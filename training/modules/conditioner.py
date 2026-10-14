"""Row assembly for TrainableTTS: the padded
[bos_prefix, voice, text, bos, audio] training sequences.

Pure input plumbing (CFG dropout draws, host-side length math, scatter
placement) with no learnable logic of its own.
"""

import torch
import torch.nn.functional as F

from pocket_tts.models.flow_lm import FlowLMModel

from ..args import TrainArgs


def build_sequences_with_conditions(
    args: TrainArgs,
    normalized_latents: torch.Tensor,
    text_tokens: list[torch.Tensor],
    voice_latents: torch.Tensor,
    cfg_dropout: bool,
    fl: FlowLMModel,
    num_voice_prompt_frames: torch.Tensor | None = None,
    force_null: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble per-sample [beginning_of_prefix, voice_conditioning,
    text_conditioning, beginning_of_sequence, target_audio] rows,
    right-padded to equal length.

    Fully tensorized (no per-row python ints): scatter-placement from
    tensor offsets, so the whole forward stays traceable and returns
    prefix lengths as a tensor. Padding sits after the audio, so with
    causal attention it never leaks into valid positions.
    """
    B = normalized_latents.shape[0]
    device = normalized_latents.device
    bos = fl.bos_emb.view(1, 1, -1).expand(B, 1, -1)
    audio_in = torch.cat([bos, normalized_latents[:, :-1]], dim=1)
    audio_emb = fl.input_linear(audio_in)
    dtype = audio_emb.dtype
    voice_emb = F.linear(voice_latents.to(fl.speaker_proj_weight.dtype), fl.speaker_proj_weight)
    Tv = voice_emb.shape[1]
    T_audio = audio_emb.shape[1]

    # Keep decisions, lengths, and the padded-buffer size are all computed
    # host-side: text_tokens and num_voice_prompt_frames arrive on CPU, and the dropout
    # draws use CPU RNG -- so shaping the buffer costs zero device syncs
    # (an int() on a CUDA tensor would stall CPU run-ahead every step).
    if force_null:
        keep_voice_prob, keep_text_prob = 0.0, 0.0
    elif cfg_dropout:
        keep_voice_prob, keep_text_prob = 1 - args.voice_dropout, 1 - args.text_dropout
    else:
        keep_voice_prob, keep_text_prob = 1.0, 1.0
    # torch.rand is in [0, 1), so prob 1.0 keeps every row and 0.0 keeps none.
    keep_voice_cpu = torch.rand(B) < keep_voice_prob
    keep_text_cpu = torch.rand(B) < keep_text_prob

    if num_voice_prompt_frames is not None:
        v_len_cpu = num_voice_prompt_frames.cpu().long().clamp(max=Tv) * keep_voice_cpu
    else:
        v_len_cpu = torch.full((B,), Tv, dtype=torch.long) * keep_voice_cpu

    # Batched text embedding over padded tokens (LUT lookup; pad id is the
    # reserved padding bin and its embeddings are never placed).
    text_lens_cpu = torch.tensor([t.shape[0] for t in text_tokens])
    Lmax = int(text_lens_cpu.max())
    pad_id = fl.conditioner.embed.num_embeddings - 1
    text_pad_cpu = torch.full((B, Lmax), pad_id, dtype=torch.long)
    for b, t in enumerate(text_tokens):
        text_pad_cpu[b, : t.shape[0]] = t
    text_emb = fl.conditioner(text_pad_cpu.to(device)).to(dtype)
    t_len_cpu = text_lens_cpu * keep_text_cpu

    prefix_lengths_cpu = 1 + v_len_cpu + t_len_cpu
    max_len = int((prefix_lengths_cpu + T_audio).max())
    v_len, t_len = v_len_cpu.to(device), t_len_cpu.to(device)
    prefix_lengths = prefix_lengths_cpu.to(device)
    x = torch.zeros(B, max_len, audio_emb.shape[-1], device=device, dtype=dtype)

    # bos_before_voice at position 0 of every row.
    x[:, 0] = fl.bos_before_voice[0].to(dtype)
    rows = torch.arange(B, device=device)
    # voice: positions 1 .. 1+v_len
    vpos = 1 + torch.arange(Tv, device=device)
    vmask = torch.arange(Tv, device=device)[None, :] < v_len[:, None]
    x[rows[:, None].expand(-1, Tv)[vmask], vpos[None, :].expand(B, -1)[vmask]] = voice_emb.to(
        dtype
    )[vmask]
    # text: positions 1+v_len .. 1+v_len+t_len
    tpos = 1 + v_len[:, None] + torch.arange(Lmax, device=device)[None, :]
    tmask = torch.arange(Lmax, device=device)[None, :] < t_len[:, None]
    x[rows[:, None].expand(-1, Lmax)[tmask], tpos[tmask]] = text_emb[tmask]
    # Zero-weighted use of the text embeddings: keeps the text LUT in the
    # autograd graph when every row of a rank's batch drops text (possible
    # under CFG dropout), which find_unused_parameters=False requires.
    x[:, 0] = x[:, 0] + 0.0 * text_emb.sum()
    # audio: positions prefix .. prefix+T_audio (always full)
    apos = prefix_lengths[:, None] + torch.arange(T_audio, device=device)[None, :]
    x[rows[:, None].expand(-1, T_audio).reshape(-1), apos.reshape(-1)] = audio_emb.reshape(
        -1, audio_emb.shape[-1]
    )
    return x, prefix_lengths


def build_interleaved_sequences(
    args: TrainArgs,
    normalized_latents: torch.Tensor,
    text_tokens: list[torch.Tensor],
    text_pieces: list[list[tuple[int, torch.Tensor]]],
    voice_latents: torch.Tensor,
    cfg_dropout: bool,
    fl: FlowLMModel,
    num_voice_prompt_frames: torch.Tensor | None = None,
    force_null: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows with text inserted mid-audio, for seamless streaming inference:

        [bos_prefix, voice, text_1, audio[0:s_2], probe, text_2, audio[s_2:s_3], probe, ...]

    Piece 1 is text_tokens[b] (starting at frame 0); text_pieces[b] lists the later
    pieces as (start frame, tokens). A probe repeats the input of the next piece's
    first frame (the previous latent) and is trained to emit EOS: at inference the
    model signals the end of the current text there, the next text is appended to
    the KV cache, and generation resumes from that same latent.

    Returns (x, frame_idx [B, T]: position of every audio frame, probe_idx [B, P],
    probe_mask [B, P]). Positions are computed host-side; rows are right-padded.
    """
    B, T_audio = normalized_latents.shape[:2]
    device = normalized_latents.device
    bos = fl.bos_emb.view(1, 1, -1).expand(B, 1, -1)
    audio_in = torch.cat([bos, normalized_latents[:, :-1]], dim=1)
    audio_emb = fl.input_linear(audio_in)
    dtype = audio_emb.dtype
    voice_emb = F.linear(voice_latents.to(fl.speaker_proj_weight.dtype), fl.speaker_proj_weight)
    Tv = voice_emb.shape[1]

    if force_null:
        keep_voice_prob, keep_text_prob = 0.0, 0.0
    elif cfg_dropout:
        keep_voice_prob, keep_text_prob = 1 - args.voice_dropout, 1 - args.text_dropout
    else:
        keep_voice_prob, keep_text_prob = 1.0, 1.0
    keep_voice_cpu = torch.rand(B) < keep_voice_prob
    keep_text_cpu = torch.rand(B) < keep_text_prob
    if num_voice_prompt_frames is not None:
        v_len_cpu = num_voice_prompt_frames.cpu().long().clamp(max=Tv) * keep_voice_cpu
    else:
        v_len_cpu = torch.full((B,), Tv, dtype=torch.long) * keep_voice_cpu

    row_tokens, tok_pos, frame_pos, probe_pos, probe_src, row_len = [], [], [], [], [], []
    for b in range(B):
        pieces = [(0, text_tokens[b]), *text_pieces[b]]
        pos = 1 + int(v_len_cpu[b])
        toks, tp, fp, pp, ps = [], [], [], [], []
        for j, (start, tokens) in enumerate(pieces):
            if keep_text_cpu[b]:
                toks.append(tokens)
                tp.extend(range(pos, pos + tokens.shape[0]))
                pos += tokens.shape[0]
            end = pieces[j + 1][0] if j + 1 < len(pieces) else T_audio
            fp.extend(range(pos, pos + end - start))
            pos += end - start
            if j + 1 < len(pieces):
                pp.append(pos)
                ps.append(end)
                pos += 1
        row_tokens.append(torch.cat(toks) if toks else torch.zeros(0, dtype=torch.long))
        tok_pos.append(tp)
        frame_pos.append(fp)
        probe_pos.append(pp)
        probe_src.append(ps)
        row_len.append(pos)

    max_len = max(row_len)
    x = torch.zeros(B, max_len, audio_emb.shape[-1], device=device, dtype=dtype)
    x[:, 0] = fl.bos_before_voice[0].to(dtype)
    rows = torch.arange(B, device=device)
    v_len = v_len_cpu.to(device)
    vpos = 1 + torch.arange(Tv, device=device)
    vmask = torch.arange(Tv, device=device)[None, :] < v_len[:, None]
    x[rows[:, None].expand(-1, Tv)[vmask], vpos[None, :].expand(B, -1)[vmask]] = voice_emb.to(
        dtype
    )[vmask]

    # Text: one LUT lookup over the padded concatenation of each row's kept pieces.
    pad_id = fl.conditioner.embed.num_embeddings - 1
    Lmax = max(1, max(t.shape[0] for t in row_tokens))
    text_pad_cpu = torch.full((B, Lmax), pad_id, dtype=torch.long)
    t_rows, t_cols, t_pos = [], [], []
    for b, t in enumerate(row_tokens):
        text_pad_cpu[b, : t.shape[0]] = t
        t_rows += [b] * t.shape[0]
        t_cols += list(range(t.shape[0]))
        t_pos += tok_pos[b]
    text_emb = fl.conditioner(text_pad_cpu.to(device)).to(dtype)
    if t_rows:
        tr = torch.tensor(t_rows, device=device)
        x[tr, torch.tensor(t_pos, device=device)] = text_emb[
            tr, torch.tensor(t_cols, device=device)
        ]
    x[:, 0] = x[:, 0] + 0.0 * text_emb.sum()  # keep the LUT in the graph (see above)

    frame_idx = torch.tensor(frame_pos, dtype=torch.long, device=device)  # [B, T_audio]
    x[rows[:, None].expand(-1, T_audio).reshape(-1), frame_idx.reshape(-1)] = audio_emb.reshape(
        -1, audio_emb.shape[-1]
    )
    P = max(1, max(len(p) for p in probe_pos))
    probe_idx_cpu = torch.zeros(B, P, dtype=torch.long)
    probe_src_cpu = torch.zeros(B, P, dtype=torch.long)
    probe_mask_cpu = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        n = len(probe_pos[b])
        probe_idx_cpu[b, :n] = torch.tensor(probe_pos[b], dtype=torch.long)
        probe_src_cpu[b, :n] = torch.tensor(probe_src[b], dtype=torch.long)
        probe_mask_cpu[b, :n] = True
    probe_idx, probe_mask = probe_idx_cpu.to(device), probe_mask_cpu.to(device)
    if probe_mask_cpu.any():
        pr = rows[:, None].expand(-1, P)[probe_mask]
        x[pr, probe_idx[probe_mask]] = audio_emb[pr, probe_src_cpu.to(device)[probe_mask]]
    return x, frame_idx, probe_idx, probe_mask
