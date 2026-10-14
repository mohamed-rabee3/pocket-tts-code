"""Seamless long-form speech for a voice agent.

`TTSModel.generate_audio_stream` splits long text into sentences and generates
each one from the bare voice prompt with a fresh Mimi decoder, so every
sentence restarts the prosody and the decoder (audible seams). Here one agent
turn is one continuous stream:

- every text chunk is generated with the voice prompt
  [reference clip latents, last `max_context_sec` of this turn's generated latents],
  the same "ref_ctx" prompt the Saudi models are trained on (the first chunk of a
  turn uses the reference alone, the "ref" training mode);
- the Mimi decoder runs continuously for the whole turn (its state is renewed
  every 20 s by warm-up + one-frame crossfade, which keeps memory bounded and
  is numerically transparent), so the waveform has no joins;
- each chunk runs a few frames past its EOS (more after a sentence end than a
  comma), so the pause is the model's own audio and the next chunk's prompt ends
  inside it; nothing is inserted or cut.

With `interleaved=True` (models trained with interleaved text, i.e. the Saudi
recipe) a turn is a single autoregressive sequence: when a chunk reaches EOS the
next chunk's text is appended to the KV cache and generation resumes from the
last latent, exactly the [text_1, audio_1, probe, text_2, audio_2, ...] layout of
training (verified by training/tests/test_interleaved.py). The model keeps the
whole turn in view, so prosody flows across chunks; the per-chunk re-prompting
above is then only used every `max_state_sec` to stay within training lengths.

Usage:
    model = TTSModel.load_model(config="...yaml")  # plus load_training_checkpoint(...)
    voice = VoiceAgentTTS(model, "reference.wav", interleaved=True)
    turn = voice.new_turn()
    for audio in turn.speak_stream(llm_text_deltas):   # or turn.say("...")
        play(audio)                                    # float32 tensor, 24 kHz
    turn.stop()                                        # barge-in, from any thread
"""

import logging
import re
import threading
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.functional as F

from pocket_tts.arabic_text import normalize_tashkeel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio, end_on_pause
from pocket_tts.models.flow_lm import drifting_decode, lsd_decode, ot_decode
from pocket_tts.modules.stateful_module import increment_steps, init_states

logger = logging.getLogger(__name__)

MIMI_REFRESH_LATENTS = 250  # 20 s per decoder state (~100 MB of KV cache)
MIMI_WARMUP_LATENTS = 32  # 2.5 s, well past the decoder's receptive field
_SENTENCE_END = re.compile(r"[.!؟?…]+[\"')\]»]*\s")
_CLAUSE_END = re.compile(r"[،,؛;:]\s")


class VoiceAgentTTS:
    def __init__(
        self,
        model: "TTSModel",  # noqa: F821
        reference_audio: str | Path | torch.Tensor,
        max_ref_sec: float = 5.0,
        max_context_sec: float = 5.0,
        frames_after_eos: int = 1,
        sentence_pause_frames: int = 4,
        clause_pause_frames: int = 2,
        max_turn_sec: float = 600.0,
        normalize_text: bool = True,
        ref_in_continuation: bool = True,
        interleaved: bool = False,
        max_state_sec: float = 25.0,
        cfg_coef: float = 1.0,
    ):
        self.model = model
        # Classifier-free guidance, as the training samples use (sample_cfg_coef):
        # a voice- and text-free copy of the sequence runs alongside and the backbone
        # output is z_null + cfg_coef * (z - z_null). Needed for the 24-layer teacher;
        # CFG-distilled models use 1.0 (one pass per frame).
        self.cfg_coef = cfg_coef
        # interleaved=True for models trained with interleaved text (the Saudi
        # recipe): a turn is one autoregressive sequence and each new chunk's text
        # is appended to it. The sequence is re-prompted (reference + recent audio)
        # once it holds max_state_sec of audio, staying within training lengths.
        self.interleaved = interleaved
        self.max_state_frames = int(max_state_sec * model.mimi.frame_rate)
        self.max_context_frames = int(max_context_sec * model.mimi.frame_rate)
        self.frames_after_eos = frames_after_eos
        self.sentence_pause_frames = sentence_pause_frames
        self.clause_pause_frames = clause_pause_frames
        self.max_turn_sec = max_turn_sec
        self.normalize_text = normalize_text
        # True for models trained with ref_ctx prompts (the Saudi recipe). Models
        # trained only on contiguous prompts (released English) need False: later
        # chunks are then prompted with the turn's own audio alone.
        self.ref_in_continuation = ref_in_continuation
        self.ref_latents = self._encode_reference(reference_audio, max_ref_sec)

    def save_voice(self, path: str | Path) -> None:
        """Save the reference as a built-in voice (Mimi latents: independent of the
        flow LM weights, so one file serves every checkpoint and distilled model)."""
        safetensors.torch.save_file({"ref_latents": self.ref_latents.cpu().contiguous()}, str(path))

    @torch.no_grad()
    def _encode_reference(self, audio: str | Path | torch.Tensor, max_sec: float) -> torch.Tensor:
        """Raw Mimi latents [1, T, D] of the reference: its start, cut to end in a pause.
        A .safetensors path loads a voice saved with save_voice()."""
        if not isinstance(audio, torch.Tensor) and str(audio).endswith(".safetensors"):
            return safetensors.torch.load_file(str(audio))["ref_latents"].to(self.model.device)
        sr = self.model.config.mimi.sample_rate
        if not isinstance(audio, torch.Tensor):
            wav, in_sr = audio_read(Path(audio))
            audio = convert_audio(wav, in_sr, sr, 1)
        audio = end_on_pause(audio[..., : int(max_sec * sr)], sr)
        return self.model.mimi.encode_to_latent(audio.unsqueeze(0).to(self.model.device)).float()

    def new_turn(self) -> "Turn":
        return Turn(self)


class Turn:
    """One uninterrupted stretch of agent speech. Not thread-safe except `stop()`."""

    def __init__(self, voice: VoiceAgentTTS):
        self.voice = voice
        model = voice.model
        self._steps_per_latent = int(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
        self._max_latents = int(voice.max_turn_sec * model.mimi.frame_rate)
        self._mimi_state = self._new_mimi_state()
        self._mimi_frames = 0
        self._recent: list[torch.Tensor] = []
        self._context = torch.zeros(1, 0, voice.ref_latents.shape[-1], device=model.device)
        self._generated = 0
        self._stop = threading.Event()
        self._first_audio = True
        self._state: dict | None = None  # interleaved mode: the turn's flow LM state
        self._state_frames = 0
        self._last_latent: torch.Tensor | None = None  # last emitted latent (normalized)

    def stop(self) -> None:
        """Barge-in: the current chunk ends at its next frame."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def speak_stream(
        self, text_deltas: Iterable[str], first_chunk_words: int = 4, max_chunk_words: int = 40
    ) -> Iterator[torch.Tensor]:
        """Speak text arriving in pieces (LLM tokens). The first chunk is cut early
        (clause end or `first_chunk_words`) for low latency; later chunks are whole
        sentences, which the continuation prompt keeps seamless."""
        for chunk in chunk_text_stream(text_deltas, first_chunk_words, max_chunk_words):
            if self.stopped:
                return
            yield from self.say(chunk)

    @torch.no_grad()
    def say(self, text: str) -> Iterator[torch.Tensor]:
        """Generate one chunk as the continuation of everything said so far in this turn."""
        model, voice = self.voice.model, self.voice
        fl = model.flow_lm
        if voice.normalize_text:
            text = normalize_tashkeel(text)
        if not text or self.stopped:
            return
        tokens = fl.conditioner.prepare(text)
        max_gen = min(
            model._estimate_max_gen_len(tokens.shape[1]), self._max_latents - self._generated
        )
        if max_gen <= 0:
            logger.warning("turn reached max_turn_sec; start a new turn")
            return

        if (
            voice.interleaved
            and self._state is not None
            and self._state_frames + max_gen <= voice.max_state_frames
        ):
            # Append this chunk's text to the running sequence and resume from the
            # last latent: the layout the interleaved training uses after an EOS probe.
            state = self._state
            for st in state:
                if st is not None:
                    end = model._flow_lm_current_end(st)
                    model._expand_kv_cache(st, sequence_length=end + tokens.shape[1] + max_gen + 1)
            self._feed(state[0], text=tokens)  # the null branch never sees text
            latent = self._last_latent
        else:
            state = self._prompted_state(tokens, max_gen)
            latent = torch.full((1, 1, fl.ldim), float("nan"), device=model.device, dtype=fl.dtype)
            self._state_frames = 0
        # Interleaved: stop exactly at EOS (the probe), the next text continues the
        # pause. Re-prompted chunks: run on a little so the pause is generated.
        frames_after_eos = 0 if voice.interleaved else self._pause_frames(text)

        eos_step = None
        new_latents = []
        for step in range(max_gen):
            if self.stopped:
                break
            next_latent, is_eos = self._step(state, latent)
            if is_eos and eos_step is None and step >= model._MIN_FRAMES_BEFORE_EOS:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break
            latent = next_latent
            raw = latent * fl.emb_std + fl.emb_mean
            new_latents.append(raw.float())
            yield self._decode(raw)
        else:
            logger.warning("chunk reached its length limit without EOS: %r", text[:60])
        if new_latents:
            self._context = torch.cat([self._context, *new_latents], dim=1)[
                :, -voice.max_context_frames :
            ]
            self._generated += len(new_latents)
        if voice.interleaved and eos_step is not None and new_latents and not self.stopped:
            self._state, self._last_latent = state, latent
            self._state_frames += len(new_latents)
        else:
            self._state = None  # no clean EOS: the next chunk starts a fresh sequence

    def _prompted_state(self, tokens: torch.Tensor, max_gen: int) -> dict:
        """A new flow LM sequence: [voice prompt, text]. The prompt is the reference
        clip, plus the turn's recent audio once there is some."""
        model, voice = self.voice.model, self.voice
        fl = model.flow_lm
        context = self._context[:, -voice.max_context_frames :]
        if context.shape[1] == 0:
            prompt = voice.ref_latents
        elif voice.ref_in_continuation:
            prompt = torch.cat([voice.ref_latents, context], 1)
        else:
            prompt = context
        cond = F.linear(prompt.to(fl.speaker_proj_weight.dtype), fl.speaker_proj_weight)
        if fl.insert_bos_before_voice:
            cond = torch.cat([fl.bos_before_voice, cond], dim=1)
        state = init_states(
            fl, batch_size=1, sequence_length=cond.shape[1] + tokens.shape[1] + max_gen + 1
        )
        self._feed(state, cond=cond)
        self._feed(state, text=tokens)
        null = None
        if voice.cfg_coef != 1.0:  # CFG null branch: bos_before_voice only (voice and text dropped)
            null = init_states(fl, batch_size=1, sequence_length=1 + max_gen + 1)
            self._feed(null, cond=fl.bos_before_voice)
        return (state, null)

    def _feed(
        self,
        state: dict,
        text: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append text tokens, conditioning embeddings or latents to a sequence;
        returns the backbone output at its last position [1, dim] (float32)."""
        fl = self.voice.model.flow_lm
        dev, dtype = fl.bos_emb.device, fl.bos_emb.dtype
        te = fl.conditioner(text) if text is not None else torch.zeros(1, 0, fl.dim, device=dev)
        if cond is not None:
            te = torch.cat([te.to(cond.dtype), cond], dim=1)
        seq = seq if seq is not None else torch.zeros(1, 0, fl.ldim, device=dev, dtype=dtype)
        inp = fl.input_linear(torch.where(torch.isnan(seq), fl.bos_emb, seq))
        out = fl.backbone(inp, te.to(inp.dtype), seq, model_state=state)
        increment_steps(fl, state, increment=te.shape[1] + seq.shape[1])
        return out[:, -1].float()

    def _step(self, state: tuple, latent: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """One frame: backbone (with CFG when a null branch exists), EOS, flow sample."""
        model, voice = self.voice.model, self.voice
        fl = model.flow_lm
        z = self._feed(state[0], seq=latent)
        if state[1] is not None:
            z_null = self._feed(state[1], seq=latent)
            z = z_null + voice.cfg_coef * (z - z_null)
        is_eos = bool((fl.out_eos(z) > model.eos_threshold).item())
        noise = torch.empty(z.shape[0], fl.ldim, device=z.device, dtype=z.dtype)
        std = model.temp**0.5
        if model.noise_clamp is None:
            torch.nn.init.normal_(noise, mean=0.0, std=std)
        else:
            torch.nn.init.trunc_normal_(
                noise, mean=0.0, std=std, a=-model.noise_clamp, b=model.noise_clamp
            )
        decode = {"lsd": lsd_decode, "flow_matching": ot_decode, "drifting": drifting_decode}[
            fl.flow_type
        ]
        latent = decode(partial(fl.flow_net, z), noise, model.sampler_decode_steps)
        return latent[:, None, :].to(fl.bos_emb.dtype), is_eos

    def _pause_frames(self, text: str) -> int:
        """Frames generated after EOS: the model's own silence, so a sentence end
        gets a real pause and the next chunk's prompt ends inside it (as the
        training prompts, cut mid-pause, do)."""
        tail = text.rstrip(" \"')]»")[-1:]
        if tail in ".!?؟…":
            return self.voice.sentence_pause_frames
        if tail in "،,؛;:":
            return self.voice.clause_pause_frames
        return self.voice.frames_after_eos

    def _new_mimi_state(self) -> dict:
        return init_states(
            self.voice.model.mimi,
            batch_size=1,
            sequence_length=(MIMI_REFRESH_LATENTS + MIMI_WARMUP_LATENTS + 2)
            * self._steps_per_latent,
        )

    def _decode_with(self, state: dict, raw_latent: torch.Tensor) -> torch.Tensor:
        mimi = self.voice.model.mimi
        audio = mimi.decode_from_latent(raw_latent, state)
        increment_steps(mimi, state, increment=self._steps_per_latent)
        return audio[0, 0]

    def _decode(self, raw_latent: torch.Tensor) -> torch.Tensor:
        """Decode one latent with the turn's decoder. The Mimi KV cache only grows,
        while the decoder sees ~1.3 s back, so every MIMI_REFRESH_LATENTS frames a
        fresh decoder is warmed up on the last MIMI_WARMUP_LATENTS latents and the
        two are crossfaded over one frame: bounded memory, no audible seam."""
        model = self.voice.model
        if self._mimi_frames >= MIMI_REFRESH_LATENTS:
            fresh = self._new_mimi_state()
            for past in self._recent:
                self._decode_with(fresh, past)
            old = self._decode_with(self._mimi_state, raw_latent)
            new = self._decode_with(fresh, raw_latent)
            w = torch.linspace(0, 1, new.numel(), device=new.device, dtype=new.dtype)
            audio = old * (1 - w) + new * w
            self._mimi_state, self._mimi_frames = fresh, len(self._recent) + 1
        else:
            audio = self._decode_with(self._mimi_state, raw_latent)
            self._mimi_frames += 1
        self._recent = (self._recent + [raw_latent])[-MIMI_WARMUP_LATENTS:]
        audio = _soft_limit(audio)
        if self._first_audio:
            # A fresh decoder starts with a small step; fade in the turn's first 5 ms only.
            n = min(audio.numel(), model.config.mimi.sample_rate // 200)
            audio[:n] *= torch.linspace(0, 1, n, device=audio.device, dtype=audio.dtype)
            self._first_audio = False
        return audio


def _soft_limit(x: torch.Tensor, knee: float = 0.9) -> torch.Tensor:
    """Identity below `knee`; above it a tanh curve that approaches but never reaches
    full scale, so rare loud onsets neither hard-clip nor wrap around in int16."""
    over = x.abs() - knee
    soft = knee + (1 - knee) * torch.tanh(over.clamp(min=0) / (1 - knee))
    return torch.where(over > 0, x.sign() * soft, x)


def chunk_text_stream(
    deltas: Iterable[str], first_chunk_words: int = 4, max_chunk_words: int = 40
) -> Iterator[str]:
    """Group streamed text into speakable chunks: sentences, except a short first
    chunk (first clause, or after `first_chunk_words` words) to start audio early,
    and a hard split at a clause end / `max_chunk_words` for run-on text."""
    buf = ""
    first = True
    for delta in deltas:
        buf += delta
        while True:
            cut = _find_cut(buf, first, first_chunk_words, max_chunk_words)
            if cut is None:
                break
            chunk, buf = buf[:cut].strip(), buf[cut:]
            if chunk:
                yield chunk
                first = False
    if buf.strip():
        yield buf.strip()


def _find_cut(buf: str, first: bool, first_words: int, max_words: int) -> int | None:
    m = _SENTENCE_END.search(buf)
    if m:
        return m.end()
    words = buf.split()
    if first:
        m = _CLAUSE_END.search(buf)
        if m:
            return m.end()
        if len(words) > first_words and buf[-1:].isspace():
            return len(buf)
    if len(words) > max_words:
        m = None
        for m in _CLAUSE_END.finditer(buf):
            pass
        if m:
            return m.end()
        if buf[-1:].isspace():
            return len(buf)
    return None
