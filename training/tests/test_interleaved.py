"""The interleaved training layout must equal streaming inference that appends text
mid-utterance: same backbone outputs at every audio frame and EOS probe."""

import types

import pytest
import torch
import torch.nn.functional as F

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.stateful_module import ModelState, increment_steps, init_states
from training.modules.conditioner import build_interleaved_sequences


@pytest.fixture(scope="module")
def flow_lm() -> FlowLMModel:
    from pocket_tts import TTSModel

    try:
        model = TTSModel.load_model(language="english")
    except Exception as exc:  # noqa: BLE001 -- gated weights unavailable offline
        pytest.skip(f"pretrained weights unavailable: {exc}")
    fl = model.flow_lm.float().eval()
    return fl


def _stream(
    fl: FlowLMModel,
    state: ModelState,
    text: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    seq: torch.Tensor | None = None,
) -> torch.Tensor:
    """One inference call, as TTSModel._run_flow_lm_and_increment_step, returning
    the backbone output instead of a sampled latent."""
    dim = fl.dim
    te = fl.conditioner(text) if text is not None else torch.zeros(1, 0, dim)
    te = torch.cat([te, cond if cond is not None else torch.zeros(1, 0, dim)], dim=1)
    seq = seq if seq is not None else torch.zeros(1, 0, fl.ldim)
    inp = fl.input_linear(torch.where(torch.isnan(seq), fl.bos_emb, seq))
    out = fl.backbone(inp, te, seq, model_state=state)
    increment_steps(fl, state, increment=te.shape[1] + seq.shape[1])
    return out


@torch.no_grad()
def test_interleaved_layout_matches_streaming(flow_lm: FlowLMModel) -> None:
    fl = flow_lm
    torch.manual_seed(0)
    T, Tv = 40, 12
    raw = torch.randn(1, T, fl.ldim) * fl.emb_std + fl.emb_mean
    norm = (raw - fl.emb_mean) / fl.emb_std
    voice = torch.randn(1, Tv, fl.ldim) * fl.emb_std + fl.emb_mean
    t1 = fl.conditioner.prepare("Hello there, and thanks for calling.")[0]
    t2 = fl.conditioner.prepare("I checked your account.")[0]
    t3 = fl.conditioner.prepare("Anything else?")[0]
    s2, s3 = 15, 28
    args = types.SimpleNamespace(voice_dropout=0.0, text_dropout=0.0)

    x, idx, pidx, pmask = build_interleaved_sequences(
        args, norm, [t1], [[(s2, t2), (s3, t3)]], voice, cfg_dropout=False, fl=fl
    )
    out = fl.out_norm(fl.transformer(x, model_state=None))
    z_train = out[0, idx[0]]
    zp_train = out[0, pidx[0][pmask[0]]]

    state = init_states(fl, batch_size=1, sequence_length=x.shape[1] + 4)
    cond = F.linear(voice, fl.speaker_proj_weight)
    cond = torch.cat([fl.bos_before_voice, cond], dim=1)
    _stream(fl, state, cond=cond)
    z_inf, zp_inf = [], []
    bounds = [0, s2, s3, T]
    texts = [t1, t2, t3]
    for j in range(3):
        _stream(fl, state, text=texts[j][None])
        for i in range(bounds[j], bounds[j + 1]):
            prev = torch.full((1, 1, fl.ldim), float("nan")) if i == 0 else norm[:, i - 1 : i]
            z_inf.append(_stream(fl, state, seq=prev)[0, -1])
        if j < 2:  # EOS probe: the next frame's input, before the next text
            zp_inf.append(_stream(fl, state, seq=norm[:, bounds[j + 1] - 1 : bounds[j + 1]])[0, -1])
    z_inf, zp_inf = torch.stack(z_inf), torch.stack(zp_inf)

    assert z_train.shape == z_inf.shape and zp_train.shape == zp_inf.shape
    torch.testing.assert_close(z_train, z_inf, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(zp_train, zp_inf, atol=2e-4, rtol=2e-4)


@torch.no_grad()
def test_single_piece_matches_original_builder(flow_lm: FlowLMModel) -> None:
    """With no extra pieces the interleaved builder reproduces the upstream layout."""
    from training.modules.conditioner import build_sequences_with_conditions

    fl = flow_lm
    torch.manual_seed(1)
    norm = torch.randn(2, 20, fl.ldim)
    voice = torch.randn(2, 8, fl.ldim)
    toks = [fl.conditioner.prepare("One two three.")[0], fl.conditioner.prepare("Four.")[0]]
    args = types.SimpleNamespace(voice_dropout=0.0, text_dropout=0.0)
    x_old, prefix = build_sequences_with_conditions(args, norm, toks, voice, False, fl)
    x_new, idx, _, pmask = build_interleaved_sequences(args, norm, toks, [[], []], voice, False, fl)
    assert not pmask.any()
    assert torch.equal(idx, prefix[:, None] + torch.arange(20)[None, :])
    for b in range(2):
        n = int(prefix[b]) + 20
        torch.testing.assert_close(x_new[b, :n], x_old[b, :n])
