"""Compare checkpoints on voice consistency across chunks and pronunciation.

usage: compare_ckpts.py <name> [<ckpt1.pt> <ckpt2.pt> ...]
With no checkpoints, only (re)scores the audio already in evals/<name>/audio/.

Same texts, seeds and voices for every checkpoint: the 8 long written test items
(includes samples/text.txt as w_l04), continuous CFG 1.0, voices ref_f and ref_f2,
plus default mode on w_l04. Per generation:
- chunk speaker similarity (WavLM) to the reference: mean, worst chunk, spread
  (std across chunks), and the lowest similarity between consecutive chunks;
- for default mode (and audio without a chunk-boundary sidecar) the same over
  3 s windows;
- CER and word substitutions, transcribed with Cohere Transcribe Arabic.
Run with `uv run --with librosa`.
Writes evals/<name>/{results.csv,summary.md,audio/}.
"""

import csv
import gc
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pocket_tts import TTSModel
from pocket_tts.arabic_text import strip_tashkeel
from pocket_tts.continuous import VoiceAgentTTS, chunk_text_stream

ROOT = Path("/home/ai2/pocket-tts-training")
SR = 24000
name, ckpts = sys.argv[1], sys.argv[2:]
out = ROOT / "evals" / name
(out / "audio").mkdir(parents=True, exist_ok=True)
items = [json.loads(line) for line in (ROOT / "testset/testset.jsonl").open()]
items = [it for it in items if it["part"] == "written" and it["length"] == "long"]
voices = [ROOT / "ref_f.wav", ROOT / "ref_f2.wav"]

jobs = [("continuous", 0, it, v) for it in items for v in voices]
user = next(it for it in items if it["id"] == "w_l04")
jobs += [("continuous", 1, user, v) for v in voices] + [("default", 0, user, v) for v in voices]

_NAME = re.compile(r"step(\d+)_(continuous|default)_s(\d+)_(w_\w+?)_(ref_f2?|[\w-]+)\.wav$")


def generate(ck: str) -> None:
    """Generate every job for one checkpoint; writes <wav> and <wav>.json (chunk lengths)."""
    step = int(re.search(r"(\d+)\.pt", ck).group(1))
    model = TTSModel.load_model(config=str(ROOT / "configs/saudi_inference_24l.yaml"), checkpoint=ck).to("cuda")
    for mode, seed, it, v in jobs:
        torch.manual_seed(seed)
        if mode == "continuous":
            turn = VoiceAgentTTS(model, str(v), interleaved=True, cfg_coef=1.0).new_turn()
            pieces = []
            for c in chunk_text_stream([w + " " for w in it["text"].split()]):
                a = [x.float().cpu() for x in turn.say(c)]
                pieces.append(torch.cat(a).numpy() if a else np.zeros(0, np.float32))
        else:
            state = model.get_state_for_audio_prompt(v)
            pieces = [torch.cat([x.float().cpu() for x in model.generate_audio_stream(state, it["text"])]).numpy()]
        wav = out / "audio" / f"step{step}_{mode}_s{seed}_{it['id']}_{v.stem}.wav"
        sf.write(wav, np.concatenate(pieces), SR)
        Path(f"{wav}.json").write_text(json.dumps({"chunk_samples": [len(p) for p in pieces]}))
    print(f"generated step {step}", flush=True)


for ck in ckpts:
    generate(ck)  # all model references die with the function's frame
    gc.collect()
    torch.cuda.empty_cache()

texts = {it["id"]: it["text"] for it in items}
gens = []
for wav in sorted((out / "audio").glob("*.wav")):
    mt = _NAME.search(wav.name)
    step, mode, seed, item_id, vname = int(mt[1]), mt[2], int(mt[3]), mt[4], mt[5]
    y, _ = sf.read(wav, dtype="float32")
    side = Path(f"{wav}.json")
    if mode == "continuous" and side.exists():
        bounds = np.cumsum([0] + json.loads(side.read_text())["chunk_samples"])
        pieces = [y[a:b] for a, b in zip(bounds, bounds[1:])]
    else:
        pieces = [y]  # no chunk boundaries recorded: scored over 3 s windows
    gens.append(dict(step=step, mode=mode, seed=seed, id=item_id, voice=ROOT / f"{vname}.wav", text=texts[item_id], pieces=pieces, wav=wav))

from asr import CohereASR  # noqa: E402
from transformers import AutoFeatureExtractor, WavLMForXVector  # noqa: E402
import jiwer  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

asr = CohereASR()
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda").eval()


def embed(x: np.ndarray, sr: int) -> torch.Tensor:
    g = np.gcd(sr, 16000)
    x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    inp = fe(x, sampling_rate=16000, return_tensors="pt").to("cuda")
    with torch.no_grad():
        return torch.nn.functional.normalize(sv(**inp).embeddings, dim=-1)[0]


def norm(t: str) -> str:
    t = re.sub(r"[^ء-ي\s]", "", strip_tashkeel(t))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


ref_emb = {}
for v in voices:
    x, sr = sf.read(v, dtype="float32")
    ref_emb[v] = embed(x if x.ndim == 1 else x.mean(1), sr)

rows, subs = [], defaultdict(Counter)
for g in gens:
    if len(g["pieces"]) > 1:
        segs = [p for p in g["pieces"] if len(p) >= int(1.5 * SR)]
    else:
        y = g["pieces"][0]
        segs = [y[i : i + 3 * SR] for i in range(0, len(y) - 3 * SR, int(1.5 * SR))]
        segs = [s for s in segs if np.sqrt(np.mean(s**2)) > 0.01]
    hyp = asr(np.concatenate(g["pieces"]), SR)
    embs = [embed(s, SR) for s in segs]
    sims = [float(e @ ref_emb[g["voice"]]) for e in embs]
    adj = [float(a @ b) for a, b in zip(embs, embs[1:])]
    r, h = norm(g["text"]), norm(hyp)
    wo = jiwer.process_words(r, h)
    for al in wo.alignments[0]:
        if al.type == "substitute":
            for a, b in zip(r.split()[al.ref_start_idx : al.ref_end_idx], h.split()[al.hyp_start_idx : al.hyp_end_idx]):
                subs[(g["step"], g["mode"])][f"{a} → {b}"] += 1
    rows.append(
        dict(
            step=g["step"], mode=g["mode"], seed=g["seed"], id=g["id"], voice=g["voice"].stem,
            n_segs=len(sims),
            sim_mean=round(st.mean(sims), 4) if sims else "",
            sim_min=round(min(sims), 4) if sims else "",
            sim_std=round(st.pstdev(sims), 4) if len(sims) > 1 else "",
            adj_min=round(min(adj), 4) if adj else "",
            cer=round(jiwer.cer(r, h) if h else 1.0, 4),
            wer=round(wo.wer, 4),
            duration=round(sum(len(p) for p in g["pieces"]) / SR, 1),
            wav=str(g["wav"]), hyp=hyp.strip(),
        )
    )

with (out / "results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def m(xs):
    xs = [x for x in xs if x != ""]
    return st.mean(xs) if xs else float("nan")


lines = [f"# {name}", "", "| step | mode | voice | n | sim mean | worst chunk sim | sim spread (std) | worst consecutive-chunk sim | CER | WER |", "|---|---|---|---|---|---|---|---|---|---|"]
for key in sorted({(r["mode"], r["voice"], r["step"]) for r in rows}):
    g = [r for r in rows if (r["mode"], r["voice"], r["step"]) == key]
    lines.append(
        f"| {key[2]} | {key[0]} | {key[1]} | {len(g)} | {m(r['sim_mean'] for r in g):.3f} | {m(r['sim_min'] for r in g):.3f} | "
        f"{m(r['sim_std'] for r in g):.3f} | {m(r['adj_min'] for r in g):.3f} | {m(r['cer'] for r in g):.3f} | {m(r['wer'] for r in g):.3f} |"
    )
for key in sorted({(r["mode"], r["step"]) for r in rows}):
    g = [r for r in rows if (r["mode"], r["step"]) == key]
    lines.append(
        f"| **{key[1]}** | **{key[0]} all** | | {len(g)} | {m(r['sim_mean'] for r in g):.3f} | {m(r['sim_min'] for r in g):.3f} | "
        f"{m(r['sim_std'] for r in g):.3f} | {m(r['adj_min'] for r in g):.3f} | {m(r['cer'] for r in g):.3f} | {m(r['wer'] for r in g):.3f} |"
    )
lines += ["", "## Word substitutions (reference → heard), all generations", ""]
for key in sorted(subs):
    lines.append(f"- step {key[0]} {key[1]}: {sum(subs[key].values())} total; top: " + "، ".join(f"{k} ×{n}" for k, n in subs[key].most_common(15)))
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("ALL_DONE")
