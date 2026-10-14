"""Evaluate a checkpoint on testset/testset.jsonl.

usage:
  eval_testset.py <checkpoint.pt> --name <run_name>
      [--setups continuous:1.0,continuous:1.25,continuous:1.5,default:1.0]
      [--seeds 0,1] [--voices ../ref_f2.wav,../ref-new-1.wav] [--parts written,heldout]
      [--lengths short,medium,long]

Written items are voiced with each --voices file; held-out items with another
clip of their own (unseen) speaker. Continuous mode streams the text through the
agent chunker as an LLM would; default mode is Pocket's generate_audio_stream
(no CFG, so only default:1.0).

Per generation: duration, speaking rate, silence, long pauses, CER, loop
and failure flags, chunks that hit the length limit without EOS, and WavLM
speaker similarity to the voice used. CER uses Cohere Transcribe Arabic;
run with `uv run --with librosa`. Writes evals/<name>/results.csv and
summary.md (means per setup x part x length), plus all audio.
"""

import argparse
import csv
import json
import logging
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pocket_tts import TTSModel
from pocket_tts.arabic_text import strip_tashkeel
from pocket_tts.continuous import VoiceAgentTTS, chunk_text_stream

ROOT = Path("/home/ai2/pocket-tts-training")
SR = 24000


class _CountNoEos(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "without EOS" in record.getMessage():
            self.n += 1


def letters(t: str) -> int:
    return len(re.sub(r"[^ء-ي]", "", strip_tashkeel(t)))


def norm(t: str) -> str:
    t = re.sub(r"[^ء-ي\s]", "", strip_tashkeel(t))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def timing(y: np.ndarray) -> tuple[float, float, int]:
    """(speech seconds, silence seconds, pauses >= 0.6 s) from 20 ms frames at -45 dB."""
    f = SR // 50
    e = np.array([np.sqrt(np.mean(y[i : i + f] ** 2)) for i in range(0, max(1, len(y) - f), f)])
    sil = 20 * np.log10(e + 1e-9) < -45
    runs, c = [], 0
    for v in list(sil) + [False]:
        if v:
            c += 1
        else:
            if c >= 5:
                runs.append(c * 0.02)
            c = 0
    speech = float((~sil).sum() * 0.02)
    return speech, len(y) / SR - speech, sum(r >= 0.6 for r in runs)


def looped(hyp: str, ref: str) -> bool:
    """A word or 3-gram repeated far more often than in the reference (stuck generation)."""
    h, r = norm(hyp).split(), norm(ref).split()
    for n in (1, 3):
        hc = defaultdict(int)
        for i in range(len(h) - n + 1):
            hc[tuple(h[i : i + n])] += 1
        rc = defaultdict(int)
        for i in range(len(r) - n + 1):
            rc[tuple(r[i : i + n])] += 1
        if any(v >= 4 and v > 2 * rc.get(k, 0) + 1 for k, v in hc.items()):
            return True
    return bool(re.search(r"(.)\1{7,}", hyp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--setups", default="continuous:1.0,continuous:1.25,continuous:1.5,default:1.0")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--voices", default=f"{ROOT}/ref_f2.wav,{ROOT}/ref-new-1.wav")
    ap.add_argument("--parts", default="written,heldout")
    ap.add_argument("--lengths", default="short,medium,long")
    ap.add_argument("--config", default=str(ROOT / "configs/saudi_inference_24l.yaml"))
    args = ap.parse_args()

    out = ROOT / "evals" / args.name
    (out / "audio").mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in (ROOT / "testset/testset.jsonl").open()]
    items = [
        it
        for it in items
        if it["part"] in args.parts.split(",") and it["length"] in args.lengths.split(",")
    ]
    voices = args.voices.split(",")
    setups = [(s.split(":")[0], float(s.split(":")[1])) for s in args.setups.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    counter = _CountNoEos()
    logging.getLogger("pocket_tts").addHandler(counter)
    model = TTSModel.load_model(config=args.config, checkpoint=args.checkpoint).to("cuda")
    jobs = []
    for mode, cfg in setups:
        for seed in seeds:
            for it in items:
                for voice in [it["voice"]] if it["part"] == "heldout" else voices:
                    jobs.append((mode, cfg, seed, it, voice))
    print(f"{len(jobs)} generations", flush=True)
    rows = []
    for k, (mode, cfg, seed, it, voice) in enumerate(jobs):
        torch.manual_seed(seed)
        before = counter.n
        if mode == "continuous":
            turn = VoiceAgentTTS(model, voice, interleaved=True, cfg_coef=cfg).new_turn()
            chunks = chunk_text_stream([w + " " for w in it["text"].split()])
            audio = [a for c in chunks for a in turn.say(c)]
        else:
            state = model.get_state_for_audio_prompt(Path(voice))
            audio = list(model.generate_audio_stream(state, it["text"]))
        y = torch.cat(audio).float().cpu().numpy() if audio else np.zeros(1, dtype=np.float32)
        vname = Path(voice).stem if it["part"] == "written" else "own"
        wav = out / "audio" / f"{mode}_cfg{cfg}_s{seed}_{it['id']}_{vname}.wav"
        sf.write(wav, y, SR)
        speech, silence, pauses = timing(y)
        rows.append(
            {
                "setup": f"{mode}:{cfg}",
                "seed": seed,
                "id": it["id"],
                "part": it["part"],
                "length": it["length"],
                "category": it["category"],
                "voice": vname,
                "duration": round(len(y) / SR, 2),
                "letters_per_s": round(letters(it["text"]) / max(speech, 0.05), 2),
                "silence_frac": round(silence / max(len(y) / SR, 0.05), 3),
                "long_pauses": pauses,
                "no_eos_chunks": counter.n - before,
                "wav": str(wav),
                "voice_path": voice,
                "text": it["text"],
            }
        )
        if (k + 1) % 25 == 0:
            print(f"generated {k + 1}/{len(jobs)}", flush=True)
    del model
    torch.cuda.empty_cache()

    from asr import CohereASR
    from transformers import AutoFeatureExtractor, WavLMForXVector

    asr = CohereASR()
    import jiwer
    from scipy.signal import resample_poly

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda").eval()

    def embed(path: str) -> torch.Tensor:
        x, sr = sf.read(path, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        inp = fe(x, sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            return torch.nn.functional.normalize(sv(**inp).embeddings, dim=-1)[0]

    voice_emb = {v: embed(v) for v in {r["voice_path"] for r in rows}}
    for r in rows:
        y, _ = sf.read(r["wav"], dtype="float32")
        hyp = asr(y, SR)
        ref = norm(r["text"])
        r["cer"] = round(jiwer.cer(ref, norm(hyp)) if norm(hyp) else 1.0, 4)
        r["loop"] = int(looped(hyp, r["text"]))
        r["failed"] = int(r["cer"] > 0.5 or r["loop"] or r["no_eos_chunks"] > 0)
        r["speaker_sim"] = round(float(embed(r["wav"]) @ voice_emb[r["voice_path"]]), 4) if len(y) > SR // 2 else ""
        r["hyp"] = hyp.strip()

    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["setup"], r["part"], r["length"])].append(r)
        groups[(r["setup"], "ALL", "all")].append(r)
    lines = [
        f"# {args.name}: {args.checkpoint}",
        "",
        "| setup | part | length | n | CER mean | CER median | failed % | loops | no-EOS | duration s | letters/s | silence % | long pauses | speaker sim |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = {"short": 0, "medium": 1, "long": 2, "all": 3}
    for key in sorted(groups, key=lambda k: (k[0], k[1] == "ALL", k[1], order[k[2]])):
        g = groups[key]
        sims = [r["speaker_sim"] for r in g if r["speaker_sim"] != ""]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(g)} | {st.mean(r['cer'] for r in g):.3f} | "
            f"{st.median(r['cer'] for r in g):.3f} | {100 * st.mean(r['failed'] for r in g):.1f} | "
            f"{sum(r['loop'] for r in g)} | {sum(r['no_eos_chunks'] > 0 for r in g)} | "
            f"{st.mean(r['duration'] for r in g):.1f} | {st.median(r['letters_per_s'] for r in g):.1f} | "
            f"{100 * st.mean(r['silence_frac'] for r in g):.0f} | {st.mean(r['long_pauses'] for r in g):.1f} | "
            f"{st.mean(sims) if sims else float('nan'):.3f} |"
        )
    worst = sorted(rows, key=lambda r: -r["cer"])[:10]
    lines += ["", "## Worst 10 generations", ""]
    lines += [f"- {r['setup']} s{r['seed']} {r['id']} ({r['voice']}): CER {r['cer']}, loop {r['loop']} — {r['hyp'][:120]}" for r in worst]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
