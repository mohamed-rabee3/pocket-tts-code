"""Build multi-sentence training segments from the Saudi clip corpora.

Clips are ~5 s; Pocket TTS trains on "prompt = start of a recording, target =
its continuation", so longer segments teach it to carry speech across
sentences. Consecutive clips of one speaker are joined when nothing else was
cut from the episode between them (consecutive clip index, so no kept or
rejected clip of any speaker sits in the gap), the gap is <= MAX_GAP_SEC and
the segment stays <= MAX_SEG_SEC.

Per segment: clips resampled to 24 kHz, their per-clip loudness gain undone
(so level steps between clips are the original ones) and one segment gain
applied, gaps filled with noise at the clips' own floor level, text joined and
passed through pocket_tts.arabic_text.normalize_tashkeel.

Writes /data/saudi-tts/segments/<corpus>/<speaker>/<id>.flac and manifests:
  train.jsonl, valid.jsonl (seen speakers + unseen speakers), valid_unseen.jsonl
"""

import argparse
import collections
import json
import logging
import random
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pocket_tts.arabic_text import normalize_tashkeel

SR = 24000
MAX_GAP_SEC = 1.0
MAX_SEG_SEC = 30.0
PEAK = 10 ** (-1 / 20)
CORPORA = ["nafas", "maha", "shakhsiya", "sara", "swalif", "socrates", "youtube-001-part02-tashkeel"]
MIN_TASHKEEL_COVERAGE = 0.6
_BAD_CHARS = re.compile(r"[0-9٠-٩A-Za-z]")
_IDX = re.compile(r"_(\d+)(?:_long)?\.(?:wav|flac)$")

logger = logging.getLogger("build_segments")


def clip_index(row: dict) -> int:
    return int(_IDX.search(row["audio_file"]).group(1))


def load_rows(raw: Path, corpus: str, stats: collections.Counter) -> list[dict]:
    rows = []
    for line in (raw / corpus / "metadata.jsonl").open():
        r = json.loads(line)
        if r.get("split") == "long":
            continue
        if "text_tashkeel" in r:  # youtube corpus: diacritized text lives here
            if (r.get("coverage") or 0) < MIN_TASHKEEL_COVERAGE:
                stats["drop_low_tashkeel"] += 1
                continue
            text = r["text_tashkeel"]
        else:
            text = r["text"]
        text = normalize_tashkeel(text or "")
        if not text:
            stats["drop_empty_text"] += 1
            continue
        if _BAD_CHARS.search(text):
            stats["drop_digits_or_latin"] += 1
            continue
        r["_text"] = text
        r["_idx"] = clip_index(r)
        r["_corpus"] = corpus
        rows.append(r)
    return rows


def group_segments(rows: list[dict]) -> list[list[dict]]:
    by_video = collections.defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    segments = []
    for clips in by_video.values():
        clips.sort(key=lambda r: r["_idx"])
        cur: list[dict] = []
        for r in clips:
            if cur:
                prev = cur[-1]
                gap = r["start"] - prev["end"]
                if (
                    r["speaker"] == prev["speaker"]
                    and r["_idx"] == prev["_idx"] + 1
                    and 0 <= gap <= MAX_GAP_SEC
                    and r["end"] - cur[0]["start"] <= MAX_SEG_SEC
                ):
                    cur.append(r)
                    continue
                segments.append(cur)
            cur = [r]
        if cur:
            segments.append(cur)
    return segments


def _floor_rms(x: np.ndarray, from_end: bool) -> float:
    edge = x[-SR // 5 :] if from_end else x[: SR // 5]
    n = SR // 50
    frames = [edge[i : i + n] for i in range(0, max(1, len(edge) - n + 1), n // 2)]
    rms = [float(np.sqrt(np.mean(f**2))) for f in frames if len(f)]
    return min(rms) if rms else 0.0


def render(job: tuple[Path, Path, list[dict], int]) -> dict | None:
    raw, out_root, clips, seed = job
    rng = np.random.default_rng(seed)
    parts = []
    for i, r in enumerate(clips):
        x, sr = sf.read(raw / r["_corpus"] / r["flac"], dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            g = np.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x *= 10 ** (-float(r.get("gain_db", 0.0)) / 20)
        if i > 0:
            gap = int(round((r["start"] - clips[i - 1]["end"]) * SR))
            if gap > 0:
                level = min(_floor_rms(parts[-1], True), _floor_rms(x, False))
                fill = (rng.standard_normal(gap) * level).astype(np.float32)
                ramp = min(len(fill) // 2, SR // 200)
                if ramp:
                    w = np.linspace(0, 1, ramp, dtype=np.float32)
                    fill[:ramp] *= w
                    fill[-ramp:] *= w[::-1]
                parts.append(fill)
        parts.append(x)
    y = np.concatenate(parts)
    y *= 10 ** (float(np.median([r.get("gain_db", 0.0) for r in clips])) / 20)
    peak = float(np.abs(y).max())
    if peak > PEAK:
        y *= PEAK / peak
    first = clips[0]
    speaker = f"{first['_corpus']}/{first['speaker']}"
    seg_id = f"{first['video_id']}_{first['_idx']:04d}_{len(clips):02d}"
    path = out_root / first["_corpus"] / first["speaker"] / f"{seg_id}.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR, subtype="PCM_16")
    return {
        "path": str(path),
        "duration": round(len(y) / SR, 3),
        "transcript": " ".join(r["_text"] for r in clips),
        "speaker": speaker,
        "corpus": first["_corpus"],
        "n_clips": len(clips),
        "orig_sr": int(first.get("sample_rate", SR)),
        "clips": [r["id"] for r in clips],
    }


def split_speakers(segments: list[list[dict]], rng: random.Random, n_unseen: int) -> set[str]:
    """Hold out whole speakers (3-15 min each, youtube + podcast guests) for the new-voice eval."""
    secs = collections.Counter()
    for s in segments:
        secs[f"{s[0]['_corpus']}/{s[0]['speaker']}"] += sum(r["duration"] for r in s)
    pool = sorted(k for k, v in secs.items() if 180 <= v <= 900 and not k.endswith("SPEAKER_00"))
    return set(rng.sample(pool, min(n_unseen, len(pool))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("/data/saudi-tts/raw"))
    ap.add_argument("--out", type=Path, default=Path("/data/saudi-tts/segments"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--n-unseen", type=int, default=30)
    ap.add_argument("--n-valid-seen", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")

    stats: collections.Counter = collections.Counter()
    segments: list[list[dict]] = []
    for c in args.corpora:
        rows = load_rows(args.raw, c, stats)
        segs = group_segments(rows)
        logger.info(f"{c}: {len(rows)} clips -> {len(segs)} segments")
        segments += segs

    rng = random.Random(args.seed)
    unseen = split_speakers(segments, rng, args.n_unseen)
    jobs = [(args.raw, args.out, s, args.seed + i) for i, s in enumerate(segments)]
    args.out.mkdir(parents=True, exist_ok=True)
    out = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(render, jobs, chunksize=64)):
            if row:
                out.append(row)
            if (i + 1) % 20000 == 0:
                logger.info(f"rendered {i + 1}/{len(jobs)}")

    out.sort(key=lambda r: r["path"])
    unseen_rows = [r for r in out if r["speaker"] in unseen]
    seen_rows = [r for r in out if r["speaker"] not in unseen]
    rng.shuffle(seen_rows)
    valid_seen = [r for r in seen_rows[: args.n_valid_seen * 5] if 4 <= r["duration"] <= 20]
    valid_seen = valid_seen[: args.n_valid_seen]
    held = {r["path"] for r in valid_seen}
    train = sorted((r for r in seen_rows if r["path"] not in held), key=lambda r: r["path"])
    valid_unseen = [r for r in unseen_rows if 4 <= r["duration"] <= 20]
    rng.shuffle(valid_unseen)
    valid_unseen = valid_unseen[:300]

    def dump(name: str, rows: list[dict]) -> None:
        with (args.out / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("train.jsonl", train)
    dump("valid_seen.jsonl", valid_seen)
    dump("valid_unseen.jsonl", valid_unseen)
    dump("valid.jsonl", valid_seen + valid_unseen)
    hours = sum(r["duration"] for r in train) / 3600
    summary = {
        "train_segments": len(train),
        "train_hours": round(hours, 1),
        "valid_seen": len(valid_seen),
        "valid_unseen": len(valid_unseen),
        "unseen_speakers": sorted(unseen),
        "dropped": dict(stats),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(json.dumps({k: v for k, v in summary.items() if k != "unseen_speakers"}))


if __name__ == "__main__":
    main()
