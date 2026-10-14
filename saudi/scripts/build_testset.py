"""Build the Saudi/Najdi TTS test set: testset/testset.jsonl.

Two parts:
- written: hand-written agent-style Najdi texts (testset/written.json), short /
  medium / long, as an LLM would send them. Voiced at eval time with the
  candidate production voices.
- heldout: real transcripts of the 30 speakers excluded from training (their
  voices and texts were never seen). Each item is voiced with a *different*
  clip of the same speaker, so it tests cloning an unseen voice, and keeps the
  original recording for reference. Long items join consecutive clips of one
  speaker within an episode.

Every text goes through normalize_tashkeel, as at inference.
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/ai2/pocket-tts-training/pocket-tts")
from pocket_tts.arabic_text import normalize_tashkeel  # noqa: E402

ROOT = Path("/home/ai2/pocket-tts-training")
RAW = Path("/data/saudi-tts/raw")
SEG = Path("/data/saudi-tts/segments")
OUT = ROOT / "testset"
_IDX = re.compile(r"_(\d+)(?:_long)?\.(?:wav|flac)$")
_BAD = re.compile(r"[0-9٠-٩A-Za-z]")
rng = random.Random(7)


_LABEL = re.compile(r"(^|\s)[\u0621-\u064a]{2,12}:\s*")  # "رغد:" speaker tags in some transcripts


def clip_text(r: dict) -> str:
    return normalize_tashkeel(_LABEL.sub(" ", r.get("text_tashkeel") or r.get("text") or ""))


def main() -> None:
    items = []
    for w in json.loads((OUT / "written.json").read_text()):
        text = w.get("text") or (ROOT / w["source_file"]).read_text()
        items.append(
            {
                "id": f"w_{w['id']}",
                "part": "written",
                "length": w["length"],
                "category": w["category"],
                "text": normalize_tashkeel(text),
            }
        )

    unseen = json.loads((SEG / "summary.json").read_text())["unseen_speakers"]
    by_spk = collections.defaultdict(list)
    wanted = {tuple(u.split("/", 1)) for u in unseen}
    for corpus in {c for c, _ in wanted}:
        for line in (RAW / corpus / "metadata.jsonl").open():
            r = json.loads(line)
            if r.get("split") == "long" or (corpus, r["speaker"]) not in wanted:
                continue
            text = clip_text(r)
            if not text or _BAD.search(text):
                continue
            r["_text"], r["_idx"], r["_path"] = text, int(_IDX.search(r["audio_file"]).group(1)), str(RAW / corpus / r["flac"])
            by_spk[f"{corpus}/{r['speaker']}"].append(r)

    used: set[str] = set()
    plan = {"short": 12, "medium": 12, "long": 8}
    counts = collections.Counter()
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    for spk in speakers * 3:  # several passes so every length fills up
        clips = sorted(by_spk[spk], key=lambda r: (r["video_id"], r["_idx"]))
        free = [r for r in clips if r["id"] not in used]
        refs = [r for r in free if 5 <= r["duration"] <= 9]
        if not refs:
            continue
        ref = rng.choice(refs)
        pool = [r for r in free if r is not ref]
        item = None
        if counts["short"] < plan["short"]:
            cand = [r for r in pool if r["duration"] < 3 and len(r["_text"].split()) <= 5]
            if cand:
                c = rng.choice(cand)
                item = ("short", [c])
        if item is None and counts["medium"] < plan["medium"]:
            cand = [r for r in pool if 4 <= r["duration"] <= 12]
            if cand:
                item = ("medium", [rng.choice(cand)])
        if item is None and counts["long"] < plan["long"]:
            runs, cur = [], []
            for r in pool:  # consecutive clips of this speaker within one episode
                if cur and r["video_id"] == cur[-1]["video_id"] and r["_idx"] == cur[-1]["_idx"] + 1 and r["start"] - cur[-1]["end"] < 3:
                    cur.append(r)
                else:
                    cur = [r]
                total = sum(x["duration"] for x in cur)
                if 25 <= total <= 60:
                    runs.append(list(cur))
            if runs:
                item = ("long", rng.choice(runs))
        if item is None:
            continue
        length, chosen = item
        counts[length] += 1
        used.update(r["id"] for r in chosen)
        used.add(ref["id"])
        items.append(
            {
                "id": f"h_{length[0]}{counts[length]:02d}",
                "part": "heldout",
                "length": length,
                "category": "real_" + spk.split("/")[0].split("-")[0],
                "text": " ".join(r["_text"] for r in chosen),
                "voice": ref["_path"],
                "speaker": spk,
                "real_audio": [r["_path"] for r in chosen],
                "real_duration": round(sum(r["duration"] for r in chosen), 2),
            }
        )

    with (OUT / "testset.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = collections.Counter((it["part"], it["length"]) for it in items)
    print(f"{len(items)} items:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
