#!/usr/bin/env bash
# Data preparation after the download: verify -> segments -> word alignment -> tokenizer.
# Every step skips itself when its output already exists, so the script can be rerun.
set -euo pipefail

RAW=/data/saudi-tts/raw
SEG=/data/saudi-tts/segments
REPO=/home/ai2/pocket-tts-training/pocket-tts
SCRIPTS=/home/ai2/pocket-tts-training/scripts
ALIGN_MODEL=jonatasgrosman/wav2vec2-large-xlsr-53-arabic
CORPORA=(nafas maha shakhsiya sara swalif socrates youtube-001-part02-tashkeel)
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for the download to finish"
while pgrep -f scripts/download.sh > /dev/null; do sleep 30; done

log "verifying checksums"
for c in "${CORPORA[@]}"; do
  (cd "$RAW/$c" && grep -v ' flac/long/' SHA256SUMS | grep ' flac/' \
    | sha256sum -c --quiet > "/data/saudi-tts/logs/sha_$c.log" 2>&1 \
    && echo "$c OK" || echo "$c FAILED (see logs/sha_$c.log)") &
done
wait
if grep -l -E 'FAILED|No such file' /data/saudi-tts/logs/sha_*.log 2>/dev/null; then
  log "checksum failures above: rerun scripts/download.sh, then this script"
  exit 1
fi

cd "$REPO"
if [ ! -f "$SEG/train.jsonl" ]; then
  log "building segments"
  uv run python "$SCRIPTS/build_segments.py" --out "$SEG" --workers 64
fi

if [ ! -f "$SEG/train_aligned.jsonl" ]; then
  log "aligning (2 processes on the GPU)"
  split -n l/2 -d "$SEG/train.jsonl" "$SEG/train_part_"
  for p in 00 01; do
    uv run python training/scripts/align_data.py "$SEG/train_part_$p" "$SEG/train_part_$p.aligned" \
      --model "$ALIGN_MODEL" --strip-marks --batch-size 32 --resume --shard "${p#0}" \
      > "/data/saudi-tts/logs/align_$p.log" 2>&1 &
  done
  wait
  uv run python training/scripts/align_data.py "$SEG/valid.jsonl" "$SEG/valid_aligned.jsonl" \
    --model "$ALIGN_MODEL" --strip-marks --batch-size 32 > /data/saudi-tts/logs/align_valid.log 2>&1
  uv run python training/scripts/align_data.py "$SEG/valid_unseen.jsonl" "$SEG/valid_unseen_aligned.jsonl" \
    --model "$ALIGN_MODEL" --strip-marks --batch-size 32 >> /data/saudi-tts/logs/align_valid.log 2>&1
  cat "$SEG/train_part_00.aligned" "$SEG/train_part_01.aligned" > "$SEG/train_aligned.jsonl.tmp"
  mv "$SEG/train_aligned.jsonl.tmp" "$SEG/train_aligned.jsonl"
fi

if [ ! -f "$SEG/tokenizer.json" ]; then
  log "training the tokenizer"
  uv run python -m training.scripts.train_tokenizer "$SEG/tokenizer" "$SEG/train_aligned.jsonl" \
    --character-coverage 1.0
  uv run python -m training.scripts.convert_tokenizer "$SEG/tokenizer.model" \
    --out "$SEG/tokenizer.json" --texts "$SEG/valid_aligned.jsonl"
fi

log "data ready: $(wc -l < "$SEG/train_aligned.jsonl") train segments"
