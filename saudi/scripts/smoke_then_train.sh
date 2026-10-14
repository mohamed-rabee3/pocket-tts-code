#!/usr/bin/env bash
# After data preparation: a short interleaved smoke run on nafas, then (only if it
# trained cleanly) the supervised full training run.
set -uo pipefail
ROOT=/home/ai2/pocket-tts-training
LOG=/data/saudi-tts/logs
SMOKE=${SMOKE_DIR:?set SMOKE_DIR to the folder holding smoke.yaml}
export HF_TOKEN=$(cat "$ROOT/.hf_token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/smoke_then_train.log"; }

until grep -q "data ready" "$LOG/prepare.log" 2>/dev/null; do
  if ! pgrep -f scripts/prepare_all.sh > /dev/null; then log "prepare_all.sh stopped without 'data ready'"; exit 1; fi
  sleep 30
done
log "data ready; smoke test"

rm -rf "$SMOKE/run_smoke"
cd "$ROOT/pocket-tts"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$LOG/smoke_gpu.log" 2>&1 &
GPID=$!
uv run python -m training.train "$SMOKE/smoke.yaml" > "$LOG/smoke_train.log" 2>&1
code=$?
kill $GPID
peak=$(sort -n "$LOG/smoke_gpu.log" | tail -1)
if [ $code -ne 0 ] || grep -q -E "Traceback|OutOfMemory|skipping unreadable" "$LOG/smoke_train.log"; then
  log "smoke test FAILED (exit $code, peak ${peak} MiB): see $LOG/smoke_train.log"
  exit 1
fi
log "smoke test passed (peak ${peak} MiB): $(grep -E 'step 120 ' "$LOG/smoke_train.log" | cut -c1-200)"

nohup "$ROOT/scripts/train_supervisor.sh" > /dev/null 2>&1 &
log "full training supervisor started (pid $!)"
