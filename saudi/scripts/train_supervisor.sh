#!/usr/bin/env bash
# Runs the teacher training until max_steps, resuming from the latest checkpoint
# after any crash. After 2 out-of-memory crashes it switches to the batch-16 x
# accum-4 config (same effective batch). `touch $RUN/STOP` ends it cleanly.
set -uo pipefail
ROOT=/home/ai2/pocket-tts-training
RUN=/data/saudi-tts/runs/teacher
LOG=/data/saudi-tts/logs
CONFIG=$ROOT/configs/saudi_teacher.yaml
export HF_TOKEN=$(cat "$ROOT/.hf_token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*" >> "$LOG/supervisor.log"; }

log "waiting for data preparation"
until grep -q "data ready" "$LOG/prepare.log" 2>/dev/null; do
  if ! pgrep -f scripts/prepare_all.sh > /dev/null; then log "prepare_all.sh stopped without 'data ready'"; exit 1; fi
  sleep 60
done

mkdir -p "$RUN"
ooms=0; restarts=0
cd "$ROOT/pocket-tts"
while [ ! -f "$RUN/STOP" ]; do
  attempt_log="$LOG/train_$(date +%Y%m%d_%H%M%S).log"
  log "starting training ($CONFIG), log $attempt_log"
  uv run python -m training.train "$CONFIG" > "$attempt_log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then log "training finished"; break; fi
  restarts=$((restarts + 1))
  if grep -q "OutOfMemoryError\|CUDA out of memory" "$attempt_log"; then
    ooms=$((ooms + 1))
    log "OOM crash #$ooms"
    if [ $ooms -ge 2 ] && [ "$CONFIG" != "$ROOT/configs/saudi_teacher_b16.yaml" ]; then
      CONFIG=$ROOT/configs/saudi_teacher_b16.yaml
      log "switching to the low-memory config"
    fi
  else
    log "crash (exit $code), see $attempt_log"
  fi
  if [ $restarts -ge 30 ]; then log "too many restarts, giving up"; exit 1; fi
  sleep 60
done
