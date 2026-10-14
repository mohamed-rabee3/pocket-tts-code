#!/usr/bin/env bash
# 6-layer depth distillation from the 140k teacher (configs/saudi_distill_6l.yaml).
# `touch $RUN/STOP` ends it cleanly.
set -uo pipefail
ROOT=/home/ai2/pocket-tts-training
CONFIG=$ROOT/configs/saudi_distill_6l.yaml
RUN=/data/saudi-tts/runs/distill_6l
LOG=/data/saudi-tts/logs
export HF_TOKEN=$(cat "$ROOT/.hf_token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*" >> "$LOG/distill_supervisor.log"; }

mkdir -p "$RUN"

restarts=0
cd "$ROOT/pocket-tts"
while [ ! -f "$RUN/STOP" ]; do
  attempt_log="$LOG/distill_$(date +%Y%m%d_%H%M%S).log"
  log "starting distillation ($CONFIG), log $attempt_log"
  uv run python -m training.train "$CONFIG" > "$attempt_log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then log "distillation finished"; break; fi
  restarts=$((restarts + 1))
  log "crash (exit $code), see $attempt_log"
  if [ $restarts -ge 10 ]; then log "too many restarts, giving up"; exit 1; fi
  sleep 60
done
