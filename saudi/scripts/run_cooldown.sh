#!/usr/bin/env bash
# Cosine cooldown of the teacher from a saved constant-LR checkpoint.
#   run_cooldown.sh [start_checkpoint.pt]   (default: teacher_saved/checkpoint_00140000.pt)
# Seeds the cooldown run dir with that checkpoint (training resumes from the newest
# checkpoint there), then trains to max_steps, resuming after crashes.
# `touch $RUN/STOP` ends it cleanly.
set -uo pipefail
ROOT=/home/ai2/pocket-tts-training
CONFIG=$ROOT/configs/saudi_teacher_cooldown.yaml
RUN=/data/saudi-tts/runs/teacher_cooldown
LOG=/data/saudi-tts/logs
START=${1:-/data/saudi-tts/runs/teacher_saved/checkpoint_00140000.pt}
export HF_TOKEN=$(cat "$ROOT/.hf_token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*" >> "$LOG/cooldown_supervisor.log"; }

mkdir -p "$RUN"
if ! ls "$RUN"/checkpoint_*.pt > /dev/null 2>&1; then
  cp -p "$START" "$RUN/" || { log "cannot copy $START"; exit 1; }
  log "seeded $RUN with $START"
fi

restarts=0
cd "$ROOT/pocket-tts"
while [ ! -f "$RUN/STOP" ]; do
  attempt_log="$LOG/cooldown_$(date +%Y%m%d_%H%M%S).log"
  log "starting cooldown ($CONFIG), log $attempt_log"
  uv run python -m training.train "$CONFIG" > "$attempt_log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then log "cooldown finished"; break; fi
  restarts=$((restarts + 1))
  log "crash (exit $code), see $attempt_log"
  if [ $restarts -ge 10 ]; then log "too many restarts, giving up"; exit 1; fi
  sleep 60
done
