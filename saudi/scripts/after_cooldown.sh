#!/usr/bin/env bash
# Waits for the cooldown to finish, then compares 140k (before) with cooldown checkpoints.
ROOT=/home/ai2/pocket-tts-training
until grep -qE "cooldown finished|giving up" /data/saudi-tts/logs/cooldown_supervisor.log; do sleep 120; done
grep -q "giving up" /data/saudi-tts/logs/cooldown_supervisor.log && exit 1
C=/data/saudi-tts/runs/teacher_cooldown
cd $ROOT/pocket-tts
HF_TOKEN=$(cat ../.hf_token) uv run --with librosa python ../scripts/compare_ckpts.py compare_cooldown \
  /data/saudi-tts/runs/teacher_saved/checkpoint_00140000.pt $C/checkpoint_00160000.pt $C/checkpoint_00165000.pt \
  > $ROOT/evals/compare_cooldown.log 2>&1
