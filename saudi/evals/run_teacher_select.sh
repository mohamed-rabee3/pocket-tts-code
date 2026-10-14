#!/usr/bin/env bash
# Teacher 140k with the same 3-seed protocol as the student selection.
cd /home/ai2/pocket-tts-training/pocket-tts
export HF_TOKEN=$(cat ../.hf_token)
R=/home/ai2/pocket-tts-training
uv run --with librosa python ../scripts/eval_testset.py /data/saudi-tts/runs/teacher_saved/checkpoint_00140000.pt --name select24l_step140000 \
  --setups continuous:1.0 --seeds 0,1,2 --voices $R/ref_f.wav,$R/ref_f2.wav 2>&1 | grep -E "generations|Traceback|Error"
echo ALL_DONE
