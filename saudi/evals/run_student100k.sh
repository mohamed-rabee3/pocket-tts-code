#!/usr/bin/env bash
# Student (6L) at step 100k vs the 140k teacher: user text pace/CER, then the 80-item test set.
cd /home/ai2/pocket-tts-training/pocket-tts
export HF_TOKEN=$(cat ../.hf_token)
R=/home/ai2/pocket-tts-training
S=/data/saudi-tts/runs/distill_6l/checkpoint_00100000.pt
T=/data/saudi-tts/runs/teacher_saved/checkpoint_00140000.pt
for v in ref_f ref_f2; do
  uv run --with librosa python ../scripts/pace_check.py $S $R/$v.wav $R/configs/saudi_inference_6l.yaml 6l_ 2>&1 | grep -E "^\||step|Traceback|Error"
done
for v in; do
  uv run --with librosa python ../scripts/pace_check.py $T $R/$v.wav 2>&1 | grep -E "^\||step|Traceback|Error"
done
uv run --with librosa python ../scripts/eval_testset.py $S --name student6l_step100000 --config $R/configs/saudi_inference_6l.yaml \
  --setups continuous:1.0 --seeds 0 --voices $R/ref_f.wav,$R/ref_f2.wav 2>&1 | grep -E "generations|Traceback|Error" 
echo ALL_DONE
