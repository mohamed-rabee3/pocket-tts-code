#!/usr/bin/env bash
# Final student selection: 200k/150k/100k/50k on the 80-item test set, 3 seeds, + user-text pace for 150k/200k.
cd /home/ai2/pocket-tts-training/pocket-tts
export HF_TOKEN=$(cat ../.hf_token)
R=/home/ai2/pocket-tts-training
C=$R/configs/saudi_inference_6l.yaml
D=/data/saudi-tts/runs/distill_6l
for st in 200000 150000; do
  for v in ref_f ref_f2; do
    uv run --with librosa python ../scripts/pace_check.py $D/checkpoint_00$st.pt $R/$v.wav $C 6l_ 2>&1 | grep -E "^\||step|Traceback|Error"
  done
done
for st in 200000 150000 100000 050000; do
  uv run --with librosa python ../scripts/eval_testset.py $D/checkpoint_00$st.pt --name select6l_step$st --config $C \
    --setups continuous:1.0 --seeds 0,1,2 --voices $R/ref_f.wav,$R/ref_f2.wav 2>&1 | grep -E "generations|Traceback|Error"
done
echo ALL_DONE
