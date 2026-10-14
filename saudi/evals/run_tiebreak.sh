#!/usr/bin/env bash
cd /home/ai2/pocket-tts-training/pocket-tts
export HF_TOKEN=$(cat ../.hf_token)
V=/home/ai2/pocket-tts-training/ref_f.wav,/home/ai2/pocket-tts-training/ref_f2.wav
uv run --with librosa python ../scripts/eval_testset.py /data/saudi-tts/runs/teacher_saved/checkpoint_00140000.pt --name tb_step140000 --setups continuous:1.0 --seeds 0 --voices $V
uv run --with librosa python ../scripts/eval_testset.py /data/saudi-tts/runs/teacher_cooldown/checkpoint_00165000.pt --name tb_cooldown165000 --setups continuous:1.0 --seeds 0 --voices $V
echo ALL_DONE
