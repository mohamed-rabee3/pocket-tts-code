# Saudi Arabic Pocket TTS: training kit

The scripts and configs used to train the Saudi (Najdi) Pocket TTS models:

- **24-layer (GPU):** [Rabe3/saudi-pocket-tts-24l-teacher](https://huggingface.co/Rabe3/saudi-pocket-tts-24l-teacher), step 140k.
- **6-layer (CPU):** [Rabe3/saudi-pocket-tts-6l-student](https://huggingface.co/Rabe3/saudi-pocket-tts-6l-student), step 150k.

Changes to the Pocket TTS code itself are in the parent folders:

- `pocket_tts/arabic_text.py`: tashkeel normalization.
- `pocket_tts/continuous.py`: `VoiceAgentTTS`, for seamless streaming in a voice agent.
- `pocket_tts/models/text_chunking.py`: Arabic punctuation (`؟ ، ؛ ..`).
- Training:
  - Interleaved multi-chunk training in `training/modules/conditioner.py` and `training/dataloader/`.
  - Tashkeel dropout.
  - A schedule offset (`optim.schedule_start_step`) for cooldowns.
  - Depth distillation with the null pass skipped at CFG 1.0.
- `training/tests/test_interleaved.py`: checks that the training layout matches streaming inference.

Paths in these files are the training server's: data under `/data/saudi-tts`, the kit under `/home/ai2/pocket-tts-training`. Adjust them before reuse. Tokens are read from local files (`.hf_token`) and are not part of this repo. Data, checkpoints and audio are not included.

## Pipeline

1. **Data.** `scripts/download.sh` pulls the data from the Storage Box. `scripts/prepare_all.sh` then verifies it, builds segments, runs word alignment and trains the 4,000-token tokenizer.
2. **Teacher.** `configs/saudi_teacher.yaml` fine-tunes from `english_2026-09_24l`, run by `scripts/train_supervisor.sh`. The constant learning rate of 2e-4 overfits after about 150k steps, so step 140k was kept.
3. **Cooldown (tested, not used).** `configs/saudi_teacher_cooldown.yaml` with `scripts/run_cooldown.sh`. It had better validation loss but more dropped words on short phrases.
4. **Distillation.** `configs/saudi_distill_6l.yaml` with `scripts/run_distill.sh` distills the 6-layer student from teacher step 140k at CFG 1.0, for 200k steps.
5. **Evaluation.**
   - `scripts/eval_testset.py`: 80-item test set.
   - `scripts/compare_ckpts.py`: chunk-to-chunk voice consistency.
   - `scripts/pace_check.py`: pace on the reference paragraph.
   - All three transcribe with Cohere Transcribe Arabic (`scripts/asr.py`); run them with `uv run --with librosa`.
6. **Release.**
   - `scripts/export_voice_pack.py`: built-in voices.
   - `scripts/export_hf_release.py`: Hugging Face release folder, verified against the checkpoint.
   - `scripts/write_model_cards.py`: model cards.

Inference configs: `configs/saudi_inference_24l.yaml` and `configs/saudi_inference_6l.yaml`. Load them with `TTSModel.load_model(config=..., checkpoint=...)`.

Base model and code: [kyutai/pocket-tts](https://github.com/kyutai-labs/pocket-tts). The code is MIT and the weights CC-BY-4.0. Do not clone or impersonate a voice without consent.
