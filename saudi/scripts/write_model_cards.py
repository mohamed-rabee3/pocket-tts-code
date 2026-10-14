"""Write README.md model cards into the two HF release folders."""

from pathlib import Path

COMMON = """
## Text input

- Saudi (Najdi) dialect Arabic **with tashkeel**, convention: no sukun, no word-final
  case marks, shadda kept (e.g. `هَلَا وَاللّه، كِيف أَقدَر أَخدِمك اليَوم؟`).
  `VoiceAgentTTS` normalizes input with `pocket_tts.arabic_text.normalize_tashkeel`.
- Write numbers as words (no digits) and avoid Latin script.

## Usage

Needs the modified Pocket TTS code (Arabic tokenizer support, Arabic-aware chunking,
`pocket_tts.continuous`). Set `HF_TOKEN` for this private repo.

```python
from pocket_tts import TTSModel
from pocket_tts.continuous import VoiceAgentTTS, chunk_text_stream

model = TTSModel.load_model(config="hf://{repo}/config.yaml"){device}

# Streaming voice agent (recommended): one continuous voice per turn, chunks appended as text arrives.
from huggingface_hub import hf_hub_download
voice = hf_hub_download("{repo}", "voices/ref_f.latents.safetensors")
agent = VoiceAgentTTS(model, voice, interleaved=True, cfg_coef=1.0)
turn = agent.new_turn()
for chunk in chunk_text_stream(llm_token_stream):   # iterable of text deltas
    for pcm in turn.say(chunk):                     # 24 kHz float tensors
        play(pcm)

# Plain Pocket TTS API with the built-in voice (KV state, tied to these weights):
state = model.get_state_for_audio_prompt(hf_hub_download("{repo}", "voices/ref_f.safetensors"))
audio = model.generate_audio(state, "هَلَا وَاللّه، كِيف أَقدَر أَخدِمك اليَوم؟")
```

## Files

| file | content |
|---|---|
| `model.safetensors` | inference weights (EMA), Pocket TTS format (`flow_lm.*` + `mimi.*`) |
| `config.yaml` | inference config, points at the files of this repo |
| `tokenizer.json` | 4000-token Saudi Arabic tokenizer |
| `voices/ref_f.latents.safetensors` | built-in voice `ref_f` for `VoiceAgentTTS` (continuous mode) |
| `voices/ref_f.safetensors` | built-in voice `ref_f` for the plain API (default mode) |
| `training/{ckpt}` | raw training checkpoint (model + EMA) for further fine-tuning |

## Evaluation

80-item Saudi test set (written short/medium/long + held-out real speakers), voices
`ref_f`/`ref_f2`, continuous mode CFG 1.0, 3 seeds (384 generations), transcribed with
Cohere Transcribe Arabic:

{eval}

## License and use

Fine-tuned from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) (CC-BY-4.0, © Kyutai);
these weights are released under the same license. Kyutai's use terms apply: no voice
cloning or impersonation without explicit, lawful consent, no deceptive use.
"""

TEACHER = """---
license: cc-by-4.0
language: [ar]
library_name: pocket-tts
base_model: kyutai/pocket-tts
tags: [text-to-speech, arabic, saudi, najdi, streaming]
---
# Saudi Pocket TTS — 24-layer (GPU)

Kyutai Pocket TTS (`english_2026-09_24l`) fine-tuned on ~600 h of Saudi (mostly Najdi)
speech with a new Arabic tokenizer and interleaved multi-chunk training, for seamless
streaming in a voice agent. Step 140,000 (constant LR 2e-4). Teacher of the 6-layer
CPU model [Rabe3/saudi-pocket-tts-6l-student](https://huggingface.co/Rabe3/saudi-pocket-tts-6l-student).
Run at CFG 1.0 (no guidance); ~42-50 s pace for a 600-letter paragraph.
"""

STUDENT = """---
license: cc-by-4.0
language: [ar]
library_name: pocket-tts
base_model: kyutai/pocket-tts
tags: [text-to-speech, arabic, saudi, najdi, streaming, distillation]
---
# Saudi Pocket TTS — 6-layer (CPU)

6-layer Pocket TTS (`english_2026-09` architecture) depth-distilled from the 24-layer
Saudi teacher [Rabe3/saudi-pocket-tts-24l-teacher](https://huggingface.co/Rabe3/saudi-pocket-tts-24l-teacher)
(step 140k, teacher at CFG 1.0; student seeded from teacher layers 0,1,2,21,22,23,
backbone + EOS-probe regression, flow/EOS heads frozen). Step 150,000 of 200,000,
picked on the test set below. Same text convention, voices and API as the teacher.
"""

EVAL = """| model | CER | CER (items without numbers) | short phrases CER | failed | speaker sim |
|---|---|---|---|---|---|
| 24L teacher, step 140k | 0.040 | 0.027 | 0.046 | 1 / 384 | 0.888 |
| 6L student, step 150k | 0.031 | 0.016 | 0.024 | 1 / 384 | 0.891 |

CER is inflated on number items because the ASR writes digits. Remaining failure mode
on both: occasionally a dropped or added word on very short phrases."""

for folder, head, repo, ckpt, device in (
    ("teacher24l", TEACHER, "Rabe3/saudi-pocket-tts-24l-teacher", "checkpoint_00140000.pt", '.to("cuda")'),
    ("student6l", STUDENT, "Rabe3/saudi-pocket-tts-6l-student", "checkpoint_00150000.pt", "  # CPU"),
):
    body = COMMON.replace("{repo}", repo).replace("{ckpt}", ckpt).replace("{device}", device).replace("{eval}", EVAL)
    Path(f"/data/saudi-tts/hf_upload/{folder}/README.md").write_text(head + body)
print("ok")
