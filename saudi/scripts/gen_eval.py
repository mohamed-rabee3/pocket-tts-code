"""Generate the user's agent text from a checkpoint and score it with Whisper.

usage: gen_eval.py <checkpoint.pt> <tag> [reference.wav]
Writes samples/user_text/<tag>_*.wav: Pocket default mode (2 takes) and the
continuous interleaved mode with CFG 2.0 (1 take).
"""

import json
import re
import sys

import jiwer
import soundfile as sf
import torch

from pocket_tts import TTSModel
from pocket_tts.arabic_text import normalize_tashkeel, strip_tashkeel
from pocket_tts.continuous import VoiceAgentTTS, chunk_text_stream

ckpt, tag = sys.argv[1], sys.argv[2]
ROOT = "/home/ai2/pocket-tts-training"
OUT = f"{ROOT}/samples/user_text"
ref = sys.argv[3] if len(sys.argv) > 3 else json.load(open(f"{ROOT}/samples/eval_set.json"))["ref"]
text = normalize_tashkeel(open(f"{ROOT}/samples/text.txt").read())

m = TTSModel.load_model(config=f"{ROOT}/configs/saudi_inference_24l.yaml", checkpoint=ckpt).to("cuda")
outs = {}
vs = m.get_state_for_audio_prompt(ref)
for take in (1, 2):
    torch.manual_seed(take - 1)
    outs[f"{tag}_default_take{take}"] = torch.cat(
        [a.cpu() for a in m.generate_audio_stream(vs, text)]
    ).numpy()
torch.manual_seed(0)
turn = VoiceAgentTTS(m, ref, interleaved=True, cfg_coef=2.0).new_turn()
chunks = chunk_text_stream([w + " " for w in text.split()])
outs[f"{tag}_continuous_cfg2"] = torch.cat([a for c in chunks for a in turn.say(c)]).cpu().numpy()
for name, y in outs.items():
    sf.write(f"{OUT}/{name}.wav", y, 24000)
del m
torch.cuda.empty_cache()

from transformers import pipeline  # noqa: E402

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device=0,
    torch_dtype=torch.float16,
)


def norm(t: str) -> str:
    t = re.sub(r"[^ء-ي\s]", "", strip_tashkeel(t))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


for name, y in outs.items():
    # Whisper's sequential long-form decoding (return_timestamps, no chunk_length_s).
    hyp = asr(
        {"raw": y, "sampling_rate": 24000},
        return_timestamps=True,
        generate_kwargs={"language": "arabic"},
    )["text"]
    print(f"{name}: {len(y) / 24000:.1f}s, CER {jiwer.cer(norm(text), norm(hyp)):.3f}")
    print(f"    {hyp.strip()}")
