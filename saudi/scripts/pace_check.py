"""User's agent text (samples/text.txt) in default and continuous CFG 1.0 mode, 2 takes each:
length, speaking rate, silence, long pauses and CER (Cohere Transcribe Arabic).
Run with `uv run --with librosa`.

usage: pace_check.py <checkpoint.pt> <reference.wav> [inference_config.yaml] [tag]
Writes samples/pace_check/<tag>step<N>_<voice>_<mode>_take<k>.wav (default config:
configs/saudi_inference_24l.yaml, tag empty; e.g. configs/saudi_inference_6l.yaml 6l_).
"""

import re
import sys
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch

from pocket_tts import TTSModel
from pocket_tts.arabic_text import normalize_tashkeel, strip_tashkeel
from pocket_tts.continuous import VoiceAgentTTS, chunk_text_stream

ck, ref = sys.argv[1], sys.argv[2]
R = "/home/ai2/pocket-tts-training"
config = sys.argv[3] if len(sys.argv) > 3 else f"{R}/configs/saudi_inference_24l.yaml"
tag = sys.argv[4] if len(sys.argv) > 4 else ""
step = int(re.search(r"(\d+)\.pt", ck).group(1))
voice = Path(ref).stem
O = f"{R}/samples/pace_check"
Path(O).mkdir(parents=True, exist_ok=True)
text = normalize_tashkeel(open(f"{R}/samples/text.txt").read())
chunks = list(chunk_text_stream([w + " " for w in text.split()]))

m = TTSModel.load_model(config=config, checkpoint=ck).to("cuda")
vs = m.get_state_for_audio_prompt(Path(ref))
outs = {}
for take in (1, 2):
    torch.manual_seed(take - 1)
    outs[f"default_take{take}"] = torch.cat([a.cpu() for a in m.generate_audio_stream(vs, text)]).numpy()
    torch.manual_seed(take - 1)
    turn = VoiceAgentTTS(m, ref, interleaved=True, cfg_coef=1.0).new_turn()
    outs[f"continuous_cfg1_take{take}"] = torch.cat([a for c in chunks for a in turn.say(c)]).cpu().numpy()
for n, y in outs.items():
    sf.write(f"{O}/{tag}step{step}_{voice}_{n}.wav", y, 24000)
del m
torch.cuda.empty_cache()

from asr import CohereASR  # noqa: E402

asr = CohereASR()


def norm(t: str) -> str:
    t = re.sub(r"[^ء-ي\s]", "", strip_tashkeel(t))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


n_letters = len(re.sub(r"[^ء-ي]", "", strip_tashkeel(text)))
print(f"{tag}step {step}, voice {voice}")
for n, y in outs.items():
    f = 480
    e = np.array([np.sqrt(np.mean(y[i : i + f] ** 2)) for i in range(0, len(y) - f, f)])
    sil = 20 * np.log10(e + 1e-9) < -45
    runs, c = [], 0
    for v in list(sil) + [False]:
        if v:
            c += 1
        else:
            if c >= 5:
                runs.append(c * 0.02)
            c = 0
    sp = (~sil).sum() * 0.02
    hyp = asr(y, 24000)
    print(
        f"| {n:22s} | {len(y) / 24000:5.1f}s | speech {sp:5.1f}s ({n_letters / sp:4.1f} letters/s) | "
        f"silence {len(y) / 24000 - sp:4.1f}s | pauses>=0.6s {sum(r >= 0.6 for r in runs):2d} | "
        f"CER {jiwer.cer(norm(text), norm(hyp)):.3f} |"
    )
