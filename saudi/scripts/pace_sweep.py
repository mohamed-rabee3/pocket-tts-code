"""Continuous mode at several CFG strengths: length, pace and Whisper CER (ref_f2, 2 seeds)."""
import re, sys, torch, numpy as np, soundfile as sf, jiwer
from pocket_tts import TTSModel
from pocket_tts.continuous import VoiceAgentTTS, chunk_text_stream
from pocket_tts.arabic_text import normalize_tashkeel, strip_tashkeel
R = "/home/ai2/pocket-tts-training"; ref = f"{R}/ref_f2.wav"; ckpt = sys.argv[1]
text = normalize_tashkeel(open(f"{R}/samples/text.txt").read())
chunks = list(chunk_text_stream([w + " " for w in text.split()]))
m = TTSModel.load_model(config=f"{R}/configs/saudi_inference_24l.yaml", checkpoint=ckpt).to("cuda")
outs = {}
for cfg in (1.0, 1.25, 1.5, 1.75, 2.0):
    for seed in (0, 1):
        torch.manual_seed(seed)
        t = VoiceAgentTTS(m, ref, interleaved=True, cfg_coef=cfg).new_turn()
        y = torch.cat([a for c in chunks for a in t.say(c)]).cpu().numpy()
        outs[(cfg, seed)] = y
        sf.write(f"{R}/samples/pace/cfg{cfg}_take{seed + 1}.wav", y, 24000)
del m; torch.cuda.empty_cache()
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0, dtype=torch.float16)
def norm(t):
    t = re.sub(r"[^ء-ي\s]", "", strip_tashkeel(t))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")): t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()
def sil(y):
    f = 480; e = np.array([np.sqrt(np.mean(y[i:i+f]**2)) for i in range(0, len(y)-f, f)]); s = 20*np.log10(e+1e-9) < -45
    return s.sum() * 0.02
for cfg in (1.0, 1.25, 1.5, 1.75, 2.0):
    L = [len(outs[(cfg, s)]) / 24000 for s in (0, 1)]
    S = [sil(outs[(cfg, s)]) for s in (0, 1)]
    C = [jiwer.cer(norm(text), norm(asr({"raw": outs[(cfg, s)], "sampling_rate": 24000}, return_timestamps=True, generate_kwargs={"language": "arabic"})["text"])) for s in (0, 1)]
    print(f"| CFG {cfg:4} | length {L[0]:4.1f}s / {L[1]:4.1f}s | silence {np.mean(S):4.1f}s | CER {C[0]:.3f} / {C[1]:.3f} |", flush=True)
