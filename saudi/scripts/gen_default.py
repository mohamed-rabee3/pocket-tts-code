"""Generate the user's text (with the hesitation) in Pocket's default mode and score it."""
import json, re, sys, torch, soundfile as sf, jiwer
from pocket_tts import TTSModel
from pocket_tts.arabic_text import normalize_tashkeel, strip_tashkeel

ckpt, step = sys.argv[1], sys.argv[2]
O = "/home/ai2/pocket-tts-training/samples/user_text"
ev = json.load(open("/home/ai2/pocket-tts-training/samples/eval_set.json"))
text = normalize_tashkeel(open("/home/ai2/pocket-tts-training/samples/text.txt").read())
m = TTSModel.load_model(config="/home/ai2/pocket-tts-training/configs/saudi_inference_24l.yaml", checkpoint=ckpt).to("cuda")
vs = m.get_state_for_audio_prompt(ev["ref"])
outs = {}
for seed in (0, 1):
    torch.manual_seed(seed)
    y = torch.cat([a.cpu() for a in m.generate_audio_stream(vs, text)]).numpy()
    name = f"D_default_with_hesitation_step{step}_take{seed + 1}"
    sf.write(f"{O}/{name}.wav", y, 24000); outs[name] = y
del m; torch.cuda.empty_cache()
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0, torch_dtype=torch.float16)
def norm(t):
    t = re.sub(r"[^ء-ي\s]", "", strip_tashkeel(t))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")): t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()
for name, y in outs.items():
    hyp = asr({"raw": y, "sampling_rate": 24000}, generate_kwargs={"language": "arabic"}, chunk_length_s=30)["text"]
    i = hyp.find("بكثر")
    print(f"{name}: {len(y)/24000:.1f}s, CER {jiwer.cer(norm(text), norm(hyp)):.2f} | around hesitation: {hyp[max(0,i-15):i+45] if i>=0 else '(بكثر not found)'}")
