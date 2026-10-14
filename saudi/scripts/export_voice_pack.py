"""Add a built-in voice to a trained model.

usage: export_voice_pack.py <model_tag> <checkpoint.pt or model.safetensors> <voice_name> <reference.wav>

Writes, under /home/ai2/pocket-tts-training/voices/:
  <model_tag>/<voice_name>.safetensors   Pocket default mode: the voice's model state
                                         (KV cache). Tied to these weights, so export
                                         again for every final model (24L GPU, 6L CPU).
  <voice_name>.latents.safetensors       Continuous mode (VoiceAgentTTS): the voice's
                                         Mimi latents; independent of the weights.
Use:
  m.generate_audio_stream(m.get_state_for_audio_prompt("voices/<tag>/<name>.safetensors"), text)
  VoiceAgentTTS(m, "voices/<name>.latents.safetensors", interleaved=True, cfg_coef=...)
"""

import sys
from pathlib import Path

from pocket_tts import TTSModel
from pocket_tts.continuous import VoiceAgentTTS
from pocket_tts.models.model_state import export_model_state

tag, weights, name, wav = sys.argv[1:5]
ROOT = Path("/home/ai2/pocket-tts-training")
out = ROOT / "voices" / tag
out.mkdir(parents=True, exist_ok=True)

model = TTSModel.load_model(
    config=str(ROOT / "configs/saudi_inference_24l.yaml") if len(sys.argv) < 6 else sys.argv[5],
    checkpoint=weights,
)
state = model.get_state_for_audio_prompt(audio_conditioning=Path(wav), truncate=True)
export_model_state(state, out / f"{name}.safetensors")
VoiceAgentTTS(model, wav).save_voice(ROOT / "voices" / f"{name}.latents.safetensors")
print(f"wrote {out / f'{name}.safetensors'} and voices/{name}.latents.safetensors")
