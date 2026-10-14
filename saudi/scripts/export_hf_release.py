"""Build a self-contained Hugging Face release folder for one trained model.

usage: export_hf_release.py <out_dir> <checkpoint.pt> <inference_config.yaml> <hf_repo_id> <voice_tag>

Writes to <out_dir>:
  model.safetensors   pocket-tts format (flow_lm.* EMA weights + mimi.*)
  tokenizer.json      the Saudi tokenizer
  config.yaml         inference config pointing at hf://<hf_repo_id>/{model.safetensors,tokenizer.json}
  config_local.yaml   same, pointing at the files next to it (for verification / offline use)
  voices/ref_f.safetensors, voices/ref_f.latents.safetensors   built-in voice
then checks that config_local.yaml loads the exact same weights as the checkpoint.
"""

import shutil
import sys
from pathlib import Path

import safetensors.torch
import torch
import yaml

from pocket_tts import TTSModel

out, ckpt, cfg_path, repo, voice_tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
ROOT = Path("/home/ai2/pocket-tts-training")
out.mkdir(parents=True, exist_ok=True)
(out / "voices").mkdir(exist_ok=True)

model = TTSModel.load_model(config=cfg_path, checkpoint=ckpt)
state = {k: v.detach().float().contiguous().cpu() for k, v in model.state_dict().items() if k.startswith(("flow_lm.", "mimi."))}
safetensors.torch.save_file(state, str(out / "model.safetensors"))

cfg = yaml.safe_load(Path(cfg_path).read_text())
shutil.copy(cfg["flow_lm"]["lookup_table"]["tokenizer_path"], out / "tokenizer.json")
cfg.pop("weights_path_without_voice_cloning", None)
for name, prefix in (("config.yaml", f"hf://{repo}/"), ("config_local.yaml", f"{out}/")):
    cfg["weights_path"] = prefix + "model.safetensors"
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = prefix + "tokenizer.json"
    (out / name).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
shutil.copy(ROOT / "voices" / voice_tag / "ref_f.safetensors", out / "voices/ref_f.safetensors")
shutil.copy(ROOT / "voices/ref_f.latents.safetensors", out / "voices/ref_f.latents.safetensors")

reloaded = TTSModel.load_model(config=str(out / "config_local.yaml")).state_dict()
bad = [k for k in state if k not in reloaded or not torch.equal(state[k], reloaded[k].float())]
missing = [k for k in reloaded if k.startswith(("flow_lm.", "mimi.")) and k not in state]
print(f"{len(state)} tensors exported; mismatched {len(bad)}; missing {len(missing)}", bad[:5], missing[:5])
assert not bad and not missing
print("VERIFIED", out)
