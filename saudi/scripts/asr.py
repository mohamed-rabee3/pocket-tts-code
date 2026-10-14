"""Arabic ASR for scoring generations: Cohere Transcribe Arabic (07-2026).

Better than Whisper on Najdi (dialect words, no skipped openings). Long audio is
split by the processor and reassembled. Needs librosa: run scripts with
`uv run --with librosa ...` (keeps the training venv unchanged).

    asr = CohereASR()
    text = asr(y, sr)
"""

import re

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"

# It writes some spoken numbers as digits; map them back to the words the TTS read.
_NUMBER_WORDS = {"60": "سكستي"}


class CohereASR:
    def __init__(self, device: str = "cuda") -> None:
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def __call__(self, y: np.ndarray, sr: int) -> str:
        y = y if y.ndim == 1 else y.mean(1)
        if len(y) < sr // 4:
            return ""
        if sr != 16000:
            g = np.gcd(sr, 16000)
            y = resample_poly(y, 16000 // g, sr // g)
        inputs = self.proc(audio=y.astype(np.float32), sampling_rate=16000, return_tensors="pt", language="ar")
        idx = inputs.get("audio_chunk_index")
        inputs.to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=256)
        text = self.proc.decode(out, skip_special_tokens=True, audio_chunk_index=idx, language="ar")
        text = text[0] if isinstance(text, list) else text
        return re.sub(r"\d+", lambda m: _NUMBER_WORDS.get(m.group(0), m.group(0)), text)
