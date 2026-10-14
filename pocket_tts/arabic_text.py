"""Saudi-dialect Arabic text normalization, shared by training and inference.

The Saudi corpora mix two tashkeel conventions: human labels without case
endings and with sparse sukun, and ASR-diacritized text with case endings and
sukun on almost every consonant. Training on both would give the model two
spellings for one sound, so every transcript, and every text sent to the model
at inference, goes through `normalize_tashkeel`:

- NFC, so a vowel and shadda on one letter always come in the same order.
- Tatweel and Quranic annotation marks are removed.
- Sukun is removed everywhere (an unvowelled consonant already means "no vowel").
- The vowel / tanween marks on the last letter of each word are removed; a final
  shadda is kept. Tanween before a final alif (شُكْرًا) is not on the last letter
  and stays.
- Latin , ? ; become the Arabic ، ؟ ؛ and whitespace is collapsed.

The LLM feeding the TTS should write text that is already in this form, but
it does not have to: anything else is mapped onto it here.
"""

import random
import re
import unicodedata

TANWEEN_AND_VOWELS = "ًٌٍَُِ"  # ً ٌ ٍ َ ُ ِ
SHADDA = "ّ"
SUKUN = "ْ"
DAGGER_ALIF = "ٰ"
ALL_MARKS = TANWEEN_AND_VOWELS + SHADDA + SUKUN + DAGGER_ALIF

_REMOVE = re.compile("[ـؐ-ؚۖ-ۭ" + SUKUN + "]")
_LETTERS = "ء-غف-يٱ-ۓ"  # incl. گ ڤ چ, which Saudi text uses
_WORD_FINAL_MARKS = re.compile(f"[{TANWEEN_AND_VOWELS}]+(?=[{SHADDA}]?(?![{_LETTERS}{ALL_MARKS}]))")
_ANY_MARK = re.compile(f"[{ALL_MARKS}]")
_PUNCT = str.maketrans({",": "،", "?": "؟", ";": "؛"})
_SPACES = re.compile(r"\s+")


def normalize_tashkeel(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _REMOVE.sub("", text)
    text = _WORD_FINAL_MARKS.sub("", text)
    text = text.translate(_PUNCT)
    return _SPACES.sub(" ", text).strip()


def strip_tashkeel(text: str) -> str:
    """All diacritics removed: the letter skeleton."""
    return _ANY_MARK.sub("", text)


def tashkeel_dropout(text: str, rng: random.Random, p_all: float, p_word: float) -> str:
    """Training augmentation: with probability p_all drop every mark, else drop
    each word's marks with probability p_word, so missing tashkeel at inference
    stays in distribution."""
    if p_all > 0 and rng.random() < p_all:
        return strip_tashkeel(text)
    if p_word > 0:
        return " ".join(strip_tashkeel(w) if rng.random() < p_word else w for w in text.split(" "))
    return text
