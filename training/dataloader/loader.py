"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import itertools
import json
import logging
import queue
import random
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch
from safetensors import safe_open

from pocket_tts.arabic_text import tashkeel_dropout

from . import audio
from .manifest import LazyEntries, load_entries
from .types import Batch, Entry

logger = logging.getLogger(__name__)


def _prefetch(iterator: Iterator[Batch], depth: int = 4) -> Iterator[Batch]:
    """Run the (synchronous, IO-bound) loader in a background thread."""
    q: queue.Queue[Batch | BaseException | None] = queue.Queue(maxsize=depth)

    def worker():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as exc:  # noqa: BLE001 -- re-raised in the consumer
            q.put(exc)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class UnalignedEntry(ValueError):
    """A manifest entry without word alignments (see training/scripts/align_data.py)."""


class DataLoader:
    # Augmentation knobs; class defaults keep instances built without __init__ (tests) valid.
    prompt_trim_max_sec: float = 0.0
    prompt_trim_prob: float = 1.0
    final_punct_dropout: float = 0.0
    tashkeel_dropout_all: float = 0.0
    tashkeel_dropout_word: float = 0.0
    ref_only_prob: float = 0.0
    ref_ctx_prob: float = 0.0
    min_ref_prompt_sec: float = 1.5
    max_ref_prompt_sec: float = 5.0
    by_speaker: ClassVar[dict[str, list[int]]] = {}
    interleave_probs: ClassVar[list[float]] = [1.0]
    interleave_min_piece_sec: float = 1.0
    interleave_punct_weight: float = 4.0

    def __init__(
        self,
        jsonl: str,
        tokenize: Callable[[str], list[int]],
        batch_size: int,
        sample_rate: int,
        frame_rate: float,
        max_duration_sec: float,
        max_voice_prompt_sec: float,
        rank: int,
        world_size: int,
        seed: int = 0,
        shuffle: bool = True,
        io_workers: int = 16,
        num_bucket_batches: int = 1,
        prompt_trim_max_sec: float = 0.0,
        final_punct_dropout: float = 0.0,
        prompt_trim_prob: float = 1.0,
        tashkeel_dropout_all: float = 0.0,
        tashkeel_dropout_word: float = 0.0,
        ref_only_prob: float = 0.0,
        ref_ctx_prob: float = 0.0,
        min_ref_prompt_sec: float = 1.5,
        max_ref_prompt_sec: float = 5.0,
        interleave_probs: list[float] | None = None,
        interleave_min_piece_sec: float = 1.0,
        interleave_punct_weight: float = 4.0,
    ):
        self.jsonl = jsonl
        self.interleave_probs = list(interleave_probs or [1.0])
        self.interleave_min_piece_sec = interleave_min_piece_sec
        self.interleave_punct_weight = interleave_punct_weight
        self.prompt_trim_prob = prompt_trim_prob
        self.tashkeel_dropout_all = tashkeel_dropout_all
        self.tashkeel_dropout_word = tashkeel_dropout_word
        self.ref_only_prob = ref_only_prob
        self.ref_ctx_prob = ref_ctx_prob
        self.min_ref_prompt_sec = min_ref_prompt_sec
        self.max_ref_prompt_sec = max_ref_prompt_sec
        self.num_bucket_batches = num_bucket_batches
        self.entries = load_entries(jsonl, rank, world_size)
        self.tokenize = tokenize
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.max_duration_sec = max_duration_sec
        self.max_voice_prompt_sec = max_voice_prompt_sec
        self.prompt_trim_max_sec = prompt_trim_max_sec
        self.final_punct_dropout = final_punct_dropout
        self.shuffle = shuffle
        self.io_workers = io_workers
        self._failures = 0
        self.rng = random.Random(seed)
        self.frame_size = int(sample_rate / frame_rate)
        meta_path = Path(jsonl).with_suffix(".meta.json")
        self.stitch_frames = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.stitch_frames = int(meta["stitch_frames"])
            self.latents_root = Path(jsonl).parent
            logger.info(f"latent mode: stitch_frames={self.stitch_frames} from {meta_path.name}")
        if self.ref_only_prob + self.ref_ctx_prob > 0:
            self.by_speaker = self._index_speakers()

    def _index_speakers(self) -> dict[str, list[int]]:
        """speaker -> entry indices of this shard, for reference voice prompts."""
        self.all_entries = load_entries(self.jsonl, 0, 1)  # references may sit in any shard
        by_speaker: dict[str, list[int]] = {}
        for i in range(len(self.all_entries)):
            speaker = json.loads(self.all_entries[i]).get("speaker")
            if speaker:
                by_speaker.setdefault(speaker, []).append(i)
        multi = sum(len(v) > 1 for v in by_speaker.values())
        logger.info(f"reference prompts: {len(by_speaker)} speakers, {multi} with 2+ entries")
        return by_speaker

    MIN_CUT_SEC = 1.0  # keep at least this much audio on both sides of a cut
    TRAIL_SEC = 0.2  # silence kept after the last word, so EOS has a consistent target

    def _cap_prompt(self, prompt: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], int]:
        """Truncate the prompt to the configured cap; collation pads to batch max."""
        if self.max_voice_prompt_sec > 0:
            prompt_samples = int(self.max_voice_prompt_sec * self.sample_rate)
            prompt = prompt[:prompt_samples]
        if self.prompt_trim_max_sec > 0:
            drop = int(self.rng.uniform(0.0, self.prompt_trim_max_sec) * self.sample_rate)
            prompt = prompt[: max(len(prompt) - drop, min(len(prompt), self.sample_rate))]
        return prompt, len(prompt)

    def _tokens(self, text: str) -> torch.Tensor:
        """Tokenize the target text, dropping a final period with probability
        final_punct_dropout so unpunctuated prompts are in distribution. Only the
        period goes: "?" and "!" change how the sentence is said."""
        if self.final_punct_dropout > 0 and self.rng.random() < self.final_punct_dropout:
            text = re.sub(r"\.\s*$", "", text) or text
        if self.tashkeel_dropout_all > 0 or self.tashkeel_dropout_word > 0:
            text = tashkeel_dropout(
                text, self.rng, self.tashkeel_dropout_all, self.tashkeel_dropout_word
            )
        return torch.tensor(self.tokenize(text), dtype=torch.long)

    @staticmethod
    def _last_word_end(entry: Entry) -> float | None:
        """End of the last aligned word, i.e. where the speech actually stops.

        None when the entry carries no alignment, which is the documented
        manifest format without `words`.
        """
        if not entry.words:
            return None
        ends = [w["end"] for w in entry.words if w.get("end") is not None]
        return max(ends) if ends else None

    def _choose_cut(self, entry: Entry) -> tuple[float, str] | None:
        """(cut in seconds, transcript of the words after it), None without alignment."""
        # Cut the utterance at a random point between two aligned words:
        # audio before the cut = voice conditioning, audio after = target,
        # paired with the remaining words as text (see training/scripts/align_data.py).
        if not entry.words:
            # Without a boundary to cut at, the prompt would be a window of the target
            # itself and the model would learn to copy it instead of reading the text.
            raise UnalignedEntry(
                f"{entry.path}: no word alignments; run training/scripts/align_data.py first"
            )
        cuts = []
        for i in range(1, len(entry.words)):
            prev, cur = entry.words[i - 1], entry.words[i]
            if prev["end"] is None or cur["start"] is None:
                continue
            cut = 0.5 * (prev["end"] + cur["start"])
            if cut >= self.MIN_CUT_SEC and entry.duration - cut >= self.MIN_CUT_SEC:
                cuts.append((cut, i))
        if cuts:
            # The prompt is the (contiguous) start of the utterance.
            # Eligible cuts are word boundaries whose preceding word ends
            # inside the prompt window; the draw is uniform over 1..k
            # eligible words, so prompt length varies and the target
            # keeps most of the utterance.
            window = self.max_voice_prompt_sec if self.max_voice_prompt_sec > 0 else float("inf")
            eligible = [
                (c, i)
                for c, i in cuts
                if entry.words[i - 1].get("end") is not None and entry.words[i - 1]["end"] < window
            ]
            cuts = eligible or cuts[:1]  # degenerate windows: earliest valid cut
        if not cuts:
            return None
        cut, i = self.rng.choice(cuts)
        return cut, " ".join(w["word"] for w in entry.words[i:])

    def _sample(self, entry: Entry) -> tuple[Any, ...]:
        """(wav, tokens, prompt wav, prompt samples), or _sample_latent's tuple."""
        if entry.latents_file is not None:
            return self._sample_latent(entry)
        chosen = self._choose_cut(entry)
        if chosen is not None:
            cut, text = chosen
            tokens = self._tokens(text)
            # Trim to the last word (plus a short tail) rather than the end
            # of the file: 12% of utterances carry >1s of trailing silence,
            # and training on it teaches the model to emit silence instead of
            # EOS, so generations never terminate. dora does the same thing
            # via align_wav_on_words.
            end = entry.duration
            last = self._last_word_end(entry)
            if last is not None and last > cut:
                end = min(entry.duration, last + self.TRAIL_SEC)
            wav = audio._load_window(
                entry.path,
                entry.start + cut,
                min(end - cut, self.max_duration_sec),
                self.sample_rate,
            )
            # The prompt is everything from the utterance start to the
            # cut (bounded by the window via cut selection).
            prompt, length = self._cap_prompt(
                audio._load_window(entry.path, entry.start, cut, self.sample_rate)
            )
            return wav, tokens, prompt, length
        last = self._last_word_end(entry)
        end = min(entry.duration, last + self.TRAIL_SEC) if last is not None else entry.duration
        duration = min(end, self.max_duration_sec)
        wav = audio._load_window(entry.path, entry.start, duration, self.sample_rate)
        tokens = self._tokens(entry.transcript)
        # No alignment: voice prompt from a random window of the same file.
        fallback_sec = self.max_voice_prompt_sec if self.max_voice_prompt_sec > 0 else 3.0
        prompt_start = self.rng.uniform(0, max(0.0, entry.duration - fallback_sec))
        prompt, length = self._cap_prompt(
            audio._load_window(
                entry.path, entry.start + prompt_start, fallback_sec, self.sample_rate
            )
        )
        return wav, tokens, prompt, length

    def _load_latents(self, latents_file: str) -> torch.Tensor:
        path = self.latents_root / latents_file
        with safe_open(str(path), framework="pt") as f:
            return f.get_tensor("latents")

    def _latent_cut(self, entry: Entry, stored: int) -> tuple[int, str]:
        chosen = self._choose_cut(entry)
        if chosen is None or stored <= 1:
            return 0, entry.transcript
        cut, text = chosen
        cut_frames = min(max(round(cut * self.frame_rate), 1), stored - 1)
        return cut_frames, text

    def _latent_target_frames(self, entry: Entry, cut_frames: int, stored: int) -> int:
        cut_sec = cut_frames / self.frame_rate
        end = entry.duration
        last = self._last_word_end(entry)
        if last is not None and last > cut_sec:
            end = min(entry.duration, last + self.TRAIL_SEC)
        target_frames = int(min(end - cut_sec, self.max_duration_sec) * self.frame_rate)
        return max(1, min(target_frames, stored - cut_frames))

    def _latent_prompt(self, lat: torch.Tensor, cut_frames: int) -> torch.Tensor:
        stored = lat.shape[0]
        cap = (
            max(1, int(self.max_voice_prompt_sec * self.frame_rate))
            if self.max_voice_prompt_sec > 0
            else stored
        )
        if cut_frames > 0:
            # The audio right before the cut, so prompt and target stay contiguous.
            return self._trim_prompt_frames(lat[max(0, cut_frames - cap) : cut_frames])
        start = self.rng.randint(0, max(0, stored - cap))
        return self._trim_prompt_frames(lat[start : start + cap])

    def _trim_prompt_frames(self, prompt: torch.Tensor) -> torch.Tensor:
        """With probability prompt_trim_prob, drop a uniform 0..prompt_trim_max_sec
        tail (in frames), keeping >= 1 s."""
        if self.prompt_trim_max_sec <= 0 or self.rng.random() >= self.prompt_trim_prob:
            return prompt
        drop = self.rng.randint(0, int(round(self.prompt_trim_max_sec * self.frame_rate)))
        keep = max(prompt.shape[0] - drop, min(prompt.shape[0], int(self.frame_rate)))
        return prompt[:keep]

    def _prompt_mode(self) -> str:
        """ "ref": reference clip only (start of an agent turn), "ref_ctx": reference
        clip + the audio just before the target (a later chunk of the turn), "ctx":
        the preceding audio only (upstream behaviour)."""
        u = self.rng.random()
        if u < self.ref_only_prob:
            return "ref"
        if u < self.ref_only_prob + self.ref_ctx_prob:
            return "ref_ctx"
        return "ctx"

    def _ref_prompt(self, entry: Entry) -> torch.Tensor | None:
        """Latents of the start of another utterance of the same speaker, ending in
        the pause after a word (as end_on_pause trims voice prompts at inference)."""
        candidates = self.by_speaker.get(entry.speaker or "", [])
        if len(candidates) < 2:
            return None
        for _ in range(4):
            other = self.get_entry(self.rng.choice(candidates), self.all_entries)
            if other.path != entry.path and other.latents_file is not None:
                break
        else:
            return None
        lo, hi = self.min_ref_prompt_sec, self.max_ref_prompt_sec
        words = [
            w for w in other.words or [] if w.get("start") is not None and w.get("end") is not None
        ]
        ends = [0.5 * (a["end"] + b["start"]) for a, b in itertools.pairwise(words)]
        if words:
            ends.append(min(other.duration, words[-1]["end"] + self.TRAIL_SEC))
        ends = [e for e in ends if lo <= e <= hi]
        end_sec = self.rng.choice(ends) if ends else min(hi, other.duration)
        lat = self._load_latents(other.latents_file)
        return lat[: max(1, min(lat.shape[0], round(end_sec * self.frame_rate)))]

    def _sample_latent(self, entry: Entry) -> tuple[Any, ...]:
        """(stitch wav, tokens, prompt latents, tail latents, target frames)."""
        assert self.stitch_frames > 0, f"{entry.path}: latents entry but no meta file loaded"
        assert entry.latents_file is not None, f"{entry.path}: not a latents entry"
        lat = self._load_latents(entry.latents_file)
        mode = self._prompt_mode()
        ref = self._ref_prompt(entry) if mode != "ctx" else None
        if ref is None:
            mode = "ctx"
        if mode == "ref":
            cut_frames, text = 0, entry.transcript
        else:
            cut_frames, text = self._latent_cut(entry, lat.shape[0])
            if cut_frames == 0:
                # Too short to cut: a prompt from this same file would let the model
                # copy the target, so use a reference clip or skip the entry.
                ref = ref if ref is not None else self._ref_prompt(entry)
                if ref is None:
                    raise ValueError(f"{entry.path}: no word cut and no reference utterance")
                mode = "ref"
        if mode == "ref":
            prompt = ref
        elif mode == "ref_ctx":
            prompt = torch.cat([ref, self._latent_prompt(lat, cut_frames)], dim=0)
        else:
            prompt = self._latent_prompt(lat, cut_frames)
        target_frames = self._latent_target_frames(entry, cut_frames, lat.shape[0])
        pieces = self._interleave_pieces(entry, cut_frames, target_frames)
        if pieces is None:
            tokens, extra = self._tokens(text), []
        else:
            tokens, extra = (
                self._tokens(pieces[0][1]),
                [(f, self._tokens(t)) for f, t in pieces[1:]],
            )
        stitch_frames = min(self.stitch_frames, target_frames)
        stitch = audio._load_window(
            entry.path,
            entry.start + cut_frames / self.frame_rate,
            stitch_frames / self.frame_rate,
            self.sample_rate,
        )
        tail = lat[cut_frames + stitch_frames : cut_frames + target_frames]
        return stitch, tokens, prompt, tail, target_frames, extra

    def _interleave_pieces(
        self, entry: Entry, cut_frames: int, target_frames: int
    ) -> list[tuple[int, str]] | None:
        """Split the target transcript into pieces at mid-pause word boundaries:
        [(start frame within the target, text), ...], the first starting at 0.
        None keeps the single-text layout."""
        if len(self.interleave_probs) <= 1 or not entry.words:
            return None
        n_extra = self.rng.choices(range(len(self.interleave_probs)), self.interleave_probs)[0]
        if n_extra == 0:
            return None
        words = entry.words
        cut_sec = cut_frames / self.frame_rate
        first = 0
        if cut_frames > 0:
            starts = [
                k
                for k, w in enumerate(words)
                if w.get("start") is not None and w["start"] >= cut_sec - 0.01
            ]
            if not starts:
                return None
            first = starts[0]
        target_sec = target_frames / self.frame_rate
        lo = self.interleave_min_piece_sec
        cands, weights = [], []
        for k in range(first + 1, len(words)):
            a, b = words[k - 1], words[k]
            if a.get("end") is None or b.get("start") is None:
                continue
            rel = 0.5 * (a["end"] + b["start"]) - cut_sec
            if rel < lo or target_sec - rel < lo:
                continue
            cands.append((rel, k))
            punct = a["word"][-1:] in "،,.؟?!؛;:"
            weights.append(self.interleave_punct_weight if punct else 1.0)
        chosen: list[tuple[float, int]] = []
        while cands and len(chosen) < n_extra:
            i = self.rng.choices(range(len(cands)), weights)[0]
            rel, k = cands[i]
            chosen.append((rel, k))
            keep = [j for j, (r, _) in enumerate(cands) if abs(r - rel) >= lo]
            cands, weights = [cands[j] for j in keep], [weights[j] for j in keep]
        if not chosen:
            return None
        chosen.sort()
        bounds = [first] + [k for _, k in chosen] + [len(words)]
        frames = [0] + [
            min(max(1, round(rel * self.frame_rate)), target_frames - 1) for rel, _ in chosen
        ]
        return [
            (f, " ".join(w["word"] for w in words[bounds[j] : bounds[j + 1]]))
            for j, f in enumerate(frames)
        ]

    @staticmethod
    def _pad_latents(seqs: tuple[torch.Tensor, ...], min_len: int) -> torch.Tensor:
        length = max(min_len, max(s.shape[0] for s in seqs))
        out = torch.zeros(len(seqs), length, seqs[0].shape[-1])
        for b, s in enumerate(seqs):
            out[b, : s.shape[0]] = s
        return out

    def _collate_stitch_audio(self, stitches: tuple[npt.NDArray[np.float32], ...]) -> torch.Tensor:
        stitch_samples = self.stitch_frames * self.frame_size
        audio = torch.zeros(len(stitches), 1, stitch_samples)
        for b, w in enumerate(stitches):
            n = min(len(w), stitch_samples)
            audio[b, 0, :n] = torch.from_numpy(w[:n])
        return audio

    def _collate_latent(self, batch: list[tuple[Any, ...]]) -> Batch:
        stitches, tokens, prompts, tails, target_frames, extra = zip(*batch, strict=True)
        num_prompt_frames = torch.tensor([max(1, p.shape[0]) for p in prompts], dtype=torch.long)
        return Batch(
            self._collate_stitch_audio(stitches),
            torch.tensor(target_frames, dtype=torch.long),
            list(tokens),
            torch.zeros(len(stitches), 1, 0),
            num_prompt_frames,
            tail_latents=self._pad_latents(tails, 0),
            prompt_latents=self._pad_latents(prompts, 1),
            text_pieces=list(extra) if any(extra) else None,
        )

    def get_entry(self, index: int, entries: LazyEntries | None = None) -> Entry:
        d = json.loads((self.entries if entries is None else entries)[index])
        return Entry(
            d["path"],
            float(d["duration"]),
            d["transcript"],
            d.get("words"),
            float(d.get("start", 0.0)),
            d.get("latents_file"),
            d.get("speaker"),
        )

    def _sample_or_none(self, entry: Entry) -> tuple[Any, ...] | None:
        try:
            return self._sample(entry)
        except UnalignedEntry:
            raise
        except Exception as exc:  # noqa: BLE001 — skip unreadable samples, whatever the cause
            self._failures += 1
            if self._failures % 1000 == 1:
                logger.warning(f"skipping unreadable sample ({self._failures} so far): {exc}")
            return None

    def __iter__(self) -> Iterator[Batch]:
        # Batches are produced in a background thread (the loader is IO-bound)
        # so the GPU never waits on network storage.
        return _prefetch(self._batches())

    def _batches(self) -> Iterator[Batch]:
        self._failures = 0
        # Each sample is two small reads from network storage, so the loader is
        # latency-bound rather than CPU-bound: fetching a batch's samples
        # concurrently keeps the GPUs fed. sphn releases the GIL, so threads are
        # enough. Without this a cold, wide corpus starves training (~2x).
        pool = ThreadPoolExecutor(max_workers=self.io_workers)
        need = max(1, self.num_bucket_batches) * self.batch_size
        if len(self.entries) < need:
            raise ValueError(
                f"{len(self.entries)} entries in this loader's shard of {self.jsonl}, but the "
                f"first batch needs num_bucket_batches x batch_size = {need}. Lower "
                "data.num_bucket_batches (1 disables length bucketing), data.loader_procs or "
                "batch_size, use fewer ranks, or point at a larger manifest."
            )
        while True:
            yielded = 0
            order = list(range(len(self.entries)))
            if self.shuffle:
                self.rng.shuffle(order)
            samples = []
            for chunk_start in range(0, len(order), self.batch_size):
                chunk = order[chunk_start : chunk_start + self.batch_size]
                # Parse sequentially, outside the pool: get_entry is GIL-bound
                # (unlike sphn's audio reads), so parsing it on the worker
                # threads just contends with itself instead of overlapping.
                chunk_entries = [self.get_entry(i) for i in chunk]
                got = [s for s in pool.map(self._sample_or_none, chunk_entries) if s is not None]
                samples.extend(got)
                # Sort a pool of batches by row length so rows in a batch have similar lengths.
                pool_size = max(1, self.num_bucket_batches) * self.batch_size
                if len(samples) < pool_size:
                    continue
                samples.sort(key=self._row_len)
                n = len(samples) // self.batch_size
                batches = [
                    samples[i * self.batch_size : (i + 1) * self.batch_size] for i in range(n)
                ]
                samples = samples[n * self.batch_size :]
                if self.shuffle:
                    self.rng.shuffle(batches)
                for batch in batches:
                    yielded += 1
                    yield self._collate(batch)
            if not yielded:
                raise ValueError(
                    f"no readable samples in {self.jsonl}: every entry failed to load "
                    f"({self._failures} failures). Check the paths in the manifest."
                )

    def _row_len(self, sample: tuple[Any, ...]) -> float:
        """Voice prompt + text tokens + target audio frames: what a row costs after padding."""
        if self.stitch_frames:  # (stitch wav, tokens, prompt, tail, target frames, extra pieces)
            extra = sum(len(t) + 1 for _, t in sample[5])  # + 1: each piece's EOS probe
            return sample[2].shape[0] + len(sample[1]) + int(sample[4]) + extra
        wav, tokens, _prompt, prompt_samples = sample
        return (prompt_samples + len(wav)) * self.frame_rate / self.sample_rate + len(tokens)

    def _collate(self, batch: list[tuple[Any, ...]]) -> Batch:
        if self.stitch_frames:
            return self._collate_latent(batch)
        wavs, tokens, prompts, prompt_lens = zip(*batch, strict=True)
        max_len = max(len(w) for w in wavs)
        audio = torch.zeros(len(wavs), 1, max_len)
        frames = torch.zeros(len(wavs), dtype=torch.long)
        for b, w in enumerate(wavs):
            audio[b, 0, : len(w)] = torch.from_numpy(w)
            frames[b] = max(1, int(len(w) * self.frame_rate / self.sample_rate))
        max_prompt = max(len(p) for p in prompts)
        voice = torch.zeros(len(prompts), 1, max_prompt)
        for b, pr in enumerate(prompts):
            voice[b, 0, : len(pr)] = torch.from_numpy(pr)
        num_voice_prompt_frames = torch.tensor(
            [max(1, int(n * self.frame_rate / self.sample_rate)) for n in prompt_lens],
            dtype=torch.long,
        )
        return Batch(audio, frames, list(tokens), voice, num_voice_prompt_frames)
