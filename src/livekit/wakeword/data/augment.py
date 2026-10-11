"""Audio augmentation pipeline."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import WakeWordConfig

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """Augmentation pipeline for wake word clips.

    Applies per-sample augmentations, RIR convolution,
    and background noise mixing.
    """

    def __init__(
        self,
        background_paths: list[Path],
        rir_paths: list[Path],
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate
        self.background_files = self._collect_wavs(background_paths)
        self.rir_files = self._collect_wavs(rir_paths)
        self._per_sample_aug = None

    @staticmethod
    def _collect_wavs(dirs: list[Path]) -> list[Path]:
        wavs: list[Path] = []
        for d in dirs:
            if d.exists():
                wavs.extend(d.glob("**/*.wav"))
        return wavs

    def _get_per_sample_augmentations(self) -> Any:
        """Lazy-load audiomentations transforms."""
        if self._per_sample_aug is None:
            from audiomentations import Compose, SevenBandParametricEQ, TanhDistortion

            self._per_sample_aug = Compose(
                [
                    SevenBandParametricEQ(p=0.25),
                    TanhDistortion(p=0.25),
                ]
            )
        return self._per_sample_aug

    def apply_rir(self, audio: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Convolve audio with a random room impulse response."""
        if random.random() > p or not self.rir_files:
            return audio
        import soundfile as sf
        from scipy.signal import fftconvolve

        rir_path = random.choice(self.rir_files)
        rir, sr = sf.read(str(rir_path))
        if rir.ndim > 1:
            rir = rir[:, 0]
        # Normalize RIR
        rir = rir / (np.max(np.abs(rir)) + 1e-8)
        convolved = fftconvolve(audio, rir, mode="full")[: len(audio)]
        return convolved.astype(np.float32)

    def augment_clip(self, audio: np.ndarray) -> np.ndarray:
        """Apply per-sample augmentations to a single clip."""
        aug = self._get_per_sample_augmentations()
        return aug(samples=audio, sample_rate=self.sample_rate)

    def mix_with_background(
        self,
        audio: np.ndarray,
        snr_db_range: tuple[float, float] = (5.0, 15.0),
        signal_power: float | None = None,
    ) -> np.ndarray:
        """Mix audio with random background noise at given SNR.

        The noise spans the whole of *audio*. The SNR is relative to *signal_power*
        when given (e.g. the power of just the speech in a padded clip), otherwise to
        the power of the whole array.
        """
        if not self.background_files:
            return audio
        import soundfile as sf

        bg_path = random.choice(self.background_files)
        bg, sr = sf.read(str(bg_path))
        if bg.ndim > 1:
            bg = bg[:, 0]

        # Loop or crop background to match audio length
        if len(bg) < len(audio):
            repeats = (len(audio) // len(bg)) + 1
            bg = np.tile(bg, repeats)
        start = random.randint(0, max(0, len(bg) - len(audio)))
        bg = bg[start : start + len(audio)]

        # Compute SNR mixing
        snr_db = random.uniform(*snr_db_range)
        audio_power = (np.mean(audio**2) if signal_power is None else signal_power) + 1e-8
        bg_power = np.mean(bg**2) + 1e-8
        scale = np.sqrt(audio_power / (bg_power * 10 ** (snr_db / 10)))
        mixed = audio + scale * bg
        return mixed.astype(np.float32)


def align_clip_to_end(
    audio: np.ndarray,
    target_length: int,
    jitter_samples: int = 3200,  # 200ms at 16kHz
) -> np.ndarray:
    """Align a clip to the END of the target window with random jitter.

    Positive and negative clips are both placed at the end of the window with
    0..*jitter_samples* jitter (``augmentation.end_jitter`` in the pipeline); clips
    longer than the window keep their end.
    """
    return _align_clip_to_end(audio, target_length, jitter_samples)[0]


def _align_clip_to_end(
    audio: np.ndarray,
    target_length: int,
    jitter_samples: int = 3200,
) -> tuple[np.ndarray, int, int]:
    """:func:`align_clip_to_end`, also returning the ``[start, end)`` span the clip fills."""
    result = np.zeros(target_length, dtype=np.float32)
    jitter = random.randint(0, jitter_samples)
    end_pos = target_length - jitter
    start_pos = max(0, end_pos - len(audio))
    clip_start = max(0, len(audio) - (end_pos - start_pos))
    result[start_pos:end_pos] = audio[clip_start : clip_start + (end_pos - start_pos)]
    return result, start_pos, end_pos


def _pad_or_crop_center(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Center-pad (with zeros) or center-crop a clip to ``target_length``."""
    if len(audio) < target_length:
        padded = np.zeros(target_length, dtype=np.float32)
        start = (target_length - len(audio)) // 2
        padded[start : start + len(audio)] = audio
        return padded
    start = (len(audio) - target_length) // 2
    return audio[start : start + target_length]


_ALL_SPLITS = [
    "positive_train",
    "positive_test",
    "negative_train",
    "negative_test",
    "background_train",
    "background_test",
]

_ORIGINAL_RE = re.compile(r"^clip_\d{6}\.wav$")


def _original_clips(clip_dir: Path) -> list[Path]:
    """Original TTS clips (``clip_000000.wav``) in *clip_dir*, sorted; no augmented ones."""
    if not clip_dir.is_dir():
        return []
    return sorted(p for p in clip_dir.glob("*.wav") if _ORIGINAL_RE.match(p.name))


def _read_mono(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, _ = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32)


@dataclass
class SpeechPlacement:
    """How a speech clip is laid out in the window before RIR and noise.

    The clip always ends ``0..end_jitter`` samples before the end of the window. The
    padding before it can be filled with other speech (*context*) that ends a short
    gap before the phrase, as in a live stream where the wake word follows talk.
    Positives can also be rebuilt from two halves of the phrase with a pause between.
    """

    end_jitter: int = 4800
    context_clips: list[Path] = field(default_factory=list)
    context_probability: float = 0.0
    near_miss_clips: list[Path] = field(default_factory=list)
    near_miss_probability: float = 0.0
    context_gap: tuple[int, int] = (0, 6400)
    split_pairs: list[tuple[Path, Path]] = field(default_factory=list)
    split_probability: float = 0.0
    split_gap: tuple[int, int] = (1600, 8000)

    def phrase_audio(self, original: Path, positive: bool) -> np.ndarray:
        """The clip to place: *original*, or for a positive maybe a split-phrase pair."""
        if positive and self.split_pairs and random.random() < self.split_probability:
            left_path, right_path = random.choice(self.split_pairs)
            gap = np.zeros(random.randint(*self.split_gap), dtype=np.float32)
            return np.concatenate([_read_mono(left_path), gap, _read_mono(right_path)])
        return _read_mono(original)

    def context_audio(self, positive: bool) -> np.ndarray | None:
        """Speech to put before the phrase, or ``None`` for none."""
        if positive and self.near_miss_clips and random.random() < self.near_miss_probability:
            return _read_mono(random.choice(self.near_miss_clips))
        if self.context_clips and random.random() < self.context_probability:
            return _read_mono(random.choice(self.context_clips))
        return None

    def place(
        self,
        audio: np.ndarray,
        target_length: int,
        context: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, int]:
        """End-align *audio* and fill the padding with *context*.

        Returns the window and the ``[start, end)`` span of *audio* in it.
        """
        window, start, end = _align_clip_to_end(audio, target_length, self.end_jitter)
        if context is not None and len(context) > 0:
            ctx_end = start - random.randint(*self.context_gap)
            if ctx_end > 0:
                ctx = context[-ctx_end:]
                # Context talk at a similar level, sometimes a bit quieter
                ref = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
                rms = float(np.sqrt(np.mean(ctx**2))) + 1e-8
                gain = (ref / rms if ref > 0 else 1.0) * 10 ** (random.uniform(-6.0, 0.0) / 20)
                window[ctx_end - len(ctx) : ctx_end] = ctx * gain
        return window, start, end


def _split_pairs(clip_dir: Path) -> list[tuple[Path, Path]]:
    """Consecutive (left, right) half-phrase clips: ``clip_2j`` and ``clip_2j+1``."""
    by_idx = {int(p.stem.split("_")[1]): p for p in _original_clips(clip_dir)}
    return [(by_idx[i], by_idx[i + 1]) for i in sorted(by_idx) if i % 2 == 0 and i + 1 in by_idx]


def build_placement(
    config: WakeWordConfig, split: str, sample_rate: int = 16000
) -> SpeechPlacement:
    """The :class:`SpeechPlacement` for *split* (``positive_train``, ``negative_test``, ...)."""
    aug = config.augmentation
    model_dir = config.model_output_dir
    suffix = split.rsplit("_", 1)[-1]  # train / test

    def samples(r: tuple[float, float]) -> tuple[int, int]:
        lo, hi = sorted(r)
        return int(lo * sample_rate), int(hi * sample_rate)

    placement = SpeechPlacement(
        end_jitter=int(aug.end_jitter * sample_rate),
        context_gap=samples(aug.context_gap),
        split_gap=samples(aug.split_phrase_gap),
    )
    if aug.context_speech_probability > 0:
        placement.context_clips = _original_clips(model_dir / "context_speech")
        placement.context_probability = aug.context_speech_probability
    if split.startswith("positive"):
        if aug.near_miss_context_probability > 0:
            placement.near_miss_clips = _original_clips(model_dir / f"negative_{suffix}")
            placement.near_miss_probability = aug.near_miss_context_probability
        if aug.split_phrase_probability > 0:
            placement.split_pairs = _split_pairs(model_dir / f"positive_{suffix}_parts")
            placement.split_probability = aug.split_phrase_probability
    return placement


def run_augment(config: WakeWordConfig) -> None:
    """Run augmentation pipeline on generated clips.

    Every round starts again from the original TTS clips and draws a new
    placement, context, RIR and noise, so ``rounds`` gives that many independent
    variants of each clip.
    """
    target_duration = config.augmentation.clip_duration

    model_dir = config.model_output_dir

    # Clean up old augmented files before starting fresh augmentation.
    # This prevents stale _rN.wav files from previous runs piling up.
    _aug_re = re.compile(r"^clip_\d{6}_r\d+\.wav$")
    for split in _ALL_SPLITS:
        clip_dir = model_dir / split
        if not clip_dir.exists():
            continue
        old_augs = [p for p in clip_dir.glob("*.wav") if _aug_re.match(p.name)]
        if old_augs:
            logger.info(f"Cleaning {len(old_augs)} old augmented files from {split}")
            for p in old_augs:
                p.unlink()

    augmentor = AudioAugmentor(
        background_paths=[Path(p) for p in config.augmentation.background_paths],
        rir_paths=[Path(p) for p in config.augmentation.rir_paths],
    )
    if not augmentor.background_files:
        logger.warning(
            f"No background noise found in {config.augmentation.background_paths}; "
            "clip padding will stay digital silence, which live audio never contains"
        )

    placements = {split: build_placement(config, split) for split in _ALL_SPLITS}
    aug = config.augmentation
    for split, placement in placements.items():
        if aug.context_speech_probability > 0 and not placement.context_clips:
            logger.warning(f"{split}: no context_speech clips; set n_context_samples and generate")
        if split.startswith("positive") and aug.split_phrase_probability > 0:
            if not placement.split_pairs:
                logger.warning(f"{split}: no split-phrase clips; set n_split_phrase_samples")

    for round_idx in range(config.augmentation.rounds):
        logger.info(f"Augmentation round {round_idx + 1}/{config.augmentation.rounds}")
        for split in _ALL_SPLITS:
            clip_dir = model_dir / split
            if not clip_dir.exists():
                logger.warning(f"Skipping {split}: directory not found")
                continue
            _augment_directory(
                clip_dir,
                augmentor,
                # Positives and negatives are aligned identically, so the phrase's
                # position can't give its class away.
                end_align=not split.startswith("background"),
                round_idx=round_idx,
                target_duration_s=target_duration,
                placement=placements[split],
                positive=split.startswith("positive"),
            )


def _augment_directory(
    clip_dir: Path,
    augmentor: AudioAugmentor,
    end_align: bool,
    target_duration_s: float = 2.0,
    sample_rate: int = 16000,
    round_idx: int = 0,
    placement: SpeechPlacement | None = None,
    positive: bool = False,
) -> None:
    """Augment all original WAV files in a directory into ``clip_000000_r{round_idx}.wav``.

    Every round reads the original TTS clips (``clip_000000.wav``) and draws a
    fresh placement, RIR and noise, so rounds are independent variants rather
    than stacked layers of reverb and noise.

    Speech clips (``end_align``: positives and negatives alike) are end-aligned
    with jitter, optionally after other speech (see :class:`SpeechPlacement`);
    background clips are center-padded/cropped. This happens *before* RIR and
    background mixing, so reverb tails stay inside the window and the noise
    covers the padding. Otherwise the classifier can learn where the digital
    silence sits instead of how the phrase sounds. A streaming listener slides
    every phrase through the end of its window, so that shortcut makes near-miss
    phrases fire.
    """
    import soundfile as sf
    from tqdm import tqdm

    target_length = int(target_duration_s * sample_rate)
    placement = placement if placement is not None else SpeechPlacement()

    wav_files = _original_clips(clip_dir)

    for wav_path in tqdm(wav_files, desc=f"Augmenting {clip_dir.name} r{round_idx}", unit="clip"):
        if end_align:
            audio = placement.phrase_audio(wav_path, positive)
            # Per-sample augmentations on the unpadded clip (and context separately)
            audio = augmentor.augment_clip(audio)
            context = placement.context_audio(positive)
            if context is not None:
                context = augmentor.augment_clip(context)
            # [start, end) is the span the SNR is measured over.
            audio, start, end = placement.place(audio, target_length, context)
        else:
            audio = augmentor.augment_clip(_read_mono(wav_path))
            audio = _pad_or_crop_center(audio, target_length)
            start, end = 0, target_length

        # Apply RIR
        audio = augmentor.apply_rir(audio)

        # Mix with background across the whole window. Measure the SNR against the
        # clip's own span: counting the padding would make the noise too quiet.
        audio = augmentor.mix_with_background(
            audio, signal_power=float(np.mean(audio[start:end] ** 2))
        )

        out_path = wav_path.with_name(f"{wav_path.stem}_r{round_idx}.wav")
        sf.write(str(out_path), audio, sample_rate)
