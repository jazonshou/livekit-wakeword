"""Tests for augmentation utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.data.augment import AudioAugmentor, align_clip_to_end, run_augment

SR = 16000
TARGET = 32000  # 2s window
JITTER = 3200  # 200ms


def _write_noise(path: Path, seconds: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(0).normal(0, 0.1, int(seconds * SR)).astype(np.float32)
    sf.write(str(path), noise, SR)


def _longest_zero_run(audio: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], (audio == 0).astype(np.int8), [0]))))
    return int((edges[1::2] - edges[::2]).max(initial=0))


class TestAlignClipToEnd:
    def test_basic_alignment(self):
        audio = np.ones(8000, dtype=np.float32)  # 0.5s at 16kHz
        target_length = 32000  # 2s
        result = align_clip_to_end(audio, target_length, jitter_samples=0)
        assert result.shape == (target_length,)
        # Audio should be at the end
        assert np.sum(result[-8000:]) > 0
        assert np.sum(result[:16000]) == 0.0

    def test_output_length(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000

    def test_longer_clip_than_target(self):
        audio = np.ones(48000, dtype=np.float32)  # 3s, longer than target
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000


class TestMixWithBackground:
    def test_snr_relative_to_signal_power(self, tmp_path: Path):
        _write_noise(tmp_path / "bg" / "noise.wav")
        augmentor = AudioAugmentor(background_paths=[tmp_path / "bg"], rir_paths=[])
        speech = 0.2 * np.sin(np.linspace(0, 400 * np.pi, 8000)).astype(np.float32)
        audio = np.zeros(TARGET, dtype=np.float32)
        audio[-len(speech) :] = speech
        power = float(np.mean(speech**2))

        mixed = augmentor.mix_with_background(audio, (10.0, 10.0), signal_power=power)

        noise = mixed - audio
        assert _longest_zero_run(noise) < 16  # the noise covers the padding too
        assert 10 * np.log10(power / np.mean(noise**2)) == pytest.approx(10.0, abs=0.01)


class TestRunAugment:
    """Positives and negatives must not be separable by where the silence is."""

    CLIP_SECONDS = [0.4, 0.8, 1.2, 2.5]
    SPLITS = ["positive_train", "negative_train"]

    @pytest.fixture
    def config(self, sample_config: WakeWordConfig, monkeypatch) -> WakeWordConfig:
        # Skip EQ/distortion so the clip's samples stay recognisable
        monkeypatch.setattr(AudioAugmentor, "augment_clip", lambda self, audio: audio)
        random.seed(0)
        for split in self.SPLITS:
            for i, seconds in enumerate(self.CLIP_SECONDS):
                path = sample_config.model_output_dir / split / f"clip_{i:06d}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), np.full(int(seconds * SR), 0.2, dtype=np.float32), SR)
        sample_config.augmentation.rir_paths = []
        sample_config.augmentation.background_paths = []
        return sample_config

    def test_negatives_end_aligned_like_positives(self, config: WakeWordConfig):
        run_augment(config)

        for split in self.SPLITS:
            for i, seconds in enumerate(self.CLIP_SECONDS):
                out, _ = sf.read(str(config.model_output_dir / split / f"clip_{i:06d}_r0.wav"))
                assert len(out) == TARGET
                nonzero = np.flatnonzero(out)
                trailing = TARGET - 1 - nonzero[-1]
                assert 0 <= trailing <= JITTER, split
                # The clip is one block ending there, cropped from the front if too long
                assert len(nonzero) == nonzero[-1] - nonzero[0] + 1
                assert len(nonzero) == min(int(seconds * SR), TARGET - trailing)

    def test_no_digital_silence_with_background(self, config: WakeWordConfig, tmp_path: Path):
        _write_noise(tmp_path / "bg" / "noise.wav")
        config.augmentation.background_paths = [str(tmp_path / "bg")]
        config.augmentation.rounds = 2

        run_augment(config)

        for split in self.SPLITS:
            outputs = sorted((config.model_output_dir / split).glob("clip_*_r*.wav"))
            assert len(outputs) == 2 * len(self.CLIP_SECONDS)
            for path in outputs:
                out, _ = sf.read(str(path))
                assert len(out) == TARGET
                assert _longest_zero_run(out) < 16, path.name  # < 1ms
