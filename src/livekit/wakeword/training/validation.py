"""Validation data shared by training and evaluation.

Validation has two kinds of negatives:

- test clips (``negative_features_test.npy``, ``background_noise_features_test.npy``),
  one ``(16, 96)`` window per clip and one decision per clip;
- the continuous ~11 h validation stream (``validation_set_features.npy``, ``(N, 96)``),
  which a live detector scores once per 80 ms hop. It is scored over stride-1 windows
  and its false accepts are counted with the listener's debounce.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import WakeWordConfig
from .metrics import HOP_SECONDS, WINDOW_STEPS, stride1_windows

logger = logging.getLogger(__name__)


@dataclass
class ValidationData:
    positive: np.ndarray  # (P, 16, 96)
    negative_clips: np.ndarray  # (N, 16, 96)
    clip_hours: float  # hours of audio covered by negative_clips
    stream: np.ndarray  # (T, 96) continuous negative stream, T may be 0

    @property
    def stream_hours(self) -> float:
        return float(max(0, self.stream.shape[0] - WINDOW_STEPS + 1) * HOP_SECONDS / 3600.0)


def load_validation_data(config: WakeWordConfig) -> ValidationData:
    """Load positive/negative test clips and the continuous validation stream."""
    model_dir = config.model_output_dir
    empty = np.zeros((0, WINDOW_STEPS, 96), dtype=np.float32)

    pos_path = model_dir / "positive_features_test.npy"
    pos = np.load(str(pos_path)) if pos_path.exists() else empty

    neg_parts = [
        np.load(str(p))
        for p in (
            model_dir / "negative_features_test.npy",
            model_dir / "background_noise_features_test.npy",
        )
        if p.exists()
    ]
    neg = np.concatenate(neg_parts, axis=0) if neg_parts else empty

    stream = np.zeros((0, 96), dtype=np.float32)
    val_path = config.data_path / "features" / "validation_set_features.npy"
    if val_path.exists():
        val = np.load(str(val_path), mmap_mode="r")
        if val.ndim == 2:
            stream = val
        else:
            # Already windowed: treat each window as an independent clip
            neg = np.concatenate([neg, np.asarray(val)], axis=0) if neg.shape[0] else val

    clip_hours = neg.shape[0] * config.augmentation.clip_duration / 3600.0
    return ValidationData(positive=pos, negative_clips=neg, clip_hours=clip_hours, stream=stream)


def score_stream(
    predict: Callable[[np.ndarray], np.ndarray],
    stream: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Score every stride-1 window of a ``(T, 96)`` stream, one score per hop."""
    windows = stride1_windows(stream)
    scores = [
        predict(np.ascontiguousarray(windows[i : i + batch_size], dtype=np.float32))
        for i in range(0, windows.shape[0], batch_size)
    ]
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def debounce_hops(config: WakeWordConfig) -> int:
    """The streaming debounce expressed in 80 ms hops."""
    return max(1, round(config.streaming_eval.debounce_seconds / HOP_SECONDS))
