"""Voice selection shared by TTS backends."""

from __future__ import annotations

import numpy as np


def split_voices(n_voices: int, test_fraction: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """Split voice indices ``0..n_voices-1`` into disjoint (train, test) lists.

    The split is a seeded shuffle, so it is the same on every run. With at least two
    voices each side gets at least one; with one voice both sides share it.
    """
    if n_voices <= 0:
        return [], []
    if n_voices == 1 or test_fraction <= 0:
        return list(range(n_voices)), list(range(n_voices))
    order = np.random.default_rng(seed).permutation(n_voices).tolist()
    n_test = min(n_voices - 1, max(1, round(n_voices * test_fraction)))
    return sorted(order[n_test:]), sorted(order[:n_test])
