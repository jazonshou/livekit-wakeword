"""Tests for feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.wakeword.data.features import extract_features_from_directory
from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path


def test_workers_match_single_process(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for i in range(6):
        audio = (rng.standard_normal(32000) * 0.05).astype(np.float32)
        sf.write(tmp_path / f"clip_{i:06d}_r0.wav", audio, 16000)
    sf.write(tmp_path / "clip_000000.wav", np.zeros(32000, dtype=np.float32), 16000)  # skipped

    mel = MelSpectrogramFrontend(get_mel_model_path())
    emb = SpeechEmbedding(get_embedding_model_path())
    single = extract_features_from_directory(tmp_path, mel, emb)
    parallel = extract_features_from_directory(tmp_path, mel, emb, workers=2)
    assert single.shape == (6, 16, 96)
    np.testing.assert_allclose(parallel, single, atol=1e-5)
