"""Tests for StreamingWakeWordModel: parity with full-window WakeWordModel.predict()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword import StreamingWakeWordModel, WakeWordModel
from livekit.wakeword.inference.streaming import HOP_SAMPLES, WINDOW_SAMPLES

EXAMPLE_MODELS = Path(__file__).parent.parent / "examples" / "resources"


@pytest.fixture(scope="module")
def model() -> WakeWordModel:
    return WakeWordModel(
        models=[EXAMPLE_MODELS / "hey_livekit.onnx", EXAMPLE_MODELS / "nihao_livekit.onnx"]
    )


def _hard_stream(seconds: float = 7.0) -> np.ndarray:
    """Int16 audio that moves the mel model's top_db floor around.

    Low noise, a stretch of digital silence, near-silence and loud tone bursts
    entering and leaving the 2 s window make the floor change from hop to hop.
    """
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(7 * 16000) * 0.003
    audio[16000:24000] = 0.0
    audio[60000:64000] = rng.standard_normal(4000) * 1e-5
    for start in (20000, 45000, 90000):
        audio[start : start + 3000] += 0.8 * np.sin(np.arange(3000) * 0.3)
    return (np.clip(audio[: int(seconds * 16000)], -1, 1) * 32767).astype(np.int16)


def _feed(stream: StreamingWakeWordModel, audio: np.ndarray, sizes: list[int]) -> list:
    out: list[dict[str, float]] = []
    i = 0
    k = 0
    while i < len(audio):
        size = sizes[k % len(sizes)]
        out += stream.process(audio[i : i + size])
        i += size
        k += 1
    return out


def test_matches_predict_on_every_hop(model: WakeWordModel):
    """Each streamed hop scores the same as predict() on the last 2 s of audio."""
    audio = _hard_stream()
    stream = StreamingWakeWordModel(model)
    # Irregular chunk sizes, including ones smaller than a hop and a single sample
    scores = _feed(stream, audio, [1280, 500, 3000, 1, 2559])

    n_hops = len(audio) // HOP_SAMPLES - WINDOW_SAMPLES // HOP_SAMPLES + 1
    assert len(scores) == n_hops
    for h, streamed in enumerate(scores):
        end = WINDOW_SAMPLES + h * HOP_SAMPLES
        expected = model.predict(audio[end - WINDOW_SAMPLES : end])
        assert streamed.keys() == expected.keys()
        for name in expected:
            assert streamed[name] == pytest.approx(expected[name], abs=1e-5)


def test_one_embedding_per_hop_in_steady_noise(model: WakeWordModel):
    """With a steady noise floor nothing is recomputed: one embedding call per hop."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 5) * 1000).astype(np.int16)
    stream = StreamingWakeWordModel(model)

    calls = 0
    embed = model._speech_embedding

    class Counting:
        def __call__(self, windows: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += len(windows)
            return embed(windows)

    model._speech_embedding = Counting()  # type: ignore[assignment]
    try:
        stream.process(audio[:WINDOW_SAMPLES])  # warm-up
        calls = 0
        n_hops = len(stream.process(audio[WINDOW_SAMPLES:]))
    finally:
        model._speech_embedding = embed

    assert n_hops == (len(audio) - WINDOW_SAMPLES) // HOP_SAMPLES
    assert calls == n_hops


def test_warm_up_and_reset(model: WakeWordModel):
    stream = StreamingWakeWordModel(model)
    audio = _hard_stream(4.0)

    # Nothing is scored until a full 2 s window has arrived
    assert stream.process(audio[: WINDOW_SAMPLES - HOP_SAMPLES]) == []
    first = stream.process(audio[WINDOW_SAMPLES - HOP_SAMPLES : WINDOW_SAMPLES])
    assert len(first) == 1

    # reset() drops all state: the same audio scores the same way again
    stream.reset()
    assert stream.process(audio[: WINDOW_SAMPLES - 1]) == []
    again = stream.process(audio[WINDOW_SAMPLES - 1 : WINDOW_SAMPLES])
    assert again == first


def test_float_and_int16_input_agree(model: WakeWordModel):
    audio = _hard_stream(3.0)
    from_int = StreamingWakeWordModel(model).process(audio)
    from_float = StreamingWakeWordModel(model).process(audio.astype(np.float32) / 32768.0)
    assert from_int == from_float


def test_no_classifiers_returns_empty_scores():
    stream = StreamingWakeWordModel(WakeWordModel())
    out = stream.process(np.zeros(WINDOW_SAMPLES, dtype=np.int16))
    assert out == [{}]
