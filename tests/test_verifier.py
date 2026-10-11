"""Tests for the second-stage verifier: inference, listener hook, eval gating and training."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword import StreamingWakeWordModel, WakeWordModel, WakeWordVerifier
from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.inference.listener import WakeWordListener
from livekit.wakeword.inference.streaming import HOP_SAMPLES, WINDOW_SAMPLES

from .test_listener import FakeModel, FakeStream, _patch_pyaudio

EXAMPLE_MODELS = Path(__file__).parent.parent / "examples" / "resources"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeVerifier(WakeWordVerifier):
    """Verifier returning pre-configured scores, one per call."""

    def __init__(self, scores: list[float], threshold: float = 0.5):
        self._scores = list(scores)
        self.threshold = threshold
        self.name = "fake_verifier"
        self.calls = 0

    def score(self, embeddings: np.ndarray) -> float:
        assert embeddings.shape == (16, 96)
        score = self._scores[min(self.calls, len(self._scores) - 1)]
        self.calls += 1
        return score


class EmbeddingFakeModel(FakeModel):
    def embeddings(self) -> np.ndarray | None:
        return np.zeros((16, 96), dtype=np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> WakeWordModel:
    return WakeWordModel(models=[EXAMPLE_MODELS / "hey_livekit.onnx"])


def test_stream_embeddings_match_window_embeddings(model: WakeWordModel):
    """The streaming model's cached embeddings equal embed() on the last 2 s."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(WINDOW_SAMPLES + 5 * HOP_SAMPLES) * 0.05).astype(np.float32)
    stream = StreamingWakeWordModel(model)
    assert stream.embeddings() is None
    stream.process(audio[: WINDOW_SAMPLES - HOP_SAMPLES])
    assert stream.embeddings() is None
    stream.process(audio[WINDOW_SAMPLES - HOP_SAMPLES :])

    expected = model.embed(audio[-WINDOW_SAMPLES:])
    got = stream.embeddings()
    assert expected is not None and got is not None
    assert got.shape == (16, 96)
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_embed_short_audio_returns_none(model: WakeWordModel):
    assert model.embed(np.zeros(8000, dtype=np.float32)) is None


def test_verifier_scores_like_the_classifier(model: WakeWordModel):
    """A classifier loaded as a verifier scores embeddings like the main model."""
    verifier = WakeWordVerifier(EXAMPLE_MODELS / "hey_livekit.onnx")
    assert verifier.threshold == 0.5  # no threshold metadata
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(WINDOW_SAMPLES) * 0.05).astype(np.float32)
    emb = model.embed(audio)
    assert emb is not None
    expected = model.predict(audio)["hey_livekit"]
    assert verifier.score(emb) == pytest.approx(expected, abs=1e-6)
    assert verifier.score_audio(model, audio) == pytest.approx(expected, abs=1e-6)
    batch = verifier.score_batch(np.stack([emb, emb]))
    np.testing.assert_allclose(batch, [expected, expected], atol=1e-6)
    accepted, score = verifier.verify(emb)
    assert accepted == (score >= 0.5)


def test_verifier_threshold_override():
    verifier = WakeWordVerifier(EXAMPLE_MODELS / "hey_livekit.onnx", threshold=0.9)
    assert verifier.threshold == 0.9


def test_verifier_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        WakeWordVerifier(tmp_path / "nope.onnx")


# ---------------------------------------------------------------------------
# Listener hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listener_without_verifier_is_unchanged():
    model = FakeModel(scores_sequence=[{"test": 0.1}, {"test": 0.9}])
    with _patch_pyaudio(FakeStream()):
        async with WakeWordListener(model, threshold=0.5, debounce=0.0) as listener:
            detection = await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)
    assert detection.verifier_score is None
    assert model._call_count == 2


@pytest.mark.asyncio
async def test_verifier_rejects_then_accepts():
    """Rejected triggers don't fire or pause; the next qualifying hop is re-checked."""
    high = {"test": 0.9}
    model = EmbeddingFakeModel(scores_sequence=[{"test": 0.1}, high, high, high])
    verifier = FakeVerifier([0.2, 0.3, 0.8])
    with _patch_pyaudio(FakeStream()):
        async with WakeWordListener(
            model, threshold=0.5, debounce=0.0, verifier=verifier
        ) as listener:
            detection = await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)
            rejected = listener.rejected_triggers

    assert verifier.calls == 3
    assert rejected == 2
    assert detection.verifier_score == pytest.approx(0.8)
    assert detection.confidence == pytest.approx(0.9)
    assert model._call_count == 4


@pytest.mark.asyncio
async def test_verifier_only_runs_on_triggers():
    model = EmbeddingFakeModel(scores_sequence=[{"test": 0.1}] * 10 + [{"test": 0.9}])
    verifier = FakeVerifier([0.9])
    with _patch_pyaudio(FakeStream()):
        async with WakeWordListener(model, threshold=0.5, verifier=verifier) as listener:
            await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_verifier_mapping_applies_per_model():
    """A mapping verifies only the models it names; others fire unchecked."""
    model = EmbeddingFakeModel(scores_sequence=[{"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}])
    verifier = FakeVerifier([0.0])
    with _patch_pyaudio(FakeStream()):
        async with WakeWordListener(
            model, threshold=0.5, debounce=0.0, verifier={"a": verifier}
        ) as listener:
            detection = await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)
    assert detection.name == "b"
    assert detection.verifier_score is None
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_rejection_does_not_start_debounce():
    model = EmbeddingFakeModel(scores_sequence=[{"test": 0.9}, {"test": 0.9}])
    verifier = FakeVerifier([0.1, 0.9])
    with _patch_pyaudio(FakeStream()):
        async with WakeWordListener(
            model, threshold=0.5, debounce=10.0, verifier=verifier
        ) as listener:
            detection = await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)
    assert detection.verifier_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Streaming eval gating
# ---------------------------------------------------------------------------


def test_verified_scorer_zeroes_rejected_hops():
    from livekit.wakeword.eval.streaming import VerifiedScorer

    class Scorer:
        def __init__(self) -> None:
            self.n = 0

        def reset(self) -> None:
            self.n = 0

        def process_frame(self, frame: np.ndarray) -> dict[str, float]:
            self.n += 1
            return {} if self.n == 1 else {"m": 0.9}

        def embeddings(self) -> np.ndarray | None:
            return None if self.n == 1 else np.zeros((16, 96), dtype=np.float32)

    gated = VerifiedScorer(Scorer(), FakeVerifier([0.1, 0.9]))
    frame = np.zeros(1280, dtype=np.float32)
    assert gated.process_frame(frame) == {}
    assert gated.process_frame(frame) == {"m": 0.0}
    assert gated.process_frame(frame) == {"m": 0.9}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_verifier_threshold_keeps_target_recall():
    from livekit.wakeword.training.verifier import verifier_threshold

    pos = np.linspace(0.0, 1.0, 101)[1:]  # 0.01 .. 1.00
    neg = np.array([0.1, 0.5, 0.95])
    m = verifier_threshold(pos, neg, target_recall=0.9)
    assert m["threshold"] == pytest.approx(0.11)
    assert m["recall"] >= 0.9
    assert m["near_miss_accept"] == pytest.approx(2 / 3)
    assert verifier_threshold(pos, neg, 1.0)["threshold"] == pytest.approx(0.01)


def test_config_verifier_defaults(sample_config: WakeWordConfig):
    assert sample_config.verifier_name == "test_wakeword_verifier"
    assert sample_config.verifier.model.model_size == "medium"
    cfg = WakeWordConfig(
        model_name="x",
        target_phrases=["hey x"],
        verifier={"model": {"model_type": "dnn", "model_size": "tiny"}, "steps": 5},
    )
    assert cfg.verifier.model.model_type == "dnn"
    assert cfg.verifier.steps == 5


def test_train_verifier_end_to_end(sample_config: WakeWordConfig):
    """Trains on separable fake features, exports ONNX with its threshold, loads it back."""
    pytest.importorskip("onnx")
    from livekit.wakeword.config import ModelConfig
    from livekit.wakeword.training.verifier import run_train_verifier

    rng = np.random.default_rng(0)
    out = sample_config.model_output_dir
    out.mkdir(parents=True)

    def feats(n: int, shift: float, steps: int = 16) -> np.ndarray:
        return (rng.standard_normal((n, steps, 96)) * 0.1 + shift).astype(np.float32)

    np.save(out / "positive_features_train.npy", feats(64, 1.0, steps=20))
    np.save(out / "negative_features_train.npy", feats(64, -1.0, steps=20))
    np.save(out / "positive_features_test.npy", feats(20, 1.0))
    np.save(out / "negative_features_test.npy", feats(20, -1.0))

    sample_config.verifier.model = ModelConfig(model_type="dnn", model_size="tiny")
    sample_config.verifier.steps = 60
    sample_config.verifier.learning_rate = 1e-2
    sample_config.verifier.batch_n_per_class = {"positive": 16, "adversarial_negative": 16}

    onnx_path = run_train_verifier(sample_config)
    assert onnx_path == out / "test_wakeword_verifier.onnx"
    assert (out / "test_wakeword_verifier.pt").exists()
    assert (out / "test_wakeword_verifier_metrics.json").exists()

    verifier = WakeWordVerifier(onnx_path)
    import onnx

    meta = {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}
    assert float(meta["verifier_threshold"]) == pytest.approx(verifier.threshold, abs=1e-6)

    pos = np.load(out / "positive_features_test.npy")
    neg = np.load(out / "negative_features_test.npy")
    assert np.mean(verifier.score_batch(pos) >= verifier.threshold) >= 0.98
    assert np.mean(verifier.score_batch(neg) >= verifier.threshold) <= 0.1


def test_train_verifier_needs_features(sample_config: WakeWordConfig):
    from livekit.wakeword.training.verifier import VerifierTrainer

    with pytest.raises(FileNotFoundError, match="train-verifier"):
        VerifierTrainer(sample_config)._build_dataloader()
