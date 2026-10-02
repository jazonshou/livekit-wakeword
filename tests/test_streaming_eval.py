"""Tests for stride-1 validation metrics and the streaming evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword.config import StreamingEvalConfig, StreamingEvalSet, WakeWordConfig
from livekit.wakeword.eval import streaming
from livekit.wakeword.eval.streaming import (
    FRAME_SAMPLES,
    PRE_ROLL_SECONDS,
    SAMPLE_RATE,
    WindowScorer,
    build_stream,
    det_curve,
    score_audio,
    select_threshold,
    summarize_stream,
)
from livekit.wakeword.training.metrics import (
    detection_indices,
    evaluate_model,
    find_best_threshold,
    stride1_windows,
)
from livekit.wakeword.training.validation import load_validation_data, score_stream


class TestStride1:
    def test_windows_match_slices(self) -> None:
        stream = np.arange(40 * 96, dtype=np.float32).reshape(40, 96)
        windows = stride1_windows(stream)
        assert windows.shape == (25, 16, 96)
        np.testing.assert_array_equal(windows[7], stream[7:23])

    def test_short_stream(self) -> None:
        assert stride1_windows(np.zeros((5, 96))).shape == (0, 16, 96)

    def test_score_stream_scores_every_hop(self) -> None:
        stream = np.random.randn(1000, 96).astype(np.float32)
        scores = score_stream(lambda x: x[:, -1, 0], stream, batch_size=100)
        np.testing.assert_allclose(scores, stream[15:, 0])


class TestDetectionIndices:
    def test_debounce_merges_nearby_hops(self) -> None:
        scores = np.zeros(100)
        scores[[10, 11, 12, 30, 40]] = 0.9
        np.testing.assert_array_equal(detection_indices(scores, 0.5, debounce_hops=25), [10, 40])

    def test_no_debounce(self) -> None:
        scores = np.zeros(10)
        scores[[2, 3]] = 0.9
        np.testing.assert_array_equal(detection_indices(scores, 0.5, debounce_hops=1), [2, 3])

    def test_min_consecutive(self) -> None:
        scores = np.zeros(50)
        scores[5] = 0.9  # isolated spike: ignored
        scores[20:23] = 0.9  # fires on the 3rd hop
        fires = detection_indices(scores, 0.5, debounce_hops=25, min_consecutive=3)
        np.testing.assert_array_equal(fires, [22])


class TestStreamMetrics:
    def test_stream_false_accepts_are_debounced(self) -> None:
        pos = np.array([0.9, 0.9])
        neg = np.array([0.1, 0.6])  # one clip false accept
        stream = np.zeros(45000)  # 1 hour of hops
        stream[1000:1010] = 0.9  # one event spanning 10 hops
        stream[5000] = 0.9
        m = evaluate_model(pos, neg, 0.5, validation_hours=1.0, stream_preds=stream)
        assert m["stream_fpph"] == pytest.approx(2.0)
        assert m["fpph"] == pytest.approx(3.0 / 2.0)

    def test_without_stream_unchanged(self) -> None:
        m = evaluate_model(np.array([0.9]), np.array([0.6]), 0.5, validation_hours=1.0)
        assert m["fpph"] == 1.0 and "stream_fpph" not in m

    def test_find_best_threshold_uses_stream(self) -> None:
        pos = np.full(10, 0.95)
        stream = np.zeros(45000)
        stream[::100] = 0.8  # 450 events/hour below 0.8
        best = find_best_threshold(
            pos, np.zeros(0), 0.0, target_fpph=0.5, stream_preds=stream, debounce_hops=1
        )
        assert best["threshold"] > 0.8
        assert best["recall"] == 1.0


class TestValidationData:
    def test_loads_2d_stream(self, sample_config: WakeWordConfig) -> None:
        out = sample_config.model_output_dir
        out.mkdir(parents=True)
        np.save(out / "positive_features_test.npy", np.zeros((4, 16, 96), np.float32))
        np.save(out / "negative_features_test.npy", np.zeros((6, 16, 96), np.float32))
        feats = sample_config.data_path / "features"
        feats.mkdir(parents=True)
        np.save(feats / "validation_set_features.npy", np.zeros((1015, 96), np.float32))

        data = load_validation_data(sample_config)
        assert data.positive.shape[0] == 4
        assert data.negative_clips.shape[0] == 6
        assert data.stream.shape == (1015, 96)
        assert data.stream_hours == pytest.approx(1000 * 0.08 / 3600)
        assert data.clip_hours == pytest.approx(6 * 2.0 / 3600)


def _clip(seconds: float, value: float = 0.5) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), value, dtype=np.float32)


class _LoudnessScorer:
    """Scores 1.0 whenever the current frame is loud, so detections track the clips."""

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray) -> dict[str, float]:
        return {"m": float(np.abs(frame).mean() > 0.3)}


class TestStreams:
    def test_layout(self) -> None:
        s = build_stream("p", [_clip(1.0), _clip(0.5)], positive=True, gap_seconds=3.0)
        pre = int(PRE_ROLL_SECONDS * SAMPLE_RATE)
        assert s.targets[0] == (pre, pre + SAMPLE_RATE)
        assert s.targets[1][0] == pre + SAMPLE_RATE + 3 * SAMPLE_RATE
        assert s.audio_seconds == pytest.approx(1.5)

    def test_speech_before_counts_as_false_accept(self) -> None:
        rng = np.random.default_rng(0)
        s = build_stream(
            "p",
            [_clip(1.0)] * 3,
            positive=True,
            gap_seconds=3.0,
            speech_clips=[_clip(1.0, 0.4)],
            rng=rng,
        )
        scores = score_audio(_LoudnessScorer(), s.audio, "m")
        r = summarize_stream(s, scores, 0.5, debounce_hops=1)
        assert r["miss_rate"] == 0.0
        assert r["false_accepts"] > 0

    def test_quiet_clips_are_missed(self) -> None:
        s = build_stream("p", [_clip(1.0, 0.1)] * 4, positive=True, gap_seconds=3.0)
        scores = score_audio(_LoudnessScorer(), s.audio, "m")
        r = summarize_stream(s, scores, 0.5, debounce_hops=25)
        assert r["misses"] == 4 and r["miss_rate"] == 1.0

    def test_negative_set(self) -> None:
        s = build_stream("n", [_clip(1.0), _clip(1.0, 0.1)], positive=False, gap_seconds=3.0)
        scores = score_audio(_LoudnessScorer(), s.audio, "m")
        r = summarize_stream(s, scores, 0.5, debounce_hops=25)
        assert r["false_accepts"] == 1
        assert r["fa_per_clip"] == 0.5
        assert r["fa_per_hour"] == pytest.approx(1 / (2.0 / 3600))

    def test_background_snr(self) -> None:
        noise = np.random.default_rng(1).standard_normal(SAMPLE_RATE).astype(np.float32)
        s = build_stream(
            "p", [_clip(1.0)], positive=True, gap_seconds=3.0, background=noise, snr_db=10.0
        )
        start, end = s.targets[0]
        gap_power = np.mean(s.audio[end + 100 : end + SAMPLE_RATE] ** 2)
        assert 10 * np.log10(0.25 / gap_power) == pytest.approx(10.0, abs=0.5)

    def test_det_and_threshold_selection(self) -> None:
        pos = build_stream("p", [_clip(1.0)] * 2, positive=True, gap_seconds=3.0)
        neg = build_stream("n", [_clip(1.0)], positive=False, gap_seconds=3.0)
        pos_scores = np.zeros(int(np.ceil(pos.audio.shape[0] / FRAME_SAMPLES)))
        for a, _ in pos.targets:
            pos_scores[a // FRAME_SAMPLES + 3] = 0.9
        neg_scores = np.zeros(int(np.ceil(neg.audio.shape[0] / FRAME_SAMPLES)))
        neg_scores[neg.targets[0][0] // FRAME_SAMPLES + 3] = 0.6
        points = det_curve([(pos, pos_scores), (neg, neg_scores)], debounce_hops=25)
        chosen, met = select_threshold(points, target_fa_per_hour=0.5)
        assert met
        assert chosen["miss_rate"] == 0.0 and chosen["fa_per_hour"] == 0.0
        assert 0.6 < chosen["threshold"] <= 0.9


class TestWindowScorer:
    def test_waits_for_full_window(self) -> None:
        class _Model:
            def predict(self, audio: np.ndarray) -> dict[str, float]:
                assert audio.shape[0] == 25 * FRAME_SAMPLES
                return {"m": 0.7}

        scorer = WindowScorer(_Model())
        outs = [scorer.process_frame(np.zeros(FRAME_SAMPLES)) for _ in range(25)]
        assert outs[23] == {} and outs[24] == {"m": 0.7}


def test_stream_scorer_matches_window_scorer() -> None:
    from livekit.wakeword.eval.streaming import StreamScorer
    from livekit.wakeword.inference.model import WakeWordModel

    model = WakeWordModel(
        models=[Path(__file__).parent.parent / "examples" / "resources" / "hey_livekit.onnx"]
    )
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(32 * FRAME_SAMPLES) * 0.05).astype(np.float32)
    window, stream = WindowScorer(model), StreamScorer(model)
    for i in range(32):
        frame = audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        expected, got = window.process_frame(frame), stream.process_frame(frame)
        assert expected.keys() == got.keys()
        for name in expected:
            assert got[name] == pytest.approx(expected[name], abs=1e-5)


def test_run_streaming_eval(
    sample_config: WakeWordConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sf = pytest.importorskip("soundfile")
    out = sample_config.model_output_dir
    for split, value in (("positive_test", 0.5), ("negative_test", 0.1)):
        (out / split).mkdir(parents=True)
        for i in range(3):
            sf.write(str(out / split / f"clip_{i:06d}.wav"), _clip(1.0, value), SAMPLE_RATE)
    loud_negatives = tmp_path / "tv"
    loud_negatives.mkdir()
    sf.write(str(loud_negatives / "a.wav"), _clip(1.0, 0.5), SAMPLE_RATE)

    model_path = out / "m.onnx"
    model_path.write_bytes(b"")
    monkeypatch.setattr(streaming, "make_scorer", lambda *_a, **_k: (_LoudnessScorer(), "test"))

    sample_config.streaming_eval = StreamingEvalConfig(
        sets=[StreamingEvalSet(name="tv", positive=False, paths=[str(loud_negatives)])],
        target_fa_per_hour=1e9,
    )
    report = streaming.run_streaming_eval(sample_config, model_path)

    sets = report["sets"]
    assert set(sets) == {"positive_silence", "positive_speech", "near_miss", "tv"}
    assert sets["positive_silence"]["miss_rate"] == 0.0
    assert sets["near_miss"]["false_accepts"] == 0
    assert sets["tv"]["false_accepts"] == 1
    assert (out / "test_wakeword_streaming_eval.json").exists()
    assert set(np.load(out / "test_wakeword_streaming_scores.npz").files) == set(sets)


def test_trainer_validation_counts_every_hop(sample_config: WakeWordConfig) -> None:
    from livekit.wakeword.training.trainer import WakeWordTrainer

    out = sample_config.model_output_dir
    out.mkdir(parents=True)
    np.save(out / "positive_features_test.npy", np.ones((4, 16, 96), np.float32))
    feats = sample_config.data_path / "features"
    feats.mkdir(parents=True)
    stream = np.zeros((45015, 96), np.float32)  # 45000 hops = 1 hour
    stream[[1000, 1008, 20000]] = 1.0  # 1008 is inside the debounce of 1000
    np.save(feats / "validation_set_features.npy", stream)

    trainer = WakeWordTrainer(sample_config)
    trainer._predict = lambda x, batch_size=512: x.max(axis=(1, 2))  # type: ignore[method-assign]
    metrics = trainer._validate()
    assert metrics["stream_fpph"] == pytest.approx(2.0)
    assert metrics["recall"] == 1.0
