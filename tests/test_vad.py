"""Tests for VoiceActivityGate and its use in the listener and streaming eval."""

from __future__ import annotations

import numpy as np
import pytest

from livekit.wakeword import VoiceActivityGate
from livekit.wakeword.eval.streaming import score_audio
from livekit.wakeword.inference.listener import WakeWordListener

SR = 16000


def _noise(seconds: float, dbfs: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR)).astype(np.float32)
    return x * np.float32(10 ** (dbfs / 20))


def _voiced(seconds: float, dbfs: float = -20.0) -> np.ndarray:
    """Speech-like signal: a 140 Hz harmonic stack, amplitude-modulated at 4 Hz."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 20))
    x = x * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
    x = x / np.sqrt(np.mean(x**2)) * 10 ** (dbfs / 20)
    return x.astype(np.float32)


def _feed(gate: VoiceActivityGate, audio: np.ndarray, frame: int = 1280) -> list[bool]:
    return [gate.process(audio[i : i + frame]) for i in range(0, len(audio), frame)]


@pytest.mark.parametrize("backend", ["energy", "webrtc"])
def test_closed_on_silence_and_steady_noise(backend: str) -> None:
    gate = VoiceActivityGate(backend)  # type: ignore[arg-type]
    assert not any(_feed(gate, np.zeros(3 * SR, dtype=np.float32)))
    gate.reset()
    assert not any(_feed(gate, _noise(3.0, -60.0)))


@pytest.mark.parametrize("backend", ["energy", "webrtc"])
def test_opens_on_speech_then_closes_after_window(backend: str) -> None:
    gate = VoiceActivityGate(backend, window_seconds=1.0)  # type: ignore[arg-type]
    audio = np.concatenate([_noise(2.0, -60.0), _voiced(0.6) + _noise(0.6, -60.0, 1)])
    states = _feed(gate, audio)
    assert not any(states[:20])  # 1.6 s of background only
    assert states[-1]
    # Still open shortly after speech ends, closed once the window has passed
    after = _feed(gate, _noise(1.5, -60.0, 2))
    assert after[0]
    assert not after[-1]


def test_energy_adapts_to_loud_background() -> None:
    """A step up in steady noise opens briefly, then the floor catches up."""
    gate = VoiceActivityGate(noise_rise_db_per_second=20.0)
    _feed(gate, _noise(1.0, -70.0))
    states = _feed(gate, _noise(6.0, -45.0, 1))
    assert not states[-1]


def test_int16_and_frame_size_invariance() -> None:
    audio = np.concatenate([_noise(1.0, -60.0), _voiced(0.5), _noise(1.0, -60.0, 1)])
    a = VoiceActivityGate()
    b = VoiceActivityGate()
    sa = _feed(a, audio, 1280)
    pcm = (audio * 32767).astype(np.int16)
    _feed(b, pcm, 333)
    assert a.speech_seconds == pytest.approx(b.speech_seconds, abs=0.02)
    assert sa[-1] == b.is_open


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        VoiceActivityGate("silero")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VoiceActivityGate(window_seconds=0)
    with pytest.raises(ValueError):
        VoiceActivityGate(window_seconds=1.0, min_speech_seconds=2.0)
    with pytest.raises(ValueError):
        VoiceActivityGate("webrtc", aggressiveness=4)


class _ConstantModel:
    def reset(self) -> None:
        pass

    def process(self, audio: np.ndarray) -> list[dict[str, float]]:
        return [{"test": 0.9}]


class _StubGate:
    def __init__(self, open_: bool) -> None:
        self.open = open_
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def process(self, audio: np.ndarray) -> bool:
        return self.open


def test_listener_gate_blocks_and_restarts_runs() -> None:
    gate = _StubGate(False)
    listener = WakeWordListener(
        _ConstantModel(),
        threshold=0.5,
        debounce=0.0,
        min_consecutive=2,
        vad=gate,  # type: ignore[arg-type]
    )
    frame = np.zeros(1280, dtype=np.int16)
    for _ in range(5):
        scores, speech = listener._process_frame(frame)
        assert listener._check_scores(scores[0], speech=speech) is None
    gate.open = True
    scores, speech = listener._process_frame(frame)
    assert listener._check_scores(scores[0], speech=speech) is None  # run restarts at 1
    scores, speech = listener._process_frame(frame)
    assert listener._check_scores(scores[0], speech=speech) is not None


def test_listener_without_gate_unchanged() -> None:
    listener = WakeWordListener(_ConstantModel(), threshold=0.5, debounce=0.0)  # type: ignore[arg-type]
    scores, speech = listener._process_frame(np.zeros(1280, dtype=np.int16))
    assert speech
    assert listener._check_scores(scores[0], speech=speech) is not None


def test_listener_resets_gate() -> None:
    gate = _StubGate(True)
    listener = WakeWordListener(_ConstantModel(), vad=gate)  # type: ignore[arg-type]
    listener._reset_state()
    assert gate.resets == 1


class _OneScorer:
    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray) -> dict[str, float]:
        return {"m": 0.9}


def test_score_audio_zeroes_hops_without_speech() -> None:
    audio = np.concatenate([np.zeros(2 * SR, dtype=np.float32), _voiced(0.6)])
    plain = score_audio(_OneScorer(), audio, "m")
    gated = score_audio(_OneScorer(), audio, "m", VoiceActivityGate())
    assert np.all(plain == pytest.approx(0.9))
    assert np.all(gated[:20] == 0.0)
    assert gated[-1] == pytest.approx(0.9)


def test_gate_is_cheap() -> None:
    import time

    gate = VoiceActivityGate()
    audio = _noise(10.0, -40.0)
    t0 = time.perf_counter()
    _feed(gate, audio)
    per_hop = (time.perf_counter() - t0) / 125
    assert per_hop < 2e-3
