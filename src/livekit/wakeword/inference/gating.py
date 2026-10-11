"""Optional checks that sit between the wake word scores and a detection.

- :class:`SpeechGate` blocks detections unless WebRTC VAD heard speech recently, so
  noise that happens to score high (music beats, clatter) can't fire.
- :func:`verifier_score` reads a second-stage model's score for a detection; the
  second model runs only when the first one fires, so it costs nothing while idle.
"""

from __future__ import annotations

from collections import deque

import numpy as np

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 320  # 20 ms, a frame size WebRTC VAD accepts


class SpeechGate:
    """Tracks whether WebRTC VAD heard speech within the last ``window_seconds``.

    Feed every audio frame to :meth:`update` (any length; 20 ms sub-frames are
    buffered across calls). The gate is open while at least ``min_speech_frames`` of
    the 20 ms frames in the window were speech.

    Args:
        aggressiveness: WebRTC VAD mode, 0 (least aggressive about filtering out
            non-speech) to 3 (most).
        window_seconds: How far back speech keeps the gate open. It should cover the
            wake phrase, so 1 s or more.
        min_speech_frames: Speech frames (20 ms each) needed in the window.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        window_seconds: float = 1.0,
        min_speech_frames: int = 3,
    ):
        import webrtcvad

        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"aggressiveness must be 0-3, got {aggressiveness}")
        self._vad = webrtcvad.Vad(aggressiveness)
        n_frames = max(1, round(window_seconds * SAMPLE_RATE / VAD_FRAME_SAMPLES))
        self._min_speech = max(1, min_speech_frames)
        self._flags: deque[bool] = deque(maxlen=n_frames)
        self._pending = np.zeros(0, dtype=np.int16)

    def reset(self) -> None:
        self._flags.clear()
        self._pending = np.zeros(0, dtype=np.int16)

    @property
    def is_open(self) -> bool:
        return sum(self._flags) >= self._min_speech

    def update(self, audio: np.ndarray) -> bool:
        """Add 16 kHz audio (int16, or float in [-1, 1]) and return :attr:`is_open`."""
        if audio.dtype != np.int16:
            audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        pending = np.concatenate([self._pending, audio.reshape(-1)])
        n_full = pending.shape[0] // VAD_FRAME_SAMPLES
        for i in range(n_full):
            frame = pending[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES]
            self._flags.append(self._vad.is_speech(frame.tobytes(), SAMPLE_RATE))
        self._pending = pending[n_full * VAD_FRAME_SAMPLES :]
        return self.is_open


def verifier_score(scores: dict[str, float], name: str) -> float:
    """The verifier's score for detection *name*.

    Uses the verifier classifier with the same name, or its only classifier when it
    has one; otherwise the detection is not verified (score 0).
    """
    if name in scores:
        return scores[name]
    if len(scores) == 1:
        return next(iter(scores.values()))
    return 0.0
