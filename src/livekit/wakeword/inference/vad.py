"""Voice activity gate: keeps the wake word from firing when no speech is present."""

from __future__ import annotations

from collections import deque
from typing import Any, Literal

import numpy as np

SAMPLE_RATE = 16000
SUBFRAME_SAMPLES = 160  # 10 ms, a frame size both backends accept
SUBFRAME_SECONDS = SUBFRAME_SAMPLES / SAMPLE_RATE

VadBackend = Literal["energy", "webrtc"]


class VoiceActivityGate:
    """Tracks recent speech in a 16 kHz stream and opens only while speech is present.

    Audio is split into 10 ms subframes, and each subframe is classified as
    speech or not.  The gate is open when at least ``min_speech_seconds`` of
    speech fell within the last ``window_seconds``.  The default window matches
    the 2 s the classifier sees, so a wake word that just ended still holds
    the gate open for the hops where its score peaks.

    Two backends:

    - ``"energy"`` (default, numpy only): a subframe is speech when its level is
      at least ``threshold_db`` above an adaptive noise floor and above
      ``min_dbfs``.  The floor follows the quietest recent level: it drops
      immediately and rises by at most ``noise_rise_db_per_second``.  Costs a
      few microseconds per 80 ms hop.
    - ``"webrtc"``: the WebRTC VAD (``pip install webrtcvad-wheels``) at the
      given ``aggressiveness`` (0-3).  Better at rejecting loud non-speech
      noise; still well under a millisecond per hop.

    Example:
        gate = VoiceActivityGate()
        for frame in microphone_frames():
            open_ = gate.process(frame)
            for scores in stream.process(frame):
                if open_ and scores["hey_livekit"] >= 0.5:
                    ...
    """

    def __init__(
        self,
        backend: VadBackend = "energy",
        *,
        window_seconds: float = 2.0,
        min_speech_seconds: float = 0.15,
        threshold_db: float = 10.0,
        min_dbfs: float = -55.0,
        noise_rise_db_per_second: float = 3.0,
        aggressiveness: int = 2,
    ):
        """Initialize the gate.

        Args:
            backend: ``"energy"`` or ``"webrtc"``.
            window_seconds: How far back speech counts toward opening the gate.
            min_speech_seconds: Speech needed within the window to open it.
            threshold_db: (energy) Margin above the noise floor that counts as speech.
            min_dbfs: (energy) Subframes quieter than this are never speech.
            noise_rise_db_per_second: (energy) How fast the noise floor may rise.
            aggressiveness: (webrtc) WebRTC VAD mode, 0 (least) to 3 (most aggressive).
        """
        if backend not in ("energy", "webrtc"):
            raise ValueError(f"Unknown VAD backend {backend!r}; expected 'energy' or 'webrtc'")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if not 0 <= min_speech_seconds <= window_seconds:
            raise ValueError(
                f"min_speech_seconds must be in [0, window_seconds], got {min_speech_seconds}"
            )
        self.backend = backend
        self._window = max(1, round(window_seconds / SUBFRAME_SECONDS))
        self._min_speech = max(1, round(min_speech_seconds / SUBFRAME_SECONDS))
        self._threshold_db = threshold_db
        self._min_dbfs = min_dbfs
        self._noise_rise = noise_rise_db_per_second * SUBFRAME_SECONDS
        self._webrtc: Any = None
        if backend == "webrtc":
            if not 0 <= aggressiveness <= 3:
                raise ValueError(f"aggressiveness must be 0-3, got {aggressiveness}")
            try:
                import webrtcvad  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "The 'webrtc' VAD backend needs webrtcvad: pip install livekit-wakeword[vad]"
                ) from exc
            self._webrtc = webrtcvad.Vad(aggressiveness)
        self.reset()

    def reset(self) -> None:
        """Forget all audio: the gate closes until speech is heard again."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._flags: deque[bool] = deque(maxlen=self._window)
        self._speech_count = 0
        self._noise_floor: float | None = None

    @property
    def is_open(self) -> bool:
        """True when enough speech fell within the window."""
        return self._speech_count >= self._min_speech

    @property
    def speech_seconds(self) -> float:
        """Seconds of speech within the window."""
        return self._speech_count * SUBFRAME_SECONDS

    def process(self, audio: np.ndarray) -> bool:
        """Add audio and return whether the gate is open afterwards.

        Args:
            audio: 16 kHz mono samples, int16 or float32 in [-1, 1], any length.
                Samples short of a full 10 ms subframe carry over to the next call.
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        buf = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32).ravel()])
        n = len(buf) // SUBFRAME_SAMPLES
        self._pending = buf[n * SUBFRAME_SAMPLES :]
        if n:
            frames = buf[: n * SUBFRAME_SAMPLES].reshape(n, SUBFRAME_SAMPLES)
            for is_speech in self._classify(frames):
                if len(self._flags) == self._window and self._flags[0]:
                    self._speech_count -= 1
                self._flags.append(is_speech)
                self._speech_count += is_speech
        return self.is_open

    def _classify(self, frames: np.ndarray) -> list[bool]:
        if self._webrtc is not None:
            pcm = (np.clip(frames, -1.0, 1.0) * 32767.0).astype(np.int16)
            return [bool(self._webrtc.is_speech(f.tobytes(), SAMPLE_RATE)) for f in pcm]
        levels = 10.0 * np.log10(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
        out = []
        for level in levels.tolist():
            floor = self._noise_floor
            floor = level if floor is None else min(level, floor + self._noise_rise)
            self._noise_floor = floor
            out.append(level >= self._min_dbfs and level >= floor + self._threshold_db)
        return out
