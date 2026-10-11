"""Streaming wake word model: one new speech embedding per 80 ms hop."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from .model import EMBEDDING_STRIDE, EMBEDDING_WINDOW, MIN_EMBEDDINGS, WakeWordModel

HOP_SAMPLES = 1280  # 80 ms; one new embedding per hop
WINDOW_SAMPLES = 32000  # 2 s, the window WakeWordModel.predict() is designed for
MEL_HOP_SAMPLES = 160
MEL_FRAME_SAMPLES = 512
MEL_FRAMES_PER_HOP = HOP_SAMPLES // MEL_HOP_SAMPLES  # 8 == EMBEDDING_STRIDE
WINDOW_MEL_FRAMES = (WINDOW_SAMPLES - MEL_FRAME_SAMPLES) // MEL_HOP_SAMPLES + 1  # 197
# melspectrogram.onnx clamps its dB output at (loudest frame in the input - 80 dB).
# After the x/10 + 2 post-processing that floor sits 8 units below the maximum.
TOP_DB = 8.0


class _CachedEmbedding(NamedTuple):
    position: int  # embedding number since reset(); its mel frames start at position * 8
    vector: np.ndarray  # (96,)
    floor: float  # mel floor the vector was computed with
    min_mel: float  # smallest unfloored mel value among its 76 frames


class StreamingWakeWordModel:
    """Incremental wake word scoring for a continuous 16 kHz audio stream.

    Wraps a :class:`WakeWordModel` and shares its ONNX sessions, so several
    streams can use one model.  Feed audio of any length to :meth:`process`.
    For every 1280 samples (80 ms hop) it computes the mel frames of the new
    audio and one new speech embedding, keeps the last 16 embeddings, and runs
    the classifiers on them: one embedding-model call per hop, instead of the
    16 that ``WakeWordModel.predict()`` makes on a 2 s window.

    At every hop the scores equal ``predict()`` on the most recent 2 s of
    audio, up to float rounding.  The mel model floors quiet frames relative
    to the loudest frame in its input, so when the loudest frame of the 2 s
    window changes, cached embeddings that touched the floor are recomputed.
    With a live microphone's noise floor that is rare.

    Example:
        from livekit.wakeword import StreamingWakeWordModel, WakeWordModel

        stream = StreamingWakeWordModel(WakeWordModel(models=["hey_livekit.onnx"]))
        for frame in microphone_frames():  # any frame size
            for scores in stream.process(frame):
                if scores["hey_livekit"] >= 0.5:
                    ...
    """

    def __init__(self, model: WakeWordModel):
        self._model = model
        self.reset()

    @property
    def model(self) -> WakeWordModel:
        """The wrapped stateless model."""
        return self._model

    def reset(self) -> None:
        """Drop all buffered audio and cached features (e.g. after a detection).

        The next 2 s of audio warm the stream up again before scores resume.
        """
        self._pending = np.zeros(0, dtype=np.float32)  # samples short of a full hop
        self._mel_audio = np.zeros(0, dtype=np.float32)  # audio from the next mel frame on
        self._n_samples = 0
        self._n_mel = 0
        # Mel frames of the current 2 s window, each floored only by its own mel call.
        self._mel = np.zeros((0, 32), dtype=np.float32)
        self._embeddings: deque[_CachedEmbedding] = deque(maxlen=MIN_EMBEDDINGS)

    def process(self, audio: np.ndarray) -> list[dict[str, float]]:
        """Feed audio and score every hop it completes.

        Args:
            audio: 16 kHz mono samples, int16 or float32, any length.

        Returns:
            One ``{model_name: score}`` dict per 80 ms hop completed by this
            call, oldest first.  Hops within the first 2 s after construction
            or :meth:`reset` are not scored, so this may be empty.
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        buf = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32).ravel()])

        n_hops = len(buf) // HOP_SAMPLES
        results = []
        for i in range(n_hops):
            scores = self._step(buf[i * HOP_SAMPLES : (i + 1) * HOP_SAMPLES])
            if scores is not None:
                results.append(scores)
        self._pending = buf[n_hops * HOP_SAMPLES :].copy()
        return results

    def _step(self, hop: np.ndarray) -> dict[str, float] | None:
        self._n_samples += HOP_SAMPLES
        self._append_mel(np.concatenate([self._mel_audio, hop]))

        mel_start = self._n_mel - len(self._mel)
        floor = float(self._mel.max()) - TOP_DB

        next_index = self._embeddings[-1].position + 1 if self._embeddings else 0
        while next_index * EMBEDDING_STRIDE + EMBEDDING_WINDOW <= self._n_mel:
            self._embeddings.append(self._embed(next_index, mel_start, floor))
            next_index += 1

        if self._n_samples < WINDOW_SAMPLES or len(self._embeddings) < MIN_EMBEDDINGS:
            return None

        # An embedding stays valid under a new floor unless one of its frames
        # sits below the higher of the two floors.
        for i, cached in enumerate(self._embeddings):
            if cached.floor != floor and cached.min_mel < max(cached.floor, floor):
                self._embeddings[i] = self._embed(cached.position, mel_start, floor)

        return self._model._score_embeddings(np.stack([e.vector for e in self._embeddings]))

    def _append_mel(self, audio: np.ndarray) -> None:
        """Compute every complete mel frame in ``audio`` (which starts at frame ``_n_mel``).

        Frames are computed in runs that never cross a multiple of 8.  The
        2 s window always starts on such a boundary, so each run's own floor
        (its loudest frame - 8) is at or below the floor of any window that
        contains it, and ``max(value, window_floor)`` reproduces the
        full-window mel exactly.
        """
        start = self._n_mel
        stop = start + (len(audio) - MEL_FRAME_SAMPLES) // MEL_HOP_SAMPLES + 1
        runs = []
        frame = start
        while frame < stop:
            run_stop = min(stop, (frame // MEL_FRAMES_PER_HOP + 1) * MEL_FRAMES_PER_HOP)
            first = (frame - start) * MEL_HOP_SAMPLES
            last = (run_stop - 1 - start) * MEL_HOP_SAMPLES + MEL_FRAME_SAMPLES
            mel = self._model._mel_frontend(audio[first:last])
            runs.append(mel[0] if mel.ndim == 3 else mel)
            frame = run_stop

        self._mel = np.concatenate([self._mel, *runs])[-WINDOW_MEL_FRAMES:]
        self._mel_audio = audio[(stop - start) * MEL_HOP_SAMPLES :]
        self._n_mel = stop

    def _embed(self, index: int, mel_start: int, floor: float) -> _CachedEmbedding:
        offset = index * EMBEDDING_STRIDE - mel_start
        frames = self._mel[offset : offset + EMBEDDING_WINDOW]
        vector = self._model._speech_embedding(np.maximum(frames, floor)[np.newaxis])[0]
        return _CachedEmbedding(index, vector, floor, float(frames.min()))
