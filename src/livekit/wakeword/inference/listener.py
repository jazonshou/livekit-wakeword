"""Async wake word listener with audio capture."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .model import WakeWordModel
from .streaming import StreamingWakeWordModel
from .verifier import WakeWordVerifier

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80ms per frame
CHUNK_SECONDS = 2.0
# Number of frames that fill a ~2-second chunk (25 × 80ms = 2000ms)
CHUNK_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)


@dataclass
class Detection:
    """Wake word detection result."""

    name: str
    confidence: float
    timestamp: float
    # Second-stage verifier score, or None when no verifier checked this detection
    verifier_score: float | None = None


class WakeWordListener:
    """Async wake word listener that handles audio capture.

    Each 80 ms microphone frame goes to a ``StreamingWakeWordModel``, which
    scores the most recent 2 s of audio with one new embedding per frame.
    Scoring starts once 2 s of audio have been captured.  After a detection
    the loop pauses automatically, drops the buffered audio, and resumes when
    the consumer calls ``wait_for_detection()`` again.

    With a ``verifier``, a trigger from the always-on model becomes a
    detection only if the verifier, run once on the same 2 s of embeddings,
    accepts it.  Rejected triggers are dropped without pausing, and the next
    hop that still meets the firing rule is checked again.

    Example:
        from livekit.wakeword import WakeWordModel, WakeWordListener

        model = WakeWordModel(models=["hey_livekit.onnx"])

        async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
            while True:
                detection = await listener.wait_for_detection()
                print(f"Detected {detection.name}! (confidence={detection.confidence:.2f})")
    """

    def __init__(
        self,
        model: WakeWordModel | StreamingWakeWordModel,
        threshold: float = 0.5,
        debounce: float = 2.0,
        min_consecutive: int = 1,
        verifier: WakeWordVerifier | Mapping[str, WakeWordVerifier] | None = None,
    ):
        """Initialize listener.

        Args:
            model: WakeWordModel instance with loaded classifiers, or a
                StreamingWakeWordModel wrapping one.
            threshold: Detection threshold (0-1).
            debounce: Minimum seconds between detections.
            min_consecutive: Number of consecutive 80 ms frames a model's
                score must stay at or above ``threshold`` before it fires.
                The default of 1 fires on the first frame above threshold.
            verifier: Optional second-stage check (off by default).  A single
                ``WakeWordVerifier`` checks every model's triggers; a mapping
                from model name to verifier checks only the models it names.
        """
        if min_consecutive < 1:
            raise ValueError(f"min_consecutive must be >= 1, got {min_consecutive}")
        if isinstance(model, WakeWordModel):
            model = StreamingWakeWordModel(model)
        self._model = model
        self._threshold = threshold
        self._debounce = debounce
        self._min_consecutive = min_consecutive
        self._verifier = verifier
        # Triggers the verifier has rejected since the listener started
        self.rejected_triggers = 0
        # Consecutive frames each model has scored at or above threshold
        self._consecutive: dict[str, int] = {}

        self._stream = None
        self._pa = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_detection_time = 0.0
        self._detection_queue: asyncio.Queue[Detection] = asyncio.Queue()

        # Error propagation: stored exception from _audio_loop crash
        self._error: BaseException | None = None

        # Single-thread executor keeps inference off the event loop
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Pause/resume control: cleared after detection, set when consumer
        # calls wait_for_detection() again.
        self._listening = asyncio.Event()

        # Signals when _audio_loop exits (success or crash) so
        # wait_for_detection() can raise instead of hanging forever.
        self._done_event = asyncio.Event()

    async def __aenter__(self) -> WakeWordListener:
        """Start audio capture."""
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )
        self._running = True
        self._listening.set()
        self._done_event.clear()
        self._error = None
        self._model.reset()
        self._consecutive.clear()
        self.rejected_triggers = 0
        self._detection_queue = asyncio.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._task = asyncio.create_task(self._audio_loop())
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop audio capture with safe shutdown sequence."""
        # 1. Signal the loop to stop
        self._running = False
        # Unblock if paused so the loop can see _running=False
        self._listening.set()

        # 2. Let the loop exit naturally (worst case: one 80ms read finishes)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        # 3. Wait for any in-flight executor work to complete
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # 4. Now safe to close stream — no thread is reading from it
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    async def _audio_loop(self) -> None:
        """Background task that captures audio and runs detection."""
        loop = asyncio.get_event_loop()

        try:
            while self._running:
                # Block here when paused (after detection, until consumer resumes)
                await self._listening.wait()
                if not self._running:
                    break

                # Read audio in executor to not block event loop
                data = await loop.run_in_executor(
                    self._executor,
                    lambda: self._stream.read(  # type: ignore[union-attr]
                        FRAME_SAMPLES, exception_on_overflow=False
                    ),
                )
                if not self._running:
                    break

                frame = np.frombuffer(data, dtype=np.int16)

                # Score the new frame in the executor (empty while warming up)
                hop_scores = await loop.run_in_executor(
                    self._executor,
                    self._model.process,
                    frame,
                )
                if not self._running:
                    break

                # Check for detections (lightweight, fine on event loop)
                detection = None
                for scores in hop_scores:
                    for name, score in self._candidates(scores):
                        verifier = self._verifier_for(name)
                        verifier_score = None
                        if verifier is not None:
                            # One classifier call on the cached embeddings
                            accepted, verifier_score = await loop.run_in_executor(
                                self._executor, self._verify, verifier
                            )
                            if not accepted:
                                self.rejected_triggers += 1
                                logger.info(
                                    "Verifier rejected '%s' trigger (score %.2f, verifier %.2f)",
                                    name,
                                    score,
                                    verifier_score,
                                )
                                continue
                        now = time.monotonic()
                        self._last_detection_time = now
                        detection = Detection(
                            name=name,
                            confidence=score,
                            timestamp=now,
                            verifier_score=verifier_score,
                        )
                        break
                    if detection is not None:
                        break
                if not self._running:
                    break
                if detection is not None:
                    # Pause the loop and drop buffered audio so no stale
                    # audio is processed while the consumer handles the
                    # detection.
                    self._listening.clear()
                    self._model.reset()
                    self._consecutive.clear()
                    await self._detection_queue.put(detection)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Audio loop crashed: %s", exc, exc_info=True)
            self._error = exc
        finally:
            self._done_event.set()

    def _candidates(self, scores: dict[str, float]) -> list[tuple[str, float]]:
        """Update consecutive counts with one frame's scores; return the models that fire."""
        now = time.monotonic()
        fired = []
        for name, score in scores.items():
            count = self._consecutive.get(name, 0) + 1 if score >= self._threshold else 0
            self._consecutive[name] = count
            if count >= self._min_consecutive and now - self._last_detection_time >= self._debounce:
                fired.append((name, score))
        return fired

    def _verifier_for(self, name: str) -> WakeWordVerifier | None:
        if self._verifier is None or isinstance(self._verifier, WakeWordVerifier):
            return self._verifier
        return self._verifier.get(name)

    def _verify(self, verifier: WakeWordVerifier) -> tuple[bool, float]:
        """Run the verifier on the embeddings the latest hop was scored on."""
        embeddings = self._model.embeddings()
        if embeddings is None:
            return False, 0.0
        return verifier.verify(embeddings)

    async def wait_for_detection(self) -> Detection:
        """Wait for and return the next wake word detection.

        Resumes the audio loop if it was paused after a previous detection.

        Raises:
            RuntimeError: If the background audio loop has crashed.
        """
        # Resume listening (no-op if already active)
        self._listening.set()

        # Race: either we get a detection, or the audio loop dies
        queue_waiter = asyncio.ensure_future(self._detection_queue.get())
        done_waiter = asyncio.ensure_future(self._done_event.wait())

        done, pending = await asyncio.wait(
            {queue_waiter, done_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if queue_waiter in done:
            return queue_waiter.result()

        # Loop ended — check if there's still a queued detection
        if not self._detection_queue.empty():
            return self._detection_queue.get_nowait()

        if self._error is not None:
            raise RuntimeError(f"Audio loop crashed: {self._error}") from self._error

        raise RuntimeError("Audio loop ended unexpectedly")
