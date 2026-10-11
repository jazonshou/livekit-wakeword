"""Second-stage verifier: re-checks a trigger before it becomes a detection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from onnxruntime.capi.onnxruntime_pybind11_state import SessionOptions

from .model import MIN_EMBEDDINGS, WakeWordModel

logger = logging.getLogger(__name__)

# ONNX metadata key holding the threshold chosen at training time
THRESHOLD_METADATA_KEY = "verifier_threshold"
DEFAULT_THRESHOLD = 0.5


class WakeWordVerifier:
    """A second classifier that confirms or rejects a wake word trigger.

    The always-on model fires first; only then is the verifier run, once, on the
    same ``(16, 96)`` speech embeddings (the last 2 s of audio).  It costs
    nothing while idle, so it can be a larger model trained only to separate
    the wake phrase from its near misses (``livekit-wakeword train-verifier``).

    Example:
        from livekit.wakeword import WakeWordListener, WakeWordModel, WakeWordVerifier

        model = WakeWordModel(models=["hey_livekit.onnx"])
        verifier = WakeWordVerifier("hey_livekit_verifier.onnx")
        async with WakeWordListener(model, verifier=verifier) as listener:
            detection = await listener.wait_for_detection()

    Without the listener, pass the embeddings of a ``StreamingWakeWordModel``
    (``stream.embeddings()``) to :meth:`score`, or ~2 s of audio to
    :meth:`score_audio`.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float | None = None,
        sess_options: SessionOptions | None = None,
    ):
        """Load a verifier classifier.

        Args:
            model_path: ONNX classifier taking ``(batch, 16, 96)`` embeddings, as
                exported by ``livekit-wakeword export-verifier``.
            threshold: Minimum verifier score to accept a trigger.  None uses the
                threshold stored in the model at training time, or 0.5 if it
                has none.
        """
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Verifier model not found: {model_path}")
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=sess_options
        )
        self._input_name = self._session.get_inputs()[0].name
        self.name = model_path.stem
        if threshold is None:
            meta = self._session.get_modelmeta().custom_metadata_map
            threshold = float(meta.get(THRESHOLD_METADATA_KEY, DEFAULT_THRESHOLD))
        self.threshold = threshold
        logger.info(
            "Loaded verifier '%s' from %s (threshold %.3f)", self.name, model_path, threshold
        )

    def score(self, embeddings: np.ndarray) -> float:
        """Score one ``(16, 96)`` embedding sequence (or ``(1, 16, 96)``)."""
        emb = np.asarray(embeddings, dtype=np.float32).reshape(-1, MIN_EMBEDDINGS, 96)[-1:]
        outputs = self._session.run(None, {self._input_name: emb})
        return float(outputs[0][0, 0])

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Score ``(batch, 16, 96)`` embeddings, one score per sequence."""
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        if emb.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: emb})
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def verify(self, embeddings: np.ndarray) -> tuple[bool, float]:
        """Return ``(accepted, score)`` for one embedding sequence."""
        score = self.score(embeddings)
        return score >= self.threshold, score

    def score_audio(self, model: WakeWordModel, audio_chunk: np.ndarray) -> float:
        """Score ~2 s of 16 kHz audio, using ``model``'s feature extractor.

        Returns 0.0 if the chunk is too short to give 16 embeddings.
        """
        emb = model.embed(audio_chunk)
        return 0.0 if emb is None else self.score(emb)
