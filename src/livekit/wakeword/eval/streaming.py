"""Streaming evaluation: replay audio through the frame-by-frame detector.

Clips are laid end to end into one stream per set, separated by silence, and fed to
the detector in 80 ms frames exactly as a live microphone would. Detections use the
listener's rules (threshold, ``min_consecutive``, debounce), so the reported numbers
and the chosen threshold apply to the deployed path rather than to isolated
2-second windows.

Each set is reported separately:

- positive sets: miss rate (a clip is detected if the detector fires between the clip's
  start and ``detection_window_seconds`` after its end) and false accepts elsewhere in
  the stream (e.g. on the speech placed before each clip);
- negative sets: share of clips that fired, and false accepts per hour of audio;
- ``validation_stream``: false accepts per hour on the ~11 h validation features,
  scored at every hop.

A DET curve (pooled miss rate vs pooled false accepts per hour) is computed over
thresholds, and the threshold with the lowest miss rate at no more than
``target_fa_per_hour`` is selected.
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..config import StreamingEvalSet, WakeWordConfig
from ..training.metrics import HOP_SECONDS, detection_indices

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms hop
WINDOW_FRAMES = 25  # 2 s window used by the listener's predict() path
PRE_ROLL_SECONDS = 2.0  # silence before the first clip so the buffers are full
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)

_ORIGINAL_CLIP_RE = re.compile(r"^clip_\d{6}\.wav$")


class FrameScorer(Protocol):
    """A detector that scores one 80 ms frame at a time."""

    def reset(self) -> None: ...

    def process_frame(self, frame: np.ndarray) -> dict[str, float]: ...


class WindowScorer:
    """Scores each hop with ``WakeWordModel.predict()`` on the last 2 s of audio.

    This is the listener's pre-streaming behaviour; it recomputes 16 embeddings
    per hop, so it is slow but needs nothing beyond :class:`WakeWordModel`.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._frames: deque[np.ndarray] = deque(maxlen=WINDOW_FRAMES)

    def reset(self) -> None:
        self._frames.clear()

    def process_frame(self, frame: np.ndarray) -> dict[str, float]:
        self._frames.append(frame)
        if len(self._frames) < WINDOW_FRAMES:
            return {}
        scores: dict[str, float] = self._model.predict(np.concatenate(list(self._frames)))
        return scores


class StreamScorer:
    """Scores each hop with :class:`StreamingWakeWordModel`, the path the listener runs."""

    def __init__(self, model: Any) -> None:
        from ..inference.streaming import StreamingWakeWordModel

        self._stream = StreamingWakeWordModel(model)

    def reset(self) -> None:
        self._stream.reset()

    def process_frame(self, frame: np.ndarray) -> dict[str, float]:
        hops = self._stream.process(frame)
        return hops[-1] if hops else {}


def make_scorer(model_path: str | Path, kind: str = "streaming") -> tuple[FrameScorer, str]:
    """Build a frame scorer for one classifier.

    Args:
        model_path: ONNX classifier.
        kind: ``"streaming"`` for ``StreamingWakeWordModel`` (cached embeddings, the
            path a live device runs) or ``"window"`` for ``predict()`` on a sliding
            2 s window.

    Returns:
        (scorer, kind actually used)
    """
    from ..inference import model as inference_model

    if kind not in ("streaming", "window"):
        raise ValueError(f"Unknown scorer {kind!r}; expected 'streaming' or 'window'")
    model = inference_model.WakeWordModel(models=[model_path])
    if kind == "streaming":
        return StreamScorer(model), "streaming"
    return WindowScorer(model), "window"


@dataclass
class EvalStream:
    """One evaluation set laid out as a single audio stream."""

    name: str
    positive: bool
    audio: np.ndarray  # float32, 16 kHz
    targets: list[tuple[int, int]] = field(default_factory=list)  # clip [start, end) samples
    audio_seconds: float = 0.0  # seconds of clip audio, excluding inserted silence


def _collect_wavs(paths: Sequence[str | Path], originals_only: bool = False) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("*.wav"))
            if originals_only:
                found = [f for f in found if _ORIGINAL_CLIP_RE.match(f.name)]
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            logger.warning("Streaming eval path not found: %s", p)
    return files


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def build_stream(
    name: str,
    clips: Sequence[np.ndarray],
    *,
    positive: bool,
    gap_seconds: float,
    speech_clips: Sequence[np.ndarray] = (),
    rng: np.random.Generator | None = None,
    background: np.ndarray | None = None,
    snr_db: float = 10.0,
) -> EvalStream:
    """Lay clips end to end with silence between them.

    With ``speech_clips``, each clip is preceded by a random speech clip that ends
    0-400 ms before it. With ``background``, a looped noise bed is mixed under the
    whole stream so that the clips sit ``snr_db`` above it.
    """
    rng = rng or np.random.default_rng(0)
    gap = np.zeros(int(gap_seconds * SAMPLE_RATE), dtype=np.float32)
    parts: list[np.ndarray] = [np.zeros(int(PRE_ROLL_SECONDS * SAMPLE_RATE), dtype=np.float32)]
    voiced: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    pos = parts[0].shape[0]
    audio_samples = 0

    for clip in clips:
        if speech_clips:
            ctx = speech_clips[int(rng.integers(len(speech_clips)))]
            pause = np.zeros(int(rng.uniform(0.0, 0.4) * SAMPLE_RATE), dtype=np.float32)
            parts += [ctx, pause]
            voiced.append((pos, pos + ctx.shape[0]))
            pos += ctx.shape[0] + pause.shape[0]
            audio_samples += ctx.shape[0]
        parts += [clip, gap]
        targets.append((pos, pos + clip.shape[0]))
        voiced.append((pos, pos + clip.shape[0]))
        pos += clip.shape[0] + gap.shape[0]
        audio_samples += clip.shape[0]

    audio = np.concatenate(parts).astype(np.float32)

    if background is not None and background.shape[0] > 0:
        reps = int(np.ceil(audio.shape[0] / background.shape[0]))
        offset = int(rng.integers(background.shape[0]))
        bed = np.roll(np.tile(background, reps), -offset)[: audio.shape[0]]
        signal_power = np.mean(np.concatenate([audio[a:b] for a, b in voiced]) ** 2)
        bed_power = np.mean(bed**2)
        if signal_power > 0 and bed_power > 0:
            bed = bed * np.sqrt(signal_power / (bed_power * 10 ** (snr_db / 10)))
            audio = np.clip(audio + bed, -1.0, 1.0).astype(np.float32)

    return EvalStream(
        name=name,
        positive=positive,
        audio=audio,
        targets=targets,
        audio_seconds=audio_samples / SAMPLE_RATE,
    )


def score_audio(scorer: FrameScorer, audio: np.ndarray, model_name: str) -> np.ndarray:
    """Feed audio through the scorer in 80 ms frames; one score per hop."""
    scorer.reset()
    n_hops = int(np.ceil(audio.shape[0] / FRAME_SAMPLES))
    padded = np.zeros(n_hops * FRAME_SAMPLES, dtype=np.float32)
    padded[: audio.shape[0]] = audio
    scores = np.zeros(n_hops, dtype=np.float32)
    for i in range(n_hops):
        out = scorer.process_frame(padded[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
        scores[i] = out.get(model_name, 0.0)
    return scores


def summarize_stream(
    stream: EvalStream,
    scores: np.ndarray,
    threshold: float,
    *,
    debounce_hops: int,
    min_consecutive: int = 1,
    detection_window_seconds: float = 1.0,
) -> dict[str, float]:
    """Hits, misses and false accepts for one stream at one threshold."""
    fires = detection_indices(scores, threshold, debounce_hops, min_consecutive)
    tail = int(detection_window_seconds * SAMPLE_RATE)
    if stream.targets:
        bounds = np.asarray(stream.targets, dtype=np.int64)
        start_hop = bounds[:, 0] // FRAME_SAMPLES
        end_hop = (bounds[:, 1] + tail) // FRAME_SAMPLES
        lo = np.searchsorted(fires, start_hop, side="left")
        hi = np.searchsorted(fires, end_hop, side="right")
        fired = hi > lo
        in_window = int(np.sum(hi - lo))
    else:
        fired = np.zeros(0, dtype=bool)
        in_window = 0

    hours = stream.audio_seconds / 3600.0
    n = len(stream.targets)
    if stream.positive:
        outside = int(fires.shape[0]) - in_window
        misses = int(n - fired.sum())
        return {
            "clips": n,
            "misses": misses,
            "miss_rate": misses / n if n else 0.0,
            "false_accepts": outside,
        }
    total = int(fires.shape[0])
    return {
        "clips": n,
        "fa_per_clip": float(fired.mean()) if n else 0.0,
        "false_accepts": total,
        "fa_per_hour": total / hours if hours > 0 else 0.0,
    }


def det_curve(
    streams: Sequence[tuple[EvalStream, np.ndarray]],
    *,
    debounce_hops: int,
    min_consecutive: int = 1,
    detection_window_seconds: float = 1.0,
    thresholds: np.ndarray = THRESHOLDS,
) -> list[dict[str, float]]:
    """Pooled miss rate and false accepts per hour at each threshold."""
    points: list[dict[str, float]] = []
    for t in thresholds:
        misses = clips = false_accepts = 0
        hours = 0.0
        for stream, scores in streams:
            s = summarize_stream(
                stream,
                scores,
                float(t),
                debounce_hops=debounce_hops,
                min_consecutive=min_consecutive,
                detection_window_seconds=detection_window_seconds,
            )
            if stream.positive:
                clips += int(s["clips"])
                misses += int(s["misses"])
            else:
                false_accepts += int(s["false_accepts"])
                hours += stream.audio_seconds / 3600.0
        points.append(
            {
                "threshold": float(t),
                "miss_rate": misses / clips if clips else 0.0,
                "fa_per_hour": false_accepts / hours if hours > 0 else 0.0,
            }
        )
    return points


def select_threshold(
    points: Sequence[dict[str, float]], target_fa_per_hour: float
) -> tuple[dict[str, float], bool]:
    """Lowest miss rate with fa_per_hour <= target (highest threshold on ties).

    Falls back to the point with the fewest false accepts per hour when no
    threshold meets the target. Returns (point, target_met).
    """
    ok = [p for p in points if p["fa_per_hour"] <= target_fa_per_hour]
    if ok:
        return min(ok, key=lambda p: (p["miss_rate"], -p["threshold"])), True
    return min(points, key=lambda p: (p["fa_per_hour"], p["miss_rate"])), False


def _default_sets(config: WakeWordConfig) -> list[StreamingEvalSet]:
    model_dir = config.model_output_dir
    sets: list[StreamingEvalSet] = []
    if (model_dir / "positive_test").is_dir():
        pos = [str(model_dir / "positive_test")]
        sets.append(StreamingEvalSet(name="positive_silence", positive=True, paths=pos))
        sets.append(
            StreamingEvalSet(name="positive_speech", positive=True, paths=pos, speech_before=True)
        )
    if (model_dir / "negative_test").is_dir():
        neg = [str(model_dir / "negative_test")]
        sets.append(StreamingEvalSet(name="near_miss", positive=False, paths=neg))
    return sets


def _stream_from_validation_features(config: WakeWordConfig) -> np.ndarray | None:
    path = config.data_path / "features" / "validation_set_features.npy"
    if not path.exists():
        return None
    val = np.load(str(path), mmap_mode="r")
    return val if val.ndim == 2 else None


def _score_validation_stream(model_path: Path, stream: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    from ..training.validation import score_stream
    from .evaluate import _predict_onnx

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return score_stream(lambda x: _predict_onnx(session, x, batch_size=x.shape[0]), stream)


def _plot(
    points: Sequence[dict[str, float]], chosen: dict[str, float], path: Path, title: str
) -> None:
    try:
        import matplotlib
    except ImportError:
        logger.warning("matplotlib not installed; skipping streaming DET plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fa = np.array([p["fa_per_hour"] for p in points])
    miss = np.array([p["miss_rate"] for p in points]) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fa, miss, linewidth=2, color="#2563eb")
    ax.plot(chosen["fa_per_hour"], chosen["miss_rate"] * 100, "o", color="#dc2626")
    ax.annotate(
        f"t={chosen['threshold']:.2f}",
        (chosen["fa_per_hour"], chosen["miss_rate"] * 100),
        textcoords="offset points",
        xytext=(8, 8),
    )
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("False accepts per hour")
    ax.set_ylabel("Miss rate (%)")
    ax.set_title(f"Streaming DET — {title}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def run_streaming_eval(
    config: WakeWordConfig,
    model_path: str | Path,
    *,
    scorer: str = "streaming",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run the streaming evaluation and write ``<model>_streaming_eval.json``.

    Args:
        config: Wake word config; ``config.streaming_eval`` controls the sets and
            detector settings.
        model_path: ONNX classifier to evaluate.
        scorer: ``"streaming"`` or ``"window"`` (see :func:`make_scorer`).
        threshold: Report per-set results at this threshold instead of the
            selected one.

    Returns:
        The report written to JSON.
    """
    cfg = config.streaming_eval
    if cfg.gap_seconds <= cfg.detection_window_seconds:
        raise ValueError("streaming_eval.gap_seconds must exceed detection_window_seconds")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model_name = model_path.stem
    debounce_hops = max(1, round(cfg.debounce_seconds / HOP_SECONDS))
    rng = np.random.default_rng(cfg.seed)

    sets = (_default_sets(config) if cfg.default_sets else []) + list(cfg.sets)
    if not sets:
        raise ValueError(
            "No streaming eval sets: generate the test splits or add streaming_eval.sets"
        )

    speech_paths = cfg.speech_paths or [str(config.model_output_dir / "negative_test")]
    speech_clips: list[np.ndarray] = []
    if any(s.speech_before for s in sets):
        speech_files = _collect_wavs(speech_paths, originals_only=True)
        if cfg.max_clips_per_set:
            speech_files = speech_files[: cfg.max_clips_per_set]
        speech_clips = [_load_wav(p) for p in speech_files]

    background: np.ndarray | None = None
    if cfg.background_paths:
        bg_files = _collect_wavs(cfg.background_paths)
        if bg_files:
            background = np.concatenate([_load_wav(p) for p in bg_files])

    frame_scorer, scorer_used = make_scorer(model_path, scorer)
    logger.info("Streaming eval of %s with the %s scorer", model_name, scorer_used)

    scored: list[tuple[EvalStream, np.ndarray]] = []
    for spec in sets:
        files = _collect_wavs(spec.paths, originals_only=True) or _collect_wavs(spec.paths)
        if cfg.max_clips_per_set and len(files) > cfg.max_clips_per_set:
            idx = rng.choice(len(files), cfg.max_clips_per_set, replace=False)
            files = [files[i] for i in sorted(idx)]
        if not files:
            logger.warning("Streaming eval set %s has no WAV files; skipping", spec.name)
            continue
        if spec.speech_before and not speech_clips:
            logger.warning("No speech clips for %s; skipping", spec.name)
            continue
        stream = build_stream(
            spec.name,
            [_load_wav(p) for p in files],
            positive=spec.positive,
            gap_seconds=cfg.gap_seconds,
            speech_clips=speech_clips if spec.speech_before else (),
            rng=rng,
            background=background,
            snr_db=cfg.snr_db,
        )
        logger.info(
            "Scoring %s: %d clips, %.1f min of audio",
            spec.name,
            len(files),
            stream.audio.shape[0] / SAMPLE_RATE / 60,
        )
        scored.append((stream, score_audio(frame_scorer, stream.audio, model_name)))

    val = _stream_from_validation_features(config)
    if val is not None:
        val_scores = _score_validation_stream(model_path, val)
        val_stream = EvalStream(
            name="validation_stream",
            positive=False,
            audio=np.zeros(0, dtype=np.float32),
            audio_seconds=val_scores.shape[0] * HOP_SECONDS,
        )
        scored.append((val_stream, val_scores))

    if not any(s.positive for s, _ in scored):
        raise ValueError("Streaming eval needs at least one positive set")

    points = det_curve(
        scored,
        debounce_hops=debounce_hops,
        min_consecutive=cfg.min_consecutive,
        detection_window_seconds=cfg.detection_window_seconds,
    )
    chosen, target_met = select_threshold(points, cfg.target_fa_per_hour)
    report_threshold = chosen["threshold"] if threshold is None else threshold

    report: dict[str, Any] = {
        "model": model_name,
        "scorer": scorer_used,
        "debounce_seconds": cfg.debounce_seconds,
        "min_consecutive": cfg.min_consecutive,
        "target_fa_per_hour": cfg.target_fa_per_hour,
        "selected_threshold": chosen["threshold"],
        "target_met": target_met,
        "selected": chosen,
        "report_threshold": report_threshold,
        "sets": {
            stream.name: {
                "positive": stream.positive,
                "hours": round(stream.audio_seconds / 3600.0, 3),
                **summarize_stream(
                    stream,
                    scores,
                    report_threshold,
                    debounce_hops=debounce_hops,
                    min_consecutive=cfg.min_consecutive,
                    detection_window_seconds=cfg.detection_window_seconds,
                ),
            }
            for stream, scores in scored
        },
        "det": points,
    }

    out_dir = config.model_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{config.model_name}_streaming_eval.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    per_hop: dict[str, Any] = {stream.name: scores for stream, scores in scored}
    np.savez_compressed(out_dir / f"{config.model_name}_streaming_scores.npz", **per_hop)
    _plot(points, chosen, out_dir / f"{config.model_name}_streaming_det.png", model_name)
    logger.info("Streaming eval saved to %s", json_path)
    return report
