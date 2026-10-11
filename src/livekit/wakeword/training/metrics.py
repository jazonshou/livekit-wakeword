"""Training and evaluation metrics for wake word detection."""

from __future__ import annotations

import numpy as np

HOP_SECONDS = 0.08  # one embedding step = one 80 ms streaming hop
WINDOW_STEPS = 16  # classifier input length in embedding steps


def false_positives_per_hour(
    predictions: np.ndarray,
    threshold: float,
    total_hours: float,
) -> float:
    """Compute false positives per hour (FPPH).

    Args:
        predictions: Model scores for negative samples
        threshold: Detection threshold
        total_hours: Total hours of audio represented

    Returns:
        False positives per hour
    """
    if total_hours <= 0:
        return float("inf")
    fp_count = np.sum(predictions >= threshold)
    return float(fp_count / total_hours)


def stride1_windows(features: np.ndarray, window: int = WINDOW_STEPS) -> np.ndarray:
    """Every ``window``-step window of a continuous ``(N, D)`` embedding stream.

    A live stream scores once per 80 ms hop, i.e. once per embedding step, so
    false accepts must be counted over stride-1 windows rather than
    back-to-back blocks. Returns a read-only view shaped ``(N - window + 1, window, D)``.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected (N, D) stream features, got shape {features.shape}")
    if features.shape[0] < window:
        return np.zeros((0, window, features.shape[1]), dtype=features.dtype)
    view = np.lib.stride_tricks.sliding_window_view(features, window, axis=0)
    # sliding_window_view puts the window axis last: (N-w+1, D, w) -> (N-w+1, w, D)
    return view.transpose(0, 2, 1)


def detection_indices(
    scores: np.ndarray,
    threshold: float,
    debounce_hops: int = 25,
    min_consecutive: int = 1,
) -> np.ndarray:
    """Hops at which a streaming detector fires, as the live listener would.

    A detection fires on the ``min_consecutive``-th consecutive hop at or above
    ``threshold``; after a detection, further hops are ignored for
    ``debounce_hops`` hops.

    Args:
        scores: Per-hop scores of one continuous stream.
        threshold: Detection threshold.
        debounce_hops: Minimum hops between two detections.
        min_consecutive: Hops above threshold needed before firing.

    Returns:
        Sorted hop indices of the detections.
    """
    above = np.asarray(scores) >= threshold
    if min_consecutive > 1:
        if above.shape[0] < min_consecutive:
            return np.zeros(0, dtype=np.int64)
        runs = np.lib.stride_tricks.sliding_window_view(above, min_consecutive).all(axis=1)
        candidates = np.flatnonzero(runs) + (min_consecutive - 1)
    else:
        candidates = np.flatnonzero(above)
    if candidates.shape[0] == 0 or debounce_hops <= 1:
        return candidates.astype(np.int64)

    fired: list[int] = []
    i = 0
    while i < candidates.shape[0]:
        hop = int(candidates[i])
        fired.append(hop)
        i = int(np.searchsorted(candidates, hop + debounce_hops, side="left"))
    return np.asarray(fired, dtype=np.int64)


def streaming_false_accepts(
    scores: np.ndarray,
    threshold: float,
    debounce_hops: int = 25,
    min_consecutive: int = 1,
) -> int:
    """Number of detections a streaming detector makes on a negative stream."""
    return int(detection_indices(scores, threshold, debounce_hops, min_consecutive).shape[0])


def recall_at_threshold(
    predictions: np.ndarray,
    threshold: float,
) -> float:
    """Compute recall (true positive rate) at a given threshold.

    Args:
        predictions: Model scores for positive samples
        threshold: Detection threshold

    Returns:
        Recall (0-1)
    """
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(predictions >= threshold))


def accuracy(
    positive_preds: np.ndarray,
    negative_preds: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute balanced accuracy.

    Args:
        positive_preds: Model scores for positive samples
        negative_preds: Model scores for negative samples
        threshold: Detection threshold

    Returns:
        Balanced accuracy (0-1): average of true positive rate and true negative rate
    """
    if len(positive_preds) == 0 and len(negative_preds) == 0:
        return 0.0
    tpr = float(np.mean(positive_preds >= threshold)) if len(positive_preds) > 0 else 0.0
    tnr = float(np.mean(negative_preds < threshold)) if len(negative_preds) > 0 else 0.0
    return (tpr + tnr) / 2.0


def evaluate_model(
    positive_preds: np.ndarray,
    negative_preds: np.ndarray,
    threshold: float = 0.5,
    validation_hours: float = 11.0,
    *,
    stream_preds: np.ndarray | None = None,
    debounce_hops: int = 25,
    min_consecutive: int = 1,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        positive_preds: Scores for positive clips.
        negative_preds: Scores for negative clips (one decision per clip).
        threshold: Detection threshold.
        validation_hours: Hours of audio covered by ``negative_preds``.
        stream_preds: Optional per-hop scores of a continuous negative stream
            (one per 80 ms hop, see :func:`stride1_windows`). Its false accepts
            are counted with :func:`streaming_false_accepts` and its duration is
            added to ``validation_hours``.
        debounce_hops: Debounce used when counting stream false accepts.
        min_consecutive: Consecutive hops needed to fire on the stream.

    Returns dict with keys: fpph, recall, accuracy, threshold, and stream_fpph
    when ``stream_preds`` is given.
    """
    if stream_preds is None or len(stream_preds) == 0:
        return {
            "fpph": false_positives_per_hour(negative_preds, threshold, validation_hours),
            "recall": recall_at_threshold(positive_preds, threshold),
            "accuracy": accuracy(positive_preds, negative_preds, threshold),
            "threshold": threshold,
        }

    stream_hours = len(stream_preds) * HOP_SECONDS / 3600.0
    stream_fp = streaming_false_accepts(stream_preds, threshold, debounce_hops, min_consecutive)
    clip_fp = int(np.sum(negative_preds >= threshold))
    total_hours = validation_hours + stream_hours
    all_negative = (
        np.concatenate([negative_preds, stream_preds]) if len(negative_preds) else stream_preds
    )
    return {
        "fpph": float((clip_fp + stream_fp) / total_hours),
        "stream_fpph": float(stream_fp / stream_hours),
        "recall": recall_at_threshold(positive_preds, threshold),
        "accuracy": accuracy(positive_preds, all_negative, threshold),
        "threshold": threshold,
    }


def find_best_threshold(
    positive_preds: np.ndarray,
    negative_preds: np.ndarray,
    validation_hours: float = 11.0,
    target_fpph: float = 0.1,
    min_recall: float = 0.5,
    *,
    stream_preds: np.ndarray | None = None,
    debounce_hops: int = 25,
    min_consecutive: int = 1,
) -> dict[str, float]:
    """Find the threshold that maximizes recall subject to FPPH constraint.

    Scans thresholds from 0.01 to 0.99 and picks the one with the highest
    recall while keeping FPPH at or below target_fpph.  Falls back to
    maximizing balanced accuracy if no threshold meets the FPPH target.

    Args:
        positive_preds: Model scores for positive samples.
        negative_preds: Model scores for negative samples.
        validation_hours: Total hours of negative audio in ``negative_preds``.
        target_fpph: Maximum acceptable false positives per hour.
        min_recall: Minimum acceptable recall (ignores thresholds below this).
        stream_preds: Optional per-hop negative stream scores (see :func:`evaluate_model`).
        debounce_hops: Debounce used when counting stream false accepts.
        min_consecutive: Consecutive hops needed to fire on the stream.

    Returns:
        Dict with keys: fpph, recall, accuracy, threshold
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best: dict[str, float] | None = None
    best_fallback: dict[str, float] | None = None

    for t in thresholds:
        t_float = float(t)
        metrics = evaluate_model(
            positive_preds,
            negative_preds,
            threshold=t_float,
            validation_hours=validation_hours,
            stream_preds=stream_preds,
            debounce_hops=debounce_hops,
            min_consecutive=min_consecutive,
        )
        if metrics["recall"] < min_recall:
            continue

        # Track best that meets FPPH constraint
        if metrics["fpph"] <= target_fpph:
            if best is None or metrics["recall"] > best["recall"]:
                best = metrics

        # Track overall best balanced accuracy as fallback
        if best_fallback is None or metrics["accuracy"] > best_fallback["accuracy"]:
            best_fallback = metrics

    if best is not None:
        return best
    if best_fallback is not None:
        return best_fallback
    # Nothing met min_recall — return default
    return evaluate_model(
        positive_preds,
        negative_preds,
        threshold=0.5,
        validation_hours=validation_hours,
        stream_preds=stream_preds,
        debounce_hops=debounce_hops,
        min_consecutive=min_consecutive,
    )
