"""Evaluate a wake word model and produce a DET curve plot."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import WakeWordConfig
from ..training.metrics import HOP_SECONDS, evaluate_model, find_best_threshold
from ..training.validation import debounce_hops, load_validation_data, score_stream

logger = logging.getLogger(__name__)


def _predict_onnx(
    session: ort.InferenceSession,
    features: np.ndarray,
    batch_size: int = 1,
) -> np.ndarray:
    """Run ONNX model on feature batches, return scores array.

    Clips longer than the classifier's 16 steps (``max_pool_steps``) are scored on every
    16-step window and keep their highest score, as in training.
    """
    if features.ndim == 3 and features.shape[1] > 16:
        n_clips, steps, dim = features.shape
        windows = np.lib.stride_tricks.sliding_window_view(features, 16, axis=1)
        windows = windows.transpose(0, 1, 3, 2).reshape(-1, 16, dim)
        scores = _predict_onnx(session, windows, batch_size)
        pooled: np.ndarray = scores.reshape(n_clips, steps - 15).max(axis=1)
        return pooled
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    shape = getattr(model_input, "shape", None)
    if shape and isinstance(shape[0], int):  # fixed batch size (e.g. some openWakeWord models)
        batch_size = shape[0]
    all_scores: list[np.ndarray] = []
    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size].astype(np.float32)
        outputs = session.run(None, {input_name: batch})
        all_scores.append(outputs[0].reshape(-1))
    return np.concatenate(all_scores, axis=0)


def _compute_det_curve(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DET curve points (FPR, FNR) across thresholds.

    Returns:
        (thresholds, fpr, fnr) arrays sorted by ascending threshold.
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    fpr = np.array([np.mean(neg_scores >= t) for t in thresholds])
    fnr = np.array([np.mean(pos_scores < t) for t in thresholds])
    return thresholds, fpr, fnr


def _compute_aut(fpr: np.ndarray, fnr: np.ndarray) -> float:
    """Compute Area Under the DET curve (AUT) using the trapezoidal rule.

    Lower is better (0 = perfect).
    We integrate FNR as a function of FPR (sorted by ascending FPR).
    """
    # Sort by FPR for proper integration
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    fnr_sorted = fnr[sort_idx]
    return float(np.trapezoid(fnr_sorted, fpr_sorted))


def _plot_det_curve(
    fpr: np.ndarray,
    fnr: np.ndarray,
    aut: float,
    model_name: str,
    output_path: Path,
    metrics: dict[str, float],
) -> None:
    """Render DET curve to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot DET curve (FPR vs FNR)
    sort_idx = np.argsort(fpr)
    ax.plot(fpr[sort_idx] * 100, fnr[sort_idx] * 100, linewidth=2, color="#2563eb")

    # Shade AUT area
    ax.fill_between(
        fpr[sort_idx] * 100,
        fnr[sort_idx] * 100,
        alpha=0.15,
        color="#2563eb",
    )

    ax.set_xlabel("False Positive Rate (%)", fontsize=13)
    ax.set_ylabel("False Negative Rate (%)", fontsize=13)
    ax.set_title(f"DET Curve \u2014 {model_name}", fontsize=15, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)

    # Diagonal reference (random classifier)
    ax.plot([0, 100], [100, 0], "--", color="gray", alpha=0.5, label="Random")

    # Annotation box with metrics
    text_lines = [
        f"AUT: {aut:.4f}",
        f"FPPH: {metrics['fpph']:.2f}",
        f"Recall: {metrics['recall']:.1%}",
        f"Threshold: {metrics['threshold']:.2f}",
        f"Optimal Thresh: {metrics['optimal_threshold']:.2f}",
    ]
    ax.text(
        0.97,
        0.97,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9),
        fontfamily="monospace",
    )

    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"DET curve saved to {output_path}")


def run_eval(config: WakeWordConfig, model_path: str | Path) -> dict[str, float]:
    """Run full evaluation: compute scores, DET curve, AUT, and save plot + metrics JSON.

    Args:
        config: Wake word configuration (used to locate validation data).
        model_path: Path to the ONNX classifier model to evaluate.

    Returns:
        Dict with keys: aut, fpph, recall, accuracy, threshold
    """
    # Load ONNX model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logger.info(f"Loaded model from {model_path}")

    # Load validation data
    data = load_validation_data(config)
    if data.positive.shape[0] == 0:
        raise ValueError(
            f"No positive validation features found in {config.model_output_dir}. "
            "Run the generate/augment pipeline first."
        )
    if data.negative_clips.shape[0] == 0 and data.stream.shape[0] == 0:
        raise ValueError(
            "No negative validation features found. "
            "Run setup and the generate/augment pipeline first."
        )

    # Run predictions. The validation stream is scored at every 80 ms hop.
    logger.info("Running predictions on validation set...")
    pos_scores = _predict_onnx(session, data.positive, batch_size=256)
    neg_scores = (
        _predict_onnx(session, data.negative_clips, batch_size=256)
        if data.negative_clips.shape[0]
        else np.zeros(0, dtype=np.float32)
    )
    stream_scores = score_stream(
        lambda x: _predict_onnx(session, x, batch_size=x.shape[0]), data.stream
    )

    # Compute DET curve (per-window rates over clips and stream hops)
    thresholds, fpr, fnr = _compute_det_curve(
        pos_scores, np.concatenate([neg_scores, stream_scores])
    )

    # Compute AUT
    aut = _compute_aut(fpr, fnr)

    # Compute summary metrics at fixed threshold 0.5 for consistent comparison
    hops = debounce_hops(config)
    min_consecutive = config.streaming_eval.min_consecutive
    fixed = evaluate_model(
        pos_scores,
        neg_scores,
        threshold=0.5,
        validation_hours=data.clip_hours,
        stream_preds=stream_scores,
        debounce_hops=hops,
        min_consecutive=min_consecutive,
    )

    optimal = find_best_threshold(
        pos_scores,
        neg_scores,
        validation_hours=data.clip_hours,
        target_fpph=config.target_fp_per_hour,
        stream_preds=stream_scores,
        debounce_hops=hops,
        min_consecutive=min_consecutive,
    )
    validation_hours = data.clip_hours + stream_scores.shape[0] * HOP_SECONDS / 3600.0

    # Build results
    results = {
        "aut": aut,
        "fpph": fixed["fpph"],
        "recall": fixed["recall"],
        "accuracy": fixed["accuracy"],
        "threshold": fixed["threshold"],
        "optimal_threshold": optimal["threshold"],
        "optimal_recall": optimal["recall"],
        "optimal_fpph": optimal["fpph"],
        "n_positive": int(pos_scores.shape[0]),
        "n_negative": int(neg_scores.shape[0] + stream_scores.shape[0]),
        "validation_hours": round(validation_hours, 2),
    }

    # Save plot
    output_dir = config.model_output_dir
    plot_path = output_dir / f"{config.model_name}_det.png"
    plot_metrics = {**fixed, "optimal_threshold": optimal["threshold"]}
    _plot_det_curve(fpr, fnr, aut, config.model_name, plot_path, plot_metrics)

    # Save metrics JSON
    metrics_path = output_dir / f"{config.model_name}_eval.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"Eval metrics saved to {metrics_path}")

    return results
