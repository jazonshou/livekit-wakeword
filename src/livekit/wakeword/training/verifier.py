"""Training for the second-stage verifier (positives vs near-miss negatives).

The verifier only ever sees audio the always-on model has already accepted, so
its job is narrow: separate the wake phrase from the phrases that sound like it.
It trains on the features the main pipeline already extracted
(``positive_features_train.npy`` against ``negative_features_train.npy``, plus
optional general speech and noise), with a single phase of focal loss and no
mixup, and picks its threshold on the test splits: the highest one that keeps
``verifier.target_recall`` of ``positive_features_test``.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

from ..config import WakeWordConfig
from ..data.dataset import create_dataloader
from ..models.classifier import build_classifier
from ..utils import get_device
from .trainer import N_CLASSIFIER_STEPS, _cosine_warmup_schedule, focal_loss

logger = logging.getLogger(__name__)


class VerifierMetrics(TypedDict):
    threshold: float
    recall: float  # share of positive test clips accepted
    near_miss_accept: float  # share of near-miss test clips accepted
    near_miss_reject: float


class _Checkpoint(TypedDict):
    step: int
    metrics: VerifierMetrics
    state_dict: dict[str, torch.Tensor]


def _last_steps(features: np.ndarray) -> np.ndarray:
    """Keep the classifier's last 16 steps of ``(N, steps, 96)`` features."""
    return np.ascontiguousarray(features[:, -N_CLASSIFIER_STEPS:], dtype=np.float32)


def verifier_threshold(
    pos_scores: np.ndarray, neg_scores: np.ndarray, target_recall: float
) -> VerifierMetrics:
    """The highest threshold accepting at least ``target_recall`` of the positives."""
    if pos_scores.size == 0:
        return {"threshold": 0.5, "recall": 0.0, "near_miss_accept": 0.0, "near_miss_reject": 1.0}
    # Accepting the top k positives needs threshold <= the k-th highest score
    k = int(np.ceil(target_recall * pos_scores.size))
    threshold = float(np.sort(pos_scores)[::-1][k - 1])
    recall = float(np.mean(pos_scores >= threshold))
    accept = float(np.mean(neg_scores >= threshold)) if neg_scores.size else 0.0
    return {
        "threshold": threshold,
        "recall": recall,
        "near_miss_accept": accept,
        "near_miss_reject": 1.0 - accept,
    }


class VerifierTrainer:
    """Trains ``config.verifier`` on the features of the main pipeline."""

    def __init__(self, config: WakeWordConfig, device: torch.device | None = None):
        self.config = config
        self.vconfig = config.verifier
        self.device = device or get_device()
        self.model = build_classifier(
            model_type=self.vconfig.model.model_type,
            model_size=self.vconfig.model.model_size,
        ).to(self.device)
        self.checkpoints: list[_Checkpoint] = []
        model_dir = config.model_output_dir
        self.metrics_path = model_dir / f"{config.verifier_name}_metrics.json"
        self._metrics_log: list[dict[str, object]] = []
        self._test: tuple[np.ndarray, np.ndarray] | None = None

    def _build_dataloader(self) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
        model_dir = self.config.model_output_dir
        data_files: dict[str, str | Path] = {
            "positive": model_dir / "positive_features_train.npy",
            "adversarial_negative": model_dir / "negative_features_train.npy",
            "ACAV100M_sample": self.config.data_path
            / "features"
            / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            "background_noise": model_dir / "background_noise_features_train.npy",
        }
        for required in ("positive", "adversarial_negative"):
            if not Path(data_files[required]).exists():
                raise FileNotFoundError(
                    f"{data_files[required]} not found. Run generate, augment and "
                    "extraction for this config before train-verifier."
                )
        n_per_class = self.vconfig.batch_n_per_class
        data_files = {
            name: path
            for name, path in data_files.items()
            if n_per_class.get(name, 0) > 0 and Path(path).exists()
        }
        label_funcs: dict[str, Callable[[np.ndarray], int]] = {
            name: (lambda _: 1) if name == "positive" else (lambda _: 0) for name in data_files
        }
        logger.info("Verifier training classes: %s", {k: n_per_class[k] for k in data_files})
        return create_dataloader(
            data_files=data_files,
            n_per_class=n_per_class,
            label_funcs=label_funcs,
            seq_len=N_CLASSIFIER_STEPS,
        )

    def _load_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Positive and near-miss test clips, as (N, 16, 96)."""
        if self._test is None:
            model_dir = self.config.model_output_dir
            empty = np.zeros((0, N_CLASSIFIER_STEPS, 96), dtype=np.float32)
            parts = []
            for name in ("positive_features_test.npy", "negative_features_test.npy"):
                path = model_dir / name
                parts.append(_last_steps(np.load(str(path))) if path.exists() else empty)
            self._test = (parts[0], parts[1])
            logger.info(
                "Verifier validation: %d positive, %d near-miss clips",
                parts[0].shape[0],
                parts[1].shape[0],
            )
        return self._test

    @torch.no_grad()
    def predict(self, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        out = [
            self.model(torch.from_numpy(features[i : i + batch_size]).to(self.device))
            .squeeze(-1)
            .cpu()
            .numpy()
            for i in range(0, len(features), batch_size)
        ]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def validate(self) -> VerifierMetrics:
        pos, neg = self._load_test()
        return verifier_threshold(self.predict(pos), self.predict(neg), self.vconfig.target_recall)

    def _log(self, step: int, metrics: dict[str, object]) -> None:
        self._metrics_log.append({"step": step, **metrics})
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(self._metrics_log, indent=2) + "\n")

    def train(self) -> tuple[nn.Module, VerifierMetrics]:
        """Train, average the best checkpoints and return ``(model, final metrics)``."""
        from tqdm import tqdm

        v = self.vconfig
        dataloader = self._build_dataloader()
        has_test = self._load_test()[0].shape[0] > 0
        if not has_test:
            logger.warning("No positive_features_test.npy; the verifier threshold defaults to 0.5")

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=v.learning_rate, weight_decay=v.weight_decay
        )
        steps = v.steps
        warmup, hold = steps // 10, steps // 3
        validation_interval = max(1, steps // 20)

        self.model.train()
        data_iter = iter(dataloader)
        pbar = tqdm(range(steps), desc="Verifier", unit="step")
        for step in pbar:
            try:
                features, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                features, labels = next(data_iter)
            features = features.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(features).squeeze(-1)
            loss = focal_loss(predictions, labels, gamma=2.0).mean()

            lr = _cosine_warmup_schedule(step, steps, warmup, hold, v.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}")

            last = step == steps - 1
            if has_test and (last or (step >= steps // 2 and step % validation_interval == 0)):
                metrics = self.validate()
                pbar.write(
                    f"  Verifier @ step {step}: near-miss reject="
                    f"{metrics['near_miss_reject']:.3f} at recall={metrics['recall']:.3f} "
                    f"(threshold {metrics['threshold']:.3f})"
                )
                self._log(step, dict(metrics))
                self.checkpoints.append(
                    {
                        "step": step,
                        "metrics": metrics,
                        "state_dict": copy.deepcopy(self.model.state_dict()),
                    }
                )
                self.model.train()

        self._average_best()
        final = (
            self.validate()
            if has_test
            else verifier_threshold(np.zeros(0), np.zeros(0), v.target_recall)
        )
        self._log(steps, {**final, "note": "final_averaged"})
        logger.info(
            "Verifier: rejects %.1f%% of near misses at %.1f%% recall (threshold %.3f)",
            100 * final["near_miss_reject"],
            100 * final["recall"],
            final["threshold"],
        )
        return self.model, final

    def _average_best(self) -> None:
        """Average the checkpoints that reject the most near misses at target recall."""
        if not self.checkpoints:
            return
        best = sorted(self.checkpoints, key=lambda c: c["metrics"]["near_miss_reject"])[
            -self.vconfig.n_average :
        ]
        logger.info("Averaging verifier checkpoints from steps %s", [c["step"] for c in best])
        avg = {
            key: torch.stack([c["state_dict"][key].float() for c in best]).mean(dim=0)
            for key in best[0]["state_dict"]
        }
        self.model.load_state_dict(avg)


def export_verifier(model: nn.Module, output_path: Path, threshold: float) -> Path:
    """Export the verifier to ONNX with its threshold in the model metadata."""
    import onnx

    from ..inference.verifier import THRESHOLD_METADATA_KEY

    model = copy.deepcopy(model).cpu().eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.randn(2, N_CLASSIFIER_STEPS, 96),),
        str(output_path),
        opset_version=18,
        input_names=["embeddings"],
        output_names=["score"],
        dynamic_axes={"embeddings": {0: "batch"}, "score": {0: "batch"}},
    )
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    del onnx_model.metadata_props[:]
    entry = onnx_model.metadata_props.add()
    entry.key, entry.value = THRESHOLD_METADATA_KEY, repr(float(threshold))
    onnx.save(onnx_model, str(output_path), save_as_external_data=False)
    external = output_path.with_suffix(".onnx.data")
    if external.exists():
        external.unlink()
    logger.info("Exported verifier ONNX to %s (threshold %.3f)", output_path, threshold)
    return output_path


def run_train_verifier(config: WakeWordConfig) -> Path:
    """Train the verifier and export ``<model_name>_verifier.onnx``; returns its path."""
    trainer = VerifierTrainer(config)
    model, metrics = trainer.train()
    model_dir = config.model_output_dir
    torch.save(model.state_dict(), model_dir / f"{config.verifier_name}.pt")
    return export_verifier(model, model_dir / f"{config.verifier_name}.onnx", metrics["threshold"])
