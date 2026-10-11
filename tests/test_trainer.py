"""Tests for training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.training.metrics import (
    accuracy,
    evaluate_model,
    false_positives_per_hour,
    recall_at_threshold,
)


class TestMetrics:
    def test_fpph(self):
        preds = np.array([0.1, 0.6, 0.8, 0.3, 0.9])
        fpph = false_positives_per_hour(preds, threshold=0.5, total_hours=1.0)
        assert fpph == 3.0

    def test_fpph_zero_hours(self):
        preds = np.array([0.9])
        assert false_positives_per_hour(preds, 0.5, 0.0) == float("inf")

    def test_recall(self):
        preds = np.array([0.9, 0.8, 0.3, 0.95])
        recall = recall_at_threshold(preds, threshold=0.5)
        assert recall == 0.75

    def test_recall_empty(self):
        assert recall_at_threshold(np.array([]), 0.5) == 0.0

    def test_accuracy(self):
        pos = np.array([0.9, 0.8, 0.3])
        neg = np.array([0.1, 0.2, 0.6])
        acc = accuracy(pos, neg, threshold=0.5)
        # 2 TP + 2 TN = 4/6
        assert abs(acc - 4 / 6) < 1e-6

    def test_evaluate_model(self):
        pos = np.array([0.9, 0.8])
        neg = np.array([0.1, 0.2])
        result = evaluate_model(pos, neg, threshold=0.5, validation_hours=1.0)
        assert "fpph" in result
        assert "recall" in result
        assert "accuracy" in result
        assert result["recall"] == 1.0
        assert result["fpph"] == 0.0


class TestDataset:
    def test_mmap_batch_generator(self, sample_features_file: Path):
        from livekit.wakeword.data.dataset import mmap_batch_generator

        gen = mmap_batch_generator(
            data_files={"pos": sample_features_file, "neg": sample_features_file},
            n_per_class={"pos": 10, "neg": 20},
            label_funcs={"pos": lambda _: 1, "neg": lambda _: 0},
        )
        features, labels = next(gen)
        assert features.shape == (30, 16, 96)
        assert labels.shape == (30,)
        assert np.sum(labels == 1) == 10
        assert np.sum(labels == 0) == 20

    def test_mmap_batch_generator_seq_len_and_class_ids(self, tmp_path: Path):
        from livekit.wakeword.data.dataset import mmap_batch_generator

        contiguous = tmp_path / "contiguous.npy"
        np.save(str(contiguous), np.random.randn(200, 96).astype(np.float32))
        long_clips = tmp_path / "long.npy"
        np.save(str(long_clips), np.random.randn(10, 24, 96).astype(np.float32))

        gen = mmap_batch_generator(
            data_files={"pos": long_clips, "neg": contiguous},
            n_per_class={"pos": 3, "neg": 5},
            label_funcs={"pos": lambda _: 1, "neg": lambda _: 0},
            seq_len=20,
            with_class_ids=True,
        )
        features, labels, class_ids = next(gen)
        assert features.shape == (8, 20, 96)
        # class ids follow data_files order and line up with labels after shuffling
        assert np.array_equal(class_ids == 0, labels == 1)

    def test_mmap_batch_generator_rejects_short_examples(self, sample_features_file: Path):
        from livekit.wakeword.data.dataset import mmap_batch_generator

        gen = mmap_batch_generator(
            data_files={"pos": sample_features_file},
            n_per_class={"pos": 2},
            label_funcs={"pos": lambda _: 1},
            seq_len=20,
        )
        with pytest.raises(ValueError, match="embedding steps"):
            next(gen)


class TestTrainingOptions:
    def test_max_pool_needs_longer_clips(self, sample_config: WakeWordConfig):
        data = sample_config.model_dump()
        with pytest.raises(ValueError, match="clip_duration"):
            WakeWordConfig(**{**data, "max_pool_steps": 4})
        data["augmentation"]["clip_duration"] = 2.32
        config = WakeWordConfig(**{**data, "max_pool_steps": 4})
        assert config.feature_steps == 20

    def test_score_takes_max_over_windows(self, sample_config: WakeWordConfig):
        import torch

        from livekit.wakeword.training.trainer import WakeWordTrainer

        trainer = WakeWordTrainer(sample_config, device=torch.device("cpu"))
        trainer.model.eval()
        features = torch.randn(3, 19, 96)
        with torch.no_grad():
            pooled = trainer._score(features)
            per_window = torch.stack(
                [trainer._score(features[:, i : i + 16]) for i in range(4)], dim=1
            )
        assert pooled.shape == (3,)
        assert torch.allclose(pooled, per_window.max(dim=1).values)

    def test_mixup_leaves_excluded_classes_alone(self, sample_config: WakeWordConfig):
        import torch

        from livekit.wakeword.training.trainer import WakeWordTrainer

        config = sample_config.model_copy(
            update={"mixup_alpha": 0.4, "mixup_exclude_classes": ["adversarial_negative"]}
        )
        trainer = WakeWordTrainer(config, device=torch.device("cpu"))
        trainer._class_names = ["positive", "adversarial_negative", "ACAV100M_sample"]
        features = torch.randn(30, 16, 96)
        labels = torch.tensor([1.0] * 10 + [0.0] * 20)
        class_ids = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)

        mixed, mixed_labels = trainer._mixup(features, labels, class_ids)
        assert torch.equal(mixed[10:20], features[10:20])
        assert torch.equal(mixed_labels[10:20], labels[10:20])

        off = trainer.config.model_copy(update={"mixup_alpha": 0.0})
        trainer.config = off
        same, same_labels = trainer._mixup(features, labels, class_ids)
        assert same is features and same_labels is labels

    def test_onnx_eval_takes_max_over_windows(self):
        from livekit.wakeword.eval.evaluate import _predict_onnx

        class _Input:
            name = "x"
            shape = ["batch", 16, 96]

        class _Session:
            def get_inputs(self) -> list[_Input]:
                return [_Input()]

            def run(self, _: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
                return [feeds["x"][:, -1, :1]]  # score = first value of the last step

        features = np.zeros((2, 18, 96), dtype=np.float32)
        features[0, 16, 0] = 0.9  # last step of the second window
        features[1, 17, 0] = 0.4
        scores = _predict_onnx(_Session(), features, batch_size=4)
        assert np.allclose(scores, [0.9, 0.4])
