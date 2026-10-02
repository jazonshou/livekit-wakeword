"""Piper VITS backend: SLERP speaker blending via ``generate_samples``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...config import WakeWordConfig
from ..piper import generate_samples, normalize_phrases_for_piper
from .voices import split_voices

logger = logging.getLogger(__name__)


class PiperVitsBackend:
    """VITS + SLERP diversification; English CMUDict phrase normalization."""

    def __init__(
        self,
        *,
        model_path: Path,
        noise_scales: list[float],
        noise_scale_ws: list[float],
        length_scales: list[float],
        slerp_weights: list[float],
        max_speakers: int | None,
        test_voice_fraction: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._model_path = model_path
        self._noise_scales = noise_scales
        self._noise_scale_ws = noise_scale_ws
        self._length_scales = length_scales
        self._slerp_weights = slerp_weights
        self._max_speakers = max_speakers
        self._test_voice_fraction = test_voice_fraction
        self._seed = seed

    @classmethod
    def from_config(cls, config: WakeWordConfig) -> PiperVitsBackend:
        return cls(
            model_path=config.piper_checkpoint_path,
            noise_scales=config.noise_scales,
            noise_scale_ws=config.noise_scale_ws,
            length_scales=config.length_scales,
            slerp_weights=config.slerp_weights,
            max_speakers=config.max_speakers,
            test_voice_fraction=config.test_voice_fraction,
            seed=config.seed,
        )

    def validate_artifacts(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"VITS model not found at {self._model_path}. "
                "Run setup with your config: livekit-wakeword setup --config <your.yaml>"
            )

    def _num_speakers(self) -> int:
        with open(self._model_path.with_suffix(".json"), encoding="utf-8") as f:
            n = int(json.load(f)["num_speakers"])
        return n if self._max_speakers is None else min(n, self._max_speakers)

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
        holdout_voices: bool = False,
        voice_group_size: int = 1,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        text = normalize_phrases_for_piper(phrases)
        train_speakers, test_speakers = split_voices(
            self._num_speakers(), self._test_voice_fraction, self._seed
        )
        generated = generate_samples(
            text=text,
            output_dir=output_dir,
            max_samples=n_samples,
            model=self._model_path,
            batch_size=batch_size,
            slerp_weights=self._slerp_weights,
            length_scales=self._length_scales,
            noise_scales=self._noise_scales,
            noise_scale_ws=self._noise_scale_ws,
            max_speakers=self._max_speakers,
            start_index=start_index,
            speaker_ids=test_speakers if holdout_voices else train_speakers,
            # Different streams for train and test, so test splits don't repeat pairs
            seed=self._seed * 2 + int(holdout_voices),
            voice_group_size=voice_group_size,
        )
        logger.info("Generated %d clips in %s", len(generated), output_dir)
        return generated
