"""Pydantic configuration models with YAML loading."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from .defaults import piper as piper_defaults
from .defaults import voxcpm as voxcpm_defaults

_logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    dnn = "dnn"
    rnn = "rnn"
    conv_attention = "conv_attention"


class ModelSize(StrEnum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"


class TtsBackend(StrEnum):
    """Synthetic speech engine for the generate stage."""

    piper_vits = "piper_vits"
    voxcpm = "voxcpm"


class ExportFormat(StrEnum):
    """Artifact format produced by the export stage."""

    onnx = "onnx"
    tflite = "tflite"


# Preset mapping: size -> (layer_dim, n_blocks)
MODEL_SIZE_PRESETS: dict[ModelSize, tuple[int, int]] = {
    ModelSize.tiny: (16, 1),
    ModelSize.small: (32, 1),
    ModelSize.medium: (128, 2),
    ModelSize.large: (256, 3),
}


class AugmentationConfig(BaseModel):
    clip_duration: float = 2.0
    batch_size: int = 16
    rounds: int = 1
    background_paths: list[str] = Field(default_factory=lambda: ["./data/backgrounds"])
    rir_paths: list[str] = Field(default_factory=lambda: ["./data/rirs"])
    end_jitter: float = Field(
        default=0.3,
        ge=0.0,
        description="Max seconds between the end of a speech clip and the end of the window",
    )
    context_speech_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Chance that other speech (the context_speech split) fills the padding "
        "before a positive or adversarial clip; needs n_context_samples > 0",
    )
    near_miss_context_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Chance that an adversarial clip precedes a positive clip "
        "(e.g. 'hey jack. hey computer'); checked before context_speech_probability",
    )
    context_gap: tuple[float, float] = Field(
        default=(0.0, 0.4),
        description="Range of seconds of gap between the context clip and the phrase",
    )
    split_phrase_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Chance that a positive clip is replaced by the two halves of the phrase "
        "(positive_*_parts splits) joined with a pause; needs n_split_phrase_samples > 0",
    )
    split_phrase_gap: tuple[float, float] = Field(
        default=(0.1, 0.5),
        description="Range of seconds of pause between the halves of a split phrase",
    )


class ModelConfig(BaseModel):
    model_type: ModelType = ModelType.conv_attention
    model_size: ModelSize = ModelSize.small

    @property
    def layer_dim(self) -> int:
        return MODEL_SIZE_PRESETS[self.model_size][0]

    @property
    def n_blocks(self) -> int:
        return MODEL_SIZE_PRESETS[self.model_size][1]


class PiperTtsConfig(BaseModel):
    """Piper VITS artifact layout under ``data_dir`` (when ``tts_backend`` is ``piper_vits``)."""

    checkpoint_relpath: str = Field(
        default=piper_defaults.CHECKPOINT_RELPATH,
        description="Path to the VITS state_dict .pt file, relative to data_dir",
    )


class VoxCpmTtsConfig(BaseModel):
    """VoxCPM2 voice-design TTS (when ``tts_backend`` is ``voxcpm``).

    Weights live under ``data_dir`` after ``setup --config`` (HF snapshot) or
    ``local_model_path``. Diversification defaults are intentionally large
    (persona × cfg × diffusion steps).
    """

    model_id: str = Field(
        default=voxcpm_defaults.MODEL_ID,
        description="Hugging Face repo id used by setup for snapshot_download",
    )
    model_cache_relpath: str = Field(
        default=voxcpm_defaults.MODEL_CACHE_RELPATH,
        description="Directory under data_dir where setup stores the model snapshot",
    )
    local_model_path: str | None = Field(
        default=None,
        description="If set, load weights from this path (relative to data_dir or absolute); "
        "setup skips HF download if directory exists and is non-empty",
    )
    load_denoiser: bool = False
    voice_design_prompts: list[str] = Field(default_factory=voxcpm_defaults.voice_design_prompts)
    cfg_values: list[float] = Field(default_factory=lambda: list(voxcpm_defaults.CFG_VALUES))
    inference_timesteps_list: list[int] = Field(
        default_factory=lambda: list(voxcpm_defaults.INFERENCE_TIMESTEPS),
    )


class StreamingEvalSet(BaseModel):
    """A labelled set of clips (or long recordings) replayed as one audio stream."""

    name: str
    positive: bool
    # WAV files, or directories searched recursively for *.wav
    paths: list[str]
    # Precede each clip with a speech clip that ends 0-400 ms before it
    speech_before: bool = False


class StreamingEvalConfig(BaseModel):
    """Streaming evaluation (``eval --streaming``) and stream-level validation settings."""

    # Detector behaviour, shared with training validation
    debounce_seconds: float = 2.0
    min_consecutive: int = 1
    # Threshold selection: lowest miss rate with at most this many false accepts per hour
    target_fa_per_hour: float = 0.5
    # A detection counts as a hit up to this long after the clip ends
    detection_window_seconds: float = 1.0
    # Silence between consecutive clips in a stream
    gap_seconds: float = 3.0
    max_clips_per_set: int | None = 1000
    # Derive positive_silence / positive_speech / near_miss sets from the test splits
    default_sets: bool = True
    sets: list[StreamingEvalSet] = Field(default_factory=list)
    # Speech clips used for speech_before (default: negative_test clips)
    speech_paths: list[str] = Field(default_factory=list)
    # Optional noise bed mixed under every stream at snr_db (e.g. DEMAND)
    background_paths: list[str] = Field(default_factory=list)
    snr_db: float = 10.0
    seed: int = 0


class WakeWordConfig(BaseModel):
    """Top-level config for a wake word model."""

    model_name: str
    target_phrases: list[str]

    # Data generation
    n_samples: int = 10000
    n_samples_val: int = 2000
    n_background_samples: int = 200
    n_background_samples_val: int = 40
    tts_batch_size: int = 50
    tts_backend: TtsBackend = TtsBackend.piper_vits
    piper_tts: PiperTtsConfig = Field(default_factory=PiperTtsConfig)
    voxcpm_tts: VoxCpmTtsConfig = Field(default_factory=VoxCpmTtsConfig)
    custom_negative_phrases: list[str] = Field(default_factory=list)

    # Negative phrase mix and voices
    seed: int = Field(default=0, description="Seed for phrase lists, voices and splits")
    custom_negative_share: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Share of negative clips spent on custom_negative_phrases (when set)",
    )
    word_swap_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of negative clips that swap one word of the wake phrase for a "
        "common word or name (e.g. 'hey there', 'hey siri')",
    )
    word_swap_words: list[str] | None = Field(
        default=None,
        description="Words for word_swap_share (None = built-in list of common words/names)",
    )
    negative_test_holdout: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Share of generated negative phrases kept out of negative_train and used "
        "only for negative_test",
    )
    test_voice_fraction: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Share of TTS voices held out from training and used for the test splits",
    )
    n_context_samples: int = Field(
        default=0,
        ge=0,
        description="Clips of generic speech (context_speech split) used as speech before "
        "the phrase during augmentation",
    )
    n_split_phrase_samples: int = Field(
        default=0,
        ge=0,
        description="Positive phrases synthesized as two halves in one voice "
        "(positive_train_parts), joined with a pause during augmentation",
    )
    n_split_phrase_samples_val: int = Field(default=0, ge=0)

    # TTS parameters (Piper VITS + SLERP speaker blending)
    noise_scales: list[float] = Field(default_factory=lambda: [0.98])
    noise_scale_ws: list[float] = Field(default_factory=lambda: [0.98])
    length_scales: list[float] = Field(default_factory=lambda: [0.75, 1.0, 1.25])
    slerp_weights: list[float] = Field(default_factory=lambda: [0.2, 0.35, 0.5, 0.65, 0.8])
    max_speakers: int | None = None

    # Paths
    data_dir: Annotated[str, Field(description="Root data directory")] = "./data"
    output_dir: str = "./output"

    # Export
    output_format: ExportFormat = ExportFormat.onnx

    # Augmentation
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    # Model
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Training
    steps: int = 50000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    max_negative_weight: float = 1500.0
    target_fp_per_hour: float = 0.2
    streaming_eval: StreamingEvalConfig = Field(default_factory=StreamingEvalConfig)
    batch_n_per_class: dict[str, int] = Field(
        default_factory=lambda: {
            "positive": 50,
            "adversarial_negative": 50,
            "ACAV100M_sample": 1024,
            "background_noise": 50,
        }
    )

    @model_validator(mode="after")
    def _check_negative_shares(self) -> Self:
        if self.custom_negative_share + self.word_swap_share > 1.0:
            raise ValueError("custom_negative_share + word_swap_share must not exceed 1.0")
        return self

    @model_validator(mode="after")
    def _warn_unknown_batch_keys(self) -> Self:
        known_keys = {"positive", "adversarial_negative", "ACAV100M_sample", "background_noise"}
        unknown = set(self.batch_n_per_class) - known_keys
        if unknown:
            _logger.warning(
                f"Unrecognized keys in batch_n_per_class: {unknown}. "
                f"Known keys: {sorted(known_keys)}"
            )
        return self

    @property
    def model_output_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def piper_checkpoint_path(self) -> Path:
        """Absolute path to the Piper VITS .pt checkpoint (JSON sits alongside)."""
        return (self.data_path / Path(self.piper_tts.checkpoint_relpath)).resolve()

    @property
    def voxcpm_local_model_path(self) -> Path:
        """Directory containing VoxCPM weights (snapshot or manual copy)."""
        raw = self.voxcpm_tts.local_model_path
        if raw:
            p = Path(raw)
            return p.resolve() if p.is_absolute() else (self.data_path / p).resolve()
        return (self.data_path / Path(self.voxcpm_tts.model_cache_relpath)).resolve()


def load_config(path: str | Path) -> WakeWordConfig:
    """Load a WakeWordConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return WakeWordConfig(**data)
