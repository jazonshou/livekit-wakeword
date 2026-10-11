"""Evaluation module for wake word models."""

from .evaluate import run_eval
from .streaming import run_streaming_eval

__all__ = ["run_eval", "run_streaming_eval"]
