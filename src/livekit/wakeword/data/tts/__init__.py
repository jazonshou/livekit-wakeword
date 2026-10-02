"""Pluggable TTS backends for synthetic data generation."""

from __future__ import annotations

from .backends import SpeechSynthesizer, get_tts_backend
from .voices import split_voices

__all__ = ["SpeechSynthesizer", "get_tts_backend", "split_voices"]
