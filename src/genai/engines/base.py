"""Contrato de motor TTS.

A arquitetura e agnostica ao motor por design (TDD 20.8): se o Chatterbox for
substituido, so esta camada muda.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class TTSEngine(ABC):
    name: str = "base"
    version: str = "0"
    sample_rate: int = 24000

    @abstractmethod
    def load(self) -> None:
        """Carrega o modelo na GPU. Chamado uma vez por processo."""

    @abstractmethod
    def synthesize(self, text: str, seed: int | None = None) -> np.ndarray:
        """Retorna audio mono float32 na taxa nativa do motor."""

    def set_voice(self, reference_wav: Path | None) -> None:
        """Congela a voz de referencia. Chamado uma vez por livro (TDD 7.1)."""
        raise NotImplementedError(f"{self.name} nao suporta clonagem de voz")
