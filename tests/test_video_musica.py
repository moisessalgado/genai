"""Trilha gerada e render de vídeo: o que dá para verificar sem olhar/ouvir."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audiofactory.audio.musica import SAMPLE_RATE, gerar_ambiente
from audiofactory.video.render import presets, renderizar

# Espectro medido na narração deste canal (2 min do sutta), em % de energia.
VOZ = {(300, 700): 49.5, (150, 300): 18.8, (700, 1500): 15.7, (80, 150): 9.8}


def energia(x, sr, lo, hi):
    X = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return X[(f >= lo) & (f < hi)].sum() / X.sum()


def test_trilha_sai_da_frente_da_voz():
    """A faixa onde a voz tem metade da energia (300-700 Hz) é justamente onde a
    trilha não pode estar."""
    a = gerar_ambiente(30.0)
    assert energia(a, SAMPLE_RATE, 300, 700) < 0.02


def test_trilha_tem_grave_e_brilho():
    """Só grave some em fone de celular; só agudo fica fino e nervoso."""
    a = gerar_ambiente(30.0)
    assert energia(a, SAMPLE_RATE, 0, 150) > 0.3
    assert energia(a, SAMPLE_RATE, 700, 12000) > 0.05


def test_trilha_nao_clipa():
    a = gerar_ambiente(30.0)
    assert 0.2 < np.abs(a).max() <= 1.0


def test_trilha_tem_a_duracao_pedida():
    a = gerar_ambiente(12.0)
    assert abs(len(a) / SAMPLE_RATE - 12.0) < 0.05


def test_trilha_e_deterministica():
    """Mesma seed, mesma trilha: um capítulo regerado não pode mudar de fundo."""
    assert np.array_equal(gerar_ambiente(5.0, seed=3), gerar_ambiente(5.0, seed=3))


def test_trilha_nao_tem_ataque_no_inicio():
    """Envelope de cosseno: entrar com nível cheio seria um evento sonoro."""
    a = gerar_ambiente(30.0)
    ini = np.abs(a[:SAMPLE_RATE]).max()
    meio = np.abs(a[SAMPLE_RATE * 10:SAMPLE_RATE * 11]).max()
    assert ini < meio * 0.7


def test_preset_desconhecido_e_recusado(tmp_path):
    with pytest.raises(ValueError, match="preset desconhecido"):
        renderizar(tmp_path / "x.wav", tmp_path / "y.mp4", preset="fractal")


def test_presets_declarados():
    assert set(presets()) == {"slides", "sincronizado", "ondas", "espectro",
                              "estatico", "gradiente"}
