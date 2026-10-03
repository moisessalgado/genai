"""Camada 1 do normalizador (narration/rules.py) -- regras deterministicas.

Não havia teste para este módulo. Cobre só o que foi tocado nesta sessão (a
regra de "E'" -> "É"), não o módulo inteiro -- as outras regras (abreviação,
datas, números) já funcionavam e não foram alteradas.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.narration.rules import normalize


def test_e_apostrofo_maiusculo_vira_e_acentuado():
    r = normalize("E' muito estranho isso que está acontecendo agora mesmo.")
    assert r.text == "É muito estranho isso que está acontecendo agora mesmo."
    assert ("E'", "É") in r.applied


def test_e_apostrofo_minusculo_vira_e_acentuado():
    r = normalize("Depois disso e' claro que tudo mudou para sempre.")
    assert r.text == "Depois disso é claro que tudo mudou para sempre."


def test_apostrofo_sem_espaco_depois_nao_e_tocado():
    """Só dispara quando "E'" é seguido de espaço (fim de contração) -- um
    apóstrofo usado para outra coisa (ex.: nome próprio) não deve ser mexido."""
    r = normalize("O sobrenome dela é E'Nunes, algo bem incomum por aqui.")
    assert "E'Nunes" in r.text


def test_texto_sem_apostrofo_fica_inalterado():
    r = normalize("Um parágrafo qualquer sem nenhuma contração antiga.")
    assert r.text == "Um parágrafo qualquer sem nenhuma contração antiga."
    assert r.applied == []
