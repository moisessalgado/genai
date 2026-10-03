"""Atribuição de personagem à fala citada (narration/elenco.py).

Contrato a testar: o LLM só pode escolher um nome de um conjunto FECHADO (ou
"indeterminado"), a validação é por igualdade exata, e qualquer coisa fora
disso -- resposta fora da lista, LLM indisponível -- mantém o papel genérico
`citacao`. Nunca deixa a fala sem papel algum.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib import error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.audiolivro.narration import elenco

PERSONAGENS = ["narizinho", "emilia"]


def test_atribui_quando_resposta_bate_com_a_lista(monkeypatch):
    monkeypatch.setattr(elenco, "_perguntar", lambda *a, **kw: "emilia")
    assert elenco.atribuir("— Que ideia!", "disse Emília, rindo.", "",
                          PERSONAGENS) == "emilia"


def test_resposta_indeterminado_devolve_none(monkeypatch):
    monkeypatch.setattr(elenco, "_perguntar", lambda *a, **kw: "indeterminado")
    assert elenco.atribuir("— Quem sabe?", "", "", PERSONAGENS) is None


def test_resposta_fora_da_lista_devolve_none(monkeypatch):
    """O LLM alucinar um nome que não está no elenco não pode virar um papel
    novo do nada -- só os nomes passados em `personagens` são aceitos."""
    monkeypatch.setattr(elenco, "_perguntar", lambda *a, **kw: "pinoquio")
    assert elenco.atribuir("— Oi!", "", "", PERSONAGENS) is None


def test_resposta_com_acento_ou_maiuscula_ainda_casa(monkeypatch):
    monkeypatch.setattr(elenco, "_perguntar", lambda *a, **kw: "Emília")
    assert elenco.atribuir("— Oi!", "", "", PERSONAGENS) == "emilia"


def test_llm_fora_do_ar_devolve_none(monkeypatch):
    def _falha(*a, **kw):
        raise error.URLError("conexão recusada")
    monkeypatch.setattr(elenco, "_perguntar", _falha)
    assert elenco.atribuir("— Alô?", "", "", PERSONAGENS) is None


def test_sem_personagens_devolve_none_sem_chamar_llm(monkeypatch):
    chamou = []
    monkeypatch.setattr(elenco, "_perguntar", lambda *a, **kw: chamou.append(1) or "x")
    assert elenco.atribuir("— Oi!", "", "", []) is None
    assert not chamou


# --- aplicar_elenco (nível de parágrafo, com vizinhos) ------------------------

def test_aplicar_elenco_promove_citacao_com_contexto_inequivoco(monkeypatch):
    monkeypatch.setattr(elenco, "atribuir",
                        lambda fala, antes, depois, nomes, **kw: "emilia")
    partes = [("narrador", "disse Emília,"), ("citacao", "Que ideia boa!")]
    saida = elenco.aplicar_elenco(partes, {"emilia": "voice-x", "_generico": "voice-y"})
    assert saida == [("narrador", "disse Emília,"), ("personagem_emilia", "Que ideia boa!")]


def test_aplicar_elenco_mantem_citacao_quando_indeterminado(monkeypatch):
    monkeypatch.setattr(elenco, "atribuir",
                        lambda fala, antes, depois, nomes, **kw: None)
    partes = [("citacao", "— E agora?")]
    saida = elenco.aplicar_elenco(partes, {"emilia": "voice-x", "_generico": "voice-y"})
    assert saida == [("citacao", "— E agora?")]


def test_aplicar_elenco_nao_toca_papel_narrador():
    partes = [("narrador", "Era uma vez.")]
    saida = elenco.aplicar_elenco(partes, {"emilia": "voice-x"})
    assert saida == partes


def test_aplicar_elenco_sem_elenco_configurado_e_no_op():
    partes = [("citacao", "— Oi!")]
    assert elenco.aplicar_elenco(partes, {}) == partes
    assert elenco.aplicar_elenco(partes, {"_generico": "voice-y"}) == partes


def test_aplicar_elenco_usa_cache_para_nao_repetir_chamada(monkeypatch):
    chamadas = []

    def _atribuir(fala, antes, depois, nomes, **kw):
        chamadas.append(fala)
        return "emilia"

    monkeypatch.setattr(elenco, "atribuir", _atribuir)
    partes = [("citacao", "— Oi!")]
    cache: dict = {}
    elenco.aplicar_elenco(partes, {"emilia": "voice-x"}, cache=cache)
    elenco.aplicar_elenco(partes, {"emilia": "voice-x"}, cache=cache)
    assert len(chamadas) == 1
