"""Modernização de ortografia pré-1943 (narration/ortografia.py).

Contrato a testar: o LLM só corrige UMA palavra isolada e a resposta só é
aceita se sobreviver ao validador (mesma letra inicial, semelhança alta com
o original) -- alucinação, resposta multi-palavra ou LLM fora do ar nunca
vira entrada de léxico.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib import error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.audiolivro.narration import ortografia


def test_validar_aceita_forma_moderna_plausivel():
    ok, _ = ortografia.validar("annos", "anos")
    assert ok


def test_validar_aceita_correcao_so_de_acento():
    ok, _ = ortografia.validar("oculos", "óculos")
    assert ok


def test_validar_rejeita_palavra_identica():
    ok, motivo = ortografia.validar("casa", "casa")
    assert not ok
    assert "identica" in motivo


def test_validar_rejeita_resposta_multi_palavra():
    ok, _ = ortografia.validar("vae", "vai mesmo")
    assert not ok


def test_validar_rejeita_letra_inicial_diferente():
    ok, _ = ortografia.validar("vae", "sai")
    assert not ok


def test_validar_rejeita_palavra_muito_diferente():
    """Alucinação: o LLM troca a palavra por outra coisa qualquer, com a
    mesma letra inicial por coincidência."""
    ok, _ = ortografia.validar("freguezia", "floresta")
    assert not ok


def test_modernizar_devolve_none_quando_llm_fora_do_ar(monkeypatch):
    def _falha(*a, **kw):
        raise error.URLError("conexão recusada")
    monkeypatch.setattr(ortografia, "_perguntar", _falha)
    assert ortografia.modernizar("annos") is None


def test_modernizar_devolve_none_quando_resposta_nao_passa_no_validador(monkeypatch):
    monkeypatch.setattr(ortografia, "_perguntar", lambda *a, **kw: "outra coisa")
    assert ortografia.modernizar("annos") is None


def test_modernizar_aceita_resposta_valida(monkeypatch):
    monkeypatch.setattr(ortografia, "_perguntar", lambda *a, **kw: "anos")
    assert ortografia.modernizar("annos") == "anos"


def test_extrair_candidatas_ignora_palavras_curtas():
    candidatas = ortografia.extrair_candidatas("Ha um dia e uma noite.")
    assert "ha" not in candidatas
    assert "um" not in candidatas
    assert "dia" in candidatas
    assert "uma" in candidatas


def test_extrair_candidatas_so_a_primeira_ocorrencia():
    candidatas = ortografia.extrair_candidatas("Annos depois, mais annos.")
    assert list(candidatas).count("annos") == 1


def test_modernizar_lote_aceita_so_palavras_pedidas(monkeypatch):
    """O LLM as vezes "corrige" uma palavra que nao estava na lista enviada --
    isso nao pode virar entrada de lexico do nada."""
    monkeypatch.setattr(ortografia, "_perguntar_lote",
                        lambda *a, **kw: '{"annos": "anos", "intruso": "algo"}')
    achadas = ortografia.modernizar_lote(["annos", "casa"])
    assert achadas == {"annos": "anos"}


def test_modernizar_lote_ignora_palavras_ja_modernas(monkeypatch):
    monkeypatch.setattr(ortografia, "_perguntar_lote", lambda *a, **kw: "{}")
    assert ortografia.modernizar_lote(["casa", "menina"]) == {}


def test_modernizar_lote_extrai_json_mesmo_com_texto_em_volta(monkeypatch):
    monkeypatch.setattr(ortografia, "_perguntar_lote",
                        lambda *a, **kw: 'Claro, aqui esta:\n{"vae": "vai"}\nEspero ter ajudado.')
    assert ortografia.modernizar_lote(["vae"]) == {"vae": "vai"}


def test_modernizar_lote_devolve_vazio_quando_llm_fora_do_ar(monkeypatch):
    def _falha(*a, **kw):
        raise error.URLError("conexão recusada")
    monkeypatch.setattr(ortografia, "_perguntar_lote", _falha)
    assert ortografia.modernizar_lote(["annos"]) == {}


def test_modernizar_lote_devolve_vazio_quando_resposta_nao_e_json(monkeypatch):
    monkeypatch.setattr(ortografia, "_perguntar_lote", lambda *a, **kw: "desculpe, nao sei")
    assert ortografia.modernizar_lote(["annos"]) == {}


def test_descobrir_devolve_so_palavras_aceitas(monkeypatch):
    monkeypatch.setattr(ortografia, "modernizar_lote",
                        lambda lote, **kw: {"annos": "anos"})
    achadas = ortografia.descobrir("Muitos annos naquela casa.")
    assert achadas == {"annos": "anos"}


def test_descobrir_usa_cache_para_nao_repetir_chamada(monkeypatch):
    chamadas = []

    def _fake(lote, **kw):
        chamadas.extend(lote)
        return {"annos": "anos"} if "annos" in lote else {}

    monkeypatch.setattr(ortografia, "modernizar_lote", _fake)
    cache: dict = {}
    ortografia.descobrir("Muitos annos, tantos annos.", cache=cache)
    ortografia.descobrir("Outra vez annos.", cache=cache)
    assert chamadas.count("annos") == 1
