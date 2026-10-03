"""hq/roteiro_llm: plano + páginas, com a recusa do pydantic voltando ao LLM."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.hq import roteiro_llm as rl  # noqa: E402
from genai.hq.roteiro import Quadro  # noqa: E402

PLANO = {"titulo": "O Jabuti e a Onça", "epoca": "Amazon forest",
         "personagens": [{"id": "jabuti", "nome": "Jabuti", "ficha": "a small tortoise"},
                         {"id": "onca", "nome": "Onça", "ficha": "a big jaguar"},
                         {"id": "macaco", "nome": "Macaco", "ficha": "a monkey"}],
         "paginas": [{"resumo": "a onça encontra o jabuti", "quadros": 3}]}


def _q(prop, pers, textos):
    return {"proporcao": prop, "personagens": pers, "cena": "a forest", "textos": textos}


PAGINA_RUIM = {"quadros": [
    _q("panoramico", ["jabuti"], [{"tipo": "recordatorio", "texto": "Na mata."}]),
    # a Onça fala sem estar no quadro: o schema recusa
    _q("largo", ["jabuti"], [{"tipo": "fala", "quem": "onca", "texto": "Vou te comer!"}]),
    _q("alto", ["onca"], [])]}
PAGINA_BOA = {"quadros": [
    _q("panoramico", ["jabuti"], [{"tipo": "recordatorio", "texto": "Na mata."}]),
    # nome em vez de id: traduzido
    _q("largo", ["Jabuti", "Onça"], [{"tipo": "fala", "quem": "Onça", "texto": "Vou te comer!"}]),
    _q("alto", ["onca"], [])]}


class _LLM:
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.prompts = []

    def __call__(self, prompt, schema, **kw):
        self.prompts.append(prompt)
        assert "$ref" not in str(schema)
        return self.respostas.pop(0)


def test_pagina_recusada_volta_com_o_erro_e_converge():
    fake = _LLM([PLANO, PAGINA_RUIM, PAGINA_BOA])
    r = rl.gerar("texto da fábula", slug="jabuti", gerar_json=fake)
    assert "REJECTED" in fake.prompts[2] and "'onca' fala mas não está" in fake.prompts[2]
    assert [q.id for q in r.quadros] == [1, 2, 3]
    assert r.quadro(2).personagens == ["jabuti", "onca"]
    assert r.quadro(2).textos[0].quem == "onca"
    assert r.paginas[0].tiras == [[1], [2, 3]]
    assert set(r.personagens) == {"jabuti", "onca"}  # o macaco não aparece: sem folha
    assert r.estilo == rl.ESTILO_PADRAO


def test_desiste_depois_das_tentativas():
    fake = _LLM([PLANO] + [PAGINA_RUIM] * rl.TENTATIVAS)
    with pytest.raises(ValueError, match="página 1"):
        rl.gerar("texto", slug="x", gerar_json=fake)


def test_plano_com_paginas_a_menos_e_pedido_de_novo():
    dois = {**PLANO, "paginas": PLANO["paginas"] * 2}
    pag = {"quadros": PAGINA_BOA["quadros"]}
    fake = _LLM([PLANO, dois, pag, pag])
    r = rl.gerar("texto", slug="x", paginas=2, gerar_json=fake)
    assert [q.id for q in r.quadros] == [1, 2, 3, 4, 5, 6]
    assert r.paginas[1].quadros == [4, 5, 6]


def test_fonte_longa_demais_e_recusada_antes_do_llm():
    with pytest.raises(ValueError, match="divida em trechos"):
        rl.gerar("x" * (rl.MAX_CHARS_FONTE + 1), slug="x", gerar_json=_LLM([]))


def _quadros(*props):
    return [Quadro(id=i, proporcao=p, cena="c") for i, p in enumerate(props, 1)]


def test_montar_tiras_pela_proporcao():
    assert rl.montar_tiras(_quadros("panoramico", "quadrado", "quadrado",
                                    "alto", "alto", "alto")) == [[1], [2, 3], [4, 5, 6]]
    # largo (1,46) + largo cabe; o terceiro abre tira nova
    assert rl.montar_tiras(_quadros("largo", "largo", "largo")) == [[1, 2], [3]]
    # panorâmico no meio fecha a tira anterior e vai sozinho
    assert rl.montar_tiras(_quadros("alto", "panoramico", "alto")) == [[1], [2], [3]]


def test_schema_da_pagina_restringe_ids_ao_elenco():
    s = rl.schema_pagina(["jabuti", "onca"])
    q = s["properties"]["quadros"]["items"]["properties"]
    assert q["personagens"]["items"]["enum"] == ["jabuti", "onca"]
    assert {"type": "null"} in q["textos"]["items"]["properties"]["quem"]["anyOf"]


def test_erro_de_digitacao_do_llm_e_consertado_pela_fonte():
    fonte = "Os adivinhos tinham dito. Viram um homem de cabelos brancos."
    assert (rl.corrigir_pela_fonte("Os adividos viram cabelos branc0s, e mais nada.", fonte)
            == "Os adivinhos viram cabelos brancos, e mais nada.")
    # palavra nova de verdade fica
    assert rl.corrigir_pela_fonte("Profetizaram!", fonte) == "Profetizaram!"


def _fora(*a, **kw):
    from genai.core.claude_cli import ClaudeIndisponivel
    raise ClaudeIndisponivel("limite de uso")


def test_claude_fora_cai_no_local_de_vez_e_avisa(capsys):
    local = _LLM([PLANO, PAGINA_BOA])
    chamadas = []

    def claude(*a, **kw):
        chamadas.append(1)
        _fora()

    g = rl._ComReserva(claude, local, "genai-local")
    r = rl.gerar("texto", slug="x", gerar_json=g)
    assert len(r.quadros) == 3
    assert chamadas == [1]  # não insiste no Claude depois da 1ª falha
    assert "seguindo no LLM local (genai-local)" in capsys.readouterr().err


def test_etapa_que_nao_converge_no_claude_e_refeita_no_local(capsys):
    claude = _LLM([PLANO] + [PAGINA_RUIM] * rl.TENTATIVAS)
    local = _LLM([PAGINA_BOA])
    g = rl._ComReserva(claude, local, "genai-local")
    r = rl.gerar("texto", slug="x", gerar_json=g)
    assert len(r.quadros) == 3 and g.na_reserva
    assert "não convergiu" in capsys.readouterr().err


def test_backend_litellm_nao_chama_o_claude():
    assert not isinstance(rl.backend("litellm"), rl._ComReserva)
    assert isinstance(rl.backend("claude"), rl._ComReserva)
    with pytest.raises(ValueError, match="roteiro_backend"):
        rl.backend("gpt")
