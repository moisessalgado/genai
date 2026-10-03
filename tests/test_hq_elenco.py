"""hq/elenco: folha -> aprovação -> limpeza -> ref, com o estado no state.db."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genai.hq import elenco, roteiro, servico  # noqa: E402
from hq_falso import InvokeFalso  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "hq" / "roteiro-quatro-visoes.yaml"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    shutil.copy(FIXTURE, tmp_path / "roteiro.yaml")
    falso = InvokeFalso()
    monkeypatch.setattr(servico, "cliente", lambda: falso)
    return tmp_path, falso


def test_fluxo_completo_ate_as_refs(proj):
    p, falso = proj
    r = roteiro.carregar(p)
    cands = elenco.gerar_folhas(p, r, n=3)
    assert set(cands) == {"siddhartha", "channa"} and all(len(v) == 3 for v in cands.values())
    assert all(g[0] == "flux" and (g[2], g[3]) == (832, 1216) for g in falso.grafos)
    with pytest.raises(ValueError, match="sem ref aprovada"):
        elenco.refs_prontas(p, r)

    for pid in r.personagens:
        assert elenco.aprovar(p, r, pid, cands[pid][1])[0] == "folha"
    # aprovadas não geram folha de novo
    assert elenco.gerar_folhas(p, r, n=3) == {}

    limpos = elenco.limpar(p, r, n=2)
    assert all(len(v) == 2 for v in limpos.values())
    assert falso.apagados == ["up-1.png", "up-2.png"]  # refs não ficam na galeria
    for pid in r.personagens:
        assert elenco.aprovar(p, r, pid, limpos[pid][0]) == ("ref", elenco.ref(p, pid))
    assert set(elenco.refs_prontas(p, r)) == {"siddhartha", "channa"}


def test_retomada_nao_regera_o_que_ja_esta_em_disco(proj):
    p, falso = proj
    r = roteiro.carregar(p)
    elenco.gerar_folhas(p, r, n=2)
    n = len(falso.grafos)
    elenco.gerar_folhas(p, r, n=3)  # pede um a mais: só o terceiro é gerado
    assert len(falso.grafos) == n + 2


def test_ficha_reescrita_invalida_folha_e_ref(proj):
    p, _ = proj
    r = roteiro.carregar(p)
    c = elenco.gerar_folhas(p, r, n=1)
    elenco.aprovar(p, r, "channa", c["channa"][0])
    elenco.aprovar(p, r, "channa", p / "elenco" / "channa" / "aprovado.png")  # já limpa
    r.personagens["channa"].ficha += " He carries a whip."
    elenco.sincronizar(p, r)
    from genai.hq.projeto import estado
    e = estado(p)
    assert e.item("elenco", "channa")["state"] == "pending"
    assert e.escolhido("limpeza", "channa") is None


def test_ref_exige_folha_aprovada(proj):
    p, _ = proj
    r = roteiro.carregar(p)
    outra = p / "retoque.png"
    shutil.copy(FIXTURE, outra)
    with pytest.raises(ValueError, match="aprove primeiro uma folha"):
        elenco.aprovar(p, r, "channa", outra)
    with pytest.raises(ValueError, match="fora do roteiro"):
        elenco.aprovar(p, r, "yasodhara", outra)
