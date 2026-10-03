"""hq/quadros: estratégias de referência, rodadas do QA, retomada e escolha humana."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genai.hq import elenco, projeto as hqp, qa, quadros, roteiro, servico  # noqa: E402
from hq_falso import InvokeFalso  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "hq" / "roteiro-quatro-visoes.yaml"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    shutil.copy(FIXTURE, tmp_path / "roteiro.yaml")
    falso = InvokeFalso()
    monkeypatch.setattr(servico, "cliente", lambda: falso)
    r = roteiro.carregar(tmp_path)
    c = elenco.gerar_folhas(tmp_path, r, n=1)
    for pid in r.personagens:
        elenco.aprovar(tmp_path, r, pid, c[pid][0])
        elenco.aprovar(tmp_path, r, pid, tmp_path / "elenco" / pid / "aprovado.png")
    falso.grafos.clear()
    falso.uploads.clear()
    return tmp_path, r, falso


def _qa_aprova(monkeypatch, aprova=lambda p: True):
    def avaliar(cands, n_pers, ancora):
        return {p.name: {"ok": aprova(p), "estilo": 0.01 * len(p.name), "motivo": ""}
                for p in cands}
    monkeypatch.setattr(qa, "avaliar", avaliar)


def test_duas_passadas_corrige_o_da_direita_com_a_folha_dele(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch)
    quadros.gerar(p, r, n=2, ids=[3], estrategia="duas-passadas")
    passada1 = [g for g in falso.grafos if "Picture 1 is Siddhartha" in g[1]]
    passada2 = [g for g in falso.grafos if g[1].startswith("In Picture 1, redraw only Channa")]
    assert len(passada1) == 2 and len(passada2) == 2
    assert all(len(g[2]) == 2 for g in passada1 + passada2)
    assert "Siddhartha on the left, Channa on the right" in passada1[0][1]
    # a ref de Channa entra na segunda passada (a latente é o resultado da primeira)
    assert falso.uploads.count(elenco.ref(p, "channa")) == 2
    assert len(quadros.candidatos(p, 3)) == 2
    assert hqp.estado(p).item("quadro", 3)["state"] == "ok"


def test_composta_manda_uma_ref_lado_a_lado(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch)
    quadros.gerar(p, r, n=1, ids=[2], estrategia="composta")
    (g,) = falso.grafos
    assert len(g[2]) == 1 and "side by side" in g[1]
    assert falso.uploads[0].name.startswith("siddhartha+channa-")


def test_um_personagem_tem_uma_ref_e_o_tamanho_da_proporcao(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch)
    quadros.gerar(p, r, n=1, ids=[1])
    (g,) = falso.grafos
    assert (g[3], g[4]) == (1536, 672) and len(g[2]) == 1


def test_qa_reprova_tudo_gera_outra_rodada_e_depois_pede_humano(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch, aprova=lambda c: False)
    rel = quadros.gerar(p, r, n=2, ids=[4], rodadas=2)
    assert len(rel[4]) == 4  # 2 rodadas x 2 candidatos, avaliados juntos
    item = hqp.estado(p).item("quadro", 4)
    assert item["state"] == "needs_review" and item["attempts"] == 2
    with pytest.raises(ValueError, match=r"quadros sem escolha: \[1, 2, 3, 4, 5, 6\]"):
        quadros.escolhidos(p, r)
    cand = quadros.candidatos(p, 4)[0]
    assert quadros.escolher(p, r, 4, cand) == cand
    assert hqp.estado(p).item("quadro", 4)["por"] == "humano"


def test_qa_escolhe_o_de_maior_estilo_e_retomada_nao_regera(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch)
    quadros.gerar(p, r, n=2, ids=[6])
    n = len(falso.grafos)
    quadros.gerar(p, r, n=2, ids=[6])
    assert len(falso.grafos) == n  # já escolhido: nada a gerar


def test_cena_reescrita_volta_para_a_fila(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch)
    quadros.gerar(p, r, n=1, ids=[6])
    r.quadro(6).cena += " It is raining."
    quadros.gerar(p, r, n=1, ids=[6])
    assert len(list(quadros.pasta(p, 6).glob("cand-*.png"))) == 2
    assert hqp.estado(p).escolhido("quadro", 6).name in {c.name for c in quadros.candidatos(p, 6)}


def test_retoque_de_fora_e_copiado_para_a_pasta_do_quadro(proj, monkeypatch):
    p, r, _ = proj
    _qa_aprova(monkeypatch)
    quadros.gerar(p, r, n=1, ids=[6])
    retoque = p / "canvas.png"
    shutil.copy(quadros.candidatos(p, 6)[0], retoque)
    destino = quadros.escolher(p, r, 6, retoque)
    assert destino == quadros.pasta(p, 6) / "humano-canvas.png" and destino.exists()
    with pytest.raises(ValueError, match="não existe no roteiro"):
        quadros.escolher(p, r, 99, retoque)


def test_qa_real_reprova_quadro_vazio_e_falta_de_rosto(tmp_path):
    from PIL import Image
    vazio = tmp_path / "vazio.png"
    Image.new("RGB", (64, 64), (10, 10, 10)).save(vazio)
    notas = qa.avaliar([vazio], 1, None)
    assert notas["vazio.png"] == {"ok": False, "motivo": "vazio"}
    assert qa.melhor(notas) is None


def test_refazer_gera_candidatos_com_seeds_novas(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch)
    quadros.gerar(p, r, n=2, ids=[6])
    quadros.refazer(p, r, [6])
    quadros.gerar(p, r, n=2, ids=[6])
    assert len(quadros.candidatos(p, 6)) == 4
    assert len({g[5] for g in falso.grafos}) == 4  # quatro seeds distintas


def test_estampa_acrescenta_a_etapa_de_estilo_com_a_lora(proj, monkeypatch):
    p, r, falso = proj
    _qa_aprova(monkeypatch)
    shutil.copy(elenco.ref(p, "channa"), p / "gravura.png")
    r.estampa = "gravura.png"
    quadros.gerar(p, r, n=1, ids=[3], estrategia="duas-passadas")
    prompts = [g[1] for g in falso.grafos]
    assert len(prompts) == 3 and prompts[2].startswith("style transfer.")
    assert falso.grafos[2][6] == ((quadros.ESTILO_LORA, quadros.PESO_ESTILO),)
    assert falso.uploads[-1] == p / "gravura.png"
    # intermediários e<k>, final cand
    nomes = sorted(x.name.split("-")[0] for x in quadros.pasta(p, 3).glob("*.png"))
    assert nomes == ["cand", "e0", "e1"]
