"""hq/estado: fila da HQ no mesmo state.db do audiolivro, sem atrapalhar o Store."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.core.estado import Store  # noqa: E402
from genai.hq.estado import EstadoHQ  # noqa: E402


def test_sincronizar_preserva_o_que_nao_mudou_e_invalida_o_que_mudou(tmp_path):
    e = EstadoHQ(tmp_path / "state.db")
    assert e.sincronizar("quadro", {"1": "a", "2": "b", "3": "c"}) == (3, 0, 0)
    e.concluir("quadro", 1, tmp_path / "q1.png")
    e.concluir("quadro", 2, tmp_path / "q2.png", por="humano")

    # quadro 2 reescrito, quadro 3 saiu do roteiro, quadro 4 entrou
    assert e.sincronizar("quadro", {"1": "a", "2": "B", "4": "d"}) == (1, 1, 1)
    assert e.escolhido("quadro", 1) == tmp_path / "q1.png"
    assert e.item("quadro", 2)["state"] == "pending"
    assert e.escolhido("quadro", 2) is None
    assert e.item("quadro", 3) is None
    assert [r["alvo"] for r in e.itens("quadro")] == ["1", "2", "4"]


def test_sincronizar_e_por_tipo(tmp_path):
    e = EstadoHQ(tmp_path / "state.db")
    e.sincronizar("elenco", {"siddhartha": "x"})
    e.sincronizar("quadro", {"1": "a"})
    assert e.item("elenco", "siddhartha") is not None  # o quadro não apagou o elenco


def test_running_preso_volta_para_a_fila(tmp_path):
    e = EstadoHQ(tmp_path / "state.db")
    e.sincronizar("quadro", {"1": "a"})
    e.iniciar("quadro", 1)
    assert EstadoHQ(tmp_path / "state.db").reset_stale() == 1
    assert e.item("quadro", 1)["attempts"] == 1


def test_ordem_numerica_dos_quadros(tmp_path):
    e = EstadoHQ(tmp_path / "state.db")
    e.sincronizar("quadro", {str(i): "a" for i in (10, 2, 1)})
    assert [r["alvo"] for r in e.itens("quadro")] == ["1", "2", "10"]


def test_convive_com_o_store_do_audiolivro_no_mesmo_arquivo(tmp_path):
    db = tmp_path / "state.db"
    s = Store(db)
    e = EstadoHQ(db)
    e.sincronizar("quadro", {"1": "a"})
    assert s.stats()["total"] == 0
    tabelas = {r[0] for r in s.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"chunks", "runs", "hq_itens"} <= tabelas
    e.revisar("quadro", 1, "QA reprovou", qa={"cand-1.png": {"rostos": 0}})
    assert e.resumo()["quadro"]["needs_review"] == 1
