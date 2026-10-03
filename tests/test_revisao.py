"""Revisao humana: aprovar o que o QA reprovou."""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- aprovacao humana de chunk reprovado --------------------------------------

def _store_com_chunk(tmp_path, estado="needs_review"):
    from genai.store.db import Store
    s = Store(tmp_path / "state.db")
    s.conn.execute("INSERT INTO chunks(chunk_id, chapter, idx, text, source, state) "
                   "VALUES ('ch01/00000-abc', 1, 0, 'Jiddu', 'Jiddu', ?)", (estado,))
    s.conn.commit()
    return s


def test_aprovar_destrava_a_montagem(tmp_path):
    """Um trecho de 2 s com nome próprio não pode travar um capítulo inteiro."""
    s = _store_com_chunk(tmp_path)
    assert s.approve("ch01/00000-abc") == 1
    assert s.needs_review() == []
    assert s.stats()["ok"] == 1


def test_aprovar_registra_que_foi_decisao_humana(tmp_path):
    s = _store_com_chunk(tmp_path)
    s.approve("ch01/00000-abc")
    erro = s.conn.execute("SELECT error FROM chunks").fetchone()[0]
    assert "humana" in erro


def test_nao_aprova_chunk_que_nao_esta_em_revisao(tmp_path):
    """Aprovar só vale para o que o QA reprovou — não é um atalho para a fila."""
    s = _store_com_chunk(tmp_path, estado="pending")
    assert s.approve("ch01/00000-abc") == 0
