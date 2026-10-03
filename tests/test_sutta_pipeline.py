"""Orquestração do comando `sutta`: o que ele recusa fazer duas vezes.

O comando existe para ser re-rodado — é assim que se retoma um lote interrompido.
Todas as etapas aguentam isso (o `run` retoma por projeto, o `export` reescreve),
menos uma: `videos.insert` não é idempotente, e um segundo upload cria OUTRO
vídeo no canal em vez de atualizar o primeiro.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.cli.fontes import ETAPAS
from genai.cli.publicar import _etapa_publish


def _projeto(tmp_path: Path) -> Path:
    (tmp_path / "output").mkdir(parents=True)
    (tmp_path / "output" / "video.mp4").write_bytes(b"nao e um mp4 de verdade")
    return tmp_path


def test_nao_republica_um_projeto_que_ja_tem_video(tmp_path, capsys):
    p = _projeto(tmp_path)
    (p / "output" / "publicado.json").write_text(
        json.dumps({"video_id": "abc123", "em": "2026-09-01T10:00:00"}),
        encoding="utf-8")

    # Sem a guarda isto tentaria autenticar e subir de novo.
    assert _etapa_publish(p, {"titulo": "T", "rights": {"status": "licenciado"}},
                          privacidade="private") == "abc123"
    assert "já publicado" in capsys.readouterr().out


def test_etapas_do_pipeline_estao_na_ordem_de_execucao():
    """`--ate` corta a fila por índice; a ordem é o contrato."""
    assert ETAPAS == ("texto", "script", "audio", "video", "publicar")
