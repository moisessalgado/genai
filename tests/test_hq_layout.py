"""hq/layout: os retângulos cobrem a área útil sem sobrepor; a página sai no formato."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.hq import layout  # noqa: E402
from genai.hq.roteiro import Roteiro  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "hq" / "roteiro-quatro-visoes.yaml"


def _r() -> Roteiro:
    return Roteiro.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))


def test_retangulos_preenchem_a_area_util_com_calhas():
    r = _r()
    f = r.formato
    rects = layout.retangulos(r, r.paginas[0])
    assert set(rects) == {1, 2, 3, 4, 5, 6}
    # cada tira termina na margem direita e a última na margem de baixo
    for tira in r.paginas[0].tiras:
        x, _, w, _ = rects[tira[-1]]
        assert x + w == f.largura - f.margem
    x, y, w, h = rects[6]
    assert y + h == f.altura - f.margem
    # calha entre quadros vizinhos
    assert rects[3][0] - (rects[2][0] + rects[2][2]) == f.calha
    assert rects[2][1] - (rects[1][1] + rects[1][3]) == f.calha
    # mesma tira, mesma altura; proporção perto da gerada (o panorâmico 1536x672)
    assert rects[4][3] == rects[5][3] == rects[6][3]
    assert abs(rects[1][2] / rects[1][3] - 1536 / 672) < 0.35


def test_pagina_montada_tem_o_tamanho_do_formato(tmp_path, monkeypatch):
    r = _r()
    artes = {}
    for q in r.quadros:
        p = tmp_path / f"{q.id}.png"
        Image.new("RGB", q.tamanho, (30 * q.id, 100, 150)).save(p)
        artes[q.id] = p
    monkeypatch.setattr(layout.rostos, "detectar", lambda arte: [])
    (pag,) = layout.paginas(tmp_path, r, artes)
    with Image.open(pag) as im:
        assert im.size == (r.formato.largura, r.formato.altura)
        assert round(im.info["dpi"][0]) == 300
