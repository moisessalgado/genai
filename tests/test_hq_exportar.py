"""hq/exportar: PDF multipágina, CBZ com ComicInfo e webtoon fatiado nos respiros."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.hq import exportar  # noqa: E402
from genai.hq.roteiro import Roteiro  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "hq" / "roteiro-quatro-visoes.yaml"


def _r() -> Roteiro:
    return Roteiro.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))


def _paginas(tmp_path, n=2):
    out = []
    for i in range(n):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (620, 877), (40 * i, 90, 120)).save(p)
        out.append(p)
    return out


def test_pdf_tem_uma_pagina_por_imagem(tmp_path):
    destino = exportar.pdf(_paginas(tmp_path, 3), tmp_path / "out" / "hq.pdf", dpi=75)
    import fitz  # pymupdf, já dependência do ingest
    with fitz.open(destino) as doc:
        assert doc.page_count == 3
        assert round(doc[0].rect.width) == round(620 / 75 * 72)  # tamanho físico pelo dpi


def test_cbz_tem_comicinfo_e_paginas_em_ordem(tmp_path):
    destino = exportar.cbz(_paginas(tmp_path, 2), tmp_path / "hq.cbz", _r(), autoria="Tradição")
    with zipfile.ZipFile(destino) as z:
        assert z.namelist() == ["ComicInfo.xml", "001.jpg", "002.jpg"]
        info = z.read("ComicInfo.xml").decode()
    assert "<Title>As Quatro Visões</Title>" in info and "<PageCount>2</PageCount>" in info
    assert "<LanguageISO>pt</LanguageISO>" in info and "<Writer>Tradição</Writer>" in info
    assert not (tmp_path / "hq.cbz.parcial").exists()


def test_cortes_caem_nos_respiros():
    # quadros de 500 px com respiro de 80: o corte nunca corta um quadro ao meio
    pos = [(80, 580), (660, 1160), (1240, 1740), (1820, 2320), (2400, 2900)]
    cortes = exportar._cortes(pos, 2980, 1280)
    assert cortes == [1200, 2360]
    assert all(not (y0 < c < y1) for c in cortes for y0, y1 in pos)


def test_quadro_maior_que_a_fatia_e_cortado_no_limite():
    assert exportar._cortes([(80, 3000)], 3080, 1280) == [1280, 2560]


def test_webtoon_na_largura_padrao_e_fatias_limitadas(tmp_path, monkeypatch):
    r = _r()
    artes = {}
    for q in r.quadros:
        p = tmp_path / f"{q.id}.png"
        Image.new("RGB", q.tamanho, (30 * q.id, 100, 150)).save(p)
        artes[q.id] = p
    monkeypatch.setattr(exportar.rostos, "detectar", lambda arte: [])
    fatias = exportar.webtoon(r, artes, tmp_path / "webtoon")
    assert len(fatias) >= 3
    for f in fatias:
        with Image.open(f) as im:
            assert im.width == 800 and im.height <= 1280
