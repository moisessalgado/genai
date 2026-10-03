"""Export da HQ: PDF (impressão), CBZ (leitores de quadrinhos) e webtoon (rolagem).

PDF e CBZ partem das páginas montadas (`paginas/`). O webtoon não: rolagem
vertical lê um quadro por vez na largura inteira da tela, então os quadros
são re-letreirados a 800 px (a largura do LINE Webtoon) com fonte de leitura
de celular, empilhados com respiro e fatiados em imagens de até 1280 px de
altura — o corte cai no respiro entre quadros sempre que dá.
"""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

from . import letreiro, rostos
from .roteiro import Roteiro

Q_JPEG = 90
WEBTOON_LARGURA = 800
WEBTOON_FATIA = 1280
WEBTOON_RESPIRO = 80
WEBTOON_FONTE = 30


def nome_de_arquivo(titulo: str) -> str:
    limpo = "".join(" " if c in '/\\:*?"<>|' else c for c in titulo)
    return " ".join(limpo.split()).strip(". ") or "hq"


def pdf(paginas: list[Path], destino: Path, dpi: int = 300, titulo: str | None = None) -> Path:
    """Páginas como JPEG embutido (o PNG de 300 dpi daria ~10 MB por página)."""
    imgs = [Image.open(p).convert("RGB") for p in paginas]
    destino.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(destino, "PDF", save_all=True, append_images=imgs[1:], resolution=dpi,
                 quality=Q_JPEG, title=titulo or destino.stem, creator="genai hq")
    return destino


def comicinfo(r: Roteiro, n_paginas: int, autoria: str | None = None) -> str:
    campos = {"Title": r.titulo, "Series": r.titulo, "Number": "1", "PageCount": str(n_paginas),
              "LanguageISO": r.idioma.split("-")[0], "Manga": "No", "Format": "Web"}
    if autoria:
        campos["Writer"] = autoria
    corpo = "\n".join(f"  <{k}>{escape(v)}</{k}>" for k, v in campos.items())
    return ('<?xml version="1.0" encoding="utf-8"?>\n<ComicInfo xmlns:xsi='
            '"http://www.w3.org/2001/XMLSchema-instance">\n' + corpo + "\n</ComicInfo>\n")


def cbz(paginas: list[Path], destino: Path, r: Roteiro, autoria: str | None = None) -> Path:
    """ZIP sem compressão (JPEG não comprime mais) com ComicInfo.xml."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".parcial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr("ComicInfo.xml", comicinfo(r, len(paginas), autoria))
        for i, p in enumerate(paginas, 1):
            with Image.open(p) as im:
                buf = BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=Q_JPEG)
            z.writestr(f"{i:03d}.jpg", buf.getvalue())
    tmp.replace(destino)
    return destino


def _cortes(alturas_quadros: list[tuple[int, int]], total: int, fatia: int) -> list[int]:
    """Pontos de corte (y) da tira vertical. `alturas_quadros` = (y0, y1) de
    cada quadro; corta no meio do respiro anterior ao limite, ou no limite se
    um quadro sozinho passar da fatia."""
    cortes, inicio = [], 0
    while total - inicio > fatia:
        limite = inicio + fatia
        respiros = [(y1 + y0n) // 2 for (_, y1), (y0n, _) in zip(alturas_quadros, alturas_quadros[1:])
                    if inicio < (y1 + y0n) // 2 <= limite]
        corte = max(respiros) if respiros else limite
        cortes.append(corte)
        inicio = corte
    return cortes


def webtoon(r: Roteiro, artes: dict[int, Path], destino_dir: Path,
            cache_rostos: dict[Path, list] | None = None) -> list[Path]:
    cache_rostos = {} if cache_rostos is None else cache_rostos
    blocos, posicoes, y = [], [], WEBTOON_RESPIRO
    for q in r.ordem_de_leitura():
        w0, h0 = q.tamanho
        h = round(WEBTOON_LARGURA * h0 / w0)
        arte = artes[q.id]
        if arte not in cache_rostos:
            cache_rostos[arte] = rostos.detectar(arte)
        img, _ = letreiro.letrar(arte, q, (WEBTOON_LARGURA, h), WEBTOON_FONTE,
                                 cache_rostos[arte])
        blocos.append((img, y))
        posicoes.append((y, y + h))
        y += h + WEBTOON_RESPIRO
    tira = Image.new("RGB", (WEBTOON_LARGURA, y), letreiro.PAPEL)
    for img, yy in blocos:
        tira.paste(img, (0, yy))
    destino_dir.mkdir(parents=True, exist_ok=True)
    for velho in destino_dir.glob("*.jpg"):
        velho.unlink()
    limites = [0] + _cortes(posicoes, y, WEBTOON_FATIA) + [y]
    saida = []
    for i, (a, b) in enumerate(zip(limites, limites[1:]), 1):
        p = destino_dir / f"{i:03d}.jpg"
        tira.crop((0, a, WEBTOON_LARGURA, b)).save(p, "JPEG", quality=Q_JPEG)
        saida.append(p)
    return saida
