"""Diagramação: retângulos dos quadros na página e a página montada e letreirada.

Algoritmo do spike: cada tira tem altura comum, a largura de cada quadro sai
da proporção em que ele foi gerado, e no fim as tiras são escaladas para
ocupar a altura útil da página. A diferença de proporção que sobra é
absorvida cortando o centro da arte (`letreiro.cobrir`) — a tira foi montada
para que essa diferença seja pequena (`roteiro_llm.montar_tiras`).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from . import letreiro, rostos
from .roteiro import Formato, Pagina, Roteiro

Retangulo = tuple[int, int, int, int]  # x, y, w, h em px


def retangulos(r: Roteiro, pagina: Pagina, formato: Formato | None = None) -> dict[int, Retangulo]:
    f = formato or r.formato
    W = f.largura - 2 * f.margem
    H = f.altura - 2 * f.margem
    g = f.calha
    prop = {q.id: q.tamanho[0] / q.tamanho[1] for q in r.quadros}
    alturas = [(W - g * (len(t) - 1)) / sum(prop[i] for i in t) for t in pagina.tiras]
    fator = (H - g * (len(pagina.tiras) - 1)) / sum(alturas)
    rects: dict[int, Retangulo] = {}
    y = f.margem
    for k, (tira, h0) in enumerate(zip(pagina.tiras, alturas)):
        h = round(h0 * fator) if k < len(pagina.tiras) - 1 else f.margem + H - y
        x = f.margem
        for i in tira:
            w = round(prop[i] * h0) if i != tira[-1] else f.margem + W - x
            rects[i] = (x, y, w, h)
            x += w + g
        y += h + g
    return rects


def fonte_px(largura_pagina: int) -> int:
    return round(largura_pagina * letreiro.FONTE_REL_PAGINA)


def montar_pagina(r: Roteiro, pagina: Pagina, artes: dict[int, Path],
                  cache_rostos: dict[Path, list] | None = None) -> Image.Image:
    f = r.formato
    folha = Image.new("RGB", (f.largura, f.altura), letreiro.PAPEL)
    d = ImageDraw.Draw(folha)
    px = fonte_px(f.largura)
    borda = max(4, round(f.largura / 310))
    cache_rostos = {} if cache_rostos is None else cache_rostos
    for qid, (x, y, w, h) in retangulos(r, pagina).items():
        arte = artes[qid]
        if arte not in cache_rostos:
            cache_rostos[arte] = rostos.detectar(arte)
        img, _ = letreiro.letrar(arte, r.quadro(qid), (w, h), px, cache_rostos[arte])
        folha.paste(img, (x, y))
        d.rectangle((x, y, x + w - 1, y + h - 1), outline="black", width=borda)
    return folha


def paginas(proj: Path, r: Roteiro, artes: dict[int, Path], progresso=None) -> list[Path]:
    """Monta e grava `paginas/pagina-N.png` (PNG com o dpi do formato)."""
    destino = proj / "paginas"
    destino.mkdir(parents=True, exist_ok=True)
    for velho in destino.glob("pagina-*.png"):  # roteiro com menos páginas que antes
        velho.unlink()
    cache: dict[Path, list] = {}
    saida = []
    for n, pg in enumerate(r.paginas, 1):
        img = montar_pagina(r, pg, artes, cache)
        p = destino / f"pagina-{n:02d}.png"
        img.save(p, dpi=(r.formato.dpi, r.formato.dpi))
        saida.append(p)
        if progresso:
            progresso(f"página {n}/{len(r.paginas)}")
    return saida
