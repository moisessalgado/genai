"""Folha de contato: os candidatos lado a lado, rotulados, numa imagem só.

É como o operador escolhe sem abrir arquivo por arquivo — o rótulo é o nome
do arquivo que vai no `elenco-aprovar`/`escolher`."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .letreiro import FONTE


def montar(imagens: list[Path], destino: Path, *, altura: int = 480, por_linha: int = 4,
           notas: dict[str, str] | None = None) -> Path:
    """`notas`: texto extra por nome de arquivo (ex.: o veredito do QA)."""
    notas = notas or {}
    fonte = ImageFont.truetype(str(FONTE), 22)
    miniaturas = []
    for p in imagens:
        with Image.open(p) as im:
            im = im.convert("RGB")
            miniaturas.append(im.resize((round(im.width * altura / im.height), altura)))
    if not miniaturas:
        raise ValueError("nenhuma imagem para a folha de contato")
    rotulo = 64
    linhas = [list(range(i, min(i + por_linha, len(miniaturas))))
              for i in range(0, len(miniaturas), por_linha)]
    larg = max(sum(miniaturas[i].width for i in ln) + 8 * (len(ln) - 1) for ln in linhas)
    folha = Image.new("RGB", (larg, (altura + rotulo) * len(linhas)), "white")
    d = ImageDraw.Draw(folha)
    for li, ln in enumerate(linhas):
        x, y = 0, li * (altura + rotulo)
        for i in ln:
            p, m = imagens[i], miniaturas[i]
            folha.paste(m, (x, y))
            d.text((x + 6, y + altura + 4), p.name, font=fonte, fill="black")
            if p.name in notas:
                d.text((x + 6, y + altura + 32), notas[p.name], font=fonte, fill=(160, 0, 0))
            x += m.width + 8
    destino.parent.mkdir(parents=True, exist_ok=True)
    folha.save(destino)
    return destino
