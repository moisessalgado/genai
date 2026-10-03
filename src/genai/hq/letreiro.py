"""Letreiramento da HQ: balões e recordatórios desenhados sobre o quadro.

O plano (onde vai cada balão) é feito em coordenadas NORMALIZADAS do quadro,
para o mesmo arranjo servir à página impressa, ao webtoon e ao vídeo; só o
tamanho da fonte muda com o meio (`fonte_px`).

Posição: busca em grade, com custo. O que mais pesa é cobrir rosto (YuNet,
`hq/rostos.py`) — achado do spike: o balão do Channa caiu em cima da cabeça
dele no quadro 3. Depois: encostar em outro balão, a ordem de leitura (o
balão seguinte fica à direita ou abaixo do anterior), ficar perto do alto do
quadro e perto de quem fala. O rabicho aponta para o rosto de quem fala: os
personagens do roteiro estão na ordem esquerda→direita, e os rostos de
primeiro plano também.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .rostos import Rosto, principais
from .roteiro import Quadro

# Comic Neue Bold, SIL OFL 1.1 (OFL.txt ao lado) — ver LICENSES.md.
FONTE = Path(__file__).resolve().parent / "fonts" / "ComicNeue-Bold.ttf"
PAPEL = (244, 236, 216)

# Fonte relativa à largura da PÁGINA (não do quadro): na página do spike, 46 px
# em 2480 — o mesmo corpo em todos os quadros, como numa HQ de verdade.
FONTE_REL_PAGINA = 46 / 2480

# Margem de segurança em volta do rosto (fração do tamanho dele): cabelo,
# coroa e turbante ficam fora da caixa do YuNet.
FOLGA_ROSTO = 0.45


@dataclass
class Balao:
    tipo: str          # recordatorio | fala | pensamento
    texto: str
    quem: str | None
    linhas: list[str]
    caixa: tuple[float, float, float, float]   # x, y, w, h normalizados
    alvo: tuple[float, float] | None = None    # ponta do rabicho, normalizada


def _fonte(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTE), max(8, px))


def quebrar(texto: str, fonte, largura_max: float) -> list[str]:
    """Quebra gulosa pela largura real do texto (não por número de caracteres)."""
    linhas, atual = [], ""
    for palavra in texto.split():
        tent = f"{atual} {palavra}".strip()
        if not atual or fonte.getlength(tent) <= largura_max:
            atual = tent
        else:
            linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas or [""]


def _formas(t, fonte, px: int, w: int) -> list[tuple[list[str], float, float, float]]:
    """Formas possíveis do balão: (linhas, largura, altura, custo da forma).

    Balão: várias larguras de quebra; a forma perto de 1,6:1 é a mais barata,
    mas uma mais larga e baixa pode ser o que cabe no céu acima das cabeças."""
    pad = px * 0.6
    if t.tipo == "recordatorio":
        out = []
        for frac in (0.82, 0.6, 0.45):
            linhas = quebrar(t.texto, fonte, w * frac - 2 * pad)
            tw = max(fonte.getlength(l) for l in linhas)
            out.append((linhas, tw + 2 * pad, px * 1.18 * len(linhas) + 2 * pad,
                        0.1 * (0.82 - frac)))
        return out
    out = []
    for frac in (0.22, 0.27, 0.32, 0.38, 0.45, 0.55):
        linhas = quebrar(t.texto, fonte, max(px * 5, w * frac))
        tw = max(fonte.getlength(l) for l in linhas)
        th = px * 1.18 * len(linhas)
        bw, bh = tw * 1.38 + 2 * pad, th * 1.38 + 2 * pad
        out.append((linhas, bw, bh, 0.4 * abs(bw / bh - 1.6)))
    return out


def _inter(a, b) -> float:
    x = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return x * y


def _rosto_expandido(r: Rosto) -> tuple[float, float, float, float]:
    fx, fy = r.w * FOLGA_ROSTO, r.h * FOLGA_ROSTO
    return r.x - fx, r.y - fy * 1.6, r.w + 2 * fx, r.h + fy * 2.6  # mais folga em cima


def falantes(quadro: Quadro, rostos: list[Rosto]) -> dict[str, Rosto]:
    """Rosto de cada personagem do quadro, pela ordem esquerda→direita.

    Com mais rostos que personagens (figurantes), ficam os maiores; com menos,
    não há como saber quem é quem e ninguém ganha rosto (o rabicho aponta para
    baixo, do lado do personagem)."""
    fs = principais(rostos)
    n = len(quadro.personagens)
    if n == 0 or len(fs) < n:
        return {}
    fs = sorted(sorted(fs, key=lambda r: -r.h)[:n], key=lambda r: r.x)
    return dict(zip(quadro.personagens, fs))


POR_BALAO = 150  # melhores posições de cada balão levadas à busca conjunta
FEIXE = 400      # combinações parciais mantidas a cada balão


def _alvo(t, quadro: Quadro, caras: dict[str, Rosto]) -> tuple[float, float] | None:
    if t.quem in caras:
        rc = caras[t.quem]
        return rc.centro[0], max(0.0, rc.y - rc.h * 0.15)
    if t.quem:
        i = quadro.personagens.index(t.quem)
        return (i + 0.5) / len(quadro.personagens), 0.55
    return None


def _corpo(r: Rosto) -> tuple[float, float, float, float]:
    """Faixa abaixo do rosto onde está o corpo: balão ali esconde o personagem."""
    return r.x - r.w * 0.6, r.y + r.h, r.w * 2.2, 1.0


def _candidatos(t, alvo, formas, w, h, margem, proibido, corpos) -> list[tuple[float, tuple, list[str]]]:
    """(custo individual, caixa, linhas) das posições em grade, as melhores primeiro."""
    out = []
    passos = 14
    for linhas, bw, bh, custo_forma in formas:
        nw, nh = min(bw, w - 2 * margem) / w, min(bh, h - 2 * margem) / h
        mx, my = margem / w, margem / h
        for iy in range(passos):
            y = my + (1 - 2 * my - nh) * iy / (passos - 1)
            for ix in range(passos):
                x = mx + (1 - 2 * mx - nw) * ix / (passos - 1)
                cx = (x, y, nw, nh)
                area = nw * nh
                custo = custo_forma + 60 * sum(_inter(cx, f) for f in proibido) / area
                custo += 12 * sum(_inter(cx, c) for c in corpos) / area
                custo += 1.2 * y  # prefere o alto do quadro
                if alvo is not None:
                    # perto (na horizontal) e acima de quem fala
                    custo += 1.5 * abs(x + nw / 2 - alvo[0])
                    if y + nh > alvo[1]:
                        custo += 5 * (y + nh - alvo[1])
                elif t.tipo == "recordatorio":
                    custo += 0.8 * x  # recordatório começa no canto esquerdo
                out.append((custo, cx, linhas))
    out.sort(key=lambda c: c[0])
    return out[:POR_BALAO]


def _custo_par(a: tuple, b: tuple) -> float:
    """`b` vem depois de `a` na leitura: não se encostam, e `b` fica à direita
    ou abaixo."""
    sobre = _inter(a, b)
    # balão sobre balão esconde texto: proibido, não só caro
    custo = 1000 + 40 * sobre / min(a[2] * a[3], b[2] * b[3]) if sobre > 0 else 0.0
    if b[0] + b[2] / 2 < a[0] + a[2] / 2 and b[1] < a[1] + a[3] * 0.5:
        custo += 3
    return custo


def planejar(quadro: Quadro, tamanho: tuple[int, int], rostos: list[Rosto],
             fonte_px: int) -> list[Balao]:
    """Posições dos balões: a melhor COMBINAÇÃO (o primeiro não toma o lugar
    de que o segundo precisava), por busca em feixe."""
    w, h = tamanho
    fonte = _fonte(fonte_px)
    margem = fonte_px * 0.8
    caras = falantes(quadro, rostos)
    proibido = [_rosto_expandido(r) for r in rostos]
    corpos = [_corpo(r) for r in principais(rostos)]
    alvos = [_alvo(t, quadro, caras) for t in quadro.textos]
    cands = [_candidatos(t, a, _formas(t, fonte, fonte_px, w), w, h, margem, proibido,
                         corpos)
             for t, a in zip(quadro.textos, alvos)]
    if not cands:
        return []
    feixe: list[tuple[float, list]] = [(0.0, [])]
    for cs in cands:
        novos = [(custo + c[0] + sum(_custo_par(m[1], c[1]) for m in comb), comb + [c])
                 for custo, comb in feixe for c in cs]
        novos.sort(key=lambda x: x[0])
        feixe = novos[:FEIXE]
    melhor = feixe[0][1]
    return [Balao(t.tipo, t.texto, t.quem, c[2], c[1], a)
            for t, c, a in zip(quadro.textos, melhor, alvos)]


def _px(b, w, h):
    x, y, bw, bh = b.caixa
    return x * w, y * h, bw * w, bh * h


def desenhar(img: Image.Image, baloes: list[Balao], fonte_px: int,
             ate: int | None = None) -> Image.Image:
    """Desenha os balões (os `ate` primeiros, para o motion comic revelar um a
    um). Desenha em 2× e reduz: o `ellipse` do Pillow não tem antialias."""
    w, h = img.size
    k = 2
    camada = Image.new("RGBA", (w * k, h * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    fonte = _fonte(fonte_px * k)
    linha = max(2, round(fonte_px * 0.11)) * k
    for b in baloes[:ate]:
        x, y, bw, bh = (v * k for v in _px(b, w, h))
        lh = fonte_px * 1.18 * k
        th = lh * len(b.linhas)
        if b.tipo == "recordatorio":
            d.rectangle((x, y, x + bw, y + bh), fill=PAPEL + (255,), outline="black",
                        width=linha)
            ty = y + (bh - th) / 2
            for i, l in enumerate(b.linhas):
                d.text((x + fonte_px * 0.6 * k, ty + i * lh), l, font=fonte, fill="black")
            continue
        cx, cy = x + bw / 2, y + bh / 2
        if b.alvo is not None:
            ax, ay = b.alvo[0] * w * k, b.alvo[1] * h * k
            # a ponta para antes do alvo: aponta para a cabeça, não a cobre
            dx, dy = ax - cx, ay - cy
            dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
            alcance = min(dist * 0.85, bh * 0.5 + fonte_px * 3.2 * k)
            px_, py_ = cx + dx / dist * alcance, cy + dy / dist * alcance
            base = bw * 0.09
            nx, ny = -dy / dist * base, dx / dist * base
            if b.tipo == "pensamento":
                for f, raio in ((0.62, 0.32), (0.8, 0.2)):
                    qx, qy = cx + (px_ - cx) * f, cy + (py_ - cy) * f
                    rr = fonte_px * raio * k
                    d.ellipse((qx - rr, qy - rr, qx + rr, qy + rr), fill="white",
                              outline="black", width=linha)
            else:
                d.polygon([(cx + nx, cy + ny), (cx - nx, cy - ny), (px_, py_)],
                          fill="white", outline="black", width=linha)
        d.ellipse((x, y, x + bw, y + bh), fill="white", outline="black", width=linha)
        if b.alvo is not None and b.tipo != "pensamento":
            # repinta a base do rabicho por cima do contorno da elipse
            d.polygon([(cx + nx * 0.8, cy + ny * 0.8), (cx - nx * 0.8, cy - ny * 0.8),
                       (cx + (px_ - cx) * 0.55, cy + (py_ - cy) * 0.55)], fill="white")
        ty = cy - th / 2
        for i, l in enumerate(b.linhas):
            lw = fonte.getlength(l)
            d.text((cx - lw / 2, ty + i * lh), l, font=fonte, fill="black")
    camada = camada.resize((w, h), Image.LANCZOS)
    out = img.convert("RGBA")
    out.alpha_composite(camada)
    return out.convert("RGB")


def cobrir(im: Image.Image, w: int, h: int) -> Image.Image:
    """Preenche w×h cortando o excesso (centro), sem distorcer."""
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def letrar(arte: Path, quadro: Quadro, tamanho: tuple[int, int], fonte_px: int,
           rostos: list[Rosto], ate: int | None = None) -> tuple[Image.Image, list[Balao]]:
    """A arte no tamanho pedido (cortada ao centro) com os balões. `rostos` são
    os da ARTE ORIGINAL; aqui são reprojetados para o recorte."""
    with Image.open(arte) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
    w, h = tamanho
    s = max(w / w0, h / h0)
    ox, oy = (w0 * s - w) / 2 / (w0 * s), (h0 * s - h) / 2 / (h0 * s)
    fx, fy = w0 * s / w, h0 * s / h
    reproj = [Rosto((r.x - ox) * fx, (r.y - oy) * fy, r.w * fx, r.h * fy, r.nota)
              for r in rostos]
    img = cobrir(im, w, h)
    baloes = planejar(quadro, (w, h), reproj, fonte_px)
    return desenhar(img, baloes, fonte_px, ate), baloes
