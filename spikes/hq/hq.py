"""Spike do pipeline de HQ: roteiro -> elenco -> quadros -> página letreirada.

    python spikes/hq/hq.py elenco --n 4          # candidatos de folha-modelo (FLUX)
    python spikes/hq/hq.py aprovar siddhartha <png>
    python spikes/hq/hq.py quadros --n 2         # candidatos por quadro (Qwen Edit + refs)
    python spikes/hq/hq.py escolher 3 <png>      # troca o candidato de um quadro
    python spikes/hq/hq.py pagina                # diagrama + letreira -> output/pagina-1.png

Tudo vai para projects/<slug>/ e para um board homônimo no InvokeAI.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import invoke  # noqa: E402

RAIZ = AQUI.parents[1]
ROTEIRO = yaml.safe_load((AQUI / "roteiro.yaml").read_text(encoding="utf-8"))
PROJ = RAIZ / "projects" / ROTEIRO["slug"]
FONTE = AQUI / "fonts" / "ComicNeue-Bold.ttf"
PAPEL = (244, 236, 216)
SELECAO = PROJ / "selecao.json"


def _selecao() -> dict:
    return json.loads(SELECAO.read_text()) if SELECAO.exists() else {}


def _salvar_selecao(s: dict) -> None:
    SELECAO.write_text(json.dumps(s, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- elenco
def elenco(n: int) -> None:
    b = invoke.board(ROTEIRO["slug"])
    for pid, p in ROTEIRO["personagens"].items():
        prompt = (f"{ROTEIRO['estilo']} Full-body character reference sheet of {p['ficha']}. "
                  "Single figure standing upright, front view, arms relaxed, whole body visible "
                  "from head to feet, on a plain empty cream paper background, nothing else.")
        for seed in range(1, n + 1):
            nome = invoke.flux(prompt, 832, 1216, seed, b)
            dst = invoke.baixar(nome, PROJ / "elenco" / pid / f"cand-{seed}.png")
            print(pid, dst)


def aprovar(pid: str, arquivo: str) -> None:
    dst = PROJ / "elenco" / f"{pid}.png"
    shutil.copy(arquivo, dst)
    print("aprovado", dst)


def _ref_composta(pids: list[str]) -> Path:
    """Folhas-modelo lado a lado numa imagem só: o InvokeAI aceita UMA referência
    latente, então quadros com dois personagens recebem os dois na mesma imagem."""
    dst = PROJ / "refs" / ("+".join(pids) + ".png")
    if dst.exists():
        return dst
    imgs = [Image.open(PROJ / "elenco" / f"{p}.png").convert("RGB") for p in pids]
    h = 1216
    imgs = [im.resize((round(im.width * h / im.height), h)) for im in imgs]
    out = Image.new("RGB", (sum(im.width for im in imgs), h), PAPEL)
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return dst


# ---------------------------------------------------------------- quadros
def _prompt_quadro(q: dict) -> str:
    ps = [ROTEIRO["personagens"][p] for p in q["personagens"]]
    if len(ps) == 1:
        quem = f"The character in Picture 1 is {ps[0]['nome']}: {ps[0]['ficha']}."
    else:
        quem = "Picture 1 shows the characters side by side: " + "; ".join(
            f"{'left' if i == 0 else 'right'}: {p['nome']}, {p['ficha']}" for i, p in enumerate(ps)) + "."
    return (f"{quem} Draw a brand-new comic book panel with a completely new composition "
            f"and background: {q['cena']}. Keep each character's face, hairstyle, ornaments "
            f"and clothing exactly as in Picture 1. Do not copy the plain background or the "
            f"side-by-side layout of Picture 1. Art style: {ROTEIRO['estilo']}")


def quadros(n: int, ids: list[int] | None) -> None:
    b = invoke.board(ROTEIRO["slug"])
    sel = _selecao()
    refs: dict[str, str] = {}
    for q in ROTEIRO["quadros"]:
        if ids and q["id"] not in ids:
            continue
        chave = "+".join(q["personagens"])
        if chave not in refs:
            refs[chave] = invoke.upload(_ref_composta(q["personagens"]), b)
        prompt = _prompt_quadro(q)
        for seed in range(1, n + 1):
            nome = invoke.qwen_edit(prompt, [refs[chave]], q["w"], q["h"], seed, b)
            dst = invoke.baixar(nome, PROJ / "quadros" / str(q["id"]) / f"cand-{seed}.png")
            print(q["id"], dst)
            sel.setdefault(str(q["id"]), str(dst))
        _salvar_selecao(sel)


def escolher(qid: int, arquivo: str) -> None:
    sel = _selecao()
    sel[str(qid)] = str(Path(arquivo).resolve())
    _salvar_selecao(sel)


# ---------------------------------------------------------------- página
def _cover(im: Image.Image, w: int, h: int) -> Image.Image:
    """Preenche w x h cortando o excesso (centro), sem distorcer."""
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def _layout(pg: dict) -> dict[int, tuple[int, int, int, int]]:
    """Retângulos dos quadros. Cada tira tem altura comum; a largura de cada
    quadro sai da proporção gerada; no fim as tiras são escaladas para ocupar a
    altura útil (o _cover absorve a diferença de proporção)."""
    qs = {q["id"]: q for q in ROTEIRO["quadros"]}
    W = pg["largura"] - 2 * pg["margem"]
    H = pg["altura"] - 2 * pg["margem"]
    g = pg["calha"]
    alturas = []
    for tira in pg["tiras"]:
        soma = sum(qs[i]["w"] / qs[i]["h"] for i in tira)
        alturas.append((W - g * (len(tira) - 1)) / soma)
    fator = (H - g * (len(pg["tiras"]) - 1)) / sum(alturas)
    rects, y = {}, pg["margem"]
    for tira, h0 in zip(pg["tiras"], alturas):
        h = round(h0 * fator)
        larg = [qs[i]["w"] / qs[i]["h"] * h0 for i in tira]
        x = pg["margem"]
        for i, lw in zip(tira, larg):
            w = round(lw) if i != tira[-1] else pg["margem"] + W - x
            rects[i] = (x, y, w, h)
            x += w + g
        y += h + g
    return rects


def _quebrar(texto: str, fonte, largura_max: int, draw) -> list[str]:
    for chars in range(40, 8, -2):
        linhas = textwrap.wrap(texto, chars)
        if max(draw.textlength(l, font=fonte) for l in linhas) <= largura_max:
            return linhas
    return textwrap.wrap(texto, 10)


def _letrar(draw: ImageDraw.ImageDraw, rect, t: dict, idx: int) -> None:
    x, y, w, h = rect
    fonte = ImageFont.truetype(str(FONTE), 46)
    pad = 28
    larg_max = int(w * (0.85 if t["tipo"] == "recordatorio" else 0.30))
    linhas = _quebrar(t["texto"], fonte, larg_max, draw)
    lh = 54
    tw = max(draw.textlength(l, font=fonte) for l in linhas)
    th = lh * len(linhas)
    pos = t.get("pos", "topo")
    if t["tipo"] == "recordatorio":
        bw, bh = tw + 2 * pad, th + 2 * pad
        bx = x + 24 if "esquerda" in pos or pos == "topo" else x + w - bw - 24
        by = y + 24
        draw.rectangle((bx, by, bx + bw, by + bh), fill=PAPEL, outline="black", width=5)
        tx, ty = bx + pad, by + pad
    else:
        # elipse que contém o retângulo do texto: semi-eixos ~ sqrt(2) x metade do texto
        bw, bh = tw * 1.42 + pad, th * 1.42 + pad
        if "esquerda" in pos:
            bx = x + 40
        elif "direita" in pos:
            bx = x + w - bw - 40
        else:
            bx = x + (w - bw) / 2
        by = y + 36 + (idx * 0.15 * h if pos == "topo" else 0)
        # rabicho: do balão para baixo, inclinado para o centro do quadro
        cx = bx + bw / 2
        alvo_x = cx + (60 if cx < x + w / 2 else -60)
        base = by + bh - 10
        draw.polygon([(cx - 26, base - 8), (cx + 26, base - 8), (alvo_x, base + 90)],
                     fill="white", outline="black", width=5)
        draw.ellipse((bx, by, bx + bw, by + bh), fill="white", outline="black", width=5)
        draw.polygon([(cx - 22, base - 14), (cx + 22, base - 14), (alvo_x, base + 82)], fill="white")
        tx, ty = bx + (bw - tw) / 2, by + (bh - th) / 2
    for i, l in enumerate(linhas):
        lx = tx + (tw - draw.textlength(l, font=fonte)) / 2 if t["tipo"] != "recordatorio" else tx
        draw.text((lx, ty + i * lh), l, font=fonte, fill="black")


def pagina() -> Path:
    pg = ROTEIRO["pagina"]
    sel = _selecao()
    folha = Image.new("RGB", (pg["largura"], pg["altura"]), PAPEL)
    draw = ImageDraw.Draw(folha)
    for qid, rect in _layout(pg).items():
        x, y, w, h = rect
        arte = _cover(Image.open(sel[str(qid)]).convert("RGB"), w, h)
        folha.paste(arte, (x, y))
        draw.rectangle((x, y, x + w, y + h), outline="black", width=8)
        q = next(q for q in ROTEIRO["quadros"] if q["id"] == qid)
        for i, t in enumerate(q.get("textos", [])):
            _letrar(draw, rect, t, i)
    dst = PROJ / "output" / "pagina-1.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    folha.save(dst, dpi=(300, 300))
    folha.resize((pg["largura"] // 3, pg["altura"] // 3), Image.LANCZOS).save(dst.with_suffix(".preview.png"))
    print(dst)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("elenco").add_argument("--n", type=int, default=4)
    a = sp.add_parser("aprovar"); a.add_argument("pid"); a.add_argument("arquivo")
    q = sp.add_parser("quadros"); q.add_argument("--n", type=int, default=2)
    q.add_argument("--ids", type=int, nargs="*")
    e = sp.add_parser("escolher"); e.add_argument("qid", type=int); e.add_argument("arquivo")
    sp.add_parser("pagina")
    args = ap.parse_args()
    if args.cmd == "elenco":
        elenco(args.n)
    elif args.cmd == "aprovar":
        aprovar(args.pid, args.arquivo)
    elif args.cmd == "quadros":
        quadros(args.n, args.ids)
    elif args.cmd == "escolher":
        escolher(args.qid, args.arquivo)
    else:
        pagina()
