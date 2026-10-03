"""Quadros da HQ: Qwen-Image-Edit com as refs do elenco, QA e retomada.

Cada quadro gera `n` candidatos por rodada; o QA (`hq/qa.py`) escolhe o melhor
aprovado. Se nenhum passar, outra rodada com seeds novas; esgotadas as
rodadas, o quadro fica em `needs_review` para o operador (`hq escolher`). O
estado vive no state.db (`hq_itens`), então um processo morto retoma de onde
parou, e os PNGs já em disco não são gerados de novo.

O InvokeAI aceita UMA referência latente (as outras entram só pelo encoder de
visão). Para quadros com dois personagens há três estratégias:

- `composta` (a do spike): as duas folhas lado a lado numa imagem só, que vai
  para a latente. Achado do spike: o quadro tende a copiar o layout lado a
  lado e o fundo liso da ref.
- `multi`: folha de A na latente + A e B pelo encoder de visão, num pedido.
- `duas-passadas`: `multi`, e depois uma edição do resultado (na latente) com
  a folha de B, pedindo só para acertar B. O dobro de GPU por candidato.

Cada quadro é uma cadeia de etapas (`_etapas`): a principal, a correção do
`duas-passadas` e, se o roteiro tiver `estampa`, uma de estilo.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import elenco
from . import projeto as hqp
from . import qa
from . import servico
from .roteiro import Quadro, Roteiro

ESTRATEGIAS = ("duas-passadas", "multi", "composta")
ESTRATEGIA_PADRAO = "duas-passadas"
PAPEL = (244, 236, 216)
# Etapa opcional de estilo: LoRA de style transfer (Apache-2.0, ver LICENSES.md)
# com uma gravura de referência (`estampa` no roteiro). Prompt do model card.
ESTILO_LORA = "Qwen Image Edit Style Transfer (dx8152)"
PESO_ESTILO = 1.0
PROMPT_ESTILO = ("style transfer. Change the style of Picture 1 to the style of Picture 2. "
                 "Keep the composition, the characters with their faces, hair, clothing "
                 "and poses, and everything depicted in Picture 1 unchanged.")
# Muda junto com qualquer prompt abaixo: entra na assinatura, e quadros já
# escolhidos com o prompt velho voltam para a fila.
VERSAO_PROMPT = 1


def pasta(proj: Path, qid: int) -> Path:
    return proj / "quadros" / str(qid)


def _quem(r: Roteiro, pid: str) -> str:
    p = r.personagens[pid]
    return f"{p.nome}: {p.ficha}"


def _lados(q: Quadro) -> list[str]:
    return ["on the left", "on the right"] if len(q.personagens) == 2 else ["in the scene"]


def prompt_quadro(r: Roteiro, q: Quadro, estrategia: str) -> str:
    final = (f"Draw a brand-new comic book panel with a completely new composition and "
             f"background: {q.cena}.")
    estilo = f" Art style: {r.estilo}"
    if not q.personagens:
        return final + estilo
    nomes = [r.personagens[p].nome for p in q.personagens]
    if len(q.personagens) == 1:
        return (f"The character in Picture 1 is {_quem(r, q.personagens[0])}. {final} "
                f"Keep {nomes[0]}'s face, hairstyle, ornaments and clothing exactly as in "
                f"Picture 1. Do not copy the plain background or the standing pose of "
                f"Picture 1.{estilo}")
    if estrategia == "composta":
        quem = "; ".join(f"{lado.replace('on the ', '')}: {_quem(r, p)}"
                         for lado, p in zip(_lados(q), q.personagens))
        return (f"Picture 1 shows the characters side by side: {quem}. {final} Keep each "
                f"character's face, hairstyle, ornaments and clothing exactly as in "
                f"Picture 1. Do not copy the plain background or the side-by-side layout "
                f"of Picture 1.{estilo}")
    quem = " ".join(f"Picture {i} is {_quem(r, p)}." for i, p in enumerate(q.personagens, 1))
    onde = ", ".join(f"{n} {lado}" for n, lado in zip(nomes, _lados(q)))
    return (f"{quem} {final} Place {onde}. Keep each character's face, hairstyle, "
            f"ornaments and clothing exactly as in their picture. Do not copy the plain "
            f"backgrounds or the standing poses of the pictures.{estilo}")


def prompt_correcao(r: Roteiro, q: Quadro) -> str:
    """Segunda passada: acerta o personagem da direita contra a folha dele."""
    a, b = (r.personagens[p] for p in q.personagens)
    return (f"In Picture 1, redraw only {b.nome}, the figure on the right, so that his or "
            f"her face, hairstyle, ornaments and clothing match the character in Picture 2 "
            f"exactly ({b.ficha}). Keep everything else in Picture 1 unchanged: the "
            f"composition, the poses, the background, {a.nome} on the left, the colors and "
            f"the art style.")


def _ref_composta(proj: Path, refs: dict[str, Path], pids: list[str]) -> Path:
    destino = proj / "quadros" / "_refs" / (
        "+".join(pids) + "-" + hqp.marca(*(hqp.marca_arquivo(refs[p]) for p in pids)) + ".png")
    if destino.exists():
        return destino
    imgs = [Image.open(refs[p]).convert("RGB") for p in pids]
    h = max(im.height for im in imgs)
    imgs = [im.resize((round(im.width * h / im.height), h)) for im in imgs]
    out = Image.new("RGB", (sum(im.width for im in imgs), h), PAPEL)
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    destino.parent.mkdir(parents=True, exist_ok=True)
    out.save(destino)
    return destino


def _refs_do_quadro(proj: Path, refs: dict[str, Path], q: Quadro, estrategia: str) -> list[Path]:
    if not q.personagens:
        return []
    if len(q.personagens) == 2 and estrategia == "composta":
        return [_ref_composta(proj, refs, q.personagens)]
    return [refs[p] for p in q.personagens]


def _duas(q: Quadro, estrategia: str) -> bool:
    return estrategia == "duas-passadas" and len(q.personagens) == 2


def assinatura(proj: Path, r: Roteiro, q: Quadro, refs: dict[str, Path], estrategia: str) -> str:
    estilo = ((ESTILO_LORA, PESO_ESTILO, PROMPT_ESTILO, hqp.marca_arquivo(proj / r.estampa))
              if r.estampa else ())
    return hqp.marca(VERSAO_PROMPT, estrategia, r.estilo, q.cena, q.proporcao, *estilo,
                     *(f"{p}={r.personagens[p].ficha}={hqp.marca_arquivo(refs[p])}"
                       for p in q.personagens))


@dataclass
class _Etapa:
    """Uma edição na cadeia do quadro. A partir da segunda, a imagem da etapa
    anterior vai na frente (latente) e `extras` vêm depois."""
    prompt: str
    extras: list[Path]
    loras: list[tuple[str, float]]


def _etapas(proj: Path, r: Roteiro, q: Quadro, refs: dict[str, Path],
            estrategia: str) -> list[_Etapa]:
    es = [_Etapa(prompt_quadro(r, q, estrategia), _refs_do_quadro(proj, refs, q, estrategia), [])]
    if _duas(q, estrategia):
        es.append(_Etapa(prompt_correcao(r, q), [refs[q.personagens[1]]], []))
    if r.estampa:
        es.append(_Etapa(PROMPT_ESTILO, [proj / r.estampa], [(ESTILO_LORA, PESO_ESTILO)]))
    return es


def _uma_etapa(c, proj: Path, alvo: list[Quadro], planos: dict, marcas: dict, seeds: dict,
               k: int, rodada: int, progresso) -> None:
    """Etapa `k` de todos os quadros, num lote só (a fila do InvokeAI não para).
    A última etapa de cada quadro grava `cand-*`; as do meio, `e<k>-*`."""
    trabalhos = []
    for q in alvo:
        es = planos[q.id]
        if k >= len(es):
            continue
        m = marcas[q.id]
        for i in seeds[q.id]:
            final = pasta(proj, q.id) / f"cand-{m}-{i}.png"
            destino = final if k == len(es) - 1 else pasta(proj, q.id) / f"e{k}-{m}-{i}.png"
            if final.exists() or destino.exists():
                continue
            anterior = pasta(proj, q.id) / f"e{k - 1}-{m}-{i}.png" if k else None
            trabalhos.append((q, i, destino, anterior, es[k]))
    if not trabalhos:
        return
    subir = [t[3] for t in trabalhos if t[3]] + [p for t in trabalhos for p in t[4].extras]
    with servico.refs_enviadas(c, subir) as nomes:
        pedidos = []
        for q, i, destino, anterior, et in trabalhos:
            rq = ([nomes[anterior]] if anterior else []) + [nomes[p] for p in et.extras]
            w, h = q.tamanho
            sd = hqp.seed(marcas[q.id] + (f"e{k}" if k else ""), i)
            g = (c.grafo_qwen_edit(et.prompt, rq, w, h, sd, loras=et.loras) if rq
                 else c.grafo_flux(et.prompt, w, h, sd))
            pedidos.append((g, destino))
        if progresso:
            progresso(f"rodada {rodada + 1}, etapa {k + 1}: {len(pedidos)} imagem(ns)")
        c.gerar_lote(pedidos, progresso=progresso)


def candidatos(proj: Path, qid: int, marca: str | None = None) -> list[Path]:
    padrao = f"cand-{marca}-*.png" if marca else "cand-*.png"
    return sorted(pasta(proj, qid).glob(padrao), key=lambda p: (len(p.name), p.name))


def _base(proj: Path, r: Roteiro, estrategia: str):
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"estratégia desconhecida: {estrategia} ({', '.join(ESTRATEGIAS)})")
    refs = elenco.refs_prontas(proj, r)
    e = hqp.estado(proj)
    marcas = {q.id: assinatura(proj, r, q, refs, estrategia) for q in r.quadros}
    e.sincronizar("quadro", {str(k): v for k, v in marcas.items()})
    e.reset_stale()
    return refs, e, marcas


def gerar(proj: Path, r: Roteiro, *, n: int = 2, estrategia: str = ESTRATEGIA_PADRAO,
          ids: list[int] | None = None, rodadas: int = 2, progresso=None) -> dict[int, dict]:
    """Gera, avalia e escolhe. Devolve {quadro: notas do QA por candidato}."""
    refs, e, marcas = _base(proj, r, estrategia)
    c = servico.cliente()
    relatorio: dict[int, dict] = {}
    for rodada in range(rodadas):
        alvo = [q for q in r.ordem_de_leitura() if (not ids or q.id in ids)
                and e.item("quadro", q.id)["state"] == "pending"]
        if not alvo:
            break
        # seeds novas: depois dos candidatos que já existem (de rodadas, ou
        # execuções, anteriores) — `--refazer` gera candidatos de verdade novos
        seeds = {}
        for q in alvo:
            ini = len(candidatos(proj, q.id, marcas[q.id]))
            seeds[q.id] = range(ini, ini + n)
            e.iniciar("quadro", q.id)
        planos = {q.id: _etapas(proj, r, q, refs, estrategia) for q in alvo}
        for k in range(max(len(v) for v in planos.values())):
            _uma_etapa(c, proj, alvo, planos, marcas, seeds, k, rodada, progresso)
        # QA: todos os candidatos da assinatura atual, inclusive de rodadas anteriores
        for q in alvo:
            cands = candidatos(proj, q.id, marcas[q.id])
            notas = qa.avaliar(cands, len(q.personagens), r.ancora_estilo)
            relatorio[q.id] = notas
            escolhido = qa.melhor(notas)
            if escolhido:
                e.concluir("quadro", q.id, pasta(proj, q.id) / escolhido, qa=notas)
            elif rodada + 1 < rodadas:
                e.reabrir("quadro", q.id)
            else:
                e.revisar("quadro", q.id, "nenhum candidato passou no QA — "
                          f"genai hq escolher {proj.name} {q.id} <arquivo>", qa=notas)
    return relatorio


def refazer(proj: Path, r: Roteiro, ids: list[int]) -> None:
    """Devolve quadros à fila para gerar candidatos NOVOS (a escolha atual cai)."""
    e = hqp.estado(proj)
    for qid in ids:
        if e.item("quadro", qid):
            e.reabrir("quadro", qid)


def escolher(proj: Path, r: Roteiro, qid: int, arquivo: Path) -> Path:
    """Escolha humana: vale qualquer imagem (um candidato, um retoque do Canvas).
    Fora da pasta do quadro, é copiada para lá."""
    arquivo = Path(arquivo)
    if not arquivo.exists():
        raise FileNotFoundError(arquivo)
    if qid not in {q.id for q in r.quadros}:
        raise ValueError(f"quadro {qid} não existe no roteiro")
    e = hqp.estado(proj)
    if e.item("quadro", qid) is None:
        raise ValueError(f"quadro {qid} ainda não foi gerado — rode `genai hq quadros`")
    destino = arquivo
    if arquivo.resolve().parent != pasta(proj, qid).resolve():
        destino = pasta(proj, qid) / f"humano-{arquivo.name}"
        shutil.copy(arquivo, destino)
    e.concluir("quadro", qid, destino, por="humano")
    return destino


def escolhidos(proj: Path, r: Roteiro) -> dict[int, Path]:
    """{quadro: imagem escolhida} — erro listando os que faltam."""
    e = hqp.estado(proj)
    out, faltam = {}, []
    for q in r.ordem_de_leitura():
        p = e.escolhido("quadro", q.id)
        (out.__setitem__(q.id, p) if p and p.exists() else faltam.append(q.id))
    if faltam:
        raise ValueError(f"quadros sem escolha: {faltam} — rode `genai hq quadros` "
                         "ou `genai hq escolher`")
    return out
