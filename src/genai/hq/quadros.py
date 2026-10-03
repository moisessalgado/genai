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
"""
from __future__ import annotations

import shutil
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
    return hqp.marca(VERSAO_PROMPT, estrategia, r.estilo, q.cena, q.proporcao,
                     *(f"{p}={r.personagens[p].ficha}={hqp.marca_arquivo(refs[p])}"
                       for p in q.personagens))


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
        seeds = range(rodada * n, (rodada + 1) * n)
        for q in alvo:
            e.iniciar("quadro", q.id)
        todas_refs = [p for q in alvo for p in _refs_do_quadro(proj, refs, q, estrategia)]
        with servico.refs_enviadas(c, todas_refs) as nomes:
            # passada 1 (única, fora do duas-passadas com 2 personagens)
            pedidos = []
            for q in alvo:
                m = marcas[q.id]
                w, h = q.tamanho
                rq = [nomes[p] for p in _refs_do_quadro(proj, refs, q, estrategia)]
                prefixo = "p1" if _duas(q, estrategia) else "cand"
                for i in seeds:
                    destino = pasta(proj, q.id) / f"{prefixo}-{m}-{i}.png"
                    final = pasta(proj, q.id) / f"cand-{m}-{i}.png"
                    if destino.exists() or final.exists():
                        continue
                    g = (c.grafo_qwen_edit(prompt_quadro(r, q, estrategia), rq, w, h,
                                           hqp.seed(m, i)) if rq
                         else c.grafo_flux(prompt_quadro(r, q, estrategia), w, h,
                                           hqp.seed(m, i)))
                    pedidos.append((g, destino))
            if pedidos:
                if progresso:
                    progresso(f"rodada {rodada + 1}: {len(pedidos)} imagem(ns)")
                c.gerar_lote(pedidos, progresso=progresso)
        # passada 2: corrige o personagem da direita contra a folha dele
        segunda = [(q, i) for q in alvo if _duas(q, estrategia) for i in seeds
                   if not (pasta(proj, q.id) / f"cand-{marcas[q.id]}-{i}.png").exists()]
        if segunda:
            p1s = [pasta(proj, q.id) / f"p1-{marcas[q.id]}-{i}.png" for q, i in segunda]
            with servico.refs_enviadas(c, p1s + [refs[q.personagens[1]] for q, _ in segunda]) as nomes:
                pedidos = []
                for (q, i), p1 in zip(segunda, p1s):
                    w, h = q.tamanho
                    m = marcas[q.id]
                    pedidos.append((c.grafo_qwen_edit(
                        prompt_correcao(r, q), [nomes[p1], nomes[refs[q.personagens[1]]]],
                        w, h, hqp.seed(m + "c", i)), pasta(proj, q.id) / f"cand-{m}-{i}.png"))
                if progresso:
                    progresso(f"rodada {rodada + 1}: segunda passada, {len(pedidos)} imagem(ns)")
                c.gerar_lote(pedidos, progresso=progresso)
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
