"""Elenco da HQ: folha-modelo por personagem, com aprovação humana.

Mesmo padrão de `imagem` → `imagem-aprovar`: a máquina gera candidatos, o
operador escolhe. Duas rodadas por personagem:

1. **folha** (FLUX schnell): `elenco/<pid>/cand-*.png` → `aprovado.png`;
2. **limpeza** (Qwen Edit sobre a aprovada): o schnell ignora "no text" e
   inventa caligrafia, selos e cartuchos (achado do spike) — e a ref vai para
   TODO quadro do personagem, então o lixo se copiaria junto. `limpo-*.png` →
   `ref.png`, a referência que os quadros usam.

Aprovar a própria `aprovado.png` como ref é permitido (folha que já saiu limpa).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import projeto as hqp
from . import servico
from .roteiro import Roteiro

LARGURA, ALTURA = 832, 1216  # retrato, corpo inteiro

PROMPT_FOLHA = ("{estilo} Full-body character reference sheet of {ficha}. Single figure "
                "standing upright, front view, arms relaxed, whole body visible from head "
                "to feet, on a plain empty cream paper background, nothing else.")

PROMPT_LIMPEZA = ("Remove every piece of text, letters, calligraphy, signatures, red seal "
                  "stamps, cartouches and frames from Picture 1, and any other figure or "
                  "object besides the main character. Keep the character exactly the same: "
                  "face, hair, ornaments, clothing, colors, pose, proportions and art style. "
                  "Plain empty cream paper background.")


def pasta(proj: Path, pid: str) -> Path:
    return proj / "elenco" / pid


def ref(proj: Path, pid: str) -> Path:
    """A referência final do personagem, usada nos quadros."""
    return pasta(proj, pid) / "ref.png"


def _marca_folha(r: Roteiro, pid: str) -> str:
    return hqp.marca("folha", r.estilo, r.personagens[pid].ficha)


def sincronizar(proj: Path, r: Roteiro) -> None:
    """Alinha o estado com o roteiro: ficha reescrita pede folha nova; folha
    aprovada nova pede limpeza nova."""
    e = hqp.estado(proj)
    e.sincronizar("elenco", {pid: _marca_folha(r, pid) for pid in r.personagens})
    e.sincronizar("limpeza", {pid: hqp.marca(_marca_folha(r, pid),
                                             hqp.marca_arquivo(pasta(proj, pid) / "aprovado.png"))
                              for pid in r.personagens})


def candidatos(proj: Path, pid: str, prefixo: str = "cand") -> list[Path]:
    return sorted(pasta(proj, pid).glob(f"{prefixo}-*.png"))


def gerar_folhas(proj: Path, r: Roteiro, n: int = 4, *, pids: list[str] | None = None,
                 progresso=None) -> dict[str, list[Path]]:
    """Gera (ou reaproveita) `n` candidatos de folha para cada personagem ainda
    sem folha aprovada. Devolve {pid: candidatos}."""
    sincronizar(proj, r)
    e = hqp.estado(proj)
    alvo = [x["alvo"] for x in e.itens("elenco") if x["state"] != "ok"
            and (not pids or x["alvo"] in pids)]
    if not alvo:
        return {}
    c = servico.cliente()
    pedidos, saida = [], {}
    for pid in alvo:
        m = _marca_folha(r, pid)
        prompt = PROMPT_FOLHA.format(estilo=r.estilo, ficha=r.personagens[pid].ficha)
        destinos = [pasta(proj, pid) / f"cand-{m}-{i}.png" for i in range(n)]
        pedidos += [(c.grafo_flux(prompt, LARGURA, ALTURA, hqp.seed(m, i)), d)
                    for i, d in enumerate(destinos) if not d.exists()]
        saida[pid] = destinos
    if pedidos:
        c.gerar_lote(pedidos, progresso=progresso)
    for pid in alvo:
        e.revisar("elenco", pid, f"escolha a folha: genai hq elenco-aprovar {proj.name} "
                                 f"{pid} <arquivo>")
    return saida


def limpar(proj: Path, r: Roteiro, n: int = 2, *, pids: list[str] | None = None,
           progresso=None) -> dict[str, list[Path]]:
    """Candidatos de folha limpa (Qwen Edit) para cada personagem com folha
    aprovada e ainda sem ref."""
    sincronizar(proj, r)
    e = hqp.estado(proj)
    alvo = [pid for pid in r.personagens
            if (not pids or pid in pids) and e.escolhido("elenco", pid)
            and e.item("limpeza", pid)["state"] != "ok"]
    if not alvo:
        return {}
    c = servico.cliente()
    saida: dict[str, list[Path]] = {}
    with servico.refs_enviadas(c, [pasta(proj, p) / "aprovado.png" for p in alvo]) as nomes:
        pedidos = []
        for pid in alvo:
            aprovado = pasta(proj, pid) / "aprovado.png"
            m = hqp.marca("limpeza", hqp.marca_arquivo(aprovado), PROMPT_LIMPEZA)
            destinos = [pasta(proj, pid) / f"limpo-{m}-{i}.png" for i in range(n)]
            pedidos += [(c.grafo_qwen_edit(PROMPT_LIMPEZA, [nomes[aprovado]], LARGURA, ALTURA,
                                           hqp.seed(m, i)), d)
                        for i, d in enumerate(destinos) if not d.exists()]
            saida[pid] = destinos
        if pedidos:
            c.gerar_lote(pedidos, progresso=progresso)
    for pid in alvo:
        e.revisar("limpeza", pid, f"escolha a ref limpa: genai hq elenco-aprovar "
                                  f"{proj.name} {pid} <arquivo>")
    return saida


def aprovar(proj: Path, r: Roteiro, pid: str, arquivo: Path) -> tuple[str, Path]:
    """Aprova um candidato. `cand-*` vira a folha (`aprovado.png`, e a limpeza
    volta para a fila); qualquer outra imagem (`limpo-*`, a própria
    `aprovado.png`, um retoque feito no Canvas) vira a ref final.
    Devolve (etapa, destino)."""
    if pid not in r.personagens:
        raise ValueError(f"personagem fora do roteiro: {pid} "
                         f"(elenco: {', '.join(r.personagens)})")
    arquivo = Path(arquivo)
    if not arquivo.exists():
        raise FileNotFoundError(arquivo)
    sincronizar(proj, r)
    e = hqp.estado(proj)
    p = pasta(proj, pid)
    p.mkdir(parents=True, exist_ok=True)
    if arquivo.name.startswith("cand-"):
        destino = p / "aprovado.png"
        shutil.copy(arquivo, destino)
        e.concluir("elenco", pid, destino, por="humano")
        sincronizar(proj, r)  # folha nova -> limpeza pendente
        return "folha", destino
    if not e.escolhido("elenco", pid):
        raise ValueError(f"{pid}: aprove primeiro uma folha (cand-*.png)")
    destino = ref(proj, pid)
    if arquivo.resolve() != destino.resolve():
        shutil.copy(arquivo, destino)
    e.concluir("limpeza", pid, destino, por="humano")
    return "ref", destino


def refs_prontas(proj: Path, r: Roteiro) -> dict[str, Path]:
    """{pid: ref.png} — erro listando quem falta, se faltar alguém."""
    sincronizar(proj, r)
    e = hqp.estado(proj)
    faltam = [pid for pid in r.personagens if not e.escolhido("limpeza", pid)]
    if faltam:
        raise ValueError("sem ref aprovada: " + ", ".join(faltam)
                         + " — rode `genai hq elenco` / `hq limpar` e aprove")
    return {pid: ref(proj, pid) for pid in r.personagens}
