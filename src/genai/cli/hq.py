"""HQ: `genai hq ...` — roteiro, elenco, quadros, páginas, export e motion comic.

Fluxo, cada passo retomável (o estado fica no state.db do projeto):

    genai hq novo <slug> --titulo "..." --from texto.txt --narrator narrador-v2
    genai hq roteiro <slug>              # LLM rascunha roteiro.yaml (edite à vontade)
    genai hq elenco <slug>               # folhas-modelo (FLUX) + folha de contato
    genai hq elenco-aprovar <slug> <personagem> <cand-*.png>
    genai hq limpar <slug>               # tira texto/selos da folha (Qwen Edit)
    genai hq elenco-aprovar <slug> <personagem> <limpo-*.png>
    genai hq quadros <slug>              # Qwen Edit + refs, QA escolhe
    genai hq escolher <slug> <quadro> <png>   # troca a escolha, ou resolve needs_review
    genai hq paginas <slug>              # diagrama e letreira
    genai hq exportar <slug>             # PDF, CBZ e webtoon
    genai hq motion <slug>               # narra e renderiza o MP4 (depois: genai publish)
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from ._comum import _proj, console
from .app import app

hq_app = typer.Typer(help="HQ: roteiro, elenco, quadros, páginas, export e motion comic",
                     no_args_is_help=True)
app.add_typer(hq_app, name="hq")


def _roteiro(p: Path):
    from ..hq import roteiro
    try:
        return roteiro.carregar(p)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    except ValueError as e:  # ValidationError do pydantic
        console.print(f"[red]roteiro.yaml inválido:[/]\n{e}")
        raise typer.Exit(1)


def _falha(e: Exception) -> None:
    console.print(f"[red]{e}[/]")
    raise typer.Exit(1)


def _ids(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out: list[int] = []
    for parte in spec.split(","):
        if "-" in parte:
            a, b = parte.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(parte))
    return out


@hq_app.command("novo")
def novo(slug: str, titulo: str = typer.Option(..., help="título da HQ"),
         fonte: Path = typer.Option(None, "--from", help="texto de origem (.txt)"),
         narrator: str = typer.Option(None, help="voz dos recordatórios no motion comic")):
    """Cria projects/<slug>/ com project.yaml (preencha `rights` antes de publicar)."""
    from ..hq import projeto as hqp
    if fonte is not None and not fonte.exists():
        _falha(FileNotFoundError(f"texto não encontrado: {fonte}"))
    try:
        p = hqp.criar(slug, titulo=titulo, fonte=fonte, narrator=narrator)
    except FileExistsError as e:
        _falha(e)
    console.print(f"[green]{p}[/] — próximo: `genai hq roteiro {slug}` "
                  "(ou escreva roteiro.yaml à mão)")


@hq_app.command("roteiro")
def roteiro_cmd(slug: str, paginas: int = typer.Option(1, help="número de páginas"),
                estilo: str = typer.Option(None, help="bloco de estilo (inglês); padrão: ukiyo-e"),
                ancora_estilo: str = typer.Option(None,
                    help="frase curta do estilo para o QA (CLIP); com --estilo próprio, "
                         "sem âncora o QA não julga estilo"),
                forcar: bool = typer.Option(False, "--forcar",
                    help="sobrescreve um roteiro.yaml existente (guarda o anterior)")):
    """O LLM rascunha o roteiro.yaml a partir do texto de origem do projeto."""
    import time

    from ..core.projeto import carregar_config
    from ..hq import roteiro, roteiro_llm

    p = _proj(slug)
    arq = p / roteiro.ARQUIVO
    if arq.exists() and not forcar:
        _falha(FileExistsError(f"{arq} já existe (edite-o, ou use --forcar)"))
    cfg = carregar_config(p)
    if not cfg.get("fonte") or not (p / cfg["fonte"]).exists():
        _falha(FileNotFoundError("projeto sem texto de origem (`hq novo --from`)"))
    fonte = (p / cfg["fonte"]).read_text(encoding="utf-8")
    t0 = time.time()
    with console.status("roteiro…") as st:
        try:
            r = roteiro_llm.gerar(
                fonte, slug=slug, paginas=paginas, titulo=cfg.get("titulo"),
                estilo=estilo or roteiro_llm.ESTILO_PADRAO,
                ancora_estilo=ancora_estilo if (ancora_estilo or estilo)
                else roteiro_llm.ANCORA_PADRAO,
                progresso=st.update)
        except (ValueError, OSError) as e:
            _falha(e)
    if arq.exists():
        arq.replace(arq.with_name("roteiro.anterior.yaml"))
    roteiro.salvar(r, p)
    console.print(f"[green]{arq}[/] — {len(r.personagens)} personagens, {len(r.quadros)} "
                  f"quadros, {len(r.paginas)} página(s), {time.time() - t0:.0f} s")
    console.print("[dim]revise o roteiro (fichas em inglês, cena, falas) antes do elenco[/]")


@hq_app.command("elenco")
def elenco_cmd(slug: str, n: int = typer.Option(4, help="candidatos por personagem"),
               personagem: list[str] = typer.Option(None, "--personagem",
                   help="só estes (repetível)")):
    """Folhas-modelo (FLUX) de quem ainda não tem folha aprovada."""
    from ..hq import contato, elenco
    p = _proj(slug)
    r = _roteiro(p)
    with console.status("folhas-modelo…") as st:
        try:
            cands = elenco.gerar_folhas(p, r, n=n, pids=personagem or None, progresso=st.update)
        except (RuntimeError, ValueError) as e:
            _falha(e)
    if not cands:
        console.print("[dim]todas as folhas já aprovadas[/]")
        return
    folha = contato.montar([c for cs in cands.values() for c in cs],
                           p / "elenco" / "contato.png", por_linha=n)
    console.print(f"folha de contato: [cyan]{folha}[/]")
    console.print(f"aprove com `genai hq elenco-aprovar {slug} <personagem> <arquivo>`")


@hq_app.command("elenco-aprovar")
def elenco_aprovar(slug: str, personagem: str, arquivo: Path):
    """Aprova folha (cand-*.png) ou ref final (limpo-*.png, aprovado.png, retoque).

    cand-*.png vira a folha aprovada; qualquer outra imagem vira a ref final
    usada nos quadros."""
    from ..hq import elenco
    p = _proj(slug)
    r = _roteiro(p)
    try:
        etapa, destino = elenco.aprovar(p, r, personagem, arquivo)
    except (ValueError, FileNotFoundError) as e:
        _falha(e)
    console.print(f"[green]{etapa}[/] de {personagem}: {destino}")
    if etapa == "folha":
        console.print(f"[dim]próximo: `genai hq limpar {slug}`[/]")


@hq_app.command("limpar")
def limpar(slug: str, n: int = typer.Option(2, help="candidatos por personagem"),
           personagem: list[str] = typer.Option(None, "--personagem")):
    """Tira caligrafia, selos e cartuchos da folha aprovada (Qwen Edit)."""
    from ..hq import contato, elenco
    p = _proj(slug)
    r = _roteiro(p)
    with console.status("limpando folhas…") as st:
        try:
            cands = elenco.limpar(p, r, n=n, pids=personagem or None, progresso=st.update)
        except (RuntimeError, ValueError) as e:
            _falha(e)
    if not cands:
        console.print("[dim]nada a limpar (sem folha aprovada, ou ref já aprovada)[/]")
        return
    imgs = []
    for pid, cs in cands.items():
        imgs += [elenco.pasta(p, pid) / "aprovado.png", *cs]
    folha = contato.montar(imgs, p / "elenco" / "contato-limpo.png", por_linha=n + 1)
    console.print(f"folha de contato: [cyan]{folha}[/]")


@hq_app.command("quadros")
def quadros_cmd(slug: str, n: int = typer.Option(2, help="candidatos por quadro e rodada"),
                ids: str = typer.Option(None, help="só estes quadros, ex.: 1,3-5"),
                rodadas: int = typer.Option(2, help="rodadas se o QA reprovar todos"),
                refazer: bool = typer.Option(False, "--refazer",
                    help="gera candidatos novos para --ids mesmo já escolhidos")):
    """Gera os quadros com as refs do elenco; o QA escolhe o melhor de cada um.

    Quadros com dois personagens seguem a `estrategia` do roteiro.yaml (multi,
    duas-passadas ou composta)."""
    from ..hq import contato, qa, quadros
    p = _proj(slug)
    r = _roteiro(p)
    alvo = _ids(ids)
    if refazer:
        if not alvo:
            _falha(ValueError("--refazer precisa de --ids"))
        quadros.refazer(p, r, alvo)
    with console.status("quadros…") as st:
        try:
            rel = quadros.gerar(p, r, n=n, ids=alvo, rodadas=rodadas, progresso=st.update)
        except (RuntimeError, ValueError) as e:
            _falha(e)
    if not rel:
        console.print("[dim]nada pendente — todos os quadros têm escolha[/]")
    for qid, notas in rel.items():
        cands = [quadros.pasta(p, qid) / k for k in notas]
        contato.montar(cands, quadros.pasta(p, qid) / "contato.png", altura=360,
                       notas={k: qa.rotulo(v) for k, v in notas.items()})
    _status(p, r)


@hq_app.command("escolher")
def escolher(slug: str, quadro: int, arquivo: Path):
    """Escolha humana de um quadro (um candidato, ou um retoque do Canvas)."""
    from ..hq import quadros
    p = _proj(slug)
    r = _roteiro(p)
    try:
        destino = quadros.escolher(p, r, quadro, arquivo)
    except (ValueError, FileNotFoundError) as e:
        _falha(e)
    console.print(f"[green]quadro {quadro}[/]: {destino}")


def _status(p: Path, r) -> None:
    from ..hq import projeto as hqp
    e = hqp.estado(p)
    t = Table(title=f"HQ {p.name}")
    for c in ("item", "estado", "por", "tent.", "escolhido / pendência"):
        t.add_column(c)
    cor = {"ok": "green", "needs_review": "yellow", "pending": "dim", "running": "cyan"}
    for tipo in ("elenco", "limpeza", "quadro"):
        for i in e.itens(tipo):
            info = Path(i["escolhido"]).name if i["state"] == "ok" else (i["error"] or "")
            t.add_row(i["item_id"], f"[{cor[i['state']]}]{i['state']}[/]", i["por"] or "",
                      str(i["attempts"]), info)
    console.print(t)


@hq_app.command("status")
def status(slug: str):
    """Onde cada personagem e quadro está."""
    p = _proj(slug)
    r = _roteiro(p)
    from ..hq import elenco, quadros
    elenco.sincronizar(p, r)
    try:
        quadros.sincronizar(p, r)
    except ValueError:
        pass  # sem refs ainda: os quadros nem começaram
    _status(p, r)


@hq_app.command("paginas")
def paginas(slug: str):
    """Diagrama e letreira as páginas em paginas/ (PNG no dpi do formato)."""
    from ..hq import layout, quadros
    p = _proj(slug)
    r = _roteiro(p)
    try:
        artes = quadros.escolhidos(p, r)
    except ValueError as e:
        _falha(e)
    with console.status("páginas…") as st:
        pags = layout.paginas(p, r, artes, progresso=st.update)
    for pg in pags:
        console.print(f"[green]{pg}[/]")


@hq_app.command("exportar")
def exportar_cmd(slug: str, formatos: str = typer.Option("pdf,cbz,webtoon",
                     help="pdf, cbz, webtoon — separados por vírgula")):
    """PDF e CBZ (das páginas) e webtoon (quadros re-letreirados a 800 px)."""
    from ..core.projeto import carregar_config
    from ..hq import exportar, layout, quadros
    p = _proj(slug)
    r = _roteiro(p)
    pedidos = {f.strip() for f in formatos.split(",") if f.strip()}
    desconhecidos = pedidos - {"pdf", "cbz", "webtoon"}
    if desconhecidos:
        _falha(ValueError(f"formato desconhecido: {', '.join(sorted(desconhecidos))}"))
    try:
        artes = quadros.escolhidos(p, r)
    except ValueError as e:
        _falha(e)
    nome = exportar.nome_de_arquivo(r.titulo)
    out = p / "output"
    autoria = (carregar_config(p).get("rights") or {}).get("autor")
    with console.status("exportando…") as st:
        pags = sorted((p / "paginas").glob("pagina-*.png"))
        if pedidos & {"pdf", "cbz"} and len(pags) != len(r.paginas):
            st.update("páginas…")
            pags = layout.paginas(p, r, artes)
        if "pdf" in pedidos:
            console.print(f"[green]{exportar.pdf(pags, out / f'{nome}.pdf', r.formato.dpi, r.titulo)}[/]")
        if "cbz" in pedidos:
            console.print(f"[green]{exportar.cbz(pags, out / f'{nome}.cbz', r, autoria)}[/]")
        if "webtoon" in pedidos:
            st.update("webtoon…")
            fatias = exportar.webtoon(r, artes, out / "webtoon")
            console.print(f"[green]{out / 'webtoon'}[/] ({len(fatias)} fatias)")


@hq_app.command("motion")
def motion_cmd(slug: str, gpu: bool = typer.Option(True, "--gpu/--cpu"),
               no_qa: bool = typer.Option(False, "--no-qa", help="pula o QA da narração"),
               nome: str = typer.Option(None, help="nome do MP4 (padrão: o título)")):
    """Narra os textos (TTS do audiolivro) e renderiza o motion comic em output/."""
    from ..hq import motion, quadros
    from .audiolivro import _etapa_run
    p = _proj(slug)
    r = _roteiro(p)
    try:
        artes = quadros.escolhidos(p, r)
        motion.montar_script(p, r)
    except ValueError as e:
        _falha(e)
    _etapa_run(p, no_qa=no_qa)
    with console.status("renderizando…"):
        try:
            mp4 = motion.renderizar(p, r, artes, gpu=gpu, nome=nome)
        except RuntimeError as e:
            _falha(e)
    console.print(f"[green]{mp4}[/] — publique com `genai publish {slug}`")
