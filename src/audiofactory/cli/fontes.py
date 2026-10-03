"""Lotes de ponta a ponta a partir de uma fonte na web: `sutta` (Acesso ao Insight) e `wikisource`."""
from __future__ import annotations

from pathlib import Path

import typer
import yaml
from click.core import ParameterSource

from .. import project as proj_mod
from ..config import settings
from ..store.db import Store
from ._comum import console
from .app import app
from .audiolivro import _etapa_build, _etapa_export, _etapa_run, _etapa_script
from .publicar import _etapa_publish
from .video import _etapa_video


# Etapas do `sutta`, na ordem. `--ate` corta a fila aqui, e o operador retoma
# rodando o mesmo comando: toda etapa a partir do texto é idempotente.
ETAPAS = ("texto", "script", "audio", "video", "publicar")


@app.command()
def sutta(ctx: typer.Context,
          urls: list[str] = typer.Argument(..., metavar="URL...",
              help="endereço em acessoaoinsight.net, ou só o código (ANIV.45)"),
          narrator: str = typer.Option("narrador-v2",
              help="voz do canal (v2 lê mais devagar — é a do sutta)"),
          ate: str = typer.Option("publicar",
              help="para depois desta etapa: " + ", ".join(ETAPAS)),
          privacidade: str = typer.Option("private",
              help="private, unlisted ou public"),
          tags: str = typer.Option(None, help="tags separadas por vírgula"),
          llm: bool = typer.Option(False, "--llm/--no-llm",
              help="Camada 2 do normalizador (Ollama) sobre os spans ambíguos"),
          workers: int = typer.Option(1, help="processos de síntese na GPU"),
          free_ollama: bool = typer.Option(False, "--free-ollama"),
          preset: str = typer.Option("slides", help="preset visual do vídeo"),
          musica: str = typer.Option("musicgen:bansuri",
              help="trilha: ace, musicgen (ex.: 'musicgen:bansuri'), gerada, "
                   "nenhuma ou arquivo"),
          slides_dir: Path = typer.Option(None, help="pasta das imagens de fundo"),
          slides_seg: float = typer.Option(45.0),
          slides_seed: int = typer.Option(None),
          legenda: bool = typer.Option(True, "--legenda/--sem-legenda"),
          gpu: bool = typer.Option(True, "--gpu/--cpu"),
          refazer: bool = typer.Option(False, "--refazer",
              help="baixa a página de novo, reescreve o project.yaml e "
                   "re-renderiza o MP4"),
          sim: bool = typer.Option(False, "--sim",
              help="não pergunta nada (necessário para --privacidade public/unlisted)")):
    """Pipeline inteiro a partir de uma página de sutta do Acesso ao Insight.

    Faz o caminho todo — baixa e limpa a página, monta o script, sintetiza,
    masteriza, renderiza o MP4 e sobe para o YouTube — para cada endereço da
    linha de comando. Um endereço que falha não derruba os seguintes.

    O texto é uma TRADUÇÃO de terceiro sob licença de distribuição gratuita: a
    procedência e os termos são colhidos da própria página e gravados em
    `rights:`, e a descrição do vídeo os reproduz. A decisão de publicar
    continua sendo do operador (ESTADO.md) — por isso o padrão é `private`.
    """
    from ..ingest import acessoaoinsight as ai

    if ate not in ETAPAS:
        console.print(f"[red]etapa desconhecida:[/] {ate} — use {', '.join(ETAPAS)}")
        raise typer.Exit(1)
    if privacidade not in ("private", "unlisted", "public"):
        console.print(f"[red]privacidade inválida:[/] {privacidade}")
        raise typer.Exit(1)
    passos = ETAPAS[:ETAPAS.index(ate) + 1]

    if "publicar" in passos and privacidade != "private" and not sim:
        # Vídeo privado se desfaz apagando; vídeo público já foi visto.
        if not typer.confirm(f"publicar {len(urls)} vídeo(s) como '{privacidade}'?"):
            raise typer.Abort()

    # A GPU é conferida ANTES do laço, pelo mesmo motivo que o acervo de slides e
    # a venv de música: um venv com o torch errado derruba TODOS os endereços da
    # lista, um por um, com a mesma pilha de CUDA. Melhor uma mensagem agora.
    if "audio" in passos:
        problema = _gpu_quebrada()
        if problema:
            console.print(f"[red]GPU indisponível:[/] {problema}")
            console.print("[dim]rode `audio-factory doctor`; se o torch estiver em "
                          "cu124, o reparo está em ESTADO.md (o `chatterbox-tts` "
                          "fixa uma versão que não roda em sm_120)[/]")
            raise typer.Exit(1)

    if "publicar" in passos:
        console.print("[yellow]atenção:[/] a licença do Acesso ao Insight permite "
                      "redistribuir \"contanto que nenhum custo seja cobrado pela "
                      "distribuição ou uso\" — monetizar estes vídeos vai contra "
                      "os termos do texto.")

    falhas: list[tuple[str, str]] = []
    for i, entrada in enumerate(urls, start=1):
        console.rule(f"[bold]{i}/{len(urls)}[/] {entrada}")
        try:
            _um_sutta(entrada, ai, passos, narrator=narrator, privacidade=privacidade,
                      tags=tags, llm=llm, workers=workers, free_ollama=free_ollama,
                      preset=preset, musica=musica,
                      musica_explicita=(ctx.get_parameter_source("musica")
                                        is not ParameterSource.DEFAULT),
                      slides_dir=slides_dir,
                      slides_seg=slides_seg, slides_seed=slides_seed,
                      legenda=legenda, gpu=gpu, refazer=refazer)
        except typer.Abort:
            raise
        except Exception as e:                      # noqa: BLE001 — lote não para
            console.print(f"[red]falhou:[/] {e}")
            falhas.append((entrada, str(e)))

    if falhas:
        console.print(f"\n[red]{len(falhas)} de {len(urls)} falharam:[/]")
        for entrada, erro in falhas:
            console.print(f"  {entrada}: {erro}")
        raise typer.Exit(1)


def _gpu_quebrada() -> str | None:
    """Devolve a queixa se a GPU não serve para sintetizar, ou None se serve.

    Não basta `cuda.is_available()`: o caso que já aconteceu é o torch cu124 sobre
    uma sm_120, que se anuncia disponível e só falha ao lançar o primeiro kernel.
    """
    try:
        import torch
    except Exception as e:                          # noqa: BLE001
        return f"torch não importa ({e})"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__} não enxerga CUDA"
    try:
        x = torch.randn(64, 64, device="cuda")
        (x @ x).sum().item()
    except Exception as e:                          # noqa: BLE001
        return f"torch {torch.__version__} não roda kernels nesta GPU — {e}"
    return None


def _um_sutta(entrada: str, ai, passos: tuple[str, ...], *, narrator: str,
              privacidade: str, tags: str | None, llm: bool, workers: int,
              free_ollama: bool, preset: str, musica: str, musica_explicita: bool,
              slides_dir: Path | None,
              slides_seg: float, slides_seed: int | None, legenda: bool, gpu: bool,
              refazer: bool) -> None:
    """Um sutta, do endereço ao YouTube. Levanta em qualquer etapa que falhar."""
    s = ai.buscar(entrada, cache=settings().cache_dir / "acessoaoinsight",
                  refazer=refazer)
    console.print(f"[green]{s.referencia}[/] · {s.pali} · {len(s.paragrafos)} parágrafos"
                  + (f" · {s.descartados} descartados (aparato)" if s.descartados else ""))

    fonte = settings().books_dir / "suttas" / f"{s.slug}.txt"
    fonte.parent.mkdir(parents=True, exist_ok=True)
    fonte.write_text(s.texto(), encoding="utf-8")

    p = proj_mod.dir_projeto(s.slug)
    if not p.exists() or refazer:
        # Fora do `refazer`, um project.yaml existente é preservado: é onde o
        # operador escreve `cast:` e ajusta `params:` depois de ouvir.
        p = proj_mod.criar(s.slug, fonte, narrator, titulo=s.titulo,
                           rights=s.rights(), extras={"fonte_url": s.url})
        console.print(f"[green]projeto[/] {p}")
    else:
        console.print(f"[dim]projeto existente:[/] {p}")
    if "script" not in passos:
        console.print(f"[dim]parou em `texto`: {fonte}[/]")
        return

    cfg = proj_mod.carregar_config(p)
    _etapa_script(p, llm=llm)
    if "audio" not in passos:
        console.print(f"[dim]parou em `script`: revise {p/'diff.md'}[/]")
        return

    _etapa_run(p, workers=workers, free_ollama=free_ollama)
    pendentes = Store(p / "state.db").stats()["needs_review"]
    if pendentes:
        # `build` recusa capítulo incompleto, então continuar só produziria um
        # erro pior lá na frente. A decisão é humana: ouvir o wav e aprovar.
        raise RuntimeError(f"{pendentes} chunk(s) em needs_review — ouça e resolva "
                           f"com `audio-factory review {s.slug}`")
    _etapa_build(p)
    _etapa_export(p)
    if "video" not in passos:
        console.print(f"[dim]parou em `audio`: {p/'output'}[/]")
        return

    prontos = sorted((p / "output").glob("*.mp4"))
    if prontos and not refazer:
        console.print(f"[dim]MP4 já existe:[/] {prontos[0].name} "
                      "(use --refazer para re-renderizar)")
    else:
        _etapa_video(p, cfg, preset=preset, musica=musica,
                     musica_explicita=musica_explicita,
                     slides_dir=slides_dir, slides_seg=slides_seg,
                     slides_seed=slides_seed, legenda=legenda, gpu=gpu)
    if "publicar" not in passos:
        return

    _etapa_publish(p, cfg, privacidade=privacidade, tags=tags,
                   miniatura_preset=preset)


_ASSETS_CLASSICOS = settings().assets_dir / "slides-classicos"


@app.command()
def wikisource(ctx: typer.Context,
              url: str = typer.Argument(...,
                  help="URL de um capítulo, ou do índice da obra (descobre os "
                       "capítulos sozinho)"),
              narrator: str = typer.Option("narrador-v1",
                  help="voz do narrador"),
              elenco: Path = typer.Option(None,
                  help="YAML personagem->voice_id (sem isso, só narrador+citação genérica)"),
              ate: str = typer.Option("publicar",
                  help="para depois desta etapa: " + ", ".join(ETAPAS)),
              privacidade: str = typer.Option("private",
                  help="private, unlisted ou public"),
              tags: str = typer.Option(None, help="tags separadas por vírgula"),
              llm: bool = typer.Option(True, "--llm/--no-llm",
                  help="Camada 2 do normalizador + atribuição de elenco (Ollama)"),
              workers: int = typer.Option(1, help="processos de síntese na GPU"),
              free_ollama: bool = typer.Option(False, "--free-ollama"),
              preset: str = typer.Option("slides", help="preset visual do vídeo"),
              musica: str = typer.Option("nenhuma",
                  help="trilha: ace, musicgen, gerada, nenhuma ou arquivo — "
                       "nenhuma paleta existente combina com livro infantil"),
              slides_dir: Path = typer.Option(None,
                  help="pasta das imagens de fundo (padrão: assets/slides-classicos, "
                       "não o acervo budista dos suttas)"),
              slides_seg: float = typer.Option(45.0),
              slides_seed: int = typer.Option(None),
              legenda: bool = typer.Option(True, "--legenda/--sem-legenda"),
              gpu: bool = typer.Option(True, "--gpu/--cpu"),
              token: Path = typer.Option(None,
                  help="padrão: config/youtube_token.audiolivros.json"),
              client_secret: Path = typer.Option(None),
              refazer: bool = typer.Option(False, "--refazer"),
              sim: bool = typer.Option(False, "--sim")):
    """Pipeline inteiro a partir de uma obra da Wikisource em português.

    Um projeto por página de capítulo (não um projeto-livro com vários
    vídeos): cada capítulo já tem título próprio e vira um vídeo, o mesmo
    desenho do `sutta`. Dada a URL do índice, descobre e processa todos os
    capítulos; dada a URL de um capítulo só, processa só esse.

    O texto é de domínio público no Brasil — ver `rights:` de cada projeto,
    colhido da própria página. Ao contrário do `sutta`, não há aviso de
    licença bloqueando monetização.
    """
    from ..ingest import wikisource as wk

    if ate not in ETAPAS:
        console.print(f"[red]etapa desconhecida:[/] {ate} — use {', '.join(ETAPAS)}")
        raise typer.Exit(1)
    if privacidade not in ("private", "unlisted", "public"):
        console.print(f"[red]privacidade inválida:[/] {privacidade}")
        raise typer.Exit(1)
    passos = ETAPAS[:ETAPAS.index(ate) + 1]

    elenco_dict: dict[str, str] = {}
    if elenco:
        if not elenco.exists():
            console.print(f"[red]arquivo de elenco não encontrado:[/] {elenco}")
            raise typer.Exit(1)
        elenco_dict = yaml.safe_load(elenco.read_text(encoding="utf-8")) or {}

    capitulos = wk.descobrir_capitulos(url) or [url]
    console.print(f"[dim]{len(capitulos)} capítulo(s) a processar[/]")

    if "publicar" in passos and privacidade != "private" and not sim:
        if not typer.confirm(f"publicar {len(capitulos)} vídeo(s) como '{privacidade}'?"):
            raise typer.Abort()

    if "audio" in passos:
        problema = _gpu_quebrada()
        if problema:
            console.print(f"[red]GPU indisponível:[/] {problema}")
            raise typer.Exit(1)

    falhas: list[tuple[str, str]] = []
    for i, cap_url in enumerate(capitulos, start=1):
        console.rule(f"[bold]{i}/{len(capitulos)}[/] {cap_url}")
        try:
            _um_capitulo(cap_url, wk, passos, narrator=narrator, elenco=elenco_dict,
                        privacidade=privacidade, tags=tags, llm=llm, workers=workers,
                        free_ollama=free_ollama, preset=preset, musica=musica,
                        musica_explicita=(ctx.get_parameter_source("musica")
                                          is not ParameterSource.DEFAULT),
                        slides_dir=slides_dir or _ASSETS_CLASSICOS,
                        slides_seg=slides_seg, slides_seed=slides_seed,
                        legenda=legenda, gpu=gpu, refazer=refazer,
                        token=token, client_secret=client_secret)
        except typer.Abort:
            raise
        except Exception as e:                      # noqa: BLE001 — lote não para
            console.print(f"[red]falhou:[/] {e}")
            falhas.append((cap_url, str(e)))

    if falhas:
        console.print(f"\n[red]{len(falhas)} de {len(capitulos)} falharam:[/]")
        for cap_url, erro in falhas:
            console.print(f"  {cap_url}: {erro}")
        raise typer.Exit(1)


def _cast_do_elenco(elenco: dict[str, str]) -> dict[str, str]:
    """personagem->voice_id (bruto do YAML) -> chave de `Segment.role`->voice_id.

    `_generico` (a voz do papel `citacao`) não leva prefixo; os demais viram
    `personagem_<chave>`, o papel que `narration/elenco.py` atribui.
    """
    cast: dict[str, str] = {}
    for chave, voice_id in elenco.items():
        cast["citacao" if chave == "_generico" else f"personagem_{chave}"] = voice_id
    return cast


def _um_capitulo(entrada: str, wk, passos: tuple[str, ...], *, narrator: str,
                elenco: dict[str, str], privacidade: str, tags: str | None,
                llm: bool, workers: int, free_ollama: bool, preset: str,
                musica: str, musica_explicita: bool, slides_dir: Path | None,
                slides_seg: float, slides_seed: int | None, legenda: bool,
                gpu: bool, refazer: bool, token: Path | None,
                client_secret: Path | None) -> None:
    """Um capítulo, da URL ao YouTube. Levanta em qualquer etapa que falhar."""
    c = wk.buscar(entrada, cache=settings().cache_dir / "wikisource", refazer=refazer)
    console.print(f"[green]{c.titulo_video}[/] · {len(c.paragrafos)} parágrafos")

    fonte = settings().books_dir / "wikisource" / f"{c.slug}.txt"
    fonte.parent.mkdir(parents=True, exist_ok=True)
    fonte.write_text(c.texto(), encoding="utf-8")

    p = proj_mod.dir_projeto(c.slug)
    if not p.exists() or refazer:
        cast = _cast_do_elenco(elenco) if elenco else {}
        p = proj_mod.criar(c.slug, fonte, narrator, titulo=c.titulo_video,
                           rights=c.rights(),
                           extras={"fonte_url": c.url, "cast": cast})
        console.print(f"[green]projeto[/] {p}")
    else:
        console.print(f"[dim]projeto existente:[/] {p}")
    if "script" not in passos:
        console.print(f"[dim]parou em `texto`: {fonte}[/]")
        return

    cfg = proj_mod.carregar_config(p)
    # Capítulo único forçado -- ver o docstring de `montar_script` sobre por
    # que `detectar_capitulos()` fragmentaria esta página em vários pedaços.
    _etapa_script(p, llm=llm, capitulos=[(c.titulo, c.texto())], elenco=elenco)
    if "audio" not in passos:
        console.print(f"[dim]parou em `script`: revise {p/'diff.md'}[/]")
        return

    _etapa_run(p, workers=workers, free_ollama=free_ollama)
    pendentes = Store(p / "state.db").stats()["needs_review"]
    if pendentes:
        raise RuntimeError(f"{pendentes} chunk(s) em needs_review — ouça e resolva "
                           f"com `audio-factory review {c.slug}`")
    _etapa_build(p)
    _etapa_export(p)
    if "video" not in passos:
        console.print(f"[dim]parou em `audio`: {p/'output'}[/]")
        return

    prontos = sorted((p / "output").glob("*.mp4"))
    if prontos and not refazer:
        console.print(f"[dim]MP4 já existe:[/] {prontos[0].name} "
                      "(use --refazer para re-renderizar)")
    else:
        _etapa_video(p, cfg, preset=preset, musica=musica,
                     musica_explicita=musica_explicita,
                     slides_dir=slides_dir, slides_seg=slides_seg,
                     slides_seed=slides_seed, legenda=legenda, gpu=gpu)
    if "publicar" not in passos:
        return

    _etapa_publish(p, cfg, privacidade=privacidade, tags=tags,
                   miniatura_preset=preset, token=token, client_secret=client_secret)
