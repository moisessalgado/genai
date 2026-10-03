"""CLI do audio-factory (TDD 13). Exposta no dia a dia como `iam voice`."""
from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from click.core import ParameterSource
from rich.table import Table

from .. import project as proj_mod
from ..config import settings
from ..engines.chatterbox_engine import ChatterboxEngine
from ..pipeline import Runner
from ..qa.verify import Verifier
from ..script.models import Script
from ..store.db import Store
from ._comum import _proj, console
from .app import app
from .publicar import _etapa_publish
from .video import _etapa_video
from . import voz  # noqa: F401  (registra os comandos)
from . import video  # noqa: F401  (registra os comandos)
from . import publicar  # noqa: F401  (registra os comandos)
from . import sistema  # noqa: F401  (registra os comandos)


def _lexicon() -> dict[str, str]:
    lex: dict[str, str] = {}
    for f in sorted(settings().lexicon_dir.glob("*.yaml")):
        lex.update(yaml.safe_load(f.read_text(encoding="utf-8")) or {})
    return lex


@app.command()
def new(slug: str, fonte: Path = typer.Option(..., "--from"),
        narrator: str = "default", titulo: str = typer.Option(None)):
    """Cria um projeto a partir de um arquivo de texto."""
    p = proj_mod.criar(slug, fonte.resolve(), narrator, titulo=titulo)
    console.print(f"[green]projeto criado[/] {p}")
    console.print("[dim]opcional: preencha `rights:` em project.yaml para registrar "
                  "a procedência do texto[/]")


@app.command()
def script(slug: str, max_chars: int = 300,
           llm: bool = typer.Option(False, "--llm/--no-llm",
                                    help="Camada 2: resolve spans ambíguos via Ollama"),
           modelo: str = typer.Option(None, help="modelo do Ollama (padrão: gemma4:12b)")):
    """Ingere, normaliza e gera script.json + diff.md."""
    _etapa_script(_proj(slug), max_chars, llm, modelo)


def _etapa_script(p: Path, max_chars: int = 300, llm: bool = False,
                  modelo: str | None = None,
                  capitulos: list[tuple[str, str]] | None = None,
                  elenco: dict[str, str] | None = None) -> None:
    proj_mod.ingerir(p)
    s = proj_mod.montar_script(p, _lexicon(), max_chars, usar_llm=llm, modelo_llm=modelo,
                               capitulos=capitulos, elenco=elenco)
    n_seg = sum(len(c.segments) for c in s.chapters)
    console.print(f"[green]script.json[/] {len(s.chapters)} capítulos, {n_seg} segmentos, "
                  f"{s.total_chars} caracteres")
    console.print(f"revise o diff: {p/'diff.md'}")


@app.command(name="lexico-antigo")
def lexico_antigo(slug: str,
                  saida: Path = typer.Option(
                      None, help="arquivo lexicon/*.yaml de destino "
                      "(padrão: lexicon/pt-BR.ortografia-1943.yaml)"),
                  modelo: str = typer.Option(None, help="modelo do Ollama (padrão: gemma4:12b)")):
    """Varre clean.txt em busca de ortografia pré-1943 e propõe entradas de léxico.

    Roda a Camada 2 (Ollama) palavra por palavra, com o mesmo validador
    conservador da atribuição de elenco (`narration/ortografia.py`): só entra no
    léxico o que sobreviver. Nada é aplicado ao texto aqui — o arquivo gerado é
    lexicon/*.yaml, revisável a mão antes do próximo `iam voice script` pegá-lo.
    """
    p = _proj(slug)
    caminho_texto = p / "clean.txt"
    if not caminho_texto.exists():
        proj_mod.ingerir(p)
    from ..narration import ortografia

    destino = saida or (settings().lexicon_dir / "pt-BR.ortografia-1943.yaml")
    existente: dict[str, str] = {}
    if destino.exists():
        existente = yaml.safe_load(destino.read_text(encoding="utf-8")) or {}

    cache_path = p / "ortografia_cache.json"
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    kwargs = {"modelo": modelo} if modelo else {}
    with console.status("varrendo vocabulário…") as st:
        def prog(i, total, palavra):
            st.update(f"{i}/{total} · {palavra}")
        achadas = ortografia.descobrir(caminho_texto.read_text(encoding="utf-8"), cache=cache,
                                       progresso=prog, **kwargs)

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    novas = {k: v for k, v in achadas.items() if k not in existente}
    if not novas:
        console.print("[dim]nenhuma palavra nova — léxico já cobre o vocabulário deste texto[/]")
        return
    existente.update(novas)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(yaml.safe_dump(existente, allow_unicode=True, sort_keys=True),
                       encoding="utf-8")
    console.print(f"[green]{len(novas)} entradas novas[/] em {destino}")
    for k, v in sorted(novas.items()):
        console.print(f"  {k} → {v}")
    console.print("[dim]revise antes de rodar `iam voice script` de novo[/]")


@app.command()
def run(slug: str, chapters: str = typer.Option(None, help="ex.: 1,3-5"),
        no_qa: bool = typer.Option(False, "--no-qa"),
        voice: Path = typer.Option(None, help="WAV de referência (sobrepõe o narrator)"),
        ptbr_pack: bool = typer.Option(True),
        workers: int = typer.Option(1, help="processos de síntese em paralelo na GPU"),
        free_ollama: bool = typer.Option(False, "--free-ollama",
                                         help="descarrega os modelos do Ollama antes de sintetizar")):
    """Sintetiza os chunks pendentes. Retomar é o comportamento padrão."""
    _etapa_run(_proj(slug), chapters=chapters, no_qa=no_qa, voice=voice,
               ptbr_pack=ptbr_pack, workers=workers, free_ollama=free_ollama)


def _etapa_run(p: Path, *, chapters: str | None = None, no_qa: bool = False,
               voice: Path | None = None, ptbr_pack: bool = True,
               workers: int = 1, free_ollama: bool = False) -> None:
    s = Script.load(p / "script.json")
    from ..voices import carregar

    def _ref(vid: str) -> Path | None:
        if vid in ("default", None):
            return None
        try:
            return carregar(settings().raiz, vid).referencia
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]voz {vid}:[/] {e}")
            raise typer.Exit(1)

    ref = voice or _ref(s.voice_id)
    # cada papel do elenco resolve para a sua própria referência
    refs = {vid: r for vid in {s.voice_id, *s.cast.values()} if (r := _ref(vid))}
    if s.cast:
        console.print(f"elenco: {s.voice_id} (narrador) + " +
                      ", ".join(f"{k}→{v}" for k, v in s.cast.items()))
    if free_ollama:
        _liberar_ollama()

    fabrica = lambda: ChatterboxEngine(s.params, use_ptbr_pack=ptbr_pack)
    runner = Runner(p, s, fabrica(), None if no_qa else Verifier(), voice_ref=ref,
                    refs_por_voz=refs, engine_factory=fabrica,
                    verifier_factory=Verifier)
    novos, limpos = runner.sync()
    console.print(f"fila: +{novos} novos, {limpos} obsoletos/recuperados")

    caps = _parse_chapters(chapters)
    if workers > 1:
        console.print(f"[dim]{workers} workers — cada um carrega o seu modelo "
                      f"(~3,5 GB de VRAM cada)[/]")
    with console.status("sintetizando…") as st:
        def prog(cid, d):
            st.update(f"{cid} {'ok' if d.aceito else '[red]review[/]'}")
        r = runner.run(caps, progress=prog, workers=workers)
    if not r["processados"]:
        # Retomar sem nada pendente é o caso NORMAL do `sutta`, que re-roda o
        # pipeline inteiro. Sem esta saída, a linha de métricas anunciava
        # "RTF None · Nones de áudio" toda vez.
        console.print("[dim]nada pendente — a fila já está completa[/]")
    else:
        console.print(f"[green]{r['ok']} ok[/] · [yellow]{r['review']} para revisão[/] · "
                      f"RTF {r.get('rtf')} · {r.get('audio_s')}s de áudio")
    if r.get("review"):
        console.print(f"[dim]revise com `iam voice review {p.name}`[/]")


@app.command()
def status(slug: str):
    """Progresso, falhas e CER médio."""
    db = Store(_proj(slug) / "state.db")
    s = db.stats()
    t = Table("estado", "chunks")
    for k in ("ok", "pending", "running", "needs_review"):
        t.add_row(k, str(s[k]))
    t.add_row("[bold]total", f"[bold]{s['total']}")
    console.print(t)
    console.print(f"áudio pronto: {s['audio_s']/60:.1f} min · "
                  f"CER médio: {s['cer_medio'] if s['cer_medio'] is None else round(s['cer_medio'],4)}")
    if s.get("speaker_medio") is not None:
        console.print(f"identidade da voz: média {s['speaker_medio']:.3f} · "
                      f"pior chunk {s['speaker_min']:.3f} (limiar 0,88)")


@app.command()
def review(slug: str,
           aprovar: str = typer.Option(None, "--aprovar", metavar="CHUNK_ID",
                                       help="aceita o chunk como está, depois de ouvir"),
           regerar: str = typer.Option(None, "--regerar", metavar="CHUNK_ID",
                                       help="devolve o chunk à fila, para outra seed")):
    """Lista — ou resolve — os chunks que precisam de revisão humana."""
    db = Store(_proj(slug) / "state.db")

    if aprovar:
        if db.approve(aprovar):
            console.print(f"[green]aprovado[/] {aprovar} — `build` já pode montar")
        else:
            console.print(f"[red]não está em needs_review:[/] {aprovar}")
            raise typer.Exit(1)
        return
    if regerar:
        db.requeue(regerar)
        console.print(f"[green]de volta à fila[/] {regerar} — rode `run` de novo")
        return

    rows = db.needs_review()
    if not rows:
        console.print("[green]nenhum chunk pendente de revisão[/]")
        return
    for r in rows:
        console.print(f"\n[bold]{r['chunk_id']}[/] — {r['error']}")
        console.print(f"  esperado: {r['text']}")
        console.print(f"  ouvido  : {r['transcript']}")
        console.print(f"  wav     : {r['wav_path']}")
    console.print(f"\n[dim]ouça o wav e decida: `review {slug} --aprovar <chunk_id>` "
                  f"ou `--regerar <chunk_id>`[/]")


@app.command()
def build(slug: str):
    """Monta os capítulos a partir dos chunks aprovados."""
    _etapa_build(_proj(slug))


def _etapa_build(p: Path) -> None:
    s = Script.load(p / "script.json")
    engine = ChatterboxEngine(s.params)
    engine.sample_rate = 24000
    runner = Runner(p, s, engine)
    for ch in runner.store.chapters():
        destino = runner.build_chapter(ch)
        console.print(f"cap {ch}: {destino or '[yellow]incompleto — faltam chunks[/]'}")


@app.command()
def export(slug: str, formato: str = typer.Option("mp3", help="mp3, aac, flac, wav — separados por vírgula"),
           juntar: bool = typer.Option(True, "--juntar/--sem-juntar",
                                       help="também gera o arquivo único contínuo"),
           bitrate: str = "192k"):
    """Masteriza, exporta e escreve o chapters.txt."""
    _etapa_export(_proj(slug), formato=formato, juntar=juntar, bitrate=bitrate)


def _etapa_export(p: Path, *, formato: str = "mp3", juntar: bool = True,
                  bitrate: str = "192k") -> None:
    from ..audio.process import chapters_txt, concatenar, duracao, exportar, masterizar

    cfg = proj_mod.carregar_config(p)
    s = Script.load(p / "script.json")
    titulos = {c.idx: c.title for c in s.chapters}
    # `rights` é registro de procedência, não autorização: o export nunca é
    # bloqueado por ele. A decisão sobre o que publicar é do operador.
    status = (cfg.get("rights") or {}).get("status")
    console.print(f"[dim]direitos declarados: {status or '(não preenchido)'}[/]")

    out = p / "output"
    out.mkdir(parents=True, exist_ok=True)
    wavs = sorted((p / "audio" / "chapters").glob("ch*.wav"))
    if not wavs:
        console.print("[red]nenhum capítulo montado[/] — rode `build` antes")
        raise typer.Exit(1)
    formatos = [f.strip() for f in formato.split(",") if f.strip()]

    masters: list[Path] = []
    duracoes: list[tuple[str, float]] = []
    for n, wav in enumerate(wavs, start=1):
        cap = int(wav.stem[2:]) if wav.stem[2:].isdigit() else n
        master = out / f"{wav.stem}-master.wav"
        masterizar(wav, master)
        masters.append(master)
        duracoes.append((titulos.get(cap, wav.stem), duracao(master)))
        meta = {"title": titulos.get(cap, wav.stem), "album": cfg["titulo"],
                "track": str(cap), "artist": cfg.get("narrator", "")}
        for fmt in formatos:
            console.print(f"[green]{exportar(master, out / f'{wav.stem}.{fmt}', fmt, meta, bitrate)}[/]")

    # Os carimbos são cumulativos e só valem contra o arquivo contínuo.
    (out / "chapters.txt").write_text(chapters_txt(duracoes), encoding="utf-8")
    total = sum(d for _, d in duracoes)
    console.print(f"[green]{out/'chapters.txt'}[/] — {len(duracoes)} capítulos, "
                  f"{total/60:.1f} min")

    if juntar and len(masters) > 1:
        completo = out / f"{p.name}-completo.wav"
        concatenar(masters, completo)
        meta = {"title": cfg["titulo"], "album": cfg["titulo"],
                "artist": cfg.get("narrator", "")}
        for fmt in formatos:
            console.print(f"[green]{exportar(completo, out / f'{p.name}-completo.{fmt}', fmt, meta, bitrate)}[/]")


@app.command()
def report(slug: str, medir: bool = typer.Option(True, "--medir/--sem-medir",
                                                 help="medir loudness dos masters (usa ffmpeg)")):
    """Confronta as métricas do projeto com os alvos objetivos do TDD."""
    from ..qa.report import coletar, markdown

    p = _proj(slug)
    cfg = proj_mod.carregar_config(p)
    db = Store(p / "state.db")
    metricas = coletar(p, db, medir_audio=medir)
    review = db.needs_review()

    t = Table("métrica", "medido", "alvo", "situação", "detalhe")
    for m in metricas:
        if m.valor is None:
            val = "—"
        elif "RTF" in m.nome:
            val = f"{m.valor:.3f}"
        elif "Loudness" in m.nome:
            val = f"{m.valor:.1f} LUFS"
        else:
            val = f"{m.valor*100:.1f}%"
        cor = {True: "green", False: "red", None: "dim"}[m.ok]
        t.add_row(m.nome, val, m.alvo, f"[{cor}]{m.marca}[/]", m.detalhe)
    console.print(t)

    destino = p / "qa" / "report.md"
    destino.write_text(markdown(cfg["titulo"], metricas, review), encoding="utf-8")
    console.print(f"[green]{destino}[/]")
    if any(m.ok is False for m in metricas):
        raise typer.Exit(1)


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


def _liberar_ollama() -> None:
    """Descarrega os modelos do Ollama da GPU antes da sintese.

    Nao e obrigatorio -- a Fase 0 mediu 3,5 GB de pico para o TTS em 16 GB, e o
    gemma4:12b cabe junto. Vira util com varios workers, ou se a GPU estiver
    dividida com outra coisa.
    """
    import shutil
    import subprocess

    if not shutil.which("ollama"):
        console.print("[yellow]ollama não encontrado — nada a liberar[/]")
        return
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    modelos = [l.split()[0] for l in ps.stdout.splitlines()[1:] if l.strip()]
    for m in modelos:
        subprocess.run(["ollama", "stop", m], capture_output=True, text=True)
    console.print(f"[dim]ollama: {len(modelos) or 'nenhum'} modelo(s) descarregado(s)[/]")


def _parse_chapters(spec: str | None) -> list[int] | None:
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


if __name__ == "__main__":
    app()
