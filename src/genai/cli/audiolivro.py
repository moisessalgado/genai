"""Audiolivro, do texto ao master: `new`, `script`, `run`, `review`, `build`, `export`, `report`..."""
from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.table import Table

from ..audiolivro import projeto as livro
from ..core import projeto as proj_mod
from ..core.config import settings
from ..core.servicos import invokeai
from ..audiolivro.engines.chatterbox_engine import ChatterboxEngine
from ..audiolivro.pipeline import Runner
from ..audiolivro.qa.verify import Verifier
from ..audiolivro.script.models import Script
from ..core.estado import Store
from ._comum import _proj, console
from .app import app


def _lexicon() -> dict[str, str]:
    lex: dict[str, str] = {}
    for f in sorted(settings().lexicon_dir.glob("*.yaml")):
        lex.update(yaml.safe_load(f.read_text(encoding="utf-8")) or {})
    return lex


@app.command()
def new(slug: str, fonte: Path = typer.Option(..., "--from"),
        narrator: str = "default", titulo: str = typer.Option(None)):
    """Cria um projeto a partir de um arquivo de texto."""
    p = livro.criar(slug, fonte.resolve(), narrator, titulo=titulo)
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
    livro.ingerir(p)
    s = livro.montar_script(p, _lexicon(), max_chars, usar_llm=llm, modelo_llm=modelo,
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
        livro.ingerir(p)
    from ..audiolivro.narration import ortografia

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
    from ..audiolivro.voices import carregar

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
    # O InvokeAI segura a VRAM entre gerações (7 GB medidos depois de um lote
    # FLUX); com 2 workers do TTS isso estoura os 16 GB. Fora do ar, nada a fazer.
    if invokeai.liberar_vram_se_no_ar():
        console.print("[dim]InvokeAI: VRAM liberada para o TTS[/]")

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
    from ..audiolivro.audio.process import chapters_txt, concatenar, duracao, exportar, masterizar

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
    from ..audiolivro.qa.report import coletar, markdown

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
