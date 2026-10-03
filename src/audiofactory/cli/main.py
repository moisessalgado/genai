"""CLI do audio-factory (TDD 13). Exposta no dia a dia como `iam voice`."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
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
from . import voz  # noqa: F401  (registra os comandos)


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


def _nome_de_arquivo(titulo: str) -> str:
    """Titulo do projeto -> nome de arquivo publicavel, CURTO.

    Mantem espacos e acentos: o YouTube usa o nome do arquivo como titulo
    sugerido, e "Dhammacakkappavattana Sutta" le melhor que um slug com hifens.

    Corta no primeiro travessao ou dois-pontos de proposito. O motivo nao e
    estetico: o player do operador desenha o nome do arquivo sobre o video ao
    iniciar, e um nome de 68 caracteres atravessa o rodape e COLIDE com a
    legenda queimada. O nome completo do projeto vira o subtitulo la no YouTube;
    aqui basta a parte que identifica.
    """
    limpo = "".join(" " if c in '/\\:*?"<>|' else c for c in titulo)
    for corte in ("—", "–", " - ", ":"):
        if corte in limpo:
            limpo = limpo.split(corte)[0]
            break
    return " ".join(limpo.split()).strip(". ") or "video"


@app.command()
def video(ctx: typer.Context, slug: str,
          preset: str = typer.Option("slides",
              help="slides, sincronizado, ondas, espectro, estatico ou gradiente"),
          musica: str = typer.Option("ace",
              help="'ace' ou 'musicgen' (modelos dedicados; 'ace:sobrio' / "
                   "'musicgen:sobrio' escolhem a paleta), 'gerada' (sintetizada), "
                   "'nenhuma', ou caminho de um arquivo"),
          capa: Path = typer.Option(None, help="imagem fixa de fundo (sobrepõe o preset)"),
          slides_dir: Path = typer.Option(None,
              help="pasta das imagens do preset slides (padrão: assets/slides)"),
          slides_seg: float = typer.Option(45.0,
              help="segundos que cada imagem fica na tela"),
          slides_seed: int = typer.Option(None,
              help="fixa o sorteio das imagens; sem isso cada render sorteia de novo"),
          trilha_lufs: float = typer.Option(None, help="nível da trilha (padrão −29,5)"),
          legenda: bool = typer.Option(True, "--legenda/--sem-legenda",
              help="queima o texto sincronizado (usa o .srt gerado pelo build)"),
          musica_pecas: int = typer.Option(1,
              help="peças distintas no leito; 1 repete a mesma o capítulo todo"),
          musica_seg: float = typer.Option(None,
              help="duração de cada peça, em s (padrão: 120 no ace, 30 no musicgen)"),
          musica_peca: int = typer.Option(None,
              help="qual peça da paleta usar (padrão: 0 no ace; varia por "
                   "capítulo no musicgen)"),
          nome: str = typer.Option(None,
              help="nome do MP4 (padrão: o título do projeto)"),
          janela_imagem: float = typer.Option(10.0,
              help="preset sincronizado: segundos de narração por imagem"),
          imagens_candidatas: int = typer.Option(2,
              help="preset sincronizado: candidatos de imagem gerados por "
                   "janela, antes do QA automático escolher o melhor"),
          sincronizado_llm: bool = typer.Option(True,
              "--sincronizado-llm/--sem-sincronizado-llm",
              help="preset sincronizado: usa o Ollama para ler cada janela "
                   "(desligue em livros de ortografia antiga — taxa de acerto "
                   "medida perto de zero, e cada tentativa falha paga o "
                   "timeout inteiro; sem isso cai direto no cenário de "
                   "reserva, bem mais rápido)"),
          gpu: bool = typer.Option(True, "--gpu/--cpu")):
    """Gera o MP4 para o YouTube, com trilha opcional sob a narração."""
    p = _proj(slug)
    _etapa_video(p, proj_mod.carregar_config(p), preset=preset, musica=musica,
                 musica_explicita=(ctx.get_parameter_source("musica")
                                   is not ParameterSource.DEFAULT),
                 capa=capa, slides_dir=slides_dir, slides_seg=slides_seg,
                 slides_seed=slides_seed, trilha_lufs=trilha_lufs, legenda=legenda,
                 musica_pecas=musica_pecas, musica_seg=musica_seg,
                 musica_peca=musica_peca, nome=nome,
                 janela_imagem=janela_imagem, imagens_candidatas=imagens_candidatas,
                 sincronizado_llm=sincronizado_llm, gpu=gpu)


def _indice_capitulo(chave: str) -> int:
    """Índice determinístico a partir do nome do capítulo (ex.: `ch01-master`).

    Dá a cada capítulo uma peça de trilha diferente das dos irmãos, sem
    depender de contador externo -- e reproduzível: o mesmo capítulo
    re-renderizado cai sempre na mesma peça, a mesma garantia que já vale para
    a seed de `musica_ace._seed`/`musica_musicgen._seed`.
    """
    h = hashlib.sha256(chave.encode()).digest()
    return int.from_bytes(h[:4], "big")


def _etapa_video(p: Path, cfg: dict, *, preset: str = "slides",
                 musica: str = "ace", musica_explicita: bool = False,
                 capa: Path | None = None, slides_dir: Path | None = None,
                 slides_seg: float = 45.0, slides_seed: int | None = None,
                 trilha_lufs: float | None = None, legenda: bool = True,
                 musica_pecas: int = 1, musica_seg: float | None = None,
                 musica_peca: int | None = None, nome: str | None = None,
                 janela_imagem: float = 10.0, imagens_candidatas: int = 2,
                 sincronizado_llm: bool = True,
                 gpu: bool = True) -> list[Path]:
    """Renderiza um MP4 por master e devolve os arquivos gerados."""
    from ..audio import musica_ace as ace
    from ..audio import musica_musicgen as mg
    from ..audio.musica import TRILHA_LUFS, mixar, preparar_trilha
    from ..audio.process import duracao
    from ..video import slides as slides_mod
    from ..video import sincronizado as sincronizado_mod
    from ..video.render import presets, renderizar
    # A trilha vem ligada de fábrica porque o canal publica com ela; o vídeo mudo
    # era uma opção que o operador tinha de lembrar de pedir, e o resultado foi
    # justamente publicar sem querer um capítulo sem música.
    #
    # `musica_explicita` distingue o que veio da linha de comando do que veio do
    # padrão (o comando resolve isso com `get_parameter_source`). É a diferença
    # entre "a máquina não tem a venv" (segue sem trilha) e "pedi ace e não veio"
    # (erro).
    pedido_explicito = musica_explicita
    if preset not in presets():
        console.print(f"[red]preset desconhecido:[/] {preset} — use {', '.join(presets())}")
        raise typer.Exit(1)

    # Acervo conferido ANTES do laço, pelo mesmo motivo da trilha: pasta vazia
    # tem de aparecer agora, e não depois de meia hora de render.
    if preset == "slides" and capa is None:
        pasta = slides_dir or slides_mod.DIRETORIO_PADRAO
        acervo = slides_mod.disponiveis(pasta)
        if not acervo:
            console.print(f"[red]nenhuma imagem em[/] {pasta} — aponte "
                          "--slides-dir para uma pasta com .jpg/.png, ou use "
                          "outro preset")
            raise typer.Exit(1)
        console.print(f"[dim]acervo de slides: {len(acervo)} imagens em {pasta}[/]")
    if preset == "sincronizado" and capa is None:
        from ..video import imagem as imagem_mod
        if not imagem_mod.disponivel():
            console.print(f"[red]venv de imagem ausente:[/] {imagem_mod.VENV} — "
                          "preset sincronizado precisa gerar imagem")
            raise typer.Exit(1)

    # Resolve o modo da trilha uma vez, antes do laço: um erro de paleta ou um
    # arquivo inexistente tem de aparecer agora, e não depois de renderizar
    # metade dos capítulos.
    fonte, paleta = None, "contemplativo"
    if musica.startswith("ace"):
        paleta = musica.split(":", 1)[1] if ":" in musica else "contemplativo"
        if paleta not in ace.PALETAS:
            console.print(f"[red]paleta desconhecida:[/] {paleta} — "
                          f"use {', '.join(ace.PALETAS)}")
            raise typer.Exit(1)
        if not ace.disponivel():
            # Sem a venv, `--musica ace` PEDIDO e erro, mas `--musica ace`
            # PADRAO nao pode derrubar o render: num clone sem GPU o operador
            # nao pediu trilha nenhuma, so nao desligou a que vem de fabrica.
            if pedido_explicito:
                console.print(f"[red]venv de música ausente:[/] {ace.VENV} — "
                              "veja ESTADO.md, ou use `--musica gerada`")
                raise typer.Exit(1)
            console.print(f"[yellow]sem trilha:[/] a venv de música não existe "
                          f"em {ace.VENV} — renderizando só a narração")
            musica = "nenhuma"
    elif musica.startswith("musicgen"):
        paleta = musica.split(":", 1)[1] if ":" in musica else "contemplativo"
        if paleta not in mg.PALETAS:
            console.print(f"[red]paleta desconhecida:[/] {paleta} — "
                          f"use {', '.join(mg.PALETAS)}")
            raise typer.Exit(1)
        if not mg.disponivel():
            if pedido_explicito:
                console.print(f"[red]venv de música (MusicGen) ausente:[/] {mg.VENV} — "
                              "veja ESTADO.md, ou use `--musica gerada`")
                raise typer.Exit(1)
            console.print(f"[yellow]sem trilha:[/] a venv do MusicGen não existe "
                          f"em {mg.VENV} — renderizando só a narração")
            musica = "nenhuma"
    elif musica not in ("nenhuma", "gerada"):
        fonte = Path(musica).resolve()
        if not fonte.exists():
            console.print(f"[red]trilha não encontrada:[/] {fonte}")
            raise typer.Exit(1)
        console.print("[yellow]trilha de terceiro:[/] confira a licença antes de "
                      "publicar — o Content ID do YouTube reclama sozinho")
    masters = sorted((p / "output").glob("*-master.wav"))
    if not masters:
        console.print("[red]nenhum master[/] — rode `export` antes")
        raise typer.Exit(1)

    gerados: list[Path] = []
    for master in masters:
        audio = master
        if musica != "nenhuma":
            alvo = p / "cache" / f"{master.stem}-trilha.wav"
            with console.status("preparando a trilha…"):
                if musica.startswith("ace"):
                    # Padrão do ACE-Step continua fixo em 0, sem variar por
                    # capítulo: é o timbre já publicado em `dhammacakka` e nos
                    # primeiros capítulos do sutta, e mudar o padrão agora faria
                    # um capítulo remontado soar diferente dos irmãos no ar.
                    peca_inicial = musica_peca if musica_peca is not None else 0
                    tr = ace.preparar_trilha(
                        alvo, duracao(master), paleta=paleta,
                        n_pecas=musica_pecas, peca_s=musica_seg or ace.PECA_S,
                        peca_inicial=peca_inicial,
                        progresso=lambda m: console.print(f"[dim]{m}[/]"))
                elif musica.startswith("musicgen"):
                    # Aqui sim varia por capítulo quando não pedida: é motor
                    # novo, sem vídeo publicado ainda para desalinhar.
                    peca_inicial = (musica_peca if musica_peca is not None
                                    else _indice_capitulo(master.stem))
                    tr = mg.preparar_trilha(
                        alvo, duracao(master), paleta=paleta,
                        n_pecas=musica_pecas, peca_s=musica_seg or mg.PECA_S,
                        peca_inicial=peca_inicial,
                        progresso=lambda m: console.print(f"[dim]{m}[/]"))
                else:
                    tr = preparar_trilha(alvo, duracao(master), fonte)
                audio = p / "cache" / f"{master.stem}-com-trilha.wav"
                mixar(master, tr, audio, trilha_lufs=trilha_lufs or TRILHA_LUFS)

        # O nome do arquivo NAO e detalhe: o YouTube pre-preenche o titulo do
        # video com ele. "ch01.mp4" viraria o titulo sugerido da publicacao.
        base = nome or _nome_de_arquivo(cfg["titulo"])
        if len(masters) > 1:
            base = f"{base} - {master.stem.replace('-master','')}"
        destino = p / "output" / f"{base}.mp4"
        srt = p / "audio" / "chapters" / f"{master.stem.replace('-master','')}.srt"
        if legenda and not srt.exists():
            console.print(f"[yellow]sem legenda:[/] {srt.name} não existe — "
                          "rode `build` de novo para gerá-la")

        sincronizado_plano = None
        if preset == "sincronizado" and capa is None:
            if not srt.exists():
                console.print(f"[red]sem legenda:[/] {srt} não existe — preset "
                              "sincronizado precisa dela para saber o que narrar "
                              "em cada janela (rode `build` de novo)")
                raise typer.Exit(1)
            destino_imgs = (p / "cache" / "imagens-sincronizadas" /
                           master.stem.replace("-master", ""))
            with console.status("gerando imagens sincronizadas (Ollama + FLUX)…") as st:
                sincronizado_plano = sincronizado_mod.plano(
                    srt, destino_imgs, duracao(audio),
                    janela_s=janela_imagem, n_candidatos=imagens_candidatas,
                    usar_llm=sincronizado_llm, progresso=lambda m: st.update(m))

        with console.status(f"renderizando {destino.name}…"):
            renderizar(audio, destino, preset=preset, capa=capa, gpu=gpu,
                       legenda=srt if (legenda and srt.exists()) else None,
                       slides_dir=slides_dir, slides_seg=slides_seg,
                       slides_seed=slides_seed,
                       sincronizado_plano=sincronizado_plano)
        console.print(f"[green]{destino}[/] ({destino.stat().st_size/1e6:.0f} MB)")
        gerados.append(destino)

    console.print("[dim]lembre do disclosure de conteúdo sintético ao publicar[/]")
    return gerados


@app.command()
def publish(slug: str,
            arquivo: Path = typer.Option(None,
                help="MP4 a enviar (padrão: o único .mp4 em output/)"),
            privacidade: str = typer.Option("private",
                help="private, unlisted ou public"),
            tags: str = typer.Option(None, help="tags separadas por vírgula"),
            categoria: str = typer.Option("27",
                help="categoria do YouTube (27 = Educação)"),
            client_secret: Path = typer.Option(None,
                help="client_secret.json (padrão: config/youtube_client_secret.json)"),
            miniatura: Path = typer.Option(None,
                help="imagem para a miniatura (padrão: frame extraído do próprio MP4)"),
            miniatura_preset: str = typer.Option("slides",
                help="preset usado no `video`, para escolher um instante fora do dissolve"),
            sem_miniatura: bool = typer.Option(False, "--sem-miniatura",
                help="não define miniatura"),
            forcar: bool = typer.Option(False, "--forcar",
                help="sobe mesmo com rights.status incompleto ou marcado como teste"),
            republicar: bool = typer.Option(False, "--republicar",
                help="sobe de novo um projeto que já tem vídeo publicado"),
            token: Path = typer.Option(None,
                help="token de upload (padrão: config/youtube_token.json) — use outro "
                     "para publicar num canal diferente do padrão")):
    """Sobe o MP4 do projeto para o YouTube, com metadados do project.yaml.

    Exige credenciais OAuth próprias (veja docs/youtube-publish.md) — a ferramenta
    não decide o que publicar, só preenche o que já está registrado no projeto.
    """
    p = _proj(slug)
    _etapa_publish(p, proj_mod.carregar_config(p), arquivo=arquivo,
                   privacidade=privacidade, tags=tags, categoria=categoria,
                   client_secret=client_secret, miniatura=miniatura,
                   miniatura_preset=miniatura_preset, sem_miniatura=sem_miniatura,
                   forcar=forcar, republicar=republicar, token=token)


def _token_upload(token: Path | None) -> Path:
    """Token de UPLOAD (`youtube.upload`) -- diferente do token de canal/branding
    (`_token_canal`, escopo `youtube`). Cada canal do YouTube tem o seu, mesmo
    `client_secret.json` (mesmo app OAuth)."""
    return token or (settings().config_dir / "youtube_token.json")


def _etapa_publish(p: Path, cfg: dict, *, arquivo: Path | None = None,
                   privacidade: str = "private", tags: str | None = None,
                   categoria: str = "27", client_secret: Path | None = None,
                   miniatura: Path | None = None, miniatura_preset: str = "slides",
                   sem_miniatura: bool = False, forcar: bool = False,
                   republicar: bool = False, token: Path | None = None) -> str:
    """Sobe o MP4 e devolve o id do vídeo publicado.

    Um upload já feito é registrado em `output/publicado.json` e barra o segundo:
    `videos.insert` não é idempotente — repetir o comando cria OUTRO vídeo no
    canal, e o pipeline em lote existe justamente para ser re-rodado.
    """
    from ..publish.youtube import autenticar, enviar, metadados

    if privacidade not in ("private", "unlisted", "public"):
        console.print(f"[red]privacidade inválida:[/] {privacidade} — "
                      "use private, unlisted ou public")
        raise typer.Exit(1)

    status_rights = str((cfg.get("rights") or {}).get("status") or "")
    suspeito = (not status_rights or status_rights == "PREENCHER"
                or "NAO-PUBLICAR" in status_rights.upper()
                or "TESTE" in status_rights.upper())
    if suspeito and not forcar:
        console.print(f"[red]rights.status = '{status_rights or None}'[/] — "
                      "confira project.yaml antes de publicar, ou use --forcar")
        raise typer.Exit(1)
    elif suspeito:
        console.print(f"[yellow]publicando apesar de rights.status = "
                      f"'{status_rights}'[/]")

    if arquivo is None:
        candidatos = sorted((p / "output").glob("*.mp4"))
        if not candidatos:
            console.print("[red]nenhum mp4[/] em output/ — rode `video` antes")
            raise typer.Exit(1)
        if len(candidatos) > 1:
            console.print("[red]mais de um mp4[/] em output/ — use --arquivo: "
                          + ", ".join(c.name for c in candidatos))
            raise typer.Exit(1)
        arquivo = candidatos[0]

    registro = p / "output" / "publicado.json"
    if registro.exists() and not republicar:
        anterior = json.loads(registro.read_text(encoding="utf-8"))
        console.print(f"[yellow]já publicado[/] em {anterior.get('em')}: "
                      f"https://studio.youtube.com/video/{anterior['video_id']}/edit")
        console.print("[dim]use --republicar para subir outra cópia[/]")
        return anterior["video_id"]

    segredo = client_secret or (settings().config_dir / "youtube_client_secret.json")
    token = _token_upload(token)
    if not segredo.exists():
        console.print(f"[red]client secret ausente:[/] {segredo} — "
                      "veja docs/youtube-publish.md")
        raise typer.Exit(1)

    corpo = metadados(cfg, p, privacidade=privacidade,
                      tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else None,
                      categoria=categoria)

    console.print(f"[dim]arquivo:[/] {arquivo.name} ({arquivo.stat().st_size/1e6:.0f} MB)")
    console.print(f"[dim]título:[/] {corpo['snippet']['title']}")
    console.print(f"[dim]privacidade:[/] {privacidade}")

    with console.status("autenticando…"):
        creds = autenticar(segredo, token)

    def _progresso(fracao: float) -> None:
        console.print(f"[dim]enviado: {fracao*100:.0f}%[/]")

    video_id = enviar(arquivo, corpo, creds, progresso=_progresso)
    registro.write_text(json.dumps(
        {"video_id": video_id, "url": f"https://youtu.be/{video_id}",
         "arquivo": arquivo.name, "privacidade": privacidade,
         "em": datetime.now().isoformat(timespec="seconds")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    aviso = "" if privacidade != "private" else " (privado até você mudar)"
    console.print(f"[green]publicado{aviso}[/] "
                  f"https://studio.youtube.com/video/{video_id}/edit")

    if not sem_miniatura:
        from ..publish.youtube import definir_miniatura
        from ..video.thumbnail import extrair_frame, instante_seguro

        img = miniatura
        if img is None:
            srt = p / "audio" / "chapters" / f"{arquivo.stem.replace('-master', '')}.srt"
            if not srt.exists():
                candidatos_srt = sorted((p / "audio" / "chapters").glob("*.srt"))
                srt = candidatos_srt[0] if len(candidatos_srt) == 1 else None
            instante = instante_seguro(arquivo, miniatura_preset, srt=srt)
            img = p / "cache" / "miniatura.jpg"
            extrair_frame(arquivo, img, instante)
        try:
            definir_miniatura(video_id, img, creds)
            console.print(f"[dim]miniatura definida:[/] {img}")
        except Exception as e:
            console.print(f"[yellow]miniatura não definida[/] ({e}) — "
                          "confira se o canal tem miniatura customizada "
                          "habilitada (verificação de telefone)")
    return video_id


canal_app = typer.Typer(help="Branding do canal no YouTube (descrição, keywords, banner)",
                        no_args_is_help=True)
app.add_typer(canal_app, name="canal")


def _token_canal(token: Path | None) -> Path:
    return token or (settings().config_dir / "youtube_canal_token.json")


def _autenticar_canal(client_secret: Path | None, token: Path | None = None):
    from ..publish.youtube import SCOPES_CANAL, autenticar

    segredo = client_secret or (settings().config_dir / "youtube_client_secret.json")
    if not segredo.exists():
        console.print(f"[red]client secret ausente:[/] {segredo} — "
                      "veja docs/youtube-publish.md")
        raise typer.Exit(1)
    with console.status("autenticando (escopo de canal — pode pedir login de novo)…"):
        return autenticar(segredo, _token_canal(token), scopes=SCOPES_CANAL)


_TOKEN_HELP = ("padrão: config/youtube_canal_token.json — use outro arquivo para "
               "gerenciar um segundo canal da mesma conta Google sem sobrescrever "
               "o token do primeiro")


@canal_app.command("mostrar")
def canal_mostrar(client_secret: Path = typer.Option(None,
                      help="padrão: config/youtube_client_secret.json"),
                  token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Mostra a descrição, keywords e país atuais do canal."""
    from ..publish.youtube import canal_atual

    creds = _autenticar_canal(client_secret, token)
    canal = canal_atual(creds)
    snip = canal.get("snippet", {})
    branding = (canal.get("brandingSettings") or {}).get("channel", {})
    console.print(f"[bold]{snip.get('title')}[/]  ({canal.get('id')})")
    console.print(f"país: {snip.get('country') or '[dim](não definido)[/]'}")
    console.print(f"keywords: {branding.get('keywords') or '[dim](vazias)[/]'}")
    console.print(f"\n[bold]descrição:[/]\n{snip.get('description') or '[dim](vazia)[/]'}")


@canal_app.command("atualizar")
def canal_atualizar(descricao: Path = typer.Option(None,
                        help="arquivo .txt com a nova descrição do canal"),
                    keywords: str = typer.Option(None,
                        help="separadas por vírgula, ex.: 'budismo,dhamma,cânone páli'"),
                    pais: str = typer.Option(None, help="código ISO, ex.: BR"),
                    client_secret: Path = typer.Option(None),
                    token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Atualiza descrição/keywords/país do canal (`brandingSettings`).

    `channels.update` substitui o recurso inteiro — este comando busca o estado
    atual primeiro e só troca os campos passados aqui, então rodar sem nenhuma
    opção não muda nada.
    """
    from ..publish.youtube import atualizar_canal, formatar_keywords

    if descricao is None and keywords is None and pais is None:
        console.print("[yellow]nada a atualizar[/] — passe --descricao, --keywords "
                      "e/ou --pais")
        raise typer.Exit(1)

    desc_texto = descricao.read_text(encoding="utf-8").strip() if descricao else None
    kw = formatar_keywords([k.strip() for k in keywords.split(",") if k.strip()]) if keywords else None

    creds = _autenticar_canal(client_secret, token)
    atualizar_canal(creds, descricao=desc_texto, keywords=kw, pais=pais)
    console.print("[green]canal atualizado[/] — confira em "
                  "https://studio.youtube.com/channel/_/editing/branding")


@canal_app.command("banner")
def canal_banner(imagem: Path, client_secret: Path = typer.Option(None),
                 token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Sobe a arte de capa do canal (recomendado: 2560×1440, ≤6 MB).

    A foto de perfil/logo do canal não tem endpoint na API do YouTube — só se
    troca manualmente em https://studio.youtube.com.
    """
    from ..publish.youtube import atualizar_banner

    if not imagem.exists():
        console.print(f"[red]imagem não encontrada:[/] {imagem}")
        raise typer.Exit(1)
    creds = _autenticar_canal(client_secret, token)
    with console.status("enviando banner…"):
        atualizar_banner(imagem, creds)
    console.print("[green]banner atualizado[/] — confira em "
                  "https://studio.youtube.com/channel/_/editing/branding")
    console.print("[yellow]a foto de perfil do canal precisa ser trocada à mão[/] — "
                  "a Data API v3 não expõe esse recurso")


@canal_app.command("marca-dagua")
def canal_marca_dagua(imagem: Path,
                      canto: str = typer.Option("bottomRight",
                          help="topLeft, topRight, bottomLeft ou bottomRight"),
                      client_secret: Path = typer.Option(None),
                      token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Define a marca d'água do canal — PNG com alfa recomendado.

    Aparece sobre o vídeo o tempo todo e funciona como botão de inscrição ao
    passar o mouse; vale para os vídeos já publicados também.
    """
    from ..publish.youtube import CANTOS_MARCA_DAGUA, canal_atual, definir_marca_dagua

    if canto not in CANTOS_MARCA_DAGUA:
        console.print(f"[red]canto inválido:[/] {canto} — use "
                      f"{', '.join(CANTOS_MARCA_DAGUA)}")
        raise typer.Exit(1)
    if not imagem.exists():
        console.print(f"[red]imagem não encontrada:[/] {imagem}")
        raise typer.Exit(1)
    creds = _autenticar_canal(client_secret, token)
    channel_id = canal_atual(creds)["id"]
    with console.status("enviando marca d'água…"):
        definir_marca_dagua(imagem, creds, channel_id, canto=canto)
    console.print("[green]marca d'água definida[/] — vale para os vídeos já "
                  "publicados e os próximos")


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


@app.command()
def musica(paleta: str = typer.Option("contemplativo",
               help="contemplativo, drone, sobrio, piano ou flauta"),
           motor: str = typer.Option("ace", help="ace ou musicgen"),
           duracao: float = typer.Option(0.0,
               help="se >0, monta também um leito desta duração, para ouvir a emenda")):
    """Gera as peças da trilha e mostra onde ficaram, para ouvir antes de renderizar.

    Existe porque a alternativa é descobrir que a paleta não serve depois de
    renderizar uma hora de vídeo.
    """
    if motor == "ace":
        from ..audio import musica_ace as m
    elif motor == "musicgen":
        from ..audio import musica_musicgen as m
    else:
        console.print(f"[red]motor desconhecido:[/] {motor} — use ace ou musicgen")
        raise typer.Exit(1)

    if paleta not in m.PALETAS:
        console.print(f"[red]paleta desconhecida:[/] {paleta} — use {', '.join(m.PALETAS)}")
        raise typer.Exit(1)
    if not m.disponivel():
        console.print(f"[red]venv de música ({motor}) ausente:[/] {m.VENV} — veja ESTADO.md")
        raise typer.Exit(1)

    with console.status(f"peças da paleta {paleta} ({motor})…"):
        pecas = m.gerar_pecas(paleta, progresso=lambda msg: console.print(f"[dim]{msg}[/]"))
    timbres = m.PALETAS[paleta]
    t = Table("peça", "timbre", "arquivo")
    for i, peca in enumerate(pecas):
        t.add_row(f"{i:02d}", timbres[i % len(timbres)], str(peca))
    console.print(t)

    if duracao > 0:
        alvo = m.CACHE / f"leito-{motor}-{paleta}-{int(duracao)}s.wav"
        with console.status("montando o leito…"):
            m.preparar_trilha(alvo, duracao, paleta=paleta)
        console.print(f"[green]{alvo}[/]")


@app.command()
def imagem(prompt: str = typer.Argument(None, help="prompt livre"),
          arquivo: Path = typer.Option(None, help="um prompt por linha, para lote"),
          modelo: str = typer.Option("flux", help="flux, sd ou sd:large"),
          n: int = typer.Option(4, help="variações por prompt"),
          largura: int = typer.Option(None, help="padrão: 1344 (~16:9)"),
          altura: int = typer.Option(None, help="padrão: 768 (~16:9)"),
          passos: int = typer.Option(None, help="padrão depende do modelo"),
          guidance: float = typer.Option(None, help="padrão depende do modelo"),
          negativo: str = typer.Option(None,
              help="negative prompt — só tem efeito com --modelo sd/sd:large")):
    """Gera candidatos em cache/imagens/, para revisar antes de `imagem-aprovar`.

    Não escreve direto no acervo: nem toda imagem gerada presta, e curadoria é
    humana. Ver `imagem-aprovar` para o passo seguinte.
    """
    from ..video import imagem as img

    if (prompt is None) == (arquivo is None):
        console.print("[red]passe um prompt OU --arquivo, não os dois nem nenhum[/]")
        raise typer.Exit(1)
    try:
        img.resolver_modelo(modelo)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    if not img.disponivel():
        console.print(f"[red]venv de imagem ausente:[/] {img.VENV} — veja ESTADO.md")
        raise typer.Exit(1)

    prompts = ([prompt] if prompt is not None else
               [l.strip() for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()])
    kwargs = {"modelo": modelo, "n": n, "passos": passos, "guidance": guidance,
              "negative": negativo}
    if largura is not None:
        kwargs["largura"] = largura
    if altura is not None:
        kwargs["altura"] = altura

    for p in prompts:
        with console.status(f"gerando '{p[:60]}'…"):
            candidatos = img.gerar(p, progresso=lambda m: console.print(f"[dim]{m}[/]"), **kwargs)
        for c in candidatos:
            console.print(f"[green]{c}[/]")


@app.command(name="imagem-aprovar")
def imagem_aprovar(arquivos: list[Path],
                   slides_dir: Path = typer.Option(None, help="padrão: assets/slides")):
    """Converte os candidatos escolhidos (JPEG q2, ≤1920px) e move para o acervo."""
    from ..video import imagem as img

    kwargs = {"slides_dir": slides_dir} if slides_dir is not None else {}
    try:
        finais = img.aprovar(arquivos, **kwargs)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    for f in finais:
        console.print(f"[green]{f}[/]")


@app.command()
def doctor():
    """Verifica GPU, torch, FFmpeg, licenças dos pesos e a configuração resolvida."""
    import shutil
    import torch

    ok = lambda b: "[green]OK[/]" if b else "[red]FALHOU[/]"
    console.print(f"torch {torch.__version__} · CUDA {ok(torch.cuda.is_available())}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        console.print(f"GPU {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
        try:
            x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()
            console.print(f"kernels na GPU {ok(True)}")
        except Exception as e:
            console.print(f"kernels na GPU {ok(False)} — {e}")
    console.print(f"ffmpeg {ok(shutil.which('ffmpeg'))} · ffprobe {ok(shutil.which('ffprobe'))}")
    lic = settings().raiz / "LICENSES.md"
    console.print(f"registro de licenças {ok(lic.exists())} — {lic}")
    _mostrar_config()


def _mostrar_config() -> None:
    """Valor final de cada campo do Settings e de onde ele veio."""
    from dataclasses import fields
    from urllib import request

    s = settings()
    t = Table(title="configuração (vf.toml / VF_*)")
    for c in ("campo", "valor", "origem", ""):
        t.add_column(c)
    for f in fields(s):
        if f.name == "origem":
            continue
        v = getattr(s, f.name)
        if isinstance(v, Path):
            estado = "[green]existe[/]" if v.exists() else "[yellow]ausente[/]"
        else:
            estado = ""
            if f.name.endswith("_url"):
                try:
                    request.urlopen(v, timeout=2).close()
                    estado = "[green]responde[/]"
                except Exception as e:
                    # 404 na raiz ainda prova que o serviço está no ar
                    estado = ("[green]responde[/]" if getattr(e, "code", None)
                              else "[yellow]fora do ar[/]")
        t.add_row(f.name, str(v), s.origem.get(f.name, ""), estado)
    console.print(t)


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
