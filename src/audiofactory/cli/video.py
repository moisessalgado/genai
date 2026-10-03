"""Vídeo: `video` (MP4 para o YouTube), trilha (`musica`) e arte (`imagem`, `imagem-aprovar`)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import typer
from click.core import ParameterSource
from rich.table import Table

from .. import project as proj_mod
from ._comum import _proj, console
from .app import app


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
