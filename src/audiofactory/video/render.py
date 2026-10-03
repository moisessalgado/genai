"""Renderizacao do MP4 para o YouTube (TDD 18, V2).

O YouTube nao aceita audio puro, e um retangulo preto por nove minutos parece
canal abandonado. Aqui o video e gerado a partir do PROPRIO sinal de audio, com
filtros do ffmpeg -- sem dependencia nova e sem arquivo de video para licenciar.

Principio de desenho: a imagem acompanha a narracao, nao compete com ela. Nada de
corte, flash ou movimento rapido; quem poe um audiolivro para tocar quase sempre
nao esta olhando para a tela, e quando olha, precisa achar a tela em repouso.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..audio.process import duracao
from . import slides as slides_mod
from . import sincronizado as sincronizado_mod

# Acima disto, um `-filter_complex` so com scale+blur+overlay+xfade por imagem
# estoura RAM: medido no piloto do preset `sincronizado` (234 imagens), o
# ffmpeg foi morto pelo OOM killer do sistema com ~48 GB de RSS (kernel log:
# "Out of memory: Killed process ... ffmpeg ... anon-rss:50450564kB"). O grafo
# inteiro fica vivo na memoria de uma vez so; passar de ~20-30 ramos e o que
# fez a diferenca entre alguns GB e travar a maquina. Acima do teto,
# `renderizar` particiona em lotes silenciosos, concatena por stream-copy
# (barato) e só entao muxa audio+legenda num passo final simples.
#
# 12, nao 24: esta maquina roda outros processos ao mesmo tempo (outro agente
# em outro projeto, editor, etc.) — a mesma RAM nao e so desta renderizacao, e
# o teto medido (48 GB para 234 imagens, ~200 MB/imagem) ja e folga curta
# perto dos 59 GB totais quando dividida com vizinhos imprevisiveis.
LOTE_MAXIMO = 12

# 1080p a 25 fps: o YouTube reencoda tudo, e mais resolucao so aumenta o upload.
LARGURA, ALTURA, FPS = 1920, 1080, 25

# Qualidade do NVENC. O `cq 31` que estava aqui rendia 380 kb/s em 1080p, e o
# codificador gastava esse orcamento onde havia detalhe -- as tarjas borradas,
# que sao quase planas, sobravam com blocos inteiros no mesmo valor e viravam
# listras de cor. `cq 20` custa ~3x o arquivo (46 MB viram ~150 MB em onze
# minutos, irrelevante num upload unico) e devolve a rampa.
#
# `spatial-aq` existe exatamente para este material: ele reparte o orcamento a
# favor das areas LISAS, que e onde o olho enxerga banda, e nao a favor do
# detalhe, que e onde o codificador iria sozinho.
CQ = 20
NVENC = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq",
         "-rc", "vbr", "-cq", str(CQ), "-b:v", "0",
         "-maxrate", "16M", "-bufsize", "32M",
         "-spatial-aq", "1", "-aq-strength", "8",
         "-bf", "3", "-rc-lookahead", "32", "-profile:v", "high"]
X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]

# Ultimo elo do grafo, comum a todos os presets: derruba a cadeia de 10 bits
# para os 8 do H.264 com difusao de erro. Sem o `dither`, a conversao arredonda
# e as bandas que os 10 bits evitaram voltam inteiras no ultimo passo.
#
# O `matrix`/`range` nao sao enfeite: o JPEG entra em faixa cheia, e sair sem
# converter marcava o MP4 como `yuvj420p`. Player que ignora a marca (nao sao
# poucos) esmagava preto e branco. Aqui a conversao e explicita e a marca,
# escrita no arquivo pelas flags de cor do ffmpeg.
#
# O `format=gbrp10le` da frente nao e redundancia com os slides (que ja chegam
# assim): ele existe para os OUTROS presets. O `zscale` recusa converter a
# partir de um quadro cuja matriz nao esta marcada -- e o `espectro`, que sai de
# um `blend`, e exatamente um desses ("no path between colorspaces"). Passando
# por RGB primeiro, nao ha matriz de entrada para adivinhar.
SAIDA_8BITS = ("format=gbrp10le,"
               "zscale=matrix=709:range=limited:dither=error_diffusion,"
               "format=yuv420p")
CORES = ["-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-color_range", "tv"]

# Paleta escura de proposito: video claro por uma hora cansa, e tela escura gasta
# menos bateria em OLED, que e onde a maioria ouve.
FUNDO_A = "0x0d1b2a"
FUNDO_B = "0x1b263b"
COR_ONDA = "0x7ec8e3"


def presets() -> list[str]:
    return ["slides", "sincronizado", "ondas", "espectro", "estatico", "gradiente"]


def _fundo(preset: str, capa: Path | None,
           plano: tuple[list[Path], float | list[float], float] | None,
           veu: bool) -> list[str]:
    """Entradas de video do ffmpeg. O audio entra depois delas."""
    if capa is not None:
        return ["-loop", "1", "-framerate", str(FPS), "-i", str(capa)]
    if preset in ("slides", "sincronizado"):
        imagens, cada, _ = plano
        return slides_mod.entradas(imagens, cada, FPS, LARGURA, veu)
    if preset == "estatico":
        return ["-f", "lavfi", "-i",
                f"color=c={FUNDO_A}:s={LARGURA}x{ALTURA}:r={FPS}"]
    # `gradiente`: o mesmo fundo dos outros presets, mas SEM visualizacao por
    # cima. O operador achou a onda cansativa em nove minutos, e cor chapada por
    # onze cai no "canal abandonado" que este arquivo existe para evitar. A
    # deriva a 0,01 nao e perceptivel quadro a quadro e ainda assim a tela nao
    # congela.
    # gradiente que se move devagar: a 0,01 nao ha movimento perceptivel quadro a
    # quadro, mas a tela nao fica congelada por nove minutos
    return ["-f", "lavfi", "-i",
            f"gradients=s={LARGURA}x{ALTURA}:c0={FUNDO_A}:c1={FUNDO_B}"
            f":speed=0.01:r={FPS}"]


def _sobreposicao(preset: str, capa: Path | None, i_audio: int,
                  plano: tuple[list[Path], float, float] | None,
                  veu: bool) -> str:
    """Filtro que desenha a visualizacao do audio sobre o fundo.

    `i_audio` nao e fixo: o preset `slides` abre uma entrada por imagem, e o
    audio passa a ser a ultima delas.
    """
    if preset in ("slides", "sincronizado") and capa is None:
        imagens, cada, cruzamento = plano
        return slides_mod.filtro(imagens, cada, cruzamento, LARGURA, ALTURA, veu)
    if preset in ("estatico", "gradiente"):
        # 10 bits aqui pelo mesmo motivo dos slides: o `gradiente` e uma rampa
        # de canto a canto, o pior caso possivel para banda em 8 bits.
        return (f"[0:v]scale={LARGURA}:{ALTURA}:force_original_aspect_ratio=increase,"
                f"crop={LARGURA}:{ALTURA},"
                f"format={slides_mod.PROFUNDIDADE},setsar=1[v]")
    if preset == "espectro":
        # `axis=0` e obrigatorio: por padrao o showcqt desenha a regua de notas
        # (A B C D E F G) sobre a imagem, o que num audiolivro nao faz sentido
        # nenhum. `sono_h=0` tira o sonograma e deixa so as barras, que sao mais
        # calmas. O blend e `lighten` e nao `screen`: screen lava o fundo inteiro.
        return (f"[{i_audio}:a]showcqt=s={LARGURA}x{ALTURA}:r={FPS}:count=2:axis=0:"
                f"sono_h=0:bar_g=2:basefreq=55:endfreq=6000:"
                f"cscheme=0.4|0.8|1.0|0.1|0.4|0.9[cqt];"
                f"[0:v]scale={LARGURA}:{ALTURA},setsar=1[bg];"
                "[bg][cqt]blend=all_mode=lighten[v]")
    # ondas: linha central sobre o fundo, na altura dos olhos
    return (f"[{i_audio}:a]showwaves=s={LARGURA}x{int(ALTURA*0.28)}:mode=cline:"
            f"colors={COR_ONDA}:r={FPS}:scale=sqrt[w];"
            f"[0:v]scale={LARGURA}:{ALTURA},setsar=1[bg];"
            f"[bg][w]overlay=0:{int(ALTURA*0.36)}:format=auto[v]")


# Estilo da legenda queimada.
#
# ATENCAO a escala: para um .srt o libass assume uma tela de referencia de 288 px
# de altura, e nao os 1080 reais. Tudo aqui e multiplicado por 1080/288 = 3,75 na
# hora de desenhar. `FontSize=13` vira ~49 px na tela, que e o corpo certo para
# 1080p; um `FontSize=26` "razoavel" viraria 97 px e cobriria o meio da imagem.
# Pelo mesmo motivo `MarginV=20` vira ~75 px do rodape.
#
# Contorno em vez de caixa opaca: caixa cobre o fundo o tempo todo, contorno so
# ocupa o traco da letra.
ESTILO_LEGENDA = (
    "FontName=DejaVu Sans,FontSize=13,PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&HC0000000,BorderStyle=1,Outline=1,Shadow=0,"
    "Alignment=2,MarginV=20"
)


def _escapar(p: Path) -> str:
    """Caminho dentro do filtro do ffmpeg: dois niveis de escape."""
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def renderizar(audio: Path, destino: Path, preset: str = "slides",
               capa: Path | None = None, gpu: bool = True,
               legenda: Path | None = None,
               slides_dir: Path | None = None,
               slides_seg: float = slides_mod.SEGUNDOS_POR_IMAGEM,
               slides_seed: int | None = None,
               sincronizado_plano: tuple[list[Path], list[float], float] | None = None
               ) -> Path:
    """Gera o MP4 a partir do audio. `capa` sobrepoe o fundo gerado.

    No preset `slides` o numero de imagens sai da duracao do audio, e quais
    imagens sao sorteadas do acervo -- diferentes a cada render, a menos que
    `slides_seed` fixe o sorteio.

    No preset `sincronizado` a geracao de imagem (LLM + FLUX em lote) e cara
    demais para acontecer aqui dentro a cada chamada -- `sincronizado_plano`
    chega PRONTO de `sincronizado.plano()`, chamado uma vez por quem orquestra
    (`cli/main.py`), e `renderizar` so monta o ffmpeg em cima dele.
    """
    if preset not in presets():
        raise ValueError(f"preset desconhecido: {preset} (use {presets()})")
    plano = None
    if preset == "slides" and capa is None:
        plano = slides_mod.plano(
            duracao(audio), slides_dir or slides_mod.DIRETORIO_PADRAO,
            segundos_por_imagem=slides_seg, seed=slides_seed)
    elif preset == "sincronizado" and capa is None:
        if sincronizado_plano is None:
            raise ValueError(
                "preset sincronizado precisa de sincronizado_plano — "
                "chame video/sincronizado.py::plano() antes de renderizar")
        plano = sincronizado_plano
    # O veu so existe para dar contraste a legenda; sem legenda ele seria um
    # escurecimento sem motivo no rodape da arte.
    veu = legenda is not None
    entrada_fundo = _fundo(preset, capa, plano, veu)
    # Contado, e nao deduzido: `slides` abre uma entrada por imagem mais o veu,
    # e o audio e sempre a proxima. Errar este indice mapeia o audio errado.
    n_video = entrada_fundo.count("-i")
    filtro = _sobreposicao(preset, capa, n_video, plano, veu)
    if legenda is not None:
        if not legenda.exists():
            raise FileNotFoundError(f"legenda nao encontrada: {legenda}")
        # queima DEPOIS da visualizacao, senao a onda passaria por cima do texto.
        # Todo preset termina com exatamente um rotulo [v]; renomea-lo e o
        # suficiente para encadear mais um filtro no fim.
        filtro = filtro.replace("[v]", "[vbase]")
        filtro += (f";[vbase]subtitles='{_escapar(legenda)}'"
                   f":force_style='{ESTILO_LEGENDA}'[v]")

    # O despejo para 8 bits e o ULTIMO elo, depois ate da legenda: feito antes,
    # o `subtitles` receberia 8 bits e o resto do grafo perderia a precisao que
    # os 10 bits existem para dar.
    filtro = filtro.replace("[v]", "[v10]") + f";[v10]{SAIDA_8BITS}[v]"

    imagens_plano = plano[0] if preset in ("slides", "sincronizado") and capa is None else None
    if imagens_plano is not None and len(imagens_plano) > LOTE_MAXIMO:
        # Grafo unico com tudo (scale+blur+overlay+xfade por imagem) e o que
        # estourou RAM — ver LOTE_MAXIMO. Particiona em vez de montar tudo de
        # uma vez.
        return _renderizar_em_lotes(plano, audio, destino, gpu, legenda, veu)

    video = NVENC if gpu else X264
    destino.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         *entrada_fundo, "-i", str(audio),
         "-filter_complex", filtro, "-map", "[v]", "-map", f"{n_video}:a",
         *video, *CORES, "-pix_fmt", "yuv420p",
         # faststart poe o indice no inicio: o YouTube processa antes de terminar
         # o upload, e um player web consegue comecar sem baixar tudo
         "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(destino)],
        capture_output=True, text=True, check=True)
    return destino


def _renderizar_segmento_silencioso(imagens: list[Path], duracoes: list[float],
                                    cruzamento: float, destino: Path, gpu: bool) -> Path:
    """Um lote de imagens encadeadas, sem audio e sem legenda — so o video."""
    entrada = slides_mod.entradas(imagens, duracoes, FPS, LARGURA, veu=False)
    filtro = slides_mod.filtro(imagens, duracoes, cruzamento, LARGURA, ALTURA, veu=False)
    filtro = filtro.replace("[v]", "[v10]") + f";[v10]{SAIDA_8BITS}[v]"
    video = NVENC if gpu else X264
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         *entrada, "-filter_complex", filtro, "-map", "[v]",
         *video, *CORES, "-pix_fmt", "yuv420p", "-an", str(destino)],
        capture_output=True, text=True, check=True)
    return destino


def _renderizar_em_lotes(plano: tuple[list[Path], list[float], float], audio: Path,
                         destino: Path, gpu: bool, legenda: Path | None,
                         veu: bool) -> Path:
    """Video longo (muitas imagens) em pedacos pequenos de memoria limitada,
    concatenados por stream-copy (barato), e so entao casados com o audio e a
    legenda num passo final — esse ultimo passo nao tem scale/blur/xfade
    nenhum, entao nao volta a estourar RAM nem em video de horas.

    Cada lote perde o veu (so faz sentido junto da legenda, que so entra no
    passo final) — sem isso o rodape escureceria duas vezes na emenda.
    """
    imagens, duracoes, cruzamento = plano
    with tempfile.TemporaryDirectory(prefix="render-lotes-") as tmp:
        tmp_dir = Path(tmp)
        segmentos: list[Path] = []
        for i in range(0, len(imagens), LOTE_MAXIMO):
            fatia_imgs = imagens[i:i + LOTE_MAXIMO]
            fatia_dur = duracoes[i:i + LOTE_MAXIMO]
            seg = tmp_dir / f"lote-{i // LOTE_MAXIMO:04d}.mp4"
            _renderizar_segmento_silencioso(fatia_imgs, fatia_dur, cruzamento, seg, gpu)
            segmentos.append(seg)

        concat_txt = tmp_dir / "concat.txt"
        concat_txt.write_text(
            "".join(f"file '{s.resolve()}'\n" for s in segmentos), encoding="utf-8")
        video_concatenado = tmp_dir / "concatenado.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(concat_txt),
             "-c", "copy", str(video_concatenado)],
            capture_output=True, text=True, check=True)

        # Passo final: so casa video (ja pronto) + audio real + legenda queimada
        # (com o veu, se pedido) — grafo trivial, sem risco de OOM.
        filtro_final = "[0:v]null[v]"
        entradas_extra: list[str] = []
        n_seguinte = 1
        if veu and legenda is not None:
            entradas_extra += ["-f", "lavfi", "-i", slides_mod.veu_lavfi(FPS, LARGURA)]
            filtro_final = (f"[0:v][{n_seguinte}:v]"
                            f"overlay=0:{ALTURA - slides_mod.VEU_ALTURA}:format=auto[v]")
            n_seguinte += 1
        if legenda is not None:
            if not legenda.exists():
                raise FileNotFoundError(f"legenda nao encontrada: {legenda}")
            filtro_final = filtro_final.replace("[v]", "[vbase]")
            filtro_final += (f";[vbase]subtitles='{_escapar(legenda)}'"
                             f":force_style='{ESTILO_LEGENDA}'[v]")

        video = NVENC if gpu else X264
        destino.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(video_concatenado), *entradas_extra, "-i", str(audio),
             "-filter_complex", filtro_final, "-map", "[v]", "-map", f"{n_seguinte}:a",
             *video, *CORES, "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-c:a", "aac", "-b:a", "192k", "-shortest", str(destino)],
            capture_output=True, text=True, check=True)
    return destino
