"""Fundo de slides: imagens do canal trocando devagar sob a narracao (TDD 18).

Substitui o `gradiente`, que existia so para a tela nao congelar. Um gradiente
que deriva a 0,01 resolve o "canal abandonado" e nao diz nada; a arte do canal
diz. O desenho continua o mesmo do resto do modulo de video: a imagem acompanha
a narracao e nao compete com ela, entao a troca e um dissolve longo e nao um
corte, e nada se move dentro do quadro.

Nada de `zoompan` (Ken Burns): num still de 1080p ele custa caro por quadro e o
passo de zoom, por ser fracionario, treme de forma visivel em movimento lento --
o oposto do que este preset quer.

A selecao e aleatoria a cada render de proposito: o operador pediu variedade
entre videos. Quem precisa repetir um render exato passa `seed`.
"""
from __future__ import annotations

import random
from pathlib import Path

# assets/slides na raiz do repo: video/ -> audiofactory/ -> src/ -> raiz.
DIRETORIO_PADRAO = Path(__file__).resolve().parents[3] / "assets" / "slides"

EXTENSOES = (".jpg", ".jpeg", ".png", ".webp")

# Uma imagem a cada 45 s: em nove minutos dao doze trocas. Mais curto vira
# apresentacao de slides e puxa o olho para a tela; mais longo volta a ser
# imagem parada.
SEGUNDOS_POR_IMAGEM = 45.0

# Dissolve de 2 s. Abaixo de 1 s ja se percebe como corte.
CRUZAMENTO = 2.0

# Veu inferior sob a legenda queimada. O `gradiente` que este preset substitui
# era escuro por baixo do texto de graca; arte clara nao e, e "Nobre Caminho
# Octuplo" em branco sobre um lotus vermelho vive do contorno de 1 px. O veu
# devolve o contraste sem escurecer a imagem toda -- ele cobre as duas linhas de
# legenda (MarginV=20 do .srt cai a ~75 px do rodape) e para bem abaixo do
# centro da composicao.
#
# 420 px e nao 300: a altura aqui nao e a do texto, e a do DISFARCE. Um veu
# curto tem de escurecer depressa para dar contraste no rodape, e a rampa curta
# e justamente o que o olho le como borda -- a faixa aparece como objeto
# desenhado sobre a arte. Esticando a rampa, o mesmo escurecimento no pe do
# quadro chega tao devagar que nao ha onde o olho fixar o comeco dela.
VEU_ALTURA = 420

# Opacidade no PE do quadro, onde a legenda esta. No topo do veu e sempre zero.
VEU_OPACIDADE = 0.85

# Expoente da rampa. Linear (1.0) distribui o escurecimento por igual e ainda
# deixa o topo do veu perceptivel; com 2,4 a metade de cima fica praticamente
# transparente (a 50% da altura o veu esta em 16% da opacidade) e quase todo o
# escurecimento se concentra nas duas linhas de legenda, que sao o motivo dele.
VEU_CURVA = 2.4

# O veu e ESTATICO. Antes ele vinha de um `gradients`, que anima por padrao (a
# cor circula em `speed`), e o rodape pulsava a noite toda -- movimento sem
# motivo, exatamente o que o resto do modulo evita. Aqui o `geq` desenha a rampa
# UMA vez (`trim=end_frame=1`) e o `loop` repete aquele quadro: alem de nao
# animar, o custo por quadro cai a zero, e `geq` sobre 1920x420 pixels em 16 mil
# quadros nao e barato.

# Resolucao em que o borrao de fundo e calculado, e o sigma nela. O par importa
# junto: sigma 15 em 480x270 cobre o mesmo raio, em fracao do quadro, que sigma
# 6 cobria em 192x108 -- o borrao tem a mesma cara, so que sem os degraus.
BORRAO_L, BORRAO_A = 480, 270
BORRAO_SIGMA = 15

# Profundidade de trabalho do grafo. Ver `filtro` para o porque.
PROFUNDIDADE = "gbrp10le"

# O video precisa sobrar sobre o audio: `-t` por imagem cai na grade de quadros
# e o arredondamento, somado em doze slides, pode deixar o video mais curto que
# a narracao -- com `-shortest` isso cortaria o fim da fala. Sobra e descartada.
MARGEM_S = 2.0


def disponiveis(diretorio: Path = DIRETORIO_PADRAO) -> list[Path]:
    """Imagens utilizaveis, em ordem estavel (a aleatoriedade vem do sorteio)."""
    if not diretorio.is_dir():
        return []
    return sorted(p for p in diretorio.iterdir()
                  if p.suffix.lower() in EXTENSOES and p.is_file())


def quantas(duracao: float, segundos_por_imagem: float = SEGUNDOS_POR_IMAGEM,
            cruzamento: float = CRUZAMENTO) -> int:
    """Numero de slides para um audio desta duracao.

    O teto nao e estetico e sim aritmetico: com `n` imagens de `t` segundos e
    `n-1` dissolves, o video dura `n*t - (n-1)*d`. Para que cada imagem ainda
    fique parada pelo menos o tempo de um dissolve, `t >= 2d`, o que limita
    `n <= duracao/d - 1`. Sem esse corte um audio de 10 s pediria tres slides
    que so fariam dissolve, um dentro do outro.
    """
    if duracao <= 0:
        raise ValueError(f"duracao invalida: {duracao}")
    teto = int(duracao / cruzamento) - 1
    return max(1, min(round(duracao / segundos_por_imagem), teto))


def sortear(imagens: list[Path], n: int, seed: int | None = None) -> list[Path]:
    """Sorteia `n` imagens sem repetir nenhuma antes de esgotar o acervo.

    Embaralha o acervo inteiro e vai consumindo; se `n` passa do que existe,
    reembaralha. A unica costura cuidada e a emenda entre voltas: repetir a
    mesma imagem em slides vizinhos apareceria como um dissolve que nao muda
    nada, que o espectador le como travamento do video.
    """
    if not imagens:
        raise ValueError("nenhuma imagem disponivel para os slides")
    rng = random.Random(seed)
    escolhidas: list[Path] = []
    while len(escolhidas) < n:
        volta = list(imagens)
        rng.shuffle(volta)
        if escolhidas and len(volta) > 1 and volta[0] == escolhidas[-1]:
            volta[0], volta[1] = volta[1], volta[0]
        escolhidas.extend(volta[:n - len(escolhidas)])
    return escolhidas


def plano(duracao: float, diretorio: Path = DIRETORIO_PADRAO,
          segundos_por_imagem: float = SEGUNDOS_POR_IMAGEM,
          cruzamento: float = CRUZAMENTO,
          seed: int | None = None) -> tuple[list[Path], float, float]:
    """(imagens sorteadas, segundos por imagem, cruzamento) para este audio."""
    acervo = disponiveis(diretorio)
    if not acervo:
        raise FileNotFoundError(
            f"nenhuma imagem em {diretorio} — use outro preset ou aponte "
            "--slides-dir para uma pasta com .jpg/.png")
    alvo = duracao + MARGEM_S
    n = quantas(alvo, segundos_por_imagem, cruzamento)
    # Inverte a formula do video encadeado para cobrir o audio exatamente.
    cada = (alvo + (n - 1) * cruzamento) / n
    return sortear(acervo, n, seed), cada, cruzamento


def veu_lavfi(fps: int, largura: int) -> str:
    """Fonte lavfi do veu: preto com alfa em rampa, desenhado uma unica vez.

    O `geq` custa uma passada por pixel, entao ele roda no primeiro quadro e
    para: `trim=end_frame=1` corta o resto e `loop` devolve aquele quadro para
    sempre. O `setpts` reconstroi os tempos, que o `loop` repete junto com a
    imagem -- sem ele o overlay recebe dezesseis mil quadros com PTS 0.

    A rampa vai de alfa 0 no topo a `VEU_OPACIDADE` no pe, com expoente
    `VEU_CURVA`. `rgba64le` (16 bits por canal) em vez de `rgba`: com o
    expoente 2,4 os primeiros ~90 px de altura mapeiam para alfa 0-3 em 8
    bits, e cada um desses valores cobre uma dezena de linhas -- um degrau
    real sobre fundo escuro, medido como listras horizontais na tarja
    esquerda/direita renderizada (era este o defeito relatado, nao a imagem
    de origem). Em 16 bits a rampa fica fina o bastante para o dither final
    do `render.py` (SAIDA_8BITS) espalhar o arredondamento em vez de deixar o
    degrau inteiro.
    """
    alfa = f"65535*{VEU_OPACIDADE}*pow(Y/H\\,{VEU_CURVA})"
    return (f"color=c=black:s={largura}x{VEU_ALTURA}:r={fps},format=rgba64le,"
            f"geq=r=0:g=0:b=0:a='{alfa}',"
            f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/{fps}/TB")


def _duracoes(cada: float | list[float], n: int) -> list[float]:
    """Normaliza `cada` (um numero, do preset `slides`, ou uma duracao por
    imagem, do preset `sincronizado`) para uma lista do tamanho certo."""
    if isinstance(cada, (int, float)):
        return [float(cada)] * n
    if len(cada) != n:
        raise ValueError(f"{len(cada)} duracoes para {n} imagens")
    return list(cada)


def entradas(imagens: list[Path], cada: float | list[float], fps: int,
             largura: int, veu: bool) -> list[str]:
    """Argumentos de entrada do ffmpeg: um bloco por imagem, mais o veu.

    `cada` aceita um numero unico (todas as imagens com a mesma duracao, uso
    do preset `slides`) ou uma duracao por imagem (preset `sincronizado`,
    onde cada janela de narracao tem um tamanho diferente)."""
    duracoes = _duracoes(cada, len(imagens))
    args: list[str] = []
    for img, d in zip(imagens, duracoes):
        args += ["-loop", "1", "-framerate", str(fps), "-t", f"{d:.3f}",
                 "-i", str(img)]
    if veu:
        args += ["-f", "lavfi", "-i", veu_lavfi(fps, largura)]
    return args


def filtro(imagens: list[Path], cada: float | list[float], cruzamento: float,
           largura: int, altura: int, veu: bool) -> str:
    """Grafo que normaliza cada imagem e as encadeia por dissolve, saindo em [v].

    As imagens do acervo sao quadradas ou panoramicas, quase nenhuma em 16:9.
    Cortar para caber jogaria fora quase metade de um 1024x1024, entao a imagem
    entra inteira e o resto do quadro recebe ela mesma, ampliada, desfocada e
    escurecida -- o mesmo material, sem borda preta e sem competir com a arte.

    O desfoque e feito pequeno e so depois esticado: `gblur` com sigma alto em
    1080p custa caro em TODO quadro de um still parado 45 s, e o resultado
    ampliado e indistinguivel.

    Mas nao TAO pequeno: em 192x108 um pixel do borrao virava um bloco de 10x10
    na tela, e o degrade que sobra tem passos largos demais para o `scale`
    interpolar -- eram essas as listras de cor que apareciam nas tarjas laterais.
    Em 480x270 o passo cai para 4 px e o borrao ainda sai de graca.

    A outra metade do conserto e `PROFUNDIDADE`: o borrao e uma rampa quase
    plana, e rampa plana em 8 bits BANDEIA por aritmetica, antes de qualquer
    codec. Em 10 bits ha degraus de sobra, e o `render` os despeja em 8 com
    difusao de erro no fim da cadeia.
    """
    duracoes = _duracoes(cada, len(imagens))
    partes: list[str] = []
    for i in range(len(imagens)):
        partes.append(
            f"[{i}:v]split=2[c{i}][f{i}];"
            f"[c{i}]scale={largura}:{altura}:force_original_aspect_ratio=increase,"
            f"crop={largura}:{altura},scale={BORRAO_L}:{BORRAO_A},"
            f"format={PROFUNDIDADE},gblur=sigma={BORRAO_SIGMA},"
            f"scale={largura}:{altura},eq=brightness=-0.18:saturation=0.5[b{i}];"
            f"[f{i}]scale={largura}:{altura}:force_original_aspect_ratio=decrease,"
            f"format={PROFUNDIDADE}[g{i}];"
            f"[b{i}][g{i}]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1[s{i}]")
    anterior = "[s0]"
    acumulado = 0.0
    for k in range(1, len(imagens)):
        acumulado += duracoes[k - 1]
        offset = acumulado - k * cruzamento
        saida = f"[x{k}]"
        partes.append(f"{anterior}[s{k}]xfade=transition=fade:"
                      f"duration={cruzamento:.3f}:"
                      f"offset={offset:.3f}{saida}")
        anterior = saida
    if veu:
        # O veu entra UMA vez, depois do encadeado: aplicado por imagem, cada
        # dissolve somaria dois veus e o rodape escureceria a cada troca.
        partes.append(f"{anterior}[{len(imagens)}:v]"
                      f"overlay=0:{altura - VEU_ALTURA}:format=auto[v]")
    else:
        partes.append(f"{anterior}null[v]")
    return ";".join(partes)
