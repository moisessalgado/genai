"""Trilha de fundo: geracao propria e mixagem com ducking (TDD 18, V2).

Por que GERAR em vez de usar uma trilha pronta: o TDD e explicito sobre o risco
("Content ID do YouTube e implacavel"), e a unica trilha com licenca
inquestionavel e a que o proprio projeto cria. Nao ha o que reclamar sobre uma
onda senoidal.

O desenho e deliberadamente simples -- um pad de drone modal, sem ritmo e sem
melodia. Narração com melodia por baixo compete pela atencao do ouvinte; um leito
harmonico estatico preenche o silencio sem disputar. E, por nao ter ataque
percussivo, sobrevive bem ao ducking.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000

# Nivel da trilha sob a narracao; o ducking abaixa mais ainda enquanto ha fala.
#
# O caminho ate este numero vale ser lembrado. Medindo uma gravacao do mesmo
# sutta que o operador aprovou (recitacao em pali), a trilha de la fica 12,2 dB
# abaixo da fala, e isso levou a subir daqui para -20. O operador ouviu e pediu o
# contrario: mais baixa. A referencia media outro conteudo -- canto com pausas de
# 5 s, onde a musica tem espaco para existir sozinha. Sob prosa continua em
# portugues, o mesmo nivel atrapalha.
#
# -29,5 e escolha de ouvido do operador entre tres opcoes medidas (-23,5, -26 e
# -29,5), e nao vem de nenhum alvo numerico. Medicao serviu para OFERECER as
# opcoes em passos conhecidos; quem decidiu foi a escuta.
TRILHA_LUFS = -29.5
DUCK_REDUCAO_DB = 9.0

# Grau da escala em semitons a partir da tonica. Modo dorico: menor, mas sem a
# sexta menor, o que evita o tom sombrio do eolio numa peca longa.
_DORICO = (0, 2, 3, 5, 7, 9, 10)

# Acordes como graus da escala, em ciclo lento. Quatro acordes, sem tensao nem
# resolucao forte: a trilha nao pode ter "eventos" que puxem a atencao.
_CICLO = ((0, 2, 4), (5, 0, 2), (3, 5, 0), (4, 6, 1))


def _nota_hz(tonica_hz: float, grau: int) -> float:
    oitava, passo = divmod(grau, len(_DORICO))
    return tonica_hz * (2 ** oitava) * (2 ** (_DORICO[passo] / 12))


# Proporcao de potencia entre as duas camadas. Ajustada medindo o espectro da
# narracao deste canal: o leito grave sustenta, o brilho garante que a trilha
# exista em fone de celular, que nao reproduz nada abaixo de 100 Hz.
PESO_GRAVE = 0.55
PESO_BRILHO = 0.45
OITAVAS_BRILHO = 4


def gerar_ambiente(duracao_s: float, tonica_hz: float = 55.0,
                   sample_rate: int = SAMPLE_RATE, compasso_s: float = 24.0,
                   seed: int = 7) -> np.ndarray:
    """Pad de drone modal, sem ritmo e sem melodia.

    Duas camadas do mesmo acorde, geradas e normalizadas em separado: o leito
    grave, abaixo da fundamental da voz, e um brilho quatro oitavas acima, onde a
    fala quase nao tem energia. Misturar por potencia medida, e nao por peso
    solto, e o que torna o equilibrio previsivel -- o grave domina qualquer soma
    ingenua, porque potencia cresce com a amplitude ao quadrado.
    """
    grave = _camada(duracao_s, tonica_hz, 0, sample_rate, compasso_s, seed)
    brilho = _camada(duracao_s, tonica_hz, OITAVAS_BRILHO * len(_DORICO),
                     sample_rate, compasso_s, seed + 1)
    mistura = PESO_GRAVE * _por_rms(grave) + PESO_BRILHO * _por_rms(brilho)
    # A mistura e por RMS (potencia audivel), mas a SAIDA e limitada por PICO:
    # normalizar por RMS nao limita o pico, e este pad tem fator de crista ~4 --
    # media casada, pico em +6 dBFS, ou seja, estourando.
    return (_por_pico(_moldar_para_voz(mistura, sample_rate), 0.5)).astype(np.float32)


def _por_rms(x: np.ndarray) -> np.ndarray:
    """Normaliza por potencia: e o que torna a mistura das camadas previsivel."""
    rms = np.sqrt((x ** 2).mean())
    return x / rms if rms > 0 else x


def _por_pico(x: np.ndarray, alvo: float) -> np.ndarray:
    pico = np.abs(x).max()
    return x * (alvo / pico) if pico > 0 else x


def _camada(duracao_s: float, tonica_hz: float, transposicao: int, sample_rate: int,
            compasso_s: float, seed: int) -> np.ndarray:
    """Uma camada do ciclo de acordes, transposta por N graus da escala."""
    rng = np.random.default_rng(seed)
    n = int(duracao_s * sample_rate)
    t = np.arange(n) / sample_rate
    saida = np.zeros(n, dtype=np.float64)
    repeticoes = int(duracao_s / (compasso_s * len(_CICLO))) + 2
    for i, acorde in enumerate(_CICLO * repeticoes):
        ini = i * compasso_s
        if ini >= duracao_s:
            break
        env = _envelope(t, ini, compasso_s)
        if not env.any():
            continue
        for grau in acorde:
            f = _nota_hz(tonica_hz, grau + transposicao)
            if f > sample_rate / 4:
                continue
            for parcial, peso in ((1, 1.0), (2, 0.35), (3, 0.12), (4, 0.06)):
                # A desafinacao entre parciais produz o batimento lento que o
                # ouvido le como textura, e nao como tom eletronico parado.
                desafino = 1 + rng.uniform(-0.0016, 0.0016)
                fase = rng.uniform(0, 2 * np.pi)
                lfo = 1 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.02, 0.06) * t + fase)
                saida += env * peso * lfo * np.sin(
                    2 * np.pi * f * parcial * desafino * t + fase)
    return saida


def _envelope(t: np.ndarray, ini: float, compasso_s: float) -> np.ndarray:
    """Rampa de cosseno subindo e descendo, com sobreposicao de meio compasso."""
    dentro = (t >= ini - compasso_s / 2) & (t < ini + compasso_s * 1.5)
    e = np.zeros_like(t)
    if not dentro.any():
        return e
    x = (t[dentro] - (ini - compasso_s / 2)) / (compasso_s * 2)
    e[dentro] = 0.5 - 0.5 * np.cos(2 * np.pi * x)
    return e


# Faixa em que a fala vive e a trilha precisa sair da frente. Medido na narracao
# deste canal (2 min do sutta): 49,5% da energia entre 300 e 700 Hz, fundamental
# em 150 Hz, e quase nada abaixo de 80 Hz ou acima de 3 kHz.
VOZ_INI_HZ = 120.0
VOZ_FIM_HZ = 1400.0
VOZ_ATENUACAO_DB = 15.0
BRILHO_FIM_HZ = 6000.0


def _moldar_para_voz(x: np.ndarray, sr: int) -> np.ndarray:
    """Deixa a trilha embaixo e em volta da voz, nunca em cima dela.

    Uma trilha "bonita sozinha" costuma ser a que mais atrapalha: ela ocupa a
    mesma regiao das consoantes. Aqui a faixa da fala e deprimida de proposito, e
    o que sobra e grave (abaixo da fundamental) mais um fio de brilho no agudo.
    """
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)

    # depressao suave na faixa da fala, em forma de cosseno levantado
    ganho = np.ones_like(f)
    faixa = (f >= VOZ_INI_HZ) & (f <= VOZ_FIM_HZ)
    pos = (np.log(f[faixa] / VOZ_INI_HZ) / np.log(VOZ_FIM_HZ / VOZ_INI_HZ))
    fundo = 10 ** (-VOZ_ATENUACAO_DB / 20)
    ganho[faixa] = 1 - (1 - fundo) * (0.5 - 0.5 * np.cos(2 * np.pi * pos)) ** 0.5
    ganho[(f > VOZ_FIM_HZ)] = fundo + (1 - fundo) * 0.35   # brilho parcial
    ganho *= 1 / (1 + (f / BRILHO_FIM_HZ) ** 4)            # sem chiado no topo
    return np.fft.irfft(X * ganho, len(x))


def preparar_trilha(destino: Path, duracao_s: float, fonte: Path | None = None,
                    sample_rate: int = SAMPLE_RATE) -> Path:
    """Deixa em `destino` uma trilha do tamanho exato da narracao.

    Sem `fonte`, gera. Com `fonte`, repete o arquivo do operador ate cobrir a
    duracao -- a licenca dele e responsabilidade de quem publica, e o `export`
    ecoa isso no aviso.
    """
    import soundfile as sf

    destino.parent.mkdir(parents=True, exist_ok=True)
    if fonte is None:
        sf.write(str(destino), gerar_ambiente(duracao_s, sample_rate=sample_rate),
                 sample_rate)
        return destino
    _ffmpeg(["-stream_loop", "-1", "-i", str(fonte), "-t", f"{duracao_s:.3f}",
             "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s24le", str(destino)])
    return destino


def mixar(narracao: Path, trilha: Path, destino: Path,
          trilha_lufs: float = TRILHA_LUFS,
          reducao_db: float = DUCK_REDUCAO_DB,
          sample_rate: int | None = None) -> Path:
    """Mixa narracao e trilha com ducking pela propria narracao (sidechain).

    Ducking e obrigatorio, nao decorativo: uma trilha de nivel fixo que caiba nas
    pausas fica alta demais sob a fala, e uma que caiba sob a fala some nas
    pausas. O sidechain resolve os dois com um numero so.

    A narracao entra em ganho unitario e NAO e comprimida: ela ja saiu do
    loudnorm no alvo, e comprimir de novo achataria a dinamica da leitura.
    """
    # `asplit` e obrigatorio: a narracao e consumida DUAS vezes (como sinal e como
    # chave do sidechain), e o ffmpeg nao ramifica um stream sozinho. Sem ele o
    # grafo entrega um resultado silenciosamente errado -- medido: com a trilha em
    # silencio, a saida vinha 4,5 dB acima da narracao de entrada.
    filtro = (
        "[0:a]asplit=2[narr][chave];"
        f"[1:a]loudnorm=I={trilha_lufs}:TP=-3:LRA=7[t];"
        # a chave do ducking: a narracao controla o ganho da trilha
        f"[t][chave]sidechaincompress="
        f"threshold=0.02:ratio={10 ** (reducao_db / 20):.1f}:attack=25:release=900"
        f":makeup=1[duck];"
        "[narr][duck]amix=inputs=2:duration=first:normalize=0[mix]"
    )
    # A taxa de saida e FIXADA de proposito: o `loudnorm` do ffmpeg reamostra
    # para 192 kHz internamente, e sem `-ar` o arquivo sai a 192 kHz. Nao quebra
    # o audio, mas multiplica o tamanho por oito -- e arruina qualquer comparacao
    # amostra a amostra com a narracao de entrada.
    import soundfile as sf

    sr = sample_rate or sf.info(str(narracao)).samplerate
    _ffmpeg(["-i", str(narracao), "-i", str(trilha), "-filter_complex", filtro,
             "-map", "[mix]", "-ar", str(sr), "-c:a", "pcm_s24le", str(destino)])
    return destino


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                   capture_output=True, text=True, check=True)
