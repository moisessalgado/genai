"""Verificacao de identidade da voz (TDD 7.1, mecanismo 3).

O CER garante que o chunk diz o texto certo; nao garante que diz na voz certa.
Deriva de timbre e o defeito que o ouvinte percebe como "trocaram o narrador no
meio do livro", e passa despercebida por qualquer checagem textual.

Limiar calibrado com medicao propria (voz template narrador-v1, 6 chunks, 3 capitulos):
  - mesma voz .............. 0,917 a 0,960 (amplitude 0,042)
  - voz diferente (controle)  0,833
0,88 fica no meio do vao, longe das duas pontas.

MAS o limiar so vale para audio LONGO. Medido cortando os MESMOS chunks em varias
duracoes (12 chunks de narrador-v1 contra a referencia; controle: citacao-v1):

    corte |  mesma voz (min)  |  impostor (max)  |  vao
    ------+-------------------+------------------+-------
      2 s |       0,798       |      0,728       | +0,070
      3 s |       0,866       |      0,756       | +0,109
      4 s |       0,898       |      0,779       | +0,119
      6 s |       0,911       |      0,791       | +0,119
     10 s |       0,928       |      0,791       | +0,137
    cheio |       0,929       |      0,791       | +0,137

O vao continua aberto, mas a escala INTEIRA desce em audio curto -- impostor
incluido. Um limiar fixo de 0,88 nao acusa voz trocada: acusa audio curto. Foi o
que aconteceu no discurso do Krishnamurti, onde 5 chunks de 2 a 4 s com CER 0,000
(texto perfeito, voz certa) foram reprovados com 0,839 a 0,879.

Por isso o limiar acompanha a duracao, mantendo a mesma folga que 0,88 tem no
audio cheio (0,049 abaixo do piso da mesma voz). Abaixo de MIN_JULGAVEL o vao
fica estreito demais (+0,070) e a checagem simplesmente nao opina -- o CER
continua garantindo o conteudo desses trechos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SIMILARIDADE_MINIMA = 0.88

# Abaixo disto o embedding e instavel demais para julgar (vao de so +0,070).
MIN_JULGAVEL_S = 2.5
# Duracao a partir da qual a similaridade ja saturou e o limiar cheio vale.
DUR_PLENA_S = 8.0
# Limiar em MIN_JULGAVEL_S, interpolado linearmente ate SIMILARIDADE_MINIMA.
SIMILARIDADE_MINIMA_CURTA = 0.82


def limiar_por_duracao(duracao_s: float,
                       minima: float = SIMILARIDADE_MINIMA) -> float:
    """Limiar que acompanha a duracao, porque a escala de similaridade acompanha."""
    if duracao_s >= DUR_PLENA_S:
        return minima
    fracao = (duracao_s - MIN_JULGAVEL_S) / (DUR_PLENA_S - MIN_JULGAVEL_S)
    return SIMILARIDADE_MINIMA_CURTA + fracao * (minima - SIMILARIDADE_MINIMA_CURTA)


class SpeakerCheck:
    """Compara cada chunk com a referencia usando o voice encoder do proprio motor."""

    def __init__(self, voice_encoder, referencia: Path,
                 minima: float = SIMILARIDADE_MINIMA):
        self.ve = voice_encoder
        self.minima = minima
        self.ref = self._embed_path(referencia)

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        v = np.asarray(self.ve.embeds_from_wavs([audio], sample_rate=sample_rate)).squeeze()
        norma = np.linalg.norm(v)
        return v / norma if norma else v

    def _embed_path(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(str(path), dtype="float32")
        return self._embed(audio, sr)

    def similaridade(self, audio: np.ndarray, sample_rate: int) -> float:
        # trechos muito curtos dao embedding instavel; nao vale julgar
        if len(audio) / sample_rate < MIN_JULGAVEL_S:
            return 1.0
        return float(self.ref @ self._embed(audio, sample_rate))

    def ok(self, audio: np.ndarray, sample_rate: int) -> tuple[bool, float]:
        s = self.similaridade(audio, sample_rate)
        return s >= limiar_por_duracao(len(audio) / sample_rate, self.minima), s
