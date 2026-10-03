"""Verificacao automatica de cada chunk gerado (TDD 8, "QA LOOP").

Motivo de existir: o Chatterbox e autoregressivo e falha de forma RARA e SILENCIOSA
-- repete, alucina ou trunca. Na Fase 0 o proprio alignment_stream_analyzer do modelo
detectou repeticao e FORCOU EOS, ou seja, entregou audio incompleto sem erro.
Por isso verificamos duas coisas independentes:

  1. CONTEUDO  -- re-transcreve com ASR e compara com o texto esperado (CER);
  2. DURACAO   -- compara a duracao real com a esperada para o numero de caracteres;
  3. AR MORTO  -- procura buracos longos ENTRE os segmentos que o ASR devolve.

A checagem de duracao e a que pega o truncamento: cortar o fim de uma frase muda
pouco o CER de um texto longo, mas mutila a narracao.

A de ar morto pega um defeito que escapa das outras duas: o modelo diz o texto
inteiro, para de falar no meio, e continua gerando chiado por segundos. O CER
fica otimo (o texto esta la) e a duracao total fica dentro da faixa, mas o
ouvinte escuta um buraco com ruido de fundo. Foi assim que 7,2 s de ar morto
foram parar num audio ja publicado, e so um ouvido humano notou.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

# Duracao esperada = OVERHEAD fixo + texto / ritmo. O overhead (ataque, respiracao,
# pausa final) NAO encolhe com o texto, entao um piso fixo de caracteres por segundo
# reprova todo chunk curto que esta perfeito.
#
# Ajustado sobre 126 chunks aprovados do discurso do Krishnamurti (28 a 298 chars):
#   dur = 0,83 s + chars / 15,4     (residuo: desvio-padrao 0,79 s)
# Nessa amostra a razao dur/esperada ficou entre 0,79 e 1,30.
OVERHEAD_S = 0.85
CPS_NOMINAL = 15.4

# Truncamento e o que esta checagem existe para pegar, e ele discrimina bem: cortar
# metade de um chunk da razao ~0,50, longe do piso 0,79 observado.
FATOR_MIN = 0.6

# O lado "lento demais" e generoso de proposito. Medido: um titulo curto lido com
# pausa deu razao 1,70 com CER 0,000 (audio correto), enquanto o arrasto real
# documentado no ESTADO deu 1,54 -- as duas populacoes se sobrepoem, e duracao
# sozinha nao as separa. Quem cuida de conteudo aqui e o CER; a duracao so barra
# o que e grosseiro (loop de repeticao passa MUITO de 1,8).
FATOR_MAX = 1.8

CER_MAX = 0.05

# Maior buraco tolerado entre segmentos do ASR. Medido em 185 chunks aprovados de
# dois projetos: mediana 0,68 s, p90 1,34 s, p99 2,58 s, com o maior legitimo em
# 2,30 s. Os defeitos reais apareceram isolados em 4,06 s e 7,22 s. O limite fica
# no vao entre as duas populacoes.
#
# Usa os tempos dos segmentos, e nao o nivel do audio, porque o ar morto do
# Chatterbox NAO e silencio: e chiado a -45 dB com picos a -38 dB, so 14 dB
# abaixo da fala. Qualquer limiar de nivel que o pegue tambem reprova pausa
# normal. Os tempos do ASR resolvem isso sem limiar de amplitude, e de graca --
# o ASR ja roda em todo chunk para medir o CER.
SILENCIO_MAX_S = 3.0


def duracao_esperada(texto: str) -> float:
    return OVERHEAD_S + len(texto) / CPS_NOMINAL


@dataclass
class QAResult:
    ok: bool
    cer: float
    transcript: str
    duration_s: float
    cps: float
    reason: str = ""


def _normalize_for_compare(text: str) -> str:
    """Reduz ambos os lados ao que importa: letras e espacos.

    Antes disso, o texto passa pelo normalizador de narracao. Sem esse passo o QA
    gera falso positivo sistematico: o Whisper faz a normalizacao INVERSA -- ouve
    "mil seiscentos e quarenta e oito" e transcreve "1648" -- e a comparacao
    penalizaria exatamente os numeros que o pipeline acabou de expandir.
    """
    from ..narration.rules import normalize as _narr

    try:
        text = _narr(text).text
    except Exception:
        pass
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def char_error_rate(esperado: str, obtido: str) -> float:
    """Levenshtein normalizado pelo comprimento do esperado."""
    a, b = _normalize_for_compare(esperado), _normalize_for_compare(obtido)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


class Verifier:
    """Wrapper do faster-whisper. Carrega o modelo uma vez por processo."""

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", language: str = "pt",
                 cpu_threads: int = 8):
        # CPU por padrao de proposito: o CTranslate2 exige cuBLAS 12 e o ambiente
        # e CUDA 13 (misturar cu12/cu13 no mesmo venv quebra o torch). Alem disso,
        # deixar o ASR na CPU libera a GPU inteira para o TTS -- que e o gargalo.
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.cpu_threads = cpu_threads
        self.model = None

    def load(self) -> None:
        if self.model is not None:      # idempotente: o worker chama sem saber
            return
        from faster_whisper import WhisperModel

        self.model = WhisperModel(self.model_size, device=self.device,
                                  compute_type=self.compute_type,
                                  cpu_threads=self.cpu_threads)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.transcribe_com_tempos(audio, sample_rate)[0]

    def transcribe_com_tempos(self, audio: np.ndarray,
                              sample_rate: int) -> tuple[str, float]:
        """Devolve (texto, maior buraco sem fala em segundos).

        O buraco inclui a cauda depois do ultimo segmento: um chunk que para de
        falar e continua chiando ate o fim tem o defeito no fim, nao no meio.
        """
        dur = len(audio) / sample_rate
        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)
        segments, _ = self.model.transcribe(audio, language=self.language,
                                            beam_size=1, vad_filter=False)
        partes, fim, maior = [], 0.0, 0.0
        for s in segments:
            maior = max(maior, s.start - fim)
            fim = s.end
            partes.append(s.text)
        return " ".join(partes).strip(), max(maior, dur - fim)

    def check(self, audio: np.ndarray, sample_rate: int, esperado: str,
              cer_max: float = CER_MAX) -> QAResult:
        dur = len(audio) / sample_rate
        cps = len(esperado) / dur if dur > 0 else float("inf")
        prevista = duracao_esperada(esperado)
        razao = dur / prevista

        # Duracao primeiro: e barata e pega o truncamento sem rodar o ASR.
        if dur < 0.2:
            return QAResult(False, 1.0, "", dur, cps, "audio vazio ou quase vazio")
        if razao < FATOR_MIN:
            return QAResult(False, 1.0, "", dur, cps,
                            f"audio curto demais para o texto ({dur:.1f}s contra "
                            f"{prevista:.1f}s previstos) - truncado?")
        if razao > FATOR_MAX:
            return QAResult(False, 1.0, "", dur, cps,
                            f"audio longo demais para o texto ({dur:.1f}s contra "
                            f"{prevista:.1f}s previstos) - loop/arrasto?")

        transcript, buraco = self.transcribe_com_tempos(audio, sample_rate)
        if buraco > SILENCIO_MAX_S:
            return QAResult(False, 1.0, transcript, dur, cps,
                            f"ar morto de {buraco:.1f}s sem fala - o modelo parou "
                            "no meio e continuou gerando ruido")
        cer = char_error_rate(esperado, transcript)
        if cer > cer_max:
            return QAResult(False, cer, transcript, dur, cps,
                            f"CER {cer:.3f} acima do limite {cer_max}")
        return QAResult(True, cer, transcript, dur, cps)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Reamostragem linear -- suficiente para ASR de verificacao."""
    n_out = int(round(len(audio) * sr_out / sr_in))
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out, dtype=np.float64),
        np.arange(len(audio), dtype=np.float64),
        audio,
    ).astype(np.float32)
