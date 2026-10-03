"""Analise objetiva de uma gravacao de referencia de voz (TDD 7.2).

A referencia e a identidade do canal: ela e congelada por sha256 e todo o audio
publicado depende dela. Um take ruim nao se conserta depois -- se o microfone
cortava em 8 kHz, todo o audiolivro sai com voz de telefone.

Por isso o registro nao aceita "parece bom": mede. As checagens abaixo pegam
justamente o que o ouvido do proprio gravador perdoa por estar acostumado.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Alvos vindos das instrucoes de gravacao (TDD 7.2 / `voice record`).
PICO_ALVO_DB = -6.0
PICO_MIN_DB = -20.0        # abaixo disto o sinal e fraco demais: sobe o ruido junto
PICO_MAX_DB = -1.0         # acima disto esta perto do teto, sem margem
RUIDO_MAX_DB = -50.0       # sala silenciosa
SNR_MIN_DB = 35.0
DC_MAX = 0.002
CLIP_MAX_FRACAO = 1e-5

# Fala tem modulacao silabica: silabas, pausas, respiracao. A medida e a distancia
# entre o percentil 90 e a mediana do nivel por janela -- as silabas fortes contra
# o corpo do sinal. Desvio-padrao NAO serve: ruido de sala afunda o desvio de uma
# fala perfeitamente audivel, e o diagnostico sai errado.
#
# Medido (janelas de 50 ms, DC removido):
#   fala narrada real, 40 chunks ....... 6,1 a 26,0  (mediana 8,0)
#   referencia narrador-v1 ............. 7,3
#   fala em sala barulhenta ............ 6,1
#   ruido de amplificador puro ......... 0,2
#   microfone mudo (caso real medido) .. 3,7
# 5,0 fica no meio do vao entre 3,7 e 6,1.
#
# E o teste que distingue "microfone mudo" de "microfone baixo". Sem ele um cabo
# solto aparece como uma lista de problemas de nivel, e a pessoa passa a tarde
# subindo ganho atras de sinal que nao existe.
MODULACAO_MIN_DB = 5.0

# Toda medida e feita na banda AUDIVEL. Sem isso, DC e infrassom entram em pico,
# ruido, SNR e modulacao, e os numeros deixam de falar sobre a voz. Caso real
# medido: uma gravacao com 97,5% da energia abaixo de 20 Hz reportava pico de
# 0,0 dBFS e clipping -- tudo rumble, com a voz 30 dB abaixo disso.
CORTE_SUBSONICO_HZ = 20.0
# Fracao de energia subsonica que denuncia defeito de cabo/aterramento.
RUMBLE_MAX_FRACAO = 0.30

# Um microfone de headset Bluetooth (HFP) corta em 4 ou 8 kHz. Voz masculina tem
# energia util ate ~12 kHz, e e a faixa de 8-14 kHz que da o "ar" da narracao.
CORTE_MIN_HZ = 11000.0
# Queda, em dB abaixo do pico do espectro, que conta como fim da banda util.
QUEDA_DB = 45.0


@dataclass
class Take:
    duracao_s: float
    sample_rate: int
    canais: int
    pico_db: float
    rms_db: float
    ruido_db: float
    snr_db: float
    dc: float
    clip_fracao: float
    corte_hz: float
    modulacao_db: float
    rumble_fracao: float

    @property
    def sem_voz(self) -> bool:
        """Sinal estacionario: nao ha fala nenhuma no arquivo."""
        return self.modulacao_db < MODULACAO_MIN_DB

    @property
    def problemas(self) -> list[str]:
        """Falhas que tornam a referencia inadequada. Vazio = pode registrar."""
        p = []
        if self.rumble_fracao > RUMBLE_MAX_FRACAO:
            # Vem antes de tudo: o rumble come o headroom e faz o pico e o
            # clipping mentirem sobre a voz.
            return [f"{self.rumble_fracao*100:.0f}% da energia abaixo de "
                    f"{CORTE_SUBSONICO_HZ:.0f} Hz — isso é rumble, não voz. "
                    "Plugue mal encaixado, cabo passando perto da fonte ou "
                    "problema de aterramento. Some o headroom todo e faz o pico "
                    "parecer bom quando não está"]
        if self.sem_voz:
            # Diagnostico primeiro: os problemas de nivel abaixo sao consequencia,
            # e listar todos junto manda a pessoa mexer no ganho a toa.
            return [f"sinal estacionário (modulação de {self.modulacao_db:.1f} dB, "
                    f"fala tem mais de {MODULACAO_MIN_DB:.0f}) — não há voz neste "
                    "arquivo: o microfone não está captando. Verifique o cabo, o "
                    "plugue e a chave de mudo do microfone, não o ganho"]
        if self.clip_fracao > CLIP_MAX_FRACAO:
            p.append(f"clipping em {self.clip_fracao*100:.2f}% das amostras — "
                     "abaixe o ganho de entrada e regrave")
        if self.pico_db > PICO_MAX_DB:
            p.append(f"pico {self.pico_db:.1f} dBFS, sem margem (alvo {PICO_ALVO_DB:.0f})")
        if self.pico_db < PICO_MIN_DB:
            p.append(f"pico {self.pico_db:.1f} dBFS, fraco demais (alvo {PICO_ALVO_DB:.0f}) — "
                     "aproxime o microfone ou suba o ganho")
        if self.ruido_db > RUIDO_MAX_DB:
            p.append(f"ruído de fundo {self.ruido_db:.1f} dBFS — sala/microfone ruidosos")
        if self.snr_db < SNR_MIN_DB:
            p.append(f"relação sinal/ruído {self.snr_db:.1f} dB (mínimo {SNR_MIN_DB:.0f})")
        if self.dc > DC_MAX:
            p.append(f"offset DC de {self.dc:.4f} — problema na placa de entrada")
        if self.corte_hz < CORTE_MIN_HZ:
            p.append(f"banda cortada em {self.corte_hz/1000:.1f} kHz — "
                     "microfone de telefone/Bluetooth? Use um microfone com fio")
        return p

    @property
    def avisos(self) -> list[str]:
        """Nao impedem o registro, mas valem uma segunda gravacao."""
        a = []
        if self.canais > 1:
            a.append(f"{self.canais} canais — será somado para mono")
        if self.sample_rate < 44100:
            a.append(f"{self.sample_rate} Hz — grave a 48 kHz")
        if abs(self.pico_db - PICO_ALVO_DB) > 4:
            a.append(f"pico {self.pico_db:.1f} dBFS, longe do alvo de {PICO_ALVO_DB:.0f}")
        return a


def analisar(caminho: Path) -> Take:
    audio, sr = sf.read(str(caminho), dtype="float32", always_2d=True)
    canais = audio.shape[1]
    mono = audio.mean(axis=1)
    return analisar_audio(mono, sr, canais)


def analisar_audio(mono: np.ndarray, sr: int, canais: int = 1) -> Take:
    if mono.size == 0:
        raise ValueError("arquivo de áudio vazio")
    # DC e clipping sao do sinal COMO GRAVADO -- e o conversor que satura, e o
    # rumble satura junto. O resto das medidas usa so a banda audivel.
    pico_bruto = float(np.abs(mono).max())

    clip = float((np.abs(mono) >= 0.99).mean())
    dc = float(abs(mono.mean()))
    rumble = _fracao_subsonica(mono, sr)
    mono = _passa_altas(mono, sr, CORTE_SUBSONICO_HZ)
    # Percentil, nao maximo: um unico estalo (o plugue, a boca, o inicio da
    # gravacao) define o maximo e faz um take de fala baixa parecer no nivel
    # certo. Medido: um take com fala entre -41 e -50 dBFS reportava pico de
    # -1,0 dBFS por causa de um clique nos primeiros 2 s.
    pico = float(np.percentile(np.abs(mono), 99.99))

    # Ruido de fundo: mediana das janelas mais silenciosas. Media nao serve --
    # uma unica pausa longa a puxaria para baixo e mascararia sala barulhenta.
    jan = max(1, int(sr * 0.05))
    n = (len(mono) // jan) * jan
    if n < jan * 4:
        raise ValueError("áudio curto demais para analisar")
    quadros = mono[:n].reshape(-1, jan)
    rms_quadro = np.sqrt((quadros.astype(np.float64) ** 2).mean(axis=1))
    ordenado = np.sort(rms_quadro)
    ruido = float(np.median(ordenado[:max(1, len(ordenado) // 10)]))
    fala = float(np.median(ordenado[-max(1, len(ordenado) // 4):]))

    # DC removido antes de medir modulacao: um offset preenche as pausas e
    # achataria o nivel, fazendo uma gravacao com defeito de placa parecer muda.
    sem_dc = mono - mono.mean()
    rms_sem_dc = np.sqrt((sem_dc[:n].reshape(-1, jan).astype(np.float64) ** 2).mean(axis=1))
    niveis_db = 20 * np.log10(np.maximum(rms_sem_dc, 1e-12))
    modulacao = float(np.percentile(niveis_db, 90) - np.percentile(niveis_db, 50))
    return Take(
        duracao_s=len(mono) / sr,
        sample_rate=sr,
        canais=canais,
        pico_db=_db(max(pico, 1e-12)),
        rms_db=_db(float(np.sqrt((mono.astype(np.float64) ** 2).mean()))),
        ruido_db=_db(ruido),
        snr_db=_db(fala) - _db(ruido),
        dc=dc,
        clip_fracao=clip,
        corte_hz=corte_espectral(mono, sr),
        modulacao_db=modulacao,
        rumble_fracao=rumble,
    )


def _fracao_subsonica(mono: np.ndarray, sr: int,
                     corte: float = CORTE_SUBSONICO_HZ) -> float:
    """Energia entre 0 Hz (exclusive) e a faixa audivel.

    O bin de 0 Hz fica de FORA porque offset DC e outro defeito, com outra causa
    e outro conserto -- juntar os dois daria o diagnostico errado. Medido no caso
    real: 1% em DC puro e 92% entre 0,1 e 10 Hz, com pico em 0,3 Hz. Deriva lenta
    de linha de base e contato, nao conversor.
    """
    X = np.abs(np.fft.rfft((mono - mono.mean()).astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    total = X.sum()
    return float(X[(freqs > 0) & (freqs < corte)].sum() / total) if total > 0 else 0.0


def _passa_altas(mono: np.ndarray, sr: int, corte: float) -> np.ndarray:
    """Passa-altas de fase zero por FFT. Suficiente para medir, e sem atraso."""
    X = np.fft.rfft(mono.astype(np.float64))
    X[np.fft.rfftfreq(len(mono), 1 / sr) < corte] = 0.0
    return np.fft.irfft(X, len(mono)).astype(np.float32)


def corte_espectral(mono: np.ndarray, sr: int, queda_db: float = QUEDA_DB) -> float:
    """Frequencia acima da qual nao ha mais sinal -- pega microfone de banda estreita.

    Procura a ultima faixa do espectro medio que ainda esta acima de (pico - queda).
    Um microfone Bluetooth despenca num degrau em 4 ou 8 kHz; um microfone decente
    vai perto de Nyquist. E a mesma tecnica usada para flagrar MP3 recodificado.
    """
    n = 4096
    if len(mono) < n * 2:
        n = 1 << max(8, int(np.log2(max(2, len(mono) // 2))))
    passo = n // 2
    janela = np.hanning(n).astype(np.float32)
    quadros = [mono[i:i + n] * janela
               for i in range(0, len(mono) - n, passo)]
    if not quadros:
        return sr / 2
    espectro = np.abs(np.fft.rfft(np.array(quadros), axis=1)).mean(axis=0)
    espectro_db = 20 * np.log10(np.maximum(espectro, 1e-12))
    freqs = np.fft.rfftfreq(n, 1 / sr)

    # ignora o gravissimo (rumble) ao procurar o pico de referencia
    util = freqs > 100
    limiar = espectro_db[util].max() - queda_db
    acima = np.where((espectro_db > limiar) & util)[0]
    return float(freqs[acima[-1]]) if acima.size else 0.0


def _db(x: float) -> float:
    return 20 * float(np.log10(max(x, 1e-12)))
