"""Pos-processamento e montagem de audio (TDD 9).

Principio: processar o MINIMO. A saida do TTS ja e limpa, sem ruido de sala e sem
clipping -- denoise e EQ agressivo so degradam. A cadeia e curta de proposito:
trim -> juncao com crossfade -> pausas -> high-pass -> loudnorm 2-pass -> limiter.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Pausas estruturais, em milissegundos (silencio digital, nao gerado pelo TTS)
# 400 ms era ritmo de audiolivro comum. Estes textos pedem outra coisa: o
# operador pediu tempo para o ouvinte compreender o que foi dito, e 900 ms foi o
# valor das amostras que ele aprovou. Vem junto com a voz `narrador-v2`, mais
# lenta -- pausa e velocidade de fala sao alavancas separadas, e so as duas
# juntas mudam a sensacao de pressa.
PAUSA_PARAGRAFO_MS = 900
PAUSA_SECAO_MS = 1000
PAUSA_CAPITULO_MS = 1800

CROSSFADE_MS = 15
TRIM_DB = -45.0
TRIM_MARGEM_MS = 30

# Alvo de loudness: o YouTube normaliza para cerca de -14 LUFS; entregar -16 evita
# que a plataforma reduza o ganho e preserva a dinamica da narracao.
LUFS_ALVO = -16.0
TP_ALVO = -1.5
LRA_ALVO = 11.0


@dataclass
class Segmento:
    audio: np.ndarray
    pausa_depois_ms: int = PAUSA_PARAGRAFO_MS


def trim_silencio(audio: np.ndarray, sample_rate: int, limiar_db: float = TRIM_DB,
                  margem_ms: int = TRIM_MARGEM_MS) -> np.ndarray:
    """Remove silencio das bordas -- chunks de TTS tem ataque e cauda irregulares."""
    if audio.size == 0:
        return audio
    limiar = 10 ** (limiar_db / 20)
    acima = np.abs(audio) > limiar
    if not acima.any():
        return audio[:0]
    margem = int(sample_rate * margem_ms / 1000)
    ini = max(0, int(np.argmax(acima)) - margem)
    fim = min(len(audio), len(audio) - int(np.argmax(acima[::-1])) + margem)
    return audio[ini:fim]


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Junta dois trechos com rampa curta -- elimina o clique de emenda."""
    n = min(n, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    rampa = np.linspace(0.0, 1.0, n, dtype=np.float32)
    meio = a[-n:] * (1 - rampa) + b[:n] * rampa
    return np.concatenate([a[:-n], meio, b[n:]])


def montar(segmentos: list[Segmento], sample_rate: int,
           crossfade_ms: int = CROSSFADE_MS) -> np.ndarray:
    """Concatena segmentos com crossfade e insere as pausas estruturais."""
    if not segmentos:
        return np.zeros(0, dtype=np.float32)
    n_cf = int(sample_rate * crossfade_ms / 1000)
    out = trim_silencio(segmentos[0].audio, sample_rate)
    for i, seg in enumerate(segmentos):
        if i > 0:
            out = _crossfade(out, trim_silencio(seg.audio, sample_rate), n_cf)
        if seg.pausa_depois_ms and i < len(segmentos) - 1:
            n = int(sample_rate * seg.pausa_depois_ms / 1000)
            out = np.concatenate([out, np.zeros(n, dtype=np.float32)])
    return out.astype(np.float32)


# Uma pausa interna de fala: curta demais e e so a oclusiva de um /p/, longa
# demais e ja e outro segmento. Entre 120 e 700 ms fica a virgula e o ponto.
PAUSA_INTERNA_MIN_S = 0.12
PAUSA_INTERNA_DB = -38.0


def pausas_internas(audio: np.ndarray, sample_rate: int) -> list[float]:
    """Centros dos silencios curtos dentro de um trecho, em segundos.

    Servem para a legenda quebrar onde a VOZ quebra. Repartir uma legenda longa
    por contagem de caracteres parece razoavel e nao e: fala nao tem taxa
    constante de caracteres por segundo, e o erro aparece como legenda adiantada.
    """
    h = max(1, int(sample_rate * 0.02))
    n = len(audio) // h
    if n < 2:
        return []
    quadros = audio[:n * h].reshape(n, h)
    db = 20 * np.log10(np.sqrt((quadros.astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
    baixo = db < PAUSA_INTERNA_DB
    saida, ini = [], None
    for i, b in enumerate(baixo):
        if b and ini is None:
            ini = i
        elif not b and ini is not None:
            if (i - ini) * h / sample_rate >= PAUSA_INTERNA_MIN_S:
                saida.append((ini + i) / 2 * h / sample_rate)
            ini = None
    return saida


def montar_com_marcas(segmentos: list[Segmento], sample_rate: int,
                      crossfade_ms: int = CROSSFADE_MS
                      ) -> tuple[np.ndarray, list[tuple[float, float]], list[list[float]]]:
    """Como `montar`, mas devolve tambem onde cada segmento caiu, em segundos.

    Os tempos precisam sair DAQUI, e nao de somar as duracoes dos chunks: o
    `trim_silencio` encurta cada um por uma quantidade diferente, e o crossfade
    engole `crossfade_ms` a cada emenda. Somar as duracoes do banco erra alguns
    segundos ao longo de um capitulo -- o bastante para a legenda descolar da
    fala.
    """
    if not segmentos:
        return np.zeros(0, dtype=np.float32), [], []
    n_cf = int(sample_rate * crossfade_ms / 1000)
    primeiro = trim_silencio(segmentos[0].audio, sample_rate)
    out = primeiro
    marcas = [(0, len(out))]
    pausas = [pausas_internas(primeiro, sample_rate)]
    for i, seg in enumerate(segmentos):
        if i > 0:
            b = trim_silencio(seg.audio, sample_rate)
            n = min(n_cf, len(out), len(b))
            ini = len(out) - n
            out = _crossfade(out, b, n_cf)
            marcas.append((ini, ini + len(b)))
            pausas.append([ini / sample_rate + t
                           for t in pausas_internas(b, sample_rate)])
        if seg.pausa_depois_ms and i < len(segmentos) - 1:
            n = int(sample_rate * seg.pausa_depois_ms / 1000)
            out = np.concatenate([out, np.zeros(n, dtype=np.float32)])
    marcas_s = [(a / sample_rate, b / sample_rate) for a, b in marcas]
    for k, (a, _) in enumerate(marcas):
        pausas[k] = [a / sample_rate + t for t in pausas[k]] if k == 0 else pausas[k]
    return out.astype(np.float32), marcas_s, pausas


def _ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True, check=True)


def duracao(path: Path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip())


def concatenar(partes: list[Path], destino: Path) -> Path:
    """Junta os masters num arquivo continuo, sem recodificar.

    Cada capitulo ja saiu do loudnorm no mesmo alvo, entao concatenar nao muda o
    loudness -- e o `chapters.txt` so faz sentido contra este arquivo unico.
    """
    lista = destino.with_suffix(".concat.txt")
    lista.write_text("".join(f"file '{p.resolve()}'\n" for p in partes),
                     encoding="utf-8")
    try:
        _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lista),
                 "-c", "copy", str(destino)])
    finally:
        lista.unlink(missing_ok=True)
    return destino


def medir_loudness(path: Path) -> dict:
    """Primeira passada do loudnorm: mede para a segunda passada corrigir com precisao."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"loudnorm=I={LUFS_ALVO}:TP={TP_ALVO}:LRA={LRA_ALVO}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True)
    saida = proc.stderr
    ini = saida.rfind("{")
    if ini == -1:
        raise RuntimeError(f"loudnorm nao retornou medicao: {saida[-400:]}")
    return json.loads(saida[ini:saida.rfind("}") + 1])


def masterizar(entrada: Path, saida: Path, sample_rate_saida: int = 48000,
               high_pass_hz: int = 65, fade_in_ms: int = 30,
               fade_out_ms: int = 300) -> dict:
    """Cadeia final: high-pass -> loudnorm 2-pass -> limiter -> fades -> resample."""
    m = medir_loudness(entrada)
    dur = duracao(entrada)

    loudnorm = (
        f"loudnorm=I={LUFS_ALVO}:TP={TP_ALVO}:LRA={LRA_ALVO}"
        f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}:linear=true:print_format=summary"
    )
    filtros = [
        f"highpass=f={high_pass_hz}",
        loudnorm,
        f"alimiter=limit={TP_ALVO}dB:level=disabled",
        f"afade=t=in:st=0:d={fade_in_ms/1000}",
        f"afade=t=out:st={max(0.0, dur - fade_out_ms/1000)}:d={fade_out_ms/1000}",
        f"aresample={sample_rate_saida}",
    ]
    _ffmpeg(["-i", str(entrada), "-af", ",".join(filtros),
             "-c:a", "pcm_s24le", str(saida)])
    _corrigir_ganho(saida)
    return m


# Desvio a partir do qual vale corrigir o ganho do master (dB LUFS).
TOLERANCIA_LUFS = 0.1


def _corrigir_ganho(master: Path, tolerancia: float = TOLERANCIA_LUFS) -> float:
    """Mede o master e corrige o residuo com ganho linear.

    Motivo medido: quando o ganho necessario estoura o true peak alvo, o `loudnorm`
    abandona o modo linear e cai no dinamico, e o resultado fica sistematicamente
    ~0,5 LUFS abaixo do alvo -- fora da tolerancia de +-0,5 do TDD. Corrigir com
    `volume` e seguro porque ganho linear nao altera a dinamica; o limiter fica
    depois so como guarda de true peak.
    """
    medido = float(medir_loudness(master)["input_i"])
    delta = LUFS_ALVO - medido
    if abs(delta) <= tolerancia:
        return medido
    tmp = master.with_suffix(".corr.wav")
    _ffmpeg(["-i", str(master), "-af",
             f"volume={delta:+.2f}dB,alimiter=limit={TP_ALVO}dB:level=disabled",
             "-c:a", "pcm_s24le", str(tmp)])
    tmp.replace(master)
    return float(medir_loudness(master)["input_i"])


def exportar(master: Path, destino: Path, formato: str, metadados: dict | None = None,
             bitrate: str = "192k") -> Path:
    """Gera o entregavel a partir do master. Voz nao precisa mais que 192 kbps."""
    args = ["-i", str(master)]
    for k, v in (metadados or {}).items():
        args += ["-metadata", f"{k}={v}"]
    if formato == "mp3":
        args += ["-c:a", "libmp3lame", "-b:a", bitrate]
    elif formato == "aac":
        args += ["-c:a", "aac", "-b:a", bitrate]
    elif formato == "flac":
        args += ["-c:a", "flac"]
    elif formato == "wav":
        args += ["-c:a", "pcm_s24le"]
    else:
        raise ValueError(f"formato nao suportado: {formato}")
    _ffmpeg([*args, str(destino)])
    return destino


def salvar_wav(audio: np.ndarray, path: Path, sample_rate: int,
               subtype: str = "PCM_24") -> None:
    sf.write(str(path), audio, sample_rate, subtype=subtype)


def chapters_txt(duracoes: list[tuple[str, float]]) -> str:
    """Timestamps no formato que o YouTube aceita na descricao do video."""
    linhas, t = [], 0.0
    for titulo, dur in duracoes:
        h, resto = divmod(int(t), 3600)
        m, s = divmod(resto, 60)
        carimbo = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        linhas.append(f"{carimbo} {titulo}")
        t += dur
    return "\n".join(linhas) + "\n"
