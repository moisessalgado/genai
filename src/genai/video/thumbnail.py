"""Miniatura automatica: um frame do proprio MP4, sem etapa manual no meio do
fluxo criacao->publicacao (TDD 18).

Nao usa `zoompan`/cena mais "interessante": o preset `slides` nao tem cena, so
uma imagem parada por vez com dissolve de 2 s entre elas (ver `video/slides.py`).
Um frame tirado durante o dissolve sai borrado, entao a escolha do instante
evita a janela de cruzamento em vez de sortear.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..audio.process import duracao
from . import slides as slides_mod

# YouTube: minimo 640x360, e recusa arquivo abaixo de ~1 KB.
_QUALIDADE_JPEG = 3

_SRT_TS = re.compile(r"(\d+):(\d+):(\d+),(\d+)")


def _ts_para_segundos(ts: str) -> float:
    h, m, s, ms = _SRT_TS.match(ts).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _cues(srt: Path) -> list[tuple[float, float]]:
    cues = []
    for bloco in srt.read_text(encoding="utf-8").strip().split("\n\n"):
        for linha in bloco.splitlines():
            if "-->" in linha:
                ini, fim = linha.split("-->")
                cues.append((_ts_para_segundos(ini.strip()), _ts_para_segundos(fim.strip())))
                break
    return cues


def instante_seguro(video: Path, preset: str = "slides", srt: Path | None = None) -> float:
    """Escolhe um instante (s) fora de qualquer dissolve entre slides e, se um
    `.srt` for passado, fora de qualquer legenda na tela -- um frame parado com
    uma legenda pela metade nao serve de miniatura sozinho.

    Para o preset `slides`, o meio do primeiro slide -- metade de
    `SEGUNDOS_POR_IMAGEM` -- fica sempre fora da janela de cruzamento
    (`CRUZAMENTO`) por construcao (TDD 18: `t >= 2*d`). Nos demais presets nao
    ha corte a evitar; 10% da duracao so foge de fade-in, se houver.
    """
    total = duracao(video)
    if preset == "slides":
        alvo = slides_mod.SEGUNDOS_POR_IMAGEM / 2
        janela = (slides_mod.CRUZAMENTO, slides_mod.SEGUNDOS_POR_IMAGEM - slides_mod.CRUZAMENTO)
    else:
        alvo = total * 0.10
        janela = (0.0, total)
    alvo = min(alvo, max(total - 0.5, 0.0))

    if srt and srt.exists():
        cues = _cues(srt)

        def coberto(t: float) -> bool:
            return any(ini <= t <= fim for ini, fim in cues)

        if coberto(alvo):
            passo = 0.5
            candidato = alvo
            while candidato < janela[1]:
                candidato += passo
                if not coberto(candidato):
                    return candidato
            candidato = alvo
            while candidato > janela[0]:
                candidato -= passo
                if not coberto(candidato):
                    return candidato

    return alvo


def extrair_frame(video: Path, destino: Path, instante: float) -> Path:
    """Salva um JPEG do frame em `instante` segundos. `-ss` antes de `-i` busca
    pelo keyframe mais proximo, o que basta para uma miniatura -- nao e preciso
    decodificar o arquivo inteiro so para um frame."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(instante), "-i", str(video),
         "-frames:v", "1", "-q:v", str(_QUALIDADE_JPEG), str(destino)],
        capture_output=True, text=True, check=True)
    return destino
