"""Utilidades de mídia comuns às áreas (ffprobe/ffmpeg como ferramenta externa)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def duracao(path: Path) -> float:
    """Duração em segundos de qualquer arquivo de áudio ou vídeo."""
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip())
