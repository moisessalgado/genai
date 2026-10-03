"""Detecção de rosto nos quadros: YuNet (OpenCV Zoo, MIT), na CPU.

Dois usos: o letreiramento não põe balão em cima de rosto, e o QA confere que
o quadro tem pelo menos tantos rostos de primeiro plano quanto personagens.

Medido nos 18 candidatos do spike (ukiyo-e, Qwen Edit): todos os protagonistas
achados com nota 0,87–0,92, inclusive no panorâmico em que o príncipe ocupa
7% da altura; os figurantes da multidão também aparecem (rostos de 2% da
altura) — o que é bom para o balão e é filtrado por `MIN_ALTURA` no QA.
Milissegundos por quadro. Pesos de 230 KB baixados do Hugging Face para o
`hf_home` da máquina, como o CLIP do `imagem_qa`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.config import settings

REPO = "opencv/face_detection_yunet"
ARQUIVO = "face_detection_yunet_2023mar.onnx"
LIMIAR = 0.6
# Rosto "de primeiro plano" para o QA: altura mínima relativa à do quadro.
MIN_ALTURA = 0.035

_modelo: Path | None = None


@dataclass(frozen=True)
class Rosto:
    """Caixa normalizada (0–1) na imagem: vale em qualquer escala do quadro."""
    x: float
    y: float
    w: float
    h: float
    nota: float

    @property
    def centro(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


def _pesos() -> Path:
    global _modelo
    if _modelo is None:
        from huggingface_hub import hf_hub_download
        _modelo = Path(hf_hub_download(REPO, ARQUIVO, cache_dir=settings().hf_home / "hub"))
    return _modelo


def detectar(imagem: Path | Image.Image, limiar: float = LIMIAR) -> list[Rosto]:
    """Rostos da esquerda para a direita."""
    import cv2

    if isinstance(imagem, Image.Image):
        rgb = np.asarray(imagem.convert("RGB"))
    else:
        with Image.open(imagem) as im:
            rgb = np.asarray(im.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(_pesos()), "", (w, h), limiar, 0.3, 200)
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    out = [Rosto(max(0.0, f[0] / w), max(0.0, f[1] / h), f[2] / w, f[3] / h, float(f[-1]))
           for f in faces]
    return sorted(out, key=lambda r: r.x)


def principais(rostos: list[Rosto], min_altura: float = MIN_ALTURA) -> list[Rosto]:
    return [r for r in rostos if r.h >= min_altura]
