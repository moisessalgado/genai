"""Projeto de HQ: project.yaml (o mesmo que o `publish` lê) + roteiro.yaml + state.db.

Layout em `projects/<slug>/`:

    project.yaml         título, rights, narrador e elenco de vozes (motion comic)
    fonte.txt            texto de origem (cópia), se houver
    roteiro.yaml         hq/roteiro.py
    state.db             hq_itens (e os chunks do audiolivro, no motion comic)
    elenco/<pid>/        cand-*.png (FLUX), aprovado.png, limpo-*.png, ref.png
    quadros/<id>/        cand-*.png (Qwen Edit)
    letreiro/            quadros letreirados
    paginas/             páginas diagramadas
    output/              PDF, CBZ, webtoon/, MP4
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml

from ..core.projeto import dir_projeto
from .estado import EstadoHQ


def criar(slug: str, *, titulo: str, fonte: Path | None = None,
          narrator: str | None = None, rights: dict | None = None,
          raiz: Path | None = None) -> Path:
    proj = dir_projeto(slug, raiz)
    if (proj / "project.yaml").exists():
        raise FileExistsError(f"projeto já existe: {proj}")
    proj.mkdir(parents=True, exist_ok=True)
    if fonte is not None:
        shutil.copy(fonte, proj / "fonte.txt")
    cfg = {
        "slug": slug,
        "area": "hq",
        "titulo": titulo,
        "fonte": "fonte.txt" if fonte is not None else None,
        # Motion comic: voz do narrador (recordatórios) e papel -> voice_id.
        # Papel sem voz no cast usa a `voz` da ficha no roteiro, senão o narrador.
        "narrator": narrator,
        "cast": {},
        "rights": rights or {"status": "PREENCHER", "autor": None, "fonte": None,
                             "licenca": None, "verificado_em": None},
    }
    (proj / "project.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return proj


def estado(proj: Path) -> EstadoHQ:
    return EstadoHQ(proj / "state.db")


def marca(*partes: object) -> str:
    """Hash curto do que define uma peça: vai no nome do arquivo e na assinatura
    do estado. Reescrever a cena não reaproveita em silêncio o PNG antigo."""
    h = hashlib.sha256()
    for p in partes:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:10]


def marca_arquivo(caminho: Path) -> str:
    return marca(caminho.read_bytes()) if caminho.exists() else "ausente"


def seed(m: str, i: int) -> int:
    """Seed derivada da marca: o mesmo candidato é reproduzível sem guardar a seed."""
    return int.from_bytes(hashlib.sha256(f"{m}:{i}".encode()).digest()[:4], "big")
