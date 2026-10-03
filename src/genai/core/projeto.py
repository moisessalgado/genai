"""Projeto, comum a todas as áreas: pasta em projects/<slug> e project.yaml.

O que cada área põe no projeto (script.json do audiolivro, roteiro da HQ...)
mora na área; aqui fica só o que todas compartilham."""
from __future__ import annotations

from pathlib import Path

import yaml

from .config import settings

# Valores convencionais de `rights.status`. Servem para consulta e para o registro
# no project.yaml -- NAO sao uma autorizacao: o export nao e bloqueado por eles.
# A decisao editorial sobre o que publicar e do operador do canal, nao da ferramenta.
RIGHTS_CONHECIDOS = {"dominio-publico", "proprio", "licenciado", "teste-local"}


def dir_projeto(slug: str, raiz: Path | None = None) -> Path:
    return raiz / "projects" / slug if raiz else settings().projects_dir / slug


def carregar_config(proj: Path) -> dict:
    return yaml.safe_load((proj / "project.yaml").read_text(encoding="utf-8"))
