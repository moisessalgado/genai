"""Peças usadas por mais de uma área da CLI."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .. import project as proj_mod

console = Console()


def _proj(slug: str) -> Path:
    p = proj_mod.dir_projeto(slug)
    if not p.exists():
        raise typer.BadParameter(f"projeto não encontrado: {slug}")
    return p
