"""CLI do audio-factory (TDD 13). Exposta no dia a dia como `iam voice`.

Os comandos vivem em um módulo por área, e cada um se registra no `app` de
`cli/app.py` ao ser importado. A ordem dos imports abaixo é a ordem do `--help`.
"""
from __future__ import annotations

from . import audiolivro, video, publicar, fontes, voz, sistema  # noqa: F401  (registram os comandos)
from .app import app

__all__ = ["app"]

if __name__ == "__main__":
    app()
