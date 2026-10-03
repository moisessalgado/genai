"""O Typer raiz. Cada módulo de `cli/` registra seus comandos nele."""
from __future__ import annotations

import typer

app = typer.Typer(help="genai — audiolivros, vídeos para o YouTube e HQ com IA, 100% local",
                  no_args_is_help=True)
