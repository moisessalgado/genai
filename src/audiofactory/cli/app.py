"""O Typer raiz. Cada módulo de `cli/` registra seus comandos nele."""
from __future__ import annotations

import typer

app = typer.Typer(help="Audio Factory — audiolivros narrados por IA, 100% local",
                  no_args_is_help=True)
