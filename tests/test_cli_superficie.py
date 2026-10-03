"""A superfície da CLI (comandos, opções, padrões, ajuda) não muda por acidente.

O `cli/main.py` foi dividido em módulos por área; este teste compara a árvore
de comandos com a foto tirada ANTES da divisão. Mudança de propósito numa opção
exige regravar a foto:

    VF_REGRAVAR_CLI=1 uv run pytest tests/test_cli_superficie.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.cli.main import app  # noqa: E402

FOTO = Path(__file__).parent / "fixtures" / "cli_superficie.json"


def _valor(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_valor(x) for x in v]
    return str(v)


def _parametro(p) -> dict:
    d = {"nome": p.name, "tipo": p.param_type_name, "click_tipo": getattr(p.type, "name", type(p.type).__name__),
         "opts": list(p.opts), "secundarias": list(p.secondary_opts),
         "obrigatorio": p.required, "multiplo": p.multiple, "padrao": _valor(p.default)}
    if p.param_type_name == "option":
        d |= {"flag": p.is_flag, "ajuda": p.help}
    return d


def _arvore(cmd) -> dict:
    d = {"ajuda": cmd.help, "parametros": [_parametro(p) for p in cmd.params]}
    # o Typer 0.27 embute o proprio click: isinstance contra `click` falha
    if hasattr(cmd, "commands"):
        d["subcomandos"] = {n: _arvore(c) for n, c in sorted(cmd.commands.items())}
    return d


def superficie() -> dict:
    return _arvore(typer.main.get_command(app))


def _caminhos(arv: dict, prefixo: tuple[str, ...] = ()):
    yield prefixo
    for nome, sub in arv.get("subcomandos", {}).items():
        yield from _caminhos(sub, prefixo + (nome,))


def test_superficie_igual_a_foto():
    atual = superficie()
    if os.environ.get("VF_REGRAVAR_CLI"):
        FOTO.write_text(json.dumps(atual, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    assert atual == json.loads(FOTO.read_text(encoding="utf-8"))


@pytest.mark.parametrize("caminho", list(_caminhos(json.loads(FOTO.read_text(encoding="utf-8")))),
                         ids=lambda c: " ".join(c) or "<raiz>")
def test_help_de_cada_comando(caminho):
    r = CliRunner().invoke(app, [*caminho, "--help"])
    assert r.exit_code == 0, r.output
