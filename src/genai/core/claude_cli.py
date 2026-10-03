"""Claude pelo Claude Code da assinatura do operador (`claude -p`), por subprocesso.

Exceção explícita à regra "todo LLM passa pelo LiteLLM" (decisão do operador,
2026-10-03): ele não tem chave de API e não quer pagar crédito de API; a
assinatura só vale dentro do próprio Claude Code, que aqui roda em modo não
interativo, como o usuário dele. O token da assinatura nunca é lido nem
repassado: quem autentica é o binário, com o login que já tem. Por isso o
ambiente do filho vai SEM `ANTHROPIC_API_KEY` (o binário cobraria na API) e sem
`CLAUDE_CODE_OAUTH_TOKEN` (o do openclaw não é deste uso).

A franquia é a das janelas de ~5 h que o operador divide com o Claude Code
interativo e o openclaw: serve para roteiro pontual, não para lote.

`ClaudeIndisponivel` cobre o que deve cair no LLM local: binário ausente,
timeout, limite de uso, erro da API. JSON que não bate com o schema sobe como
`ValueError`, como no `core/llm.gerar_json`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import settings

# O binário que a extensão do VSCode instala, quando não há `claude` no PATH.
_EXTENSOES = "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
_SEM = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
SISTEMA = ("You are a careful comic-script writer. Answer only through the structured "
           "output requested; never call tools.")


class ClaudeIndisponivel(RuntimeError):
    """O `claude -p` não respondeu: quem chama usa o LLM local."""


def _versao(p: Path) -> tuple[int, ...]:
    nome = p.parents[2].name.rsplit("-", 2)[0]  # anthropic.claude-code-2.1.288
    return tuple(int(x) for x in nome.rsplit("-", 1)[-1].split(".") if x.isdigit())


def binario() -> str | None:
    """`claude_bin` da configuração (nome no PATH ou caminho); sem ele no PATH,
    o binário da extensão do VSCode mais nova."""
    alvo = settings().claude_bin
    if achado := shutil.which(os.path.expanduser(alvo)):
        return achado
    if alvo != "claude":
        return None
    candidatos = sorted(Path.home().glob(_EXTENSOES.removeprefix("~/")), key=_versao)
    return str(candidatos[-1]) if candidatos else None


def gerar_json(prompt: str, schema: dict, *, modelo: str | None = None,
               timeout: float = 600, **_ignorado) -> dict:
    """Uma resposta estruturada (`--json-schema`). `**_ignorado` engole o
    `num_predict` que os chamadores do `core/llm` passam."""
    exe = binario()
    if exe is None:
        raise ClaudeIndisponivel("binário `claude` não encontrado (claude_bin)")
    cmd = [exe, "-p", "--output-format", "json", "--model", modelo or settings().claude_modelo,
           "--json-schema", json.dumps(schema), "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--system-prompt", SISTEMA]
    env = {k: v for k, v in os.environ.items() if k not in _SEM}
    # Pasta vazia: sem CLAUDE.md nem settings de projeto entrando no contexto.
    with tempfile.TemporaryDirectory(prefix="genai-claude-") as cwd:
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                               timeout=timeout, cwd=cwd, env=env)
        except subprocess.TimeoutExpired as e:
            raise ClaudeIndisponivel(f"`claude -p` passou de {timeout:.0f} s") from e
    try:
        saida = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise ClaudeIndisponivel(f"`claude -p` saiu com {r.returncode}: "
                                 f"{(r.stderr or r.stdout).strip()[:300]}") from None
    if r.returncode != 0 or saida.get("is_error") or saida.get("subtype") != "success":
        raise ClaudeIndisponivel(
            f"`claude -p` falhou ({saida.get('subtype')}, status "
            f"{saida.get('api_error_status')}): {str(saida.get('result'))[:300]}")
    estruturada = saida.get("structured_output")
    if not isinstance(estruturada, dict):
        raise ValueError(f"`claude -p` sem saída estruturada; resultado: "
                         f"{str(saida.get('result'))[:200]!r}")
    return estruturada
