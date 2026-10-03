"""core/claude_cli: `claude -p` por subprocesso, com um binário falso."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.core import claude_cli  # noqa: E402
from genai.core import config  # noqa: E402


def _binario(tmp_path, monkeypatch, saida: dict | str, rc: int = 0) -> Path:
    """Um `claude` que grava argv, stdin e ambiente e imprime `saida`."""
    exe = tmp_path / "claude"
    corpo = saida if isinstance(saida, str) else json.dumps(saida)
    exe.write_text(f"""#!{sys.executable}
import json, os, sys
json.dump({{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd(),
           "env": sorted(k for k in os.environ if "ANTHROPIC" in k or "OAUTH" in k)}},
          open({str(tmp_path / "visto.json")!r}, "w"))
print({corpo!r})
sys.exit({rc})
""")
    exe.chmod(0o755)
    monkeypatch.setattr(claude_cli, "settings", lambda: config.carregar(
        env={"VF_CLAUDE_BIN": str(exe)}, arquivo=tmp_path / "x.toml"))
    return tmp_path / "visto.json"


def test_resposta_estruturada_sem_credencial_no_ambiente(tmp_path, monkeypatch):
    visto = _binario(tmp_path, monkeypatch, {"type": "result", "subtype": "success",
                                             "is_error": False, "structured_output": {"a": 7}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-nao-pode")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "nao-pode")
    schema = {"type": "object"}
    assert claude_cli.gerar_json("o prompt", schema, num_predict=3000) == {"a": 7}
    v = json.loads(visto.read_text())
    assert v["stdin"] == "o prompt"
    assert v["env"] == []
    a = v["argv"]
    assert a[a.index("--json-schema") + 1] == json.dumps(schema)
    assert a[a.index("--model") + 1] == "sonnet"
    assert a[a.index("--tools") + 1] == ""
    assert "-p" in a and "--no-session-persistence" in a
    assert not Path(v["cwd"]).exists()  # pasta temporária, já apagada


def test_limite_de_uso_vira_claude_indisponivel(tmp_path, monkeypatch):
    _binario(tmp_path, monkeypatch, {"type": "result", "subtype": "success", "is_error": True,
                                     "api_error_status": 429, "result": "usage limit"}, rc=1)
    with pytest.raises(claude_cli.ClaudeIndisponivel, match="429"):
        claude_cli.gerar_json("p", {})


def test_saida_que_nao_e_json_vira_claude_indisponivel(tmp_path, monkeypatch):
    _binario(tmp_path, monkeypatch, "Not logged in", rc=1)
    with pytest.raises(claude_cli.ClaudeIndisponivel, match="Not logged in"):
        claude_cli.gerar_json("p", {})


def test_sem_saida_estruturada_e_value_error(tmp_path, monkeypatch):
    _binario(tmp_path, monkeypatch, {"type": "result", "subtype": "success",
                                     "is_error": False, "result": "texto solto"})
    with pytest.raises(ValueError, match="sem saída estruturada"):
        claude_cli.gerar_json("p", {})


def test_binario_ausente(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_cli, "settings", lambda: config.carregar(
        env={"VF_CLAUDE_BIN": str(tmp_path / "nao-ha")}, arquivo=tmp_path / "x.toml"))
    with pytest.raises(claude_cli.ClaudeIndisponivel, match="não encontrado"):
        claude_cli.gerar_json("p", {})


def test_extensao_do_vscode_mais_nova(tmp_path, monkeypatch):
    for v in ("2.1.9", "2.1.288", "2.1.30"):
        b = tmp_path / f".vscode/extensions/anthropic.claude-code-{v}-linux-x64/resources/native-binary/claude"
        b.parent.mkdir(parents=True)
        b.write_text("")
    monkeypatch.setattr(claude_cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_cli.shutil, "which", lambda x: None)
    assert "2.1.288" in claude_cli.binario()


def test_fora_do_path_prefere_a_instalacao_nativa(tmp_path, monkeypatch):
    nativo = tmp_path / ".local/bin/claude"
    nativo.parent.mkdir(parents=True)
    nativo.write_text("")
    nativo.chmod(0o755)
    b = tmp_path / ".vscode/extensions/anthropic.claude-code-2.1.288-linux-x64/resources/native-binary/claude"
    b.parent.mkdir(parents=True)
    b.write_text("")
    monkeypatch.setattr(claude_cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_cli.shutil, "which", lambda x: None)
    assert claude_cli.binario() == str(nativo)
