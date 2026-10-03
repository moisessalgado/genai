"""core/llm: a requisição que os quatro usos do LLM faziam cada um por conta."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.core import llm  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_gerar_monta_o_payload_do_ollama_e_limpa_a_resposta(monkeypatch):
    visto = {}

    def urlopen(req, timeout):
        visto.update(url=req.full_url, corpo=json.loads(req.data), timeout=timeout)
        return _Resp(json.dumps({"response": "  Segundo \n"}).encode())

    monkeypatch.setattr(llm.request, "urlopen", urlopen)
    assert llm.gerar("p", temperatura=0.4, num_predict=900, timeout=45) == "Segundo"
    assert visto["url"] == llm.URL_GENERATE
    assert visto["timeout"] == 45
    assert visto["corpo"] == {"model": llm.MODELO_PADRAO, "prompt": "p", "stream": False,
                              "options": {"temperature": 0.4, "num_predict": 900}}


def test_modelo_e_url_explicitos_vencem_a_configuracao(monkeypatch):
    visto = {}

    def urlopen(req, timeout):
        visto.update(url=req.full_url, modelo=json.loads(req.data)["model"])
        return _Resp(b'{"response": "x"}')

    monkeypatch.setattr(llm.request, "urlopen", urlopen)
    llm.gerar("p", modelo="outro:7b", url="http://outro:1/api/generate")
    assert visto == {"url": "http://outro:1/api/generate", "modelo": "outro:7b"}


def test_gerar_json_manda_o_schema_no_format_e_decodifica(monkeypatch):
    visto = {}
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

    def urlopen(req, timeout):
        visto.update(corpo=json.loads(req.data))
        return _Resp(json.dumps({"response": ' {"a": 3} '}).encode())

    monkeypatch.setattr(llm.request, "urlopen", urlopen)
    assert llm.gerar_json("p", schema) == {"a": 3}
    assert visto["corpo"]["format"] == schema


def test_gerar_json_resposta_cortada_vira_value_error(monkeypatch):
    monkeypatch.setattr(llm.request, "urlopen",
                        lambda req, timeout: _Resp(b'{"response": "{\\"a\\": "}'))
    import pytest
    with pytest.raises(ValueError, match="JSON"):
        llm.gerar_json("p", {"type": "object"})


def test_dialeto_openai_fala_chat_completions_com_chave(monkeypatch):
    """O LiteLLM do ai-stack: troca de backend só pela configuração."""
    visto = {}

    def urlopen(req, timeout):
        visto.update(corpo=json.loads(req.data), auth=req.get_header("Authorization"))
        return _Resp(json.dumps({"choices": [{"message": {"content": '{"a": 1}'}}]}).encode())

    monkeypatch.setattr(llm, "API", "openai")
    monkeypatch.setenv("VF_LLM_CHAVE", "sk-teste")
    monkeypatch.setattr(llm.request, "urlopen", urlopen)
    assert llm.gerar_json("p", {"type": "object"}, num_predict=700,
                          url="http://x:4000/v1/chat/completions") == {"a": 1}
    c = visto["corpo"]
    assert c["messages"] == [{"role": "user", "content": "p"}]
    assert c["max_tokens"] == 700
    assert c["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert visto["auth"] == "Bearer sk-teste"
