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
