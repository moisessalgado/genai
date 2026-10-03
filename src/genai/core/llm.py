"""Cliente do LLM local, num lugar só.

Hoje fala com a API nativa do Ollama (`/api/generate`) em `llm_url`, com o
modelo `llm_modelo` da configuração central. Antes, quatro módulos montavam a
mesma requisição cada um por conta própria; trocar o backend (LiteLLM, por
exemplo) agora é mexer só aqui.

Erros de rede sobem como `urllib.error.URLError`/`TimeoutError`, como sempre
subiram: quem chama decide o fallback (todos os usos aqui degradam em vez de
bloquear o pipeline).
"""
from __future__ import annotations

import json
from urllib import request

from .config import settings

URL_GENERATE = settings().llm_url.rstrip("/") + "/api/generate"
MODELO_PADRAO = settings().llm_modelo


def gerar(prompt: str, *, modelo: str | None = None, temperatura: float = 0.0,
          num_predict: int = 600, timeout: float = 60, url: str | None = None) -> str:
    """Uma resposta completa (sem streaming), já sem espaços nas pontas.

    `num_predict` baixo é armadilha medida: o gemma4:12b gasta ~230 tokens
    raciocinando antes de responder, e com orçamento curto para por 'length'
    devolvendo string vazia. 600 é o piso seguro para respostas curtas.
    """
    corpo = json.dumps({
        "model": modelo or MODELO_PADRAO,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperatura, "num_predict": num_predict},
    }).encode()
    req = request.Request(url or URL_GENERATE, data=corpo,
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()
