"""Cliente do LLM, num lugar só.

Dois dialetos, escolhidos por `llm_api` na configuração central:

- `openai` (padrão): API compatível com a OpenAI (`/v1/chat/completions`), que
  é o que o LiteLLM do ai-stack (`:4000`) fala. A chave vem de `VF_LLM_CHAVE`
  no ambiente ou, sem ela, do arquivo `config/litellm.chave` (fora do git) —
  não é campo do Settings para o `doctor` não imprimi-la;
- `ollama`: API nativa do Ollama (`/api/generate`) em `llm_url`.

Antes, quatro módulos montavam a mesma requisição cada um por conta própria;
trocar o backend agora é mexer na configuração, não no código.

Erros de rede sobem como `urllib.error.URLError`/`TimeoutError`, como sempre
subiram: quem chama decide o fallback (todos os usos do audiolivro degradam em
vez de bloquear o pipeline). Como a degradação é silenciosa lá, a primeira
falha de conexão do processo vira um aviso em stderr, com o endereço e o erro.
"""
from __future__ import annotations

import json
import os
import sys
from urllib import error, request

from .config import settings

API = settings().llm_api
if API not in ("ollama", "openai"):
    raise ValueError(f"llm_api desconhecida: {API!r} — use 'ollama' ou 'openai'")
_CAMINHO = {"ollama": "/api/generate", "openai": "/v1/chat/completions"}[API]
# Nome histórico: é o endpoint completo do dialeto ativo, não só o do Ollama.
URL_GENERATE = settings().llm_url.rstrip("/") + _CAMINHO
MODELO_PADRAO = settings().llm_modelo
ARQUIVO_CHAVE = settings().config_dir / "litellm.chave"

_avisado = False


def _chave() -> str | None:
    if os.environ.get("VF_LLM_CHAVE"):
        return os.environ["VF_LLM_CHAVE"]
    try:
        return ARQUIVO_CHAVE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def _avisar(url: str, e: Exception) -> None:
    """Uma vez por processo: quem chama costuma engolir o erro e seguir sem o
    LLM, e sem este aviso o resultado só piora, sem dizer por quê."""
    global _avisado
    if _avisado:
        return
    _avisado = True
    detalhe = repr(e)
    if isinstance(e, error.HTTPError):
        try:
            corpo = e.read(300).decode("utf-8", "replace")
        except Exception:
            corpo = ""
        detalhe = f"HTTP {e.code} {e.reason} {corpo}".strip()
    dica = (" (chave do LiteLLM ausente ou recusada: VF_LLM_CHAVE ou "
            f"{ARQUIVO_CHAVE})" if isinstance(e, error.HTTPError) and e.code in (401, 403)
            else "")
    print(f"\n!!! LLM indisponível em {url}: {detalhe}{dica}.\n"
          "!!! As etapas que dependem dele vão seguir sem ele (resultado pior).\n",
          file=sys.stderr, flush=True)


def _corpo(prompt: str, modelo: str, temperatura: float, num_predict: int,
           schema: dict | None) -> dict:
    if API == "ollama":
        corpo = {"model": modelo, "prompt": prompt, "stream": False,
                 "options": {"temperature": temperatura, "num_predict": num_predict}}
        if schema is not None:
            corpo["format"] = schema
        return corpo
    corpo = {"model": modelo, "messages": [{"role": "user", "content": prompt}],
             "stream": False, "temperature": temperatura, "max_tokens": num_predict}
    if schema is not None:
        corpo["response_format"] = {"type": "json_schema",
                                    "json_schema": {"name": "saida", "schema": schema}}
    return corpo


def _texto(resposta: dict) -> tuple[str, str | None]:
    """(texto, motivo de parada) — 'length' quer dizer orçamento estourado."""
    if API == "ollama":
        return resposta["response"], resposta.get("done_reason")
    escolha = resposta["choices"][0]
    return escolha["message"]["content"] or "", escolha.get("finish_reason")


def _chamar(prompt: str, *, modelo: str | None, temperatura: float, num_predict: int,
            timeout: float, url: str | None, schema: dict | None) -> tuple[str, str | None]:
    cabecalhos = {"Content-Type": "application/json"}
    if API == "openai" and (chave := _chave()):
        cabecalhos["Authorization"] = "Bearer " + chave
    corpo = json.dumps(_corpo(prompt, modelo or MODELO_PADRAO, temperatura,
                              num_predict, schema)).encode()
    req = request.Request(url or URL_GENERATE, data=corpo, headers=cabecalhos)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            texto, motivo = _texto(json.loads(r.read()))
    except (error.URLError, TimeoutError, ConnectionError) as e:
        _avisar(req.full_url, e)
        raise
    return texto.strip(), motivo


def gerar(prompt: str, *, modelo: str | None = None, temperatura: float = 0.0,
          num_predict: int = 600, timeout: float = 60, url: str | None = None) -> str:
    """Uma resposta completa (sem streaming), já sem espaços nas pontas.

    `url`, se dado, é o endpoint completo do dialeto ativo.

    `num_predict` baixo é armadilha medida: o gemma4:12b gasta ~230 tokens
    raciocinando antes de responder, e com orçamento curto para por 'length'
    devolvendo string vazia. 600 é o piso seguro para respostas curtas.
    """
    return _chamar(prompt, modelo=modelo, temperatura=temperatura,
                   num_predict=num_predict, timeout=timeout, url=url, schema=None)[0]


def gerar_json(prompt: str, schema: dict, *, modelo: str | None = None,
               temperatura: float = 0.0, num_predict: int = 4000, timeout: float = 300,
               url: str | None = None):
    """Saída estruturada: o servidor restringe a geração ao JSON Schema dado
    (`format` no Ollama, `response_format` na API OpenAI) e aqui ela volta já
    decodificada. Validar o CONTEÚDO (ids coerentes etc.) é de quem chama —
    o schema garante a forma, não o sentido.

    `ValueError` se a resposta não for JSON: cortada por `num_predict`
    (parada 'length') ou, raramente, malformada mesmo com a gramática — visto
    com o gemma4:12b, string sem fechar e parada 'stop'. Quem chama tenta de
    novo. Com `format`, o gemma4 não emite raciocínio (medido: `thinking`
    vazio), então o piso de 600 do `gerar` não se aplica aqui."""
    texto, motivo = _chamar(prompt, modelo=modelo, temperatura=temperatura,
                    num_predict=num_predict, timeout=timeout, url=url, schema=schema)
    try:
        return json.loads(texto)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM não devolveu JSON ({e}; parada: {motivo}); "
                         f"início: {texto[:200]!r}") from e
