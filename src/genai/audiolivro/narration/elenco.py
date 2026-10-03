"""Atribuição de personagem à fala citada (multivoz por elenco nominal).

Mesmo contrato de segurança da Camada 2 (`narration/llm.py`): o LLM nunca
reescreve nada, só CLASSIFICA — aqui, em conjunto fechado (um nome do elenco
conhecido, ou "indeterminado"). A resposta é validada por igualdade exata
contra a lista permitida; falhou a validação, ou o LLM está fora do ar, o
papel genérico `citacao` (`text/roles.py`) é mantido — nunca bloqueia.

Isto é deliberadamente mais conservador que rastrear turnos de diálogo: uma
fala sem atribuição inequívoca por perto (ex.: alternância de dois
personagens sem repetir o nome a cada linha) fica em `citacao`. Medido no
capítulo 1 de *As Reinações de Narizinho*: de 182 falas por travessão, ~90 têm
um nome inequívoco por perto -- só essas ganham voz de personagem; o resto
soa na voz genérica de diálogo. É uma melhoria real sobre uma citação única,
não uma tentativa de resolver o problema inteiro.
"""
from __future__ import annotations

import json
import re
import unicodedata
from urllib import error, request

from ...core.config import settings

OLLAMA_URL = settings().llm_url.rstrip("/") + "/api/generate"
MODELO_PADRAO = settings().llm_modelo

INDETERMINADO = "indeterminado"

PROMPT = """Você identifica QUEM fala uma frase de diálogo num livro em português.

Personagens conhecidos nesta obra: {personagens}

Contexto antes da fala: {antes}
FALA: {fala}
Contexto depois da fala: {depois}

Responda com o NOME EXATO de um dos personagens da lista acima (só o nome,
minúsculo, sem acento, como aparece na lista), ou com a palavra
"indeterminado" se não houver como saber com segurança quem fala.
Não explique, não escreva mais nada além do nome ou de "indeterminado".

Resposta:"""

# Janela de contexto: o suficiente para pegar uma marcação de fala tipo
# "disse Emília" logo antes/depois, sem arrastar o parágrafo inteiro para o
# prompt (o LLM as vezes se distrai com contexto longo demais).
JANELA_CONTEXTO = 200


def _sem_acento(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _perguntar(fala: str, antes: str, depois: str, personagens: list[str],
              modelo: str, url: str, timeout: int) -> str:
    corpo = json.dumps({
        "model": modelo,
        "prompt": PROMPT.format(personagens=", ".join(personagens),
                                fala=fala, antes=antes[-JANELA_CONTEXTO:],
                                depois=depois[:JANELA_CONTEXTO]),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 700},
    }).encode()
    req = request.Request(url, data=corpo, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()


def atribuir(fala: str, antes: str, depois: str, personagens: list[str], *,
            modelo: str = MODELO_PADRAO, url: str = OLLAMA_URL,
            timeout: int = 60) -> str | None:
    """Devolve a CHAVE do personagem (uma de `personagens`), ou `None` se
    indeterminado, resposta inválida, ou o LLM estiver indisponível."""
    if not personagens:
        return None
    try:
        resp = _perguntar(fala, antes, depois, personagens, modelo, url, timeout)
    except (error.URLError, TimeoutError, KeyError, OSError):
        return None
    candidato = _sem_acento(resp.strip().strip('."\''))
    if candidato == INDETERMINADO:
        return None
    permitidos = {_sem_acento(p): p for p in personagens}
    return permitidos.get(candidato)


def aplicar_elenco(partes: list[tuple[str, str]], personagens: dict[str, str], *,
                   modelo: str = MODELO_PADRAO, url: str = OLLAMA_URL,
                   timeout: int = 60, cache: dict[str, str | None] | None = None
                   ) -> list[tuple[str, str]]:
    """Recebe o resultado de `dividir_por_papel` e tenta promover falas
    `citacao` para `personagem_<chave>`, usando o narrador vizinho (na mesma
    lista) como contexto.

    `personagens` é o elenco do livro (chave -> voice_id) SEM a entrada
    `_generico` -- essa é o fallback de voz do `citacao`, não um personagem
    para o LLM escolher.
    """
    nomes = [k for k in personagens if not k.startswith("_")]
    if not nomes:
        return partes
    cache = cache if cache is not None else {}

    saida: list[tuple[str, str]] = []
    for i, (papel, texto) in enumerate(partes):
        if papel != "citacao":
            saida.append((papel, texto))
            continue
        antes = partes[i - 1][1] if i > 0 else ""
        depois = partes[i + 1][1] if i + 1 < len(partes) else ""
        chave_cache = f"{texto}|{antes[-JANELA_CONTEXTO:]}|{depois[:JANELA_CONTEXTO]}"
        if chave_cache in cache:
            personagem = cache[chave_cache]
        else:
            personagem = atribuir(texto, antes, depois, nomes, modelo=modelo,
                                  url=url, timeout=timeout)
            cache[chave_cache] = personagem
        saida.append((f"personagem_{personagem}" if personagem else papel, texto))
    return saida
