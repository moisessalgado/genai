"""Modernização de ortografia pré-1943 (irmã de `elenco.py` e `llm.py`).

Texto de domínio público anterior à Reforma Ortográfica de 1943 traz grafia que
o TTS não sabe pronunciar ("vae" em vez de "vai", "annos" em vez de "anos",
"oculos" sem o acento de "óculos"). Mesmo contrato de segurança da Camada 2: o
LLM nunca reescreve a frase, só responde a forma moderna de palavras isoladas,
e cada resposta só é aceita se sobreviver a um validador determinístico (mesma
letra inicial, semelhança alta com o original depois de tirar acento). Falhou a
validação, ou o LLM está fora do ar, a palavra é descartada -- nunca vira lixo
no lexico.

As palavras vao ao LLM em LOTE (`descobrir`/`modernizar_lote`), nao uma por
requisicao: medido com gemma4:12b, uma pergunta isolada custa ~5-40s (o modelo
"pensa" ~230 tokens antes de responder, e pensa de novo quando a resposta e
"sem correcao" -- ver mesma observacao em `narration/llm.py`). Um livro tem
milhares de palavras unicas; em lote esse custo fixo por requisicao e diluido
entre dezenas de palavras. `modernizar` (uma palavra por vez) continua
disponivel para uso pontual/depuracao, so nao e o caminho usado por `descobrir`.

A saída não é aplicada direto no texto: vira entrada de lexicon (grafia ->
forma falada), o mesmo mecanismo de `narration/rules.py` (`_expand_lexicon`),
para ficar auditável em diff.md e reaproveitável entre livros da mesma época
(ver `lexicon/pt-BR.ortografia-1943.yaml`).
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from urllib import error

from ...core.config import settings
from ...core.llm import URL_GENERATE, gerar as gerar_llm

OLLAMA_URL = URL_GENERATE
MODELO_PADRAO = settings().llm_modelo

PROMPT = """Você conhece a Reforma Ortográfica de 1943 do português do Brasil.

Contexto onde a palavra aparece: {contexto}
PALAVRA: {palavra}

Se "{palavra}" está escrita na ortografia ANTERIOR a 1943 (ex.: "annos",
"vae", "oculos", "apezar", "panno"), responda com a grafia MODERNA da MESMA
palavra, com os acentos corretos ("anos", "vai", "óculos", "apesar", "pano").
Se a palavra já está em ortografia moderna, ou é um nome próprio, responda
exatamente a mesma palavra sem mudar nada.

Responda com UMA ÚNICA PALAVRA, sem explicação, sem pontuação, sem aspas.

Resposta:"""

PROMPT_LOTE = """Você conhece a Reforma Ortográfica de 1943 do português do Brasil, que
eliminou grafias como "ph", "th" e consoantes dobradas desnecessárias, e
corrigiu acentos.

Lista de palavras extraídas de um texto brasileiro antigo:
{lista}

Para CADA palavra da lista que estiver escrita na ortografia ANTERIOR a 1943,
ou sem um acento que muda a pronúncia, dê a forma moderna da MESMA palavra
(ex.: "annos" -> "anos", "vae" -> "vai", "oculos" -> "óculos", "panno" ->
"pano", "sitio" -> "sítio"). NÃO inclua palavras que já estão em ortografia
moderna. NÃO inclua nomes próprios de pessoas ou lugares.

Responda SÓ com um objeto JSON, palavra original -> forma moderna, contendo
somente as palavras que precisam de correção. Se nenhuma precisar, responda
{{}}. Nada além do JSON -- sem explicação, sem markdown.

JSON:"""

# Abaixo disto o token quase sempre e preposicao/artigo/conjuncao -- mandar
# pro LLM so gera ruido (falso positivo em cima de "da", "se", "ou"...).
# Excecoes curtas conhecidas (ex.: "ha" -> "há") entram no lexicon a mao,
# como qualquer outra descoberta de QA.
MIN_CARACTERES = 3

# Custo fixo por requisicao (raciocinio interno do gemma4) domina o tempo de
# uma pergunta isolada -- por isso `descobrir` manda as palavras em lotes.
# Lote grande demais aumenta a chance de ter DUAS palavras "dificeis" (medido:
# contracao como "dagua", que nao cabe em resposta de uma palavra so) no mesmo
# lote, o que multiplica bisecoes em `modernizar_lote`. 15 e um meio termo.
TAMANHO_LOTE = 15

_PALAVRA = re.compile(r"[A-Za-zÀ-ÿ]+")
_BLOCO_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _sem_acento(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def extrair_candidatas(texto: str) -> dict[str, str]:
    """Palavras unicas (minusculas) -> uma ocorrencia com contexto, para servir
    de entrada ao LLM. So a primeira ocorrencia de cada palavra e guardada --
    o contexto e so uma dica, nao precisa ser exaustivo."""
    candidatas: dict[str, str] = {}
    for m in _PALAVRA.finditer(texto):
        palavra = m.group(0)
        if len(palavra) < MIN_CARACTERES:
            continue
        chave = palavra.lower()
        if chave in candidatas:
            continue
        candidatas[chave] = texto[max(0, m.start() - 40):m.end() + 40]
    return candidatas


def validar(antiga: str, moderna: str) -> tuple[bool, str]:
    """Aceita a forma moderna so se for verificavelmente a MESMA palavra."""
    moderna = moderna.strip().strip('."\'').strip()
    if not moderna:
        return False, "resposta vazia"
    if " " in moderna or "\n" in moderna:
        return False, "resposta com mais de uma palavra"
    if not re.fullmatch(r"[A-Za-zÀ-ÿ]+", moderna):
        return False, "resposta com caracteres fora do alfabeto"
    if antiga.lower() == moderna.lower():
        return False, "identica ao original (nada a corrigir)"
    base_antiga, base_moderna = _sem_acento(antiga), _sem_acento(moderna)
    if base_antiga[0] != base_moderna[0]:
        return False, "letra inicial diferente"
    ratio = difflib.SequenceMatcher(None, base_antiga, base_moderna).ratio()
    if ratio < 0.6:
        return False, f"forma sugerida muito diferente do original (similaridade {ratio:.2f})"
    return True, ""


def _chamar_ollama(prompt: str, modelo: str, url: str, timeout: int, num_predict: int) -> str:
    # gemma4:12b consome ~230 tokens de raciocinio interno antes de emitir a
    # resposta (mesma observacao de narration/llm.py): com num_predict baixo
    # ele para por 'length' e devolve string vazia.
    return gerar_llm(prompt, modelo=modelo, temperatura=0.0, num_predict=num_predict,
                     timeout=timeout, url=url)


def _perguntar(palavra: str, contexto: str, modelo: str, url: str, timeout: int,
               num_predict: int = 600) -> str:
    return _chamar_ollama(PROMPT.format(palavra=palavra, contexto=contexto),
                          modelo, url, timeout, num_predict)


def modernizar(palavra: str, contexto: str = "", *, modelo: str = MODELO_PADRAO,
               url: str = OLLAMA_URL, timeout: int = 60) -> str | None:
    """Devolve a forma moderna de `palavra`, ou `None` se ja esta moderna,
    a resposta nao passou no validador, ou o LLM esta fora do ar.

    Uma palavra por requisicao -- usar `descobrir`/`modernizar_lote` para
    varrer um texto inteiro, bem mais rapido (ver docstring do modulo)."""
    for orcamento in (600, 1200):  # gemma4 as vezes estoura o 1o orcamento
        try:
            resp = _perguntar(palavra, contexto, modelo, url, timeout, num_predict=orcamento)
        except (error.URLError, TimeoutError, KeyError, OSError):
            return None
        ok, _ = validar(palavra, resp)
        if ok:
            return resp.strip().strip('."\'')
    return None


def _perguntar_lote(palavras: list[str], modelo: str, url: str, timeout: int,
                    num_predict: int) -> str:
    lista = "\n".join(palavras)
    return _chamar_ollama(PROMPT_LOTE.format(lista=lista), modelo, url, timeout, num_predict)


def _tentar_lote(palavras: list[str], modelo: str, url: str, timeout: int) -> dict[str, str] | None:
    """Uma tentativa (com escalada de orcamento). `None` = nao saiu JSON valido
    dentro do orcamento -- distinto de `{}`, que e uma resposta valida dizendo
    "nenhuma destas palavras precisa de correcao"."""
    permitidas = {p.lower(): p for p in palavras}
    # lote maior precisa de mais espaco de resposta; 40 tokens/palavra cobre
    # raciocinio + JSON com folga.
    for orcamento in (900, 900 + 40 * len(palavras)):
        try:
            resp = _perguntar_lote(palavras, modelo, url, timeout, num_predict=orcamento)
        except (error.URLError, TimeoutError, OSError):
            return None
        m = _BLOCO_JSON.search(resp)
        if not m:
            continue
        try:
            bruto = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(bruto, dict):
            continue
        aceitas: dict[str, str] = {}
        for antiga, moderna in bruto.items():
            if not isinstance(antiga, str) or not isinstance(moderna, str):
                continue
            original = permitidas.get(antiga.lower())
            if original is None:
                continue  # o LLM respondeu por uma palavra que nao foi perguntada
            ok, _ = validar(original, moderna)
            if ok:
                aceitas[original.lower()] = moderna.strip().strip('."\'')
        return aceitas
    return None


def modernizar_lote(palavras: list[str], *, modelo: str = MODELO_PADRAO, url: str = OLLAMA_URL,
                    timeout: int = 90) -> dict[str, str]:
    """Manda um lote de palavras de uma vez; devolve {palavra: forma_moderna} so
    das que o validador aceitou. Palavra fora da lista enviada (o LLM as vezes
    "corrige" algo que nao foi perguntado) e descartada -- nunca vira surpresa.

    Se o lote inteiro nao produzir JSON valido dentro do orcamento -- medido com
    "dagua" (contracao "d'água" nao cabe na resposta de uma palavra so) e
    "Amarello" (ambiguo entre nome proprio e adjetivo): o modelo "trava"
    raciocinando e nunca emite o JSON, mesmo com orcamento bem maior -- biseca o
    lote e tenta cada metade. Isola e descarta so a palavra problematica, sem
    perder as correcoes boas do resto do lote."""
    resultado = _tentar_lote(palavras, modelo, url, timeout)
    if resultado is not None:
        return resultado
    if len(palavras) == 1:
        return {}
    meio = len(palavras) // 2
    esquerda = modernizar_lote(palavras[:meio], modelo=modelo, url=url, timeout=timeout)
    direita = modernizar_lote(palavras[meio:], modelo=modelo, url=url, timeout=timeout)
    return {**esquerda, **direita}


def descobrir(texto: str, *, modelo: str = MODELO_PADRAO, url: str = OLLAMA_URL,
             timeout: int = 90, cache: dict[str, str | None] | None = None,
             tamanho_lote: int = TAMANHO_LOTE, progresso=None) -> dict[str, str]:
    """Varre `texto` inteiro em lotes e devolve {grafia_antiga: forma_moderna}
    so das palavras que o validador aceitou. `cache` evita reconsultar palavra
    ja vista (entre chamadas, ou entre capitulos do mesmo livro)."""
    cache = cache if cache is not None else {}
    candidatas = list(extrair_candidatas(texto))
    pendentes = [p for p in candidatas if p not in cache]
    lotes = [pendentes[i:i + tamanho_lote] for i in range(0, len(pendentes), tamanho_lote)]
    for i, lote in enumerate(lotes):
        if progresso:
            progresso(i, len(lotes), lote[0])
        aceitas = modernizar_lote(lote, modelo=modelo, url=url, timeout=timeout)
        for palavra in lote:
            cache[palavra] = aceitas.get(palavra)
    return {p: cache[p] for p in candidatas if cache.get(p)}
