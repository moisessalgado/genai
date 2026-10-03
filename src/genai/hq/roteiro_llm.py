"""Roteiro da HQ rascunhado pelo LLM, a partir de um texto-fonte.

Dois estágios, para cada resposta caber folgada no orçamento do modelo local
(o gemma4:12b ainda gasta ~230 tokens raciocinando antes de responder):

1. **plano**: título, elenco com ficha visual e um resumo por página;
2. **página**: os quadros de uma página, um pedido por página, com o elenco e
   o plano já fixados — o que mantém nomes e fichas iguais do começo ao fim.

O LLM devolve JSON restrito pelo schema (`core/llm.gerar_json`); o sentido é
conferido pelo `Roteiro` do pydantic. Se ele recusar, a mesma página é pedida
de novo com o erro no prompt (até `TENTATIVAS`).

A diagramação (quais quadros dividem uma tira) NÃO é pedida ao LLM: sai de
`montar_tiras`, pela proporção de cada quadro. É regra geométrica, e o modelo
erraria à toa.

Fidelidade: falas e recordatórios saem do texto-fonte, condensados mas sem
eufemismo (decisão do operador para textos de domínio público).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from ..core import llm
from .roteiro import (MAX_CHARS_TEXTO, MAX_PERSONAGENS_QUADRO, MAX_QUADROS_TIRA, TAMANHOS,
                      Pagina, Personagem, Proporcao, Quadro, Roteiro, Texto)

TENTATIVAS = 3
# Texto-fonte enviado por pedido. Fábulas e contos cabem inteiros; um capítulo
# longo deve ser dividido antes (uma HQ por trecho).
MAX_CHARS_FONTE = 12000

ESTILO_PADRAO = (
    "ukiyo-e Japanese woodblock print, Edo period, in the style of Hokusai and "
    "Hiroshige, bold black ink outlines, flat areas of color, limited palette of "
    "indigo blue, vermilion, ochre and pale green, soft bokashi gradient sky, "
    "visible washi paper texture and woodgrain. No text, no letters, no "
    "calligraphy, no signature, no seals, no cartouche."
)

ANCORA_PADRAO = "a traditional Japanese woodblock print by Hokusai, flat colors, washi paper"


# ------------------------------------------------------------ schemas do LLM
class _PersonagemLLM(BaseModel):
    id: str = Field(description="lowercase ascii slug, e.g. 'jabuti'")
    nome: str
    ficha: str = Field(description="fixed visual description in ENGLISH: age, build, "
                                   "face, hair, clothing, colors, accessories")


class _PaginaPlano(BaseModel):
    resumo: str
    quadros: int = Field(ge=3, le=7)


class _Plano(BaseModel):
    titulo: str
    epoca: str = Field(description="time and place of the story, in English")
    personagens: list[_PersonagemLLM] = Field(min_length=1, max_length=6)
    paginas: list[_PaginaPlano] = Field(min_length=1)


class _TextoLLM(BaseModel):
    tipo: Literal["recordatorio", "fala", "pensamento"]
    quem: str | None = None
    texto: str = Field(min_length=1)


class _QuadroLLM(BaseModel):
    proporcao: Proporcao
    personagens: list[str]
    cena: str
    textos: list[_TextoLLM] = Field(max_length=2)


class _PaginaLLM(BaseModel):
    quadros: list[_QuadroLLM] = Field(min_length=1, max_length=7)


def schema(modelo: type[BaseModel]) -> dict:
    """JSON Schema sem `$ref`: a gramática do Ollama lida melhor com tudo inline."""
    s = modelo.model_json_schema()
    defs = s.pop("$defs", {})

    def inline(no):
        if isinstance(no, dict):
            if "$ref" in no:
                return inline(defs[no["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in no.items() if k != "title"}
        if isinstance(no, list):
            return [inline(v) for v in no]
        return no
    return inline(s)


def schema_pagina(ids: list[str]) -> dict:
    """Schema da página com `personagens` e `quem` restritos aos ids do elenco:
    a gramática do servidor não deixa o modelo inventar um id (visto: 'channo'
    no lugar de 'channa')."""
    s = schema(_PaginaLLM)
    q = s["properties"]["quadros"]["items"]["properties"]
    q["personagens"]["items"] = {"type": "string", "enum": ids}
    q["textos"]["items"]["properties"]["quem"] = {"anyOf": [
        {"type": "string", "enum": ids}, {"type": "null"}]}
    return s


# ------------------------------------------------------------------ prompts
PROMPT_PLANO = """You are adapting a text into a comic book (HQ) in {idioma}.

SOURCE TEXT:
\"\"\"
{fonte}
\"\"\"

Plan the adaptation in exactly {paginas} page(s).
- "titulo": the comic's title, in {idioma}.
- "epoca": the time and place where the story happens, in ENGLISH, short
  (e.g. "ancient India, 5th century BC", "rural Brazil, early 1900s").
- "personagens": the recurring characters who appear on panels (at most 6).
  "id" is a short lowercase ascii slug; "nome" as in the text; "ficha" is a
  FIXED visual description in ENGLISH, 40 to 70 words: age, build, face, hair,
  clothing with colors and fabrics, accessories, footwear — all faithful to
  the time, place and culture of the story (no modern items). Concrete enough
  for an illustrator to draw the same figure every time.
  Animals that talk are characters too: describe species, size, colors.
- "paginas": one entry per page with "resumo" (what happens on that page, in
  {idioma}, following the text in order) and "quadros" (3 to 7 panels).
Cover the whole story, in order, across the pages.
"""

PROMPT_PAGINA = """You are writing page {n} of {total} of a comic book in {idioma},
adapted from the source text below.

SOURCE TEXT:
\"\"\"
{fonte}
\"\"\"

CHARACTERS (use only these ids in "personagens" and "quem"):
{elenco}

WHOLE PLAN:
{plano}

Write the panels of page {n}: about {quadros}, up to 7 — enough to show EVERY
event of this page's summary, in order ({resumo}).
For each panel:
- "proporcao": "panoramico" (very wide establishing shot), "largo" (wide),
  "quadrado" (square) or "alto" (tall, close-up or single standing figure).
  Vary them like a real comic page: establishing shots wide, emotional
  moments and reactions as "alto" or "quadrado" close-ups.
- "personagens": ids of the characters drawn in the panel, ordered LEFT to RIGHT,
  at most {max_p}. Other people or animals go only in "cena".
- "cena": ONE paragraph in ENGLISH describing only what is visible: setting,
  action, poses, camera framing. Refer to characters by "nome" exactly as
  listed; do not describe their looks (their fixed description is added
  automatically). Leave empty space at the top of
  the panel for the speech balloons, and say so ("empty sky at the top").
- "textos": 0 to 2 items, in reading order. "recordatorio" = narrator caption
  (no "quem"); "fala" = speech, "pensamento" = thought, both with "quem" = id
  of a character IN THIS PANEL. Text in {idioma}, at most {max_c} characters
  each, taken from the source text and copied without typos: condense, but
  stay faithful to it — keep
  its words, its harshness and its archaic terms; do not soften or sanitize.
  Never repeat in a caption the words someone says in a balloon.
{erro}"""


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "personagem"


# Dígito conta como letra: "branc0s" tem de ser UMA palavra para ser consertada.
_PALAVRA = re.compile(r"\w+(?:[-']\w+)*")


def corrigir_pela_fonte(texto: str, fonte: str, corte: float = 0.8) -> str:
    """Conserta erro de digitação do LLM usando o próprio texto-fonte.

    Sob a gramática do `format`, o gemma4 troca letras de vez em quando
    (visto: "adividos" por "adivinhos", "branc0s" por "brancos"). Palavra que
    não existe na fonte mas é quase igual a uma que existe vira a da fonte;
    palavra nova de verdade (o LLM condensou com termo próprio) fica."""
    vocab = {w.lower() for w in _PALAVRA.findall(fonte)}

    def troca(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() in vocab or len(w) < 4:
            return w
        perto = difflib.get_close_matches(w.lower(), vocab, n=1, cutoff=corte)
        if not perto:
            return w
        novo = perto[0]
        return novo.capitalize() if w[0].isupper() else novo
    return _PALAVRA.sub(troca, texto)


# ------------------------------------------------------------- diagramação
def montar_tiras(quadros: list[Quadro], alvo: float = 2.2) -> list[list[int]]:
    """Divide os quadros, em ordem, em tiras de até 3, com a soma das proporções
    (L/A) de cada tira o mais perto possível de `alvo`.

    ~2,2 é a proporção de uma tira de A4 quando a página tem 3 tiras (o spike:
    panorâmico 2,29 | 2 quadrados 2,0 | 3 altos 2,25). Partição ótima por
    programação dinâmica — guloso juntava quadrado+quadrado+alto numa tira só.
    Panorâmico vai sempre sozinho."""
    r = [TAMANHOS[q.proporcao][0] / TAMANHOS[q.proporcao][1] for q in quadros]
    n = len(quadros)
    melhor: list[tuple[float, list[int]]] = [(0.0, [])] + [(float("inf"), [])] * n
    for fim in range(1, n + 1):
        for k in range(1, min(MAX_QUADROS_TIRA, fim) + 1):
            ini = fim - k
            grupo = quadros[ini:fim]
            if k > 1 and any(q.proporcao == "panoramico" for q in grupo):
                continue
            custo = melhor[ini][0] + (sum(r[ini:fim]) - alvo) ** 2
            if custo < melhor[fim][0]:
                melhor[fim] = (custo, melhor[ini][1] + [fim])
    cortes = [0] + melhor[n][1]
    return [[q.id for q in quadros[a:b]] for a, b in zip(cortes, cortes[1:])]


# ------------------------------------------------------------------ geração
GerarJson = Callable[..., dict]


def _plano(fonte: str, paginas: int, idioma: str, gerar_json: GerarJson) -> _Plano:
    ultimo = None
    for _ in range(TENTATIVAS):
        try:
            bruto = gerar_json(PROMPT_PLANO.format(idioma=idioma, fonte=fonte,
                                                   paginas=paginas),
                               schema(_Plano), num_predict=3000)
            plano = _Plano.model_validate(bruto)
        except (ValidationError, ValueError) as e:
            ultimo = e
            continue
        if len(plano.paginas) == paginas:
            return plano
        ultimo = ValueError(f"plano com {len(plano.paginas)} páginas, pedi {paginas}")
    raise ValueError(f"o LLM não devolveu um plano válido: {ultimo}")


def _uma_pagina(n: int, plano: _Plano, pers: dict[str, Personagem], ids: dict[str, str],
                fonte: str, idioma: str, primeiro_id: int, base: dict,
                gerar_json: GerarJson) -> tuple[list[Quadro], Pagina]:
    pp = plano.paginas[n - 1]
    elenco = "\n".join(f"- {pid}: {p.nome}" for pid, p in pers.items())
    resumo_plano = "\n".join(f"page {i}: {p.resumo}" for i, p in enumerate(plano.paginas, 1))
    erro = ""
    for _ in range(TENTATIVAS):
        prompt = PROMPT_PAGINA.format(
            n=n, total=len(plano.paginas), idioma=idioma, fonte=fonte, elenco=elenco,
            plano=resumo_plano, quadros=pp.quadros, resumo=pp.resumo,
            max_p=MAX_PERSONAGENS_QUADRO, max_c=MAX_CHARS_TEXTO, erro=erro)
        try:
            bruto = gerar_json(prompt, schema_pagina(list(pers)), num_predict=4000)
            pag = _PaginaLLM.model_validate(bruto)
            quadros = []
            for i, q in enumerate(pag.quadros):
                # o LLM às vezes usa o nome em vez do id
                trad = lambda x: ids.get(x, ids.get(_slug(x or ""), x))
                quadros.append(Quadro(
                    id=primeiro_id + i, proporcao=q.proporcao,
                    personagens=[trad(p) for p in q.personagens], cena=q.cena,
                    textos=[Texto(tipo=t.tipo, quem=trad(t.quem) if t.quem else None,
                                  texto=corrigir_pela_fonte(t.texto, fonte))
                            for t in q.textos]))
            pagina = Pagina(tiras=montar_tiras(quadros))
            # valida a página sozinha contra o elenco: o erro volta para ESTA página
            Roteiro.model_validate({**base, "quadros": [q.model_dump() for q in quadros],
                                    "paginas": [pagina.model_dump()]})
            return quadros, pagina
        except (ValidationError, ValueError) as e:
            erro = ("\nYOUR PREVIOUS ANSWER WAS REJECTED. Fix this and answer again:\n"
                    + str(e)[:1500] + "\n")
    raise ValueError(f"página {n}: o LLM não chegou a um quadro válido. Último erro:{erro}")


def gerar(fonte: str, *, slug: str, paginas: int = 1, estilo: str = ESTILO_PADRAO,
          ancora_estilo: str | None = ANCORA_PADRAO,
          idioma: str = "pt-BR", titulo: str | None = None,
          gerar_json: GerarJson | None = None, progresso=None) -> Roteiro:
    """Rascunho completo do roteiro. Sobe `ValueError` se o LLM não convergir."""
    gerar_json = gerar_json or llm.gerar_json
    fonte = fonte.strip()
    if len(fonte) > MAX_CHARS_FONTE:
        raise ValueError(f"texto-fonte com {len(fonte)} caracteres (máx. {MAX_CHARS_FONTE}):"
                         " divida em trechos, uma HQ por trecho")
    if progresso:
        progresso("plano: elenco e páginas")
    plano = _plano(fonte, paginas, idioma, gerar_json)
    pers: dict[str, Personagem] = {}
    ids: dict[str, str] = {}
    for p in plano.personagens:
        pid = _slug(p.id)
        while pid in pers:
            pid += "-2"
        pers[pid] = Personagem(nome=p.nome, ficha=p.ficha)
        ids.update({p.id: pid, p.nome: pid, _slug(p.nome): pid, pid: pid})
    base = {"slug": slug, "titulo": titulo or plano.titulo, "idioma": idioma,
            "estilo": estilo, "ancora_estilo": ancora_estilo, "epoca": plano.epoca, "personagens": {k: v.model_dump() for k, v in pers.items()}}
    quadros: list[Quadro] = []
    pags: list[Pagina] = []
    for n in range(1, paginas + 1):
        if progresso:
            progresso(f"página {n}/{paginas}")
        qs, pg = _uma_pagina(n, plano, pers, ids, fonte, idioma, len(quadros) + 1,
                             base, gerar_json)
        quadros += qs
        pags.append(pg)
    # personagens que o plano listou mas nenhum quadro desenha não precisam de folha
    usados = {p for q in quadros for p in q.personagens}
    base["personagens"] = {k: v for k, v in base["personagens"].items() if k in usados}
    return Roteiro.model_validate({**base, "quadros": [q.model_dump() for q in quadros],
                                   "paginas": [p.model_dump() for p in pags]})
