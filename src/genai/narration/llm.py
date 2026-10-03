"""Camada 2: o LLM resolve APENAS spans ambiguos (TDD 6.3).

Contrato de seguranca, nao negociavel:
  - o LLM nunca recebe o paragrafo para reescrever;
  - recebe {span, contexto antes, contexto depois} e devolve so a expansao do span;
  - a resposta passa por um VALIDADOR automatico antes de ser aceita;
  - falhou a validacao -> descarta a sugestao e mantem a regra deterministica.

O objetivo nao e fluidez, e fidelidade. Na duvida, o texto original vence.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from urllib import error, request

from num2words import num2words

from ..config import settings

OLLAMA_URL = settings().llm_url.rstrip("/") + "/api/generate"
MODELO_PADRAO = settings().llm_modelo

PROMPT = """Você prepara texto para narração em português brasileiro.

Contexto anterior: {antes}
TRECHO A EXPANDIR: {span}
Contexto seguinte: {depois}

Escreva APENAS como esse trecho deve ser FALADO em voz alta, por extenso.
Regras:
- responda somente com a forma falada do trecho, nada mais;
- não explique, não use aspas, não repita o contexto;
- não acrescente nem remova informação;
- se for um ano, leia como ano; se for quantidade, leia como número;
- se for ordinal (como "Pedro I"), escreva o ordinal por extenso;
- CONCORDE O GÊNERO com a pessoa citada: "Dom Pedro I" é "Primeiro" (masculino),
  "Rainha Elizabeth II" é "Segunda" (feminino).

Forma falada:"""

# Palavras que o LLM pode legitimamente introduzir ao expandir um numero.
_VOCABULARIO = set("""
zero um uma dois duas tres três quatro cinco seis sete oito nove dez onze doze treze
catorze quatorze quinze dezesseis dezessete dezoito dezenove vinte trinta quarenta
cinquenta sessenta setenta oitenta noventa cem cento duzentos duzentas trezentos
trezentas quatrocentos quatrocentas quinhentos quinhentas seiscentos seiscentas
setecentos setecentas oitocentos oitocentas novecentos novecentas mil milhao milhão
milhoes milhões bilhao bilhão bilhoes bilhões e de do da primeiro primeira segundo
segunda terceiro terceira quarto quarta quinto quinta sexto sexta setimo sétimo
setima sétima oitavo oitava nono nona decimo décimo decima décima undecimo undécimo undecima undécima duodecimo duodécimo duodecima duodécima
""".split())


@dataclass
class Sugestao:
    span: str
    expansao: str
    aceita: bool
    motivo: str = ""


def _sem_acento(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _numeros_de(texto: str) -> set[int]:
    """Reconstroi os valores numericos presentes numa expansao por extenso."""
    encontrados = {int(m) for m in re.findall(r"\d+", texto)}
    # Compara por reconstrucao: gera o extenso de cada candidato e procura no texto
    return encontrados


def validar(span: str, expansao: str) -> tuple[bool, str]:
    """Aceita a expansao apenas se ela for verificavelmente equivalente ao span."""
    exp = expansao.strip().strip('"').strip("'")
    if not exp:
        return False, "resposta vazia"
    if len(exp) > max(120, len(span) * 25):
        return False, "resposta longa demais para uma expansão de span"
    if "\n" in exp.strip():
        return False, "resposta com múltiplas linhas (provável explicação)"

    palavras = [p for p in re.findall(r"[a-zà-úA-ZÀ-Ú]+", exp)]
    fora = [p for p in palavras if _sem_acento(p) not in _VOCABULARIO]
    if fora:
        return False, f"palavras fora do vocabulário permitido: {fora[:4]}"

    # Reversibilidade: o numero do span tem de ser reproduzivel pela expansao.
    numeros = re.findall(r"\d+", span)
    if numeros:
        alvo = int(numeros[0])
        candidatos = {
            _sem_acento(num2words(alvo, lang="pt_BR").replace(",", "")),
            _sem_acento(num2words(alvo, lang="pt_BR", to="ordinal").replace(",", "")),
        }
        if _sem_acento(exp) not in candidatos:
            return False, (f"expansão não reproduz o valor {alvo} "
                           f"(esperado uma de {sorted(candidatos)})")
    return True, ""


def fallback_romano(span: str) -> str | None:
    """Se o LLM falhar num romano apos nome proprio, o ordinal masculino e a leitura
    correta na esmagadora maioria dos casos historicos ("Dom Pedro I" -> "Primeiro").
    O genero e a unica parte de fato ambigua -- e por isso o LLM tenta primeiro."""
    from .rules import ROMANOS

    valor = ROMANOS.get(span.upper())
    if valor is None:
        return None
    return num2words(valor, lang="pt_BR", to="ordinal").replace(",", "").capitalize()


def perguntar(span: str, antes: str, depois: str, modelo: str = MODELO_PADRAO,
              url: str = OLLAMA_URL, timeout: int = 60, num_predict: int = 600) -> str:
    corpo = json.dumps({
        "model": modelo,
        "prompt": PROMPT.format(span=span, antes=antes[-160:], depois=depois[:160]),
        "stream": False,
        # gemma4:12b consome ~230 tokens de raciocinio interno antes de emitir a
        # resposta: com num_predict baixo ele para por 'length' e devolve string
        # vazia. 600 e folgado o bastante para a resposta sair.
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }).encode()
    req = request.Request(url, data=corpo, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()


def resolver(texto: str, ambiguos, modelo: str = MODELO_PADRAO,
             cache: dict[str, str] | None = None) -> tuple[str, list[Sugestao]]:
    """Aplica a Camada 2 sobre os spans ambíguos. Mantém o original se algo falhar."""
    cache = cache if cache is not None else {}
    sugestoes: list[Sugestao] = []
    # de tras para frente para os offsets nao invalidarem
    for amb in sorted(ambiguos, key=lambda a: a.start, reverse=True):
        # A chave inclui a palavra anterior: sem isso "Elizabeth II" -> "Segunda"
        # contaminaria "Dom Pedro II", que deve ser "Segundo". O genero depende do
        # contexto, logo o contexto faz parte da identidade da entrada em cache.
        anterior = (re.findall(r"([\wÀ-ú]+)\s*$", texto[:amb.start]) or [""])[0]
        chave = f"{amb.text}|{amb.reason}|{anterior.lower()}"
        if chave in cache:
            exp, ok, motivo = cache[chave], True, "cache"
        else:
            exp, ok, motivo = "", False, ""
            for orcamento in (600, 1200):  # gemma4 as vezes estoura o 1o orcamento
                try:
                    exp = perguntar(amb.text, texto[:amb.start], texto[amb.end:],
                                    modelo, num_predict=orcamento)
                except (error.URLError, TimeoutError, KeyError) as e:
                    motivo = f"LLM indisponível: {e}"
                    break
                ok, motivo = validar(amb.text, exp)
                if ok:
                    break
            if not ok and (alt := fallback_romano(amb.text)):
                exp, ok, motivo = alt, True, "fallback determinístico (ordinal masculino)"
            if ok:
                cache[chave] = exp.strip().strip('"')
        if not ok:
            sugestoes.append(Sugestao(amb.text, exp, False, motivo))
            continue
        exp = cache[chave]
        texto = texto[:amb.start] + exp + texto[amb.end:]
        sugestoes.append(Sugestao(amb.text, exp, True, motivo))
    return texto, sugestoes
