"""Deteccao de papeis de fala num paragrafo (narracao vs. fala citada).

Objetivo: dar voz propria ao discurso direto sem que ninguem tenha de anotar o
livro a mao. A deteccao e CONSERVADORA -- na duvida, e narracao. Um falso positivo
troca a voz no meio de uma frase, defeito bem mais audivel do que uma citacao lida
na voz do narrador.

Nao tenta identificar QUEM fala (isso exige analise de correferencia e erra muito).
Entrega papeis genericos; quem quiser um elenco nominal usa marcacao explicita.
"""
from __future__ import annotations

import re

PAPEL_NARRADOR = "narrador"
PAPEL_CITACAO = "citacao"

# Aspas retas, curvas e simples; travessao de dialogo no inicio da linha.
_ASPAS = re.compile(r'"([^"]{12,})"|[“]([^”]{12,})[”]|\'([^\']{12,})\'')
_TRAVESSAO = re.compile(r"^\s*[—–-]\s*(.+)$", re.MULTILINE)
# So para o paragrafo INTEIRO ser uma fala (nao a heuristica frouxa de
# `tem_dialogo`, que aceita ate hifen solto): exige travessao/meia-risca de
# verdade (nao hifen "-", que aparece toda hora em prosa comum e nao e
# dialogo) e SO conta quando o paragrafo tem exatamente UM travessao no total
# -- um segundo travessao costuma ser interjeicao do narrador embutida
# ("-- Fala -- disse Fulano."), e essa mistura fica de fora de proposito.
_TRAVESSAO_PARAGRAFO_INTEIRO = re.compile(r"^[—–]\s*(.+)$", re.DOTALL)
# Marcacao explicita do usuario, para quando ele quiser controlar o elenco.
_EXPLICITO = re.compile(r"\[\[voz:([a-z0-9_-]+)\]\]\s*(.+?)(?=\[\[voz:|$)",
                        re.IGNORECASE | re.DOTALL)

MIN_CARACTERES = 12

# Abaixo disto o Chatterbox nao sintetiza de forma confiavel: arrasta o audio
# (medido: "respondeu o diabo." saiu a 5,8 c/s, contra 14-17 c/s do normal) ou
# alucina. Se a divisao por papel produzir QUALQUER caco menor que isto, a
# divisao inteira e abandonada e o paragrafo vira narracao -- e o comportamento
# conservador prometido: na duvida, e narracao.
MIN_SINTETIZAVEL = 25


def dividir_por_papel(paragrafo: str) -> list[tuple[str, str]]:
    """Devolve [(papel, trecho)] na ordem do texto."""
    explicitos = list(_EXPLICITO.finditer(paragrafo))
    if explicitos:
        return [(m.group(1).lower(), m.group(2).strip()) for m in explicitos
                if m.group(2).strip()]

    partes: list[tuple[str, str]] = []
    pos = 0
    for m in _ASPAS.finditer(paragrafo):
        fala = next(g for g in m.groups() if g)
        if len(fala.strip()) < MIN_CARACTERES:
            continue
        antes = paragrafo[pos:m.start()].strip()
        if antes:
            partes.append((PAPEL_NARRADOR, antes))
        partes.append((PAPEL_CITACAO, fala.strip()))
        pos = m.end()
    resto = paragrafo[pos:].strip()
    if resto:
        partes.append((PAPEL_NARRADOR, resto))
    partes = _consolidar(partes)

    if not partes or (len(partes) == 1 and partes[0][0] == PAPEL_NARRADOR):
        # Nenhuma aspa casou -- tenta o travessão de diálogo, só no caso
        # inequívoco (parágrafo inteiro é UMA fala só, sem interjeição do
        # narrador embutida). Sem isto, um livro que usa travessão em vez de
        # aspas (a esmagadora maioria da ficção brasileira, travessão por
        # parágrafo de fala) narra todo o diálogo na voz do narrador, e o
        # elenco por personagem não tem o que promover.
        m = _TRAVESSAO_PARAGRAFO_INTEIRO.match(paragrafo)
        if m and (paragrafo.count("—") + paragrafo.count("–")) == 1:
            fala = m.group(1).strip()
            if len(fala) >= MIN_CARACTERES:
                partes = [(PAPEL_CITACAO, fala)]

    if any(len(tr) < MIN_SINTETIZAVEL for _, tr in partes):
        # Diálogo picado (fala curta, atribuição curta, fala curta) não sobrevive à
        # troca de voz: cada caco vira um chunk defeituoso. Lido inteiro na voz do
        # narrador fica correto -- só menos teatral.
        return [(PAPEL_NARRADOR, paragrafo.strip())]
    return partes or [(PAPEL_NARRADOR, paragrafo.strip())]


def _consolidar(partes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Funde restos curtos e junta partes vizinhas do mesmo papel.

    A separacao por aspas deixa cacos: a pontuacao entre duas falas vira um
    segmento "." de um caractere, que o TTS transforma em ruido. Esses cacos sao
    absorvidos pelo vizinho anterior, mantendo o papel dele.
    """
    out: list[tuple[str, str]] = []
    for papel, trecho in partes:
        trecho = trecho.strip()
        if not trecho:
            continue
        so_pontuacao = not re.search(r"[a-zA-ZÀ-ú]", trecho)
        curto = len(trecho) < MIN_CARACTERES
        if out and (so_pontuacao or curto):
            papel_ant, texto_ant = out[-1]
            sep = "" if so_pontuacao and trecho[0] in ".,;:!?" else " "
            out[-1] = (papel_ant, _limpar_pontuacao(f"{texto_ant}{sep}{trecho}"))
            continue
        if out and out[-1][0] == papel:
            out[-1] = (papel, _limpar_pontuacao(f"{out[-1][1]} {trecho}"))
            continue
        out.append((papel, _limpar_pontuacao(trecho)))
    return out


def _limpar_pontuacao(texto: str) -> str:
    """Colapsa pontuacao dobrada criada pela fusao de cacos ("?." -> "?")."""
    texto = re.sub(r"([.!?])[\s]*[.,;:]+", r"\1", texto)
    texto = re.sub(r"\s+([.,;:!?])", r"\1", texto)
    # trecho nao comeca com virgula/ponto-e-virgula: o corte da aspa deixa esse
    # resto pendurado (", respondeu o diabo") e o TTS abre com uma pausa estranha
    texto = re.sub(r"^[\s,;:]+", "", texto)
    return re.sub(r"\s{2,}", " ", texto).strip()


def tem_dialogo(texto: str) -> bool:
    """Heuristica barata para avisar o usuario que vale configurar um cast."""
    return bool(_ASPAS.search(texto)) or bool(_TRAVESSAO.search(texto))
