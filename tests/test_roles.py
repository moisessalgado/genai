"""Detecção de papel de fala por parágrafo (text/roles.py).

Não havia teste nenhum para este módulo antes — escrito ao adicionar o
travessão de diálogo (ver `dividir_por_papel`), para travar o comportamento
por aspas que já existia e cobrir o caso novo. Exemplos de frase são
inventados, não vêm de nenhum livro real.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.text.roles import (
    PAPEL_CITACAO, PAPEL_NARRADOR, dividir_por_papel, tem_dialogo,
)


# --- aspas (comportamento pré-existente, travado aqui) ------------------------

def test_fala_entre_aspas_retas_vira_citacao():
    # Cada pedaço (antes, fala, depois) precisa passar de MIN_SINTETIZAVEL,
    # senão a divisão inteira é abandonada -- ver test_fala_curta_demais_*.
    partes = dividir_por_papel(
        'Ele parou no meio do caminho e disse, hesitante, '
        '"isto aqui é realmente muito mais incrível do que eu esperava" '
        'e depois continuou andando devagar pela estrada.')
    papeis = [p for p, _ in partes]
    assert PAPEL_CITACAO in papeis
    assert PAPEL_NARRADOR in papeis


def test_paragrafo_sem_fala_e_so_narrador():
    partes = dividir_por_papel("Era uma manhã tranquila no vilarejo distante.")
    assert partes == [(PAPEL_NARRADOR, "Era uma manhã tranquila no vilarejo distante.")]


def test_marcacao_explicita_de_personagem():
    partes = dividir_por_papel("[[voz:fulano]]Isto é o que fulano diz agora mesmo.")
    assert partes == [("fulano", "Isto é o que fulano diz agora mesmo.")]


def test_fala_curta_demais_fica_toda_em_narracao():
    """Abaixo de MIN_SINTETIZAVEL o Chatterbox arrasta o áudio -- a divisão
    inteira é abandonada, não só o pedaço curto."""
    original = 'Ele disse "oi" e saiu correndo depressa pela rua vazia.'
    partes = dividir_por_papel(original)
    assert partes == [(PAPEL_NARRADOR, original)]


# --- travessão (caso novo) -----------------------------------------------------

def test_paragrafo_travessao_unico_vira_citacao():
    partes = dividir_por_papel("— Isto é uma fala inventada só para o teste.")
    assert partes == [(PAPEL_CITACAO, "Isto é uma fala inventada só para o teste.")]


def test_travessao_meia_risca_tambem_conta():
    partes = dividir_por_papel("– Outra fala inventada, com meia-risca em vez de travessão.")
    assert partes[0][0] == PAPEL_CITACAO


def test_travessao_duplo_no_mesmo_paragrafo_fica_em_narracao():
    """Segundo travessão costuma ser interjeição do narrador embutida
    ("— fala — disse alguém — mais fala") -- caso ambíguo, fica de fora de
    propósito em vez de tentar separar errado."""
    original = "— Isto é uma fala qualquer — disse a personagem, virando-se."
    partes = dividir_por_papel(original)
    assert partes == [(PAPEL_NARRADOR, original)]


def test_hifen_solto_no_inicio_nao_e_travessao_de_dialogo():
    """Hífen de verdade (não travessão/meia-risca) não deve disparar a
    divisão -- evita falso positivo em texto comum com hífen de abertura."""
    original = "- isto não é diálogo, é só uma linha começando com hífen comum."
    partes = dividir_por_papel(original)
    assert partes == [(PAPEL_NARRADOR, original)]


def test_travessao_seguido_de_fala_curta_demais_fica_em_narracao():
    original = "— Não."
    partes = dividir_por_papel(original)
    assert partes == [(PAPEL_NARRADOR, original)]


def test_aspas_tem_prioridade_sobre_travessao():
    """Um parágrafo com aspas passa pelo caminho de aspas normalmente, mesmo
    que também comece com travessão em outro lugar -- o travessão só é
    tentado quando as aspas não acham nada."""
    original = ('— Isto aqui é um parágrafo bem mais longo que começa com '
                'travessão, mas tem uma "citação entre aspas relativamente '
                'longa o suficiente para sobreviver ao corte" e depois '
                'continua com mais um bom pedaço de narração no final.')
    partes = dividir_por_papel(original)
    assert any(p == PAPEL_CITACAO for p, _ in partes)


# --- heurística de aviso -------------------------------------------------------

def test_tem_dialogo_detecta_aspas_e_travessao():
    assert tem_dialogo('Ele disse "algo relativamente longo aqui" e saiu.')
    assert tem_dialogo("— Uma fala qualquer introduzida por travessão.")
    assert not tem_dialogo("Um parágrafo qualquer, sem nenhuma fala citada.")
