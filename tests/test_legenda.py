"""Legenda sincronizada: o que dá para verificar sem olhar o vídeo.

Os tempos vêm do mesmo ato de montagem que gera o áudio, então o que se testa
aqui é a repartição de um segmento em várias legendas — que foi onde os dois
defeitos apareceram.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_legenda_quebra_na_pausa_da_fala_e_nao_na_contagem_de_letras():
    """Regressão: repartir uma legenda longa proporcionalmente aos caracteres
    supõe fala com taxa constante de letras por segundo. Não tem — e o erro
    aparece como legenda adiantada em relação ao áudio."""
    from genai.video.legenda import cues

    marcas = [(0.0, 20.0)]
    textos = ["Palavra " * 30]

    sem = cues(marcas, textos)
    assert len(sem) > 1, "o texto precisa render mais de uma legenda"
    alvo = sem[0][1]

    # a fala pausou 1 s DEPOIS do alvo proporcional: é esse o erro que se corrige
    com = cues(marcas, textos, pausas=[[alvo + 1.0]])
    assert abs(com[0][1] - (alvo + 1.0)) < 0.01
    # e a legenda seguinte começa onde a anterior terminou, sem buraco nem sobra
    assert com[1][0] == com[0][1]


def test_legenda_sem_pausa_por_perto_mantem_o_alvo_proporcional():
    """O ímã só puxa se houver pausa dentro da janela; longe dela, nada muda."""
    from genai.video.legenda import cues

    marcas = [(0.0, 20.0)]
    textos = ["Palavra " * 30]
    sem = cues(marcas, textos)
    com = cues(marcas, textos, pausas=[[sem[0][1] + 5.0]])
    assert [c[:2] for c in sem] == [c[:2] for c in com]


def test_legenda_nunca_volta_no_tempo():
    """Duas quebras não podem escolher a mesma pausa, nem sair de ordem."""
    from genai.video.legenda import cues

    marcas = [(0.0, 40.0)]
    textos = ["Palavra " * 90]
    c = cues(marcas, textos, pausas=[[9.0, 9.1, 9.2, 20.0, 31.0]])
    assert len(c) > 2
    for a, b in zip(c, c[1:]):
        assert a[1] <= b[0] + 1e-9, "legenda começa antes da anterior terminar"
        assert a[0] < a[1], "legenda de duração nula ou negativa"


def test_legenda_nunca_tem_duracao_zero():
    """Regressão medida no capítulo real: 44 das 339 legendas saíam com duração
    zero ou negativa — texto que nunca chega a aparecer na tela. O ímã consumia
    o tempo do pedaço seguinte quando o alvo proporcional já estava atrasado."""
    from genai.video.legenda import MIN_CUE_S, cues

    marcas = [(0.0, 6.0)]                     # curto para o tanto de texto
    textos = ["Palavra " * 60]
    # pausas amontoadas logo no início: o pior caso para o ímã
    c = cues(marcas, textos, pausas=[[0.2, 0.25, 0.3, 0.35, 0.4]])
    assert len(c) > 3
    for ini, fim, _ in c:
        assert fim - ini >= MIN_CUE_S * 0.5, f"legenda de {fim - ini:.3f}s"
    for a, b in zip(c, c[1:]):
        assert a[1] <= b[0] + 1e-9


def test_legenda_recupera_a_grafia_escondida_pelo_respelling():
    """O `text` vai respelado para o motor pronunciar; na tela tem de aparecer a
    grafia de verdade. Defeito visto pelo operador no vídeo: a abertura exibia
    'dama tchaca pavátana súta'."""
    from genai.video.legenda import desfazer_lexico

    lex = {"Dhammacakkapavattana": "dama tchaca pavátana", "Sutta": "súta",
           "sutta": "súta", "Tathagata": "tatágata", "Kondañña": "condánha"}
    assert desfazer_lexico("Samyutta Nikaya dama tchaca pavátana súta", lex) == \
        "Samyutta Nikaya Dhammacakkapavattana Sutta"
    assert desfazer_lexico("o tatágata despertou", lex) == "o Tathagata despertou"


def test_legenda_nao_troca_pedaco_de_palavra():
    """A substituição é por palavra inteira: 'súta' dentro de outra palavra não
    pode virar 'Sutta'."""
    from genai.video.legenda import desfazer_lexico

    assert desfazer_lexico("consúmio", {"Sutta": "súta"}) == "consúmio"


def test_legenda_nao_usa_o_source_do_segmento():
    """`source` guarda o PARÁGRAFO inteiro, repetido em cada pedaço cortado dele
    — no ch01, 25.914 caracteres contra 7.527 de `text`. Usá-lo faz a legenda
    exibir texto ainda não falado, que é como ela 'adianta'."""
    from genai.video.legenda import cues

    paragrafo = "Uma frase. " * 12
    pedaco = "Uma frase."
    # o segmento dura 4 s e fala só o pedaço; o parágrafo levaria muito mais
    poucas = cues([(0.0, 4.0)], [pedaco])
    muitas = cues([(0.0, 4.0)], [paragrafo])
    assert len(poucas) == 1
    assert len(muitas) > len(poucas), "o parágrafo inteiro estoura o tempo do segmento"
