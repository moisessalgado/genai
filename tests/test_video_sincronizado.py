"""Slideshow sincronizado: o que dá para verificar sem GPU/Ollama."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audiofactory.video.slides import _duracoes, entradas, filtro
from audiofactory.video.sincronizado import janelas_do_srt

SRT_EXEMPLO = """1
00:00:00,000 --> 00:00:03,000
Era uma vez uma menina

2
00:00:03,000 --> 00:00:07,500
que morava numa fazenda.

3
00:00:07,500 --> 00:00:12,000
Ela tinha um nariz arrebitado

4
00:00:12,000 --> 00:00:14,000
e uma boneca de pano.

5
00:00:14,000 --> 00:00:16,500
Um dia, foi passear no quintal.
"""


def test_janelas_agrupa_cues_ate_acumular_a_janela(tmp_path):
    srt = tmp_path / "ch01.srt"
    srt.write_text(SRT_EXEMPLO, encoding="utf-8")
    janelas = janelas_do_srt(srt, janela_s=10.0)
    # cues 1-3 somam 12s (>=10), fecha a primeira janela; cues 4-5 fecham a segunda.
    assert len(janelas) == 2
    assert janelas[0].inicio_s == 0.0
    assert janelas[0].fim_s == 12.0
    assert "menina" in janelas[0].texto and "arrebitado" in janelas[0].texto
    assert janelas[1].inicio_s == 12.0
    assert janelas[1].fim_s == 16.5


def test_janelas_nao_corta_no_meio_de_uma_cue(tmp_path):
    """Nenhuma janela pode terminar num instante que nao seja o fim de uma cue —
    senao a imagem descreveria uma frase pela metade."""
    srt = tmp_path / "ch01.srt"
    srt.write_text(SRT_EXEMPLO, encoding="utf-8")
    fins_de_cue = {0.0, 3.0, 7.5, 12.0, 14.0, 16.5}
    for j in janelas_do_srt(srt, janela_s=10.0):
        assert j.inicio_s in fins_de_cue
        assert j.fim_s in fins_de_cue


def test_janelas_srt_vazio(tmp_path):
    srt = tmp_path / "vazio.srt"
    srt.write_text("", encoding="utf-8")
    assert janelas_do_srt(srt) == []


def test_duracoes_aceita_numero_unico():
    assert _duracoes(5.0, 3) == [5.0, 5.0, 5.0]


def test_duracoes_aceita_lista():
    assert _duracoes([1.0, 2.0, 3.0], 3) == [1.0, 2.0, 3.0]


def test_duracoes_lista_com_tamanho_errado_falha():
    import pytest
    with pytest.raises(ValueError):
        _duracoes([1.0, 2.0], 3)


def test_entradas_com_duracao_variavel_por_imagem(tmp_path):
    imgs = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    args = entradas(imgs, [3.0, 7.0], fps=25, largura=1920, veu=False)
    # cada imagem carrega o seu proprio "-t"
    assert args[args.index("-t") + 1] == "3.000"
    resto = args[args.index("-t", args.index("-t") + 1):]
    assert resto[1] == "7.000"


def test_filtro_offset_variavel_reduz_ao_caso_uniforme(tmp_path):
    """Com todas as durações iguais, o offset do preset sincronizado tem de
    bater com a fórmula já usada pelo preset `slides` (k*(cada-cruzamento))."""
    imgs = [tmp_path / f"{i}.jpg" for i in range(4)]
    cada, cruzamento = 10.0, 2.0
    f_uniforme = filtro(imgs, cada, cruzamento, 1920, 1080, veu=False)
    f_lista = filtro(imgs, [cada] * len(imgs), cruzamento, 1920, 1080, veu=False)
    assert f_uniforme == f_lista


def test_filtro_offset_variavel_acumula_duracoes_reais(tmp_path):
    imgs = [tmp_path / f"{i}.jpg" for i in range(3)]
    duracoes = [8.0, 12.0, 5.0]
    cruzamento = 2.0
    f = filtro(imgs, duracoes, cruzamento, 1920, 1080, veu=False)
    # offset da 1a transicao: soma(duracoes[:1]) - 1*cruzamento = 8-2=6
    assert "offset=6.000" in f
    # offset da 2a transicao: soma(duracoes[:2]) - 2*cruzamento = 20-4=16
    assert "offset=16.000" in f
