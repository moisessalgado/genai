"""Deteccao de capitulos: o que e titulo e o que e prosa."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.ingest.loader import detectar_capitulos

# -- deteccao de capitulos ----------------------------------------------------

@pytest.mark.parametrize("linha", [
    "Discurso de Dissolução", "Como os homens adoram", "Livro dos dias",
    "Vocês me ouvem há três anos", "Isso não me preocupa", "Minha preocupação",
])
def test_prosa_que_comeca_por_letra_romana_nao_e_titulo(linha):
    """I, V, X, L, C, D e M abrem meia lingua portuguesa: sem separador nao ha titulo."""
    texto = f"{linha}\n\nCorpo do parágrafo seguinte, com prosa de verdade."
    assert [t for t, _ in detectar_capitulos(texto)] == ["Texto completo"]


@pytest.mark.parametrize("linha,titulo", [
    ("IV", "IV"), ("XII.", "XII."), ("I — A partida", "I — A partida"),
    ("III · O regresso", "III · O regresso"), ("II: O retorno", "II: O retorno"),
])
def test_numeral_romano_isolado_ainda_e_titulo(linha, titulo):
    caps = detectar_capitulos(f"Prefácio da obra.\n\n{linha}\n\nCorpo do capítulo.")
    assert titulo in [t for t, _ in caps]


def test_texto_antes_do_primeiro_capitulo_nao_se_perde():
    """Rosto, autor e epigrafe vinham antes do primeiro marco e eram descartados."""
    texto = ("Jiddu Krishnamurti\n\nNota do editor sobre a ocasião do discurso.\n\n"
             "Capítulo I — A abertura\n\nCorpo do primeiro capítulo.")
    caps = detectar_capitulos(texto)
    assert caps[0][0] == "Abertura"
    assert "Jiddu Krishnamurti" in caps[0][1]
    assert "Nota do editor" in caps[0][1]


def test_nenhum_caractere_se_perde_na_divisao_em_capitulos():
    texto = ("Autor da obra\n\nEpígrafe.\n\nCapítulo I — A partida\n\nCorpo um.\n\n"
             "Capítulo II — O retorno\n\nCorpo dois.")
    caps = detectar_capitulos(texto)
    for trecho in ("Autor da obra", "Epígrafe.", "Corpo um.", "Corpo dois."):
        assert any(trecho in corpo for _, corpo in caps), trecho


# -- notas de rodape ----------------------------------------------------------

def test_marcador_de_nota_de_rodape_e_removido():
    """Caso real (Dhammacakkappavattana Sutta): "[1]" e "[2]" chegavam ao
    normalizador e viravam "um" e "dois" no meio da narração."""
    from genai.ingest.loader import limpar

    t = limpar("despegar desse mesmo desejo.[1]\n\nnas suas três fases, [2] não estava")
    assert "[1]" not in t and "[2]" not in t
    assert "desse mesmo desejo." in t
    assert "três fases" in t


def test_colchete_com_texto_nao_e_removido():
    """Só marcador numérico sai; interpolação do tradutor é conteúdo."""
    from genai.ingest.loader import limpar

    assert "[o Buda]" in limpar("Então ele [o Buda] disse.")


# -- paragrafos curtos demais para sintetizar ---------------------------------

def test_titulo_curto_e_fundido_ao_paragrafo_seguinte():
    """Medido em dois livros: parágrafo isolado curto vira gibberish no TTS.
    "Dhammacakkapavattana Sutta" sozinho saiu "Deu uma chaca pavada na sota";
    com a linha seguinte junto, sai correto."""
    from genai.ingest.loader import paragrafos

    ps = paragrafos("Dhammacakkapavattana Sutta\n\n"
                    "Colocando a roda do Dhamma em movimento, disse ele.")
    assert len(ps) == 1
    assert ps[0].startswith("Dhammacakkapavattana Sutta Colocando")


def test_varios_cacos_seguidos_viram_um_paragrafo_so():
    from genai.ingest.loader import paragrafos

    ps = paragrafos("Autor\n\nTítulo\n\n1929\n\n" + "Corpo do texto com tamanho normal. " * 2)
    assert len(ps) == 1


def test_caco_no_fim_cola_no_anterior():
    """Sem isto o último fragmento iria sozinho ao TTS, que é o caso ruim."""
    from genai.ingest.loader import paragrafos

    ps = paragrafos("Corpo do texto com tamanho perfeitamente normal aqui.\n\nFim.")
    assert len(ps) == 1 and ps[0].endswith("Fim.")


def test_paragrafos_normais_nao_sao_fundidos():
    from genai.ingest.loader import paragrafos

    ps = paragrafos("Primeiro parágrafo com tamanho normal e suficiente.\n\n"
                    "Segundo parágrafo, também com tamanho normal.")
    assert len(ps) == 2
