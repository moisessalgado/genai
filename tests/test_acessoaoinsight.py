"""Ingest das páginas de sutta do acessoaoinsight.net.

O que importa testar aqui é o que o pipeline NÃO consegue perceber depois: um
parágrafo de aparato editorial que vira narração, uma quebra de linha do Word
que corta uma frase no meio, ou uma página inexistente que passa por sutta vazio.
Tudo isso sai do outro lado como um MP4 de som correto e conteúdo errado.

A fixture é a página real de ANIV.45, byte a byte como o servidor a entrega
(windows-1252, HTML exportado do Word). Não é editada de propósito: o valor dela
está justamente em conter as sujeiras que o parser precisa aguentar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.ingest.acessoaoinsight import (
    codigo_da_url, decodificar, extrair, normalizar_url,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ANIV.45.php.html"
URL = "https://www.acessoaoinsight.net/sutta/ANIV.45.php.html"


@pytest.fixture
def sutta():
    return extrair(decodificar(FIXTURE.read_bytes()), URL)


# --- codificação -------------------------------------------------------------

def test_a_pagina_e_cp1252_e_nao_o_latin1_que_ela_declara():
    """A página diz `charset=ISO-8859-1` e mente: as aspas curvas do Word ocupam
    0x93/0x94, que em latin-1 são caracteres de controle. `roles.py` procura
    justamente essas aspas para achar a fala citada."""
    bruto = FIXTURE.read_bytes()
    assert "“É possível" in decodificar(bruto)
    assert "\x93É possível" in bruto.decode("iso-8859-1")   # o que NÃO queremos


# --- recorte da página -------------------------------------------------------

def test_extrai_referencia_e_os_dois_titulos(sutta):
    assert sutta.referencia == "Anguttara Nikaya IV.45"
    assert sutta.pali == "Rohitassa Sutta"
    assert sutta.portugues == "Rohitassa"


def test_pagina_inexistente_e_erro_e_nao_sutta_vazio():
    """O servidor devolve HTTP 200 com uma página genérica para endereço que não
    existe. Sem esta guarda o pipeline criaria um projeto vazio e só falharia
    lá no `export`, depois de o operador ter esperado a síntese."""
    with pytest.raises(ValueError, match="INICIO DO TEXTO"):
        extrair("<html><body>página qualquer</body></html>", URL)


def test_notas_e_navegacao_ficam_de_fora(sutta):
    """Notas, 'Veja também' e '>> Próximo Sutta' vivem depois do último `<hr>`."""
    texto = sutta.texto()
    assert "Próximo Sutta" not in texto
    assert "Veja também" not in texto
    assert "SN II.26" not in texto


def test_a_licenca_de_distribuicao_e_colhida_da_propria_pagina(sutta):
    assert sutta.licenca.startswith("Somente para distribuição gratuita.")
    assert "nenhum custo seja cobrado" in sutta.licenca


def test_nenhum_residuo_de_html_no_texto(sutta):
    """O último `<p>` da página carrega o `<!--` que abre o rodapé, sem `>` para
    fechar — e a remoção de tags só casa tag fechada."""
    texto = sutta.texto()
    for lixo in ("<", ">", "&nbsp;", "&quot;", "o:p"):
        assert lixo not in texto, lixo


# --- o que é quebra de linha e o que não é -----------------------------------

def test_br_e_verso_mas_a_quebra_do_word_nao_e():
    """O Word quebra a fonte na coluna 78; `<br>` é o verso do sutta. Preservar
    as duas cortaria "Assim ouvi." no meio."""
    pagina = (
        "<!-- INICIO DO TEXTO -->"
        "<p class=Tit3>Ref 1.1</p><p class=Tit1>Teste Sutta</p>"
        "<p class=Tit1>Teste</p><hr>"
        "<p class=Normal>Assim\novi. Em certa ocasião o Abençoado estava lá.</p>"
        "<p class=Normal>Primeiro verso,<br>\nsegundo verso.</p>"
        "<!-- FIM DO TEXTO -->"
    )
    s = extrair(pagina, URL)
    assert s.paragrafos[0] == "Assim ovi. Em certa ocasião o Abençoado estava lá."
    assert s.paragrafos[1] == "Primeiro verso,\nsegundo verso."


def test_numeracao_de_paragrafo_da_edicao_e_removida():
    """"1. Assim ouvi." é referência de edição. Narrada, vira "um. Assim ouvi."."""
    pagina = (
        "<!-- INICIO DO TEXTO -->"
        "<p class=Tit1>Teste Sutta</p><p class=Tit1>Teste</p><hr>"
        "<p class=Normal>1. Assim ouvi. Em certa ocasião o Abençoado estava lá.</p>"
        "<!-- FIM DO TEXTO -->"
    )
    assert extrair(pagina, URL).paragrafos[0].startswith("Assim ouvi.")


# --- o texto que vai ao TTS --------------------------------------------------

def test_a_referencia_da_colecao_nao_entra_na_narracao(sutta):
    """Medido: o normalizador transforma "Anguttara Nikaya IV.45" em "Anguttara
    Nikaya IVquarenta e cinco" — o romano fica sem leitura e o `.45` vira
    cardinal colado. A referência vive no título e na descrição."""
    assert "IV.45" not in sutta.texto()
    assert "Anguttara" not in sutta.texto()
    assert "Anguttara Nikaya IV.45" in sutta.titulo


def test_titulo_em_portugues_repetido_nao_e_dito_duas_vezes(sutta):
    """"Rohitassa Sutta" / "Rohitassa": o segundo não acrescenta nada."""
    assert sutta.texto().startswith("Rohitassa Sutta.\n\n")


def test_titulo_em_portugues_distinto_abre_a_narracao():
    pagina = (
        "<!-- INICIO DO TEXTO -->"
        "<p class=Tit3>Samyutta Nikaya LVI.11</p>"
        "<p class=Tit1>Dhammacakkapavattana Sutta</p>"
        "<p class=Tit1>Colocando a roda do Dhamma em movimento</p><hr>"
        "<p class=Normal>Assim ouvi, em certa ocasião, e assim por diante.</p>"
        "<!-- FIM DO TEXTO -->"
    )
    s = extrair(pagina, URL)
    assert s.texto().startswith(
        "Dhammacakkapavattana Sutta. Colocando a roda do Dhamma em movimento.")
    assert s.titulo.endswith("(Samyutta Nikaya LVI.11)")


# --- procedência -------------------------------------------------------------

def test_rights_registra_licenciado_e_nao_dominio_publico(sutta):
    """O sutta é antigo; a TRADUÇÃO é obra derivada com direito próprio do
    tradutor (TDD 14.3). Declarar domínio público aqui seria falso."""
    r = sutta.rights()
    assert r["status"] == "licenciado"
    assert r["tradutor"].startswith("Acesso ao Insight")
    assert URL in r["fonte"]
    assert "nenhum custo seja cobrado" in r["licenca"]
    # Data de download não é data de conferência: quem verifica direitos é gente.
    assert r["verificado_em"] is None


# --- endereços ---------------------------------------------------------------

@pytest.mark.parametrize("entrada", [
    URL,
    "ANIV.45",
    "sutta/ANIV.45.php.html",
    "/sutta/ANIV.45.php.html",
])
def test_normalizar_url_aceita_url_caminho_ou_codigo(entrada):
    assert normalizar_url(entrada) == URL


def test_codigo_da_url(sutta):
    assert codigo_da_url(URL) == "ANIV.45"
    assert sutta.slug == "aniv45-rohitassa"
