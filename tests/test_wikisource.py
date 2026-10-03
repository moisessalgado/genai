"""Ingest de obras da Wikisource em português (pt.wikisource.org).

O que importa testar aqui é o mesmo tipo de coisa que `test_acessoaoinsight.py`
cobre para o outro ingest: aparato editorial que vira narração por engano, e uma
garantia estrutural específica desta fonte — uma página de capítulo não pode
fragmentar em vários `Chapter` só porque a prosa tem marcadores internos em
romano (medido: "I — NARIZINHO" faz `detectar_capitulos()` genérico produzir 8
pedaços de uma página que é UM capítulo só).

As fixtures são páginas reais (índice e capítulo 1 de "As Reinações de
Narizinho", Monteiro Lobato, 1931 — domínio público no Brasil), baixadas ao
vivo e não editadas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.core.ingest import wikisource as wk
from genai.core.ingest.loader import detectar_capitulos

FIXTURES = Path(__file__).parent / "fixtures" / "wikisource"
CAPITULO_HTML = FIXTURES / "narizinho-arrebitado.html"
INDICE_HTML = FIXTURES / "indice.html"
URL_CAPITULO = "https://pt.wikisource.org/wiki/As_Reinações_de_Narizinho/Narizinho_Arrebitado"
URL_INDICE = "https://pt.wikisource.org/wiki/As_Reinações_de_Narizinho"

CAPITULOS_ESPERADOS = [
    "Narizinho_Arrebitado", "O_Sitio_do_Picapau_amarello", "O_Marquez_de_Rabicó",
    "O_Casamento_de_Narizinho", "Aventuras_do_Principe", "O_Gato_Felix",
    "Cara_de_Coruja", "O_Irmão_de_Pinocchio", "O_Circo_de_Escavallinho",
    "A_Penna_de_Papagaio", "O_Pó_de_Pirlimpimpim",
]


@pytest.fixture
def capitulo():
    return wk.extrair(CAPITULO_HTML.read_text(encoding="utf-8"), URL_CAPITULO)


# --- metadados estruturados (#ws-data) ----------------------------------------

def test_extrai_titulo_obra_autor_ano_e_id(capitulo):
    assert capitulo.titulo == "Narizinho Arrebitado"
    assert capitulo.obra == "As Reinações de Narizinho"
    assert capitulo.autor == "Monteiro Lobato"
    assert capitulo.ano_publicacao == 1931
    assert capitulo.artigo_id == "225930"


def test_slug_e_titulo_video(capitulo):
    assert capitulo.slug == "narizinho-arrebitado"
    assert capitulo.titulo_video == "Narizinho Arrebitado — As Reinações de Narizinho"


# --- limpeza do aparato editorial ----------------------------------------------

def test_aparato_editorial_nao_sobrevive_no_texto(capitulo):
    """Tudo que mora em `.ws-noexport` (nav anterior/próximo, linha de edição,
    selo de domínio público, o próprio bloco #ws-data) precisa sumir."""
    texto = capitulo.texto()
    for marca in ("225930", "edição de referência", "domínio público",
                  "Companhia Editora Nacional", "ws-noexport", "ws-data"):
        assert marca not in texto, f"{marca!r} vazou para o texto final"


def test_sem_residuo_de_html(capitulo):
    texto = capitulo.texto()
    for marca in ("<", ">", "&nbsp;", "&quot;"):
        assert marca not in texto


def test_pagenum_vazio_no_meio_de_frase_nao_quebra_paragrafo():
    """Reproduz o padrão real (`<span class="pagenum ws-pagenum" ...></span>`
    vazio, inserido pelo ProofreadPage no meio do texto) com um HTML sintético
    pequeno — a fixture real não tem nenhuma ocorrência no meio de palavra por
    sorte, então testar só contra ela deixaria essa regressão passar batido."""
    html = f"""
    <div id="mw-content-text"><div class="mw-parser-output">
    <div id="ws-data" class="ws-noexport">
      <span id="ws-article-id">1</span>
      <span id="ws-title"><a href="{URL_INDICE}">Obra Teste</a>
        <a class="mw-selflink" href="{URL_CAPITULO}2">Capítulo Teste</a></span>
      <span id="ws-author">Autor Teste</span>
      <span id="ws-year">1900</span>
    </div>
    <p>Primeira parte da frase<span class="pagenum ws-pagenum" id="9"></span> continua sem quebra.</p>
    </div></div>
    """
    c = wk.extrair(html, URL_CAPITULO + "2")
    assert c.paragrafos == ["Primeira parte da frase continua sem quebra."]


def test_pagina_sem_ws_data_e_erro_claro():
    """Sem `#ws-data` não há como saber título/obra/autor — em vez de devolver
    um Capitulo com campos vazios (que o resto do pipeline aceitaria calado),
    é erro explícito. Mesma filosofia do "sem marcador -> ValueError" do
    Acesso ao Insight, para o caso de página inexistente/layout inesperado."""
    html = '<div id="mw-content-text"><div class="mw-parser-output"><p>Sem dados.</p></div></div>'
    with pytest.raises(ValueError, match="ws-data"):
        wk.extrair(html, URL_CAPITULO)


def test_corpo_vazio_depois_da_limpeza_e_erro():
    html = f"""
    <div id="mw-content-text"><div class="mw-parser-output">
    <div id="ws-data" class="ws-noexport">
      <span id="ws-article-id">1</span>
      <span id="ws-title"><a href="{URL_INDICE}">Obra</a>
        <a class="mw-selflink" href="{URL_CAPITULO}">Cap</a></span>
      <span id="ws-author">Autor</span><span id="ws-year">1900</span>
    </div>
    </div></div>
    """
    with pytest.raises(ValueError, match="vazio"):
        wk.extrair(html, URL_CAPITULO)


# --- garantia estrutural: um capítulo do Wikisource é UM Chapter --------------

def test_pagina_de_capitulo_nunca_fragmenta_em_varios_capitulos(capitulo):
    """A prosa de origem tem marcadores internos em romano ("I — NARIZINHO"
    etc.) que o detector genérico de capítulos casaria como títulos — mas
    aqui `montar_script` recebe `capitulos=[(c.titulo, c.texto())]` explícito
    (ver `project.montar_script`), então isto nunca chega a `detectar_capitulos`
    de verdade em produção. Este teste documenta POR QUE isso é necessário:
    passado sem a lista forçada, o texto fragmenta."""
    caps = detectar_capitulos(capitulo.texto())
    assert len(caps) > 1, (
        "se isto passar a dar 1, o forçar capitulos= em cli/main.py pode ter "
        "virado desnecessário — mas não remova sem re-confirmar")


# --- descoberta de capítulos a partir do índice -------------------------------

def test_descobre_os_11_capitulos_na_ordem(monkeypatch):
    pagina = INDICE_HTML.read_text(encoding="utf-8")
    monkeypatch.setattr(wk, "baixar", lambda url, **kw: pagina)
    urls = wk.descobrir_capitulos(URL_INDICE)
    assert len(urls) == 11
    assert len(set(urls)) == 11  # sem duplicatas
    esperados = [f"{URL_INDICE}/{nome}" for nome in CAPITULOS_ESPERADOS]
    assert urls == esperados


def test_descobrir_capitulos_exclui_links_de_pagina_de_scan(monkeypatch):
    pagina = INDICE_HTML.read_text(encoding="utf-8")
    monkeypatch.setattr(wk, "baixar", lambda url, **kw: pagina)
    urls = wk.descobrir_capitulos(URL_INDICE)
    assert not any("Página:" in u for u in urls)


def test_pagina_de_capitulo_nao_tem_capitulos_filhos(monkeypatch):
    """`descobrir_capitulos` numa página de CAPÍTULO (não índice) devolve
    lista vazia -- é assim que o comando decide "isto já é o capítulo"."""
    pagina = CAPITULO_HTML.read_text(encoding="utf-8")
    monkeypatch.setattr(wk, "baixar", lambda url, **kw: pagina)
    assert wk.descobrir_capitulos(URL_CAPITULO) == []


# --- rights --------------------------------------------------------------------

def test_rights_e_dominio_publico_nao_licenciado(capitulo):
    """Diferente do Acesso ao Insight (`licenciado` -- a tradução é obra
    derivada): aqui o texto ORIGINAL é de domínio público no Brasil."""
    r = capitulo.rights()
    assert r["status"] == "dominio-publico"
    assert r["autor"] == "Monteiro Lobato"
    assert r["ano_morte"] == 1948
    assert r["tradutor"] is None
    assert r["verificado_em"] is None
    assert "Wikisource" in r["fonte"]


# --- normalização de URL ---------------------------------------------------------

@pytest.mark.parametrize("entrada", [
    URL_CAPITULO,
    "wiki/As_Reinações_de_Narizinho/Narizinho_Arrebitado",
    "/wiki/As_Reinações_de_Narizinho/Narizinho_Arrebitado",
    "As_Reinações_de_Narizinho/Narizinho_Arrebitado",
])
def test_normalizar_url_aceita_varias_formas(entrada):
    assert wk.normalizar_url(entrada) == URL_CAPITULO


# --- codificação da URL para a requisição HTTP --------------------------------

def test_url_com_unicode_cru_vira_percent_encoded_para_a_requisicao():
    """`descobrir_capitulos` extrai `href` do HTML em unicode cru (é assim que
    o MediaWiki os gera) -- `http.client` exige ASCII no request line, e sem
    codificar isso derruba com `UnicodeEncodeError` (bug real, medido rodando
    contra o site de verdade)."""
    codificada = wk._url_para_requisicao(URL_CAPITULO)
    codificada.encode("ascii")  # não deve levantar
    assert "%C3%A7" in codificada  # o "ç" de "Reinações"


def test_url_ja_percent_encoded_nao_fica_com_percent_duplo():
    ja_codificada = "https://pt.wikisource.org/wiki/As_Reina%C3%A7%C3%B5es_de_Narizinho"
    assert wk._url_para_requisicao(ja_codificada) == ja_codificada
