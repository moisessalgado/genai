"""Metadados do upload para o YouTube: o que dá para verificar sem subir vídeo.

A chamada de rede (`videos.insert`) não é testável aqui -- exige credenciais OAuth
reais. O que importa testar é o que vira metadado visível no canal: título,
descrição (atribuição + capítulos + disclosure) e as flags de status, porque um
erro aí só aparece depois de publicado, quando já é tarde para corrigir sem
editar manualmente no Studio.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.publish.youtube import (DISCLOSURE, descricao, formatar_keywords,
                                          mesclar_branding, metadados)


def _cfg(rights: dict) -> dict:
    return {"titulo": "Discurso de Dissolução", "rights": rights}


def test_descricao_credita_dominio_publico_e_fonte(tmp_path):
    cfg = _cfg({"status": "dominio-publico", "autor": "Cânone Páli",
               "fonte": "SN 56.11"})
    texto = descricao(cfg, tmp_path)
    assert "Domínio público — Cânone Páli" in texto
    assert "Fonte: SN 56.11" in texto


def test_descricao_sempre_traz_o_disclosure_de_sintese(tmp_path):
    """Toda narração deste canal sai de TTS -- não é opcional por projeto."""
    cfg = _cfg({"status": "proprio", "autor": "Moises"})
    assert DISCLOSURE in descricao(cfg, tmp_path)


def test_descricao_inclui_timestamps_quando_ha_mais_de_um_capitulo(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "chapters.txt").write_text(
        "00:00 Introdução\n12:30 Capítulo 1\n", encoding="utf-8")
    texto = descricao(_cfg({}), tmp_path)
    assert "12:30 Capítulo 1" in texto


def test_descricao_omite_capitulo_unico_generico(tmp_path):
    """Um projeto de capítulo só tem chapters.txt = 'Texto completo': não soma
    nada à descrição, e incluir seria ruído."""
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "chapters.txt").write_text("00:00 Texto completo\n",
                                                       encoding="utf-8")
    texto = descricao(_cfg({}), tmp_path)
    assert "Texto completo" not in texto


def test_metadados_privacidade_padrao_e_titulo_truncado(tmp_path):
    cfg = _cfg({}) | {"titulo": "T" * 150}
    corpo = metadados(cfg, tmp_path, privacidade="private")
    assert corpo["status"]["privacyStatus"] == "private"
    assert len(corpo["snippet"]["title"]) == 100


def test_metadados_declara_conteudo_sintetico_sempre(tmp_path):
    corpo = metadados(_cfg({}), tmp_path)
    assert corpo["status"]["containsSyntheticMedia"] is True
    assert corpo["status"]["selfDeclaredMadeForKids"] is False


def test_descricao_reproduz_a_licenca_do_texto(tmp_path):
    """Texto de terceiro sob licença de distribuição gratuita costuma exigir que
    os termos viajem com a cópia. Num vídeo, o lugar disso é a descrição."""
    cfg = _cfg({"status": "licenciado", "tradutor": "Fulano",
                "licenca": "Somente para distribuição gratuita."})
    texto = descricao(cfg, tmp_path)
    assert "Licença do texto: Somente para distribuição gratuita." in texto
    assert "Tradução: Fulano" in texto


def test_descricao_sem_licenca_declarada_nao_inventa_linha(tmp_path):
    assert "Licença do texto" not in descricao(_cfg({"status": "proprio"}), tmp_path)


def _canal(**kwargs) -> dict:
    base = {"id": "UCxyz", "snippet": {"title": "Suttas em Português",
                                       "description": "antiga"},
           "brandingSettings": {"channel": {"keywords": "antigas"}}}
    base.update(kwargs)
    return base


def test_mesclar_branding_troca_so_o_campo_pedido():
    """`channels.update` substitui o recurso inteiro -- sem partir do estado
    atual, atualizar só a descrição apagaria as keywords existentes."""
    corpo = mesclar_branding(_canal(), descricao="nova descrição")
    assert corpo["snippet"]["description"] == "nova descrição"
    assert corpo["brandingSettings"]["channel"]["keywords"] == "antigas"


def test_mesclar_branding_preserva_titulo_do_snippet():
    corpo = mesclar_branding(_canal(), keywords="budismo dhamma")
    assert corpo["snippet"]["title"] == "Suttas em Português"
    assert corpo["brandingSettings"]["channel"]["keywords"] == "budismo dhamma"


def test_mesclar_branding_sem_argumentos_nao_muda_nada():
    canal = _canal()
    corpo = mesclar_branding(canal)
    assert corpo["snippet"]["description"] == "antiga"
    assert corpo["brandingSettings"]["channel"]["keywords"] == "antigas"


def test_mesclar_branding_banner_url_entra_em_image():
    corpo = mesclar_branding(_canal(), banner_url="https://yt3.googleusercontent.com/x")
    assert corpo["brandingSettings"]["image"]["bannerExternalUrl"] == \
        "https://yt3.googleusercontent.com/x"


def test_formatar_keywords_aspeia_termos_de_mais_de_uma_palavra():
    """O campo é texto corrido: sem aspas, "cânone páli" viraria duas keywords
    separadas em vez de uma."""
    texto = formatar_keywords(["budismo", "cânone páli", "dhamma"])
    assert texto == 'budismo "cânone páli" dhamma'
