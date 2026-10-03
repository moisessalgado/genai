"""Geração de imagem (FLUX via InvokeAI): o que dá para verificar sem a GPU.

A geração em si é do InvokeAI (`test_invokeai.py` cobre o cliente). O que é
testável aqui é tudo o que fica *ao redor* dela: resolução de modelo,
defaults, determinismo da seed, o atalho de cache e a conversão para o
formato do acervo. Mesma filosofia de `test_musica_ace.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.video import imagem as img


# --- resolução de modelo ------------------------------------------------------

def test_resolve_flux_para_o_modelo_do_invokeai():
    assert img.resolver_modelo("flux") == "FLUX.1 schnell (quantized)"


def test_modelo_desconhecido_e_recusado():
    with pytest.raises(ValueError, match="modelo desconhecido"):
        img.resolver_modelo("sd")


# --- defaults -----------------------------------------------------------------

def test_defaults_flux_sao_poucos_passos_sem_guidance():
    """FLUX-schnell é destilado: guidance != 0 não faz CFG nenhum, só custa tempo."""
    assert img._defaults(None, None) == (4, 0.0)


def test_defaults_explicitos_sobrepoem():
    assert img._defaults(10, 3.5) == (10, 3.5)


# --- seed ----------------------------------------------------------------------

def test_seed_e_deterministica_e_distinta():
    """Mesmo prompt, mesma imagem: um candidato aprovado tem de poder ser
    reproduzido se o arquivo se perder, sem guardar a seed em outro lugar."""
    assert img._seed("paisagem", 0) == img._seed("paisagem", 0)
    assert img._seed("paisagem", 0) != img._seed("paisagem", 1)
    assert img._seed("paisagem", 0) != img._seed("retrato", 0)


# --- gerar(): atalho de cache --------------------------------------------------

def test_gerar_pula_destinos_que_ja_existem(tmp_path, monkeypatch):
    monkeypatch.setattr(img, "CACHE", tmp_path)
    monkeypatch.setattr(img, "disponivel", lambda: True)
    prompt = "paisagem oriental"
    marca = __import__("hashlib").sha256(prompt.encode()).hexdigest()[:8]
    for i in range(2):
        (tmp_path / f"flux-{marca}-{img._seed(prompt, i)}.png").write_bytes(b"x")

    with patch.object(img, "gerar_lote") as lote:
        destinos = img.gerar(prompt, modelo="flux", n=2)
        lote.assert_not_called()
    assert len(destinos) == 2
    assert all(d.exists() for d in destinos)


def test_gerar_pede_ao_lote_so_os_pendentes(tmp_path, monkeypatch):
    monkeypatch.setattr(img, "CACHE", tmp_path)
    monkeypatch.setattr(img, "disponivel", lambda: True)
    prompt = "retrato histórico"
    marca = __import__("hashlib").sha256(prompt.encode()).hexdigest()[:8]
    (tmp_path / f"flux-{marca}-{img._seed(prompt, 0)}.png").write_bytes(b"x")

    with patch.object(img, "gerar_lote") as lote:
        img.gerar(prompt, modelo="flux", n=3)
        (pendentes,), _ = lote.call_args
    assert [p["seed"] for p in pendentes] == [img._seed(prompt, 1), img._seed(prompt, 2)]
    assert all((p["passos"], p["guidance"]) == (4, 0.0) for p in pendentes)
    assert all((p["largura"], p["altura"]) == (1344, 768) for p in pendentes)


def test_gerar_sem_invokeai_e_recusado(monkeypatch):
    monkeypatch.setattr(img, "disponivel", lambda: False)
    with pytest.raises(RuntimeError, match="InvokeAI fora do ar"):
        img.gerar("qualquer coisa")


class _ClienteFalso:
    def __init__(self):
        self.pedidos = []

    def grafo_flux(self, prompt, largura, altura, seed, passos, guidance):
        return ({"prompt": prompt, "seed": seed}, [])

    def gerar_lote(self, grafos, progresso=None):
        for g, destino in grafos:
            self.pedidos.append(g[0])
            _png(destino)


def test_gerar_lote_gera_so_o_que_falta_e_pontua_tudo(tmp_path, monkeypatch):
    falso = _ClienteFalso()
    monkeypatch.setattr(img, "disponivel", lambda: True)
    monkeypatch.setattr(img.invokeai, "cliente", lambda: falso)
    chamados = {}

    def pontuar(itens, relevancia, estilo):
        chamados["itens"] = itens
        return {str(c): 0.3 for c, _ in itens}, {str(c): -0.1 for c, _ in itens}

    monkeypatch.setattr(img.imagem_qa, "pontuar", pontuar)
    existe = _png(tmp_path / "a.png")
    pedidos = [{"prompt": p, "seed": s, "destino": str(d), "largura": 64, "altura": 64,
                "passos": 4, "guidance": 0.0}
               for p, s, d in (("um", 1, existe), ("dois", 2, tmp_path / "b.png"))]

    notas, estilos = img.gerar_lote(pedidos, avaliar_clip=True, avaliar_estilo=True)
    assert falso.pedidos == [{"prompt": "dois", "seed": 2}]
    assert [c for c, _ in chamados["itens"]] == [existe, tmp_path / "b.png"]
    assert notas[str(existe)] == 0.3 and estilos[str(tmp_path / "b.png")] == -0.1


def test_gerar_lote_sem_avaliacao_nao_carrega_o_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(img, "disponivel", lambda: True)
    monkeypatch.setattr(img.invokeai, "cliente", lambda: _ClienteFalso())
    monkeypatch.setattr(img.imagem_qa, "pontuar",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("CLIP")))
    pedidos = [{"prompt": "x", "seed": 1, "destino": str(tmp_path / "x.png"), "largura": 64,
                "altura": 64, "passos": 4, "guidance": 0.0}]
    assert img.gerar_lote(pedidos) == ({}, {})


# --- aprovar(): conversão para o acervo ----------------------------------------

def _png(caminho: Path) -> Path:
    from PIL import Image
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(caminho)
    return caminho


def test_aprovar_converte_para_jpeg_no_acervo_e_limpa_o_cache(tmp_path):
    cache = tmp_path / "cache"
    acervo = tmp_path / "acervo"
    cache.mkdir()
    origem = _png(cache / "flux-abc123-42.png")

    finais = img.aprovar([origem], slides_dir=acervo)

    assert len(finais) == 1
    assert finais[0] == acervo / "flux-abc123-42.jpg"
    assert finais[0].exists()
    assert not origem.exists()  # rascunho descartado depois de virar acervo


def test_aprovar_recusa_arquivo_inexistente_sem_converter_nada(tmp_path):
    cache = tmp_path / "cache"
    acervo = tmp_path / "acervo"
    cache.mkdir()
    ok = _png(cache / "flux-ok-1.png")
    falta = cache / "flux-falta-2.png"

    with pytest.raises(FileNotFoundError, match="flux-falta-2.png"):
        img.aprovar([ok, falta], slides_dir=acervo)

    assert ok.exists()  # nada foi convertido nem apagado
    assert not (acervo / "flux-ok-1.jpg").exists()
