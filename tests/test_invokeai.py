"""Cliente do InvokeAI, sem servidor: um _req falso registra o que foi pedido."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audiofactory.servicos import invokeai  # noqa: E402

MODELOS = [{"key": f"k-{n}", "hash": "h", "name": n, "base": "b", "type": "t", "extra": 1}
           for n in (invokeai.FLUX_SCHNELL, invokeai.FLUX_VAE, invokeai.FLUX_T5,
                     invokeai.FLUX_CLIP)]


class Falso(invokeai.InvokeAI):
    """Fila que completa cada item na N-ésima consulta, e registra a ordem."""

    def __init__(self, falhar: set[int] = frozenset()):
        super().__init__(base="http://falso", intervalo_s=0)
        self.log: list[tuple] = []
        self.em_fila: set[int] = set()
        self.max_em_fila = 0
        self.proximo = 1
        self.falhar = falhar

    def _req(self, method, path, body=None, headers=None):
        self.log.append((method, path))
        if path == "/api/v2/models/":
            return {"models": MODELOS}
        if path.endswith("enqueue_batch"):
            item = self.proximo
            self.proximo += 1
            self.em_fila.add(item)
            self.max_em_fila = max(self.max_em_fila, len(self.em_fila))
            return {"item_ids": [item]}
        if path.startswith("/api/v1/queue/default/i/"):
            item = int(path.rsplit("/", 1)[1])
            self.em_fila.discard(item)
            if item in self.falhar:
                return {"status": "failed", "error_traceback": "OOM"}
            return {"status": "completed", "session": {"results": {
                "dec": {"type": "image_output", "image": {"image_name": f"img-{item}.png"}}}}}
        if path.endswith("/full"):
            return b"PNG"
        if method == "DELETE":
            return None
        raise AssertionError(path)


def test_grafo_flux_usa_os_modelos_registrados_e_os_padroes_do_schnell():
    c = Falso()
    nodes, edges = c.grafo_flux("um jabuti", 1344, 768, 7)
    assert nodes["loader"]["model"] == {"key": f"k-{invokeai.FLUX_SCHNELL}", "hash": "h",
                                        "name": invokeai.FLUX_SCHNELL, "base": "b", "type": "t"}
    assert nodes["dn"].items() >= {"num_steps": 4, "guidance": 0.0, "seed": 7,
                                   "width": 1344, "height": 768}.items()
    assert nodes["te"]["prompt"] == "um jabuti"
    assert {(e["source"]["node_id"], e["destination"]["node_id"]) for e in edges} >= {
        ("loader", "te"), ("te", "dn"), ("dn", "dec")}


def test_modelo_ausente_diz_quais_existem():
    with pytest.raises(invokeai.InvokeAIErro, match="não instalado.*FLUX.1 schnell"):
        Falso().modelo("SD3.5")


def test_gerar_lote_respeita_a_janela_baixa_na_ordem_e_apaga(tmp_path):
    c = Falso()
    gs = [(c.grafo_flux(f"p{i}", 64, 64, i), tmp_path / f"{i}.png") for i in range(7)]
    feitos = c.gerar_lote(gs, em_voo=3)
    assert feitos == [d for _, d in gs]
    assert all(d.read_bytes() == b"PNG" for d in feitos)
    assert c.max_em_fila == 3
    apagados = [p for m, p in c.log if m == "DELETE"]
    assert apagados == [f"/api/v1/images/i/img-{i}.png" for i in range(1, 8)]
    assert not list(tmp_path.glob("*.parcial"))


def test_manter_no_invokeai_nao_apaga(tmp_path):
    c = Falso()
    c.gerar_lote([(c.grafo_flux("p", 64, 64, 1), tmp_path / "a.png")], manter_no_invokeai=True)
    assert not any(m == "DELETE" for m, _ in c.log)


def test_item_falho_vira_erro_com_o_traceback(tmp_path):
    c = Falso(falhar={1})
    with pytest.raises(invokeai.InvokeAIErro, match="item 1 failed: OOM"):
        c.gerar_lote([(c.grafo_flux("p", 64, 64, 1), tmp_path / "a.png")])
    assert not (tmp_path / "a.png").exists()


def test_servidor_fora_do_ar_e_erro_claro():
    c = invokeai.InvokeAI(base="http://127.0.0.1:9", timeout_s=1)
    assert not c.disponivel()
    with pytest.raises(invokeai.InvokeAIErro, match="fora do ar"):
        c.versao()


def test_liberar_vram_esvazia_o_cache_de_modelos():
    c = Falso()
    c._req = lambda m, p, *a, **k: c.log.append((m, p))
    c.liberar_vram()
    assert c.log == [("POST", "/api/v2/models/empty_model_cache")]


def test_liberar_vram_com_servico_fora_do_ar_nao_e_erro(monkeypatch):
    monkeypatch.setattr(invokeai, "cliente",
                        lambda: invokeai.InvokeAI(base="http://127.0.0.1:9", timeout_s=1))
    assert invokeai.liberar_vram_se_no_ar() is False
