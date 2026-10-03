"""InvokeAI falso para os testes da HQ: grava PNGs de verdade nos destinos."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


class InvokeFalso:
    def __init__(self, cor=(200, 120, 40)):
        self.cor = cor
        self.grafos: list[tuple] = []
        self.uploads: list[Path] = []
        self.apagados: list[str] = []

    def disponivel(self):
        return True

    def grafo_flux(self, prompt, largura, altura, seed, **kw):
        return ("flux", prompt, largura, altura, seed)

    def grafo_qwen_edit(self, prompt, refs, largura, altura, seed, **kw):
        return ("qwen", prompt, tuple(refs), largura, altura, seed, tuple(kw.get("loras", ())))

    def upload(self, caminho, board_id=None):
        self.uploads.append(Path(caminho))
        return f"up-{len(self.uploads)}.png"

    def apagar(self, nome):
        self.apagados.append(nome)

    def gerar_lote(self, pedidos, progresso=None, **kw):
        for g, destino in pedidos:
            self.grafos.append(g)
            destino.parent.mkdir(parents=True, exist_ok=True)
            tam = (g[3], g[4]) if g[0] == "qwen" else (g[2], g[3])
            Image.new("RGB", tam, self.cor).save(destino)
        return [d for _, d in pedidos]
