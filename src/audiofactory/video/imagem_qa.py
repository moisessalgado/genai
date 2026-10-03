"""QA automatica de imagem, sem olho humano (TDD 18 — slideshow sincronizado).

Espelha o desenho de `qa/verify.py`: checagens baratas e numericas no lugar da
curadoria manual que `imagem.aprovar()` sempre pediu. Necessario aqui porque o
slideshow sincronizado gera uma imagem por janela de ~10s (centenas por
capitulo) e o operador decidiu explicitamente pular a revisao humana — mas
"sem revisao humana" nao pode virar "sem revisao nenhuma": FLUX-schnell as
vezes devolve quadro quase solido (colapso do modelo num prompt ruim), e isso
tem de ser pego antes de entrar no video.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Desvio padrao minimo (0-255, luminancia) para uma imagem contar como "tem
# conteudo". Medido contra um recorte solido de proposito: uma imagem quase
# uniforme fica bem abaixo de 8; arte normal, mesmo minimalista, passa de 20.
DESVIO_MINIMO = 8.0

# Diferenca (sim. CLIP a uma ancora "foto real" menos sim. a uma ancora
# "ilustracao infantil") acima da qual a imagem conta como fotorrealista
# demais para o video. 0.0 = neutro; medido contra alguns pares foto/desenho
# de teste, imagens claramente desenhadas ficam bem abaixo de 0, fotos reais
# bem acima.
LIMIAR_FOTO = 0.0


def eh_fotorealista(pontuacao: float, limiar: float = LIMIAR_FOTO) -> bool:
    """`pontuacao` e a diferenca foto-ilustracao calculada por
    `_imagem_runner._pontuar` (ver `avaliar_estilo`)."""
    return pontuacao > limiar


def nao_e_vazia(caminho: Path, desvio_minimo: float = DESVIO_MINIMO) -> bool:
    """Rejeita quadro quase solido — o modo de falha mais comum sem curadoria.

    FLUX-schnell colapsando num prompt ruim tende a devolver um campo de cor
    quase uniforme (bug conhecido do passo-unico destilado), nao ruido — por
    isso desvio padrao de luminancia, nao entropia, e o suficiente e barato.
    """
    with Image.open(caminho) as img:
        cinza = np.asarray(img.convert("L"), dtype=np.float32)
    return float(cinza.std()) >= desvio_minimo


def escolher_melhor(candidatos: list[Path], pontuacoes: dict[Path, float] | None = None,
                    estilos: dict[Path, float] | None = None) -> Path | None:
    """Primeiro candidato que passa em `nao_e_vazia` e (se `estilos` for dado)
    em `eh_fotorealista`, ordenado por pontuacao de relevancia (se houver) do
    maior para o menor. `None` se nenhum candidato prestar — quem chama decide
    o fallback (repetir com outro prompt, pular a janela)."""
    validos = [c for c in candidatos if c.exists() and nao_e_vazia(c)
              and not (estilos is not None and eh_fotorealista(estilos.get(c, -1.0)))]
    if not validos:
        return None
    if pontuacoes:
        validos.sort(key=lambda c: pontuacoes.get(c, 0.0), reverse=True)
    return validos[0]
