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


# Ancoras fixas para o QA de estilo: a diferenca de similaridade CLIP entre a
# imagem e cada uma diz se ela pendeu para foto real em vez de desenho — sem
# olho humano, e o unico sinal barato que temos de que o FLUX ignorou o pedido
# de estilo "storybook illustration" (medido: acontece sobretudo em prompts
# com pessoas, onde o modelo tende a foto-realismo mesmo com o estilo pedido).
ANCORA_FOTO = "a realistic photograph of a real person"
ANCORA_ILUSTRACAO = "a hand-drawn cartoon illustration from a children's picture book, flat colors"

# CLIP pequeno, na CPU do processo principal (o `transformers` ja vem com o
# chatterbox). Morava no runner da antiga `.venv-imagem`; medido em 18 imagens
# reais do Narizinho, as notas dos dois ambientes diferem em < 3e-4 e todas as
# decisoes foto/ilustracao coincidem — o LIMIAR_FOTO nao precisou de ajuste.
MODELO_CLIP = "openai/clip-vit-base-patch32"
_clip = None


def eh_fotorealista(pontuacao: float, limiar: float = LIMIAR_FOTO) -> bool:
    """`pontuacao` e a diferenca foto-ilustracao calculada por `pontuar`."""
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


def _carregar_clip():
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        from transformers.utils import logging as tlog

        # No runner antigo isto ia para um stdout capturado; aqui sujaria o
        # `console.status` da CLI com a barra de carga e o relatorio de pesos.
        tlog.set_verbosity_error()
        tlog.disable_progress_bar()
        modelo = CLIPModel.from_pretrained(MODELO_CLIP)
        modelo.eval()
        _clip = (modelo, CLIPProcessor.from_pretrained(MODELO_CLIP))
    return _clip


def pontuar(itens: list[tuple[Path, str]], *, relevancia: bool, estilo: bool
            ) -> tuple[dict[str, float], dict[str, float]]:
    """Notas CLIP de cada (imagem, prompt) que existe em disco: relevancia ao
    prompt e diferenca foto-menos-ilustracao. Chaveadas pelo caminho em texto.

    Um carregamento do CLIP para as duas notas. `relevancia` so serve para
    ESCOLHER a melhor entre candidatos da mesma janela — quem aprova ou
    reprova e `nao_e_vazia`/`eh_fotorealista`.

    A forward completa do modelo (`modelo(**entradas)`), e nao
    `get_text_features`/`get_image_features`: esses chegaram a devolver o
    output cru do encoder, sem a projecao, numa versao de transformers.
    """
    if not (relevancia or estilo):
        return {}, {}
    import torch

    modelo, processador = _carregar_clip()
    notas_relevancia: dict[str, float] = {}
    notas_estilo: dict[str, float] = {}
    with torch.no_grad():
        for caminho, prompt in itens:
            if not caminho.exists():
                continue
            with Image.open(caminho) as im:
                imagem = im.convert("RGB")
            textos = [prompt[:300]] if relevancia else []
            offset_estilo = len(textos)
            if estilo:
                textos += [ANCORA_FOTO, ANCORA_ILUSTRACAO]
            entradas = processador(text=textos, images=[imagem], return_tensors="pt",
                                   padding=True, truncation=True)
            saida = modelo(**entradas)
            sims = torch.nn.functional.cosine_similarity(
                saida.image_embeds, saida.text_embeds).tolist()
            if relevancia:
                notas_relevancia[str(caminho)] = sims[0]
            if estilo:
                notas_estilo[str(caminho)] = sims[offset_estilo] - sims[offset_estilo + 1]
    return notas_relevancia, notas_estilo
