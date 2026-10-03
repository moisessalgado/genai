"""Slideshow sincronizado ao conteudo: uma imagem nova a cada ~10s, gerada a
partir do trecho da narracao que esta tocando naquele momento (TDD 18 — preset
"sincronizado").

Diferenca central para o preset `slides`: aquele sorteia de um acervo
compartilhado sem relacao com o audio; este gera um acervo PROPRIO por
capitulo, amarrado ao tempo, e sem curadoria humana — decisao explicita do
operador (volume grande demais para revisar candidato a candidato). Por isso
duas pecas que `imagem.py` sempre deixou para o olho humano viram codigo aqui:
`imagem_qa.py` (nao aceitar quadro quase solido) e a nota CLIP (escolher a
melhor entre candidatos da mesma janela).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

from . import imagem as imagem_mod
from . import imagem_qa
from . import slides as slides_mod
from ..core.config import settings
from ..audiolivro.narration.llm import OLLAMA_URL

MODELO_OLLAMA = settings().llm_modelo

# ~10s por janela: pequeno o bastante para acompanhar a historia, grande o
# bastante para uma janela sempre conter pelo menos uma cue inteira de legenda
# (MIN_CUE_S de legenda.py e 0.35s — nunca vai faltar texto para descrever).
JANELA_S = 10.0


@dataclass
class Janela:
    inicio_s: float
    fim_s: float
    texto: str


# Ficha fixa de personagem — mesma descricao usada nas imagens aprovadas do
# acervo da Narizinho, reaproveitada aqui para o LLM manter consistencia
# visual entre janelas sem depender de memoria de imagem a imagem (FLUX nao
# tem isso; o que da consistencia e repetir o MESMO texto descritivo sempre
# que o personagem aparece). Personagem original nosso, nao o design de
# nenhuma midia de terceiros.
FICHA_MENINA = (
    "a young Brazilian girl around 8 years old with a small upturned nose, "
    "two brown braids tied with red ribbons, wearing a faded blue cotton "
    "pinafore dress, barefoot, early 1900s rural Brazil"
)
FICHA_BONECA = (
    "a small handmade rag doll with a round stitched face, black button "
    "eyes, a red stitched smile, yellow yarn pigtails, patched cotton dress"
)

# Contexto fixo do livro, enviado em TODO prompt — o LLM nao precisa mais
# adivinhar epoca/cenario a partir da grafia arcaica do trecho, so encaixar o
# trecho dentro de um cenario que ja foi dado pronto. Ideia do operador: o
# gemma4:12b se perdia tentando "traduzir" cada palavra em ortografia de 1920
# (medido: `num_predict` estourava sem nunca emitir resposta); pedir so o
# GIST da cena, com o contexto do livro ja resolvido, e uma tarefa mais curta
# e mais barata de raciocinar.
CONTEXTO_LIVRO = (
    "This is a passage from a 1920s Brazilian children's book. Setting: a "
    "small farm (\"sítio\") in the Brazilian countryside, early 1900s. Main "
    "character: a young girl who lives there with her grandmother, alongside "
    "a talking rag doll and other farm characters."
)

PROMPT_CENA = """{contexto}

Below is one passage from the book, in archaic 1920s Portuguese spelling —
don't try to translate it word for word, just get the GIST of what is
physically happening (who/what, doing what, where):

"{texto}"

Write ONE line in ENGLISH describing a single scene that matches that gist,
as a children's picture-book illustration: flat 2D cartoon art, hand-painted
gouache and crayon texture, thick black outlines — never photorealistic,
never a photo, never a realistic render of a person's face.

If the girl protagonist appears in the scene, describe her exactly as:
{ficha_menina}
If her rag doll appears, describe it exactly as:
{ficha_boneca}
Otherwise just describe the setting/other characters, consistent with the
book's setting above.

Rules:
- output ONLY the scene description, nothing else, no quotes, no explanation;
- one line, under 60 words;
- never mention real people, brands, or characters from other media.

Scene description:"""

# Estilo reforcado nas DUAS pontas do prompt (prefixo + sufixo): medido no
# render anterior que o sufixo sozinho ("storybook illustration ... not
# photorealistic") nao bastava — FLUX-schnell (4 passos, sem CFG de verdade)
# ainda saia fotorrealista sobretudo em cenas com pessoas (a menina, a avo).
# So dizer "not X" tem pouco efeito num modelo assim; reforcar com varias
# palavras POSITIVAS de estilo (cartoon, hand-drawn, thick outlines) e repeti-
# -las no comeco E no fim do prompt e o que de fato muda a saida, complementado
# pelo QA automatico de estilo em `imagem_qa.eh_fotorealista` (que descarta o
# candidato se mesmo assim sair parecendo foto).
ESTILO_PREFIXO = ("Children's picture-book illustration, flat 2D cartoon art style, "
                  "hand-painted gouache and crayon texture, thick black outlines:")
_ESTILO = ("children's picture-book illustration, flat cartoon art, hand-painted "
          "gouache and crayon texture, thick outlines, bold flat colors, whimsical "
          "children's book style, not a photo, not photorealistic, not 3D render")


def _com_estilo(cena: str) -> str:
    return f"{ESTILO_PREFIXO} {cena}, {_ESTILO}"

# Cenarios genericos de reserva, para quando o Ollama nao responder a tempo —
# medido: para certos trechos (ortografia arcaica do livro confunde o
# gemma4:12b, que entra num "raciocinio" interno e nunca emite a resposta),
# nem 1200 tokens de orcamento bastam, e insistir custaria ~25s por janela a
# troco de nada. Uma lista pequena, escolhida por indice, evita que toda janela
# de reserva vire a MESMA imagem repetida — o problema que este preset existe
# para resolver.
CENARIOS_RESERVA = [
    _com_estilo("a sunlit Brazilian farmyard with mango trees"),
    _com_estilo("a rustic farmhouse veranda at golden hour"),
    _com_estilo("a lush backyard vegetable garden with pumpkins and banana trees"),
    _com_estilo("a rustic Brazilian farmhouse kitchen with a wood stove"),
    _com_estilo("a moonlit Brazilian garden with wildflowers"),
    _com_estilo("a dirt path through a tropical orchard"),
]

# Palavras que indicam que a janela fala da menina ou da boneca — se baterem,
# o cenario de reserva leva a ficha fixa do personagem tambem, em vez de so a
# cena vazia. Lista curta e literal de proposito: e so para o caso de RESERVA,
# o LLM (que entende contexto de verdade) e sempre a primeira tentativa.
_PALAVRAS_MENINA = ("narizinho", "menina", "netta")
_PALAVRAS_BONECA = ("boneca", "pano")


def _fallback(janela: Janela, indice: int) -> str:
    base = CENARIOS_RESERVA[indice % len(CENARIOS_RESERVA)]
    texto = janela.texto.lower()
    if any(p in texto for p in _PALAVRAS_BONECA):
        return f"{FICHA_BONECA}, {base}"
    if any(p in texto for p in _PALAVRAS_MENINA):
        return f"{FICHA_MENINA}, {base}"
    return base

_TS = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)")


def _segundos(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _ler_cues(srt_path: Path) -> list[tuple[float, float, str]]:
    """(inicio, fim, texto) por cue do .srt — parser mínimo, so o que
    `legenda.escrever_srt` gera (sem estilos, sem cues aninhadas)."""
    blocos = srt_path.read_text(encoding="utf-8").strip().split("\n\n")
    cues = []
    for bloco in blocos:
        linhas = bloco.strip().splitlines()
        if len(linhas) < 3:
            continue
        m = _TS.search(linhas[1])
        if not m:
            continue
        ini = _segundos(*m.groups()[:4])
        fim = _segundos(*m.groups()[4:])
        texto = " ".join(l.strip() for l in linhas[2:]).replace("\n", " ")
        cues.append((ini, fim, texto))
    return cues


def janelas_do_srt(srt_path: Path, janela_s: float = JANELA_S) -> list[Janela]:
    """Agrupa cues consecutivas ate acumular ~`janela_s`, sem cortar no meio
    de uma cue — o corte de relogio e so um ALVO, a fronteira real segue a
    cue mais proxima dele, igual `legenda._fronteiras` faz para a legenda."""
    cues = _ler_cues(srt_path)
    if not cues:
        return []
    janelas: list[Janela] = []
    inicio = cues[0][0]
    textos: list[str] = []
    fim_atual = cues[0][0]
    for ini, fim, texto in cues:
        textos.append(texto)
        fim_atual = fim
        if fim_atual - inicio >= janela_s:
            janelas.append(Janela(inicio, fim_atual, " ".join(textos)))
            inicio = fim_atual
            textos = []
    if textos:
        janelas.append(Janela(inicio, fim_atual, " ".join(textos)))
    return janelas


def _perguntar_ollama(texto: str, modelo: str, timeout: int, num_predict: int) -> str:
    corpo = json.dumps({
        "model": modelo,
        "prompt": PROMPT_CENA.format(contexto=CONTEXTO_LIVRO, texto=texto[:600],
                                     ficha_menina=FICHA_MENINA,
                                     ficha_boneca=FICHA_BONECA),
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": num_predict},
    }).encode()
    req = request.Request(OLLAMA_URL, data=corpo, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()


def _validar_prompt(saida: str) -> bool:
    return bool(saida) and "\n" not in saida.strip() and len(saida) <= 400


def prompt_da_janela(janela: Janela, indice: int, modelo: str = MODELO_OLLAMA,
                     timeout: int = 45, num_predict: int = 900,
                     usar_llm: bool = True) -> tuple[str, bool]:
    """(prompt em ingles, veio_do_llm). Cai no cenario de reserva (determinístico,
    por indice) se o Ollama estiver fora do ar ou a saida nao validar.

    Uma tentativa so, nao duas escalando o orcamento como `narration/llm.py`
    faz: medido nesta janela (texto em ortografia arcaica) que quando
    `num_predict=600` estoura por 'length' sem emitir resposta, `1200`
    tambem estoura quase sempre — o gemma4:12b entra num raciocinio interno
    longo demais para este tipo de trecho, nao perto do limite. Insistir
    custaria ~2x o tempo (~40s por janela) para ganhar pouco; 900 fica no
    meio do caminho e o cenario de reserva ja tem variedade propria.

    `usar_llm=False` pula a chamada de rede inteira — medido em livros de
    ortografia antiga (ex.: Narizinho, 1920): taxa de acerto do Ollama caiu
    perto de zero, e cada tentativa falha ainda paga o timeout inteiro. Sem
    ele, ~300 janelas saem em minutos em vez de horas, so com o cenario de
    reserva (que ja varia por indice e detecta menina/boneca por palavra-chave).
    """
    if not usar_llm:
        return _fallback(janela, indice), False
    try:
        saida = _perguntar_ollama(janela.texto, modelo, timeout, num_predict)
    except (error.URLError, TimeoutError, KeyError, ValueError):
        return _fallback(janela, indice), False
    saida = saida.strip().strip('"').strip("'")
    if not _validar_prompt(saida):
        return _fallback(janela, indice), False
    return saida, True


def gerar_imagens(janelas: list[Janela], destino_dir: Path, *,
                  n_candidatos: int = 2, modelo: str = "flux",
                  usar_llm: bool = True, prompts_prontos: list[str] | None = None,
                  progresso=None) -> list[tuple[Path | None, bool]]:
    """Gera e escolhe automaticamente uma imagem por janela, em UMA invocacao
    em lote da venv isolada (nao uma por janela — recarregar o FLUX ~300
    vezes custaria a maior parte do tempo so em carga de modelo).

    `prompts_prontos` pula o Ollama E o cenario de reserva: um prompt por
    janela, já escrito por fora (ex.: pelo operador ou por outro processo que
    entenda o texto de verdade — medido que o gemma4:12b local não dava conta
    da ortografia arcaica deste livro nem com contexto extra, ver
    `prompt_da_janela`). Precisa bater 1:1 com `janelas` em tamanho e ordem.

    Devolve `(caminho_ou_None, veio_do_llm)` por janela, na mesma ordem — o
    chamador decide o que fazer com janelas sem imagem (None: todos os
    candidatos falharam no QA de "nao vazia").
    """
    if prompts_prontos is not None and len(prompts_prontos) != len(janelas):
        raise ValueError(
            f"{len(prompts_prontos)} prompts prontos para {len(janelas)} janelas")
    destino_dir.mkdir(parents=True, exist_ok=True)
    imagem_mod.resolver_modelo(modelo)
    passos, guidance = imagem_mod._defaults(None, None)

    prompts: list[str] = []
    veio_llm: list[bool] = []
    for i, j in enumerate(janelas):
        if prompts_prontos is not None:
            p, veio = prompts_prontos[i], True
        else:
            p, veio = prompt_da_janela(j, i, usar_llm=usar_llm)
        prompts.append(p)
        veio_llm.append(veio)
        if progresso:
            progresso(f"prompt pronto ({'ollama' if veio else 'fallback'}): {p[:70]}")

    imagem_mod.CACHE.mkdir(parents=True, exist_ok=True)
    pendentes: list[dict] = []
    candidatos_por_janela: list[list[Path]] = []
    for i, prompt in enumerate(prompts):
        destinos = [imagem_mod.CACHE /
                   f"sinc-{i:04d}-{imagem_mod._seed(prompt, k)}.png"
                   for k in range(n_candidatos)]
        candidatos_por_janela.append(destinos)
        for k, destino in enumerate(destinos):
            if destino.exists():
                continue
            pendentes.append({
                "prompt": prompt, "seed": imagem_mod._seed(prompt, k),
                "destino": str(destino), "largura": imagem_mod.LARGURA_PADRAO,
                "altura": imagem_mod.ALTURA_PADRAO, "passos": passos,
                "guidance": guidance,
            })

    if progresso:
        progresso(f"gerando {len(pendentes)} imagens em lote ({modelo})…")
    notas, estilos = imagem_mod.gerar_lote(
        pendentes, avaliar_clip=n_candidatos > 1, avaliar_estilo=True,
        progresso=progresso) if pendentes else ({}, {})

    resultado: list[tuple[Path | None, bool]] = []
    rejeitadas_foto = 0
    for candidatos, veio in zip(candidatos_por_janela, veio_llm):
        pontuacoes = {c: notas.get(str(c), 0.0) for c in candidatos}
        estilos_c = {c: estilos.get(str(c), -1.0) for c in candidatos}
        melhor = imagem_qa.escolher_melhor(candidatos, pontuacoes, estilos_c)
        if melhor is None and any(c.exists() and imagem_qa.nao_e_vazia(c) for c in candidatos):
            rejeitadas_foto += 1
        if melhor is None:
            resultado.append((None, veio))
            continue
        for c in candidatos:
            if c != melhor and c.exists():
                c.unlink()
        finais = imagem_mod.aprovar([melhor], slides_dir=destino_dir)
        resultado.append((finais[0], veio))
    if progresso and rejeitadas_foto:
        progresso(f"{rejeitadas_foto} janela(s) descartada(s) por parecerem "
                 f"foto real em vez de ilustracao (todos os candidatos)")
    return resultado


def plano(srt_path: Path, destino_dir: Path, audio_duracao_s: float, *,
         janela_s: float = JANELA_S, n_candidatos: int = 2,
         modelo: str = "flux", usar_llm: bool = True,
         prompts_prontos: list[str] | None = None, progresso=None
         ) -> tuple[list[Path], list[float], float]:
    """(imagens, duracao por imagem, cruzamento) — mesmo formato de retorno de
    `slides.plano()`, para o `render.py` nao precisar de um caminho separado
    por preset alem de qual funcao chamar.

    Janela sem imagem aproveitavel (todos os candidatos vazios no QA) nao vira
    buraco no video: a duracao dela e somada a imagem anterior (ou a proxima,
    se for a primeira janela) em vez de pular o trecho de audio.

    `usar_llm=False` pula o Ollama inteiro (ver `prompt_da_janela`) — util em
    livros de ortografia antiga, onde a taxa de acerto do LLM medida foi
    perto de zero e cada tentativa falha ainda paga o timeout completo.
    `prompts_prontos`, ver `gerar_imagens`, tem prioridade sobre `usar_llm`.
    """
    janelas = janelas_do_srt(srt_path, janela_s)
    if not janelas:
        raise FileNotFoundError(
            f"nenhuma cue em {srt_path} — preset sincronizado precisa da legenda")
    resultados = gerar_imagens(janelas, destino_dir, n_candidatos=n_candidatos,
                               modelo=modelo, usar_llm=usar_llm,
                               prompts_prontos=prompts_prontos, progresso=progresso)

    imagens: list[Path] = []
    duracoes: list[float] = []
    arrasto = 0.0  # duracao de janelas falhas ainda sem imagem para carregar
    for janela, (caminho, _veio) in zip(janelas, resultados):
        dur = janela.fim_s - janela.inicio_s
        if caminho is None:
            arrasto += dur
            continue
        imagens.append(caminho)
        duracoes.append(dur + arrasto)
        arrasto = 0.0
    if arrasto and duracoes:
        # janelas falhas no FINAL: nao ha "proxima imagem" para carregar,
        # entao ficam com a ultima que saiu.
        duracoes[-1] += arrasto

    if not imagens:
        raise RuntimeError(
            "nenhuma imagem sincronizada aproveitavel — todas as janelas "
            "falharam no QA (`imagem_qa.nao_e_vazia`)")

    # Sobra sobre o audio, mesmo raciocinio de `slides.plano` (MARGEM_S): o
    # video nao pode terminar antes da narracao.
    falta = (audio_duracao_s + slides_mod.MARGEM_S) - sum(duracoes)
    if falta > 0:
        duracoes[-1] += falta

    if progresso:
        falhas = sum(1 for _, veio in resultados if not veio)
        progresso(f"{len(imagens)} imagens sincronizadas prontas "
                  f"({falhas} janelas caíram no prompt de reserva)")
    return imagens, duracoes, slides_mod.CRUZAMENTO
