"""Ingest e limpeza de texto, com deteccao de capitulos (TDD 8, TEXT PROCESSOR)."""
from __future__ import annotations

import re
from pathlib import Path

# Numero do capitulo por extenso. Lista FECHADA de proposito: com um `[a-zà-ú]+`
# solto, "Livro dos dias" e "Parte de mim" viravam titulo de capitulo, e todo o
# texto antes deles ia junto para outro capitulo.
_ORDINAIS = (
    "primeir|segund|terceir|quart|quint|sext|s[eé]tim|oitav|non|d[eé]cim|"
    "und[eé]cim|duod[eé]cim|vig[eé]sim"
)
# O romano fica MAIUSCULO a ferro: o resto do padrao e IGNORECASE, e com isso o
# "d" de "Livro dos dias" casava como numeral romano.
_NUMERO_CAP = rf"(?-i:[IVXLCDM]+)|\d+|(?:{_ORDINAIS})[oa]|[úu]nic[oa]|final"

# Titulos de capitulo em obras em portugues
_CAPITULO = re.compile(
    rf"^\s*(cap[ií]tulo|cap\.|parte|livro|se[cç][aã]o)\s+"
    rf"({_NUMERO_CAP})\s*[-–—.:·]?\s*(.*)$",
    re.IGNORECASE)
# O separador e OBRIGATORIO quando ha texto depois do numeral. Sem isso, "D" e um
# romano valido e a linha "Discurso de Dissolucao" virava titulo de secao -- e como
# I, V, X, L, C, D e M abrem meia lingua portuguesa ("Como", "Livro", "Vocês",
# "Isso", "Minha"), qualquer paragrafo curto virava capitulo.
_TITULO_ROMANO = re.compile(r"^\s*([IVXLCDM]{1,7})\s*(?:[-–—.:·]\s*(.{0,60}))?\s*$")


def ler(path: Path) -> str:
    """Le TXT, MD, EPUB ou PDF e devolve texto bruto."""
    suf = path.suffix.lower()
    if suf in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suf == ".epub":
        return _ler_epub(path)
    if suf == ".pdf":
        return ler_pdf(path)
    raise ValueError(f"formato nao suportado: {suf}")


def _ler_epub(path: Path) -> str:
    from bs4 import BeautifulSoup
    import ebooklib
    from ebooklib import epub

    livro = epub.read_epub(str(path))
    partes = []
    for item in livro.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        sopa = BeautifulSoup(item.get_content(), "html.parser")
        texto = sopa.get_text("\n")
        if texto.strip():
            partes.append(texto)
    return "\n\n".join(partes)


# Fracao de paginas em que uma linha precisa aparecer para ser cabecalho/rodape.
# Abaixo disso pode ser texto legitimo repetido (um refrao, um nome de secao).
_REPETICAO_MIN = 0.30
# Linhas de borda examinadas em cada ponta da pagina.
_BORDAS = 2


def ler_pdf(path: Path) -> str:
    """Extrai texto de PDF digital, por blocos, removendo cabecalho e rodape.

    Le por BLOCOS e nao por linhas: num PDF cada linha visual vira uma quebra, e
    reconstruir paragrafo a partir de pontuacao erra sempre que uma frase termina
    no meio do paragrafo. O bloco do PyMuPDF ja e, na pratica, o paragrafo.

    PDF escaneado (sem camada de texto) e recusado: OCR esta fora do escopo
    (TDD 17, "Fora"), e um OCR silencioso entregaria lixo ao TTS.
    """
    import pymupdf

    doc = pymupdf.open(str(path))
    try:
        paginas = [_blocos_da_pagina(pg) for pg in doc]
    finally:
        doc.close()

    if not any(paginas):
        raise ValueError(
            f"{path.name}: nenhum texto extraivel — provavelmente um PDF escaneado. "
            "OCR esta fora do escopo; passe um PDF com camada de texto ou um TXT.")

    descartar = _bordas_repetidas(paginas)
    partes: list[str] = []
    for blocos in paginas:
        for b in blocos:
            if _chave(b) in descartar:
                continue
            partes.append(b)
    return "\n\n".join(partes)


def _blocos_da_pagina(pagina) -> list[str]:
    """Blocos de texto da pagina, em ordem de leitura, com linhas ja unidas."""
    out = []
    for bloco in pagina.get_text("blocks", sort=True):
        if len(bloco) > 6 and bloco[6] != 0:   # 1 = imagem
            continue
        texto = _unir_linhas(bloco[4])
        if texto:
            out.append(texto)
    return out


def _unir_linhas(bloco: str) -> str:
    """Une as linhas visuais de um bloco num paragrafo unico."""
    linhas = [l.strip() for l in bloco.splitlines() if l.strip()]
    if not linhas:
        return ""
    texto = ""
    for linha in linhas:
        if not texto:
            texto = linha
        elif texto.endswith("-"):          # palavra quebrada pela margem
            texto = texto[:-1] + linha
        else:
            texto = f"{texto} {linha}"
    return texto.strip()


def _chave(bloco: str) -> str:
    """Identidade de cabecalho/rodape: sem digitos, para casar 'Pagina 12' com 'Pagina 13'."""
    return re.sub(r"\d+", "#", bloco.strip().lower())


def _bordas_repetidas(paginas: list[list[str]]) -> set[str]:
    """Blocos curtos que se repetem no topo/rodape da maioria das paginas.

    So olha paginas com corpo suficiente para que "borda" signifique alguma coisa:
    numa pagina de dois blocos, todo bloco e borda, e o proprio texto do livro
    seria descartado. Como `_chave` apaga os digitos (para casar "Pagina 12" com
    "Pagina 13"), sem essa guarda uma pagina esparsa perderia o corpo.
    """
    from collections import Counter

    # Toda pagina precisa sobrar pelo menos um bloco no meio: se cabecalho e rodape
    # cobrissem a pagina inteira, o corpo do livro entraria na conta de repeticao.
    candidatas = [p for p in paginas if len(p) >= 3]
    if len(candidatas) < 4:
        return set()
    contagem: Counter[str] = Counter()
    for blocos in candidatas:
        k = _BORDAS if len(blocos) >= _BORDAS * 2 + 1 else 1
        bordas = blocos[:k] + blocos[-k:]
        for b in {_chave(x) for x in bordas if len(x) <= 80}:
            contagem[b] += 1
    minimo = max(2, int(len(candidatas) * _REPETICAO_MIN))
    return {k for k, n in contagem.items() if n >= minimo}


def limpar(texto: str) -> str:
    """Limpeza conservadora: nao reescreve prosa, so remove artefato de digitalizacao."""
    # de-hifenizacao de quebra de linha: "expedi-\ncao" -> "expedicao"
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    # marcadores de nota de rodape: "[1]", "[12]". Sem isto o normalizador os
    # expande e a narracao ganha um "um" ou "doze" no meio da frase.
    texto = re.sub(r"\[\d{1,3}\]", "", texto)
    # numeros de pagina isolados
    texto = re.sub(r"^\s*\d{1,4}\s*$", "", texto, flags=re.MULTILINE)
    # cabecalho/rodape repetido em CAIXA ALTA curto
    texto = re.sub(r"^[A-ZÁ-Ú\s]{4,40}$\n", "", texto, flags=re.MULTILINE)
    # aspas e travessoes tipograficos preservados; so normaliza espaco
    texto = texto.replace("\r\n", "\n").replace(" ", " ")
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    # junta linhas quebradas dentro do mesmo paragrafo
    texto = re.sub(r"(?<![.!?:;\n])\n(?![\n\s])", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def detectar_capitulos(texto: str) -> list[tuple[str, str]]:
    """Devolve [(titulo, corpo)]. Sem marcacao clara, retorna um capitulo unico."""
    linhas = texto.split("\n")
    marcos: list[tuple[int, str]] = []
    for i, linha in enumerate(linhas):
        s = linha.strip()
        if not s or len(s) > 80:
            continue
        m = _CAPITULO.match(s)
        if m:
            marcos.append((i, s))
            continue
        # Romano isolado so conta se a linha seguinte estiver vazia (titulo de secao)
        if _TITULO_ROMANO.match(s) and i + 1 < len(linhas) and not linhas[i + 1].strip():
            marcos.append((i, s))

    if not marcos:
        return [("Texto completo", texto)]

    caps = []
    # O que vem ANTES do primeiro marco e conteudo: rosto, autor, epigrafe,
    # nota de edicao. Sem isto ele era descartado em silencio -- o pior tipo de
    # defeito neste pipeline, porque o audio sai correto e so falta um pedaco.
    preambulo = "\n".join(linhas[:marcos[0][0]]).strip()
    if preambulo:
        caps.append(("Abertura", preambulo))
    for n, (ini, titulo) in enumerate(marcos):
        fim = marcos[n + 1][0] if n + 1 < len(marcos) else len(linhas)
        corpo = "\n".join(linhas[ini + 1:fim]).strip()
        if corpo:
            caps.append((titulo, corpo))
    return caps or [("Texto completo", texto)]


# Paragrafo isolado menor que isto e fundido ao seguinte. Medido em dois textos
# diferentes: "Jiddu Krishnamurti" (18 chars) e "Dhammacakkapavattana Sutta"
# (25 chars) viraram gibberish quando sintetizados sozinhos -- o Chatterbox
# precisa de contexto. O mesmo termo, com a linha seguinte junto, sai correto.
# E o mesmo limite do chunker (chunk/splitter.MIN_CHARS), que ja funde fragmentos
# curtos DENTRO de um paragrafo; aqui a fusao e ENTRE paragrafos.
MIN_PARAGRAFO = 30


def paragrafos(corpo: str, min_chars: int = MIN_PARAGRAFO) -> list[str]:
    """Paragrafos do corpo, com os curtos demais fundidos ao seguinte."""
    brutos = [p.strip() for p in re.split(r"\n\s*\n", corpo) if p.strip()]
    out: list[str] = []
    pendente = ""
    for par in brutos:
        atual = f"{pendente} {par}".strip() if pendente else par
        if len(atual) < min_chars:
            # ainda curto: segura e tenta juntar com o proximo
            pendente = atual
            continue
        pendente = ""
        out.append(atual)
    if pendente:
        # sobrou um caco no fim: cola no ultimo, senao ele iria sozinho ao TTS
        if out:
            out[-1] = f"{out[-1]} {pendente}".strip()
        else:
            out.append(pendente)
    return out
