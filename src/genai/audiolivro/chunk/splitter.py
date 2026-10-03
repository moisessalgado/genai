"""Segmentacao para o TTS (TDD 8.1).

Duas restricoes vindas do comportamento medido do Chatterbox:
  - chunks longos alucinam/repetem (limite ~300 chars);
  - chunks muito curtos ("Sim.", "Por quê?") produzem gibberish -- por isso
    fragmentos curtos sao fundidos ao vizinho em vez de sintetizados sozinhos.
"""
from __future__ import annotations

import re

MAX_CHARS = 300
MIN_CHARS = 25

# Nao quebrar dentro destes: numeros por extenso longos e siglas soletradas
_FIM_SENTENCA = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁ-Ú«\"'—])")
_FIM_CLAUSULA = re.compile(r"(?<=[;:])\s+")
_VIRGULA = re.compile(r"(?<=,)\s+")


def _split_by(pattern: re.Pattern, text: str) -> list[str]:
    return [p.strip() for p in pattern.split(text) if p.strip()]


def _hard_split(text: str, limit: int) -> list[str]:
    """Ultimo recurso: quebra por espaco respeitando o limite."""
    palavras, out, atual = text.split(), [], ""
    for w in palavras:
        if atual and len(atual) + 1 + len(w) > limit:
            out.append(atual)
            atual = w
        else:
            atual = f"{atual} {w}".strip()
    if atual:
        out.append(atual)
    return out


def split_paragraph(text: str, max_chars: int = MAX_CHARS,
                    min_chars: int = MIN_CHARS) -> list[str]:
    """Divide um paragrafo em chunks sintetizaveis, do mais semantico ao mais bruto."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pedacos: list[str] = []
    for sent in _split_by(_FIM_SENTENCA, text) or [text]:
        if len(sent) <= max_chars:
            pedacos.append(sent)
            continue
        for cl in _split_by(_FIM_CLAUSULA, sent) or [sent]:
            if len(cl) <= max_chars:
                pedacos.append(cl)
                continue
            for vg in _split_by(_VIRGULA, cl) or [cl]:
                if len(vg) <= max_chars:
                    pedacos.append(vg)
                else:
                    pedacos.extend(_hard_split(vg, max_chars))

    return _merge_short(pedacos, max_chars, min_chars)


def _merge_short(pedacos: list[str], max_chars: int, min_chars: int) -> list[str]:
    """Funde fragmentos curtos ao vizinho -- evita a alucinacao de chunk curto."""
    out: list[str] = []
    for p in pedacos:
        if out and (len(p) < min_chars or len(out[-1]) < min_chars):
            candidato = f"{out[-1]} {p}"
            if len(candidato) <= max_chars:
                out[-1] = candidato
                continue
        out.append(p)
    return out
