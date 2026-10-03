"""Camada 1 do normalizador de narracao: regras deterministicas pt-BR (TDD 6.1).

Faz 90-95% do trabalho sem LLM. O que a regra nao resolve com seguranca e marcado
como span ambiguo, para a Camada 2 decidir -- nunca adivinhado aqui.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from num2words import num2words

# --- tabelas -----------------------------------------------------------------

ABREVIACOES = {
    "Sr.": "senhor", "Sra.": "senhora", "Srta.": "senhorita",
    "Dr.": "doutor", "Dra.": "doutora", "Prof.": "professor", "Profa.": "professora",
    "D.": "dom", "Exmo.": "excelentissimo", "Ilmo.": "ilustrissimo",
    "pág.": "página", "págs.": "páginas", "p.": "página",
    "cap.": "capítulo", "séc.": "século", "sécs.": "séculos",
    "art.": "artigo", "cf.": "conforme", "op. cit.": "obra citada",
    "etc.": "etcétera", "ex.": "exemplo", "obs.": "observação",
    "vol.": "volume", "ed.": "edição", "org.": "organizador",
    "n.": "número", "nº": "número", "núm.": "número",
    "a.C.": "antes de Cristo", "d.C.": "depois de Cristo",
}

UNIDADES = {
    "km": "quilômetros", "km²": "quilômetros quadrados", "m": "metros",
    "m²": "metros quadrados", "cm": "centímetros", "mm": "milímetros",
    "kg": "quilos", "g": "gramas", "t": "toneladas",
    "l": "litros", "ml": "mililitros", "h": "horas", "min": "minutos", "s": "segundos",
    "°C": "graus celsius", "%": "por cento",
}

SIMBOLOS = {
    "&": " e ", "@": " arroba ", "#": " número ", "+": " mais ",
    "=": " igual a ", "§": "parágrafo ", "…": "...",
}

MESES = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio", 6: "junho",
    7: "julho", 8: "agosto", 9: "setembro", 10: "outubro", 11: "novembro",
    12: "dezembro",
}

ROMANOS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21,
}


@dataclass
class AmbiguousSpan:
    """Trecho que a regra recusa decidir. Vai para a Camada 2 (LLM)."""

    start: int
    end: int
    text: str
    reason: str


@dataclass
class NormalizationResult:
    text: str
    ambiguous: list[AmbiguousSpan] = field(default_factory=list)
    applied: list[tuple[str, str]] = field(default_factory=list)


def _num(n: int | float, ordinal: bool = False) -> str:
    """num2words em pt_BR insere virgulas ("mil, seiscentos e quarenta e oito").

    Numa narracao isso vira uma pausa espuria no meio do numero, entao a virgula
    interna e removida -- a conjuncao "e" ja basta para a prosodia.
    """
    out = num2words(n, lang="pt_BR", to="ordinal" if ordinal else "cardinal")
    return out.replace(",", "")


def _preservar_caixa(original: str, novo: str) -> str:
    """A grafia do lexico e minuscula; devolve `novo` na caixa que o texto usava
    (inicio de frase capitaliza, mas o lexico nao precisa de uma entrada por caixa)."""
    if original.isupper():
        return novo.upper()
    if original[:1].isupper():
        return novo[:1].upper() + novo[1:]
    return novo


def _expand_lexicon(text: str, lexicon: dict[str, str]) -> tuple[str, list]:
    """Aplica o lexico de pronuncia (nomes proprios, termos indigenas, ortografia
    antiga). Casa sem diferenciar caixa -- a mesma palavra aparece capitalizada no
    inicio de frase e minuscula no meio -- preservando a caixa encontrada."""
    applied = []
    for grafia in sorted(lexicon, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(grafia)}\b", re.IGNORECASE)
        novo = lexicon[grafia]
        if pattern.search(text):
            text = pattern.sub(lambda m, novo=novo: _preservar_caixa(m.group(0), novo), text)
            applied.append((grafia, novo))
    return text, applied


def detect_ambiguous(text: str) -> list[AmbiguousSpan]:
    """Acha spans ambiguos no texto CRU, antes de qualquer substituicao.

    A ordem importa: se as regras rodarem primeiro, os offsets viram lixo (o texto
    encolheu ou cresceu) e a Camada 2 aplicaria a expansao no lugar errado. Por isso
    a Camada 2 opera sobre o texto original, e as regras rodam depois.
    """
    spans: list[AmbiguousSpan] = []
    for m in re.finditer(r"\b\d{4}\b", text):
        spans.append(AmbiguousSpan(m.start(), m.end(), m.group(0),
                                   "numero de 4 digitos: ano ou quantidade?"))
    for m in re.finditer(r"\b([A-ZÁ-Ú][a-zá-ú]+)\s+([IVXL]{1,5})\b", text):
        # "século XVII" ja e resolvido por regra; so nomes proprios sao ambiguos
        if m.group(1).lower().startswith("s\u00e9culo"):
            continue
        spans.append(AmbiguousSpan(m.start(2), m.end(2), m.group(2),
                                   "romano apos nome proprio: ordinal contextual"))
    return sorted(spans, key=lambda s: s.start)


def normalize(text: str, lexicon: dict[str, str] | None = None) -> NormalizationResult:
    """Normaliza um paragrafo para narracao. Nao reescreve prosa."""
    applied: list[tuple[str, str]] = []
    # Ambiguidade e detectada no texto de ENTRADA -- depois das regras os offsets
    # nao valeriam mais nada.
    ambiguous = detect_ambiguous(text)

    if lexicon:
        text, lex_applied = _expand_lexicon(text, lexicon)
        applied += lex_applied

    # Convenção tipográfica de textos antigos (pré-reforma ortográfica de 1943):
    # "E'" para "É" quando a máquina de escrever/tipografia não tinha o acento.
    # Medido em texto de 1931 (Wikisource): 13 ocorrências, e sem normalizar o
    # TTS trava/derrapa nesse token incomum -- mesma classe de problema que
    # abreviação, resolvida antes do resto para não interferir com as regras
    # de pontuação seguintes. Só "E'" -- não há evidência de "A'" no texto
    # medido, então não generalizar sem medir de novo.
    def _apostrofo_acento(m):
        novo = "É" if m.group(1) == "E" else "é"
        applied.append((m.group(0), novo))
        return novo

    text = re.sub(r"\b([Ee])'(?=\s)", _apostrofo_acento, text)

    # Abreviacoes (antes dos numeros: "séc. XVII" depende disso)
    for abbr, full in sorted(ABREVIACOES.items(), key=lambda kv: -len(kv[0])):
        pattern = re.compile(rf"(?<!\w){re.escape(abbr)}")
        if pattern.search(text):
            text = pattern.sub(full, text)
            applied.append((abbr, full))

    # Seculos em romanos: "século XVII" -> "século dezessete"
    def _seculo(m):
        val = ROMANOS.get(m.group(2).upper())
        if val is None:
            return m.group(0)
        applied.append((m.group(0), f"{m.group(1)} {_num(val)}"))
        return f"{m.group(1)} {_num(val)}"

    text = re.sub(r"\b(séculos?)\s+([IVXL]+)\b", _seculo, text, flags=re.IGNORECASE)

    # Datas por extenso: "5 de maio de 1648"
    def _data_completa(m):
        dia, mes, ano = int(m.group(1)), m.group(2), int(m.group(3))
        novo = f"{_num(dia)} de {mes} de {_num(ano)}"
        applied.append((m.group(0), novo))
        return novo

    meses_re = "|".join(MESES.values())
    text = re.sub(rf"\b(\d{{1,2}})\s+de\s+({meses_re})\s+de\s+(\d{{3,4}})\b",
                  _data_completa, text, flags=re.IGNORECASE)

    # Datas numericas: 05/05/1648
    def _data_num(m):
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mth <= 12):
            return m.group(0)
        novo = f"{_num(d)} de {MESES[mth]} de {_num(y)}"
        applied.append((m.group(0), novo))
        return novo

    text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", _data_num, text)

    # Horas: 14h30
    def _hora(m):
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        novo = f"{_num(h)} horas" + (f" e {_num(mi)}" if mi else "")
        applied.append((m.group(0), novo))
        return novo

    text = re.sub(r"\b(\d{1,2})h(\d{2})?\b", _hora, text)

    # Percentual e unidades: "12 km", "30%"
    def _unidade(m):
        val, un = m.group(1), m.group(2)
        if un not in UNIDADES:
            return m.group(0)
        num = float(val.replace(".", "").replace(",", "."))
        num_i = int(num) if num == int(num) else num
        novo = f"{_num(num_i)} {UNIDADES[un]}"
        applied.append((m.group(0), novo))
        return novo

    unidades_re = "|".join(re.escape(u) for u in sorted(UNIDADES, key=len, reverse=True))
    text = re.sub(rf"\b([\d.,]+)\s*({unidades_re})(?!\w)", _unidade, text)

    # Ordinais explicitos: 1º, 2ª
    def _ordinal(m):
        n = int(m.group(1))
        fem = m.group(2) in ("ª", "a")
        palavra = num2words(n, lang="pt_BR", to="ordinal")
        if fem:
            palavra = re.sub(r"o\b", "a", palavra)
        applied.append((m.group(0), palavra))
        return palavra

    text = re.sub(r"\b(\d+)\s*(º|ª|°)", _ordinal, text)

    # Simbolos soltos
    for sym, spoken in SIMBOLOS.items():
        if sym in text:
            text = text.replace(sym, spoken)
            applied.append((sym, spoken.strip()))

    # Numeros restantes
    def _numero(m):
        raw = m.group(0)
        limpo = raw.replace(".", "").replace(",", ".")
        try:
            val = float(limpo)
        except ValueError:
            return raw
        val_i = int(val) if val == int(val) else val
        # Anos de 4 digitos sao ambiguos: podem ser data ou quantidade.
        # A regra expande como cardinal (leitura correta em pt-BR para anos)
        # mas registra o span para a Camada 2 confirmar quando houver duvida.
        novo = _num(val_i)
        applied.append((raw, novo))
        return novo

    text = re.sub(r"\b[\d.,]*\d\b", _numero, text)

    text = re.sub(r"\s{2,}", " ", text).strip()
    return NormalizationResult(text=text, ambiguous=ambiguous, applied=applied)
