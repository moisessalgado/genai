"""Legenda sincronizada a partir dos tempos reais do capitulo montado.

Por que o texto na tela: o operador pediu, e faz sentido para este material --
sutta e texto para acompanhar, nao so ouvir. O .srt tambem sobe separado no
YouTube como closed caption, o que torna o video buscavel e acessivel.

Os tempos vem de `montar_com_marcas`, nunca de somar duracoes do banco: o trim e
o crossfade encurtam cada chunk por uma quantidade diferente.
"""
from __future__ import annotations

import re
from pathlib import Path

# Uma legenda de 300 caracteres em 12 s exige ler a 25 caracteres por segundo --
# rapido demais. Confortavel fica perto de 15. Por isso um segmento longo vira
# VARIAS legendas, repartidas pelo proprio texto.
MAX_CHARS = 84          # duas linhas de ~42, que e o que cabe em 1080p sem poluir
LARGURA_LINHA = 42


def desfazer_lexico(texto: str, lexicon: dict[str, str]) -> str:
    """Devolve a grafia de verdade num texto ja respelado para o TTS.

    A legenda NAO pode sair do `source` do segmento: `source` guarda o paragrafo
    INTEIRO, repetido em cada pedaco cortado dele (medido no ch01: `text` soma
    7.527 caracteres e `source`, 25.914). Usar `source` faz cada legenda exibir
    texto que so sera falado nos pedacos seguintes -- e o que o operador viu como
    "a legenda adianta".

    O `text` tem o tamanho certo, porque e exatamente o que foi falado. So carrega
    o respelling fonetico do lexico, e e isso que se desfaz aqui.

    Limite conhecido: quando duas entradas caem no mesmo respelling (`Sutta` e
    `sutta` viram `súta`), volta a forma MAIUSCULA. Sao nomes proprios na quase
    totalidade dos casos, e um `Sutta` no meio da frase incomoda menos que a
    grafia fonetica na tela.
    """
    inverso: dict[str, str] = {}
    for original, respelling in lexicon.items():
        chave = respelling.lower()
        atual = inverso.get(chave)
        if atual is None or (original[:1].isupper() and not atual[:1].isupper()):
            inverso[chave] = original
    # frases mais longas primeiro: "dama tchaca pavátana" antes de "dama"
    for chave in sorted(inverso, key=len, reverse=True):
        texto = re.sub(rf"(?<!\w){re.escape(chave)}(?!\w)", inverso[chave],
                       texto, flags=re.IGNORECASE)
    return texto


def _quebrar(texto: str, maximo: int = MAX_CHARS) -> list[str]:
    """Reparte um texto longo em pedacos, sempre em fronteira de palavra."""
    palavras = texto.split()
    if not palavras:
        return []
    pedacos, atual = [], ""
    for p in palavras:
        cand = f"{atual} {p}".strip()
        if len(cand) > maximo and atual:
            pedacos.append(atual)
            atual = p
        else:
            atual = cand
    if atual:
        pedacos.append(atual)
    return pedacos


def _duas_linhas(texto: str, largura: int = LARGURA_LINHA) -> str:
    """Quebra em ate duas linhas equilibradas -- linha unica muito longa cansa."""
    if len(texto) <= largura:
        return texto
    palavras = texto.split()
    melhor, corte = None, None
    for i in range(1, len(palavras)):
        a = " ".join(palavras[:i])
        b = " ".join(palavras[i:])
        custo = abs(len(a) - len(b))
        if melhor is None or custo < melhor:
            melhor, corte = custo, (a, b)
    return "\n".join(corte) if corte else texto


# Quanto uma fronteira pode ser puxada para encostar numa pausa da fala. Larga
# porque o alvo proporcional erra justamente por ai; se nao houver pausa dentro
# desta janela, o alvo proporcional fica como estava.
IMA_S = 1.5

# Nenhuma legenda pode durar menos que isto. Nao e conforto de leitura: sem um
# piso, o ima consome o tempo do pedaco seguinte e produz legenda de duracao
# ZERO -- texto que nunca chega a aparecer. Medido: 44 das 339 do capitulo,
# antes deste piso existir.
MIN_CUE_S = 0.35


def _fronteiras(ini: float, fim: float, pedacos: list[str],
                pausas: list[float]) -> list[float]:
    """Onde cortar entre um pedaco e o proximo, em segundos absolutos.

    O alvo e proporcional ao numero de caracteres, mas ele so acerta se a fala
    tiver taxa constante de caracteres por segundo -- e nao tem. Entao cada alvo
    e puxado para a pausa de fala mais proxima: e onde o ouvinte percebe a quebra,
    e onde a legenda tem de virar.

    Toda escolha e espremida numa janela que garante MIN_CUE_S para o pedaco que
    fecha E para todos os que ainda faltam. Sem isso o ima engole pedacos.
    """
    n = len(pedacos)
    total = sum(len(p) for p in pedacos)
    alvos, acc = [], 0
    for p in pedacos[:-1]:
        acc += len(p)
        alvos.append(ini + (fim - ini) * acc / total if total else fim)

    livres = sorted(pausas)
    saida, minimo = [], ini
    for k, alvo in enumerate(alvos):
        restantes = n - 1 - k              # pedacos que ainda comecam depois deste corte
        piso = minimo + MIN_CUE_S
        teto = fim - MIN_CUE_S * (restantes + 1)
        if teto < piso:                    # segmento curto demais para tanto pedaco
            piso = teto = (piso + teto) / 2
        perto = [x for x in livres if abs(x - alvo) <= IMA_S and piso <= x <= teto]
        if perto:
            escolha = min(perto, key=lambda x: abs(x - alvo))
            livres.remove(escolha)         # uma pausa nao serve a duas quebras
        else:
            escolha = min(max(alvo, piso), teto)
        saida.append(escolha)
        minimo = escolha
    return saida


def cues(marcas: list[tuple[float, float]], textos: list[str],
         pausas: list[list[float]] | None = None
         ) -> list[tuple[float, float, str]]:
    """(inicio, fim, texto) por legenda, repartindo segmentos longos."""
    saida: list[tuple[float, float, str]] = []
    for k, ((ini, fim), texto) in enumerate(zip(marcas, textos)):
        pedacos = _quebrar(texto)
        if not pedacos:
            continue
        ps = pausas[k] if pausas and k < len(pausas) else []
        cortes = _fronteiras(ini, fim, pedacos, ps)
        bordas = [ini, *cortes, fim]
        for i, p in enumerate(pedacos):
            saida.append((bordas[i], bordas[i + 1], _duas_linhas(p)))
    return saida


def _ts(s: float) -> str:
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    seg, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{seg:02d},{ms:03d}"


def escrever_srt(destino: Path, marcas: list[tuple[float, float]],
                 textos: list[str],
                 pausas: list[list[float]] | None = None) -> Path:
    linhas = []
    for i, (ini, fim, txt) in enumerate(cues(marcas, textos, pausas), start=1):
        linhas.append(f"{i}\n{_ts(ini)} --> {_ts(fim)}\n{txt}\n")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas), encoding="utf-8")
    return destino
