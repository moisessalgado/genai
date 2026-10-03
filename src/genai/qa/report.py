"""Relatorio de QA do projeto (TDD, "Verificacao").

Junta as tres fontes que ja existem -- a fila em SQLite, os logs de execucao e os
masters no disco -- e confronta cada metrica com o alvo objetivo do TDD. E o que
responde "o audiolivro esta pronto para publicar?" sem depender de memoria.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..audio.process import LUFS_ALVO, medir_loudness
from ..qa.speaker import SIMILARIDADE_MINIMA as LIMIAR_VOZ
from ..qa.speaker import limiar_por_duracao

CER_ALVO = 0.02
REGEN_ALVO = 0.05
REVIEW_ALVO = 0.003
VOZ_ALVO = 0.99
RTF_ALVO = 0.6


@dataclass
class Metrica:
    nome: str
    valor: float | None
    alvo: str
    ok: bool | None
    detalhe: str = ""

    @property
    def marca(self) -> str:
        return "—" if self.ok is None else ("OK" if self.ok else "FORA DO ALVO")


def _pct(n: int, total: int) -> float:
    return n / total if total else 0.0


def ler_logs(logs_dir: Path) -> list[dict]:
    """Todas as linhas de todos os runs, na ordem cronologica dos arquivos."""
    linhas = []
    for f in sorted(logs_dir.glob("run-*.jsonl")):
        for linha in f.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                try:
                    linhas.append(json.loads(linha))
                except json.JSONDecodeError:
                    continue
    return linhas


def coletar(projeto: Path, store, medir_audio: bool = True) -> list[Metrica]:
    st = store.stats()
    total = st["total"] or 0
    linhas = ler_logs(projeto / "logs")

    # Fidelidade: a media esconde o chunk ruim, entao o alvo e sobre a FRACAO de
    # chunks dentro do limite, nao sobre a media.
    cers = [r["cer"] for r in store.conn.execute(
        "SELECT cer FROM chunks WHERE cer IS NOT NULL")]
    dentro = sum(1 for c in cers if c <= CER_ALVO)
    m = [Metrica("Fidelidade de conteúdo (CER ≤ 2%)",
                 _pct(dentro, len(cers)) if cers else None,
                 "100% dos chunks",
                 (dentro == len(cers)) if cers else None,
                 f"CER médio {st['cer_medio']:.4f}" if st["cer_medio"] is not None else
                 "QA desligada (--no-qa)")]

    # Regeneracao: contada nos logs, porque a coluna `attempts` do banco conta
    # reivindicacoes de run, nao as tentativas internas de melhor-de-N.
    if linhas:
        regen = sum(1 for r in linhas if (r.get("tentativas") or 1) > 1)
        m.append(Metrica("Taxa de regeneração", _pct(regen, len(linhas)), "< 5%",
                         _pct(regen, len(linhas)) < REGEN_ALVO,
                         f"{regen} de {len(linhas)} chunks gerados"))
        # RTF pelo relogio de PAREDE das execucoes, nao pela soma dos chunks: com
        # varios workers a soma conta o mesmo intervalo N vezes. Inclui a carga do
        # modelo de proposito -- e o tempo que o operador espera de verdade.
        runs = store.runs()
        parede = sum(r["parede_s"] or 0 for r in runs)
        aud = sum(r["audio_s"] or 0 for r in runs)
        rtf = parede / aud if aud else None
        if rtf is not None:
            m.append(Metrica("Velocidade (RTF)", rtf, "< 0,6", rtf < RTF_ALVO,
                             f"{aud/60:.1f} min de áudio em {parede/60:.1f} min "
                             f"de relógio, em {len(runs)} execução(ões)"))

    rev = st["needs_review"]
    m.append(Metrica("Chunks irrecuperáveis", _pct(rev, total), "< 0,3%",
                     _pct(rev, total) < REVIEW_ALVO,
                     f"{rev} em {total}"))

    # O limiar acompanha a duracao do trecho, entao cada chunk e medido contra o
    # SEU limiar -- comparar tudo com 0,88 reprovaria chunk curto que foi aceito.
    medidos = store.conn.execute(
        "SELECT speaker_sim, duration_s FROM chunks WHERE speaker_sim IS NOT NULL "
        "AND duration_s IS NOT NULL").fetchall()
    sims = [r["speaker_sim"] for r in medidos]
    if sims:
        acima = sum(1 for r in medidos
                    if r["speaker_sim"] >= limiar_por_duracao(r["duration_s"]))
        m.append(Metrica("Consistência de voz", _pct(acima, len(sims)),
                         f"> 99% acima do limiar ({LIMIAR_VOZ} em áudio longo)",
                         _pct(acima, len(sims)) >= VOZ_ALVO,
                         f"média {sum(sims)/len(sims):.3f} · pior {min(sims):.3f}"))

    m.append(Metrica("Progresso", _pct(st["ok"], total), "100% dos chunks 'ok'",
                     st["ok"] == total and total > 0,
                     f"{st['ok']}/{total} · {st['audio_s']/60:.1f} min de áudio"))

    if medir_audio:
        m += _loudness(projeto)
    return m


def _loudness(projeto: Path) -> list[Metrica]:
    """Mede os masters entregues -- a unica prova de que a cadeia de audio funcionou."""
    masters = sorted((projeto / "output").glob("*-master.wav"))
    if not masters:
        return []
    fora = []
    valores = []
    for w in masters:
        try:
            med = medir_loudness(w)
        except Exception:
            continue
        i, tp = float(med["input_i"]), float(med["input_tp"])
        valores.append(i)
        if abs(i - LUFS_ALVO) > 0.5 or tp > -1.0:
            fora.append(f"{w.stem}: {i:.1f} LUFS / {tp:.1f} dBTP")
    if not valores:
        return []
    return [Metrica("Loudness dos masters", sum(valores) / len(valores),
                    "−16 LUFS ±0,5 · TP ≤ −1,0 dBTP", not fora,
                    "; ".join(fora) if fora else f"{len(valores)} capítulo(s) no alvo")]


def markdown(titulo: str, metricas: list[Metrica], review: list) -> str:
    linhas = [f"# Relatório de QA — {titulo}\n",
              "| Métrica | Medido | Alvo | Situação | Detalhe |",
              "|---|---|---|---|---|"]
    for m in metricas:
        val = "—" if m.valor is None else (
            f"{m.valor*100:.1f}%" if 0 <= m.valor <= 1 and "RTF" not in m.nome
            and "Loudness" not in m.nome else f"{m.valor:.3f}")
        if "Loudness" in m.nome:
            val = f"{m.valor:.1f} LUFS"
        linhas.append(f"| {m.nome} | {val} | {m.alvo} | {m.marca} | {m.detalhe} |")
    if review:
        linhas.append(f"\n## Chunks em revisão ({len(review)})\n")
        for r in review:
            linhas.append(f"- `{r['chunk_id']}` — {r['error']}")
            linhas.append(f"  - esperado: {r['text'][:160]}")
            linhas.append(f"  - ouvido: {(r['transcript'] or '')[:160]}")
    else:
        linhas.append("\nNenhum chunk pendente de revisão.\n")
    return "\n".join(linhas) + "\n"
