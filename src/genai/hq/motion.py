"""Motion comic: a HQ narrada, quadro a quadro, com os balões entrando na fala.

Nada de motor novo — três áreas reaproveitadas:

1. **narração (audiolivro)**: os textos do roteiro viram um `script.json`
   (recordatório → narrador; fala/pensamento → a voz do personagem) e passam
   pelo mesmo `run` do audiolivro: TTS, QA por ASR, retomada pelo state.db;
2. **render (video)**: o plano de imagens + durações vai ao preset
   `sincronizado` do `render`, com dissolve curto e as durações compensadas
   para a troca cair no instante da fala (`render.compensar_cruzamento`);
3. **publish**: o MP4 fica em `output/`, e o `genai publish <slug>` funciona
   sem mudança (lê o mesmo project.yaml).

A faixa de áudio é montada AQUI, a partir dos chunks, e não pelo `build` do
audiolivro: o tempo de cada balão precisa ser conhecido ao milissegundo, e
quem monta a faixa é quem sabe onde cada fala começa.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audiolivro.audio.process import masterizar, trim_silencio
from ..audiolivro.narration.rules import normalize
from ..audiolivro.script.models import Chapter, Rights, Script, Segment
from ..core.estado import Store
from ..core.projeto import carregar_config
from ..video import render
from . import letreiro, rostos
from .roteiro import Roteiro

# Ritmo (s). Entrada: o quadro aparece limpo antes do primeiro balão.
ENTRADA_QUADRO = 0.8
PAUSA_BALAO = 0.5
PAUSA_FIM_QUADRO = 0.9
QUADRO_MUDO = 3.0
# Dissolve curto: é o balão aparecendo, não a troca lenta de slide.
CRUZAMENTO = 0.35
FONTE_VIDEO = 40


def _tamanho_video(proporcao_wh: float) -> tuple[int, int]:
    """O quadro inteiro dentro de 1920×1080 (o render preenche o resto com o
    próprio quadro desfocado)."""
    w, h = render.LARGURA, render.ALTURA
    if proporcao_wh >= w / h:
        return w, round(w / proporcao_wh)
    return round(h * proporcao_wh), h


def montar_script(proj: Path, r: Roteiro) -> Script:
    """script.json da narração. Capítulo único; um segmento por texto, na
    ordem de leitura."""
    cfg = carregar_config(proj)
    narrador = cfg.get("narrator")
    if not narrador:
        raise ValueError("project.yaml sem `narrator` — a voz dos recordatórios "
                         "(ver `genai voice list`)")
    cast = {pid: v for pid, p in r.personagens.items() if (v := (cfg.get("cast") or {})
                                                           .get(pid) or p.voz)}
    segs = []
    for q in r.ordem_de_leitura():
        for t in q.textos:
            texto = normalize(t.texto).text
            segs.append(Segment(idx=len(segs), source=t.texto, text=texto,
                                kind="prose" if t.tipo == "recordatorio" else "dialogue",
                                role="narrador" if t.tipo == "recordatorio" else t.quem))
    if not segs:
        raise ValueError("roteiro sem nenhum texto para narrar")
    rights = cfg.get("rights") or {}
    s = Script(title=r.titulo, voice_id=narrador, cast=cast,
               rights=Rights(**rights) if rights.get("status") else None,
               chapters=[Chapter(idx=1, title=r.titulo, segments=segs)])
    s.save(proj / "script.json")
    return s


@dataclass
class Cena:
    """Uma imagem do vídeo: o quadro com os `ate` primeiros balões."""
    quadro: int
    ate: int
    duracao: float


def montar_audio(proj: Path, r: Roteiro) -> tuple[Path, list[Cena]]:
    """Faixa masterizada + a lista de imagens com a duração de cada uma.

    Exige todos os chunks `ok` — o mesmo critério do `build` do audiolivro."""
    store = Store(proj / "state.db")
    try:
        chunks = store.chapter_chunks(1)
    finally:
        store.close()
    n_textos = sum(len(q.textos) for q in r.quadros)
    pendentes = [c["chunk_id"] for c in chunks if c["state"] != "ok"]
    if len(chunks) != n_textos or pendentes:
        raise RuntimeError(f"narração incompleta ({len(chunks) - len(pendentes)}/{n_textos} "
                           f"ok) — rode `genai hq motion` de novo ou revise com "
                           f"`genai review {proj.name}`")
    falas = []
    for c in chunks:
        audio, sr = sf.read(c["wav_path"], dtype="float32")
        falas.append((audio.mean(axis=1) if audio.ndim > 1 else audio, sr))
    sr = falas[0][1]
    if any(x != sr for _, x in falas):
        raise RuntimeError("chunks com taxas de amostragem diferentes")
    partes: list[np.ndarray] = []
    cenas: list[Cena] = []
    silencio = lambda s: np.zeros(int(sr * s), dtype=np.float32)
    k = 0
    for q in r.ordem_de_leitura():
        if not q.textos:
            partes.append(silencio(QUADRO_MUDO))
            cenas.append(Cena(q.id, 0, QUADRO_MUDO))
            continue
        partes.append(silencio(ENTRADA_QUADRO))
        cenas.append(Cena(q.id, 0, ENTRADA_QUADRO))
        for j in range(len(q.textos)):
            fala = trim_silencio(falas[k][0], sr)
            pausa = PAUSA_FIM_QUADRO if j == len(q.textos) - 1 else PAUSA_BALAO
            partes += [fala, silencio(pausa)]
            # a duração sai das amostras, não da soma de segundos: sem deriva
            cenas.append(Cena(q.id, j + 1, (len(fala) + int(sr * pausa)) / sr))
            k += 1
    bruto = proj / "audio" / "motion-bruto.wav"
    bruto.parent.mkdir(parents=True, exist_ok=True)
    sf.write(bruto, np.concatenate(partes), sr)
    master = proj / "audio" / "motion.wav"
    masterizar(bruto, master)
    return master, cenas


def imagens(proj: Path, r: Roteiro, artes: dict[int, Path], cenas: list[Cena]) -> list[Path]:
    """Uma imagem por cena em `motion/`: o quadro letreirado até o balão `ate`.
    A posição dos balões é a mesma em todas (planejada uma vez por quadro)."""
    destino = proj / "motion"
    destino.mkdir(parents=True, exist_ok=True)
    cache: dict[int, tuple] = {}
    saida = []
    for c in cenas:
        q = r.quadro(c.quadro)
        if q.id not in cache:
            w0, h0 = q.tamanho
            tam = _tamanho_video(w0 / h0)
            img, baloes = letreiro.letrar(artes[q.id], q, tam, FONTE_VIDEO,
                                          rostos.detectar(artes[q.id]), ate=0)
            cache[q.id] = (img, baloes)
        base, baloes = cache[q.id]
        p = destino / f"q{q.id:03d}-{c.ate}.png"
        letreiro.desenhar(base, baloes, FONTE_VIDEO, ate=c.ate).save(p)
        saida.append(p)
    return saida


def renderizar(proj: Path, r: Roteiro, artes: dict[int, Path], *, gpu: bool = True,
               nome: str | None = None) -> Path:
    from .exportar import nome_de_arquivo

    master, cenas = montar_audio(proj, r)
    imgs = imagens(proj, r, artes, cenas)
    duracoes = render.compensar_cruzamento([c.duracao for c in cenas], CRUZAMENTO)
    # sobra no fim (o vídeo não pode acabar antes do áudio; o -shortest corta)
    duracoes[-1] += 1.0
    destino = proj / "output" / f"{nome or nome_de_arquivo(r.titulo)}.mp4"
    return render.renderizar(master, destino, preset="sincronizado", gpu=gpu,
                             sincronizado_plano=(imgs, duracoes, CRUZAMENTO))

