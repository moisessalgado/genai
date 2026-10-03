"""Orquestracao: fila -> TTS -> QA -> disco (TDD 8).

Invariantes:
  - retomar e o comportamento padrao: processa tudo que nao esta 'ok';
  - um chunk irrecuperavel vira 'needs_review' e o pipeline SEGUE;
  - os conditionals da voz sao preparados UMA vez, antes do primeiro chunk.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .audio.process import (
    PAUSA_CAPITULO_MS,
    PAUSA_PARAGRAFO_MS,
    Segmento,
    montar_com_marcas,
    salvar_wav,
)
from .engines.base import TTSEngine
from .qa.policy import Decisao, Tentativa, deve_repetir, escolher
from .qa.verify import Verifier
from .script.models import Script
from ..core.estado import Store


def _lexicon_do_projeto() -> dict[str, str]:
    """Mesmos arquivos de lexico que o `script` usou para respelar."""
    import yaml

    from ..core.config import settings

    lex: dict[str, str] = {}
    for f in sorted(settings().lexicon_dir.glob("*.yaml")):
        lex.update(yaml.safe_load(f.read_text(encoding="utf-8")) or {})
    return lex


class Runner:
    def __init__(self, projeto: Path, script: Script, engine: TTSEngine,
                 verifier: Verifier | None = None, voice_ref: Path | None = None,
                 seed_base: int = 0, refs_por_voz: dict[str, Path] | None = None,
                 engine_factory=None, verifier_factory=None):
        self.projeto = projeto
        self.script = script
        self.engine = engine
        self.verifier = verifier
        # Fabricas para os workers extras: cada worker precisa do SEU modelo, porque
        # `set_voice` guarda estado (os conditionals) dentro do modelo.
        self.engine_factory = engine_factory
        self.verifier_factory = verifier_factory
        self.voice_ref = voice_ref
        # voice_id -> WAV de referencia. Vazio = voz embutida do modelo.
        self.refs_por_voz = refs_por_voz or {}
        self.seed_base = seed_base
        self.store = Store(projeto / "state.db")
        self.chunks_dir = projeto / "audio" / "chunks"
        self.logs_dir = projeto / "logs"
        for d in (self.chunks_dir, self.logs_dir, projeto / "audio" / "chapters",
                  projeto / "qa", projeto / "output"):
            d.mkdir(parents=True, exist_ok=True)

    # -- preparacao -----------------------------------------------------------

    def sync(self) -> tuple[int, int]:
        novos, obsoletos = self.store.sync_script(self.script)
        recuperados = self.store.reset_stale()
        return novos, obsoletos + recuperados

    # -- execucao -------------------------------------------------------------

    def run(self, chapters: list[int] | None = None, progress=None,
            workers: int = 1) -> dict:
        pendentes = self.store.pending(chapters)
        if not pendentes:
            return {"processados": 0, "ok": 0, "review": 0}

        # Processar AGRUPADO POR VOZ: preparar conditionals custa segundos, e
        # alternar voz a cada fala de diálogo pagaria esse custo milhares de vezes.
        # A ordem de narração é restaurada na montagem do capítulo, que lê do banco.
        pendentes = sorted(pendentes, key=lambda r: (r["voice_id"] or "",
                                                     r["chapter"], r["idx"]))
        workers = max(1, workers)
        if workers > 1 and self.engine_factory is None:
            workers = 1

        log_path = self.logs_dir / f"run-{int(time.time())}.jsonl"
        self._log = log_path.open("a")
        self._lock = threading.Lock()
        self._tot = {"ok": 0, "review": 0, "audio_s": 0.0, "gen_s": 0.0}

        t_parede = time.time()
        run_id = self.store.begin_run(self.engine.name, self.script.voice_id, workers)
        lotes = _repartir(pendentes, workers)
        if len(lotes) == 1:
            self._worker(lotes[0], self.engine, self.verifier, self.store, progress)
        else:
            threads = []
            for i, lote in enumerate(lotes):
                t = threading.Thread(target=self._worker_isolado,
                                     args=(i, lote, progress), daemon=False)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        self._log.close()
        tot = self._tot
        parede = time.time() - t_parede
        self.store.end_run(run_id, tot["ok"], tot["review"], tot["audio_s"],
                           tot["gen_s"])
        # RTF pelo relogio de parede, nao pela soma dos workers: com 2 workers a
        # soma conta o mesmo intervalo duas vezes e o numero perde o sentido.
        return {"processados": len(pendentes), "ok": tot["ok"], "review": tot["review"],
                "audio_s": round(tot["audio_s"], 1), "gen_s": round(tot["gen_s"], 1),
                "workers": workers, "parede_s": round(parede, 1),
                "rtf": round(parede / tot["audio_s"], 3) if tot["audio_s"] else None}

    def _worker_isolado(self, i: int, lote: list, progress) -> None:
        """Worker extra: modelo, ASR e conexao SQLite proprios.

        Conexao propria porque sqlite3 amarra a conexao a thread que a criou; modelo
        proprio porque `set_voice` guarda os conditionals DENTRO do modelo, e dois
        workers compartilhando um modelo trocariam a voz um do outro no meio do lote.
        """
        engine = self.engine_factory()
        verifier = None
        if self.verifier is not None:
            verifier = self.verifier_factory() if self.verifier_factory else self.verifier
        store = Store(self.projeto / "state.db")
        try:
            self._worker(lote, engine, verifier, store, progress, seed_offset=i * 97)
        finally:
            store.close()

    def _worker(self, pendentes: list, engine: TTSEngine, verifier: Verifier | None,
                store: Store, progress, seed_offset: int = 0) -> None:
        engine.load()
        if verifier is not None:
            verifier.load()

        voz_atual = object()
        speaker = None
        for row in pendentes:
            cid, texto = row["chunk_id"], row["text"]
            if row["voice_id"] != voz_atual:
                voz_atual = row["voice_id"]
                speaker = self._trocar_voz(voz_atual, engine)
            store.claim(cid)
            t0 = time.time()
            decisao, tentativas = self._gerar_com_qa(texto, engine, verifier,
                                                     seed_offset, speaker)
            gen_s = time.time() - t0

            wav_path = self.chunks_dir / f"{cid.replace('/', '_')}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            salvar_wav(decisao.melhor.audio, wav_path, engine.sample_rate)
            dur = len(decisao.melhor.audio) / engine.sample_rate

            sim = decisao.melhor.speaker_sim

            if decisao.aceito:
                store.finish_ok(cid, str(wav_path), dur, decisao.melhor.qa.cer,
                                decisao.melhor.qa.transcript, decisao.melhor.seed,
                                speaker_sim=sim)
            else:
                store.finish_review(cid, decisao.motivo, str(wav_path),
                                    decisao.melhor.qa.cer, decisao.melhor.qa.transcript)

            with self._lock:
                self._tot["ok" if decisao.aceito else "review"] += 1
                self._tot["audio_s"] += dur
                self._tot["gen_s"] += gen_s
                self._log.write(json.dumps({
                    "chunk_id": cid, "voz": row["voice_id"], "papel": row["role"],
                    "aceito": decisao.aceito, "tentativas": tentativas,
                    "cer": decisao.melhor.qa.cer,
                    "speaker_sim": round(sim, 4) if sim is not None else None,
                    "duracao_s": round(dur, 2),
                    "gen_s": round(gen_s, 2), "motivo": decisao.motivo,
                    "texto": texto,
                }, ensure_ascii=False) + "\n")
                self._log.flush()
                if progress:
                    progress(cid, decisao)

    def _trocar_voz(self, voice_id: str | None, engine: TTSEngine):
        """Congela os conditionals da voz do grupo e devolve a checagem de identidade."""
        ref = self.refs_por_voz.get(voice_id) if voice_id else None
        ref = ref or self.voice_ref
        if ref is None:
            return None
        engine.set_voice(ref)
        ve = getattr(engine.model, "ve", None)
        if ve is None:
            return None
        from .qa.speaker import SpeakerCheck

        return SpeakerCheck(ve, ref)

    def _gerar_com_qa(self, texto: str, engine: TTSEngine, verifier: Verifier | None,
                      seed_offset: int = 0, speaker=None) -> tuple[Decisao, int]:
        """Gera, verifica e regenera com nova seed enquanto valer a pena.

        A identidade da voz e medida AQUI, por tentativa, e nao depois de escolher:
        assim uma seed que derivou de timbre e descartada como qualquer outro
        defeito, em vez de mandar o chunk direto para revisao humana.
        """
        tentativas: list[Tentativa] = []
        for n in range(1, 4):
            seed = self.seed_base + seed_offset + n * 1000 + len(tentativas)
            audio = engine.synthesize(texto, seed=seed)
            if verifier is None:
                from .qa.verify import QAResult
                dur = len(audio) / engine.sample_rate
                qa = QAResult(True, 0.0, "", dur, len(texto) / dur if dur else 0.0)
            else:
                qa = verifier.check(audio, engine.sample_rate, texto)
            voz_ok = sim = None
            if speaker is not None:
                voz_ok, sim = speaker.ok(audio, engine.sample_rate)
            tentativas.append(Tentativa(audio, qa, seed, sim, voz_ok))
            if not deve_repetir(qa, n, voz_ok=voz_ok):
                break
        return escolher(tentativas), len(tentativas)

    # -- montagem -------------------------------------------------------------

    def build_chapter(self, chapter: int) -> Path | None:
        """Monta um capitulo a partir dos chunks 'ok'. Falta chunk -> nao monta."""
        rows = self.store.chapter_chunks(chapter)
        if not rows or any(r["state"] != "ok" for r in rows):
            return None
        import soundfile as sf

        segs = []
        for i, r in enumerate(rows):
            audio, _ = sf.read(r["wav_path"], dtype="float32")
            pausa = PAUSA_CAPITULO_MS if i == len(rows) - 1 else PAUSA_PARAGRAFO_MS
            segs.append(Segmento(audio, pausa))
        full, marcas, pausas = montar_com_marcas(segs, self.engine.sample_rate)
        destino = self.projeto / "audio" / "chapters" / f"ch{chapter:02d}.wav"
        salvar_wav(full, destino, self.engine.sample_rate)
        # A legenda sai junto com o audio, do MESMO ato de montagem: e a unica
        # forma de os tempos serem os reais. Gerada depois, por outro caminho,
        # ela descolaria da fala.
        from .legenda import desfazer_lexico, escrever_srt
        # O `text` e o respelling fonetico ("dama tchaca pavátana súta"), que
        # existe para o motor pronunciar e nunca deve ser lido por gente. Mas o
        # `source` tambem nao serve: ele guarda o PARAGRAFO inteiro, repetido em
        # cada pedaco cortado dele -- no ch01, `text` soma 7.527 caracteres e
        # `source`, 25.914. Usar `source` faz a legenda exibir texto que so sera
        # falado nos pedacos seguintes, e e assim que ela "adianta".
        lex = _lexicon_do_projeto()
        escrever_srt(destino.with_suffix(".srt"), marcas,
                     [desfazer_lexico(r["text"], lex) for r in rows], pausas)
        return destino


def _repartir(rows: list, n: int) -> list[list]:
    """Divide a fila em n lotes preservando os grupos de voz contiguos.

    Cada lote leva fatias contiguas de cada voz: assim um worker chama `set_voice`
    uma vez por voz do elenco, e nao uma vez por chunk.
    """
    if n <= 1:
        return [rows]
    lotes: list[list] = [[] for _ in range(n)]
    grupo: list = []
    voz = object()
    def despejar(g):
        if not g:
            return
        tam = (len(g) + n - 1) // n
        for i in range(n):
            lotes[i] += g[i * tam:(i + 1) * tam]
    for row in rows:
        if row["voice_id"] != voz:
            despejar(grupo)
            grupo, voz = [], row["voice_id"]
        grupo.append(row)
    despejar(grupo)
    return [l for l in lotes if l] or [rows]
