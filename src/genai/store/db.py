"""Fila e estado por chunk em SQLite (TDD 8.1).

O estado E a fila: nao existe arquivo de checkpoint separado, entao nao ha o que
dessincronizar. `run` sempre processa o que nao esta 'ok' -- retomar e o padrao.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    chapter     INTEGER NOT NULL,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'narrador',
    voice_id    TEXT,
    state       TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    seed        INTEGER,
    wav_path    TEXT,
    duration_s  REAL,
    cer         REAL,
    speaker_sim REAL,
    transcript  TEXT,
    error       TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_state ON chunks(state);
CREATE INDEX IF NOT EXISTS idx_order ON chunks(chapter, idx);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    engine      TEXT,
    voice_id    TEXT,
    n_ok        INTEGER DEFAULT 0,
    n_failed    INTEGER DEFAULT 0,
    audio_s     REAL DEFAULT 0,
    gen_s       REAL DEFAULT 0
);
"""

# pending -> running -> ok | needs_review
STATES = ("pending", "running", "ok", "needs_review")


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        # migracao para bancos criados antes da checagem de identidade de voz
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(chunks)")}
        if "speaker_sim" not in cols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN speaker_sim REAL")
        if "role" not in cols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN role TEXT "
                              "NOT NULL DEFAULT 'narrador'")
        if "voice_id" not in cols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN voice_id TEXT")
        self.conn.commit()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def sync_script(self, script) -> tuple[int, int]:
        """Insere os chunks do script. Chunks ja 'ok' com o mesmo id sao preservados
        -- e isso que faz corrigir um capitulo nao regerar o livro inteiro."""
        novos = 0
        ids = []
        with self.tx() as c:
            for ch, seg, cid in script.iter_segments():
                ids.append(cid)
                cur = c.execute(
                    "INSERT OR IGNORE INTO chunks(chunk_id, chapter, idx, text, source, "
                    "role, voice_id) VALUES (?,?,?,?,?,?,?)",
                    (cid, ch.idx, seg.idx, seg.text, seg.source, seg.role,
                     script.voice_of(seg)))
                novos += cur.rowcount
            # Remove chunks orfaos (texto mudou -> id mudou)
            marks = ",".join("?" * len(ids))
            obsoletos = c.execute(
                f"DELETE FROM chunks WHERE chunk_id NOT IN ({marks})", ids).rowcount
        return novos, obsoletos

    def reset_stale(self) -> int:
        """Chunks presos em 'running' (kill -9) voltam para a fila."""
        with self.tx() as c:
            return c.execute(
                "UPDATE chunks SET state='pending' WHERE state='running'").rowcount

    def pending(self, chapters: list[int] | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM chunks WHERE state != 'ok'"
        args: list = []
        if chapters:
            q += f" AND chapter IN ({','.join('?' * len(chapters))})"
            args += chapters
        return self.conn.execute(q + " ORDER BY chapter, idx", args).fetchall()

    def claim(self, chunk_id: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE chunks SET state='running', attempts=attempts+1, "
                      "updated_at=CURRENT_TIMESTAMP WHERE chunk_id=?", (chunk_id,))

    def finish_ok(self, chunk_id: str, wav_path: str, duration_s: float,
                  cer: float | None, transcript: str | None, seed: int,
                  speaker_sim: float | None = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE chunks SET state='ok', wav_path=?, duration_s=?, cer=?, "
                      "speaker_sim=?, transcript=?, seed=?, error=NULL, "
                      "updated_at=CURRENT_TIMESTAMP WHERE chunk_id=?",
                      (wav_path, duration_s, cer, speaker_sim, transcript, seed, chunk_id))

    def finish_review(self, chunk_id: str, error: str, wav_path: str | None = None,
                      cer: float | None = None, transcript: str | None = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE chunks SET state='needs_review', error=?, wav_path=?, "
                      "cer=?, transcript=?, updated_at=CURRENT_TIMESTAMP WHERE chunk_id=?",
                      (error, wav_path, cer, transcript, chunk_id))

    def approve(self, chunk_id: str) -> int:
        """Aprova um chunk que o QA reprovou, depois de um humano ouvir.

        Existe porque o QA automatico tem limite conhecido: nome proprio incomum
        derruba o CER pela grafia do ASR, nao pela pronuncia. Sem esta porta, um
        trecho de 2 s trava a montagem de um capitulo inteiro.
        """
        with self.tx() as c:
            return c.execute(
                "UPDATE chunks SET state='ok', error='aprovado por revisão humana', "
                "updated_at=CURRENT_TIMESTAMP WHERE chunk_id=? AND state='needs_review'",
                (chunk_id,)).rowcount

    def requeue(self, chunk_id: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE chunks SET state='pending', attempts=0 WHERE chunk_id=?",
                      (chunk_id,))

    # -- historico de execucoes ------------------------------------------------

    def begin_run(self, engine: str, voice_id: str | None, workers: int = 1) -> int:
        """Abre um registro de execucao. O relogio de parede daqui e o RTF real."""
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO runs(engine, voice_id, n_ok, n_failed) VALUES (?,?,0,0)",
                (f"{engine} x{workers}", voice_id))
            return cur.lastrowid

    def end_run(self, run_id: int, n_ok: int, n_failed: int, audio_s: float,
                gen_s: float) -> None:
        with self.tx() as c:
            c.execute("UPDATE runs SET finished_at=CURRENT_TIMESTAMP, n_ok=?, "
                      "n_failed=?, audio_s=?, gen_s=? WHERE id=?",
                      (n_ok, n_failed, audio_s, gen_s, run_id))

    def runs(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT *, (julianday(finished_at)-julianday(started_at))*86400 AS parede_s "
            "FROM runs WHERE finished_at IS NOT NULL ORDER BY id").fetchall()

    def stats(self) -> dict:
        rows = self.conn.execute(
            "SELECT state, COUNT(*) n, COALESCE(SUM(duration_s),0) d "
            "FROM chunks GROUP BY state").fetchall()
        out = {s: 0 for s in STATES}
        audio = 0.0
        for r in rows:
            out[r["state"]] = r["n"]
            audio += r["d"]
        out["total"] = sum(out[s] for s in STATES)
        out["audio_s"] = audio
        cer = self.conn.execute(
            "SELECT AVG(cer) a FROM chunks WHERE cer IS NOT NULL").fetchone()["a"]
        out["cer_medio"] = cer
        r = self.conn.execute("SELECT AVG(speaker_sim) a, MIN(speaker_sim) m FROM chunks "
                              "WHERE speaker_sim IS NOT NULL").fetchone()
        out["speaker_medio"], out["speaker_min"] = r["a"], r["m"]
        return out

    def chapter_chunks(self, chapter: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE chapter=? ORDER BY idx", (chapter,)).fetchall()

    def chapters(self) -> list[int]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT chapter FROM chunks ORDER BY chapter")]

    def needs_review(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE state='needs_review' ORDER BY chapter, idx"
        ).fetchall()

    def close(self) -> None:
        self.conn.close()
