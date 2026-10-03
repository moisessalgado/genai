"""Estado da HQ no state.db do projeto: o que já foi gerado, escolhido ou aprovado.

Mesmo princípio do `Store` do audiolivro (core/estado.py): o estado É a fila,
e retomar é o padrão. Tabela própria (`hq_itens`) no mesmo arquivo, para o
motion comic poder narrar a HQ pelo audiolivro sem dois bancos por projeto.

Um item é uma peça que passa por geração + escolha:

- `elenco:<pid>`   folha-modelo do personagem (FLUX), escolhida por humano;
- `limpeza:<pid>`  a folha aprovada sem texto/selos (Qwen Edit) — a ref final;
- `quadro:<id>`    o quadro (Qwen Edit + refs), escolhido pelo QA ou por humano.

A `assinatura` resume tudo o que define a peça (prompt, refs, tamanho). Mudou
a assinatura — cena reescrita, ref trocada —, o item volta para `pending` e
perde a escolha: é o equivalente do `sync_script` preservar só o que não mudou.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ..core.estado import conectar

SCHEMA = """
CREATE TABLE IF NOT EXISTS hq_itens (
    item_id     TEXT PRIMARY KEY,
    tipo        TEXT NOT NULL,
    alvo        TEXT NOT NULL,
    assinatura  TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    escolhido   TEXT,
    por         TEXT,
    qa          TEXT,
    error       TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_hq_tipo ON hq_itens(tipo, state);
"""

TIPOS = ("elenco", "limpeza", "quadro")
# pending -> running -> ok | needs_review ; needs_review -> ok (escolha humana)
STATES = ("pending", "running", "ok", "needs_review")


def item_id(tipo: str, alvo: str | int) -> str:
    return f"{tipo}:{alvo}"


class EstadoHQ:
    def __init__(self, path: Path):
        self.path = path
        self.conn = conectar(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def sincronizar(self, tipo: str, assinaturas: dict[str, str]) -> tuple[int, int, int]:
        """Alinha os itens de um tipo com o que o roteiro pede agora.

        Novo -> `pending`; assinatura mudou -> `pending` sem escolha; sumiu do
        roteiro -> apagado. Devolve (novos, invalidados, apagados)."""
        assert tipo in TIPOS, tipo
        novos = invalidados = 0
        with self.tx() as c:
            for alvo, ass in assinaturas.items():
                iid = item_id(tipo, alvo)
                row = c.execute("SELECT assinatura FROM hq_itens WHERE item_id=?",
                                (iid,)).fetchone()
                if row is None:
                    c.execute("INSERT INTO hq_itens(item_id, tipo, alvo, assinatura) "
                              "VALUES (?,?,?,?)", (iid, tipo, str(alvo), ass))
                    novos += 1
                elif row["assinatura"] != ass:
                    c.execute("UPDATE hq_itens SET assinatura=?, state='pending', "
                              "attempts=0, escolhido=NULL, por=NULL, qa=NULL, error=NULL, "
                              "updated_at=CURRENT_TIMESTAMP WHERE item_id=?", (ass, iid))
                    invalidados += 1
            alvos = [str(a) for a in assinaturas]
            marks = ",".join("?" * len(alvos)) or "''"
            apagados = c.execute(f"DELETE FROM hq_itens WHERE tipo=? AND alvo NOT IN ({marks})",
                                 [tipo, *alvos]).rowcount
        return novos, invalidados, apagados

    def reset_stale(self) -> int:
        """Itens presos em 'running' (kill -9) voltam para a fila."""
        with self.tx() as c:
            return c.execute("UPDATE hq_itens SET state='pending' "
                             "WHERE state='running'").rowcount

    def item(self, tipo: str, alvo: str | int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM hq_itens WHERE item_id=?",
                                 (item_id(tipo, alvo),)).fetchone()

    def itens(self, tipo: str, states: tuple[str, ...] | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM hq_itens WHERE tipo=?", [tipo]
        if states:
            q += f" AND state IN ({','.join('?' * len(states))})"
            args += list(states)
        rows = self.conn.execute(q, args).fetchall()
        # alvo numérico (quadros) em ordem numérica, não lexical
        return sorted(rows, key=lambda r: (not r["alvo"].isdigit(),
                                           int(r["alvo"]) if r["alvo"].isdigit() else 0,
                                           r["alvo"]))

    def iniciar(self, tipo: str, alvo: str | int) -> None:
        with self.tx() as c:
            c.execute("UPDATE hq_itens SET state='running', attempts=attempts+1, "
                      "updated_at=CURRENT_TIMESTAMP WHERE item_id=?", (item_id(tipo, alvo),))

    def concluir(self, tipo: str, alvo: str | int, escolhido: Path | str, *,
                 por: str = "qa", qa: dict | None = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE hq_itens SET state='ok', escolhido=?, por=?, "
                      "qa=COALESCE(?, qa), error=NULL, updated_at=CURRENT_TIMESTAMP "
                      "WHERE item_id=?",
                      (str(escolhido), por, json.dumps(qa) if qa is not None else None,
                       item_id(tipo, alvo)))

    def revisar(self, tipo: str, alvo: str | int, motivo: str, *,
                qa: dict | None = None) -> None:
        """Precisa de olho humano: candidatos prontos para escolher, ou o QA
        reprovou todos."""
        with self.tx() as c:
            c.execute("UPDATE hq_itens SET state='needs_review', error=?, "
                      "qa=COALESCE(?, qa), updated_at=CURRENT_TIMESTAMP WHERE item_id=?",
                      (motivo, json.dumps(qa) if qa is not None else None,
                       item_id(tipo, alvo)))

    def reabrir(self, tipo: str, alvo: str | int) -> None:
        """Volta para a fila (para gerar mais candidatos), sem zerar tentativas."""
        with self.tx() as c:
            c.execute("UPDATE hq_itens SET state='pending', escolhido=NULL, por=NULL "
                      "WHERE item_id=?", (item_id(tipo, alvo),))

    def escolhido(self, tipo: str, alvo: str | int) -> Path | None:
        r = self.item(tipo, alvo)
        return Path(r["escolhido"]) if r and r["state"] == "ok" and r["escolhido"] else None

    def resumo(self) -> dict[str, dict[str, int]]:
        out = {t: {s: 0 for s in STATES} for t in TIPOS}
        for r in self.conn.execute("SELECT tipo, state, COUNT(*) n FROM hq_itens "
                                   "GROUP BY tipo, state"):
            out[r["tipo"]][r["state"]] = r["n"]
        return out

    def close(self) -> None:
        self.conn.close()
