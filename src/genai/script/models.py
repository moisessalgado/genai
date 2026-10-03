"""Documento canonico do projeto (script.json).

Cada segmento carrega um id derivado de hash do conteudo + parametros de sintese.
Esse id e a chave de cache, de retomada e de deduplicacao: mudar o texto, a voz ou
os parametros muda o id, e so o que mudou e regerado.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field


class SynthParams(BaseModel):
    """Parametros de sintese. Congelados por perfil de voz (TDD 7.1)."""

    engine: str = "chatterbox"
    language_id: str = "pt"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 2.0
    seed: int = 0

    def fingerprint(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)


class Segment(BaseModel):
    """Menor unidade sintetizavel. `text` e o que vai ao TTS; `source` e o original."""

    idx: int
    source: str
    text: str
    kind: str = "prose"  # prose | heading | quote | dialogue
    role: str = "narrador"  # papel; o cast do script mapeia papel -> voice_id
    pause_after_ms: int = 0

    def chunk_id(self, chapter_idx: int, voice_id: str, params: SynthParams,
                 engine_version: str) -> str:
        payload = "\x00".join([
            self.text, voice_id, params.fingerprint(), engine_version,
        ])
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"ch{chapter_idx:02d}/{self.idx:05d}-{digest}"


class Chapter(BaseModel):
    idx: int
    title: str
    segments: list[Segment] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.segments)


class Rights(BaseModel):
    """Procedencia do texto. Sem isso o export e bloqueado (TDD 14.3)."""

    status: str  # dominio-publico | proprio | licenciado
    autor: str | None = None
    ano_morte: int | None = None
    tradutor: str | None = None
    fonte: str | None = None
    # Termos da licenca, nas palavras de quem publicou o texto. Guardados na
    # integra e nao resumidos: uma licenca de distribuicao gratuita costuma ter
    # condicoes ("desde que nenhum custo seja cobrado") que precisam viajar
    # junto com a obra derivada -- e a descricao do video as reproduz.
    licenca: str | None = None
    verificado_em: str | None = None


class Script(BaseModel):
    title: str
    author: str | None = None
    voice_id: str  # voz do papel "narrador"
    # cast: papel -> voice_id. Papel ausente cai no narrador.
    cast: dict[str, str] = Field(default_factory=dict)
    params: SynthParams = Field(default_factory=SynthParams)
    engine_version: str = "chatterbox-mtl-v3"
    rights: Rights | None = None
    chapters: list[Chapter] = Field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(c.char_count for c in self.chapters)

    def voice_of(self, seg: "Segment") -> str:
        """Voz efetiva de um segmento. Papel sem voz declarada usa a do narrador."""
        return self.cast.get(seg.role, self.voice_id)

    def iter_segments(self):
        """Rende (chapter, segment, chunk_id) na ordem de narracao.

        O chunk_id ja embute a voz: trocar a voz de um papel invalida so os chunks
        daquele papel, e o resto do livro continua valendo em cache.
        """
        for ch in self.chapters:
            for seg in ch.segments:
                yield ch, seg, seg.chunk_id(ch.idx, self.voice_of(seg), self.params,
                                            self.engine_version)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Script":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
