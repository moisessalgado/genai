"""Roteiro da HQ: o schema validado que todas as etapas leem.

O arquivo é `projects/<slug>/roteiro.yaml`, gerado pelo LLM (`roteiro_llm.py`)
ou escrito à mão — e sempre editável: o LLM rascunha, o operador corrige, e o
que vale é o que passa por aqui.

Convenções que o resto do pipeline assume, e que por isso são validadas, não
só documentadas:

- `cena` e `ficha` em INGLÊS (o FLUX e o Qwen entendem melhor); `texto` dos
  balões no idioma da HQ;
- `personagens` de um quadro na ordem da ESQUERDA para a DIREITA: o prompt
  pede essa disposição e o letreiramento aponta o rabicho por ela;
- no máximo 2 personagens com folha-modelo por quadro (o InvokeAI aceita uma
  referência latente só; figurantes vão descritos na `cena`);
- quem fala está no quadro.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ARQUIVO = "roteiro.yaml"

# Resolução de geração (múltiplos de 16, ~1 MP como o Qwen e o FLUX preferem).
# A proporção define o quadro na página; o layout absorve diferenças pequenas.
Proporcao = Literal["panoramico", "largo", "quadrado", "alto"]
TAMANHOS: dict[str, tuple[int, int]] = {
    "panoramico": (1536, 672),
    "largo": (1216, 832),
    "quadrado": (1024, 1024),
    "alto": (768, 1024),
}

MAX_PERSONAGENS_QUADRO = 2
MAX_QUADROS_TIRA = 3
# Acima disto o balão come o quadro: melhor quebrar a fala em dois.
MAX_CHARS_TEXTO = 160

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class Personagem(BaseModel):
    nome: str
    ficha: str = Field(description="descrição visual fixa, em inglês")
    voz: str | None = Field(None, description="voice_id do motion comic")


class Texto(BaseModel):
    tipo: Literal["recordatorio", "fala", "pensamento"]
    quem: str | None = None
    texto: str

    @model_validator(mode="after")
    def _quem(self):
        if self.tipo == "recordatorio" and self.quem:
            raise ValueError("recordatório é do narrador: não leva 'quem'")
        if self.tipo != "recordatorio" and not self.quem:
            raise ValueError(f"{self.tipo} precisa de 'quem'")
        if len(self.texto) > MAX_CHARS_TEXTO:
            raise ValueError(f"texto com {len(self.texto)} caracteres (máx. "
                             f"{MAX_CHARS_TEXTO}): divida em dois balões")
        return self


class Quadro(BaseModel):
    id: int = Field(ge=1)
    proporcao: Proporcao = "largo"
    personagens: list[str] = Field(default_factory=list,
                                   description="da esquerda para a direita")
    cena: str = Field(description="o que se vê, em inglês")
    textos: list[Texto] = Field(default_factory=list)

    @property
    def tamanho(self) -> tuple[int, int]:
        return TAMANHOS[self.proporcao]

    @model_validator(mode="after")
    def _coerente(self):
        if len(self.personagens) > MAX_PERSONAGENS_QUADRO:
            raise ValueError(f"quadro {self.id}: {len(self.personagens)} personagens "
                             f"(máx. {MAX_PERSONAGENS_QUADRO}; figurantes vão na cena)")
        if len(set(self.personagens)) != len(self.personagens):
            raise ValueError(f"quadro {self.id}: personagem repetido")
        for t in self.textos:
            if t.quem and t.quem not in self.personagens:
                raise ValueError(f"quadro {self.id}: '{t.quem}' fala mas não está no quadro "
                                 f"({', '.join(self.personagens) or 'ninguém'})")
        return self


class Pagina(BaseModel):
    tiras: list[list[int]] = Field(min_length=1)

    @field_validator("tiras")
    @classmethod
    def _tiras(cls, v):
        for t in v:
            if not 1 <= len(t) <= MAX_QUADROS_TIRA:
                raise ValueError(f"tira com {len(t)} quadros (1 a {MAX_QUADROS_TIRA})")
        return v

    @property
    def quadros(self) -> list[int]:
        return [q for t in self.tiras for q in t]


class Formato(BaseModel):
    """Página impressa. Padrão: A4 a 300 dpi."""
    largura: int = 2480
    altura: int = 3508
    margem: int = 120
    calha: int = 40
    dpi: int = 300


class Roteiro(BaseModel):
    slug: str
    titulo: str
    idioma: str = "pt-BR"
    estilo: str = Field(description="bloco de estilo repetido em todo prompt, em inglês")
    # Frase curta (inglês) que descreve o estilo para o CLIP do QA; a nota é a
    # similaridade a ela menos a similaridade a "HQ moderna" (hq/qa.py). Sem
    # âncora, o QA não julga estilo.
    ancora_estilo: str | None = None
    # Gravura de referência (caminho relativo ao projeto) para a etapa de
    # style transfer dos quadros (hq/quadros.py). Sem ela, a etapa não roda.
    estampa: str | None = None
    personagens: dict[str, Personagem]
    quadros: list[Quadro] = Field(min_length=1)
    paginas: list[Pagina] = Field(min_length=1)
    formato: Formato = Field(default_factory=Formato)

    @model_validator(mode="after")
    def _referencias(self):
        for pid in self.personagens:
            if not _ID.match(pid):
                raise ValueError(f"id de personagem inválido: {pid!r} (minúsculas, "
                                 "dígitos e hífen)")
        ids = [q.id for q in self.quadros]
        if len(set(ids)) != len(ids):
            raise ValueError("ids de quadro repetidos")
        for q in self.quadros:
            for p in q.personagens:
                if p not in self.personagens:
                    raise ValueError(f"quadro {q.id}: personagem desconhecido {p!r} "
                                     f"(elenco: {', '.join(self.personagens)})")
        nas_paginas = [i for p in self.paginas for i in p.quadros]
        if sorted(nas_paginas) != sorted(ids):
            faltam = set(ids) - set(nas_paginas)
            sobram = set(nas_paginas) - set(ids)
            repetidos = {i for i in nas_paginas if nas_paginas.count(i) > 1}
            raise ValueError("cada quadro deve estar em exatamente uma tira: "
                             + "; ".join(f"{k}: {sorted(v)}" for k, v in
                                         (("fora das páginas", faltam),
                                          ("inexistentes", sobram),
                                          ("repetidos", repetidos)) if v))
        return self

    def quadro(self, qid: int) -> Quadro:
        return next(q for q in self.quadros if q.id == qid)

    def ordem_de_leitura(self) -> list[Quadro]:
        return [self.quadro(i) for p in self.paginas for i in p.quadros]


def carregar(proj: Path) -> Roteiro:
    arq = proj / ARQUIVO
    if not arq.exists():
        raise FileNotFoundError(f"sem {ARQUIVO} em {proj} — rode `genai hq roteiro` "
                                "ou escreva um à mão")
    return Roteiro.model_validate(yaml.safe_load(arq.read_text(encoding="utf-8")))


def salvar(roteiro: Roteiro, proj: Path) -> Path:
    arq = proj / ARQUIVO
    dados = roteiro.model_dump(mode="json", exclude_defaults=False)
    arq.write_text(yaml.safe_dump(dados, allow_unicode=True, sort_keys=False, width=88),
                   encoding="utf-8")
    return arq
