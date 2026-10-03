"""O InvokeAI visto pela HQ: cliente conferido e refs enviadas só durante o uso."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from ..core.config import settings
from ..core.servicos import invokeai


def cliente() -> invokeai.InvokeAI:
    c = invokeai.cliente()
    if not c.disponivel():
        raise RuntimeError(f"InvokeAI fora do ar em {settings().invokeai_url} — suba o "
                           "serviço (ai-stack) ou ajuste `invokeai_url` no vf.toml")
    return c


@contextmanager
def refs_enviadas(c: invokeai.InvokeAI, caminhos: list[Path]):
    """Sobe as referências e devolve {caminho: image_name}; apaga ao sair.

    O disco do projeto é a fonte da verdade (como no `gerar_lote`): a galeria
    do InvokeAI não precisa acumular uma cópia de cada ref a cada execução."""
    nomes: dict[Path, str] = {}
    try:
        for p in dict.fromkeys(caminhos):
            nomes[p] = c.upload(p)
        yield nomes
    finally:
        for n in nomes.values():
            try:
                c.apagar(n)
            except invokeai.InvokeAIErro:
                pass
