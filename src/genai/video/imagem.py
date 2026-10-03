"""Geração de imagens (FLUX.1-schnell, via InvokeAI) para o acervo de slides.

A geração roda no InvokeAI da máquina (mantido pelo ai-stack), por HTTP —
`servicos/invokeai.py`. Antes era uma venv de diffusers isolada atrás de um
subprocesso (`.venv-imagem`), com o FLUX em bf16 e offload sequencial; o
InvokeAI serve o mesmo modelo quantizado (NF4), já carregado, com fila.

Uma diferença importa em relação à música: a paleta inteira da trilha vira
acervo automaticamente, mas nem toda imagem gerada presta — precisa de
curadoria humana antes de entrar em `assets/slides/`. Por isso este módulo
separa duas etapas: `gerar()` produz candidatos descartáveis em
`cache/imagens/` (custa só GPU para regerar), e `aprovar()` converte os
escolhidos pelo operador para o formato do acervo (JPEG q2, ≤1920px, mesma
faixa já usada nas imagens do Midjourney) e move para `assets/slides/`.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from ..config import settings
from ..servicos import invokeai
from . import imagem_qa
from .slides import DIRETORIO_PADRAO

# Mesma divisão de `assets/musica` vs `cache/musica`: o que pode ser apagado
# sem consequência (rascunho, custa só GPU) de um lado, o acervo do canal do
# outro. `ACERVO` é `assets/slides/`, que já existe — a arte gerada localmente
# entra no mesmo lugar que a do Midjourney.
CACHE = settings().cache_dir / "imagens"
ACERVO = DIRETORIO_PADRAO

# Apelido na linha de comando -> modelo registrado no InvokeAI.
#
# FLUX.1-**schnell**, não o `dev`: schnell é Apache-2.0, dev é licença
# não-comercial da Black Forest Labs — decisão do operador, ver LICENSES.md.
# O SD3.5 (`sd`, `sd:large`) saiu junto com a venv de diffusers: os pesos nunca
# chegaram a ser baixados nesta máquina. Para voltar, instale-o no InvokeAI e
# acrescente o grafo em `servicos/invokeai.py`.
MODELOS: dict[str, str] = {
    "flux": invokeai.FLUX_SCHNELL,
}

# FLUX-schnell é destilado por passo-de-tempo: poucos passos bastam, e
# guidance diferente de 0 não faz CFG nenhum (a rede não foi treinada para
# usar) — só desperdiça tempo.
PASSOS_PADRAO = 4
GUIDANCE_PADRAO = 0.0

# Resolução ~16:9 (múltiplo de 16, como o FLUX exige) em vez de quadrada:
# mais perto do formato final do vídeo do que o acervo atual do Midjourney,
# ainda dentro do bucket de resolução em que o modelo foi treinado.
# `slides.py` já sabe preencher com blur o que sobrar, então nada quebra para
# quem preferir gerar quadrado.
LARGURA_PADRAO = 1344
ALTURA_PADRAO = 768

MAX_LADO_ACERVO = 1920
Q_JPEG_ACERVO = "2"


def disponivel() -> bool:
    """O InvokeAI responde? Sem ele, `imagem` não tem como rodar."""
    return invokeai.cliente().disponivel()


def resolver_modelo(modelo: str) -> str:
    if modelo not in MODELOS:
        raise ValueError(f"modelo desconhecido: {modelo} — use {', '.join(MODELOS)}")
    return MODELOS[modelo]


def _defaults(passos: int | None, guidance: float | None) -> tuple[int, float]:
    return (passos if passos is not None else PASSOS_PADRAO,
            guidance if guidance is not None else GUIDANCE_PADRAO)


def _seed(prompt: str, i: int) -> int:
    """Seed derivada do prompt: o mesmo prompt no mesmo índice rende sempre a
    mesma imagem — permite reproduzir um candidato aprovado se o PNG se
    perder, sem depender de guardar a seed em outro lugar."""
    h = hashlib.sha256(f"{prompt}:{i}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _erro_indisponivel() -> RuntimeError:
    return RuntimeError(f"InvokeAI fora do ar em {settings().invokeai_url} — "
                        "suba o serviço (ai-stack) ou ajuste `invokeai_url` no vf.toml")


def gerar(prompt: str, modelo: str = "flux", n: int = 4,
          largura: int = LARGURA_PADRAO, altura: int = ALTURA_PADRAO,
          passos: int | None = None, guidance: float | None = None,
          progresso=None) -> list[Path]:
    """Gera (ou reaproveita) `n` candidatos para `prompt`, em `CACHE`.

    O nome do arquivo carrega o hash do prompt e a seed: reescrever o prompt
    não reaproveita silenciosamente um arquivo antigo com o nome antigo — a
    mesma armadilha já documentada em `musica_ace.gerar_pecas`.
    """
    if not disponivel():
        raise _erro_indisponivel()
    resolver_modelo(modelo)
    passos, guidance = _defaults(passos, guidance)
    CACHE.mkdir(parents=True, exist_ok=True)

    marca = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    prefixo = modelo.replace(":", "-")
    destinos = [CACHE / f"{prefixo}-{marca}-{_seed(prompt, i)}.png" for i in range(n)]

    pendentes = [{"prompt": prompt, "seed": _seed(prompt, i), "destino": str(destinos[i]),
                  "largura": largura, "altura": altura, "passos": passos, "guidance": guidance}
                 for i in range(n) if not destinos[i].exists()]
    if pendentes:
        if progresso:
            progresso(f"gerando {len(pendentes)} imagem(ns) com {modelo}…")
        gerar_lote(pendentes)
    return destinos


def gerar_lote(pedidos: list[dict], *, avaliar_clip: bool = False,
               avaliar_estilo: bool = False, progresso=None
               ) -> tuple[dict[str, float], dict[str, float]]:
    """Gera cada pedido (`prompt`, `seed`, `destino`, `largura`, `altura`,
    `passos`, `guidance`) que ainda não existe em disco, pelo InvokeAI.

    `avaliar_clip` pede uma nota de similaridade texto-imagem por pedido
    (usada só para desempate entre candidatos da mesma janela, ver
    `video/sincronizado.py`); `avaliar_estilo` pede a nota foto-vs-ilustração
    usada para rejeitar candidatos fotorrealistas demais
    (`imagem_qa.eh_fotorealista`). Devolve `(notas_relevancia, notas_estilo)`,
    chaveadas pelo `destino` em texto, cada uma vazia se não pedida."""
    if not disponivel():
        raise _erro_indisponivel()
    c = invokeai.cliente()
    grafos = [(c.grafo_flux(p["prompt"], p["largura"], p["altura"], p["seed"],
                            passos=p["passos"], guidance=p["guidance"]), Path(p["destino"]))
              for p in pedidos if not Path(p["destino"]).exists()]
    if grafos:
        c.gerar_lote(grafos, progresso=progresso)
    if not (avaliar_clip or avaliar_estilo):
        return {}, {}
    return imagem_qa.pontuar([(Path(p["destino"]), p["prompt"]) for p in pedidos],
                             relevancia=avaliar_clip, estilo=avaliar_estilo)


def aprovar(arquivos: list[Path], slides_dir: Path = ACERVO) -> list[Path]:
    """Converte os candidatos escolhidos para o formato do acervo e move para lá.

    Confere que todos os arquivos existem ANTES de converter qualquer um — um
    caminho errado no meio de um lote de dez não pode deixar meia curadoria
    feita. A conversão (JPEG q2, maior lado ≤1920px) segue a mesma faixa já
    usada nas 52 imagens do Midjourney (ver ESTADO.md), para o acervo não
    ganhar dois padrões de tamanho/qualidade dependendo da origem.
    """
    arquivos = [Path(a) for a in arquivos]
    faltando = [a for a in arquivos if not a.exists()]
    if faltando:
        raise FileNotFoundError(
            f"arquivo(s) não encontrado(s): {', '.join(str(a) for a in faltando)}")
    slides_dir.mkdir(parents=True, exist_ok=True)

    finais = []
    for a in arquivos:
        destino = slides_dir / f"{a.stem}.jpg"
        escala = (f"scale='min({MAX_LADO_ACERVO},iw)':'min({MAX_LADO_ACERVO},ih)':"
                  f"force_original_aspect_ratio=decrease")
        _ffmpeg(["-i", str(a), "-vf", escala, "-q:v", Q_JPEG_ACERVO, str(destino)])
        # O PNG em cache é descartável e já virou o JPEG do acervo — mantê-lo
        # só duplicaria o disco sem função (diferente da música, que guarda o
        # bruto de 48 kHz porque um sample_rate diferente ainda o reaproveita).
        a.unlink()
        finais.append(destino)
    return finais


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                   capture_output=True, text=True, check=True)
