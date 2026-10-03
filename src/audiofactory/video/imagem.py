"""Geração local de imagens (FLUX + Stable Diffusion) para o acervo de slides.

Mesmo desenho de `audio/musica_ace.py`, e pelo mesmo motivo: os pesos de imagem
não convivem com as versões de `torch`/`transformers` fixadas pelo resto do
pipeline, então o modelo roda numa venv isolada (`.venv-imagem`) atrás de uma
ponte por subprocesso que troca só JSON e arquivo em disco
(`_imagem_runner.py`) — nenhum objeto Python atravessa a fronteira.

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
import json
import subprocess
import tempfile
from pathlib import Path

from ..project import RAIZ
from .slides import DIRETORIO_PADRAO

VENV = RAIZ / ".venv-imagem"

# Mesma divisão de `assets/musica` vs `cache/musica`: o que pode ser apagado
# sem consequência (rascunho, custa só GPU) de um lado, o acervo do canal do
# outro. `ACERVO` é `assets/slides/`, que já existe — a arte gerada localmente
# entra no mesmo lugar que a do Midjourney.
CACHE = RAIZ / "cache" / "imagens"
ACERVO = DIRETORIO_PADRAO

# "flux" e "sd" (SD3.5-Medium, cabe em 16 GB sem offload) cobrem o uso comum;
# "sd:large" abre para SD3.5-Large (mais VRAM, precisa de model_cpu_offload) —
# mesmo padrão de sufixo que `musica_ace` usa para escolher paleta ("ace:sobrio").
#
# FLUX.1-**schnell**, não o `dev`: schnell é Apache-2.0, dev é licença
# não-comercial da Black Forest Labs — decisão do operador, ver LICENSES.md.
MODELOS: dict[str, tuple[str, str]] = {
    "flux": ("black-forest-labs/FLUX.1-schnell", "flux"),
    "sd": ("stabilityai/stable-diffusion-3.5-medium", "sd3"),
    "sd:large": ("stabilityai/stable-diffusion-3.5-large", "sd3"),
}

# Passos e guidance por FAMÍLIA, não por modelo: as duas famílias são
# incompatíveis entre si. FLUX-schnell é destilado por passo-de-tempo — poucos
# passos bastam, e `guidance_scale` diferente de 0 não faz CFG nenhum, só
# desperdiça tempo (a rede não foi treinada para usar). SD3.5 é convencional:
# precisa de mais passos e de CFG de verdade para não sair borrado/genérico.
_PADRAO_FAMILIA = {
    "flux": (4, 0.0),
    "sd3": (28, 4.5),
}

# Resolução ~16:9 (múltiplo de 16, como os dois modelos exigem) em vez de
# quadrada: mais perto do formato final do vídeo do que o acervo atual do
# Midjourney, ainda dentro do bucket de resolução em que os dois foram
# treinados. `slides.py` já sabe preencher com blur o que sobrar, então nada
# quebra para quem preferir gerar quadrado.
LARGURA_PADRAO = 1344
ALTURA_PADRAO = 768

MAX_LADO_ACERVO = 1920
Q_JPEG_ACERVO = "2"


def disponivel() -> bool:
    """A venv isolada existe? Sem ela, `imagem` não tem como rodar."""
    return (VENV / "bin" / "python").exists()


def resolver_modelo(modelo: str) -> tuple[str, str]:
    if modelo not in MODELOS:
        raise ValueError(f"modelo desconhecido: {modelo} — use {', '.join(MODELOS)}")
    return MODELOS[modelo]


def _defaults(familia: str, passos: int | None, guidance: float | None) -> tuple[int, float]:
    p, g = _PADRAO_FAMILIA[familia]
    return (passos if passos is not None else p, guidance if guidance is not None else g)


def _seed(prompt: str, i: int) -> int:
    """Seed derivada do prompt: o mesmo prompt no mesmo índice rende sempre a
    mesma imagem — permite reproduzir um candidato aprovado se o PNG se
    perder, sem depender de guardar a seed em outro lugar."""
    h = hashlib.sha256(f"{prompt}:{i}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def gerar(prompt: str, modelo: str = "flux", n: int = 4,
          largura: int = LARGURA_PADRAO, altura: int = ALTURA_PADRAO,
          passos: int | None = None, guidance: float | None = None,
          negative: str | None = None,
          progresso=None) -> list[Path]:
    """Gera (ou reaproveita) `n` candidatos para `prompt`, em `CACHE`.

    O nome do arquivo carrega o hash do prompt e a seed: reescrever o prompt
    não reaproveita silenciosamente um arquivo antigo com o nome antigo — a
    mesma armadilha já documentada em `musica_ace.gerar_pecas`.

    `negative` só tem efeito nos modelos "sd"/"sd:large" (CFG de verdade,
    guidance > 0) — FLUX-schnell não faz CFG (`guidance=0`), então um
    negative_prompt aí seria só decoração; o runner descarta silenciosamente
    nesse caso.
    """
    if not disponivel():
        raise RuntimeError(
            f"venv de imagem ausente em {VENV} — rode `uv venv --python 3.12 "
            f".venv-imagem && uv pip install --python .venv-imagem/bin/python "
            f"diffusers transformers accelerate sentencepiece protobuf torch "
            f"torchvision --index-url https://download.pytorch.org/whl/cu130`")
    repo, familia = resolver_modelo(modelo)
    passos, guidance = _defaults(familia, passos, guidance)
    CACHE.mkdir(parents=True, exist_ok=True)

    marca = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    prefixo = modelo.replace(":", "-")
    destinos = [CACHE / f"{prefixo}-{marca}-{_seed(prompt, i)}.png" for i in range(n)]

    pendentes = [{"prompt": prompt, "seed": _seed(prompt, i), "destino": str(destinos[i]),
                  "largura": largura, "altura": altura, "passos": passos, "guidance": guidance,
                  "negative_prompt": negative}
                 for i in range(n) if not destinos[i].exists()]
    if pendentes:
        if progresso:
            progresso(f"gerando {len(pendentes)} imagem(ns) com {modelo}…")
        _rodar(pendentes, repo, familia)
    return destinos


def _rodar(pedidos: list[dict], modelo_repo: str, familia: str,
          avaliar_clip: bool = False, avaliar_estilo: bool = False
          ) -> tuple[dict[str, float], dict[str, float]]:
    """Roda o lote inteiro numa unica invocacao da venv isolada — o modelo
    carrega uma vez so, nao uma vez por prompt. `avaliar_clip` pede uma nota
    de similaridade texto-imagem por pedido (usada so para desempate entre
    candidatos da mesma janela, ver `video/sincronizado.py`); `avaliar_estilo`
    pede a nota foto-vs-ilustracao usada para rejeitar candidatos
    fotorrealistas demais (`imagem_qa.eh_fotorealista`). Devolve
    `(notas_relevancia, notas_estilo)`, cada uma vazia se nao pedida."""
    pedido = {"pedidos": pedidos, "modelo_repo": modelo_repo, "familia": familia,
              "hf_home": str(RAIZ / "models"), "avaliar_clip": avaliar_clip,
              "avaliar_estilo": avaliar_estilo}
    runner = Path(__file__).with_name("_imagem_runner.py")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as tf:
        json.dump(pedido, tf)
        arquivo_pedido = Path(tf.name)
    try:
        r = subprocess.run([str(VENV / "bin" / "python"), str(runner), str(arquivo_pedido)],
                           capture_output=True, text=True)
    finally:
        arquivo_pedido.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"geração de imagem falhou:\n{r.stderr[-2000:]}")
    for linha in reversed(r.stdout.splitlines()):
        linha = linha.strip()
        if linha.startswith("{"):
            saida = json.loads(linha)
            return saida.get("scores", {}), saida.get("estilos", {})
    return {}, {}


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
