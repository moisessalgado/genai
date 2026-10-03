"""Trilha de fundo por modelo dedicado -- MusicGen Stereo 3.3B (TDD 18, comparativo).

Por que testar ao lado do ACE-Step: das seis peças da paleta `flauta` gerada
pelo ACE-Step, o operador aproveitou uma só -- as outras cinco soaram feias ao
ouvido. MusicGen é outro modelo, outra arquitetura (autorregressivo sobre
códigos EnCodec, não difusão), e pode acertar timbres que o ACE-Step erra --
ou não. Só a escuta decide, por isso o desenho aqui espelha o do ACE-Step:
paleta por paleta, peça por peça, para comparar lado a lado.

Por que MusicGen e não antes: os pesos são CC-BY-NC-4.0 (Meta AudioCraft), e o
motivo original para descartá-lo (registrado em LICENSES.md) era o canal
publicar comercialmente. Não publica -- decisão do operador, 2026-09-01.

O modelo NAO roda no processo do pipeline: vive na `.venv-musica-mg`, atrás de
`_musicgen_runner.py`. Venv separada da do ACE-Step de propósito: nenhuma razão
para as duas dependerem da mesma fixação de `transformers`/`torch`, e um erro
de instalação numa não arrisca quebrar a outra.

As paletas espelham as do `musica_ace` -- mesmo nome, mesma família de timbre
-- de propósito: é o que permite comparar os dois motores no mesmo instrumento,
em vez de comparar o timbre A do ACE contra o timbre B do MusicGen.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from ..config import settings
from .musica import SAMPLE_RATE, _ffmpeg
from .musica_ace import _por_pico, moldar, montar  # genéricas, não específicas do ACE

VENV = settings().venv_musica_mg
CHECKPOINT = "facebook/musicgen-stereo-large"

# Mesma divisão acervo/cache de `musica_ace` -- ver o cabeçalho de lá para o
# porquê. As duas pastas são COMPARTILHADAS com o ACE-Step; o prefixo `mg-` no
# nome do arquivo é o que evita colisão entre os dois motores.
ACERVO = settings().assets_dir / "musica"
CACHE = settings().cache_dir / "musica"

# MusicGen degrada (repete, deriva de tom) em geração contínua além de ~30 s --
# é o tamanho de trecho predominante no treino. Por isso a peça aqui é mais
# curta que a do ACE-Step (120 s): a variedade continua vindo de VÁRIAS peças
# emendadas, não de uma peça longa. Mais peças (8, não 6) para cobrir duração
# parecida de material inédito.
PECA_S = 30.0
N_PECAS = 8
CRUZAMENTO_S = 4.0

# CFG padrão do MusicGen (ver model card) -- diferente da escala do ACE-Step,
# não é o mesmo número por coincidência de nome.
GUIDANCE = 3.0

# Frases descritivas, não lista de tags: MusicGen foi treinado com legendas em
# linguagem natural (estilo "a calm piano piece with..."), e é assim que a
# própria model card do Meta descreve o prompt. Uma pilha de tags soltas é o
# estilo do ACE-Step/Suno, não o daqui -- por isso os prompts foram reescritos
# como frases, mesmo espelhando o timbre de cada paleta do ACE-Step.
PALETAS: dict[str, tuple[str, ...]] = {
    "contemplativo": (
        "A calm instrumental piece with felt piano and sustained strings, "
        "warm and slow, meditative mood, no drums, no percussion, no vocals.",
        "A warm string ensemble with cello and viola, slow sustained chords, "
        "ambient and contemplative, instrumental only, no drums, no vocals.",
        "Solo grand piano with soft touch and long pedal, slow tempo, "
        "spacious and warm, instrumental, no percussion, no vocals.",
        "Piano and cello duet, sparse notes, gentle and warm, ambient "
        "instrumental music, no drums, no vocals.",
        "A muted string quartet playing slow sustained chords, soft and "
        "contemplative, instrumental ambient music, no drums, no vocals.",
        "Upright piano with felt dampers, distant and warm tone, slow "
        "meditative instrumental piece, no drums, no vocals.",
    ),
    "drone": (
        "A continuous tanpura drone with ringing open strings and rich "
        "overtones, sustained and hypnotic, no melody, no drums, no vocals.",
        "A sustained harmonium and shruti box drone, warm and resonant, "
        "unchanging, meditative ambient instrumental music.",
        "A single long bowed viola note, sustained and singing, drone-like, "
        "instrumental ambient, no drums, no vocals.",
        "A deep cello drone with a long slow bow, warm and resonant, "
        "continuous ambient instrumental music, no drums, no vocals.",
        "Sustained strings in unison, organ-like and warm, drone ambient "
        "instrumental, unchanging, no drums, no vocals.",
        "A continuous tambura drone with shimmering overtones, hypnotic and "
        "meditative, instrumental ambient music, no drums, no vocals.",
    ),
    "sobrio": (
        "Solo cello playing long bowed notes, slow and sparse, warm "
        "instrumental ambient music, no drums, no vocals.",
        "A sustained dark double bass drone, slow and minimal, instrumental "
        "ambient music, no drums, no vocals.",
        "Piano in the low register, sparse notes, slow and contemplative, "
        "instrumental ambient music, no drums, no vocals.",
        "Viola and cello duet, slow sustained playing, warm instrumental "
        "ambient music, no drums, no vocals.",
        "Bass clarinet with breathy long tones, slow and sparse, "
        "instrumental ambient music, no drums, no vocals.",
        "Muted felt piano, distant and warm, slow sparse notes, "
        "instrumental ambient music, no drums, no vocals.",
    ),
    "piano": (
        "Solo piano performance, pentatonic melody, Japanese ambient style, "
        "slow tempo, instrumental, no drums, no percussion, no vocals.",
        "Minimalist solo piano with sparse pentatonic notes, slow and "
        "floating, instrumental ambient music, no drums, no vocals.",
        "Solo piano performance, Japanese ambient style, floating and "
        "unhurried, instrumental, no drums, no vocals.",
        "Solo piano, East Asian ambient style, minimalist and soft "
        "dynamics, instrumental, no drums, no vocals.",
        "Solo piano performance, warm pentatonic melody, sparse and "
        "unhurried, instrumental ambient music, no drums, no vocals.",
        "Solo piano performance, Japanese ambient style, spacious and "
        "quiet, floating, instrumental, no drums, no vocals.",
    ),
    "flauta": (
        "Solo shakuhachi bamboo flute, Japanese ambient style, breathy "
        "tone, slow and meditative, instrumental, no drums, no vocals.",
        "Solo bansuri flute, Indian classical style, meditative and slow, "
        "instrumental ambient music, no drums, no vocals.",
        "Solo Native American flute, breathy sustained notes, slow "
        "instrumental ambient music, no drums, no vocals.",
        "Solo Andean quena flute, breathy high register, Andean folk "
        "style, slow instrumental ambient music, no drums, no vocals.",
        "Solo shakuhachi flute, Zen ambient style, floating and unhurried, "
        "instrumental, no drums, no vocals.",
        "Solo Chinese dizi flute, breathy tone, soft dynamics, ambient "
        "instrumental music, no drums, no vocals.",
    ),
    # Paleta dedicada (TDD 18, escolha do operador em 2026-09-01, ouvindo a
    # paleta `flauta`): "bansuri, indian classical" foi o único timbre das duas
    # famílias (ACE-Step e MusicGen) aprovado para os suttas. Seis variações do
    # MESMO instrumento/estilo -- registro, dinâmica, humor -- em vez de seis
    # instrumentos diferentes: o objetivo é padrão sonoro do canal, não catálogo
    # de timbres. É a paleta padrão do pipeline `sutta`.
    "bansuri": (
        "Solo bansuri flute, Indian classical style, meditative and slow, "
        "instrumental ambient music, no drums, no vocals.",
        "Solo bansuri flute, Hindustani classical raga style, breathy and "
        "warm, slow tempo, instrumental, no drums, no vocals.",
        "Solo bansuri flute, low register, deep and mellow tone, "
        "meditative Indian classical style, instrumental, no vocals.",
        "Solo bansuri flute, high register, bright and airy, Indian "
        "classical ambient, slow and unhurried, instrumental, no vocals.",
        "Solo bansuri flute, floating melody, Indian classical style, "
        "spacious and quiet, instrumental, no drums, no vocals.",
        "Solo bansuri flute, sustained breathy notes, Hindustani classical "
        "mood, warm and contemplative, instrumental, no vocals.",
    ),
}


def prompts(paleta: str = "contemplativo") -> tuple[str, ...]:
    if paleta not in PALETAS:
        raise ValueError(f"paleta desconhecida: {paleta} — use {', '.join(PALETAS)}")
    return PALETAS[paleta]


def disponivel() -> bool:
    """A venv isolada existe? Sem ela, o `video` cai na trilha sintetizada."""
    return (VENV / "bin" / "python").exists()


def _seed(paleta: str, i: int) -> int:
    """Mesma lógica de `musica_ace._seed`: determinístico por paleta+índice,
    para que um capítulo regerado não troque de trilha no meio de uma série já
    publicada."""
    h = hashlib.sha256(f"musicgen:{paleta}:{i}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def gerar_pecas(paleta: str = "contemplativo", n: int = N_PECAS,
                peca_s: float = PECA_S, sample_rate: int = SAMPLE_RATE,
                inicio: int = 0, progresso=None) -> list[Path]:
    """Gera (ou reaproveita) as peças do leito, já em 24 kHz mono.

    Mesmo contrato de `musica_ace.gerar_pecas` -- ver lá para o porquê de
    reaproveitar por hash do prompt e separar bruto (`CACHE`) de acervo
    (`ACERVO`).
    """
    if not disponivel():
        raise RuntimeError(
            f"venv de música (MusicGen) ausente em {VENV} — rode `uv venv "
            f"--python 3.12 .venv-musica-mg && uv pip install --python "
            f".venv-musica-mg/bin/python transformers accelerate soundfile "
            f"scipy sentencepiece`")
    CACHE.mkdir(parents=True, exist_ok=True)
    ACERVO.mkdir(parents=True, exist_ok=True)
    ps = prompts(paleta)
    idx = list(range(inicio, inicio + n))
    marca = [hashlib.sha256(ps[i % len(ps)].encode()).hexdigest()[:8] for i in idx]
    finais = [ACERVO / f"mg-{paleta}-{i:02d}-{marca[k]}-{int(peca_s)}s-{sample_rate}.wav"
              for k, i in enumerate(idx)]
    brutos = [CACHE / f"mg-{paleta}-{i:02d}-{marca[k]}-{int(peca_s)}s-bruto.wav"
              for k, i in enumerate(idx)]

    pendentes = [{"prompt": ps[i % len(ps)], "seed": _seed(paleta, i),
                  "destino": str(brutos[k])}
                 for k, i in enumerate(idx)
                 if not finais[k].exists() and not brutos[k].exists()]
    if pendentes:
        if progresso:
            progresso(f"gerando {len(pendentes)} peça(s) no MusicGen…")
        _rodar(pendentes, peca_s)

    for bruto, final in zip(brutos, finais):
        if not final.exists():
            _ffmpeg(["-i", str(bruto), "-ar", str(sample_rate), "-ac", "1",
                     "-c:a", "pcm_s24le", str(final)])
    return finais


def _rodar(pecas: list[dict], peca_s: float) -> None:
    pedido = {"pecas": pecas, "duracao_s": peca_s, "guidance": GUIDANCE,
              "checkpoint": CHECKPOINT, "hf_home": str(settings().hf_home)}
    runner = Path(__file__).with_name("_musicgen_runner.py")
    r = subprocess.run([str(VENV / "bin" / "python"), str(runner), json.dumps(pedido)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"MusicGen falhou:\n{r.stderr[-2000:]}")


def preparar_trilha(destino: Path, duracao_s: float, paleta: str = "contemplativo",
                    sample_rate: int = SAMPLE_RATE, dip_db: float = 5.0,
                    n_pecas: int = 1, peca_s: float = PECA_S, peca_inicial: int = 0,
                    progresso=None) -> Path:
    """Deixa em `destino` um leito do tamanho exato da narração.

    Mesmo contrato de `musica_ace.preparar_trilha`, reaproveitando `montar()`
    (emenda com crossfade) e `moldar()` (dip na faixa da voz) de lá -- as duas
    são processamento de sinal genérico, não específico do ACE-Step.
    """
    import soundfile as sf

    pecas = gerar_pecas(paleta, n=n_pecas, peca_s=peca_s, inicio=peca_inicial,
                        sample_rate=sample_rate, progresso=progresso)
    leito = montar(duracao_s, pecas, sample_rate=sample_rate,
                   cruzamento_s=CRUZAMENTO_S)
    leito = _por_pico(moldar(leito, sample_rate, dip_db), 0.5).astype(np.float32)
    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destino), leito, sample_rate)
    return destino
