"""Trilha de fundo por modelo dedicado -- ACE-Step v1 3.5B (TDD 18, V3).

Por que trocar o sintetizador do `musica.py`: o drone modal cumpria o contrato
espectral, mas soava abstrato e ate sinistro. O motivo e de desenho, nao de
ajuste: parciais desafinados, sem ataque, com o meio do espectro cavado em 15 dB,
e a receita de pad de ficcao cientifica. Nenhum instrumento real soa assim.

Por que ACE-Step e nao outro: os pesos sao **Apache-2.0** (model card
`ACE-Step/ACE-Step-v1-3.5B`, conferido). O MusicGen esta fora -- CC-BY-NC nos
pesos, e este canal publica. A licenca era a razao original de sintetizar tudo, e
ela continua satisfeita aqui.

O modelo NAO roda no processo do pipeline: ele vive na `.venv-musica`, atras de
`_ace_runner.py`. Ver o cabecalho daquele arquivo para o motivo.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from ..config import settings
from ..servicos import invokeai
from .musica import SAMPLE_RATE, _ffmpeg

VENV = settings().venv_musica
CHECKPOINT = settings().ace_checkpoint

# Duas pastas, e a divisao entre elas nao e organizacao: e o que pode ser
# apagado sem consequencia.
#
# `cache/musica` guarda o BRUTO do modelo (48 kHz estereo, 23 MB por peca) e as
# sobras de experimento. Apagar custa GPU e mais nada -- a peca volta identica,
# porque a seed sai da paleta.
#
# `assets/musica` guarda a peca PRONTA, no formato que entra no video. Ela e
# acervo do canal, do mesmo tipo que `assets/slides`: os videos ja publicados
# soam com ela, e apagar significa que um capitulo remontado amanha nao combina
# mais com os irmaos que foram ao ar. O determinismo da seed protege contra
# gerar diferente, nao contra perder o arquivo.
ACERVO = settings().assets_dir / "musica"
CACHE = settings().cache_dir / "musica"

# Duracao de cada peca gerada. O ACE-Step aguenta ~4 min, mas peca longa nao
# compra nada aqui: o que combate a monotonia e a VARIEDADE entre pecas, e duas
# de 2 min custam o mesmo que uma de 4 e rendem duas paisagens.
PECA_S = 120.0
# Quantas pecas distintas formam o leito. Seis dao 12 min de material inedito;
# sob a narracao a -20 LUFS, ninguem percebe o retorno de uma peca depois disso.
N_PECAS = 6
# Cruzamento longo de proposito: e ele que faz a troca de peca virar mudanca de
# paisagem em vez de corte. Peca nova entrando em 8 s nao e um "evento".
CRUZAMENTO_S = 8.0

PASSOS = 60
GUIDANCE = 15.0

# Termos que valem para toda paleta. `no vocals` e `no drums` sao os dois que
# nao podem faltar: voz cantada disputa com a narracao, e ataque percussivo
# atravessa o ducking (o sidechain e lento demais para pegar transiente).
_BASE = ("ambient, instrumental, meditative, slow tempo, sparse, gentle, warm, "
         "contemplative, soft reverb, spacious, quiet, "
         "no drums, no percussion, no beat, no vocals")

# A paleta `drone` precisa da base OPOSTA. `sparse` pede notas separadas por
# silencio, e um bordao e o contrario disso: som continuo que nunca resolve.
# Manter o _BASE aqui produziria notas longas com buraco entre elas, que e outra
# coisa -- e justamente a que soa como espera de telefone.
_BASE_DRONE = ("ambient, instrumental, meditative, drone, sustained, continuous, "
               "unchanging, mid register, rich overtones, warm, resonant, hypnotic, "
               "no melody, no chord changes, no drums, no percussion, no beat, "
               "no vocals")

# Tentativa 1 (nao repetir): lista longa de "no jazz, no strings, no synth..."
# saiu pior, nao melhor -- o ACE-Step nao tem campo de negative prompt (conferido
# na assinatura de `ACEStepPipeline.__call__`), so um `prompt` unico. Uma pilha
# de negacoes nesse campo parece ser lida como TAG, nao como proibicao: pediu
# "no jazz" e saiu jazz. As tres pecas que sobreviveram ao descarte do operador
# sao todas da familia `contemplativo`, cujos prompts sao curtos e so afirmam o
# que devem ser. `_BASE_PIANO` segue a mesma receita: afirma "solo piano
# performance" e "unaccompanied" em vez de negar orquestra, e usa generos reais
# de catalogo (new age, meditation music) em vez de excluir jazz/blues/pop.
# So mantem a negacao ja provada no `_BASE` (bateria/vocal), que funciona nas
# pecas aprovadas.
_BASE_PIANO = ("solo piano performance, unaccompanied, "
               "new age, meditation music, "
               "ambient, instrumental, slow tempo, sparse, gentle, warm, "
               "contemplative, soft reverb, spacious, quiet, "
               "no drums, no percussion, no beat, no vocals")

_BASE_FLAUTA = ("solo bamboo flute performance, unaccompanied, "
                "ambient, instrumental, meditative, slow tempo, sparse, gentle, warm, "
                "contemplative, soft reverb, spacious, quiet, "
                "no drums, no percussion, no beat, no vocals")

# Timbres acusticos por padrao. A queixa original era "extraterrestre", e pad de
# sintetizador e exatamente o caminho de volta para la.
PALETAS: dict[str, tuple[str, ...]] = {
    # Familia estreitada pelo ouvido do operador, depois de ouvir as seis
    # primeiras: piano e cordas friccionadas acompanham a leitura de sutta;
    # violao, harpa, sinos, flauta e marimba nao. O padrao dos quatro rejeitados
    # e claro -- corda pincada e percussao melodica tem ATAQUE, e ataque vira
    # evento. Um leito para texto recitado precisa de som que comeca sem que se
    # perceba onde.
    #
    # As duas aprovadas ficam nas posicoes 0 e 1 de proposito: a chave do cache
    # carrega o indice, entao mexer nelas descartaria a musica ja aprovada.
    "contemplativo": (
        "felt piano, sustained strings",
        "warm string ensemble, cello, viola",
        "grand piano, soft touch, long pedal",
        "piano and cello, sparse duet",
        "string quartet, muted, slow sustained chords",
        "upright piano, felt dampers, distant",
    ),
    # Medida na referencia que o operador aprovou (recitacao em pali, 59 min):
    # 77% da energia da trilha entre 150 e 300 Hz, centroide em 340 Hz, e nada
    # acima de 1,5 kHz. Isso nao e um conjunto tocando baixo -- e um bordao, a
    # base continua da tradicao (tanpura, caixa de shruti). Os timbres aqui miram
    # aquela faixa: corda grave solta e palheta livre, que ressoam em 150-300 Hz
    # sem que seja preciso forcar com equalizacao.
    "drone": (
        "tanpura drone, open strings, ringing overtones",
        "shruti box, harmonium drone, sustained",
        "bowed viola, one long sustained note",
        "cello drone, long slow bow, singing register",
        "sustained strings in unison, organ-like",
        "tambura, continuous, shimmering overtones",
    ),
    "sobrio": (
        "solo cello, long bowed notes",
        "double bass, sustained, dark",
        "piano, low register, sparse notes",
        "viola and cello duet, slow",
        "bass clarinet, breathy, long tones",
        "muted piano, felt, distant",
    ),
    # Piano solo, nada mais -- sem cordas, sem duo. Diferente de "contemplativo",
    # que mistura piano com cordas friccionadas; aqui o instrumento e um so, do
    # inicio ao fim. Pegada oriental/meditativa como afirmacao (generos reais de
    # catalogo), nunca como negacao de jazz/blues -- ver o comentario de
    # `_BASE_PIANO` sobre por que negacao nao funciona aqui.
    # 01 e 02 sao as aprovadas pelo operador -- texto intocado de proposito
    # (mexer invalidaria o hash e as pecas ja salvas). As duas tem em comum um
    # genero real de catalogo (pentatonic melody / japanese ambient) em vez de
    # so adjetivo solto (zen, healing, tranquil), que foi o que as 4 rejeitadas
    # tinham. O refinamento aqui persegue essa pista: mais ancora de genero,
    # menos adjetivo vago.
    "piano": (
        "solo piano performance, japanese ambient, pentatonic melody, slow tempo",
        "solo piano performance, pentatonic melody, minimalist, sparse notes",
        "solo piano performance, japanese ambient, floating, unhurried",
        "solo piano performance, east asian ambient, minimalist, soft dynamics",
        "solo piano performance, pentatonic melody, warm, sparse, unhurried",
        "solo piano performance, japanese ambient, spacious, quiet, floating",
    ),
    # Candidata a testar (TDD 18, pedido do operador) -- nenhuma peca aqui foi
    # ouvida ainda, entao o texto pode mudar sem custo de invalidar cache aprovado.
    # Sopro nao tem ataque percussivo -- mesma familia de motivo que aprovou piano
    # e cordas friccionadas e rejeitou corda pinçada. Seis tradicoes de flauta de
    # bambu distintas (nao seis frases do mesmo instrumento) para o operador
    # escolher timbre, nao so confirmar que "flauta funciona".
    "flauta": (
        "shakuhachi, japanese ambient, breathy tone",
        "bansuri, indian classical, meditative",
        "native american flute, breathy, sustained notes",
        "andean quena, andean folk, breathy, high register",
        "shakuhachi, zen, floating, unhurried",
        "dizi, chinese ambient, breathy, soft dynamics",
    ),
}


def prompts(paleta: str = "contemplativo") -> tuple[str, ...]:
    if paleta not in PALETAS:
        raise ValueError(f"paleta desconhecida: {paleta} — use {', '.join(PALETAS)}")
    base = {"drone": _BASE_DRONE, "piano": _BASE_PIANO, "flauta": _BASE_FLAUTA}.get(paleta, _BASE)
    return tuple(f"{timbre}, {base}" for timbre in PALETAS[paleta])


def disponivel() -> bool:
    """A venv isolada existe? Sem ela, o `video` cai na trilha sintetizada."""
    return (VENV / "bin" / "python").exists()


def _seed(paleta: str, i: int) -> int:
    """Seed derivada do nome da paleta: a mesma paleta rende sempre o mesmo leito.

    Determinismo importa aqui por uma razao pratica: um capitulo regerado nao pode
    trocar de musica de fundo no meio de uma serie ja publicada.
    """
    h = hashlib.sha256(f"{paleta}:{i}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def gerar_pecas(paleta: str = "contemplativo", n: int = N_PECAS,
                peca_s: float = PECA_S, sample_rate: int = SAMPLE_RATE,
                inicio: int = 0, progresso=None) -> list[Path]:
    """Gera (ou reaproveita) as pecas do leito, ja em 24 kHz mono.

    Reaproveitar e o que torna a coisa viavel: gerar 12 min de musica custa
    minutos de GPU, e as pecas nao dependem do capitulo -- so da paleta. Um
    projeto novo reusa o leito do anterior sem gastar nada.

    Sai em `ACERVO`; o bruto de 48 kHz fica em `CACHE`. Ver o cabecalho das duas
    constantes para o porque da separacao.
    """
    if not disponivel():
        raise RuntimeError(
            f"venv de musica ausente em {VENV} — rode `uv venv --python 3.12 "
            f"{VENV} && uv pip install --python {VENV}/bin/python "
            f"-r envs/musica-ace.txt` (ou ajuste `[venvs] musica` no vf.toml)")
    CACHE.mkdir(parents=True, exist_ok=True)
    ACERVO.mkdir(parents=True, exist_ok=True)
    ps = prompts(paleta)
    # O prompt entra na chave do cache. Sem isso, reescrever um timbre nao
    # invalidava nada e o leito continuava sendo o antigo -- um erro silencioso,
    # do pior tipo: o codigo diz uma coisa e o arquivo em disco e outra.
    idx = list(range(inicio, inicio + n))
    marca = [hashlib.sha256(ps[i % len(ps)].encode()).hexdigest()[:8] for i in idx]
    finais = [ACERVO / f"{paleta}-{i:02d}-{marca[k]}-{int(peca_s)}s-{sample_rate}.wav"
              for k, i in enumerate(idx)]
    # As pecas nasceram no cache, antes de `assets/musica` existir. Mover na
    # primeira passada e mais barato do que gerar de novo, e mantem a musica dos
    # videos ja publicados bit a bit igual.
    for final in finais:
        antigo = CACHE / final.name
        if not final.exists() and antigo.exists():
            antigo.replace(final)
    # O modelo entrega 48 kHz estereo; o bruto fica ao lado do convertido para
    # que uma troca de sample_rate nao obrigue a gerar tudo de novo.
    brutos = [CACHE / f"{paleta}-{i:02d}-{marca[k]}-{int(peca_s)}s-bruto.wav"
              for k, i in enumerate(idx)]

    pendentes = [{"prompt": ps[i % len(ps)], "seed": _seed(paleta, i),
                  "destino": str(brutos[k])}
                 for k, i in enumerate(idx)
                 if not finais[k].exists() and not brutos[k].exists()]
    if pendentes:
        if progresso:
            progresso(f"gerando {len(pendentes)} peça(s) no ACE-Step…")
        _rodar(pendentes, peca_s)

    for bruto, final in zip(brutos, finais):
        if not final.exists():
            _ffmpeg(["-i", str(bruto), "-ar", str(sample_rate), "-ac", "1",
                     "-c:a", "pcm_s24le", str(final)])
    return finais


def _rodar(pecas: list[dict], peca_s: float) -> None:
    pedido = {"pecas": pecas, "duracao_s": peca_s, "passos": PASSOS,
              "guidance": GUIDANCE, "checkpoint": str(CHECKPOINT),
              "hf_home": str(settings().hf_home)}
    runner = Path(__file__).with_name("_ace_runner.py")
    # O InvokeAI segura a VRAM entre geracoes; o runner precisa dela inteira.
    invokeai.liberar_vram_se_no_ar()
    r = subprocess.run([str(VENV / "bin" / "python"), str(runner), json.dumps(pedido)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ACE-Step falhou:\n{r.stderr[-2000:]}")


def montar(duracao_s: float, pecas: list[Path], sample_rate: int = SAMPLE_RATE,
           cruzamento_s: float = CRUZAMENTO_S) -> np.ndarray:
    """Encadeia as pecas com cruzamento ate cobrir a duracao pedida."""
    import soundfile as sf

    if not pecas:
        raise ValueError("nenhuma peça para montar")
    audios = [sf.read(str(p), dtype="float64", always_2d=False)[0] for p in pecas]
    audios = [a if a.ndim == 1 else a.mean(axis=1) for a in audios]
    audios = [_por_pico(a, 0.7) for a in audios]

    n_total = int(duracao_s * sample_rate)
    n_cruz = int(cruzamento_s * sample_rate)
    saida = np.zeros(n_total, dtype=np.float64)

    # Rampas de potencia constante (seno/cosseno). Um cruzamento linear entre
    # sinais descorrelacionados cava um buraco de 3 dB no meio da emenda --
    # audivel justamente como o "respiro" que denuncia o corte.
    x = np.linspace(0, np.pi / 2, n_cruz)
    sobe, desce = np.sin(x), np.cos(x)

    pos, i = 0, 0
    while pos < n_total:
        a = audios[i % len(audios)]
        n = min(len(a), n_total - pos)
        trecho = a[:n].copy()
        if pos > 0:
            m = min(n_cruz, n)
            trecho[:m] *= sobe[:m]
            saida[pos:pos + m] *= desce[:m]
        saida[pos:pos + n] += trecho
        # O avanco e o tamanho da peca MENOS o cruzamento, porque a proxima entra
        # sobreposta. Quando ele nao e positivo, a peca atual ja cobre tudo o que
        # faltava e o leito esta pronto -- e preciso PARAR.
        #
        # Aqui havia `pos += max(n - n_cruz, 1)`, posto para evitar laco infinito.
        # Ele evitava o travamento e criava coisa pior: com o resto igual ao
        # cruzamento, o avanco virava 1 amostra por volta e a cauda era
        # multiplicada pela rampa e somada a uma peca nova 192 MIL vezes. A soma
        # coerente disso e uma senoide pura -- medida em 4 kHz, 74 dB acima da
        # vizinhanca, +26 dB acima do material de origem. Um leito de 44,4 s
        # entrava com 192.001 iteracoes no lugar de 1.
        avanco = n - n_cruz
        if avanco <= 0:
            break
        pos += avanco
        i += 1

    # Entrada e saida suaves: a trilha nao pode comecar com um acorde ja tocando.
    return _bordas(saida, sample_rate).astype(np.float32)


def _bordas(x: np.ndarray, sample_rate: int, s: float = 3.0) -> np.ndarray:
    n = min(int(s * sample_rate), len(x) // 2)
    if n <= 0:
        return x
    r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    x[:n] *= r
    x[-n:] *= r[::-1]
    return x


def _por_pico(x: np.ndarray, alvo: float) -> np.ndarray:
    pico = np.abs(x).max()
    return x * (alvo / pico) if pico > 0 else x


# --- moldagem espectral -----------------------------------------------------
# Aqui esta a segunda metade do conserto. O `_moldar_para_voz` do sintetizador
# derruba 15 dB de 120 a 1400 Hz -- em som harmonico real isso arranca o corpo do
# instrumento e deixa grave + brilho sem meio, que e o timbre "oco" de radio
# quebrado. Musica de verdade precisa de um dip largo e raso; quem tira a trilha
# da frente da fala no momento da fala e o ducking, que ja funciona.
DIP_CENTRO_HZ = 500.0    # centro geometrico de 300-700 Hz, onde a voz tem 49,5%
DIP_LARGURA_OIT = 1.6    # desvio-padrao em oitavas: cobre ~150 Hz a 1,7 kHz
DIP_DB = 5.0
TOPO_HZ = 11000.0        # chiado acima disso so gasta bitrate


def moldar(x: np.ndarray, sr: int, dip_db: float = DIP_DB) -> np.ndarray:
    """Dip gaussiano largo e raso na faixa da fala, em escala logaritmica.

    Gaussiana em log-frequencia, e nao um corte com bordas, porque o ouvido lê
    borda de filtro como coloracao (o "efeito telefone"); uma depressao suave de
    5 dB some como timbre e sobra como espaco.
    """
    X = np.fft.rfft(x.astype(np.float64))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    f = np.maximum(f, 1e-6)
    oitavas = np.log2(f / DIP_CENTRO_HZ)
    db = -dip_db * np.exp(-0.5 * (oitavas / DIP_LARGURA_OIT) ** 2)
    ganho = 10 ** (db / 20) / (1 + (f / TOPO_HZ) ** 4)
    return np.fft.irfft(X * ganho, len(x))


def preparar_trilha(destino: Path, duracao_s: float, paleta: str = "contemplativo",
                    sample_rate: int = SAMPLE_RATE, dip_db: float = DIP_DB,
                    n_pecas: int = 1, peca_s: float = PECA_S, peca_inicial: int = 0,
                    progresso=None) -> Path:
    """Deixa em `destino` um leito do tamanho exato da narracao.

    `n_pecas=1` por padrao: uma peca so, repetida. A intuicao inicial era o
    contrario -- seis pecas distintas para combater a monotonia -- e ela estava
    errada na pratica. Sob narracao, a TROCA de peca e um evento: o ouvinte, que
    ja tinha desistido de prestar atencao na musica, volta a nota-la. Medido no
    ch01: no cruzamento aos 3:44 a trilha sobe de -39 para -29 dB. Repeticao passa
    despercebida; mudanca, nao.
    """
    import soundfile as sf

    pecas = gerar_pecas(paleta, n=n_pecas, peca_s=peca_s, inicio=peca_inicial,
                        sample_rate=sample_rate, progresso=progresso)
    leito = montar(duracao_s, pecas, sample_rate=sample_rate)
    leito = _por_pico(moldar(leito, sample_rate, dip_db), 0.5).astype(np.float32)
    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destino), leito, sample_rate)
    return destino
