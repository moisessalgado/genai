"""Leito por modelo dedicado (ACE-Step): o que dá para verificar sem a GPU.

A geração em si não é testável aqui — depende de 8 GB de pesos e de placa. O que
é testável é tudo o que fica *em volta* dela: o prompt, a emenda entre peças e a
moldagem espectral. Foi justamente nessas três coisas que a versão sintetizada
errou, então é onde os testes têm de morar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.video.musica_ace import (
    CRUZAMENTO_S,
    DIP_CENTRO_HZ,
    PALETAS,
    SAMPLE_RATE,
    _seed,
    moldar,
    montar,
    prompts,
)

# Espectro medido na narração deste canal, em % de energia (ver test_video_musica).
VOZ_FAIXA = (300, 700)


def energia(x, sr, lo, hi):
    X = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return X[(f >= lo) & (f < hi)].sum() / X.sum()


def db(x):
    return 20 * np.log10(np.sqrt((np.asarray(x, dtype=np.float64) ** 2).mean()) + 1e-12)


def peca(tmp_path, nome, seed, dur=12.0, sr=SAMPLE_RATE):
    """Peça sintética no lugar da saída do modelo: ruído rosa com tom.

    Descorrelacionadas entre si de propósito — é exatamente o caso em que um
    cruzamento linear cavaria o buraco de 3 dB que o teste da emenda procura.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    x = rng.normal(0, 0.3, len(t)) + 0.5 * np.sin(2 * np.pi * (200 + seed * 40) * t)
    p = tmp_path / nome
    sf.write(str(p), (x / np.abs(x).max() * 0.8).astype(np.float32), sr)
    return p


# --- prompt -----------------------------------------------------------------

@pytest.mark.parametrize("paleta", sorted(PALETAS))
def test_prompt_proibe_voz_e_percussao(paleta):
    """Voz cantada disputa com a narração; ataque percussivo atravessa o ducking,
    porque o sidechain é lento demais para pegar transiente."""
    for p in prompts(paleta):
        assert "no vocals" in p and "no drums" in p and "no percussion" in p


def test_paletas_nao_repetem_timbre():
    """Seis peças iguais não combatem monotonia nenhuma."""
    for paleta in PALETAS:
        assert len(set(PALETAS[paleta])) == len(PALETAS[paleta])


def test_paleta_desconhecida_e_recusada():
    with pytest.raises(ValueError, match="paleta desconhecida"):
        prompts("cosmica")


def test_seed_e_deterministica_e_distinta():
    """Mesma paleta, mesmo leito: um capítulo regerado não pode trocar de fundo
    no meio de uma série já publicada."""
    assert _seed("contemplativo", 0) == _seed("contemplativo", 0)
    assert _seed("contemplativo", 0) != _seed("contemplativo", 1)
    assert _seed("contemplativo", 0) != _seed("sobrio", 0)


# --- montagem ---------------------------------------------------------------

def test_leito_tem_a_duracao_pedida(tmp_path):
    ps = [peca(tmp_path, f"p{i}.wav", i) for i in range(3)]
    a = montar(40.0, ps, cruzamento_s=2.0)
    assert abs(len(a) / SAMPLE_RATE - 40.0) < 0.05


def test_leito_nao_clipa(tmp_path):
    ps = [peca(tmp_path, f"p{i}.wav", i) for i in range(3)]
    a = montar(40.0, ps, cruzamento_s=2.0)
    assert 0.1 < np.abs(a).max() <= 1.0


def test_emenda_nao_cava_buraco(tmp_path):
    """Cruzamento de potência constante: entre sinais descorrelacionados, o
    cruzamento linear perde 3 dB no meio da emenda — e é esse 'respiro' que o
    ouvido reconhece como corte."""
    ps = [peca(tmp_path, f"p{i}.wav", i, dur=12.0) for i in range(3)]
    cruz = 2.0
    a = montar(30.0, ps, cruzamento_s=cruz)
    sr = SAMPLE_RATE
    meio = int((12.0 - cruz / 2) * sr)          # centro da primeira emenda
    janela = int(0.3 * sr)
    na_emenda = db(a[meio - janela:meio + janela])
    antes = db(a[int(6 * sr):int(6 * sr) + 2 * janela])
    assert na_emenda > antes - 1.5, f"emenda {na_emenda - antes:.1f} dB abaixo"


def test_leito_entra_e_sai_sem_ataque(tmp_path):
    """Começar com um acorde já tocando é um evento sonoro."""
    ps = [peca(tmp_path, f"p{i}.wav", i) for i in range(2)]
    a = montar(30.0, ps, cruzamento_s=2.0)
    sr = SAMPLE_RATE
    assert np.abs(a[:sr // 4]).max() < np.abs(a[10 * sr:11 * sr]).max() * 0.5
    assert np.abs(a[-sr // 4:]).max() < np.abs(a[10 * sr:11 * sr]).max() * 0.5


def test_montar_sem_pecas_e_recusado():
    with pytest.raises(ValueError, match="nenhuma peça"):
        montar(10.0, [])


# --- moldagem espectral -----------------------------------------------------

def test_dip_atenua_a_faixa_da_voz():
    """A faixa onde a voz tem metade da energia sai da frente — de leve."""
    sr = SAMPLE_RATE
    t = np.arange(int(4.0 * sr)) / sr
    x = np.sin(2 * np.pi * DIP_CENTRO_HZ * t)
    y = moldar(x, sr)
    atenuacao = db(x) - db(y)
    assert 4.0 < atenuacao < 6.5, f"{atenuacao:.1f} dB no centro do dip"


def test_dip_preserva_o_corpo_do_instrumento():
    """A regressão que este teste guarda: o sintetizador antigo derrubava 15 dB
    de 120 a 1400 Hz, e um instrumento sem médio soa oco e irreal. Música de
    verdade tem de sair daqui com o corpo inteiro."""
    rng = np.random.default_rng(0)
    sr = SAMPLE_RATE
    t = np.arange(int(4.0 * sr)) / sr
    # série harmônica em 220 Hz: o médio é onde vive quase toda a energia dela
    x = sum(np.sin(2 * np.pi * 220 * k * t + rng.uniform(0, 6.28)) / k
            for k in range(1, 12))
    y = moldar(x, sr)
    assert energia(y, sr, *VOZ_FAIXA) > 0.10


def test_dip_nao_e_um_corte_com_borda():
    """Gaussiana em log-frequência: borda de filtro o ouvido lê como coloração
    ('efeito telefone'). A resposta tem de ser monótona descendo até o centro."""
    sr = SAMPLE_RATE
    fs = [80, 150, 300, 500, 900, 1800, 4000]
    t = np.arange(int(2.0 * sr)) / sr
    ganhos = [db(moldar(np.sin(2 * np.pi * f * t), sr)) - db(np.sin(2 * np.pi * f * t))
              for f in fs]
    subindo = ganhos[:fs.index(500) + 1]
    descendo = ganhos[fs.index(500):]
    assert all(b <= a + 0.2 for a, b in zip(subindo, subindo[1:]))
    assert all(b >= a - 0.2 for a, b in zip(descendo, descendo[1:]))


def test_dip_e_muito_mais_raso_que_o_do_sintetizador():
    """O contraste que motivou a troca, medido lado a lado."""
    from genai.video.musica import _moldar_para_voz

    sr = SAMPLE_RATE
    t = np.arange(int(4.0 * sr)) / sr
    x = np.sin(2 * np.pi * DIP_CENTRO_HZ * t)
    novo = db(x) - db(moldar(x, sr))
    antigo = db(x) - db(_moldar_para_voz(x, sr))
    assert antigo - novo > 6.0, f"antigo {antigo:.1f} dB, novo {novo:.1f} dB"


def test_montar_nao_repete_a_cauda_quando_o_resto_cabe_no_cruzamento():
    """Regressão: o laço avançava 1 amostra por volta quando o que faltava era
    menor que o cruzamento, somando a cauda milhares de vezes. A acumulação
    coerente virava senoide pura — 4 kHz a 74 dB acima da vizinhança, audível
    como microfonia do 0:36 até o fim."""
    sr = 24000
    rng = np.random.default_rng(0)
    pecas = [rng.standard_normal(int(120 * sr)) * 0.1 for _ in range(3)]
    import soundfile as sf
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ps = []
        for i, a in enumerate(pecas):
            q = Path(d) / f"p{i}.wav"
            sf.write(str(q), a, sr)
            ps.append(q)
        # 44,4 s com cruzamento de 8 s: o resto cai exatamente no cruzamento
        leito = montar(44.4, ps, sample_rate=sr, cruzamento_s=8.0)
    assert len(leito) == int(44.4 * sr)
    # nenhum tom estreito: ruído branco montado continua ruído branco
    X = np.abs(np.fft.rfft(leito.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(leito), 1 / sr)
    b = f > 2000
    pico = X[b].max()
    fp = f[b][X[b].argmax()]
    viz = (f > fp * 0.8) & (f < fp * 1.25)
    proeminencia = 10 * np.log10(pico / np.median(X[viz]))
    assert proeminencia < 25, f"tom estreito em {fp:.0f} Hz, {proeminencia:.1f} dB acima"
