"""Fundo de slides: o que dá para verificar sem assistir ao vídeo.

O risco real deste preset não é estético, é o grafo de filtros: um rótulo solto
ou um `offset` errado só aparece como erro do ffmpeg no fim de um render de meia
hora. Por isso a maior parte daqui checa aritmética de duração e, no fim, um
render de verdade num vídeo minúsculo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.video import slides
from genai.video.render import LARGURA, ALTURA, presets, renderizar


@pytest.fixture
def acervo(tmp_path: Path) -> Path:
    """Seis imagens de teste, em três proporções — como o acervo real."""
    d = tmp_path / "slides"
    d.mkdir()
    for i, (w, h) in enumerate([(1024, 1024), (1680, 720), (1344, 896)] * 2):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc=s={w}x{h}:d=1", "-frames:v", "1",
             str(d / f"img{i}.jpg")], check=True)
    return d


def duracao_do_encadeado(n: int, cada: float, cruzamento: float) -> float:
    """`n` imagens sobrepostas duas a duas duram menos que a soma delas."""
    return n * cada - (n - 1) * cruzamento


def test_preset_declarado():
    assert "slides" in presets()


def test_uma_imagem_a_cada_45_segundos():
    """Nove minutos de capítulo dão doze trocas, não duas nem duzentas."""
    assert slides.quantas(9 * 60) == 12
    assert slides.quantas(60 * 60) == 80


def test_audio_curto_nao_vira_so_dissolve():
    """Com o teto aritmético, cada imagem ainda fica parada o tempo de um
    dissolve — sem ele, 10 s pediriam slides que nascem já saindo."""
    for d in (3.0, 6.0, 10.0, 30.0):
        n = slides.quantas(d)
        cada = (d + (n - 1) * slides.CRUZAMENTO) / n
        assert cada >= 2 * slides.CRUZAMENTO or n == 1


def test_duracao_invalida_e_recusada():
    with pytest.raises(ValueError):
        slides.quantas(0)


def test_plano_cobre_o_audio_inteiro(acervo):
    """O vídeo tem de SOBRAR sobre a narração: `-shortest` corta o que sobra,
    mas não inventa o que falta."""
    for d in (30.0, 540.0, 3600.0):
        imgs, cada, cz = slides.plano(d, acervo)
        assert duracao_do_encadeado(len(imgs), cada, cz) >= d


def test_sorteio_nao_repete_antes_de_esgotar_o_acervo(acervo):
    imgs = slides.sortear(slides.disponiveis(acervo), 6, seed=7)
    assert len(set(imgs)) == 6


def test_sorteio_repete_sem_encostar_em_si_mesmo(acervo):
    """Acervo de 6 e 20 slides: repetir é inevitável, repetir em sequência não
    — um dissolve entre duas cópias da mesma imagem parece vídeo travado."""
    imgs = slides.sortear(slides.disponiveis(acervo), 20, seed=7)
    assert len(imgs) == 20
    assert all(a != b for a, b in zip(imgs, imgs[1:]))


def test_sorteio_muda_entre_renders(acervo):
    """Sem seed, dois vídeos seguidos não podem sair com as mesmas imagens."""
    a = [slides.sortear(slides.disponiveis(acervo), 3) for _ in range(8)]
    assert len({tuple(x) for x in a}) > 1


def test_seed_fixa_reproduz_o_mesmo_video(acervo):
    disp = slides.disponiveis(acervo)
    assert slides.sortear(disp, 5, seed=42) == slides.sortear(disp, 5, seed=42)


def test_acervo_vazio_e_recusado(tmp_path):
    with pytest.raises(FileNotFoundError, match="nenhuma imagem"):
        slides.plano(60.0, tmp_path / "nao-existe")


def test_filtro_sai_num_unico_rotulo_v(acervo):
    """`renderizar` queima a legenda trocando o `[v]` final: se houvesse dois,
    o grafo quebraria com a legenda ligada."""
    imgs, cada, cz = slides.plano(300.0, acervo)
    f = slides.filtro(imgs, cada, cz, LARGURA, ALTURA, veu=True)
    assert f.count("[v]") == 1
    assert f.endswith("[v]")


def test_filtro_de_imagem_unica_tambem_sai_em_v(acervo):
    f = slides.filtro(slides.disponiveis(acervo)[:1], 10.0, 2.0,
                      LARGURA, ALTURA, veu=False)
    assert f.count("[v]") == 1


def test_render_de_verdade(acervo, tmp_path):
    """O teste que pega grafo quebrado: 12 s de áudio, slides de 3 s, ffmpeg
    real. Verifica que o vídeo tem a duração do áudio (e não a do encadeado) e
    que saiu em 1080p."""
    audio = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=f=220:d=12", str(audio)], check=True)
    destino = renderizar(audio, tmp_path / "v.mp4", preset="slides",
                         gpu=False, slides_dir=acervo, slides_seg=3.0,
                         slides_seed=1)
    saida = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries",
         "format=duration", "-of", "csv=p=0", str(destino)],
        capture_output=True, text=True, check=True).stdout.split()
    assert saida[0] == f"{LARGURA},{ALTURA}"
    assert abs(float(saida[1]) - 12.0) < 0.5


def _luminancia(frame: Path, y: int) -> float:
    """Brilho médio de uma faixa de 40 px do quadro, na altura `y`."""
    bruto = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(frame),
         "-vf", f"crop={LARGURA}:40:0:{y},format=gray,scale=1:1",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return bruto[0]


def test_veu_escurece_o_rodape_so_quando_ha_legenda(acervo, tmp_path):
    """O véu existe para a legenda sobreviver a uma arte clara. Renderiza o
    MESMO vídeo com e sem legenda e compara o rodapé: sem ela, a arte tem de
    sair intocada; com ela, o pé do quadro escurece e o meio não.

    Também é o teste que pega índice de entrada errado — o véu abre mais uma
    entrada no ffmpeg, e o áudio passa a ser a seguinte."""
    audio = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=f=220:d=8", str(audio)], check=True)
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:07,000\nlinha de teste\n\n")

    quadros = {}
    for rotulo, leg in (("sem", None), ("com", srt)):
        mp4 = renderizar(audio, tmp_path / f"{rotulo}.mp4", preset="slides",
                         gpu=False, slides_dir=acervo, slides_seg=8.0,
                         slides_seed=5, legenda=leg)
        # o áudio tem de continuar mapeado, com ou sem a entrada extra do véu
        assert subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, check=True).stdout.strip() == "aac"
        quadro = tmp_path / f"{rotulo}.png"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "4",
             "-i", str(mp4), "-frames:v", "1", str(quadro)], check=True)
        quadros[rotulo] = quadro

    pe, meio = ALTURA - 60, ALTURA // 2
    assert _luminancia(quadros["com"], pe) < _luminancia(quadros["sem"], pe) - 15
    assert abs(_luminancia(quadros["com"], meio)
               - _luminancia(quadros["sem"], meio)) < 6


def test_acervo_local_serve_para_um_capitulo():
    """O acervo NÃO é versionado (25 MB; ver `.gitignore` e `LICENSES.md`), então
    num clone limpo este teste não tem o que verificar e pula. Na máquina que
    publica ele vale: `slides` é o preset padrão do `video`, e um acervo curto
    demais faria a mesma imagem voltar várias vezes no mesmo capítulo."""
    acervo = slides.disponiveis()
    if not acervo:
        pytest.skip(f"sem acervo em {slides.DIRETORIO_PADRAO} (não versionado)")
    assert len(acervo) >= slides.quantas(20 * 60)


def _quadro(mp4: Path, t: float, destino: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t),
         "-i", str(mp4), "-frames:v", "1", str(destino)], check=True)
    return destino


def test_veu_nao_anima(acervo, tmp_path):
    """O véu tem de ser a MESMA imagem do começo ao fim do slide.

    Ele nasceu de um `gradients`, que anima por padrão: a cor circula em `speed`
    e o rodapé pulsa a noite toda. O teste compara dois instantes distantes
    DENTRO do mesmo slide — 8 s de áudio dão um slide só, então não há dissolve
    para confundir o resultado — e exige que o pé do quadro não tenha mudado."""
    audio = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=f=220:d=8", str(audio)], check=True)
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:07,500\nlinha fixa\n\n")
    mp4 = renderizar(audio, tmp_path / "v.mp4", preset="slides", gpu=False,
                     slides_dir=acervo, slides_seg=8.0, slides_seed=5,
                     legenda=srt)
    a, b = (_quadro(mp4, t, tmp_path / f"q{t}.png") for t in (1.0, 6.0))
    # a legenda é a mesma nos dois instantes, então qualquer diferença no rodapé
    # é do véu (ou do codificador, daí a folga de 1 nível)
    pe = ALTURA - 100
    assert abs(_luminancia(a, pe) - _luminancia(b, pe)) <= 1


def test_veu_desaparece_antes_do_meio(acervo, tmp_path):
    """A rampa do véu tem de chegar ao topo dele já invisível.

    O véu antigo tinha 300 px e escurecia depressa: a borda de cima aparecia
    como uma faixa desenhada sobre a arte. O conserto foi esticar a rampa e
    curvá-la, e o que garante isso é o formato do degradê — escuro no pé,
    imperceptível onde ele começa."""
    audio = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=f=220:d=8", str(audio)], check=True)
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:07,500\nlinha fixa\n\n")
    quadros = {}
    for rotulo, leg in (("sem", None), ("com", srt)):
        mp4 = renderizar(audio, tmp_path / f"{rotulo}.mp4", preset="slides",
                         gpu=False, slides_dir=acervo, slides_seg=8.0,
                         slides_seed=5, legenda=leg)
        quadros[rotulo] = _quadro(mp4, 4.0, tmp_path / f"{rotulo}.png")

    topo = ALTURA - slides.VEU_ALTURA + 10       # logo abaixo da borda do véu
    pe = ALTURA - 40                             # onde a legenda mora
    escuro = lambda y: (_luminancia(quadros["sem"], y)
                        - _luminancia(quadros["com"], y))
    assert escuro(topo) <= 3, "a borda de cima do véu está visível"
    assert escuro(pe) > 20, "o véu não está dando contraste para a legenda"


def test_saida_em_faixa_limitada_bt709(acervo, tmp_path):
    """O MP4 saía como `yuvj420p` — faixa cheia, só marcada, nunca convertida.

    Player que ignora a marca (não são poucos) esmagava preto e branco. A cadeia
    agora converte de verdade e escreve as flags de cor no arquivo."""
    audio = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=f=220:d=6", str(audio)], check=True)
    mp4 = renderizar(audio, tmp_path / "v.mp4", preset="slides", gpu=False,
                     slides_dir=acervo, slides_seg=6.0, slides_seed=2)
    campos = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt,color_range,color_space", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert campos == "yuv420p,tv,bt709"
