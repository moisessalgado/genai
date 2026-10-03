"""hq/letreiro: balão não cobre rosto, não cobre outro balão e aponta para quem fala."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.hq import letreiro, rostos  # noqa: E402
from genai.hq.rostos import Rosto  # noqa: E402
from genai.hq.roteiro import Quadro, Texto  # noqa: E402

# dois personagens com a cabeça no alto do quadro: o caso do quadro 3 do spike
A = Rosto(0.15, 0.20, 0.07, 0.10, 0.9)
B = Rosto(0.72, 0.18, 0.07, 0.10, 0.9)
FIGURANTE = Rosto(0.45, 0.40, 0.02, 0.02, 0.7)


def _quadro(*textos):
    return Quadro(id=1, personagens=["a", "b"], cena="x", textos=list(textos))


def _inter(a, b):
    return letreiro._inter(a, b)


def test_baloes_nao_cobrem_rostos_nem_um_ao_outro():
    q = _quadro(Texto(tipo="fala", quem="a", texto="Quem é aquele homem curvado?"),
                Texto(tipo="fala", quem="b", texto="Um velho, senhor. A velhice chega a todos."))
    bs = letreiro.planejar(q, (1100, 960), [A, B, FIGURANTE], 46)
    caixas_rosto = [(r.x, r.y, r.w, r.h) for r in (A, B, FIGURANTE)]
    for b in bs:
        assert all(_inter(b.caixa, f) == 0 for f in caixas_rosto), b
    assert _inter(bs[0].caixa, bs[1].caixa) == 0
    # rabicho de cada um aponta para o rosto de quem fala
    assert abs(bs[0].alvo[0] - A.centro[0]) < 1e-9
    assert abs(bs[1].alvo[0] - B.centro[0]) < 1e-9
    # ordem de leitura: o segundo balão não fica à esquerda e acima do primeiro
    a, b = bs[0].caixa, bs[1].caixa
    assert not (b[0] + b[2] / 2 < a[0] + a[2] / 2 and b[1] < a[1])


def test_falantes_ignoram_figurantes_e_seguem_a_ordem_esquerda_direita():
    q = _quadro()
    assert letreiro.falantes(q, [B, FIGURANTE, A]) == {"a": A, "b": B}
    # rosto a menos: não dá para saber quem é quem
    assert letreiro.falantes(q, [A]) == {}


def test_recordatorio_vai_para_o_canto_de_cima_a_esquerda():
    q = Quadro(id=1, personagens=[], cena="x", textos=[
        Texto(tipo="recordatorio", texto="Kapilavastu. O príncipe cresceu cercado de luxo.")])
    (b,) = letreiro.planejar(q, (2240, 980), [], 46)
    assert b.caixa[0] < 0.05 and b.caixa[1] < 0.06 and b.alvo is None


def test_letrar_desenha_no_tamanho_pedido(tmp_path):
    arte = tmp_path / "arte.png"
    Image.new("RGB", (1024, 1024), (90, 140, 200)).save(arte)
    q = _quadro(Texto(tipo="pensamento", quem="b", texto="Também eu?"))
    img, bs = letreiro.letrar(arte, q, (600, 800), 30, [A, B])
    assert img.size == (600, 800) and len(bs) == 1
    # o balão é branco: algum pixel branco apareceu sobre o azul
    assert (255, 255, 255) in {img.getpixel((x, y)) for x in range(0, 600, 7)
                               for y in range(0, 800, 7)}
    vazio, _ = letreiro.letrar(arte, q, (600, 800), 30, [A, B], ate=0)
    assert (255, 255, 255) not in {vazio.getpixel((x, y)) for x in range(0, 600, 7)
                                   for y in range(0, 800, 7)}


def test_yunet_acha_rosto_desenhado(tmp_path):
    """Sanidade do detector real com um quadro do spike, se estiver na máquina."""
    spike = Path(__file__).resolve().parents[1] / "projects/hq-buda-ukiyoe/quadros/2/cand-1.png"
    if not spike.exists():
        import pytest
        pytest.skip("quadro do spike não está nesta máquina")
    rs = rostos.principais(rostos.detectar(spike))
    assert len(rs) == 2 and rs[0].x < rs[1].x
