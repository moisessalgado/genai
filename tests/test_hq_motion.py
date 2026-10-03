"""hq/motion: script da narração, faixa e tempos dos balões; compensação do dissolve."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.core.estado import Store  # noqa: E402
from genai.hq import motion, roteiro  # noqa: E402
from genai.video import render  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "hq" / "roteiro-quatro-visoes.yaml"
SR = 24000


@pytest.fixture
def proj(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "roteiro.yaml")
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "slug": "buda", "titulo": "As Quatro Visões", "narrator": "narrador-v2",
        "cast": {"channa": "citacao-v1"}, "rights": {"status": "dominio-publico"}}),
        encoding="utf-8")
    return tmp_path, roteiro.carregar(tmp_path)


def test_script_recordatorio_e_narrador_e_fala_e_a_voz_do_personagem(proj):
    p, r = proj
    r.personagens["siddhartha"].voz = "dora-v1"
    s = motion.montar_script(p, r)
    segs = s.chapters[0].segments
    assert len(segs) == 7  # 1 + 1 + 2 + 1 + 1 + 1 textos
    assert segs[0].role == "narrador" and s.voice_of(segs[0]) == "narrador-v2"
    assert segs[1].role == "siddhartha" and s.voice_of(segs[1]) == "dora-v1"
    assert s.voice_of(segs[3]) == "citacao-v1"  # cast do project.yaml vence
    assert (p / "script.json").exists()


def _narrar(p, s, duracoes):
    """Simula o `run`: chunks ok com WAVs de duração conhecida (com silêncio
    nas bordas, que a montagem apara)."""
    store = Store(p / "state.db")
    store.sync_script(s)
    for row, d in zip(store.chapter_chunks(1), duracoes):
        wav = p / f"{row['idx']}.wav"
        tom = 0.3 * np.sin(np.linspace(0, 2000, int(SR * d))).astype(np.float32)
        sf.write(wav, np.concatenate([np.zeros(SR // 5, np.float32), tom,
                                      np.zeros(SR // 5, np.float32)]), SR)
        store.finish_ok(row["chunk_id"], str(wav), d, 0.0, "", 0)
    store.close()


def test_faixa_e_cenas_batem_amostra_a_amostra(proj, monkeypatch):
    p, r = proj
    s = motion.montar_script(p, r)
    _narrar(p, s, [1.0, 1.5, 2.0, 1.2, 0.8, 1.1, 2.5])
    monkeypatch.setattr(motion, "masterizar", lambda a, b: shutil.copy(a, b))
    master, cenas = motion.montar_audio(p, r)
    total = sf.info(master).duration
    assert abs(sum(c.duracao for c in cenas) - total) < 1e-3
    # quadro 3 tem duas falas: entrada limpa, depois 1 balão, depois 2
    assert [(c.quadro, c.ate) for c in cenas if c.quadro == 3] == [(3, 0), (3, 1), (3, 2)]
    assert cenas[0].duracao == pytest.approx(motion.ENTRADA_QUADRO)
    # a fala aparada (~1 s de tom, margem do trim) + pausa de fim de quadro
    assert cenas[1].duracao == pytest.approx(1.0 + motion.PAUSA_FIM_QUADRO, abs=0.08)


def test_narracao_incompleta_e_erro(proj):
    p, r = proj
    s = motion.montar_script(p, r)
    Store(p / "state.db").sync_script(s)
    with pytest.raises(RuntimeError, match="narração incompleta"):
        motion.montar_audio(p, r)


def test_sem_narrador_e_erro(proj):
    p, r = proj
    (p / "project.yaml").write_text("slug: x\ntitulo: X\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sem `narrator`"):
        motion.montar_script(p, r)


def test_tamanho_do_quadro_no_video():
    assert motion._tamanho_video(1536 / 672) == (1920, 840)
    assert motion._tamanho_video(768 / 1024) == (810, 1080)


def test_compensar_cruzamento_faz_cada_troca_comecar_no_instante_pedido():
    d = [1.0, 2.0, 3.0]
    c = render.compensar_cruzamento(d, 0.5)
    assert c == [1.5, 2.5, 3.0]
    # offset do xfade k = soma(c[:k]) - k*cruzamento = soma(d[:k])
    assert [sum(c[:k]) - k * 0.5 for k in (1, 2)] == [1.0, 3.0]


def test_compensar_cruzamento_respeita_os_lotes_do_render():
    n = render.LOTE_MAXIMO + 3
    c = render.compensar_cruzamento([1.0] * n, 0.5)
    # a última do primeiro lote emenda por concatenação: sem compensação
    assert c[render.LOTE_MAXIMO - 1] == 1.0
    assert c[0] == 1.5 and c[-1] == 1.0
