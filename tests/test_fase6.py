"""Testes da Fase 6: ingest de PDF, workers, politica de voz, chapters.txt."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.audio.process import chapters_txt
from genai.ingest.loader import (_bordas_repetidas, _unir_linhas, detectar_capitulos,
                                        limpar, ler)
from genai.pipeline import _repartir
from genai.qa.policy import Tentativa, escolher, deve_repetir


from genai.qa.verify import QAResult


AUDIO = np.zeros(2400, dtype=np.float32)
QA_BOM = QAResult(True, 0.01, "", 1.0, 15.0)

# -- ingest de PDF ------------------------------------------------------------

@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    """PDF de 5 paginas com cabecalho e rodape repetidos e palavra hifenizada."""
    pymupdf = pytest.importorskip("pymupdf")
    caminho = tmp_path_factory.mktemp("pdf") / "livro.pdf"
    doc = pymupdf.open()
    corpo = [
        ("Capítulo I — A partida", ["Naquela manhã a expedi-", "ção partiu do porto."]),
        (None, ["A viagem durou semanas."]),
        ("Capítulo II — O retorno", ["Poucos homens retornaram."]),
        (None, ["O rei nada perguntou."]),
        (None, ["E a história ficou nos livros."]),
    ]
    for n, (titulo, linhas) in enumerate(corpo, 1):
        pg = doc.new_page()
        pg.insert_text((60, 50), "A HISTÓRIA DAS BANDEIRAS", fontsize=8)
        y = 90
        if titulo:
            pg.insert_text((60, y), titulo, fontsize=14)
            y += 40
        for linha in linhas:
            pg.insert_text((60, y), linha, fontsize=11)
            y += 16
        pg.insert_text((60, 780), f"Página {n}", fontsize=8)
    doc.save(str(caminho))
    return caminho


def test_pdf_e_um_formato_aceito(pdf):
    assert "expedição partiu" in limpar(ler(pdf))


def test_pdf_descarta_cabecalho_e_rodape(pdf):
    texto = limpar(ler(pdf))
    assert "BANDEIRAS" not in texto
    assert "Página" not in texto


def test_pdf_reconstroi_capitulos(pdf):
    caps = detectar_capitulos(limpar(ler(pdf)))
    assert [t for t, _ in caps] == ["Capítulo I · A partida", "Capítulo II · O retorno"]


def test_pdf_escaneado_e_recusado(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    vazio = tmp_path / "escaneado.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(vazio))
    with pytest.raises(ValueError, match="escaneado"):
        ler(vazio)


def test_une_linhas_reparando_hifen():
    assert _unir_linhas("expedi-\nção partiu") == "expedição partiu"
    assert _unir_linhas("o vento\nsoprava") == "o vento soprava"


def _pagina(n: int, extras: list[str] | None = None) -> list[str]:
    """Pagina realista: cabecalho, tres blocos de prosa e o numero no rodape."""
    corpo = [f"Parágrafo {i} da página {n}, " + "com prosa suficiente para não " * 2
             for i in range(3)]
    return ["A HISTÓRIA DAS BANDEIRAS", *(extras or []), *corpo, f"Página {n}"]


def test_borda_repetida_ignora_a_numeracao():
    """'Página 12' e 'Página 13' sao o mesmo rodape: a chave apaga os digitos."""
    bordas = _bordas_repetidas([_pagina(n) for n in range(1, 6)])
    assert "página #" in bordas
    assert "a história das bandeiras" in bordas
    assert not any("parágrafo" in b for b in bordas)


def test_texto_repetido_em_poucas_paginas_nao_e_cabecalho():
    paginas = [_pagina(1, ["Refrão"]), *(_pagina(n) for n in range(2, 6))]
    assert "refrão" not in _bordas_repetidas(paginas)


def test_pagina_esparsa_nao_perde_o_corpo():
    """Sem bloco no meio nao ha como distinguir borda de corpo: nao se descarta nada."""
    paginas = [[f"Único parágrafo da página {n}.", f"Página {n}"] for n in range(1, 6)]
    assert _bordas_repetidas(paginas) == set()


def test_pagina_curta_ainda_perde_cabecalho_e_rodape():
    """Com um bloco de corpo no meio, ja da para reconhecer as bordas."""
    paginas = [["A HISTÓRIA DAS BANDEIRAS", f"Corpo da página {n}.", f"Página {n}"]
               for n in range(1, 6)]
    bordas = _bordas_repetidas(paginas)
    assert {"a história das bandeiras", "página #"} <= bordas
    assert "corpo da página #" not in bordas


# -- reparticao entre workers -------------------------------------------------

def test_repartir_preserva_todos_os_chunks():
    rows = [{"voice_id": "a"}] * 7 + [{"voice_id": "b"}] * 3
    lotes = _repartir(rows, 2)
    assert sum(len(l) for l in lotes) == len(rows)


def test_repartir_mantem_as_vozes_agrupadas():
    """Cada lote deve ver cada voz num bloco contiguo: e o que evita recarregar
    os conditionals a cada chunk."""
    rows = [{"voice_id": "a"}] * 7 + [{"voice_id": "b"}] * 3
    for lote in _repartir(rows, 2):
        vozes = [r["voice_id"] for r in lote]
        assert vozes == sorted(vozes, key=["a", "b"].index)
        assert len(set(vozes)) == len(_blocos(vozes))


def _blocos(seq):
    return [v for i, v in enumerate(seq) if i == 0 or seq[i - 1] != v]


def test_um_worker_nao_reparte():
    rows = [{"voice_id": "a"}] * 3
    assert _repartir(rows, 1) == [rows]


# -- politica: identidade da voz entra no melhor-de-N -------------------------

def test_prefere_a_tentativa_dentro_da_voz():
    d = escolher([Tentativa(AUDIO, QA_BOM, 1, 0.879, False),
                  Tentativa(AUDIO, QA_BOM, 2, 0.94, True)])
    assert d.aceito and d.melhor.speaker_sim == 0.94


def test_reprova_quando_nenhuma_tentativa_bate_a_voz():
    d = escolher([Tentativa(AUDIO, QA_BOM, 1, 0.80, False),
                  Tentativa(AUDIO, QA_BOM, 2, 0.83, False)])
    assert not d.aceito and "voz divergente" in d.motivo
    assert d.melhor.speaker_sim == 0.83     # guarda a melhor, para a revisao humana


def test_sem_referencia_de_voz_a_checagem_nao_reprova():
    assert escolher([Tentativa(AUDIO, QA_BOM, 1, None, None)]).aceito


def test_voz_baixa_pede_nova_seed():
    assert deve_repetir(QA_BOM, 1, voz_ok=False)
    assert not deve_repetir(QA_BOM, 1, voz_ok=True)


def test_nao_repete_alem_do_limite_de_tentativas():
    assert not deve_repetir(QA_BOM, 3, voz_ok=False)


def test_duracao_invalida_nao_e_salva_pela_similaridade():
    """Truncamento e defeito objetivo: voz perfeita nao o torna aceitavel."""
    truncado = QAResult(False, 1.0, "", 0.5, 60.0, "audio curto demais - truncado?")
    d = escolher([Tentativa(AUDIO, truncado, 1, 0.99, True)])
    assert not d.aceito


# -- chapters.txt -------------------------------------------------------------

def test_chapters_txt_acumula_e_passa_de_uma_hora():
    saida = chapters_txt([("Capítulo I", 125.4), ("Capítulo II", 3700.2),
                          ("Capítulo III", 60.0)])
    assert saida.splitlines() == ["00:00 Capítulo I", "02:05 Capítulo II",
                                  "1:03:45 Capítulo III"]
