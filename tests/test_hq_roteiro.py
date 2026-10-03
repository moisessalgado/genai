"""hq/roteiro: o schema recusa o que quebraria as etapas seguintes."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.hq import roteiro as rot  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "hq" / "roteiro-quatro-visoes.yaml"


def _base() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def _invalido(dados: dict, trecho: str) -> None:
    with pytest.raises(ValidationError, match=trecho):
        rot.Roteiro.model_validate(dados)


def test_roteiro_do_spike_valida_e_ida_e_volta_pelo_yaml(tmp_path):
    r = rot.Roteiro.model_validate(_base())
    assert [q.id for q in r.ordem_de_leitura()] == [1, 2, 3, 4, 5, 6]
    assert r.quadro(1).tamanho == (1536, 672)
    rot.salvar(r, tmp_path)
    assert rot.carregar(tmp_path) == r


def test_quem_fala_precisa_estar_no_quadro():
    d = _base()
    d["quadros"][3]["textos"][0]["quem"] = "channa"  # quadro 4 só tem siddhartha
    _invalido(d, "'channa' fala mas não está no quadro")


def test_personagem_fora_do_elenco():
    d = _base()
    d["quadros"][0]["personagens"] = ["yasodhara"]
    d["quadros"][0]["textos"] = []
    _invalido(d, "personagem desconhecido 'yasodhara'")


def test_tres_personagens_com_ficha_num_quadro_nao_cabem():
    d = _base()
    d["personagens"]["yasodhara"] = {"nome": "Yasodhara", "ficha": "a princess"}
    d["quadros"][1]["personagens"] = ["siddhartha", "channa", "yasodhara"]
    _invalido(d, "3 personagens")


def test_cada_quadro_em_exatamente_uma_tira():
    d = _base()
    d["paginas"][0]["tiras"] = [[1], [2, 3], [4, 5]]
    _invalido(d, r"fora das páginas: \[6\]")
    d["paginas"][0]["tiras"] = [[1, 2], [2, 3], [4, 5, 6]]
    _invalido(d, r"repetidos: \[2\]")
    d["paginas"][0]["tiras"] = [[1], [2, 3], [4, 5, 6], [7]]
    _invalido(d, r"inexistentes: \[7\]")


def test_recordatorio_nao_tem_dono_e_fala_tem():
    d = _base()
    d["quadros"][0]["textos"][0]["quem"] = "siddhartha"
    _invalido(d, "recordatório é do narrador")
    d = _base()
    del d["quadros"][1]["textos"][0]["quem"]
    _invalido(d, "fala precisa de 'quem'")


def test_texto_longo_demais_para_um_balao():
    d = _base()
    d["quadros"][1]["textos"][0]["texto"] = "palavra " * 30
    _invalido(d, "divida em dois balões")


def test_id_de_personagem_vira_nome_de_pasta_entao_e_restrito():
    d = _base()
    d["personagens"]["Channa X"] = copy.deepcopy(d["personagens"].pop("channa"))
    for q in d["quadros"]:
        q["personagens"] = ["Channa X" if p == "channa" else p for p in q["personagens"]]
        for t in q["textos"]:
            if t.get("quem") == "channa":
                t["quem"] = "Channa X"
    _invalido(d, "id de personagem inválido")


def test_tira_com_quatro_quadros():
    d = _base()
    d["paginas"][0]["tiras"] = [[1], [2, 3, 4, 5], [6]]
    _invalido(d, "tira com 4 quadros")
