"""Precedência da configuração central: código < HF_HOME < vf.toml < VF_*."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.core import config  # noqa: E402


def _toml(tmp_path: Path, texto: str) -> Path:
    f = tmp_path / "vf.toml"
    f.write_text(texto, encoding="utf-8")
    return f


def test_padroes_reproduzem_o_layout_de_sempre(tmp_path):
    s = config.carregar(env={}, arquivo=tmp_path / "nao-existe.toml")
    r = config.RAIZ_PADRAO
    assert s.raiz == r
    assert s.projects_dir == r / "projects"
    assert s.hf_home == Path("~/ai/hf").expanduser()
    assert s.ace_checkpoint == Path("~/ai/hf/ace-step").expanduser()
    assert s.venv_musica == Path("~/ai/envs/musica-ace").expanduser()
    assert s.llm_url == "http://localhost:11434"
    assert s.llm_modelo == "gemma4:12b"
    assert set(s.origem.values()) == {"padrão"}


def test_hf_home_do_ambiente_vence_o_padrao(tmp_path):
    s = config.carregar(env={"HF_HOME": "/srv/hf"}, arquivo=tmp_path / "x.toml")
    assert s.hf_home == Path("/srv/hf")
    assert s.origem["hf_home"] == "HF_HOME"


def test_arquivo_vence_hf_home_e_relativo_e_da_raiz(tmp_path):
    f = _toml(tmp_path, '[caminhos]\nhf_home = "pesos"\nprojects = "~/p"\n'
                        '[venvs]\nmusica = "/opt/envs/musica"\n'
                        '[servicos]\nllm_url = "http://litellm:4000"\n')
    s = config.carregar(env={"HF_HOME": "/srv/hf"}, arquivo=f)
    assert s.hf_home == config.RAIZ_PADRAO / "pesos"
    assert s.projects_dir == Path("~/p").expanduser()
    assert s.venv_musica == Path("/opt/envs/musica")
    assert s.llm_url == "http://litellm:4000"
    assert s.origem["hf_home"] == "vf.toml"
    assert s.origem["cache_dir"] == "padrão"


def test_variavel_vf_vence_o_arquivo(tmp_path):
    f = _toml(tmp_path, '[servicos]\nllm_url = "http://arquivo"\n')
    s = config.carregar(env={"VF_LLM_URL": "http://env", "VF_HF_HOME": "/x"}, arquivo=f)
    assert s.llm_url == "http://env"
    assert s.origem["llm_url"] == "VF_LLM_URL"
    assert s.hf_home == Path("/x")


def test_vf_raiz_move_os_padroes_relativos_mas_nao_os_absolutos(tmp_path):
    s = config.carregar(env={"VF_RAIZ": str(tmp_path)})
    assert s.raiz == tmp_path
    assert s.projects_dir == tmp_path / "projects"
    assert s.venv_musica == Path("~/ai/envs/musica-ace").expanduser()


def test_vf_raiz_le_o_vf_toml_da_nova_raiz(tmp_path):
    _toml(tmp_path, '[caminhos]\ncache = "/tmp/c"\n')
    s = config.carregar(env={"VF_RAIZ": str(tmp_path)})
    assert s.cache_dir == Path("/tmp/c")


def test_vf_config_inexistente_e_erro(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.carregar(env={"VF_CONFIG": str(tmp_path / "sumiu.toml")})


def test_exemplo_versionado_e_toml_valido_e_so_tem_campos_conhecidos():
    import tomllib
    exemplo = config.RAIZ_PADRAO / "vf.example.toml"
    dados = tomllib.loads(exemplo.read_text(encoding="utf-8"))
    conhecidas = set(config._SECOES.values())
    for secao, chaves in dados.items():
        for chave in chaves:
            assert (secao, chave) in conhecidas
    # e descomentar tudo continua valido: cada chave comentada existe no mapa
    for linha in exemplo.read_text(encoding="utf-8").splitlines():
        if linha.startswith("# ") and " = " in linha and not linha.startswith("#  "):
            chave = linha[2:].split("=")[0].strip()
            assert any(chave == c for _, c in conhecidas), chave
