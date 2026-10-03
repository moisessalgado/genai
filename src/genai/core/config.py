"""Configuração central: caminhos e serviços desta máquina.

Antes, cada módulo deduzia a raiz do repositório (`parents[2]`, `parents[3]`) e
montava `RAIZ / "models"`, `RAIZ / ".venv-imagem"`, a URL do Ollama etc. por
conta própria. Agora tudo sai daqui, e trocar um caminho (por exemplo, as venvs
isoladas de música) é editar um arquivo, não o código.

Precedência, do mais fraco ao mais forte:

1. padrão no código (dados dentro do repositório; pesos em `~/ai/hf` e venvs
   isoladas em `~/ai/envs`, o layout do ai-stack);
2. `HF_HOME` do ambiente, só para `hf_home` (é a variável da máquina inteira);
3. o arquivo TOML: `$VF_CONFIG`, ou `vf.toml` na raiz (fora do git; ver
   `vf.example.toml`);
4. variáveis `VF_<CAMPO>` (`VF_PROJECTS_DIR`, `VF_LLM_URL`, ...).

Caminho relativo (no arquivo ou no ambiente) é relativo à raiz; `~` expande.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

RAIZ_PADRAO = Path(__file__).resolve().parents[3]  # core/ -> genai/ -> src/ -> raiz

# Seção do TOML onde cada campo mora. O nome dentro da seção é o do campo sem o
# prefixo/sufixo redundante: [caminhos] projects = ..., [venvs] imagem = ...
_SECOES: dict[str, tuple[str, str]] = {
    "projects_dir": ("caminhos", "projects"),
    "assets_dir": ("caminhos", "assets"),
    "cache_dir": ("caminhos", "cache"),
    "voices_dir": ("caminhos", "voices"),
    "lexicon_dir": ("caminhos", "lexicon"),
    "books_dir": ("caminhos", "books"),
    "config_dir": ("caminhos", "config"),
    "hf_home": ("caminhos", "hf_home"),
    "venv_musica": ("venvs", "musica"),
    "venv_musica_mg": ("venvs", "musica_mg"),
    "llm_api": ("servicos", "llm_api"),
    "llm_url": ("servicos", "llm_url"),
    "llm_modelo": ("servicos", "llm_modelo"),
    "invokeai_url": ("servicos", "invokeai_url"),
}

# Padrões relativos à raiz -- exatamente o que o código fazia antes.
_PADROES: dict[str, str] = {
    "projects_dir": "projects",
    "assets_dir": "assets",
    "cache_dir": "cache",
    "voices_dir": "voices",
    "lexicon_dir": "lexicon",
    "books_dir": "books",
    "config_dir": "config",
    "hf_home": "~/ai/hf",  # o HF_HOME unico da maquina (ai-stack)
    # Venvs isoladas ficam no ~/ai/envs da maquina (layout do ai-stack); as
    # listas congeladas para recria-las estao em envs/ neste repo.
    "venv_musica": "~/ai/envs/musica-ace",
    "venv_musica_mg": "~/ai/envs/musica-musicgen",
    # "ollama" (API nativa) ou "openai" (API compatível; o LiteLLM do
    # ai-stack em :4000). Ver core/llm.py.
    "llm_api": "ollama",
    "llm_url": "http://localhost:11434",
    "llm_modelo": "gemma4:12b",
    "invokeai_url": "http://127.0.0.1:9090",
}


@dataclass(frozen=True)
class Settings:
    raiz: Path
    projects_dir: Path
    assets_dir: Path
    cache_dir: Path
    voices_dir: Path
    lexicon_dir: Path
    books_dir: Path
    config_dir: Path
    hf_home: Path
    venv_musica: Path
    venv_musica_mg: Path
    llm_api: str
    llm_url: str
    llm_modelo: str
    invokeai_url: str
    # campo -> de onde veio o valor ("padrão", "HF_HOME", "vf.toml", "VF_..."),
    # para o `doctor` poder dizer por que um caminho é o que é.
    origem: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    @property
    def ace_checkpoint(self) -> Path:
        return self.hf_home / "ace-step"


def _caminho(valor: str, raiz: Path) -> Path:
    p = Path(valor).expanduser()
    return p if p.is_absolute() else raiz / p


def carregar(env: dict[str, str] | None = None, arquivo: Path | None = None) -> Settings:
    """Monta as Settings. `env` e `arquivo` existem para os testes; o normal é
    `settings()`, que lê `os.environ` uma vez."""
    env = dict(os.environ) if env is None else env
    raiz = _caminho(env["VF_RAIZ"], Path.cwd()) if env.get("VF_RAIZ") else RAIZ_PADRAO
    origem = {"raiz": "VF_RAIZ" if env.get("VF_RAIZ") else "padrão"}

    if arquivo is None:
        arquivo = _caminho(env["VF_CONFIG"], raiz) if env.get("VF_CONFIG") else raiz / "vf.toml"
    toml: dict = {}
    if arquivo.exists():
        toml = tomllib.loads(arquivo.read_text(encoding="utf-8"))
    elif env.get("VF_CONFIG"):
        raise FileNotFoundError(f"VF_CONFIG aponta para um arquivo que não existe: {arquivo}")

    valores: dict[str, object] = {}
    for f in fields(Settings):
        if f.name in ("raiz", "origem"):
            continue
        secao, chave = _SECOES[f.name]
        var = "VF_" + f.name.upper()
        if env.get(var):
            bruto, de = env[var], var
        elif chave in toml.get(secao, {}):
            bruto, de = str(toml[secao][chave]), arquivo.name
        elif f.name == "hf_home" and env.get("HF_HOME"):
            bruto, de = env["HF_HOME"], "HF_HOME"
        else:
            bruto, de = _PADROES[f.name], "padrão"
        valores[f.name] = bruto if f.type == "str" else _caminho(bruto, raiz)
        origem[f.name] = de
    return Settings(raiz=raiz, origem=origem, **valores)


_atual: Settings | None = None


def settings() -> Settings:
    """As Settings do processo, carregadas na primeira chamada."""
    global _atual
    if _atual is None:
        _atual = carregar()
    return _atual


def recarregar() -> Settings:
    """Esquece o cache e lê de novo (testes, ou depois de mudar o ambiente)."""
    global _atual
    _atual = None
    return settings()
