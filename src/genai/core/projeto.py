"""Ciclo de vida de um projeto: pastas, project.yaml e construcao do script.json."""
from __future__ import annotations

from pathlib import Path

import yaml

from .config import settings
from ..audiolivro.chunk.splitter import split_paragraph
from .ingest.loader import detectar_capitulos, limpar, ler, paragrafos
from ..audiolivro.text.roles import dividir_por_papel, tem_dialogo
from ..audiolivro.narration.rules import detect_ambiguous, normalize
from ..audiolivro.script.models import Chapter, Rights, Script, Segment, SynthParams

# Alias de compatibilidade: o lugar certo agora e `config.settings()`.
RAIZ = settings().raiz

# Valores convencionais de `rights.status`. Servem para consulta e para o registro
# no project.yaml -- NAO sao uma autorizacao: o export nao e bloqueado por eles.
# A decisao editorial sobre o que publicar e do operador do canal, nao da ferramenta.
RIGHTS_CONHECIDOS = {"dominio-publico", "proprio", "licenciado", "teste-local"}


def dir_projeto(slug: str, raiz: Path | None = None) -> Path:
    return raiz / "projects" / slug if raiz else settings().projects_dir / slug


def criar(slug: str, fonte: Path, narrator: str, raiz: Path | None = None,
          titulo: str | None = None, rights: dict | None = None,
          extras: dict | None = None) -> Path:
    """Cria projects/<slug>/project.yaml.

    `rights` e `extras` existem para ingest automatizado (o `sutta`, que colhe
    procedencia da propria pagina): quem digita o project.yaml a mao passa None e
    preenche depois. `extras` vai para o topo do arquivo, ao lado de `slug`.
    """
    proj = dir_projeto(slug, raiz)
    proj.mkdir(parents=True, exist_ok=True)
    cfg = {
        "slug": slug,
        "titulo": titulo or fonte.stem,
        "fonte": str(fonte),
        "narrator": narrator,
        # papel -> voice_id. Papel ausente usa a voz do narrator.
        # Papéis detectados automaticamente: "citacao" (fala entre aspas).
        # Marcação explícita no texto: [[voz:personagem_a]] ...
        "cast": {},
        "rights": rights or {"status": "PREENCHER", "autor": None, "ano_morte": None,
                             "tradutor": None, "fonte": None, "licenca": None,
                             "verificado_em": None},
        "params": SynthParams().model_dump(),
    }
    cfg.update(extras or {})
    (proj / "project.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return proj


def carregar_config(proj: Path) -> dict:
    return yaml.safe_load((proj / "project.yaml").read_text(encoding="utf-8"))


def ingerir(proj: Path) -> str:
    cfg = carregar_config(proj)
    texto = limpar(ler(Path(cfg["fonte"])))
    (proj / "clean.txt").write_text(texto, encoding="utf-8")
    return texto


def montar_script(proj: Path, lexicon: dict[str, str] | None = None,
                  max_chars: int = 300, usar_llm: bool = False,
                  modelo_llm: str | None = None,
                  capitulos: list[tuple[str, str]] | None = None,
                  elenco: dict[str, str] | None = None) -> Script:
    """clean.txt -> script.json + diff.md. Nada vai ao TTS sem o diff em disco.

    Ordem obrigatoria: Camada 2 (LLM, sobre o texto CRU) e so depois as regras.
    Invertido, os offsets dos spans nao valem mais nada.

    `capitulos` deixa quem chama IMPOR a divisão em capítulos, pulando
    `detectar_capitulos()` -- necessário quando a fonte já vem com a divisão
    certa e o texto tem marcadores internos que o detector genérico casaria
    por engano (medido: uma página de capítulo da Wikisource com subseções em
    romano tipo "I — NARIZINHO" fragmenta em 8 pedaços via o detector, quando
    a página inteira é UM capítulo só).

    `elenco` (chave de personagem -> voice_id) liga `narration/elenco.py`: cada
    fala `citacao` tenta ser promovida a `personagem_<chave>` usando o
    narrador vizinho como contexto, e cai em `citacao` (voz genérica) quando o
    LLM não tem como dizer com segurança quem fala. `_generico` no dicionário
    é a voz do `citacao` em si, não um personagem -- fica de fora da lista que
    o LLM escolhe (ver `elenco.aplicar_elenco`).
    """
    cfg = carregar_config(proj)
    texto = (proj / "clean.txt").read_text(encoding="utf-8")

    script = Script(
        title=cfg["titulo"],
        voice_id=cfg["narrator"],
        cast=cfg.get("cast") or {},
        params=SynthParams(**cfg.get("params", {})),
        rights=Rights(**cfg["rights"]) if cfg.get("rights", {}).get("status") != "PREENCHER" else None,
    )

    cache_path = proj / "llm_cache.json"
    cache: dict[str, str] = {}
    if usar_llm and cache_path.exists():
        import json as _json
        cache = _json.loads(cache_path.read_text(encoding="utf-8"))

    elenco_cache_path = proj / "elenco_cache.json"
    elenco_cache: dict[str, str | None] = {}
    if elenco and elenco_cache_path.exists():
        import json as _json
        elenco_cache = _json.loads(elenco_cache_path.read_text(encoding="utf-8"))
    personagens_atribuidos = 0

    linhas_diff: list[str] = [f"# Diff de narração — {cfg['titulo']}\n",
                              "Alterações do texto original para o texto narrado.\n"]
    ambiguos = rejeitadas = 0
    for c_idx, (titulo, corpo) in enumerate(capitulos or detectar_capitulos(texto), start=1):
        cap = Chapter(idx=c_idx, title=titulo)
        linhas_diff.append(f"\n## {titulo}\n")
        s_idx = 0
        for par in paragrafos(corpo):
            partes = dividir_por_papel(par)
            if elenco:
                from ..audiolivro.narration.elenco import aplicar_elenco

                partes = aplicar_elenco(partes, elenco, cache=elenco_cache)
                personagens_atribuidos += sum(1 for p, _ in partes if p.startswith("personagem_"))
            for papel, fonte in partes:
                if usar_llm:
                    from ..audiolivro.narration.llm import resolver

                    amb = detect_ambiguous(fonte)
                    ambiguos += len(amb)
                    if amb:
                        fonte, sugestoes = resolver(fonte, amb, cache=cache,
                                                    **({"modelo": modelo_llm} if modelo_llm else {}))
                        for s in sugestoes:
                            if s.aceita:
                                linhas_diff.append(f"  - [LLM] `{s.span}` → `{s.expansao}`")
                            else:
                                rejeitadas += 1
                                linhas_diff.append(
                                    f"  - [LLM rejeitado, mantido original] `{s.span}`: {s.motivo}")
                r = normalize(fonte, lexicon)
                if not usar_llm:
                    ambiguos += len(r.ambiguous)
                if r.applied:
                    linhas_diff.append(f"- `{par[:70]}…`")
                    for orig, novo in r.applied:
                        linhas_diff.append(f"  - `{orig}` → `{novo}`")
                for pedaco in split_paragraph(r.text, max_chars=max_chars):
                    cap.segments.append(Segment(idx=s_idx, source=par, text=pedaco,
                                                role=papel))
                    s_idx += 1
        if cap.segments:
            script.chapters.append(cap)

    papeis = {s.role for c in script.chapters for s in c.segments}
    if papeis - {"narrador"} and not script.cast:
        linhas_diff.append(
            f"\n> ⚠️ Papéis detectados sem voz atribuída: {sorted(papeis - {'narrador'})}. "
            "Defina `cast:` no project.yaml para dar voz própria a eles.\n")

    script.save(proj / "script.json")
    if usar_llm:
        import json as _json
        cache_path.write_text(_json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        linhas_diff.append(f"\n---\n\n{ambiguos} spans ambíguos processados pela Camada 2; "
                           f"{rejeitadas} sugestões rejeitadas (original mantido).\n")
    else:
        linhas_diff.append(f"\n---\n\n{ambiguos} spans ambíguos marcados "
                           "(Camada 2 desligada — use `--llm`).\n")
    if elenco:
        import json as _json
        elenco_cache_path.write_text(_json.dumps(elenco_cache, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        n_citacao = sum(1 for c in script.chapters for s in c.segments if s.role == "citacao")
        linhas_diff.append(f"\n{personagens_atribuidos} falas atribuídas a um personagem "
                           f"nomeado; {n_citacao} permanecem em `citacao` (voz genérica de "
                           "diálogo) por atribuição ambígua ou indeterminada.\n")
    (proj / "diff.md").write_text("\n".join(linhas_diff), encoding="utf-8")
    return script
