"""QA automático dos quadros: reprova o que é defeito objetivo e ordena o resto.

Mesmo espírito do `video/imagem_qa.py` (de onde vêm o teste de quadro vazio e
o CLIP na CPU): checagens baratas e numéricas, e o humano só entra quando
nenhum candidato passa — ou quando quer trocar a escolha (`hq escolher`).

Reprova:
- **vazio**: quadro quase sólido (colapso do modelo);
- **rostos**: menos rostos de primeiro plano que personagens no quadro — o
  modelo perdeu alguém (ou o virou de costas, caso que o humano resolve);
- **estilo**: nota de estilo abaixo de `LIMIAR_ESTILO`.

A nota de estilo é a similaridade CLIP à `ancora_estilo` do roteiro menos a
similaridade a `ANCORA_CONTRA` (HQ moderna). Medida nos 16 quadros do spike:
de +0,058 a −0,066; os de cima têm céu em bokashi e textura de papel, os de
baixo são HQ moderna com cor digital. A nota separa bem — mas no piloto da
estratégia `multi` (a padrão, que encena melhor) TODOS os candidatos ficaram
entre −0,056 e −0,115: com o corte em −0,03 do spike, todo quadro iria para
revisão humana. Por isso ela ORDENA (vence a maior nota entre os aprovados) e
só reprova abaixo de −0,15, fora de tudo o que foi medido. Ressalva medida:
um quadro com artefatos de glitch tirou +0,001 — a nota não é QA de defeito.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..video import imagem_qa
from . import rostos as rostos_mod

ANCORA_CONTRA = "a modern western comic book illustration with digital shading"
LIMIAR_ESTILO = -0.15


def nota_estilo(imagens: list[Path], ancora: str) -> dict[str, float]:
    """{nome do arquivo: similaridade(ancora) - similaridade(HQ moderna)}."""
    import torch

    modelo, proc = imagem_qa._carregar_clip()
    out: dict[str, float] = {}
    with torch.no_grad():
        for p in imagens:
            with Image.open(p) as im:
                img = im.convert("RGB")
            ent = proc(text=[ancora, ANCORA_CONTRA], images=[img], return_tensors="pt",
                       padding=True, truncation=True)
            o = modelo(**ent)
            s = torch.nn.functional.cosine_similarity(o.image_embeds, o.text_embeds).tolist()
            out[p.name] = s[0] - s[1]
    return out


def avaliar(candidatos: list[Path], n_personagens: int, ancora: str | None) -> dict[str, dict]:
    """Notas e veredito de cada candidato: {nome: {...,"ok": bool, "motivo": str}}."""
    estilos = nota_estilo(candidatos, ancora) if ancora else {}
    out: dict[str, dict] = {}
    for p in candidatos:
        n: dict = {"motivo": ""}
        if not imagem_qa.nao_e_vazia(p):
            n.update(ok=False, motivo="vazio")
            out[p.name] = n
            continue
        n["rostos"] = len(rostos_mod.principais(rostos_mod.detectar(p)))
        if p.name in estilos:
            n["estilo"] = round(estilos[p.name], 4)
        if n["rostos"] < n_personagens:
            n["motivo"] = f"{n['rostos']} rosto(s) para {n_personagens} personagem(ns)"
        elif "estilo" in n and n["estilo"] < LIMIAR_ESTILO:
            n["motivo"] = f"estilo {n['estilo']:+.3f} (HQ moderna)"
        n["ok"] = not n["motivo"]
        out[p.name] = n
    return out


def melhor(notas: dict[str, dict]) -> str | None:
    """O aprovado de maior nota de estilo (ou o primeiro, sem nota)."""
    ok = [k for k, v in notas.items() if v.get("ok")]
    if not ok:
        return None
    return max(ok, key=lambda k: notas[k].get("estilo", 0.0))


def rotulo(n: dict) -> str:
    """Resumo curto para a folha de contato."""
    partes = []
    if "estilo" in n:
        partes.append(f"estilo {n['estilo']:+.3f}")
    if "rostos" in n:
        partes.append(f"{n['rostos']} rosto(s)")
    return ("OK " if n.get("ok") else "X ") + " · ".join(partes) + (
        f" — {n['motivo']}" if n.get("motivo") else "")
