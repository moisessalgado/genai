"""Cliente mínimo da API do InvokeAI (fila + imagens + boards).

Spike de HQ: o InvokeAI é o servidor de geração (modelos já ajustados para os
16 GB, fila serializando a GPU, Canvas para retoque manual). Este módulo monta
os grafos na mão — os nomes de nós/campos são os do InvokeAI 6.14.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:9090"


def _req(method: str, path: str, body: bytes | None = None,
         headers: dict | None = None):
    r = urllib.request.Request(BASE + path, data=body, method=method,
                               headers=headers or {})
    with urllib.request.urlopen(r) as resp:
        data = resp.read()
    return json.loads(data) if data and resp.headers.get_content_type() == "application/json" else data


def _get(path): return _req("GET", path)


def _post_json(path, obj):
    return _req("POST", path, json.dumps(obj).encode(), {"Content-Type": "application/json"})


_modelos: list[dict] | None = None


def modelo(nome: str) -> dict:
    global _modelos
    if _modelos is None:
        _modelos = _get("/api/v2/models/")["models"]
    m = next(m for m in _modelos if m["name"] == nome)
    return {k: m[k] for k in ("key", "hash", "name", "base", "type")}


def board(nome: str) -> str:
    """Board do projeto no InvokeAI (cria se não existir) — é onde o operador vê e retoca."""
    for b in _get("/api/v1/boards/?all=true"):
        if b["board_name"] == nome:
            return b["board_id"]
    return _req("POST", "/api/v1/boards/?board_name=" + urllib.parse.quote(nome))["board_id"]


def upload(caminho: Path, board_id: str | None = None) -> str:
    """Sobe uma imagem local; devolve o image_name no InvokeAI."""
    limite = uuid.uuid4().hex
    corpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{caminho.name}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    corpo += caminho.read_bytes() + f"\r\n--{limite}--\r\n".encode()
    q = "?image_category=user&is_intermediate=false"
    if board_id:
        q += f"&board_id={board_id}"
    r = _req("POST", "/api/v1/images/upload" + q, corpo,
             {"Content-Type": f"multipart/form-data; boundary={limite}"})
    return r["image_name"]


def baixar(image_name: str, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(_req("GET", f"/api/v1/images/i/{image_name}/full"))
    return destino


def _executar(nodes: dict, edges: list) -> str:
    """Enfileira o grafo, espera, devolve o image_name da saída (nó não-intermediário)."""
    graph = {"id": uuid.uuid4().hex, "nodes": nodes, "edges": edges}
    item = _post_json("/api/v1/queue/default/enqueue_batch",
                      {"batch": {"graph": graph, "runs": 1}, "prepend": False})["item_ids"][0]
    while True:
        q = _get(f"/api/v1/queue/default/i/{item}")
        if q["status"] in ("completed", "failed", "canceled"):
            break
        time.sleep(1.5)
    if q["status"] != "completed":
        raise RuntimeError(f"item {item} {q['status']}: {(q.get('error_traceback') or '')[-800:]}")
    for res in q["session"]["results"].values():
        if res.get("type") == "image_output":
            return res["image"]["image_name"]
    raise RuntimeError(f"item {item} sem imagem de saída")


def _e(src, sf, dst, df):
    return {"source": {"node_id": src, "field": sf}, "destination": {"node_id": dst, "field": df}}


def flux(prompt: str, w: int, h: int, seed: int, board_id: str | None = None) -> str:
    """FLUX.1-schnell NF4: 4 passos, guidance 0 (destilado)."""
    dec = {"id": "dec", "type": "flux_vae_decode", "is_intermediate": False}
    if board_id:
        dec["board"] = {"board_id": board_id}
    nodes = {
        "loader": {"id": "loader", "type": "flux_model_loader",
                   "model": modelo("FLUX.1 schnell (quantized)"),
                   "t5_encoder_model": modelo("T5-XXL (video-factory)"),
                   "clip_embed_model": modelo("CLIP-L (video-factory)"),
                   "vae_model": modelo("FLUX.1-schnell_ae")},
        "te": {"id": "te", "type": "flux_text_encoder", "t5_max_seq_len": 256, "prompt": prompt},
        "dn": {"id": "dn", "type": "flux_denoise", "width": w, "height": h,
               "num_steps": 4, "guidance": 0.0, "seed": seed},
        "dec": dec,
    }
    edges = [_e("loader", "clip", "te", "clip"), _e("loader", "t5_encoder", "te", "t5_encoder"),
             _e("loader", "transformer", "dn", "transformer"),
             _e("te", "conditioning", "dn", "positive_text_conditioning"),
             _e("dn", "latents", "dec", "latents"), _e("loader", "vae", "dec", "vae")]
    return _executar(nodes, edges)


def qwen_edit(prompt: str, refs: list[str], w: int, h: int, seed: int,
              board_id: str | None = None, lightning: bool = True) -> str:
    """Qwen-Image-Edit 2511. `refs[0]` vai também para o espaço latente (o InvokeAI
    só aceita UMA reference_latents); as demais entram só pelo encoder de visão."""
    dec = {"id": "dec", "type": "qwen_image_l2i", "is_intermediate": False}
    if board_id:
        dec["board"] = {"board_id": board_id}
    nodes = {
        "loader": {"id": "loader", "type": "qwen_image_model_loader",
                   "model": modelo("Qwen Image Edit 2511 (Q4_K_M)"),
                   "vae_model": modelo("Qwen Image VAE"),
                   "qwen_vl_encoder_model": modelo("Qwen2.5-VL Encoder (fp8 scaled)")},
        "te": {"id": "te", "type": "qwen_image_text_encoder", "prompt": prompt,
               "reference_images": [{"image_name": r} for r in refs]},
        "i2l": {"id": "i2l", "type": "qwen_image_i2l", "image": {"image_name": refs[0]}},
        "dn": {"id": "dn", "type": "qwen_image_denoise", "width": w, "height": h, "seed": seed,
               **({"steps": 4, "cfg_scale": 1.0, "shift": 3.0} if lightning
                  else {"steps": 40, "cfg_scale": 4.0})},
        "dec": dec,
    }
    edges = [_e("loader", "qwen_vl_encoder", "te", "qwen_vl_encoder"),
             _e("loader", "vae", "i2l", "vae"), _e("loader", "vae", "dec", "vae"),
             _e("te", "conditioning", "dn", "positive_conditioning"),
             _e("i2l", "latents", "dn", "reference_latents"),
             _e("dn", "latents", "dec", "latents")]
    if lightning:
        nodes["lora"] = {"id": "lora", "type": "qwen_image_lora_loader", "weight": 1.0,
                         "lora": modelo("Qwen Image Edit Lightning (4-step, bf16)")}
        edges += [_e("loader", "transformer", "lora", "transformer"),
                  _e("lora", "transformer", "dn", "transformer")]
    else:
        edges.append(_e("loader", "transformer", "dn", "transformer"))
    return _executar(nodes, edges)
