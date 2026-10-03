"""Cliente da API do InvokeAI (fila, imagens, boards, modelos).

O InvokeAI é o servidor de geração de imagem da máquina (mantido pelo
`ai-stack`, não por este repo): modelos já quantizados para os 16 GB, fila
serializando a GPU e Canvas para retoque manual. Este repo só monta os grafos
e fala HTTP — nenhum peso, nenhuma venv de diffusers aqui.

Os nomes de nós/campos são os do InvokeAI 6.14 (validados no spike de HQ,
`spikes/hq/invoke.py`, hoje só no histórico do git). Os nomes de modelo são
os registrados no InvokeAI.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from ..config import settings

# Nomes como estão registrados no InvokeAI desta máquina (ver ai-stack).
FLUX_SCHNELL = "FLUX.1 schnell (quantized)"
FLUX_VAE = "FLUX.1-schnell_ae"
FLUX_T5 = "T5-XXL (video-factory)"
FLUX_CLIP = "CLIP-L (video-factory)"
QWEN_EDIT = "Qwen Image Edit 2511 (Q4_K_M)"
QWEN_VAE = "Qwen Image VAE"
QWEN_VL = "Qwen2.5-VL Encoder (fp8 scaled)"
QWEN_LIGHTNING = "Qwen Image Edit Lightning (4-step, bf16)"


class InvokeAIErro(RuntimeError):
    pass


def _e(src: str, sf: str, dst: str, df: str) -> dict:
    return {"source": {"node_id": src, "field": sf}, "destination": {"node_id": dst, "field": df}}


class InvokeAI:
    def __init__(self, base: str | None = None, intervalo_s: float = 0.5,
                 timeout_s: float = 60.0):
        self.base = (base or settings().invokeai_url).rstrip("/")
        self.intervalo_s = intervalo_s
        self.timeout_s = timeout_s
        self._modelos: list[dict] | None = None

    # ------------------------------------------------------------- HTTP
    def _req(self, method: str, path: str, body: bytes | None = None,
             headers: dict | None = None):
        r = urllib.request.Request(self.base + path, data=body, method=method,
                                   headers=headers or {})
        try:
            with urllib.request.urlopen(r, timeout=self.timeout_s) as resp:
                data = resp.read()
                tipo = resp.headers.get_content_type()
        except urllib.error.HTTPError as e:
            raise InvokeAIErro(f"{method} {path}: HTTP {e.code} {e.read()[:500]!r}") from e
        except urllib.error.URLError as e:
            raise InvokeAIErro(f"InvokeAI fora do ar em {self.base} ({e.reason})") from e
        return json.loads(data) if data and tipo == "application/json" else data

    def _get(self, path: str):
        return self._req("GET", path)

    def _post_json(self, path: str, obj):
        return self._req("POST", path, json.dumps(obj).encode(),
                         {"Content-Type": "application/json"})

    # ------------------------------------------------------- utilidades
    def versao(self) -> str:
        return self._get("/api/v1/app/version")["version"]

    def disponivel(self) -> bool:
        try:
            self.versao()
            return True
        except InvokeAIErro:
            return False

    def liberar_vram(self) -> None:
        """Descarrega os modelos que o InvokeAI mantém na GPU entre gerações
        (medido: 7 GB parados depois de um lote FLUX, o bastante para o
        ACE-Step estourar os 16 GB). A próxima geração recarrega sozinha."""
        self._req("POST", "/api/v2/models/empty_model_cache")

    def modelo(self, nome: str) -> dict:
        if self._modelos is None:
            self._modelos = self._get("/api/v2/models/")["models"]
        for m in self._modelos:
            if m["name"] == nome:
                return {k: m[k] for k in ("key", "hash", "name", "base", "type")}
        raise InvokeAIErro(f"modelo não instalado no InvokeAI: {nome!r} "
                           f"(instalados: {', '.join(m['name'] for m in self._modelos)})")

    def board(self, nome: str) -> str:
        """Board homônimo (cria se não existir) — onde o operador vê e retoca."""
        for b in self._get("/api/v1/boards/?all=true"):
            if b["board_name"] == nome:
                return b["board_id"]
        return self._req("POST", "/api/v1/boards/?board_name=" + urllib.parse.quote(nome))["board_id"]

    def upload(self, caminho: Path, board_id: str | None = None) -> str:
        """Sobe uma imagem local; devolve o image_name no InvokeAI."""
        limite = uuid.uuid4().hex
        corpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"{caminho.name}\"\r\nContent-Type: image/png\r\n\r\n").encode()
        corpo += caminho.read_bytes() + f"\r\n--{limite}--\r\n".encode()
        q = "?image_category=user&is_intermediate=false"
        if board_id:
            q += f"&board_id={board_id}"
        r = self._req("POST", "/api/v1/images/upload" + q, corpo,
                      {"Content-Type": f"multipart/form-data; boundary={limite}"})
        return r["image_name"]

    def baixar(self, image_name: str, destino: Path) -> Path:
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_name(destino.name + ".parcial")
        tmp.write_bytes(self._req("GET", f"/api/v1/images/i/{image_name}/full"))
        tmp.replace(destino)  # nunca deixa PNG pela metade com o nome final
        return destino

    def apagar(self, image_name: str) -> None:
        self._req("DELETE", f"/api/v1/images/i/{image_name}")

    # ------------------------------------------------------------- fila
    def enfileirar(self, nodes: dict, edges: list) -> int:
        graph = {"id": uuid.uuid4().hex, "nodes": nodes, "edges": edges}
        r = self._post_json("/api/v1/queue/default/enqueue_batch",
                            {"batch": {"graph": graph, "runs": 1}, "prepend": False})
        return r["item_ids"][0]

    def esperar(self, item: int) -> str:
        """Espera o item terminar e devolve o image_name da saída."""
        while True:
            q = self._get(f"/api/v1/queue/default/i/{item}")
            if q["status"] in ("completed", "failed", "canceled"):
                break
            time.sleep(self.intervalo_s)
        if q["status"] != "completed":
            raise InvokeAIErro(f"item {item} {q['status']}: "
                               f"{(q.get('error_traceback') or q.get('error_message') or '')[-800:]}")
        for res in q["session"]["results"].values():
            if res.get("type") == "image_output":
                return res["image"]["image_name"]
        raise InvokeAIErro(f"item {item} terminou sem imagem de saída")

    # ----------------------------------------------------------- grafos
    def grafo_flux(self, prompt: str, largura: int, altura: int, seed: int,
                   passos: int = 4, guidance: float = 0.0,
                   board_id: str | None = None) -> tuple[dict, list]:
        """FLUX.1-schnell NF4: destilado, 4 passos e guidance 0 por padrão."""
        dec = {"id": "dec", "type": "flux_vae_decode", "is_intermediate": False}
        if board_id:
            dec["board"] = {"board_id": board_id}
        nodes = {
            "loader": {"id": "loader", "type": "flux_model_loader",
                       "model": self.modelo(FLUX_SCHNELL),
                       "t5_encoder_model": self.modelo(FLUX_T5),
                       "clip_embed_model": self.modelo(FLUX_CLIP),
                       "vae_model": self.modelo(FLUX_VAE)},
            "te": {"id": "te", "type": "flux_text_encoder", "t5_max_seq_len": 256,
                   "prompt": prompt},
            "dn": {"id": "dn", "type": "flux_denoise", "width": largura, "height": altura,
                   "num_steps": passos, "guidance": guidance, "seed": seed},
            "dec": dec,
        }
        edges = [_e("loader", "clip", "te", "clip"), _e("loader", "t5_encoder", "te", "t5_encoder"),
                 _e("loader", "transformer", "dn", "transformer"),
                 _e("te", "conditioning", "dn", "positive_text_conditioning"),
                 _e("dn", "latents", "dec", "latents"), _e("loader", "vae", "dec", "vae")]
        return nodes, edges

    def grafo_qwen_edit(self, prompt: str, refs: list[str], largura: int, altura: int,
                        seed: int, board_id: str | None = None,
                        lightning: bool = True,
                        loras: list[tuple[str, float]] = ()) -> tuple[dict, list]:
        """Qwen-Image-Edit 2511. `refs[0]` vai também para o espaço latente (o
        InvokeAI só aceita UMA reference_latents); as demais entram só pelo
        encoder de visão.

        `loras`: (nome no InvokeAI, peso), encadeadas depois da Lightning —
        por exemplo a de style transfer da HQ."""
        dec = {"id": "dec", "type": "qwen_image_l2i", "is_intermediate": False}
        if board_id:
            dec["board"] = {"board_id": board_id}
        nodes = {
            "loader": {"id": "loader", "type": "qwen_image_model_loader",
                       "model": self.modelo(QWEN_EDIT),
                       "vae_model": self.modelo(QWEN_VAE),
                       "qwen_vl_encoder_model": self.modelo(QWEN_VL)},
            "te": {"id": "te", "type": "qwen_image_text_encoder", "prompt": prompt,
                   "reference_images": [{"image_name": r} for r in refs]},
            "i2l": {"id": "i2l", "type": "qwen_image_i2l", "image": {"image_name": refs[0]}},
            "dn": {"id": "dn", "type": "qwen_image_denoise", "width": largura,
                   "height": altura, "seed": seed,
                   **({"steps": 4, "cfg_scale": 1.0, "shift": 3.0} if lightning
                      else {"steps": 40, "cfg_scale": 4.0})},
            "dec": dec,
        }
        edges = [_e("loader", "qwen_vl_encoder", "te", "qwen_vl_encoder"),
                 _e("loader", "vae", "i2l", "vae"), _e("loader", "vae", "dec", "vae"),
                 _e("te", "conditioning", "dn", "positive_conditioning"),
                 _e("i2l", "latents", "dn", "reference_latents"),
                 _e("dn", "latents", "dec", "latents")]
        cadeia = ([(QWEN_LIGHTNING, 1.0)] if lightning else []) + list(loras)
        anterior = "loader"
        for i, (nome, peso) in enumerate(cadeia):
            nid = f"lora{i}"
            nodes[nid] = {"id": nid, "type": "qwen_image_lora_loader", "weight": peso,
                          "lora": self.modelo(nome)}
            edges.append(_e(anterior, "transformer", nid, "transformer"))
            anterior = nid
        edges.append(_e(anterior, "transformer", "dn", "transformer"))
        return nodes, edges

    # --------------------------------------------------------- em lote
    def gerar_lote(self, grafos: list[tuple[tuple[dict, list], Path]], *,
                   em_voo: int = 3, manter_no_invokeai: bool = False,
                   progresso=None) -> list[Path]:
        """Gera e baixa cada grafo para o seu destino, na ordem.

        Mantém `em_voo` itens na fila do InvokeAI: o bastante para a GPU não
        esperar o cliente baixar a imagem anterior, pouco o bastante para que
        um processo morto no meio de um lote de 700 não deixe a fila gerando
        sozinha — a retomada é pelo destino já existir, e itens órfãos seriam
        gerados em dobro.

        A cópia no InvokeAI é apagada depois de baixada, a não ser que
        `manter_no_invokeai`: o disco do repo é a fonte da verdade, e as
        imagens de um capítulo não têm por que se acumular na galeria."""
        pendentes = list(grafos)
        voando: list[tuple[int, Path]] = []
        feitos: list[Path] = []
        while pendentes or voando:
            while pendentes and len(voando) < em_voo:
                g, destino = pendentes.pop(0)
                voando.append((self.enfileirar(*g), destino))
            item, destino = voando.pop(0)
            nome = self.esperar(item)
            self.baixar(nome, destino)
            if not manter_no_invokeai:
                self.apagar(nome)
            feitos.append(destino)
            if progresso:
                progresso(f"imagem {len(feitos)}/{len(grafos)}: {destino.name}")
        return feitos


_padrao: InvokeAI | None = None


def liberar_vram_se_no_ar() -> bool:
    """Antes de qualquer trabalho local de GPU (TTS, música): pede ao InvokeAI
    que solte a VRAM. Fora do ar não é erro — não há nada a soltar."""
    try:
        cliente().liberar_vram()
        return True
    except InvokeAIErro:
        return False


def cliente() -> InvokeAI:
    """Cliente do InvokeAI da configuração central (`invokeai_url`)."""
    global _padrao
    if _padrao is None:
        _padrao = InvokeAI()
    return _padrao
