"""Ponte para FLUX/SD3.5, executada na venv isolada (`.venv-imagem`).

Este arquivo NAO e importado pelo pacote. Ele roda como script sob outro
interpretador — mesmo motivo do `_ace_runner.py`: os pesos de imagem exigem
versoes de `torch`/`diffusers`/`transformers` que nao precisam (e nao devem)
conviver com as fixadas pelo resto do pipeline. A unica coisa que atravessa a
fronteira e um JSON (por ARQUIVO, nao argv — um lote de ~300 pedidos do preset
`sincronizado` estoura o limite de tamanho de linha de comando do SO) e PNGs
no disco.
"""
import json
import os
import sys


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        pedido = json.load(f)
    os.environ.setdefault("HF_HOME", pedido["hf_home"])

    import torch
    from diffusers import FluxPipeline, StableDiffusion3Pipeline

    familia = pedido["familia"]
    classe = FluxPipeline if familia == "flux" else StableDiffusion3Pipeline
    pipe = classe.from_pretrained(pedido["modelo_repo"], torch_dtype=torch.bfloat16)
    if familia == "flux":
        # FLUX-schnell tem um transformer de 12B sozinho -- em bf16 isso e
        # ~24 GB, maior que os 16 GB da placa. O offload por MODULO (que basta
        # para SD3.5) ainda tenta subir esse submodulo inteiro de uma vez e
        # estoura (medido: OOM pedindo so mais 54 MiB com 14,7 GiB ja
        # alocados). O offload SEQUENCIAL sobe camada por camada -- mais
        # lento, mas e o unico jeito de caber.
        pipe.enable_sequential_cpu_offload()
    else:
        # Mantem so o submodulo ativo (transformer, text encoders, VAE) na GPU
        # por vez -- e o que faz SD3.5-Large (8B + T5-XXL) caber em 16 GB. Para
        # SD3.5-Medium (2.5B, cabe inteiro) o custo e so um pouco de troca de
        # dispositivo entre submodulos, irrelevante perto do tempo de
        # inferencia.
        pipe.enable_model_cpu_offload()

    for p in pedido["pedidos"]:
        if os.path.exists(p["destino"]):
            continue
        # Gerador em CPU, nao GPU: com o offload ligado, os submodulos entram e
        # saem da GPU entre passos, e um gerador preso a um device especifico
        # quebraria ou perderia a reprodutibilidade da seed.
        gerador = torch.Generator(device="cpu").manual_seed(p["seed"])
        kwargs = dict(
            prompt=p["prompt"],
            width=p["largura"],
            height=p["altura"],
            num_inference_steps=p["passos"],
            guidance_scale=p["guidance"],
            generator=gerador,
        )
        # FluxPipeline não aceita negative_prompt (schnell não faz CFG).
        if familia != "flux" and p.get("negative_prompt"):
            kwargs["negative_prompt"] = p["negative_prompt"]
        imagem = pipe(**kwargs).images[0]
        imagem.save(p["destino"])
        print(f"ok {p['destino']}", flush=True)

    if pedido.get("avaliar_clip") or pedido.get("avaliar_estilo"):
        relevancia, estilo = _pontuar(pedido["pedidos"], pedido.get("avaliar_clip", False),
                                      pedido.get("avaliar_estilo", False))
        saida: dict = {}
        if relevancia:
            saida["scores"] = relevancia
        if estilo:
            saida["estilos"] = estilo
        print(json.dumps(saida), flush=True)
    return 0


# Ancoras fixas para o QA de estilo: a diferenca de similaridade CLIP entre a
# imagem e cada uma diz se ela pendeu para foto real em vez de desenho — sem
# olho humano, e o unico sinal barato que temos de que o FLUX ignorou o pedido
# de estilo "storybook illustration" (medido: acontece sobretudo em prompts
# com pessoas, onde o modelo tende a foto-realismo mesmo com o estilo pedido).
ANCORA_FOTO = "a realistic photograph of a real person"
ANCORA_ILUSTRACAO = "a hand-drawn cartoon illustration from a children's picture book, flat colors"


def _pontuar(pedidos: list[dict], relevancia: bool, estilo: bool
            ) -> tuple[dict[str, float], dict[str, float]]:
    """Um unico carregamento do CLIP para as duas notas (relevancia ao prompt
    e diferenca foto-vs-ilustracao) — evita recarregar o modelo/reabrir cada
    imagem duas vezes. `relevancia` so serve para ESCOLHER a melhor entre
    candidatos da mesma janela (nao aprova/reprova, isso e
    `imagem_qa.nao_e_vazia`/`eh_fotorealista` no processo principal)."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    modelo = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processador = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    modelo.eval()

    # `get_text_features`/`get_image_features` sozinhos se mostraram
    # inconsistentes nesta versao de transformers (chegaram a devolver o
    # output cru do encoder, sem a projecao); a forward completa do modelo
    # (`modelo(**entradas)`) e o caminho testado em `qa`/no runner original,
    # e ja devolve `image_embeds`/`text_embeds` normalizados.
    notas_relevancia: dict[str, float] = {}
    notas_estilo: dict[str, float] = {}
    with torch.no_grad():
        for p in pedidos:
            if not os.path.exists(p["destino"]):
                continue
            imagem = Image.open(p["destino"]).convert("RGB")

            textos = []
            if relevancia:
                textos.append(p["prompt"][:300])
            offset_estilo = len(textos)
            if estilo:
                textos.extend([ANCORA_FOTO, ANCORA_ILUSTRACAO])
            if not textos:
                continue

            entradas = processador(text=textos, images=[imagem], return_tensors="pt",
                                   padding=True, truncation=True)
            saida = modelo(**entradas)
            sims = torch.nn.functional.cosine_similarity(
                saida.image_embeds, saida.text_embeds).tolist()

            if relevancia:
                notas_relevancia[p["destino"]] = sims[0]
            if estilo:
                notas_estilo[p["destino"]] = sims[offset_estilo] - sims[offset_estilo + 1]
    return notas_relevancia, notas_estilo


if __name__ == "__main__":
    sys.exit(main())
