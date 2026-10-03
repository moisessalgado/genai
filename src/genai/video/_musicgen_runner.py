"""Ponte para o MusicGen Stereo, executada na venv isolada (`~/ai/envs/musica-musicgen`).

Este arquivo NAO e importado pelo pacote. Ele roda como script sob outro
interpretador -- mesmo motivo do `_ace_runner.py`: o MusicGen tem seu proprio
conjunto de dependencias (`transformers`, `torch`), e nao ha razao para elas
conviverem com as fixadas pela venv do ACE-Step ou pelo resto do pipeline.

A unica coisa que atravessa a fronteira e um JSON pela linha de comando e um
WAV no disco: nenhum objeto Python, nenhuma versao compartilhada.
"""
import json
import os
import sys


def main() -> int:
    pedido = json.loads(sys.argv[1])
    # Atribuicao, nao setdefault: o HF_HOME herdado do ambiente perderia para
    # o que a configuracao central (vf.toml) mandou.
    os.environ["HF_HOME"] = pedido["hf_home"]

    import soundfile as sf
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(pedido["checkpoint"])
    modelo = MusicgenForConditionalGeneration.from_pretrained(
        pedido["checkpoint"], torch_dtype=torch.bfloat16).to("cuda")
    sr = modelo.config.audio_encoder.sampling_rate
    # Quadros de codigo por segundo -- especifico do codec (EnCodec) que este
    # checkpoint usa, e o que traduz duracao em `max_new_tokens`.
    quadros_por_s = modelo.config.audio_encoder.frame_rate

    for peca in pedido["pecas"]:
        if os.path.exists(peca["destino"]):
            continue
        # Seed fixada por peca, nao globalmente: `do_sample=True` consome o
        # gerador padrao do torch, entao sem isso a segunda peca do lote
        # dependeria da primeira ter rodado antes -- reprodutibilidade quebrada.
        torch.manual_seed(peca["seed"])
        entradas = processor(text=[peca["prompt"]], padding=True,
                             return_tensors="pt").to("cuda")
        audio = modelo.generate(
            **entradas, do_sample=True, guidance_scale=pedido["guidance"],
            max_new_tokens=int(pedido["duracao_s"] * quadros_por_s))
        # (1, canais, amostras) -> (amostras, canais), formato que soundfile espera.
        onda = audio[0].to(torch.float32).cpu().numpy().T
        sf.write(peca["destino"], onda.clip(-1.0, 1.0), sr)
        print(f"ok {peca['destino']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
