"""Ponte para o ACE-Step, executada na venv isolada (`~/ai/envs/musica-ace`).

Este arquivo NAO e importado pelo pacote. Ele roda como script sob outro
interpretador, porque o `acestep` fixa `transformers==4.50` e `datasets==3.4`,
versoes que nao convivem com as do pipeline de TTS. Manter os dois em uma venv so
significaria escolher qual dos dois quebra.

Por isso a unica coisa que atravessa a fronteira e um JSON pela linha de comando
e um WAV no disco: nenhum objeto Python, nenhuma versao compartilhada.
"""
import json
import os
import sys


def _salvar_com_soundfile() -> None:
    """Faz `torchaudio.save` voltar a gravar WAV, com soundfile.

    O ACE-Step chama `torchaudio.save(..., backend="soundfile")`, mas o torchaudio
    2.11+ ignora o argumento e roteia tudo para o `torchcodec`, que nao esta
    instalado -- a geracao roda inteira na GPU e morre na ultima linha, ao gravar.

    E a mesma armadilha ja registrada no ESTADO.md para a venv do TTS, e a mesma
    saida: soundfile. Instalar `torchcodec` resolveria tambem, ao custo de mais um
    pacote que precisa casar com a versao do torch E com o FFmpeg do sistema --
    exatamente o tipo de acoplamento que motivou separar as venvs.
    """
    import soundfile as sf
    import torchaudio

    def save(uri, src, sample_rate, **kwargs):
        # torchaudio entrega (canais, amostras); soundfile quer (amostras, canais)
        sf.write(str(uri), src.detach().cpu().numpy().T, int(sample_rate))

    torchaudio.save = save


def main() -> int:
    pedido = json.loads(sys.argv[1])
    # Atribuicao, nao setdefault: o HF_HOME herdado do ambiente perderia para
    # o que a configuracao central (vf.toml) mandou.
    os.environ["HF_HOME"] = pedido["hf_home"]

    _salvar_com_soundfile()
    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline(checkpoint_dir=pedido["checkpoint"], dtype="bfloat16")
    for peca in pedido["pecas"]:
        if os.path.exists(peca["destino"]):
            continue
        pipe(
            prompt=peca["prompt"],
            # `[inst]` e o que desliga o cantor. Sem isso o modelo inventa uma
            # voz -- e uma segunda voz por baixo da narracao seria o pior
            # resultado possivel aqui.
            lyrics="[inst]",
            audio_duration=pedido["duracao_s"],
            infer_step=pedido["passos"],
            guidance_scale=pedido["guidance"],
            manual_seeds=[peca["seed"]],
            batch_size=1,
            save_path=peca["destino"],
        )
        print(f"ok {peca['destino']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
