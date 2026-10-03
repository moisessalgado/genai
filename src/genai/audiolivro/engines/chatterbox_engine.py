"""Motor principal: Chatterbox Multilingual V3 / pack pt-br (TDD 5).

Os conditionals da voz sao calculados uma vez e reaproveitados em todos os chunks
do livro -- e o que impede a identidade da voz de derivar entre capitulos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..script.models import SynthParams
from .base import TTSEngine

REPO_PTBR = "ResembleAI/Chatterbox-Multilingual-pt-br"
REPO_BASE = "ResembleAI/chatterbox"

# O pack pt-br so publica o T3 refinado, o s3gen e o tokenizer -- nao traz o voice
# encoder (ve.pt) nem os conds.pt do modelo base, e usa outros nomes de arquivo.
# `from_local` espera nomes fixos, entao montamos um diretorio composto:
# arquivos do pack quando existem, do base quando faltam.
# O refinamento pt-br esta no T3 (o modelo de linguagem/prosodia). O `s3gen_v3.pt`
# que o pack traz e de uma versao mais nova da lib e nao casa com o S3Token2Wav
# instalado ("Missing key(s): tokenizer._mel_filters, tokenizer.window"), entao o
# vocoder vem do modelo base -- ele e agnostico ao idioma.
_MAPA_PACK = {
    "t3_mtl23ls_v2.safetensors": "t3_pt_br.safetensors",
    "grapheme_mtl_merged_expanded_v1.json": "grapheme_mtl_merged_expanded_v1.json",
}
_DO_BASE = ("ve.pt", "conds.pt", "Cangjie5_TC.json", "s3gen.pt")


def montar_ckpt_ptbr(cache_dir: Path | None = None) -> Path:
    """Compoe o checkpoint pt-br a partir do pack + o que falta do modelo base."""
    from huggingface_hub import snapshot_download

    # allow_patterns e obrigatorio: sem ele o snapshot_download puxa o repo inteiro
    # (varios GB de variantes que nao usamos) em vez dos 5 arquivos necessarios.
    pack = Path(snapshot_download(repo_id=REPO_PTBR, repo_type="model",
                                  allow_patterns=list(_MAPA_PACK.values())))
    base = Path(snapshot_download(repo_id=REPO_BASE, repo_type="model",
                                  allow_patterns=list(_DO_BASE)))
    destino = (cache_dir or pack.parent) / "composed-ptbr"
    destino.mkdir(parents=True, exist_ok=True)

    for alvo, origem in _MAPA_PACK.items():
        src = pack / origem
        if src.exists():
            _link(src, destino / alvo)
    for nome in _DO_BASE:
        src = base / nome
        if src.exists() and not (destino / nome).exists():
            _link(src, destino / nome)
    return destino


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"

    def __init__(self, params: SynthParams, device: str = "cuda",
                 ckpt_dir: Path | None = None, use_ptbr_pack: bool = True):
        self.params = params
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.use_ptbr_pack = use_ptbr_pack
        self.model = None
        self.version = "chatterbox-mtl-v3-ptbr" if use_ptbr_pack else "chatterbox-mtl-v3"

    def load(self) -> None:
        if self.model is not None:      # idempotente: o worker chama sem saber
            return
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if self.ckpt_dir is not None:
            self.model = ChatterboxMultilingualTTS.from_local(str(self.ckpt_dir), self.device)
        elif self.use_ptbr_pack:
            self.model = ChatterboxMultilingualTTS.from_local(
                str(montar_ckpt_ptbr()), self.device)
        else:
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr

    def set_voice(self, reference_wav: Path | None) -> None:
        """Prepara os conditionals a partir da voz de referencia, uma unica vez."""
        if reference_wav is None:
            return
        self.model.prepare_conditionals(str(reference_wav),
                                        exaggeration=self.params.exaggeration)

    def synthesize(self, text: str, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.inference_mode():
            wav = self.model.generate(
                text,
                language_id=self.params.language_id,
                exaggeration=self.params.exaggeration,
                cfg_weight=self.params.cfg_weight,
                temperature=self.params.temperature,
                repetition_penalty=self.params.repetition_penalty,
            )
        return wav.detach().cpu().squeeze().numpy().astype(np.float32)
