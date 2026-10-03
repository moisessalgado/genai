"""Diagnóstico da máquina: `doctor`."""
from __future__ import annotations

from pathlib import Path

from rich.table import Table

from ..core.config import settings
from ._comum import console
from .app import app


@app.command()
def doctor():
    """Verifica GPU, torch, FFmpeg, licenças dos pesos e a configuração resolvida."""
    import shutil
    import torch

    ok = lambda b: "[green]OK[/]" if b else "[red]FALHOU[/]"
    console.print(f"torch {torch.__version__} · CUDA {ok(torch.cuda.is_available())}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        console.print(f"GPU {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
        try:
            x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()
            console.print(f"kernels na GPU {ok(True)}")
        except Exception as e:
            console.print(f"kernels na GPU {ok(False)} — {e}")
    console.print(f"ffmpeg {ok(shutil.which('ffmpeg'))} · ffprobe {ok(shutil.which('ffprobe'))}")
    lic = settings().raiz / "LICENSES.md"
    console.print(f"registro de licenças {ok(lic.exists())} — {lic}")
    _mostrar_config()


def _mostrar_config() -> None:
    """Valor final de cada campo do Settings e de onde ele veio."""
    from dataclasses import fields
    from urllib import request

    s = settings()
    t = Table(title="configuração (vf.toml / VF_*)")
    for c in ("campo", "valor", "origem", ""):
        t.add_column(c)
    for f in fields(s):
        if f.name == "origem":
            continue
        v = getattr(s, f.name)
        if isinstance(v, Path):
            estado = "[green]existe[/]" if v.exists() else "[yellow]ausente[/]"
        else:
            estado = ""
            if f.name.endswith("_url"):
                try:
                    request.urlopen(v, timeout=2).close()
                    estado = "[green]responde[/]"
                except Exception as e:
                    # 404 na raiz ainda prova que o serviço está no ar
                    estado = ("[green]responde[/]" if getattr(e, "code", None)
                              else "[yellow]fora do ar[/]")
        t.add_row(f.name, str(v), s.origem.get(f.name, ""), estado)
    console.print(t)
