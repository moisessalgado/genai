"""Registry de vozes do canal: `voice check|record|new|template|list|test`."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from ..core.config import settings
from ..audiolivro.engines.chatterbox_engine import ChatterboxEngine
from ._comum import console
from .app import app

voice_app = typer.Typer(help="Registry de vozes do canal", no_args_is_help=True)
app.add_typer(voice_app, name="voice")


def _dispositivos_captura() -> list[tuple[str, str, str]]:
    """Entradas de captura, como (backend, device, descrição).

    PipeWire primeiro: quando ele está rodando, ele abre a placa em modo
    exclusivo, e gravar direto de `hw:X,Y` falha com "Device or resource busy".
    ALSA cru fica como reserva para máquina sem servidor de áudio.
    """
    import json
    import re
    import subprocess

    try:
        dump = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=10)
        nos = json.loads(dump.stdout)
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        nos = []
    achados = []
    for n in nos:
        props = ((n.get("info") or {}).get("props") or {})
        if props.get("media.class") == "Audio/Source" and props.get("node.name"):
            achados.append(("pulse", props["node.name"],
                            props.get("node.description") or props["node.name"]))
    if achados:
        return achados

    saida = subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout
    for m in re.finditer(r"^card (\d+): (\S+) \[([^\]]+)\], device (\d+): ([^\[]+)",
                         saida, re.MULTILINE):
        card, _, desc, dev, nome = m.groups()
        achados.append(("alsa", f"hw:{card},{dev}", f"{desc} — {nome.strip()}"))
    return achados


def _mostrar_take(take, titulo: str) -> bool:
    """Imprime a análise e devolve True se a gravação serve como referência."""
    t = Table("medida", "valor", "alvo", title=titulo)
    from ..audiolivro.audio.analise import CORTE_MIN_HZ, PICO_ALVO_DB, RUIDO_MAX_DB, SNR_MIN_DB
    t.add_row("duração", f"{take.duracao_s:.1f} s", "20–180 s")
    t.add_row("taxa/canais", f"{take.sample_rate} Hz · {take.canais}", "48000 Hz · 1")
    t.add_row("pico", f"{take.pico_db:.1f} dBFS", f"{PICO_ALVO_DB:.0f} dBFS")
    t.add_row("ruído de fundo", f"{take.ruido_db:.1f} dBFS", f"< {RUIDO_MAX_DB:.0f} dBFS")
    t.add_row("sinal/ruído", f"{take.snr_db:.1f} dB", f"> {SNR_MIN_DB:.0f} dB")
    t.add_row("banda útil", f"{take.corte_hz/1000:.1f} kHz", f"> {CORTE_MIN_HZ/1000:.0f} kHz")
    t.add_row("clipping", f"{take.clip_fracao*100:.3f}%", "0%")
    from ..audiolivro.audio.analise import MODULACAO_MIN_DB, RUMBLE_MAX_FRACAO
    t.add_row("modulação (voz?)", f"{take.modulacao_db:.1f} dB",
              f"> {MODULACAO_MIN_DB:.0f} dB")
    t.add_row("energia < 20 Hz", f"{take.rumble_fracao*100:.1f}%",
              f"< {RUMBLE_MAX_FRACAO*100:.0f}%")
    console.print(t)
    for a in take.avisos:
        console.print(f"[yellow]aviso:[/] {a}")
    for pr in take.problemas:
        console.print(f"[red]problema:[/] {pr}")
    if take.problemas:
        console.print("\n[red]este take não serve como referência[/] — regrave")
        return False
    console.print("\n[green]take aprovado[/] — ouça antes de registrar")
    return True


@voice_app.command("check")
def voice_check(arquivo: Path):
    """Mede uma gravação e diz se ela serve como referência."""
    from ..audiolivro.audio.analise import analisar

    if not _mostrar_take(analisar(arquivo), f"Análise — {arquivo.name}"):
        raise typer.Exit(1)


@voice_app.command("record")
def voice_record(
        saida: Path = typer.Option(None, "--saida", help="WAV a gravar"),
        segundos: int = typer.Option(90, help="duração da gravação"),
        dispositivo: str = typer.Option(None, help="entrada ALSA, ex.: hw:2,0"),
        listar: bool = typer.Option(False, "--listar", help="só lista as entradas")):
    """Instruções, texto de calibração e — com --saida — a gravação em si."""
    import subprocess

    from ..audiolivro.voices import TEXTO_CALIBRACAO

    entradas = _dispositivos_captura()
    if listar:
        if not entradas:
            console.print("[red]nenhuma entrada de captura ALSA encontrada[/]")
            raise typer.Exit(1)
        for backend, dev, desc in entradas:
            console.print(f"  [bold]{dev}[/]  [dim]({backend})[/]  {desc}")
        console.print("\n[yellow]Não use microfone de headset Bluetooth:[/] o perfil "
                      "HFP corta a banda em 8 kHz e aplica denoise que não se desliga.")
        return

    console.print("[bold]Como gravar a referência[/] (TDD §7.2)\n")
    for linha in [
        "60–90 s de fala contínua, em [bold]tom de narração[/] — não de conversa",
        "microfone [bold]com fio[/], fixo, sala com pouco eco (um closet com roupas serve)",
        "WAV 48 kHz / 24-bit mono · [bold]sem[/] compressor, EQ, denoise ou reverb",
        "picos por volta de −6 dBFS: se clipar, o registro é recusado",
        "[bold]deixe 2 s de silêncio[/] antes de começar a falar — é o que permite "
        "medir o ruído da sala",
        "grave [bold]3 takes[/] e escolha o melhor por teste cego com `voice test`",
    ]:
        console.print(f"  • {linha}")
    console.print("\n[bold]Texto de calibração[/] (fonética variada — leia duas vezes):\n")
    console.print(f"[italic]{TEXTO_CALIBRACAO}[/]\n")

    if saida is None:
        console.print("Para gravar aqui: [bold]iam voice voice record --saida take1.wav[/]")
        console.print("Entradas disponíveis: [bold]iam voice voice record --listar[/]")
        console.print("Já tem o arquivo? [bold]iam voice voice check take1.wav[/]")
        return

    if not entradas:
        console.print("[red]nenhuma entrada de captura encontrada[/]")
        raise typer.Exit(1)
    backend, padrao, _ = entradas[0]
    if dispositivo:
        backend = "alsa" if dispositivo.startswith("hw:") else backend
    dispositivo = dispositivo or padrao

    saida.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"gravando de [bold]{dispositivo}[/] por {segundos}s em {saida}")
    with console.status("3…"):
        subprocess.run(["sleep", "3"])
    console.print("[bold green]FALE AGORA[/]")
    bruto = saida.with_suffix(".bruto.wav")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", backend, "-i", dispositivo,
         "-t", str(segundos), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le",
         str(bruto)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        bruto.unlink(missing_ok=True)
        console.print(f"[red]falha na gravação:[/] {proc.stderr.strip()[:300]}")
        raise typer.Exit(1)

    from ..audiolivro.audio.analise import CORTE_SUBSONICO_HZ, RUMBLE_MAX_FRACAO, analisar

    # O subsônico é medido no sinal CRU e depois removido. Medido nesta máquina:
    # a captura do ALC897 tem deriva lenta abaixo de 20 Hz mesmo sem nada plugado,
    # e ela some do arquivo com um passa-altas. Não é processar a voz -- é tirar o
    # que não é som, e a cadeia de masterização já corta em 65 Hz de qualquer jeito.
    cru = analisar(bruto)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(bruto), "-af", f"highpass=f={CORTE_SUBSONICO_HZ:.0f}",
                    "-c:a", "pcm_s24le", str(saida)], capture_output=True, text=True)
    bruto.unlink(missing_ok=True)

    console.print()
    if cru.rumble_fracao > RUMBLE_MAX_FRACAO:
        console.print(f"[dim]sinal cru tinha {cru.rumble_fracao*100:.0f}% de energia "
                      f"abaixo de {CORTE_SUBSONICO_HZ:.0f} Hz — removida no arquivo "
                      f"final[/]")
        if cru.clip_fracao > 0:
            console.print("[yellow]atenção:[/] esse subsônico chegou a saturar o "
                          "conversor. Baixe o ganho de entrada — o corte não desfaz "
                          "o que já clipou")
    if not _mostrar_take(analisar(saida), f"Análise — {saida.name}"):
        raise typer.Exit(1)
    console.print(f"\nRegistre: [bold]iam voice voice new moises-v1 --reference {saida}[/]")


@voice_app.command("new")
def voice_new(voice_id: str, reference: Path = typer.Option(..., "--reference"),
              quem: str = typer.Option("Moises", help="nome no termo de consentimento"),
              forcar: bool = typer.Option(False, "--forcar",
                  help="registra apesar dos defeitos medidos, que ficam no profile.yaml")):
    """Registra uma voz. Exige consentimento documentado."""
    from ..audiolivro.voices import criar, modelo_consentimento

    try:
        v = criar(settings().raiz, voice_id, reference.resolve(),
                  consentimento=modelo_consentimento(voice_id, quem), forcar=forcar)
    except ValueError as e:
        console.print(f"[red]recusado:[/] {e}")
        raise typer.Exit(1)
    console.print(f"[green]voz registrada[/] {v.dir}")
    if forcar:
        console.print("[yellow]registrada com ressalvas[/] — os defeitos medidos "
                      f"ficaram anotados em {v.dir/'profile.yaml'}")
    console.print(f"assine o termo: {v.dir/'CONSENT.md'}")
    console.print("[yellow]voices/ não é versionado — inclua no backup cifrado[/]")


@voice_app.command("template")
def voice_template(voice_id: str = typer.Argument(..., help="id a registrar, ex.: dora-v1"),
                   kokoro_voice: str = typer.Option("pf_dora",
                       help="voz do Kokoro: pf_dora, pm_alex, pm_santa"),
                   segundos: int = typer.Option(16, help="duração da referência"),
                   velocidade: float = typer.Option(1.0,
                       help="ritmo da referência (0,75 = mais pausado)")):
    """Cria uma voz template a partir de uma voz do Kokoro (Apache-2.0).

    Sintetiza uma referência com o Kokoro e a registra como voz de clonagem do
    Chatterbox: timbre brasileiro, sem trocar de motor quando a voz própria chegar.

    `velocidade` existe porque o Chatterbox clona o ANDAMENTO junto com o timbre:
    referência apressada rende narração apressada, e nenhum parâmetro de geração
    corrige isso depois — medido, `cfg_weight` de 0,5 a 0,2 não move as palavras
    por minuto. O lugar de decidir o ritmo é aqui.

    Vem do Kokoro, e não de esticar o WAV com `atempo`, porque assim não há
    artefato de time-stretch no sinal que condiciona o clone.
    """
    import warnings

    import numpy as np
    import soundfile as sf

    warnings.filterwarnings("ignore")
    from kokoro import KPipeline

    from ..audiolivro.voices import TEXTO_CALIBRACAO, criar

    with console.status(f"sintetizando referência com {kokoro_voice}…"):
        pipe = KPipeline(lang_code="p")
        audio = np.concatenate([g.audio.numpy() for g in pipe(TEXTO_CALIBRACAO,
                                                             voice=kokoro_voice,
                                                             speed=velocidade)])
    tmp = settings().cache_dir / f"ref-{voice_id}.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp), audio, 24000)
    v = criar(settings().raiz, voice_id, tmp,
              template_de=f"Kokoro-82M (Apache-2.0), voz {kokoro_voice}"
                          + (f", velocidade {velocidade:g}" if velocidade != 1.0 else ""))
    console.print(f"[green]voz template registrada[/] {v.dir}")
    console.print(f"procedência: {v.dir/'PROVENANCE.md'}")
    console.print(f"teste: [bold]iam voice voice test {voice_id}[/]")


@voice_app.command("list")
def voice_list():
    """Lista as vozes registradas."""
    from ..audiolivro.voices import listar, raiz_vozes

    vozes = listar(settings().raiz)
    if not vozes:
        console.print("nenhuma voz registrada — use [bold]voice record[/] para começar")
        return
    import yaml as _yaml

    for v in vozes:
        cfg = _yaml.safe_load(
            (raiz_vozes(settings().raiz) / v / "profile.yaml").read_text(encoding="utf-8"))
        tipo = cfg.get("tipo", "pessoa")
        origem = f" — {cfg['origem']}" if cfg.get("origem") else ""
        console.print(f"  [bold]{v}[/] ({tipo}){origem}")


@voice_app.command("test")
def voice_test(voice_id: str, ptbr_pack: bool = True):
    """Sintetiza o texto de calibração com a voz, para conferência auditiva."""
    import soundfile as sf

    from ..audiolivro.voices import TEXTO_CALIBRACAO, carregar

    v = carregar(settings().raiz, voice_id)
    engine = ChatterboxEngine(v.params, use_ptbr_pack=ptbr_pack)
    with console.status("carregando modelo…"):
        engine.load()
        engine.set_voice(v.referencia)
    with console.status("sintetizando…"):
        audio = engine.synthesize(TEXTO_CALIBRACAO, seed=v.params.seed or 1234)
    destino = v.dir / "samples" / f"calibracao-{voice_id}.wav"
    sf.write(str(destino), audio, engine.sample_rate)
    console.print(f"[green]{destino}[/] ({len(audio)/engine.sample_rate:.1f}s) — ouça antes de usar")
