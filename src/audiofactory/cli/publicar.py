"""Publicação no YouTube: `publish` e o branding do canal (`canal ...`)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from .. import project as proj_mod
from ..config import settings
from ._comum import _proj, console
from .app import app


@app.command()
def publish(slug: str,
            arquivo: Path = typer.Option(None,
                help="MP4 a enviar (padrão: o único .mp4 em output/)"),
            privacidade: str = typer.Option("private",
                help="private, unlisted ou public"),
            tags: str = typer.Option(None, help="tags separadas por vírgula"),
            categoria: str = typer.Option("27",
                help="categoria do YouTube (27 = Educação)"),
            client_secret: Path = typer.Option(None,
                help="client_secret.json (padrão: config/youtube_client_secret.json)"),
            miniatura: Path = typer.Option(None,
                help="imagem para a miniatura (padrão: frame extraído do próprio MP4)"),
            miniatura_preset: str = typer.Option("slides",
                help="preset usado no `video`, para escolher um instante fora do dissolve"),
            sem_miniatura: bool = typer.Option(False, "--sem-miniatura",
                help="não define miniatura"),
            forcar: bool = typer.Option(False, "--forcar",
                help="sobe mesmo com rights.status incompleto ou marcado como teste"),
            republicar: bool = typer.Option(False, "--republicar",
                help="sobe de novo um projeto que já tem vídeo publicado"),
            token: Path = typer.Option(None,
                help="token de upload (padrão: config/youtube_token.json) — use outro "
                     "para publicar num canal diferente do padrão")):
    """Sobe o MP4 do projeto para o YouTube, com metadados do project.yaml.

    Exige credenciais OAuth próprias (veja docs/youtube-publish.md) — a ferramenta
    não decide o que publicar, só preenche o que já está registrado no projeto.
    """
    p = _proj(slug)
    _etapa_publish(p, proj_mod.carregar_config(p), arquivo=arquivo,
                   privacidade=privacidade, tags=tags, categoria=categoria,
                   client_secret=client_secret, miniatura=miniatura,
                   miniatura_preset=miniatura_preset, sem_miniatura=sem_miniatura,
                   forcar=forcar, republicar=republicar, token=token)


def _token_upload(token: Path | None) -> Path:
    """Token de UPLOAD (`youtube.upload`) -- diferente do token de canal/branding
    (`_token_canal`, escopo `youtube`). Cada canal do YouTube tem o seu, mesmo
    `client_secret.json` (mesmo app OAuth)."""
    return token or (settings().config_dir / "youtube_token.json")


def _etapa_publish(p: Path, cfg: dict, *, arquivo: Path | None = None,
                   privacidade: str = "private", tags: str | None = None,
                   categoria: str = "27", client_secret: Path | None = None,
                   miniatura: Path | None = None, miniatura_preset: str = "slides",
                   sem_miniatura: bool = False, forcar: bool = False,
                   republicar: bool = False, token: Path | None = None) -> str:
    """Sobe o MP4 e devolve o id do vídeo publicado.

    Um upload já feito é registrado em `output/publicado.json` e barra o segundo:
    `videos.insert` não é idempotente — repetir o comando cria OUTRO vídeo no
    canal, e o pipeline em lote existe justamente para ser re-rodado.
    """
    from ..publish.youtube import autenticar, enviar, metadados

    if privacidade not in ("private", "unlisted", "public"):
        console.print(f"[red]privacidade inválida:[/] {privacidade} — "
                      "use private, unlisted ou public")
        raise typer.Exit(1)

    status_rights = str((cfg.get("rights") or {}).get("status") or "")
    suspeito = (not status_rights or status_rights == "PREENCHER"
                or "NAO-PUBLICAR" in status_rights.upper()
                or "TESTE" in status_rights.upper())
    if suspeito and not forcar:
        console.print(f"[red]rights.status = '{status_rights or None}'[/] — "
                      "confira project.yaml antes de publicar, ou use --forcar")
        raise typer.Exit(1)
    elif suspeito:
        console.print(f"[yellow]publicando apesar de rights.status = "
                      f"'{status_rights}'[/]")

    if arquivo is None:
        candidatos = sorted((p / "output").glob("*.mp4"))
        if not candidatos:
            console.print("[red]nenhum mp4[/] em output/ — rode `video` antes")
            raise typer.Exit(1)
        if len(candidatos) > 1:
            console.print("[red]mais de um mp4[/] em output/ — use --arquivo: "
                          + ", ".join(c.name for c in candidatos))
            raise typer.Exit(1)
        arquivo = candidatos[0]

    registro = p / "output" / "publicado.json"
    if registro.exists() and not republicar:
        anterior = json.loads(registro.read_text(encoding="utf-8"))
        console.print(f"[yellow]já publicado[/] em {anterior.get('em')}: "
                      f"https://studio.youtube.com/video/{anterior['video_id']}/edit")
        console.print("[dim]use --republicar para subir outra cópia[/]")
        return anterior["video_id"]

    segredo = client_secret or (settings().config_dir / "youtube_client_secret.json")
    token = _token_upload(token)
    if not segredo.exists():
        console.print(f"[red]client secret ausente:[/] {segredo} — "
                      "veja docs/youtube-publish.md")
        raise typer.Exit(1)

    corpo = metadados(cfg, p, privacidade=privacidade,
                      tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else None,
                      categoria=categoria)

    console.print(f"[dim]arquivo:[/] {arquivo.name} ({arquivo.stat().st_size/1e6:.0f} MB)")
    console.print(f"[dim]título:[/] {corpo['snippet']['title']}")
    console.print(f"[dim]privacidade:[/] {privacidade}")

    with console.status("autenticando…"):
        creds = autenticar(segredo, token)

    def _progresso(fracao: float) -> None:
        console.print(f"[dim]enviado: {fracao*100:.0f}%[/]")

    video_id = enviar(arquivo, corpo, creds, progresso=_progresso)
    registro.write_text(json.dumps(
        {"video_id": video_id, "url": f"https://youtu.be/{video_id}",
         "arquivo": arquivo.name, "privacidade": privacidade,
         "em": datetime.now().isoformat(timespec="seconds")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    aviso = "" if privacidade != "private" else " (privado até você mudar)"
    console.print(f"[green]publicado{aviso}[/] "
                  f"https://studio.youtube.com/video/{video_id}/edit")

    if not sem_miniatura:
        from ..publish.youtube import definir_miniatura
        from ..video.thumbnail import extrair_frame, instante_seguro

        img = miniatura
        if img is None:
            srt = p / "audio" / "chapters" / f"{arquivo.stem.replace('-master', '')}.srt"
            if not srt.exists():
                candidatos_srt = sorted((p / "audio" / "chapters").glob("*.srt"))
                srt = candidatos_srt[0] if len(candidatos_srt) == 1 else None
            instante = instante_seguro(arquivo, miniatura_preset, srt=srt)
            img = p / "cache" / "miniatura.jpg"
            extrair_frame(arquivo, img, instante)
        try:
            definir_miniatura(video_id, img, creds)
            console.print(f"[dim]miniatura definida:[/] {img}")
        except Exception as e:
            console.print(f"[yellow]miniatura não definida[/] ({e}) — "
                          "confira se o canal tem miniatura customizada "
                          "habilitada (verificação de telefone)")
    return video_id


canal_app = typer.Typer(help="Branding do canal no YouTube (descrição, keywords, banner)",
                        no_args_is_help=True)
app.add_typer(canal_app, name="canal")


def _token_canal(token: Path | None) -> Path:
    return token or (settings().config_dir / "youtube_canal_token.json")


def _autenticar_canal(client_secret: Path | None, token: Path | None = None):
    from ..publish.youtube import SCOPES_CANAL, autenticar

    segredo = client_secret or (settings().config_dir / "youtube_client_secret.json")
    if not segredo.exists():
        console.print(f"[red]client secret ausente:[/] {segredo} — "
                      "veja docs/youtube-publish.md")
        raise typer.Exit(1)
    with console.status("autenticando (escopo de canal — pode pedir login de novo)…"):
        return autenticar(segredo, _token_canal(token), scopes=SCOPES_CANAL)


_TOKEN_HELP = ("padrão: config/youtube_canal_token.json — use outro arquivo para "
               "gerenciar um segundo canal da mesma conta Google sem sobrescrever "
               "o token do primeiro")


@canal_app.command("mostrar")
def canal_mostrar(client_secret: Path = typer.Option(None,
                      help="padrão: config/youtube_client_secret.json"),
                  token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Mostra a descrição, keywords e país atuais do canal."""
    from ..publish.youtube import canal_atual

    creds = _autenticar_canal(client_secret, token)
    canal = canal_atual(creds)
    snip = canal.get("snippet", {})
    branding = (canal.get("brandingSettings") or {}).get("channel", {})
    console.print(f"[bold]{snip.get('title')}[/]  ({canal.get('id')})")
    console.print(f"país: {snip.get('country') or '[dim](não definido)[/]'}")
    console.print(f"keywords: {branding.get('keywords') or '[dim](vazias)[/]'}")
    console.print(f"\n[bold]descrição:[/]\n{snip.get('description') or '[dim](vazia)[/]'}")


@canal_app.command("atualizar")
def canal_atualizar(descricao: Path = typer.Option(None,
                        help="arquivo .txt com a nova descrição do canal"),
                    keywords: str = typer.Option(None,
                        help="separadas por vírgula, ex.: 'budismo,dhamma,cânone páli'"),
                    pais: str = typer.Option(None, help="código ISO, ex.: BR"),
                    client_secret: Path = typer.Option(None),
                    token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Atualiza descrição/keywords/país do canal (`brandingSettings`).

    `channels.update` substitui o recurso inteiro — este comando busca o estado
    atual primeiro e só troca os campos passados aqui, então rodar sem nenhuma
    opção não muda nada.
    """
    from ..publish.youtube import atualizar_canal, formatar_keywords

    if descricao is None and keywords is None and pais is None:
        console.print("[yellow]nada a atualizar[/] — passe --descricao, --keywords "
                      "e/ou --pais")
        raise typer.Exit(1)

    desc_texto = descricao.read_text(encoding="utf-8").strip() if descricao else None
    kw = formatar_keywords([k.strip() for k in keywords.split(",") if k.strip()]) if keywords else None

    creds = _autenticar_canal(client_secret, token)
    atualizar_canal(creds, descricao=desc_texto, keywords=kw, pais=pais)
    console.print("[green]canal atualizado[/] — confira em "
                  "https://studio.youtube.com/channel/_/editing/branding")


@canal_app.command("banner")
def canal_banner(imagem: Path, client_secret: Path = typer.Option(None),
                 token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Sobe a arte de capa do canal (recomendado: 2560×1440, ≤6 MB).

    A foto de perfil/logo do canal não tem endpoint na API do YouTube — só se
    troca manualmente em https://studio.youtube.com.
    """
    from ..publish.youtube import atualizar_banner

    if not imagem.exists():
        console.print(f"[red]imagem não encontrada:[/] {imagem}")
        raise typer.Exit(1)
    creds = _autenticar_canal(client_secret, token)
    with console.status("enviando banner…"):
        atualizar_banner(imagem, creds)
    console.print("[green]banner atualizado[/] — confira em "
                  "https://studio.youtube.com/channel/_/editing/branding")
    console.print("[yellow]a foto de perfil do canal precisa ser trocada à mão[/] — "
                  "a Data API v3 não expõe esse recurso")


@canal_app.command("marca-dagua")
def canal_marca_dagua(imagem: Path,
                      canto: str = typer.Option("bottomRight",
                          help="topLeft, topRight, bottomLeft ou bottomRight"),
                      client_secret: Path = typer.Option(None),
                      token: Path = typer.Option(None, help=_TOKEN_HELP)):
    """Define a marca d'água do canal — PNG com alfa recomendado.

    Aparece sobre o vídeo o tempo todo e funciona como botão de inscrição ao
    passar o mouse; vale para os vídeos já publicados também.
    """
    from ..publish.youtube import CANTOS_MARCA_DAGUA, canal_atual, definir_marca_dagua

    if canto not in CANTOS_MARCA_DAGUA:
        console.print(f"[red]canto inválido:[/] {canto} — use "
                      f"{', '.join(CANTOS_MARCA_DAGUA)}")
        raise typer.Exit(1)
    if not imagem.exists():
        console.print(f"[red]imagem não encontrada:[/] {imagem}")
        raise typer.Exit(1)
    creds = _autenticar_canal(client_secret, token)
    channel_id = canal_atual(creds)["id"]
    with console.status("enviando marca d'água…"):
        definir_marca_dagua(imagem, creds, channel_id, canto=canto)
    console.print("[green]marca d'água definida[/] — vale para os vídeos já "
                  "publicados e os próximos")
