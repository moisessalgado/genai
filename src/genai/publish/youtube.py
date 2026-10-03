"""Upload do MP4 para o YouTube via Data API v3, com metadados do project.yaml.

A ferramenta preenche o que já está registrado (título, direitos, capítulos) — ela
não decide o que publicar nem quando. Essa decisão é do operador do canal (ver
ESTADO.md), por isso o padrão de privacidade é sempre `private` e o comando avisa
em vez de bloquear quando `rights.status` está incompleto ou marcado como teste.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# thumbnails.set aceita esse mesmo escopo (nao precisa do youtube.force-ssl
# completo, que abriria delete/update de qualquer vídeo do canal).

# Branding (channels.update, channelBanners.insert) exige o escopo geral --
# youtube.upload não alcança esses métodos. Fica separado do SCOPES de cima,
# com token próprio (ver `canal_token` em cli/main.py), para o pipeline
# automático (`sutta`/`publish`) continuar autorizado só para subir vídeo.
SCOPES_CANAL = ["https://www.googleapis.com/auth/youtube"]

PARTES_CANAL = "snippet,brandingSettings"

TITULO_MAX = 100

DISCLOSURE = ("Este vídeo contém narração sintética gerada por IA (clonagem de "
              "voz), com marca d'água neural do modelo de síntese.")


def _atribuicao(rights: dict) -> str:
    """Linha de crédito para a descrição, a partir do `rights:` do project.yaml."""
    status = (rights.get("status") or "").lower()
    autor = rights.get("autor")
    linhas = []
    if status == "dominio-publico":
        linhas.append("Domínio público" + (f" — {autor}" if autor else ""))
    elif autor:
        linhas.append(f"Autoria: {autor}")
    if rights.get("tradutor"):
        linhas.append(f"Tradução: {rights['tradutor']}")
    if rights.get("fonte"):
        linhas.append(f"Fonte: {rights['fonte']}")
    return "\n".join(linhas)


def descricao(cfg: dict, proj: Path) -> str:
    """Monta a descrição: atribuição + licença + timestamps de capítulo (se houver
    mais de um) + o aviso de conteúdo sintético, padrão em toda publicação do canal.

    A licença entra literal quando o `project.yaml` a registra. Texto de terceiro
    sob licença de distribuição gratuita costuma exigir que os termos viajem com
    a cópia; reproduzi-los aqui é a forma de cumprir isso num vídeo.
    """
    rights = cfg.get("rights") or {}
    blocos = [cfg["titulo"]]

    atrib = _atribuicao(rights)
    if atrib:
        blocos.append(atrib)

    if rights.get("licenca"):
        blocos.append(f"Licença do texto: {rights['licenca']}")

    capitulos = proj / "output" / "chapters.txt"
    if capitulos.exists():
        texto = capitulos.read_text(encoding="utf-8").strip()
        if texto.count("\n") >= 1:
            blocos.append(texto)

    # Trilha de terceiro (gravação real, não gerada) -- diferente do `rights`
    # do texto, cobre a MÚSICA. Só existe quando `video --musica <arquivo>` foi
    # usado com algo sob licença que exige atribuição (ex.: CC BY-SA); a
    # trilha gerada pelo próprio canal (ACE-Step/MusicGen/senoide) não precisa
    # disso -- ver LICENSES.md.
    if cfg.get("trilha", {}).get("atribuicao"):
        blocos.append(f"Trilha sonora: {cfg['trilha']['atribuicao']}")

    blocos.append(DISCLOSURE)
    return "\n\n".join(blocos) + "\n"


def metadados(cfg: dict, proj: Path, *, privacidade: str = "private",
              tags: list[str] | None = None, categoria: str = "27") -> dict:
    """Corpo do `videos.insert`. `containsSyntheticMedia` fica sempre `True`: todo
    áudio deste canal sai de um TTS (ver voices.py), não é uma opção por projeto."""
    return {
        "snippet": {
            "title": cfg["titulo"][:TITULO_MAX],
            "description": descricao(cfg, proj),
            "tags": tags or [],
            "categoryId": categoria,
        },
        "status": {
            "privacyStatus": privacidade,
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True,
        },
    }


def autenticar(client_secret: Path, token: Path, *, scopes: list[str] = SCOPES):
    """Carrega um token salvo, renova se preciso, ou abre o navegador para o
    consentimento OAuth na primeira vez (`InstalledAppFlow`, fluxo local).

    `scopes` deve casar com o que foi pedido quando `token` foi gerado -- um
    token autorizado com `SCOPES` (upload) não serve para `SCOPES_CANAL`
    (branding); apague o arquivo para reautorizar com outro escopo.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret.exists():
                raise FileNotFoundError(
                    f"client secret não encontrado: {client_secret} — "
                    "veja docs/youtube-publish.md")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), scopes)
            creds = flow.run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json(), encoding="utf-8")

    return creds


def enviar(video: Path, corpo: dict, creds,
           progresso: Callable[[float], None] | None = None) -> str:
    """Envia `video` em chunks resumíveis; devolve o id do vídeo publicado."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    youtube = build("youtube", "v3", credentials=creds)
    media = MediaFileUpload(str(video), chunksize=-1, resumable=True,
                            mimetype="video/mp4")
    pedido = youtube.videos().insert(part="snippet,status", body=corpo,
                                     media_body=media)
    resposta = None
    while resposta is None:
        status, resposta = pedido.next_chunk()
        if status and progresso:
            progresso(status.progress())
    return resposta["id"]


def definir_miniatura(video_id: str, imagem: Path, creds) -> None:
    """Sobe `imagem` como miniatura de `video_id`. Exige canal com verificacao
    de telefone habilitada (miniatura customizada e recurso do YouTube, nao da
    API) -- sem isso a chamada falha com `youtubeSignupRequired`."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    youtube = build("youtube", "v3", credentials=creds)
    media = MediaFileUpload(str(imagem), mimetype="image/jpeg")
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def formatar_keywords(tags: list[str]) -> str:
    """Lista -> string única que `brandingSettings.channel.keywords` espera.

    O campo é um texto corrido, não um array: termos de mais de uma palavra
    precisam de aspas, senão o YouTube os quebra em keywords separadas.
    """
    return " ".join(f'"{t}"' if " " in t else t for t in tags)


def canal_atual(creds) -> dict:
    """`snippet` + `brandingSettings` do canal autenticado (o próprio, `mine=True`)."""
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", credentials=creds)
    resp = youtube.channels().list(part=PARTES_CANAL, mine=True).execute()
    itens = resp.get("items") or []
    if not itens:
        raise RuntimeError("nenhum canal associado a esta conta Google")
    return itens[0]


def mesclar_branding(canal: dict, *, descricao: str | None = None,
                     keywords: str | None = None, pais: str | None = None,
                     banner_url: str | None = None) -> dict:
    """Corpo do `channels.update` a partir do canal atual + só os campos pedidos.

    `channels.update` substitui o recurso INTEIRO em cada `part` enviado -- um
    corpo parcial apagaria os campos omitidos (título, outras keywords etc.).
    Por isso parte sempre do estado atual (`canal_atual`) e só troca o que foi
    passado explicitamente.
    """
    snippet = dict(canal.get("snippet") or {})
    branding = dict(canal.get("brandingSettings") or {})
    branding_channel = dict(branding.get("channel") or {})
    branding_image = dict(branding.get("image") or {})

    if descricao is not None:
        snippet["description"] = descricao
        branding_channel["description"] = descricao
    if keywords is not None:
        branding_channel["keywords"] = keywords
    if pais is not None:
        snippet["country"] = pais
        branding_channel["country"] = pais
    if banner_url is not None:
        branding_image["bannerExternalUrl"] = banner_url

    branding["channel"] = branding_channel
    if branding_image:
        branding["image"] = branding_image
    return {"id": canal["id"], "snippet": snippet, "brandingSettings": branding}


def _atualizar_parte(youtube, corpo: dict, parte: str) -> dict:
    return youtube.channels().update(
        part=parte, body={"id": corpo["id"], parte: corpo[parte]}).execute()


def atualizar_canal(creds, *, descricao: str | None = None, keywords: str | None = None,
                    pais: str | None = None) -> dict:
    """Aplica descrição/keywords/país ao canal. Devolve o `brandingSettings` atualizado.

    Só `part=brandingSettings` -- confirmado contra a API real: `channels.update`
    não aceita `part=snippet` sozinho (`ERROR_PART_UNEXPECTED`) nem combinado com
    `brandingSettings` (`"branding_settings cannot be used with other parts"`).
    `snippet.description`/`snippet.country`, que `channels.list` devolve, são
    espelho de leitura de `brandingSettings.channel` -- não têm parte própria
    gravável.
    """
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", credentials=creds)
    corpo = mesclar_branding(canal_atual(creds), descricao=descricao,
                             keywords=keywords, pais=pais)
    return _atualizar_parte(youtube, corpo, "brandingSettings")


def atualizar_banner(imagem: Path, creds) -> str:
    """Sobe `imagem` como capa do canal (arte 2560×1440) e devolve a URL.

    Só a arte de capa é suportada pela API -- a foto de perfil/logo do canal
    NÃO tem endpoint na Data API v3 (é atributo da Conta/Marca Google) e só se
    troca manualmente em https://studio.youtube.com.
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    youtube = build("youtube", "v3", credentials=creds)
    media = MediaFileUpload(str(imagem), mimetype="image/png")
    url = youtube.channelBanners().insert(media_body=media).execute()["url"]
    corpo = mesclar_branding(canal_atual(creds), banner_url=url)
    _atualizar_parte(youtube, corpo, "brandingSettings")
    return url


CANTOS_MARCA_DAGUA = ("topLeft", "topRight", "bottomLeft", "bottomRight")


def definir_marca_dagua(imagem: Path, creds, channel_id: str,
                        canto: str = "bottomRight") -> None:
    """Define a marca d'água do canal (`watermarks.set`) -- aparece por cima do
    vídeo o tempo todo e funciona como botão de inscrição ao passar o mouse.
    Vale para os vídeos já publicados também, não só os próximos.

    PNG com alfa é o formato certo: o player não recorta a imagem, então fundo
    opaco viraria um quadrado sólido sobre o vídeo.
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    youtube = build("youtube", "v3", credentials=creds)
    media = MediaFileUpload(str(imagem), mimetype="image/png")
    # A API não tem um `type` para "vídeo inteiro" -- só offsetFromStart/End
    # (confirmado no schema `InvideoTiming`, não na doc). offsetMs=0 desde o
    # início, durationMs bem maior que qualquer vídeo do canal (~um dia)
    # produz o mesmo efeito prático.
    corpo = {"timing": {"type": "offsetFromStart", "offsetMs": "0",
                        "durationMs": "86400000"},
             "position": {"type": "corner", "cornerPosition": canto}}
    pedido = youtube.watermarks().set(channelId=channel_id, body=corpo,
                                      media_body=media)
    # `watermarks.set` devolve 204 sem corpo -- bug do httplib2 (0.32.0) ao
    # descomprimir uma resposta gzip VAZIA derruba com
    # "ValueError: range() arg 3 must not be zero". Pedir a resposta sem gzip
    # evita o caminho de descompressão inteiro; a chamada em si funciona normal.
    pedido.headers["accept-encoding"] = "identity"
    pedido.execute()
