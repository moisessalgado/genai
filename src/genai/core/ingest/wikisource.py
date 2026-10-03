"""Ingest de obras da Wikisource em português (pt.wikisource.org).

Diferente do Acesso ao Insight (`acessoaoinsight.py`), aqui o HTML é Parsoid
limpo (MediaWiki de verdade, não export malformado do Word) — dá para usar
BeautifulSoup em vez de regex à mão.

Duas coisas medidas ao vivo contra a página real que valem registro:

1. **Todo o aparato editorial mora dentro de `class="ws-noexport"`** — nav
   anterior/próximo, linha de edição, selo de domínio público. Um seletor só
   remove tudo de uma vez; não precisa de regra por tipo de aparato.
2. **Metadados estruturados vêm de graça**: cada página de capítulo carrega um
   `<div id="ws-data" class="ws-noexport">` (oculto, `display:none`) com
   `#ws-article-id`, `#ws-title` (dois links — obra e capítulo), `#ws-author`,
   `#ws-year`. Extrai-se ISSO antes de remover `.ws-noexport`, em vez de tentar
   ler o título visível (`#firstHeading` vem como "Obra/Capítulo", não separado).

Uma página de capítulo não tem marcador de subseção em romano sobrevivendo como
heading na transcrição (verificado no capítulo 1 de *As Reinações de
Narizinho*: 296 `<p>` mas nenhum `<h1-6>` de subseção) — cada página de
capítulo vira exatamente UM `Chapter` depois de `loader.detectar_capitulos()`,
sem risco de fragmentação.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

BASE = "https://pt.wikisource.org"
TITULO_MAX = 100  # limite do snippet.title do YouTube


@dataclass
class Capitulo:
    """Uma página de capítulo já reduzida ao que interessa ao pipeline."""

    url: str
    titulo: str             # "Narizinho Arrebitado"
    obra: str                # "As Reinações de Narizinho"
    autor: str               # "Monteiro Lobato"
    ano_publicacao: int | None
    artigo_id: str | None
    paragrafos: list[str]

    @property
    def slug(self) -> str:
        return _ascii(self.titulo).lower().strip()

    @property
    def titulo_video(self) -> str:
        """Título do vídeo: capítulo primeiro (é o que diferencia os 11 vídeos
        da série), obra depois, mesmo corte por travessão que `_nome_de_arquivo`
        (cli/main.py) usa para o nome do arquivo não atravessar a legenda."""
        titulo = f"{self.titulo} — {self.obra}"
        return titulo[:TITULO_MAX]

    def texto(self) -> str:
        """Texto a narrar: o título do capítulo primeiro (anuncia o capítulo,
        como um audiolivro faz), depois os parágrafos, em branco entre si."""
        return "\n\n".join([f"{self.titulo}.", *self.paragrafos]) + "\n"

    def rights(self) -> dict:
        """`rights:` do project.yaml.

        Diferente do Acesso ao Insight (`status: licenciado` — a tradução é
        obra derivada): aqui o texto ORIGINAL é de domínio público no Brasil
        (autor falecido há mais de 70 anos, Lei 9.610/1998 art. 41) — a própria
        página confirma isso com o selo de domínio público. Não é tradução,
        então `tradutor` fica vazio.
        """
        return {
            "status": "dominio-publico",
            "autor": self.autor,
            "ano_morte": 1948 if self.autor == "Monteiro Lobato" else None,
            "tradutor": None,
            "fonte": f"{self.obra} — {self.titulo} — Wikisource — {self.url}",
            "licenca": (
                f"Texto original em domínio público no Brasil (Lei 9.610/1998, "
                f"art. 41 — 70 anos post-mortem). Transcrição hospedada na "
                f"Wikisource em português (CC BY-SA 4.0 para a transcrição/"
                f"marcação em si, não para o texto original)."),
            "verificado_em": None,
        }


def normalizar_url(entrada: str) -> str:
    """Aceita a URL inteira ou só o caminho depois de `/wiki/`."""
    e = entrada.strip().rstrip("/")
    if e.startswith("http://") or e.startswith("https://"):
        return e
    e = e.lstrip("/")
    if not e.startswith("wiki/"):
        e = f"wiki/{e}"
    return f"{BASE}/{e}"


def _url_para_requisicao(url: str) -> str:
    """Codifica o caminho em percent-encoding antes de enviar a requisição.

    Os `href` que `descobrir_capitulos` extrai do HTML vêm em unicode cru
    (é assim que o MediaWiki os gera, válido em HTML/IRI) -- mas o request
    line HTTP tem de ser ASCII, e `http.client` não codifica nada sozinho:
    "ç"/"õ"/"í" direto na URL derruba com `UnicodeEncodeError`. `unquote`
    antes do `quote` normaliza os dois casos de entrada (unicode cru, ou já
    percent-encoded por quem colou a URL do navegador) sem dar % duplo.
    """
    partes = urlsplit(url)
    caminho = quote(unquote(partes.path), safe="/:@")
    return urlunsplit((partes.scheme, partes.netloc, caminho, partes.query, partes.fragment))


def baixar(url: str, *, timeout: float = 30.0) -> str:
    """Baixa a página. UTF-8 direto — Parsoid não mente sobre o charset (ao
    contrário do Acesso ao Insight, que declara latin-1 e é cp1252 de verdade)."""
    pedido = Request(_url_para_requisicao(url),
                     headers={"User-Agent": "audio-factory/0.1 (+local)"})
    try:
        with urlopen(pedido, timeout=timeout) as r:
            bruto = r.read()
    except HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"{url}: {e.reason}") from e
    return bruto.decode("utf-8", errors="replace")


def _limpar(soup: BeautifulSoup) -> None:
    """Remove em uma tacada só o aparato editorial (nav, edição, selo de
    domínio público) e o resíduo de transclusão do ProofreadPage (marcador de
    página, nota de rodapé) — modifica `soup` no lugar."""
    for sel in (".ws-noexport", ".mw-editsection", ".pagenum",
                "sup.reference", "ol.references"):
        for el in soup.select(sel):
            el.decompose()


def extrair(pagina: str, url: str) -> Capitulo:
    """Página crua -> `Capitulo`. Levanta `ValueError` se não for página de
    capítulo (sem `#ws-data`, ou corpo vazio depois da limpeza)."""
    soup = BeautifulSoup(pagina, "html.parser")
    content = soup.select_one("#mw-content-text")
    if content is None:
        raise ValueError(f"{url}: sem #mw-content-text — não é página de "
                          "artigo da Wikisource")

    dados = content.select_one("#ws-data")
    if dados is None:
        raise ValueError(
            f"{url}: sem #ws-data — página inexistente ou não é um capítulo "
            "transcrito (confira o link no índice da obra)")
    artigo_id = _texto_ou_none(dados.select_one("#ws-article-id"))
    links_titulo = (dados.select_one("#ws-title") or dados).select("a")
    if len(links_titulo) < 2:
        raise ValueError(f"{url}: #ws-title sem os dois links esperados "
                          "(obra e capítulo) — layout inesperado")
    obra = links_titulo[0].get_text(strip=True)
    titulo = links_titulo[-1].get_text(strip=True)
    autor = _texto_ou_none(dados.select_one("#ws-author")) or ""
    ano_txt = _texto_ou_none(dados.select_one("#ws-year"))
    ano = int(ano_txt) if ano_txt and ano_txt.isdigit() else None

    _limpar(content)
    paragrafos = [p for p in (_texto_paragrafo(p) for p in content.select("p")) if p]
    if not paragrafos:
        raise ValueError(f"{url}: corpo vazio depois da limpeza — layout inesperado")

    return Capitulo(url=url, titulo=titulo, obra=obra, autor=autor,
                     ano_publicacao=ano, artigo_id=artigo_id, paragrafos=paragrafos)


def descobrir_capitulos(url_indice: str) -> list[str]:
    """URL (índice OU capítulo) -> lista ordenada de URLs de capítulo.

    Devolve `[]` se `url_indice` já for uma página de capítulo (sem links
    filhos) — quem chama trata lista vazia como "é capítulo único", sem
    heurística frágil de "isto é índice ou não".
    """
    url = normalizar_url(url_indice)
    pagina = baixar(url)
    soup = BeautifulSoup(pagina, "html.parser")
    content = soup.select_one("#mw-content-text")
    if content is None:
        return []
    _limpar(content)

    prefixo = f"{url}/"
    encontrados: list[str] = []
    for a in content.select("a[href]"):
        href = a["href"]
        if href.startswith("/"):
            href = f"{BASE}{href}"
        elif href.startswith("./"):
            href = f"{BASE}/wiki/{href[2:]}"
        if href.startswith(prefixo) and href not in encontrados:
            # Exclui os links de página de scan (`.../Página:Obra.pdf/9`), que
            # também começam com prefixos de caminho da obra em alguns casos,
            # mas nunca são filhos diretos da URL do índice.
            if "Página:" in href or "Special:" in href:
                continue
            encontrados.append(href)
    return encontrados


def buscar(entrada: str, *, cache: Path | None = None, refazer: bool = False) -> Capitulo:
    """Resolve o endereço, baixa (ou lê do cache) e extrai.

    A página crua fica em disco de propósito, mesmo motivo do
    `acessoaoinsight.buscar`: o texto publicado é a fonte de uma obra, e um
    `run` longo não pode depender de o site estar no ar.
    """
    url = normalizar_url(entrada)
    nome_cache = _ascii(url.rsplit("/", 1)[-1]).lower()
    guardado = cache / f"{nome_cache}.html" if cache else None
    if guardado and guardado.exists() and not refazer:
        pagina = guardado.read_text(encoding="utf-8")
    else:
        pagina = baixar(url)
        if guardado:
            guardado.parent.mkdir(parents=True, exist_ok=True)
            guardado.write_text(pagina, encoding="utf-8")
    return extrair(pagina, url)


# --- interno -----------------------------------------------------------------


def _texto_ou_none(el) -> str | None:
    return el.get_text(strip=True) if el is not None else None


def _texto_paragrafo(p) -> str:
    return " ".join(p.get_text(" ", strip=True).split())


def _ascii(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")
