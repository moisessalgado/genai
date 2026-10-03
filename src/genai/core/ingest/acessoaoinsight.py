"""Ingest de suttas do Acesso ao Insight (acessoaoinsight.net).

O site publica traducoes do Canon Pali em portugues, em paginas exportadas do
Word e servidas com a cara de HTML dos anos 2000. A estrutura, medida em 40
paginas sorteadas do indice, e teimosamente regular:

    <!-- INICIO DO TEXTO -->
    <p class=Tit3> Anguttara Nikaya IV.45      <- referencia da colecao
    <p class=Tit1> Rohitassa Sutta             <- nome em pali
    <p class=Tit1> Rohitassa                   <- titulo em portugues
    <p class=Normal> Somente para distribuicao gratuita...   <- licenca
    <hr>
    ... corpo, em <p class=Normal> ...
    <hr>                                       <- so quando ha notas
    ... notas, "Veja tambem", ">> Proximo Sutta" ...
    <!-- FIM DO TEXTO -->

Medido na amostra: 40/40 com um Tit3 e dois Tit1, 40/40 com o bloco de licenca,
37/40 com dois `<hr>` e 3/40 com um so -- os tres sao suttas curtos que nao tem
secao de notas. Dai a regra: o corpo comeca depois do PRIMEIRO `<hr>` e termina
no ULTIMO; havendo um so, vai ate o fim.

Duas armadilhas que custaram medicao e nao devem ser re-descobertas:

1. **A pagina declara `charset=ISO-8859-1` e mente.** E `windows-1252`: as aspas
   curvas do Word (0x93/0x94) sao caracteres de controle em latin-1. Decodificar
   como latin-1 entrega `\\x93E possivel` no lugar de `"E possivel`.
2. **URL inexistente devolve HTTP 200.** O servidor entrega uma pagina de busca
   generica, sem o marcador `INICIO DO TEXTO`. Por isso a ausencia do marcador e
   erro explicito, e nao "sutta vazio".
"""
from __future__ import annotations

import html as _html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://www.acessoaoinsight.net"

# O Word exporta `<p>` sem fechar e inventa `</Tit1>`; nao da para confiar em
# tag fechada. Todo recorte aqui e feito por delimitador de ABERTURA.
_CORPO = re.compile(r"INICIO DO TEXTO(.*?)FIM DO TEXTO", re.S)
_HR = re.compile(r"<hr\b[^>]*>", re.I)
_P = re.compile(r"<p\b([^>]*)>", re.I)
_CLASSE = re.compile(r"""class=["']?([A-Za-z][A-Za-z0-9]*)""")
_BR = re.compile(r"<br\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]*>")
_COMENTARIO = re.compile(r"<!--.*?-->", re.S)
# O ultimo `<p>` de cada pagina carrega o `<!--` que abre o rodape, sem `>`
# para fechar -- e `_TAG` so casa tag fechada. Sem esta guarda, "<!--" virava
# um paragrafo e ia ao TTS.
_TAG_ABERTA = re.compile(r"<[^>]*$")
# Sentinela da quebra de VERSO. As quebras de linha que sobram no HTML sao do
# Word quebrando a fonte na coluna 78 -- nao sao do texto, e preserva-las
# cortaria frases no meio ("Assim / ouvi.").
_SENTINELA = "\x00"

# Paragrafos que sao aparato de edicao, nao texto a narrar. As notas ficam depois
# do ultimo `<hr>` e ja saem no recorte; isto aqui e o cinto alem do suspensorio,
# para as paginas em que o editor esqueceu o `<hr>`.
_APARATO = re.compile(
    r"^(?:notas?\s*:|nota\b|veja\s+tamb[ée]m|veja\s+tb|para\s+ouvir\b|>>|<<|\[\d{1,3}\])",
    re.I)
# Marcador de nota no corpo: `[1]`, `[12]`. `limpar()` tambem os remove, mas
# aqui eles somem antes de virar clean.txt -- e o diff.md fica legivel.
_MARCA_NOTA = re.compile(r"\[\d{1,3}\]")
# Numeracao de paragrafo das edicoes de Bhikkhu Bodhi/Nanamoli ("1. Assim ouvi.").
# Narrada, ela vira "um. Assim ouvi." -- um numero lido em voz alta no meio do
# sutta. E numeracao de referencia, nao texto.
_NUMERO_PARAGRAFO = re.compile(r"^\d{1,3}\s*[.)]\s+")

TITULO_MAX = 100   # limite do snippet.title do YouTube


@dataclass
class Sutta:
    """Uma pagina de sutta ja reduzida ao que interessa ao pipeline."""

    url: str
    codigo: str            # ANIV.45 -- como aparece na URL
    referencia: str        # "Anguttara Nikaya IV.45"
    pali: str              # "Rohitassa Sutta"
    portugues: str         # "Rohitassa"
    paragrafos: list[str]
    licenca: str
    descartados: int = 0   # paragrafos de aparato removidos, para o operador ver

    @property
    def slug(self) -> str:
        """`ANIV.45` + `Rohitassa Sutta` -> `aniv45-rohitassa`.

        O codigo sozinho ja e unico; o nome em pali entra porque o operador le
        `projects/` e `books/` com os proprios olhos.
        """
        nome = _ascii(self.pali).lower().replace("sutta", "").strip()
        nome = re.sub(r"[^a-z0-9]+", "-", nome).strip("-")
        return f"{_ascii(self.codigo).lower().replace('.', '')}-{nome}".strip("-")

    @property
    def titulo(self) -> str:
        """Titulo do projeto -- e, dai, do video no YouTube.

        O travessao nao e enfeite: `_nome_de_arquivo` (cli/main.py) corta ali, e
        o MP4 fica com o nome em pali, curto o bastante para o player nao
        atravessar a legenda queimada.
        """
        sub = self.portugues if _distinto(self.pali, self.portugues) else self.referencia
        titulo = f"{self.pali} — {sub}"
        if sub is not self.referencia and len(titulo) + len(self.referencia) + 3 <= TITULO_MAX:
            titulo = f"{titulo} ({self.referencia})"
        return titulo[:TITULO_MAX]

    def texto(self) -> str:
        """Texto a narrar, em paragrafos separados por linha em branco.

        A referencia da colecao NAO entra: medido, o normalizador transforma
        "Anguttara Nikaya IV.45" em "Anguttara Nikaya IVquarenta e cinco" -- o
        numeral romano fica sem leitura e o `.45` vira cardinal colado. A
        referencia vive no titulo e na descricao, que ninguem le em voz alta.
        """
        abertura = self.pali
        if _distinto(self.pali, self.portugues):
            abertura = f"{self.pali}. {self.portugues}"
        return "\n\n".join([abertura.rstrip(".") + ".", *self.paragrafos]) + "\n"

    def rights(self) -> dict:
        """`rights:` do project.yaml -- registro de procedencia, nao autorizacao.

        `status` e `licenciado` e nao `dominio-publico`: o texto original esta em
        dominio publico havia dois milenios, mas a TRADUCAO e obra derivada, com
        direito proprio do tradutor (TDD 14.3, o risco alto da tabela). O que o
        site concede esta em `licenca`, com as palavras dele.
        """
        return {
            "status": "licenciado",
            "autor": "Cânone Páli",
            "ano_morte": None,
            "tradutor": "Acesso ao Insight (Michael Beisert, editor)",
            "fonte": f"{self.referencia} — {self.url}",
            "licenca": self.licenca,
            # Fica vazio de proposito: quem verifica direitos e uma pessoa, e a
            # data de download nao e data de conferencia.
            "verificado_em": None,
        }


def normalizar_url(entrada: str) -> str:
    """Aceita a URL inteira, o caminho ou so o codigo do sutta.

    `ANIV.45`, `sutta/ANIV.45.php.html` e a URL completa levam ao mesmo lugar --
    o operador copia do navegador ou digita o codigo do indice, tanto faz.
    """
    e = entrada.strip()
    if e.startswith("http://") or e.startswith("https://"):
        return e
    e = e.lstrip("/")
    if not e.startswith("sutta/") and not e.startswith("dhp/"):
        e = f"sutta/{e}"
    if not e.endswith(".html"):
        e = f"{e}.php.html" if not e.endswith(".php") else f"{e}.html"
    return f"{BASE}/{e}"


def codigo_da_url(url: str) -> str:
    """`.../sutta/ANIV.45.php.html` -> `ANIV.45`."""
    nome = url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"\.php(\.html)?$|\.html$", "", nome)


def decodificar(bruto: bytes) -> str:
    """Bytes da pagina -> texto. Sempre windows-1252, apesar do que a pagina diz.

    Ela declara `charset=ISO-8859-1`, mas o conteudo saiu do Word: as aspas
    curvas ocupam 0x93/0x94, que em latin-1 sao caracteres de CONTROLE. Decodificar
    pelo charset declarado entrega `\x93E possivel` no lugar de `"E possivel` --
    e essas aspas sao o que `text/roles.py` usa para achar a fala citada.
    """
    return bruto.decode("cp1252", errors="replace")


def baixar(url: str, *, timeout: float = 30.0) -> str:
    """Baixa a pagina e a decodifica com `decodificar`."""
    pedido = Request(url, headers={"User-Agent": "audio-factory/0.1 (+local)"})
    try:
        with urlopen(pedido, timeout=timeout) as r:
            bruto = r.read()
    except HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"{url}: {e.reason}") from e
    return decodificar(bruto)


def extrair(pagina: str, url: str) -> Sutta:
    """Pagina crua -> `Sutta`. Levanta `ValueError` se nao for pagina de sutta."""
    m = _CORPO.search(pagina)
    if not m:
        raise ValueError(
            f"{url}: sem o marcador 'INICIO DO TEXTO' — o site devolve HTTP 200 "
            "com uma página genérica para endereço inexistente; confira o código "
            "do sutta no índice (acessoaoinsight.net/sutta/indice_suttas.php.html)")
    interior = m.group(1)

    partes = _HR.split(interior)
    cabecalho = partes[0]
    corpo_html = "".join(partes[1:-1]) if len(partes) > 2 else "".join(partes[1:])

    titulos: dict[str, list[str]] = {}
    licenca = ""
    for classe, bruto in _paragrafos(cabecalho):
        texto = _texto(bruto)
        if not texto:
            continue
        if classe.startswith("Tit"):
            titulos.setdefault(classe, []).append(texto)
        elif "distribui" in texto.lower() and not licenca:
            licenca = " ".join(texto.split())

    tit1 = titulos.get("Tit1", [])
    if not tit1:
        raise ValueError(f"{url}: nenhum título (p class=Tit1) — layout inesperado")
    referencia = (titulos.get("Tit3") or [""])[0]
    pali = tit1[0]
    portugues = tit1[1] if len(tit1) > 1 else (titulos.get("Tit2") or [""])[0]

    paragrafos, descartados = _corpo(corpo_html)
    if not paragrafos:
        raise ValueError(f"{url}: corpo vazio depois da limpeza — layout inesperado")

    return Sutta(url=url, codigo=codigo_da_url(url),
                 referencia=referencia or pali, pali=pali, portugues=portugues,
                 paragrafos=paragrafos, licenca=licenca, descartados=descartados)


def buscar(entrada: str, *, cache: Path | None = None, refazer: bool = False) -> Sutta:
    """Resolve o endereco, baixa (ou le do cache) e extrai.

    A pagina crua fica em disco de proposito: o texto publicado e a fonte de uma
    obra derivada, e um `run` de meia hora nao pode depender de o site estar no ar.
    """
    url = normalizar_url(entrada)
    guardado = cache / f"{codigo_da_url(url)}.html" if cache else None
    if guardado and guardado.exists() and not refazer:
        pagina = guardado.read_text(encoding="utf-8")
    else:
        pagina = baixar(url)
        if guardado:
            guardado.parent.mkdir(parents=True, exist_ok=True)
            guardado.write_text(pagina, encoding="utf-8")
    return extrair(pagina, url)


# --- interno -----------------------------------------------------------------


def _paragrafos(fragmento: str) -> list[tuple[str, str]]:
    """Divide um fragmento em [(classe, html)] por `<p ...>`, sem exigir `</p>`."""
    pedacos = _P.split(fragmento)
    saida = []
    for attrs, conteudo in zip(pedacos[1::2], pedacos[2::2]):
        m = _CLASSE.search(attrs)
        saida.append((m.group(1) if m else "", conteudo))
    return saida


def _texto(bruto: str) -> str:
    """HTML do Word -> texto. So o `<br>` vira quebra de linha; o resto colapsa.

    A distincao importa: `<br>` e o verso do sutta ("O fim do mundo nunca podera
    ser alcancado / por meio de viagens"), enquanto a quebra de linha crua e o
    Word quebrando a fonte na coluna 78, no meio de "Assim ouvi".
    """
    t = _COMENTARIO.sub("", bruto)
    t = _BR.sub(_SENTINELA, t)
    t = _TAG.sub("", t)
    t = _TAG_ABERTA.sub("", t)
    t = _html.unescape(t)
    t = t.replace(" ", " ").replace("​", "")
    linhas = [" ".join(l.split()) for l in t.split(_SENTINELA)]
    return "\n".join(l for l in linhas if l).strip()


def _corpo(fragmento: str) -> tuple[list[str], int]:
    """Paragrafos narraveis do corpo, e quantos foram descartados como aparato."""
    saida, descartados = [], 0
    for _, bruto in _paragrafos(fragmento):
        texto = _texto(bruto)
        if not texto:
            continue
        if _APARATO.match(texto):
            descartados += 1
            continue
        texto = _MARCA_NOTA.sub("", texto)
        texto = _NUMERO_PARAGRAFO.sub("", texto)
        texto = "\n".join(" ".join(l.split()) for l in texto.split("\n")).strip()
        if texto:
            saida.append(texto)
    return saida, descartados


def _ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _distinto(pali: str, portugues: str) -> bool:
    """O titulo em portugues acrescenta alguma coisa ao nome em pali?

    Metade dos suttas se chama "Rohitassa Sutta" e "Rohitassa": repetir isso na
    abertura da narracao e no titulo do video e ruido.
    """
    if not portugues:
        return False
    a, b = _ascii(pali).lower(), _ascii(portugues).lower()
    return b not in a and a not in b
