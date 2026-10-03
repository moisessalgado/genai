"""Registry de vozes (TDD 7).

Uma voz e um diretorio auditavel: referencia, perfil de parametros congelado,
consentimento e amostras de validacao. Duas regras que o codigo impoe:

  1. sem CONSENT.md nao se cria voz -- clonagem exige consentimento documentado;
  2. os parametros ficam no profile.yaml e sao versionados: mudar expressividade
     ou seed e decisao explicita, nunca acidente de linha de comando.

`voices/` nunca vai para o git (ver .gitignore) e e o unico diretorio insubstituivel
do projeto -- e o que entra no backup cifrado.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import soundfile as sf
import yaml

from .script.models import SynthParams

TEXTO_CALIBRACAO = (
    "Em mil seiscentos e quarenta e oito, a expedição partiu rumo ao sertão "
    "desconhecido. Trezentos e cinquenta homens atravessaram rios, serras e "
    "florestas. Poucos retornaram: exaustos, doentes, irreconhecíveis. "
    "O cronista registrou apenas três palavras sobre o chefe da bandeira."
)

MIN_SEGUNDOS = 20.0
MAX_SEGUNDOS = 180.0


@dataclass
class Voz:
    id: str
    dir: Path
    referencia: Path
    params: SynthParams

    @property
    def conditionals(self) -> Path:
        return self.dir / "conditionals.pt"


def raiz_vozes(raiz: Path) -> Path:
    """Pasta do registry. Para a raiz do repositorio vale a configuracao central
    (`voices_dir`); qualquer outra raiz (testes) continua com `<raiz>/voices`."""
    from .core.config import settings

    s = settings()
    return s.voices_dir if raiz == s.raiz else raiz / "voices"


def criar(raiz: Path, voice_id: str, referencia: Path, consentimento: str | None = None,
          params: SynthParams | None = None, template_de: str | None = None,
          forcar: bool = False) -> Voz:
    """Registra uma voz a partir de um WAV de referencia.

    `template_de` marca uma voz SINTETICA (ex.: amostra do Kokoro usada como
    referencia). Nesse caso nao existe pessoa para consentir, entao em vez de
    CONSENT.md grava-se PROVENANCE.md dizendo de onde a voz veio e sob qual licenca
    -- a exigencia de rastreabilidade continua, muda so a natureza do documento.
    """
    ressalvas: list[str] = []
    audio, sr = sf.read(str(referencia), dtype="float32")
    dur = len(audio) / sr
    if template_de and dur < MIN_SEGUNDOS:
        # voz template vem de TTS: 15s ja bastam e nao ha ganho em exigir mais
        pass
    elif dur < MIN_SEGUNDOS:
        raise ValueError(f"referência curta demais: {dur:.1f}s (mínimo {MIN_SEGUNDOS:.0f}s)")
    if dur > MAX_SEGUNDOS:
        raise ValueError(f"referência longa demais: {dur:.1f}s (máximo {MAX_SEGUNDOS:.0f}s)")
    pico = float(abs(audio).max()) if audio.size else 0.0

    # Voz de pessoa passa pela analise completa da gravacao. Voz template nao:
    # ela vem de um TTS a 24 kHz, entao medir microfone e sala nao faz sentido.
    if not template_de:
        from .audio.analise import analisar

        take = analisar(referencia)
        if take.problemas and not forcar:
            raise ValueError("gravação inadequada como referência:\n  - "
                             + "\n  - ".join(take.problemas))
        # `forcar` nao apaga o defeito: ele fica gravado no profile.yaml. Uma voz
        # e um diretorio auditavel, e "por que essa voz soa assim?" precisa ter
        # resposta seis meses depois.
        ressalvas = take.problemas if forcar else []
    elif pico > 0.99:
        raise ValueError("referência com clipping — regrave com mais headroom (-6 dBFS)")

    d = raiz_vozes(raiz) / voice_id
    (d / "reference").mkdir(parents=True, exist_ok=True)
    (d / "samples").mkdir(exist_ok=True)
    destino = d / "reference" / referencia.name
    shutil.copy2(referencia, destino)

    if template_de:
        proc = d / "PROVENANCE.md"
        if not proc.exists():
            proc.write_text(_provenance(voice_id, template_de), encoding="utf-8")
    else:
        consent = d / "CONSENT.md"
        if not consent.exists():
            if consentimento is None:
                raise ValueError(
                    f"crie {consent} declarando o consentimento antes de registrar a voz")
            consent.write_text(consentimento, encoding="utf-8")

    p = params or SynthParams()
    (d / "profile.yaml").write_text(yaml.safe_dump({
        "id": voice_id,
        "criada_em": date.today().isoformat(),
        "referencia": destino.name,
        "sha256_referencia": _sha256(destino),
        "duracao_s": round(dur, 1),
        "sample_rate": sr,
        "pico": round(pico, 3),
        "tipo": "template" if template_de else "pessoa",
        "origem": template_de,
        "ressalvas": ressalvas if not template_de else [],
        "params": p.model_dump(),
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")

    try:
        d.chmod(0o700)
    except OSError:
        pass
    return Voz(voice_id, d, destino, p)


def carregar(raiz: Path, voice_id: str) -> Voz:
    d = raiz_vozes(raiz) / voice_id
    cfg_path = d / "profile.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"voz não registrada: {voice_id}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ref = d / "reference" / cfg["referencia"]
    if _sha256(ref) != cfg["sha256_referencia"]:
        raise ValueError(
            f"referência de {voice_id} mudou desde o registro — os conditionals e todo "
            "o áudio já gerado deixam de ser reproduzíveis. Registre uma nova voz.")
    return Voz(voice_id, d, ref, SynthParams(**cfg["params"]))


def listar(raiz: Path) -> list[str]:
    base = raiz_vozes(raiz)
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if (d / "profile.yaml").exists())


def _provenance(voice_id: str, origem: str) -> str:
    return f"""# Procedência da voz template — {voice_id}

Data: {date.today().isoformat()}

Esta é uma voz **sintética**, não a voz de uma pessoa. Origem: **{origem}**.

Não há pessoa identificável sendo imitada, portanto não há consentimento a colher.
O que existe é a licença do modelo de origem, que deve permitir uso comercial —
ver `LICENSES.md` na raiz do projeto.

Uso pretendido: voz provisória do canal, até a gravação da voz própria do operador.
As publicações são marcadas como conteúdo sintético, e o áudio gerado mantém o
watermark neural do Chatterbox (Perth / Resemble AI).
"""


def modelo_consentimento(voice_id: str, quem: str) -> str:
    return f"""# Consentimento de uso de voz — {voice_id}

Data: {date.today().isoformat()}

Eu, {quem}, autorizo o uso da gravação de minha voz contida em `reference/` para
treinar/condicionar um modelo de síntese de fala, e para produzir narração
sintética publicada no canal do YouTube deste projeto, inclusive em conteúdo
monetizado.

Escopo autorizado: narração de audiolivros e roteiros do canal.
Escopo NÃO autorizado: qualquer uso que atribua a esta voz declarações que eu não
fiz em contexto factual (entrevistas, depoimentos, notícias), ou uso por terceiros.

O áudio gerado mantém o watermark neural do modelo (Perth/Resemble AI), e as
publicações são marcadas como conteúdo sintético conforme exigido pela plataforma.

Assinatura: ______________________
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()
