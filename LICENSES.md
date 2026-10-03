# Registro de licenças — Audio Factory

Auditoria exigida pela §14 do TDD. **Nenhum peso entra no pipeline sem uma linha aqui.**
Reconferir o model card a cada atualização de versão — licença de peso pode mudar entre releases.

Última verificação: 2026-10-02.

## Pesos de TTS

| Modelo | Licença | Uso comercial | Fonte verificada |
|---|---|---|---|
| `ResembleAI/chatterbox-multilingual` (V3) | MIT | ✅ Sim | model card HF + repo oficial |
| `ResembleAI/Chatterbox-Multilingual-pt-br` | MIT | ✅ Sim | model card HF (`license: mit`) |
| `ResembleAI/chatterbox-turbo` | MIT | ✅ Sim (só inglês) | model card HF |
| `hexgrad/Kokoro-82M` | Apache-2.0 | ✅ Sim | model card HF |
| `rhasspy/piper-voices` pt_BR | MIT (verificar por voz) | 🟡 Conferir voz a voz | repo HF |

## Pesos de música

| Modelo | Licença | Uso comercial | Fonte verificada |
|---|---|---|---|
| `ACE-Step/ACE-Step-v1-3.5B` | Apache-2.0 | ✅ Sim | model card HF (`license: apache-2.0`) + `LICENSE` do repo, 2026-08-30 |
| `facebook/musicgen-stereo-large` (Meta AudioCraft) | CC-BY-NC-4.0 | ❌ Não, mas **aceito aqui** — ver ressalva abaixo | model card HF (`license: cc-by-nc-4.0`), 2026-09-01 |

Roda na venv isolada `~/ai/envs/musica-ace` (ACE-Step) ou `~/ai/envs/musica-musicgen` (MusicGen), nunca no processo
do pipeline — ver `video/_ace_runner.py` e `video/_musicgen_runner.py`. Venvs separadas entre si
também: nenhuma razão para as duas dependerem da mesma fixação de `transformers`/`torch`.

⚠️ **MusicGen é NC — por que entra mesmo assim.** Esta tabela registrava o MusicGen como excluído
(ver histórico) porque a restrição de uso não-comercial não convivia com um canal monetizado. Em
2026-09-01 o operador confirmou que **o canal não publica comercialmente** — é público, mas sem
monetização. Sob essa condição, CC-BY-NC-4.0 permite o uso: a trilha entra como teste comparativo
ao lado do ACE-Step, não como substituição automática (`--musica musicgen`, motor explícito na
linha de comando). **Se o canal passar a monetizar, este peso sai do pipeline** — reconferir esta
linha nesse momento, junto com qualquer outro conteúdo já publicado com ele.

⚠️ **Ressalva honesta, registrada de propósito (vale para os dois modelos).** A senoide da V2 tinha
risco de Content ID *zero* — não havia gravação a que se parecer. Um modelo generativo é outra
coisa: o próprio disclaimer do ACE-Step alerta para "unintentional copyright infringement due to
stylistic similarity", e o mesmo raciocínio vale para o MusicGen. O risco real continua baixo,
porque o Content ID casa **gravações**, não estilos, e o leito aqui é instrumental esparso e sem
melodia reconhecível. Mas deixou de ser nulo, e essa é a moeda com que se paga o som melhor. Se um
vídeo levar reclamação de Content ID, trocar a paleta (ou motor, ou voltar para `--musica gerada`)
é a saída, e as peças aprovadas ficam em `assets/musica/` — é o material exato que gerou cada
trilha publicada, guardado como acervo justamente para poder ser apontado numa contestação. O
`cache/musica/` guarda o bruto de cada motor e as sobras de experimento, e é descartável.

## Trilhas de terceiro (gravações reais, não geradas)

| Arquivo | Composição | Gravações | Licença |
|---|---|---|---|
| `assets/musica-terceiros/mozart-k545-k333.flac` | Mozart, Sonatas K.545 e K.333 (domínio público, autor morto em 1791) | Robin Alciatore e Brendan Kinsella (Musopen, domínio público) + Bernd Krueger (piano-midi.de, **CC BY-SA 3.0 DE — exige atribuição**) | ver `assets/musica-terceiros/PROVENANCE.md` para a tabela faixa a faixa |

Usada via `--musica <arquivo>` (`video/musica.py:preparar_trilha`), que só repete o arquivo do
operador até cobrir a duração — nenhum modelo generativo envolvido. O aviso do próprio comando
("trilha de terceiro: confira a licença antes de publicar") existe porque, ao contrário das
trilhas geradas acima, aqui existe uma gravação real específica a que o Content ID do YouTube
pode casar. Pedida pelo operador para conteúdo infantil (ver ESTADO.md, 2026-09-15): sonata de
piano alegre e reconhecível, no lugar da trilha ambiente/contemplativa usada nos suttas.

## Pesos de imagem

Servidos pelo **InvokeAI** da máquina (`~/invokeai`, mantido pelo ai-stack), que o pipeline chama
por HTTP (`core/servicos/invokeai.py`). Nenhum peso de imagem roda no processo do pipeline.

| Modelo (nome no InvokeAI) | Origem do arquivo | Licença | Uso comercial | Fonte verificada |
|---|---|---|---|---|
| FLUX.1 schnell (quantized) | `InvokeAI/flux_schnell` (NF4 do `black-forest-labs/FLUX.1-schnell`) | Apache-2.0 | ✅ Sim | model cards HF (`license: apache-2.0`), 2026-10-02 |
| FLUX.1-schnell_ae (VAE) | `black-forest-labs/FLUX.1-schnell::ae.safetensors` | Apache-2.0 | ✅ Sim | model card HF |
| T5-XXL / CLIP-L (video-factory) | `text_encoder_2`/`text_encoder` do `black-forest-labs/FLUX.1-schnell`, por symlink em `~/ai/hf` | Apache-2.0 (repo do schnell) | ✅ Sim | model card HF |
| Qwen Image Edit 2511 (Q4_K_M) | `unsloth/Qwen-Image-Edit-2511-GGUF` (GGUF do `Qwen/Qwen-Image-Edit-2511`) | Apache-2.0 | ✅ Sim | model cards HF (`license: apache-2.0`, "Qwen-Image is licensed under Apache 2.0"), 2026-10-02 |
| Qwen Image VAE | `Qwen/Qwen-Image-Edit-2511::vae` | Apache-2.0 | ✅ Sim | model card HF, 2026-10-02 |
| Qwen Image Edit Lightning (4-step, bf16) | `lightx2v/Qwen-Image-Edit-2511-Lightning` | Apache-2.0 | ✅ Sim | model card HF (`license: apache-2.0`), 2026-10-02 |
| Qwen2.5-VL Encoder (fp8 scaled) | `Comfy-Org/Qwen-Image_ComfyUI` (fp8 do `Qwen/Qwen2.5-VL-7B-Instruct`) | Apache-2.0 | ✅ Sim | model cards HF (os dois `license: apache-2.0`), 2026-10-02 — atenção: o Qwen2.5-VL **3B e 72B** têm licença própria, só o 7B é Apache |

**Nenhum peso é usado sem revisar o resultado**: `gerar()` só produz rascunhos descartáveis em
`cache/imagens/`; a curadoria (`imagem-aprovar`) é o operador escolhendo o que entra em
`assets/slides/`, não uma etapa automática. A exceção deliberada é o preset `sincronizado`, em que
a revisão humana foi trocada por QA automático (`video/imagem_qa.py`) por decisão do operador.

**SD3.5 saiu do pipeline em 2026-10-02**, junto com a venv de diffusers (`.venv-imagem`): os pesos
nunca chegaram a ser baixados nesta máquina. A linha fica no histórico — Stability AI Community
License, comercial só abaixo de US$1M de receita anual; reconferir se voltar via InvokeAI.

### Modelo auxiliar de QA (não gera conteúdo)

| Modelo | Licença | Uso | Fonte verificada |
|---|---|---|---|
| `openai/clip-vit-base-patch32` | MIT (repo `openai/CLIP`) | 🟡 Só como filtro interno: nota de relevância e foto-vs-ilustração no preset `sincronizado` | repo GitHub (MIT), 2026-10-02 |

🟡 O model card do CLIP declara "**any** deployed use case of the model — whether commercial or
not — is currently out of scope" e o recomenda para pesquisa. É recomendação do card, não termo
da licença MIT; e aqui ele não produz nada que vá ao ar — só decide qual candidato do FLUX
descartar. Registrado para a decisão ficar visível: se isso incomodar, a alternativa é um
classificador com licença sem ressalva (ou voltar à curadoria humana).

Não usado para teste cujo áudio venha a ser publicado: `FLUX.1-**dev**` (licença não-comercial da
Black Forest Labs) fica fora do pipeline por esse motivo, mesmo tendo qualidade superior ao
`schnell` — decisão do operador em 2026-08-31.

## Excluídos do pipeline — pesos não-comerciais

| Modelo | Licença dos pesos | Motivo |
|---|---|---|
| F5-TTS (e todos os fine-tunes pt-br) | CC-BY-NC-4.0 | Dataset Emilia; a restrição NC é herdada por fine-tunes |
| Fish Speech / OpenAudio S1-mini | CC-BY-NC-SA-4.0 | Código Apache, pesos NC |
| XTTS-v2 (Coqui) | CPML | Não-comercial |
| IndexTTS-2 | Restritiva | Comercial exige contato com os autores |
| `black-forest-labs/FLUX.1-dev` | Não-comercial (BFL) | Melhor qualidade que o `schnell`, mas licença não permite monetização sem acordo à parte |
| `black-forest-labs/FLUX.1-Kontext-dev` | FLUX.1 [dev] Non-Commercial License | Edição com referência; seria o caminho óbvio para consistência de personagem na HQ — fica fora, o Qwen-Image-Edit (Apache-2.0) faz o papel. Verificado 2026-10-02 |
| `black-forest-labs/FLUX.1-Fill-dev` | FLUX.1 [dev] Non-Commercial License | Inpainting. O card diz que as **saídas** podem ser usadas comercialmente, mas os pesos estão sob a licença NC — fica fora pelo mesmo critério do `dev`. Verificado 2026-10-02 |
| `black-forest-labs/FLUX.2-klein-9B` | FLUX Non-Commercial License | Verificado 2026-10-02. (O Klein **4B** é outro caso — conferir o card antes de cogitar.) |

**Não usar nem para teste cujo áudio venha a ser publicado.**

## Software

| Item | Licença | Nota |
|---|---|---|
| `chatterbox-tts` (código) | MIT | — |
| `acestep` (código) | Apache-2.0 | Geração da trilha, em venv separada |
| `transformers` (código, MusicGen) | Apache-2.0 | Geração da trilha, em venv separada (`~/ai/envs/musica-musicgen`) |
| InvokeAI 6.14 (código) | Apache-2.0 (+ licenças de componentes no repo) | Servidor de geração de imagem, chamado por HTTP — não é importado nem redistribuído. Verificado 2026-10-02 |
| `diffusers` (código) | Apache-2.0 | Usado até 2026-10-02 na `.venv-imagem`, aposentada em favor do InvokeAI |
| `transformers` (código, CLIP de QA) | Apache-2.0 | No processo do pipeline, só para o CLIP do `imagem_qa` |
| PyTorch | BSD-3 | wheels cu130 |
| faster-whisper / CTranslate2 | MIT | QA por ASR |
| FFmpeg | LGPL/GPL conforme build | Usado como ferramenta, não redistribuído |

## Fontes tipográficas

| Fonte | Licença | Uso | Nota |
|---|---|---|---|
| Comic Neue Bold (`spikes/hq/fonts/ComicNeue-Bold.ttf`) | SIL Open Font License 1.1 | Letreiramento da HQ | `OFL.txt` acompanha o arquivo; uso comercial e embutir em PDF permitidos, vender a fonte sozinha não |

## Watermark

Todo áudio do Chatterbox carrega o watermark neural **Perth** (Resemble AI), resistente a MP3.
**Política do projeto: manter sempre.** Alinha-se ao disclosure de conteúdo sintético do YouTube.

## Voz

A voz do narrador é a voz do próprio operador do canal, com `voices/<id>/CONSENT.md`.
Proibido usar áudio de terceiros como referência de clonagem sem consentimento escrito.

## Imagens do vídeo

| Item | Origem | Uso comercial | Nota |
|---|---|---|---|
| `assets/slides/*.jpg` (52) | Gerações próprias do operador no **Midjourney** (conta `moisescomsal`, 2023, plano pago) | ✅ **Confirmado** | Os Termos do Midjourney atribuem os direitos sobre a saída ao assinante pago. Confirmado pelo operador em 2026-08-31 que a conta era paga na época das gerações. |
| `assets/slides/*.jpg` (novas, a partir de 2026-08-31) | Geração local com FLUX.1-schnell / SD3.5, curadas via `imagem-aprovar` | ✅ Sim, sob as licenças da tabela "Pesos de imagem" acima | Sem terceiro envolvido — pesos rodam localmente, saída é do próprio operador |

Não há terceiro envolvido: nenhum upload de imagem alheia, nenhum banco de imagens.

## Texto

Cada projeto declara `rights:` em `project.yaml`. `export` é bloqueado sem esse campo.
Atenção ao caso mais perigoso: **traduções têm direito autoral próprio do tradutor**, com prazo
próprio — uma tradução recente de um autor antigo continua protegida.
