# Estado da implementação

Documento de handoff. **Leia primeiro [`docs/TDD.md`](docs/TDD.md)** — é o design aprovado e a
justificativa de cada decisão. Este arquivo diz só onde a implementação parou.

Atualizado: 2026-08-30.

## Ambiente (já provisionado)

```bash
cd /home/moises/dev/audio-factory
./.venv/bin/python -c "import torch; print(torch.__version__)"   # 2.13.0+cu130
```

⚠️ **Armadilha de instalação (custou tempo, não repita):** `chatterbox-tts` fixa
`torch==2.6.0+cu124`, que **não roda em sm_120** (`no kernel image is available`). A ordem correta é:

```bash
uv venv --python 3.12 .venv
uv pip install chatterbox-tts                       # arrasta torch 2.6 cu124
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 torchaudio==2.11.0
uv pip uninstall $(uv pip list | grep -oE '^nvidia-[a-z-]+-cu12')   # senão o NCCL cu12 sequestra o link
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 torchaudio==2.11.0
```

Também: `torchaudio.save` no torch 2.11 exige `torchcodec` — use **`soundfile`**.
Modelos ficam em `models/` (aponte `HF_HOME` para lá).

## Configuração central e CLI por área (2026-10-02)

- **`models/` é symlink para `~/ai/hf`**, o `HF_HOME` único da máquina (definido em
  `~/.profile` e `~/.config/environment.d/ai.conf`; `~/.cache/huggingface` aponta para lá
  também). A infra de IA (Ollama/LiteLLM, InvokeAI, pesos, venvs) passa a ser do
  `~/dev/ai-workspace/ai-stack`; este repo fala com os serviços por HTTP.
- **Caminhos, venvs e URLs saem de `src/audiofactory/config.py`**, não de `RAIZ / "..."`
  espalhado. Precedência: padrão no código < `HF_HOME` < `vf.toml` (ou `$VF_CONFIG`) <
  `VF_<CAMPO>`. Os padrões reproduzem o layout de sempre; `vf.example.toml` lista os campos.
  `audio-factory doctor` mostra o valor final de cada campo, de onde veio e se o serviço
  responde.
- Os runners isolados **atribuem** o `HF_HOME` que recebem (antes: `setdefault`, e o
  herdado do ambiente venceria o `vf.toml`).
- **`cli/main.py` só registra as áreas**: `audiolivro`, `video`, `publicar`, `fontes`
  (`sutta`/`wikisource`), `voz`, `sistema`. Os nomes dos comandos não mudaram.
  `tests/test_cli_superficie.py` compara a árvore de comandos/opções/padrões com uma foto;
  mudança de propósito numa opção: `VF_REGRAVAR_CLI=1 uv run pytest tests/test_cli_superficie.py`.

## Fase 0 — CONCLUÍDA ✅

Benchmark real (`bench/fase0_bench.py`, dados em `bench/out/bench.json`), RTX 5060 Ti:

| Métrica | Medido |
|---|---|
| RTF global | **0,291** (1 h de áudio em ~17,5 min) |
| VRAM de pico | **3,49 GB** de 16 GB |
| Carga do modelo | 5,0 s |
| Sample rate | 24 kHz |

- Qualidade do pt-BR **ouvida e aprovada pelo usuário** (WAVs em `bench/out/`).
- Sobra VRAM: o `gemma4:12b` do Ollama pode coexistir; `--free-ollama` virou opcional.
- **A alucinação é real:** o `alignment_stream_analyzer` do Chatterbox detectou repetição de token e
  **forçou EOS**, ou seja, truncou o áudio silenciosamente. Por isso o QA precisa checar
  **duração real vs. esperada**, não só o texto transcrito.
- Licenças auditadas em [`LICENSES.md`](LICENSES.md).

## Fases 1–2 — EM ANDAMENTO

| Módulo | Estado | Arquivo |
|---|---|---|
| Documento canônico (`script.json`, ids por hash) | ✅ pronto e testado | `src/audiofactory/script/models.py` |
| Normalizador determinístico pt-BR (Camada 1) | ✅ pronto e testado | `src/audiofactory/narration/rules.py` |
| Chunker (≤300 chars, funde fragmentos curtos) | ✅ pronto e testado | `src/audiofactory/chunk/splitter.py` |
| Contrato de motor TTS | ✅ pronto | `src/audiofactory/engines/base.py` |
| Motor Chatterbox (+ pack pt-br, conditionals congelados) | ✅ escrito, **falta testar com voz de referência** | `src/audiofactory/engines/chatterbox_engine.py` |
| Fila SQLite / resume | ✅ pronto e testado | `src/audiofactory/store/db.py` |
| QA por ASR (CER + duração) | ✅ pronto, validado em áudio real | `src/audiofactory/qa/verify.py` |
| Política de retry melhor-de-N | ✅ pronta e testada | `src/audiofactory/qa/policy.py` |
| Pós-processamento (FFmpeg, loudnorm) | ✅ pronto, entrega −16,0 LUFS medido | `src/audiofactory/audio/process.py` |
| Orquestrador (fila → TTS → QA → disco) | ✅ pronto, resume validado | `src/audiofactory/pipeline.py` |
| Ingest TXT/EPUB + detecção de capítulos | ✅ pronto | `src/audiofactory/ingest/loader.py` |
| Projeto (project.yaml, script.json, diff.md) | ✅ pronto | `src/audiofactory/project.py` |
| CLI Typer | ✅ pronta | `src/audiofactory/cli/main.py` |
| **Pack pt-BR dedicado** | ✅ **resolvido — é o padrão** | `engines/chatterbox_engine.py` |
| Camada 2 do LLM + validador | ✅ pronta, testada com gemma4:12b real | `src/audiofactory/narration/llm.py` |
| Subcomando `iam voice` | ✅ pronto | `ai-workspace/ai-stack/bin/iam` (`cmd_voice`) |
| Registry de vozes (Fase 5) | ✅ código pronto e validado | `src/audiofactory/voices.py` |
| **Multivoz (elenco por papel)** | ✅ pronto e validado | `src/audiofactory/text/roles.py` |
| Voz template do canal | ✅ `narrador-v1` registrada e em uso | `voices/narrador-v1/` |
| Motor Kokoro (rascunho/template) | ✅ instalado e funcionando | `iam voice voice template` |
| Checagem de identidade da voz | ✅ pronta e validada contra impostores | `src/audiofactory/qa/speaker.py` |
| Gravar a voz do Moises | ⏸️ adiado — melhoria de autenticidade, não bloqueio | `iam voice voice record` |

## Fase 6 — CONCLUÍDA ✅

| Entrega | Estado | Arquivo |
|---|---|---|
| Ingest de **PDF** (por blocos, sem OCR) | ✅ pronto e testado | `src/audiofactory/ingest/loader.py` |
| `chapters.txt` + arquivo único contínuo | ✅ pronto | `cli/main.py` (`export`) |
| Export multi-formato (`--formato mp3,aac`) | ✅ pronto | `cli/main.py` |
| Relatório de QA contra os alvos do TDD | ✅ pronto | `src/audiofactory/qa/report.py` |
| 2 workers na GPU | ✅ pronto, **ganho medido de 1,25×** | `pipeline.py` (`_repartir`) |
| `--free-ollama` | ✅ pronto | `cli/main.py` (`_liberar_ollama`) |
| Histórico de execuções (tabela `runs`) | ✅ pronto — é o RTF de relógio | `store/db.py` |
| Testes automatizados | ✅ 46 testes | `tests/` |

### PDF: extração por BLOCOS, não por linhas

`ler_pdf()` usa `page.get_text("blocks")`. Reconstruir parágrafo a partir de
pontuação erra sempre que uma frase termina no meio do parágrafo — e num PDF cada
linha visual é uma quebra. O bloco do PyMuPDF já é, na prática, o parágrafo.

Cabeçalho e rodapé saem por repetição: um bloco curto (≤ 80 chars) na borda da
página que aparece em ≥ 30% das páginas é descartado. A chave de comparação apaga
os dígitos, senão "Página 12" e "Página 13" nunca casariam.

**Guarda que um teste obrigou a existir:** só entram na contagem páginas com ≥ 3
blocos, e a janela de borda encolhe para 1 bloco em páginas curtas. Sem isso, numa
página de dois blocos *todo* bloco é borda — e como a chave apaga dígitos, o corpo
do livro ("Único parágrafo da página 3.") seria descartado como rodapé.

PDF escaneado é **recusado com erro claro**: OCR está fora do escopo (TDD §17), e
um OCR silencioso entregaria lixo ao TTS.

### 2 workers: ganho real de 1,25×, não de 2×

Medido em 46 chunks / 12,4 min de áudio, com QA ligada:

| | 1 worker | 2 workers |
|---|---|---|
| Relógio de parede | 305 s | **244 s** |
| RTF (parede/áudio) | 0,407 | **0,324** |
| VRAM de pico | 6,6 GB | **10,5 GB** de 16 |

**Não é 2×** porque o ASR de QA roda na CPU e a GPU já estava bem ocupada com um
worker. E **em carga pequena 2 workers é mais LENTO**: em 9 chunks deu 48 s contra
36 s de 1 worker — cada worker carrega o seu próprio modelo, e dois carregamentos
não se pagam em 50 s de áudio. Regra prática: `--workers 2` só a partir de alguns
minutos de áudio pendente.

Cada worker tem modelo, ASR e conexão SQLite próprios. Modelo próprio é
obrigatório: `set_voice` guarda os conditionals **dentro** do modelo, e dois
workers compartilhando um trocariam a voz um do outro no meio do lote.
`_repartir()` dá a cada worker fatias contíguas de cada voz, para que `set_voice`
seja chamado uma vez por voz, não uma vez por chunk.

### RTF passou a ser medido pelo relógio de parede

A tabela `runs` (que existia no schema e nunca era usada) agora registra cada
execução. O RTF do relatório é `parede / áudio` somado sobre as execuções, e
**inclui a carga do modelo** — é o tempo que o operador espera de verdade. A soma
de `gen_s` por chunk não serve: com 2 workers ela conta o mesmo intervalo duas
vezes.

### Identidade da voz entrou no melhor-de-N

**Era um furo real:** a similaridade de voz era medida *depois* de escolher a
melhor tentativa, então um chunk que só falhava na voz ia direto para
`needs_review` sem que outra seed fosse tentada. Medido no teste do PDF: um chunk
deu 0,879 (limiar 0,88) e passou com folga na seed seguinte.

Agora a similaridade é medida **por tentativa**, dentro de `_gerar_com_qa`, e
entra em `deve_repetir`/`escolher` como qualquer outro defeito. Reprovar por voz
só acontece se **nenhuma** das 3 tentativas bater o limiar.

### Loudness: o `loudnorm` sozinho não chega ao alvo

Os masters saíam a **−16,4 / −16,5 LUFS**, na borda da tolerância de ±0,5 do TDD.
Investigado filtro a filtro: **não é o limiter nem os fades** — é o próprio
`loudnorm`. Quando o ganho necessário estouraria o true peak alvo (aqui: +8,8 dB
sobre um material de −4,1 dBTP), ele abandona o modo linear e cai no dinâmico
(`normalization_type: dynamic` na medição), e o resultado fica sistematicamente
~0,5 LUFS abaixo mesmo com o `target_offset` aplicado.

`_corrigir_ganho()` mede o master e corrige o resíduo com ganho linear (`volume`),
com o limiter depois só como guarda de true peak. Ganho linear não altera
dinâmica, então é seguro. **Medido depois da correção: −16,10 e −16,14 LUFS, TP
−1,44.** O limiter também passou de −1,0 para `TP_ALVO` (−1,5), que era a
inconsistência que sobrava na cadeia.

## Detalhes que já foram decididos e não devem ser re-litigados

- **`num2words` pt_BR insere vírgula** ("mil, seiscentos e quarenta e oito") — `_num()` remove, senão
  vira pausa espúria na narração.
- **Fragmentos curtos são fundidos ao vizinho** (`MIN_CHARS=25`) — chunk curto faz o Chatterbox
  produzir gibberish.
- **`Store.sync_script` preserva os chunks já `ok`** e apaga os órfãos: corrigir o capítulo 3 não
  regenera o livro inteiro.
- **`reset_stale()`** devolve chunks presos em `running` para a fila — é a recuperação de `kill -9`.
- **Conditionals de voz são calculados uma vez por livro** (`engine.set_voice`), nunca por capítulo —
  é o que impede a voz de derivar de identidade entre capítulos.

## Teste ponta a ponta que JÁ PASSA

```bash
cd /home/moises/dev/audio-factory
./.venv/bin/audio-factory doctor
./.venv/bin/audio-factory new bandeiras --from books/teste.txt --titulo "A História das Bandeiras"
./.venv/bin/audio-factory script bandeiras       # gera script.json + diff.md
HF_HOME=$PWD/models ./.venv/bin/audio-factory run bandeiras --no-ptbr-pack
./.venv/bin/audio-factory status bandeiras
./.venv/bin/audio-factory build bandeiras
./.venv/bin/audio-factory export bandeiras
```

Verificado: CER médio 0,019 · resume após `running` órfão regenera **só** o chunk morto ·
entregável medido em **−16,0 / −16,3 LUFS**.

**Ponta a ponta a partir de PDF** (Fase 6), verificado em 30/08/2026:

```bash
./.venv/bin/audio-factory new pdfteste --from livro.pdf --narrator narrador-v1
./.venv/bin/audio-factory script pdfteste
HF_HOME=$PWD/models ./.venv/bin/audio-factory run pdfteste --workers 2 --free-ollama
./.venv/bin/audio-factory build pdfteste
./.venv/bin/audio-factory export pdfteste --formato mp3,aac   # + chapters.txt
./.venv/bin/audio-factory report pdfteste
./.venv/bin/python -m pytest tests/ -q                        # 19 passam
```

Resultado: 9/9 chunks ok · CER médio 0,0048 · identidade de voz 0,928 (pior 0,886)
· masters a **−16,10 / −16,14 LUFS**, TP −1,44 · `chapters.txt` com carimbos
cumulativos + `pdfteste-completo.mp3`.

`report` sai com **status 1** se alguma métrica ficar fora do alvo — dá para usar
como portão antes de publicar.

## Pack pt-BR — RESOLVIDO (é o padrão do motor)

O `ResembleAI/Chatterbox-Multilingual-pt-br` não é um checkpoint completo: publica só
`t3_pt_br.safetensors`, `s3gen_v3.pt` e o tokenizer. `montar_ckpt_ptbr()` compõe por symlink
um diretório com os nomes fixos que `from_local` exige, pegando do pack o que existe e do
`ResembleAI/chatterbox` o que falta (`ve.pt`, `conds.pt`, `Cangjie5_TC.json`, `s3gen.pt`).

Duas armadilhas já resolvidas, **não repita**:
1. `snapshot_download` **sem `allow_patterns` puxa o repo inteiro** (vários GB de variantes
   inúteis) e demora minutos. Use sempre a lista de arquivos.
2. O `s3gen_v3.pt` do pack é de uma versão mais nova da lib e falha com
   `Missing key(s): tokenizer._mel_filters, tokenizer.window`. **O vocoder vem do base** — o
   refinamento pt-BR está no T3, e o s3gen é agnóstico ao idioma.

**Ganho medido** (mesmos 5 chunks, mesma seed, QA com whisper small):

| | genérico | pack pt-BR |
|---|---|---|
| CER médio | 0,0274 | **0,0125** (2,2× melhor) |
| RTF | 0,291 | 0,305 |
| VRAM | 3,49 GB | 3,49 GB |

O chunk que eu havia atribuído a ruído do ASR era defeito real do genérico: ele produzia
"Ou que os homens retornaram" onde o pack produz "Poucos homens retornaram". O custo é ~5% de
RTF. Padrão do motor: `use_ptbr_pack=True`; `--no-ptbr-pack` desliga.

## Voz do canal — `narrador-v1` (template, em uso)

Decisão do usuário: usar uma voz template por enquanto; a voz própria fica como
melhoria de autenticidade mais adiante.

A voz em uso é o **Chatterbox clonando uma referência sintetizada pelo Kokoro
(`pm_alex`, Apache-2.0)**. Escolhida em teste cego entre 5 candidatas. O motivo de não
usar o Kokoro direto como motor: assim o motor continua sendo o Chatterbox, e trocar
pela voz real do Moises depois é só registrar outra voz — sem mudar engine, sem
reprocessar nada além do áudio.

`voices/narrador-v1/` tem `PROVENANCE.md` (voz sintética não tem consentimento a
colher, mas tem procedência a documentar) e o hash SHA-256 da referência.

**Para trocar pela voz real depois:**
```bash
iam voice voice record                 # instruções + texto de calibração
iam voice voice new moises-v1 --reference take2.wav
# depois, em cada project.yaml: narrator: moises-v1
```

## Identidade da voz — limiar calibrado com medição própria

`qa/speaker.py` compara cada chunk com a referência usando o voice encoder do próprio
Chatterbox. Medido com a `narrador-v1` (6 chunks, 3 capítulos):

| | similaridade |
|---|---|
| mesma voz (6 chunks, 3 capítulos) | 0,917 – 0,960 |
| voz feminina diferente | 0,833 |
| outra voz masculina (kokoro pm_santa) | 0,823 |
| voz embutida do Chatterbox | 0,679 |

**Limiar 0,88**, no meio do vão. Um chunk abaixo disso vai para `needs_review` mesmo
com CER perfeito — o CER garante que o texto está certo, não que a voz é a mesma.

## Multivoz — elenco por papel

`project.yaml` ganha `cast:` mapeando papel → voice_id. O papel `narrador` é o padrão;
`citacao` (fala entre aspas) é detectado automaticamente; marcação explícita
`[[voz:personagem_a]]` no texto permite elenco nominal.

```yaml
narrator: narrador-v1
cast:
  citacao: citacao-v1
```

Três decisões que o teste real impôs:

1. **Síntese agrupada por voz.** Preparar conditionals custa segundos; alternar voz a
   cada fala pagaria isso milhares de vezes. O pipeline ordena por `voice_id`, e a ordem
   de narração é restaurada na montagem do capítulo (que lê do banco).
2. **`MIN_SINTETIZAVEL = 25`.** Se a divisão por papel produzir qualquer caco menor que
   isso, a divisão inteira é abandonada e o parágrafo vira narração. Motivo medido:
   "respondeu o diabo." (18 chars) saiu a **5,8 c/s** contra 14–17 c/s do normal — o
   modelo arrasta o áudio. Diálogo picado (fala curta / atribuição curta / fala curta)
   **não** sobrevive à troca de voz; lido inteiro pelo narrador fica correto, só menos
   teatral. Citação longa mantém voz própria.
3. **Checagem de identidade por voz.** Cada papel é comparado com a *sua* referência.
   Verificado: chunk de citação deu 0,955 contra `citacao-v1` (contra a do narrador daria
   ~0,83, e seria barrado indevidamente).

O `chunk_id` embute a voz: trocar a voz de um papel invalida só os chunks daquele papel.

## `rights` é registro, não autorização

**Decisão do operador (2026-08-30): o export não é bloqueado por `rights.status`.**
O campo continua no `project.yaml` como registro de procedência — útil para o próprio
histórico e para os metadados —, e o `export` apenas ecoa o valor declarado.

A decisão editorial sobre o que publicar é do operador do canal, não da ferramenta.
**Não reintroduza um gate aqui** sem que ele peça: já foi removido deliberadamente.

`RIGHTS_CONHECIDOS` em `project.py` lista só os valores convencionais, para consulta.

## Depois disso

1. **Primeiro livro real** de domínio público (~5 h), ponta a ponta. O discurso do
   Krishnamurti (17 min) já validou o caminho completo; falta a escala e um texto
   publicável.
2. **V2**: música/trilha com ducking no Chapter Builder.
3. **UI de revisão** (TDD §18, Fase 7): só se `needs_review` virar gargalo real.

## Primeiro texto real ponta a ponta — CONCLUÍDO ✅

Discurso de Dissolução da Ordem da Estrela, 15 KB, `projects/krishnamurti-completo/`,
uma voz só (`narrador-v1`, `cast` vazio). **17,3 min de MP3 entregues a −16,10 LUFS
/ −1,35 dBTP**, 133 de 133 chunks aprovados, CER médio 0,0088, identidade de voz
média 0,939. RTF de 0,44 com 2 workers (17,8 min de áudio em 7,9 min de relógio).

O valor do teste não foi o áudio: foi expor **quatro defeitos que só aparecem em
texto real**, todos de calibração feita em amostra pequena.

### 1. A detecção de capítulos comia texto

Duas linhas sumiram em silêncio — o pior tipo de defeito aqui, porque o áudio sai
correto e só falta um pedaço.

- `_TITULO_ROMANO` tinha o separador **opcional**, então "**D**iscurso de Dissolução"
  casava como numeral romano "D" mais título. I, V, X, L, C, D e M abrem meia língua
  portuguesa ("Como", "Livro", "Vocês", "Isso", "Minha").
- O que vinha **antes do primeiro marco era descartado**: rosto, autor, epígrafe.
  Agora vira um capítulo "Abertura".
- Em `_CAPITULO`, o `[a-zà-ú]+` do número casava com qualquer palavra ("Livro dos
  dias", "Parte de mim"). Virou lista fechada de ordinais. E o romano precisou de
  `(?-i:...)`: o padrão é IGNORECASE, e o "d" de "dos" casava como numeral.

### 2. O piso de 10 c/s reprovava chunk curto perfeito

7 dos 8 primeiros reprovados tinham **CER 0,000**. O overhead (ataque, respiração,
pausa final) não encolhe com o texto. Ajustado sobre os 126 chunks aprovados:

```
dur = 0,83 s + chars / 15,4        razão dur/esperada observada: 0,79 a 1,30
```

O lado **curto demais** (truncamento) continua apertado, porque discrimina bem:
cortar metade dá razão ~0,50. O lado **longo demais** ficou generoso de propósito —
medido, um título lido com pausa deu razão **1,70 com CER 0,000**, e o arrasto real
documentado neste arquivo deu **1,54**. As duas populações se sobrepõem: duração
sozinha não as separa, e ali quem manda é o CER.

### 3. O limiar de voz 0,88 acusava áudio curto, não voz trocada

Sobraram 5 chunks reprovados por similaridade (0,839–0,879) com texto perfeito.
Medido cortando os **mesmos** chunks em várias durações, controle `citacao-v1`:

| corte | mesma voz (min) | impostor (máx) | vão |
|---|---|---|---|
| 2 s | 0,798 | 0,728 | +0,070 |
| 3 s | 0,866 | 0,756 | +0,109 |
| 4 s | 0,898 | 0,779 | +0,119 |
| 6 s | 0,911 | 0,791 | +0,119 |
| cheio | 0,929 | 0,791 | +0,137 |

O vão continua aberto, mas **a escala inteira desce** — impostor incluído. O limiar
passou a acompanhar a duração (`limiar_por_duracao`), e abaixo de **2,5 s não opina**:
+0,070 não dá para julgar. **Os 0,88 do áudio longo ficam intactos** — foram
calibrados contra um impostor mais próximo (0,833) que o deste experimento, e é a
medição conservadora que vale.

### 4. `needs_review` não tinha saída

`build` recusa capítulo com qualquer chunk fora de `ok`, então **um trecho de 2,5 s
travava 17 min de entregável**. O caso: o nome "Jiddu Krishnamurti". O whisper small
transcreveu o mesmo áudio como `Jidu-Criis-Namuji`, `Jidu-Klišnamurči` e
`Jidu-Crišna-Murchi` — três decodificações foneticamente certas, ortograficamente
diferentes. O CER media a grafia de um nome sânscrito, não a pronúncia; **nenhuma
seed resolve**, porque o defeito não está no áudio.

`review --aprovar <chunk_id>` aceita depois de um humano ouvir, e grava no banco que
a decisão foi humana. `--regerar` devolve à fila.

### Armadilha de cache que este teste revelou

O `chunk_id` embute o **índice** do segmento (`ch01/00042-<hash>`). Uma entrada de
léxico que muda o número de segmentos desloca todos os índices seguintes e
**regenera o capítulo inteiro** — aqui, 131 de 133 chunks. A promessa de "corrigir um
capítulo não regenera o livro" vale **entre** capítulos, não dentro de um. Num livro
com capítulos de verdade o estrago é limitado ao capítulo editado; num texto de
capítulo único, é tudo. Editar o léxico **antes** do primeiro `run` evita o retrabalho.

### Sobre a taxa de regeneração no relatório

O `report` deste projeto acusa 13,3%, fora do alvo de 5%. O número soma as **4
execuções**, incluindo as duas que refizeram chunks só porque os limiares mudaram no
meio da investigação. Não é a taxa de um run limpo.

### Krishnamurti: o teste original rodou só com um trecho

`projects/krishnamurti/` foi gerado a partir de `books/krishnamurti-trecho.txt`,
que tem **1 KB** — três parágrafos, escolhidos para testar multivoz. O `ch01.mp3`
que está em `output/` é esse trecho, e é por isso que soa incompleto: **o texto é
que está incompleto**, não o pipeline. Para o discurso inteiro basta substituir a
fonte e rodar de novo; o `rights:` do projeto registra que a tradução é do site da
KF e o status é `TESTE-LOCAL-NAO-PUBLICAR`.

## Trilha de fundo: ACE-Step em venv separada

A trilha sintetizada (`audio/musica.py`) soava **abstrata e sinistra** — pad de ficção científica.
Ver `docs/TDD.md` §9.1 para o diagnóstico completo. A trilha padrão agora vem do **ACE-Step v1
3.5B** (Apache-2.0), gerado localmente.

⚠️ **Venv separada, não negociável:** o `acestep` fixa `transformers==4.50.0` e `datasets==3.4.1`.
Instalar na `.venv` do TTS quebra um dos dois. A fronteira é um subprocesso que troca JSON e WAV
(`audio/_ace_runner.py`) — nenhum objeto Python atravessa.

```bash
uv venv --python 3.12 .venv-musica
uv pip install --python .venv-musica/bin/python git+https://github.com/ace-step/ACE-Step.git
```

Boa notícia: **a armadilha do cu124 não se repete aqui.** O `torch` do PyPI já resolve para
`2.13.0+cu130`, que roda em sm_120 — não é preciso o passo de reinstalação por índice.

Os pesos (8,3 GB) baixam sozinhos na primeira geração, para `models/ace-step/`. As peças **não
dependem do capítulo nem do projeto**, só da paleta: gerar uma vez serve para todos os livros.

**A trilha é o padrão do `video` desde 2026-08-31.** Antes o padrão era `nenhuma`, e o resultado
previsível: o vídeo do sutta foi renderizado mudo sem que ninguém tivesse decidido isso. O canal
publica com trilha; quem quer sem, pede. Numa máquina sem a `.venv-musica`, o padrão degrada com
aviso e renderiza só a narração — mas `--musica ace` *pedido* na linha de comando continua sendo
erro, que é a diferença entre "esta máquina não tem" e "pedi e não veio".

```bash
audio-factory video dhammacakka                         # trilha contemplativo, de fábrica
audio-factory video dhammacakka --musica ace:sobrio     # cordas graves, mais sóbrio
audio-factory video dhammacakka --musica gerada         # sintetizador antigo, sem GPU
audio-factory video dhammacakka --musica nenhuma        # sem trilha
```

### Onde a música fica

Duas pastas, e a divisão é **o que pode ser apagado sem consequência**:

| Pasta | O quê | Apagar custa |
|---|---|---|
| `assets/musica/` | as peças prontas (24 kHz mono), 12 aprovadas, ~100 MB | os vídeos já publicados soam com elas — um capítulo remontado depois de perder o arquivo não combina mais com os irmãos que foram ao ar |
| `cache/musica/` | o bruto do ACE-Step (48 kHz estéreo, 23 MB/peça) e as sobras de experimento | GPU, e só |

É a mesma distinção de `assets/slides`: acervo do canal de um lado, subproduto regenerável do
outro. As peças que estavam em `cache/musica/` **são movidas sozinhas** para `assets/musica/` na
primeira vez que o `video` roda — mover em vez de regerar mantém a música dos vídeos publicados
bit a bit igual.

`assets/` inteiro continua fora do git (ver `.gitignore`). Num clone limpo a pasta não existe e as
peças são geradas de novo; como a seed sai do nome da paleta, voltam idênticas.

## Trilha de fundo: MusicGen Stereo, comparativo ao ACE-Step (2026-09-01)

O operador achou a maioria das peças do ACE-Step feia — da paleta `flauta`, só uma das seis
sobreviveu à escuta. `audio/musica_musicgen.py` traz o **MusicGen Stereo 3.3B** (Meta AudioCraft)
como motor alternativo, mesmas paletas (mesmo nome, mesmo timbre-alvo) para comparação lado a
lado. Os pesos são CC-BY-NC-4.0 — antes descartado por isso (ver histórico do `LICENSES.md`), mas
o operador confirmou em 2026-09-01 que **o canal não publica comercialmente**, o que torna o uso
aceitável. Ver `LICENSES.md` para a ressalva completa, inclusive o que fazer se o canal passar a
monetizar.

⚠️ **Venv separada da do ACE-Step, não só do resto do pipeline:** nenhuma razão para as duas
dependerem da mesma fixação de `transformers`/`torch`. A fronteira é um subprocesso que troca JSON
e WAV (`audio/_musicgen_runner.py`), mesmo desenho do `_ace_runner.py`.

```bash
uv venv --python 3.12 .venv-musica-mg
uv pip install --python .venv-musica-mg/bin/python transformers accelerate soundfile scipy sentencepiece
```

O `torch` do PyPI resolveu sozinho para `2.13.0+cu130` aqui também — mesma sorte do ACE-Step, sem
precisar do índice `cu130` à parte (diferente da venv de imagem).

Diferença de desenho que importa: o MusicGen degrada (repete, deriva de tom) em geração contínua
além de ~30 s — é o tamanho de trecho predominante no treino. Por isso a peça do MusicGen é mais
curta que a do ACE-Step (30 s contra 120 s) e o leito usa mais peças por padrão (8 contra 6) para
cobrir duração parecida de material inédito — a variedade continua vindo de peças emendadas, não
de uma peça longa.

```bash
audio-factory musica --motor musicgen --paleta flauta      # gera as peças, para ouvir antes
audio-factory video dhammacakka --musica musicgen           # trilha padrão do MusicGen
audio-factory video dhammacakka --musica musicgen:sobrio    # paleta específica
```

As peças do MusicGen entram no mesmo `assets/musica/` do ACE-Step, com prefixo `mg-` no nome do
arquivo para não colidir — mesma divisão acervo/cache, mesmo motivo (ver seção anterior).

## Slides: geração local com FLUX + SD3.5 (2026-08-31)

Antes, todo o acervo de `assets/slides/` vinha do Midjourney. Agora é possível gerar novas imagens
localmente com **FLUX.1-schnell** (Apache-2.0) e **Stable Diffusion 3.5** (Stability AI Community
License), pelo mesmo motivo de sempre: licença comercial em todas as camadas, sem serviço pago em
runtime. Ver `LICENSES.md` para a auditoria completa, incluindo por que `FLUX.1-dev` fica de fora
(licença não-comercial).

⚠️ **Venv separada, não negociável** — mesma razão do ACE-Step: `diffusers`/`transformers` não
precisam conviver com as versões fixadas pelo resto do pipeline. A fronteira é um subprocesso que
troca JSON e PNG (`video/_imagem_runner.py`), nenhum objeto Python atravessa.

```bash
uv venv --python 3.12 .venv-imagem
uv pip install --python .venv-imagem/bin/python diffusers transformers accelerate \
    sentencepiece protobuf torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**SD3.5 é *gated* no Hugging Face** (FLUX.1-schnell não é): é preciso aceitar a licença no model
card (`stabilityai/stable-diffusion-3.5-medium` e/ou `-large`) e autenticar — `huggingface-cli
login` ou `HF_TOKEN` no ambiente — antes do primeiro `audio-factory imagem --modelo sd`.

Fluxo de uso, em duas etapas separadas de propósito — nem toda imagem gerada presta, e a curadoria
é humana, diferente da trilha (onde a paleta inteira vira acervo automaticamente):

```bash
audio-factory imagem "paisagem oriental ao entardecer, aquarela" --modelo flux --n 4
# candidatos em cache/imagens/ — olhe, escolha os que prestam
audio-factory imagem-aprovar cache/imagens/flux-a1b2c3d4-123456789.png
# vira JPEG q2 ≤1920px em assets/slides/, mesmo padrão das imagens do Midjourney
```

`--modelo sd` usa SD3.5-**Medium** por padrão (2,5B parâmetros, cabe em 16 GB sem offload, mais
rápido); `--modelo sd:large` troca para o Large (8B, melhor qualidade, mais VRAM — o runner liga
`enable_model_cpu_offload()` para caber mesmo assim). Passos e guidance têm padrão por família
(FLUX-schnell é destilado: 4 passos, guidance 0 — CFG não faz nada nesse modelo; SD3.5 é
convencional: 28 passos, guidance 4,5) e podem ser sobrepostos com `--passos`/`--guidance`.

## Fundo do vídeo: slides (padrão desde 2026-08-31)

O preset padrão do `video` passou de `ondas` para **`slides`**: as imagens do canal em
`assets/slides/` trocando por dissolve de 2 s, uma a cada 45 s. Substitui o `gradiente`, que
existia só para a tela não congelar e não dizia nada. Os presets antigos continuam todos lá.

```bash
audio-factory video dhammacakka                          # slides, sorteio novo a cada render
audio-factory video dhammacakka --slides-seed 7          # repete o sorteio de um render
audio-factory video dhammacakka --slides-seg 30          # troca mais rápida
audio-factory video dhammacakka --preset gradiente       # o fundo antigo
```

- **52 imagens** no acervo (`assets/slides/`, 25 MB), JPEG q2 com no máximo 1920 px — as originais
  em PNG somavam 110 MB. Duplicatas exatas removidas.
- ⚠️ **`assets/` não é versionado** (está no `.gitignore`): 25 MB entrariam no histórico para
  sempre. Num clone limpo a pasta não existe e o preset padrão falha com uma mensagem explícita —
  use `--slides-dir` apontando para o acervo, ou `--preset gradiente`. O acervo vive na máquina
  que publica; origem e licença em `LICENSES.md`.
- O acervo é conferido **antes** do laço de capítulos, como a paleta da trilha.
- As imagens são quadradas ou panorâmicas, quase nenhuma em 16:9. Entram inteiras, e o resto do
  quadro recebe a própria imagem ampliada e desfocada — o desfoque é feito pequeno e esticado,
  porque `gblur` em 1080p custaria caro em todo quadro de um still parado 45 s.
- **Véu no rodapé** (420 px, até 85% de preto, rampa com expoente 2,4) só quando há legenda
  queimada: o `gradiente` era escuro por baixo do texto de graça, arte clara não é.

### As listras nas tarjas laterais (consertado em 2026-08-31)

O primeiro MP4 com slides saiu com listras de cor nas tarjas borradas. Não era uma causa, eram
três, e as três estão no caminho do degradê quase plano do borrão:

1. **`-cq 31` no NVENC** rendia **380 kb/s em 1080p**. O codificador gastava esse orçamento onde
   havia detalhe, e as áreas lisas saíam com blocos inteiros no mesmo valor. Agora `cq 20` com
   `spatial-aq` — que existe justamente para repartir o orçamento a favor das áreas lisas, que é
   onde o olho enxerga banda. Custa ~3× o arquivo (46 MB → ~150 MB em onze minutos), irrelevante
   num upload único.
2. **Borrão calculado em 192×108.** Um pixel virava um bloco de 10×10 na tela, e o degradê que
   sobrava tinha passos largos demais para o `scale` interpolar. Agora 480×270 com `sigma=15` —
   mesmo raio em fração do quadro, passo de 4 px.
3. **8 bits.** Rampa quase plana bandeia por aritmética, antes de qualquer codec. O grafo agora
   trabalha em `gbrp10le` e só despeja para 8 bits no último elo, com `zscale=dither=error_diffusion`.

Medido no degrau médio da tarja lateral (comprimento das corridas de valor idêntico ao longo
de Y — quanto maior, mais larga a listra), no mesmo slide do ch01: **13,2 px → 1,9 px**. Num
still isolado, onde o codificador tem orçamento de sobra e o defeito é mais difícil de
provocar: 5,1 → 2,5 px, com PSNR contra o quadro não codificado subindo de 37,7 para 46,4 dB.

De brinde, a saída deixou de ser `yuvj420p` (faixa cheia, marcada) e passa a ser `yuv420p` bt709
faixa limitada, convertida de verdade — player que ignora a marca esmagava preto e branco.

- Custo medido no ch01 (11 min 10 s de áudio, `h264_nvenc`): **2 min 22 s de parede**, ou
  ~4,7× tempo real, contra ~6× antes. O arquivo foi de 46 MB para 124 MB.
- **O véu é estático.** Ele vinha de um `gradients`, que anima por padrão — a cor circula em
  `speed` — e o rodapé pulsava a noite toda. Movimento sem motivo, exatamente o que o resto do
  módulo evita. Agora um `geq` desenha a rampa **uma vez** (`trim=end_frame=1` + `loop`), o que de
  quebra tira o custo por quadro: `geq` sobre 1920×420 px em 16 mil quadros não é barato.

## Suttas do Acesso ao Insight — `sutta <url>` ✅

Um comando faz o pipeline inteiro a partir de uma página de
[acessoaoinsight.net](https://www.acessoaoinsight.net): baixa, limpa, monta o script,
sintetiza, masteriza, renderiza o MP4 e sobe para o YouTube como privado. Aceita
vários endereços de uma vez, porque o objetivo é uma série de vídeos e não um.
Documentação de uso em [`docs/suttas.md`](docs/suttas.md).

```bash
audio-factory sutta ANIV.45 SNLVI.11 MN58        # o código do índice basta
audio-factory sutta ANIV.45 --ate script         # para para você ler o diff.md
```

Voz padrão: **`narrador-v2`** (`pm_alex` a 0,75 da velocidade). Sutta se acompanha
devagar, e é a voz que `dhammacakka` e `satipatthana` já usavam — o default nasceu
errado, na `v1`, e foi corrigido em 01/09/2026. Medido nos projetos em `v2`:
identidade média 0,934–0,942, na mesma faixa da `v1`.

**Verificado ponta a ponta em 01/09/2026** com o `ANIV.45` (Rohitassa Sutta):
17/17 chunks `ok`, RTF 0,45, identidade de voz média 0,947 (pior 0,914), master a
**−16,1 LUFS**, MP4 1920×1080 de 3:04, publicado como privado
(`bBexVhAk4tI`) com miniatura extraída do próprio MP4. A segunda execução do
mesmo comando recusou o upload pelo `output/publicado.json`, como previsto.

### O que a página exigiu que o parser soubesse

Estrutura medida em **40 páginas sorteadas do índice**: 40/40 com um `Tit3`
(referência) e dois `Tit1` (pali, português), 40/40 com o bloco de licença, 37/40
com dois `<hr>` e 3/40 com um só — os três são suttas curtos sem seção de notas.
Daí a regra: o corpo vai do PRIMEIRO `<hr>` ao ÚLTIMO; havendo um só, até o fim.

Quatro armadilhas que custaram medição:

1. **`charset=ISO-8859-1` é mentira** — é `windows-1252`. As aspas curvas do Word
   ocupam 0x93/0x94, que em latin-1 são caracteres de *controle*. Decodificar pelo
   charset declarado entrega `\x93É possível`, e são justamente essas aspas que o
   `text/roles.py` usa para achar a fala citada.
2. **Endereço inexistente devolve HTTP 200** com uma página de busca genérica. Sem
   checar o marcador `INICIO DO TEXTO`, um código errado criaria um projeto vazio e
   só quebraria no `export`, depois da síntese.
3. **`<br>` é verso; a quebra de linha crua não é.** O Word quebra a fonte na coluna
   78, no meio de "Assim ouvi". Preservar as duas corta frases ao meio.
4. **O último `<p>` carrega o `<!--` que abre o rodapé**, sem `>` para fechar — e a
   remoção de tags só casa tag fechada. Sem guarda, `<!--` virava parágrafo e ia ao TTS.

### A referência da coleção não é narrada — por medição

`normalize("Anguttara Nikaya IV.45")` devolve **"Anguttara Nikaya IVquarenta e
cinco"**: o romano fica sem leitura e o `.45` vira cardinal colado. A referência vive
no título do vídeo e na descrição. A abertura narrada é só `<pali>. <título em
português>.`, e o português é omitido quando repete o pali ("Rohitassa Sutta" /
"Rohitassa").

### `rights` é `licenciado`, e monetizar contraria os termos

O sutta é antigo; a **tradução** é obra derivada com direito próprio do tradutor —
o risco alto do TDD §14.3. Declarar `dominio-publico` aqui seria falso. A licença
("Somente para distribuição gratuita … contanto que nenhum custo seja cobrado pela
distribuição ou uso") é colhida da própria página, gravada em `rights.licenca` e
**reproduzida literal na descrição do vídeo** — é assim que os termos viajam com a
obra derivada.

O comando **avisa** que monetizar vai contra a licença e **não bloqueia** nada: a
decisão editorial continua sendo do operador, como já decidido neste arquivo.
`verificado_em` fica `null` de propósito — data de download não é data de
conferência.

### Publicar duas vezes era um furo real

`videos.insert` **não é idempotente**: repetir o comando cria OUTRO vídeo no canal.
Como o `sutta` existe para ser re-rodado (é assim que se retoma um lote), um upload
feito passou a ser registrado em `output/publicado.json`, e o segundo é recusado.
`--republicar` sobe outra cópia de propósito. A guarda vale para o `publish`
avulso também.

### Refactor que isto obrigou

Os corpos de `script`, `run`, `build`, `export`, `video` e `publish` viraram
`_etapa_*(p, ...)` com defaults de verdade, e os comandos Typer viraram invólucros.
Motivo: os defaults de um comando Typer são objetos `OptionInfo`, não valores —
chamar `run(slug)` de dentro de outro comando passaria `OptionInfo` como `workers`.

### O `uv sync` derrubou a GPU — agora o pyproject impede

**01/09/2026:** um `uv sync` rodado só para instalar as dependências novas do
`publish` re-resolveu o `torch` para o **`2.6.0+cu124`** que o `chatterbox-tts`
fixa, e toda síntese passou a morrer com `no kernel image is available for
execution on the device`. É a armadilha do topo deste arquivo, disparada sem que
ninguém tocasse no torch de propósito.

O reparo manual continua válido (topo do arquivo). O que mudou é que ele não
deve ser preciso de novo: o `pyproject.toml` passou a declarar

```toml
[tool.uv]
override-dependencies = ["torch==2.13.0", "torchaudio==2.11.0"]
```

mais um `[[tool.uv.index]]` apontando as rodas cu130. O `uv lock` agora resolve
para `torch 2.13.0+cu130`, e `uv sync --dry-run` não mexe mais no torch.

**Duas dependências não declaradas apareceram na mesma investigação** — o
`uv sync` as arrancava porque nenhum arquivo dizia que elas eram necessárias:

- **`kokoro`**, que o `voice template` usa para sintetizar a referência. Foi assim
  que `narrador-v1`, `narrador-v2` e `citacao-v1` nasceram.
- **`pytest`**, agora em `[dependency-groups] dev`.

### O `sutta` confere a GPU antes do laço

Um venv quebrado derrubava **todos** os endereços da lista, um por um, cada um com
a mesma pilha de CUDA. A checagem foi para antes do laço, junto com as que já
existiam por esse motivo (acervo de slides, venv de música). E ela lança um kernel
de verdade: `cuda.is_available()` devolve `True` com o torch cu124 sobre sm_120 —
a falha só aparece no primeiro `matmul`.

## Calibrações medidas (não re-derivar)

- Whisper **`small` na CPU**: QA custa ~8% do tempo de áudio. `medium` na CPU é inviável (estourou
  10 min para 34 s de áudio). CTranslate2 exige cuBLAS 12 → **não** rodar o ASR na GPU neste venv,
  sob pena de recriar o conflito cu12/cu13.
- **A transcrição do ASR passa pelo mesmo normalizador** antes da comparação. Sem isso o Whisper
  faz a normalização inversa ("mil seiscentos e quarenta e oito" → "1648") e o CER dispara de
  0,0000 para 0,25 — falso positivo sistemático em todo número do livro.
- Limiares: CER ≤ 0,05 aceita; ≤ 0,15 aceita a melhor de N (ruído do ASR); acima, `needs_review`.
  Falha de **duração** nunca é aceita por melhor-de-N — truncamento é defeito objetivo.
- O TTS cru sai com true peak **+0,11 dBTP (clipando)**; o limiter da cadeia não é opcional.
- **Pack pt-BR: CER 0,0125 vs 0,0274 do genérico** (2,2× melhor), mesmo VRAM, +5% de RTF.
  É o padrão; `--no-ptbr-pack` desliga.
- **Camada 2 (LLM)**: `gemma4:12b` gasta ~230 tokens raciocinando antes de responder —
  `num_predict` precisa ser ≥600, senão devolve string vazia com `done_reason: length`.
  Ela roda **antes** das regras (sobre o texto cru), senão os offsets ficam obsoletos.
  A chave do cache inclui a palavra anterior: sem isso "Elizabeth II"→"Segunda"
  contaminaria "Dom Pedro II"→"Segundo". Valor real da camada: concordância de gênero.
