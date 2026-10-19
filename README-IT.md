# SUC — Semantic Universal Compression

**Una rappresentazione compressa dei large language model che preserva il *comportamento* del modello per bit
memorizzato, e che la GPU Apple esegue direttamente dai codici — senza mai ricostruire i pesi FP16.**

**Obiettivo:** far girare modelli grandi su Mac con meno memoria unificata di quella che normalmente richiedono.
Risultati finora: Qwen3-8B in 4.5 GB (Mac da 8–16 GB), Qwen3.8-27B e Qwen3-30B-A3B in ≈ 17 GB (Mac da 24–32 GB), e
modelli MoE da 177–313 B parametri (Qwen3.8-Flash-Next, DeepSeek-V4-Flash, GLM-5.3-Flash, 350–630 GB in BF16) in
74–107 GB su un singolo Mac da 128 GB, con velocità di decode vicine a llama.cpp / ds4. Per comprimere i modelli serve un Mac Studio con chip
Ultra e ≥ 128 GB (dall'M1 Ultra in su: M2, M3 … M5 Ultra); tutti i numeri qui vengono da un M1 Ultra 128 GB, il più
vecchio della serie.

English: [README.md](README.md) · Specifica: [docs/spec-IT.md](docs/spec-IT.md) · Tutte le misure:
[docs/compression_report-IT.md](docs/compression_report-IT.md) · Runtime C++: [cpp/README-IT.md](cpp/README-IT.md) · App macOS:
[SUCStudio/](SUCStudio/README.md) | Modelli LLM: [https://huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz)

## Risultati (Mac Studio M1 Ultra, batch 1, stato al 19 ottobre 2026)

Ogni numero viene da un JSON di risultato (`experiments/results/`, conservati nel repository di lavoro privato e citati per nome file) elencato in
[docs/compression_report-IT.md](docs/compression_report-IT.md) §1 e nelle [verifiche del runtime](docs/optimization/FINAL_QUALIFICATION_1.4.7.md). I protocolli sono indicati nelle tabelle; le celle invariate conservano le misure precedenti. Qwen3-0.6B è misurato
su un portatile Apple M3 (16 GB). Separatore decimale: punto.

### Tabella 1 — Dimensione e qualità

Il file SUC1 scelto per ogni modello. Originale = dimensione equivalente BF16 del modello a precisione piena rispetto
al quale si misura la qualità. KL / top-1 = divergenza / accordo del runtime C++ rispetto alla reference BF16 originale su
16 × 512 token di validazione (8176 posizioni valutate); PPL = misura esistente sul test WikiText-2.

| Modello | Tipo | Parametri | Originale GB | SUC1 GB | bpp | Riduzione | Memoria in esecuzione GB | KL | top-1 | PPL originale → SUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | denso | 0.6 B | 1.50 | 0.32 | 4.26 | 4.7× | 0.42 | 0.092 | 0.853 | 20.97 → 22.83 |
| Qwen3-8B | denso | 8.2 B | 16.38 | 4.32 | 4.22 | 3.8× | 4.63 | 0.050 | 0.902 | 9.72 → 9.99 |
| Qwen3.8-27B (side8) | denso | 26.9 B | 53.79 | 16.36 | 4.87 | 3.3× | 16.75 | 0.023 | 0.949 | 7.05 → 7.15 |
| Qwen3-30B-A3B | MoE | 30.5 B | 61.06 | 15.93 | 4.17 | 3.8× | 17.11 | 0.065 | 0.894 | 8.70 → 9.14 |
| Qwen3.8-Flash-Next (side8) | MoE | 176.9 B | 353.89 | 96.32 | 4.35 | 3.7× | 73.82 ¹ | 0.060 | 0.915 | 4.70 → 4.70 |
| DeepSeek-V4-Flash (tcq) | MoE | 284.3 B | 568.75 ² | 96.82 | 2.72 | 5.9× | 102.57 | 0.235 | 0.826 | 4.78 → 5.59 |
| GLM-5.3-Flash (tcq) | MoE | 313.3 B | 626.65 ² | 99.00 | 2.53 | 6.3× | 106.59 | 0.304 | 0.796 | 3.27 → 4.27 |

bpp = bit del file / tutti i parametri, ogni overhead incluso (codebook, scale, residui, embedding, head, metadati).
¹ Le tabelle n-gram a 4 bit (26.9 GB) sono lette dal file riga per riga, non tenute in memoria GPU. ² DeepSeek e GLM
sono pubblicati già a bassa precisione (FP8 + MXFP4: 166.88 GB; FP8: 328.33 GB); rispetto a quei file SUC1 è 1.7× e
3.3× più piccolo. Tutti i file girano interamente in memoria; il file DeepSeek di qualità superiore che legge gli
esperti da SSD è nella tabella 3.

### Tabella 2 — Velocità del runtime C++ di SUC

`suc1-generate`, greedy: prompt brevi con le misure da 256 token; 2k con la mediana di tre prove da 64 token (8B conserva la prova precedente da 256 token); 8k/32k con una prova limitata a 8 token per lunghezza. *Breve* = prompt di 12–13 token; *2k / 32k* = i primi 2.048 / 32.768
token dello stesso testo lungo. TTFT = lettura del prompt + primo token.

| Modello | Decode tok/s (prompt breve) | Prefill tok/s · TTFT, breve | Prefill tok/s · TTFT, 2k | Prefill tok/s, 8k | Prefill tok/s · TTFT, 32k | Decode tok/s a 32k |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B (M3) | 143.9 | — | — | — | — | — |
| Qwen3-8B | 76.3 | 128 · 0.11 s | 304 · 6.8 s | 231 | 106 · 308 s | 42.5 |
| Qwen3.8-27B (side8) | 25.3 | 43 · 0.35 s | 93 · 22.0 s | 83 | 59 · 558 s | 20.6 |
| Qwen3-30B-A3B | 82.4 | 199 · 0.08 s | 367 · 5.6 s | 243 | 101 · 324 s | 37.6 |
| Qwen3.8-Flash-Next (side8) | 28.3 | 55 · 0.27 s | 366 · 5.6 s | 343 | 219 · 150 s | 29.7 |
| DeepSeek-V4-Flash (tcq) | 24.0 | 46 · 0.30 s | 182 · 11.3 s | 173 | 142 · 231 s | 13.7 |
| GLM-5.3-Flash (tcq) | 22.2 | 35 · 0.41 s | 128 · 16.1 s | 113 | 97 · 339 s | 13.2 |

Il prefill a blocchi non è bit-identico al prefill token per token (KL 0.0006–0.03 tra i due, report §1c). Il decode
è bit-identico al runtime Python.

### Tabella 3 — Confronto con altri runtime e formati

Stessa macchina. KL sulle stesse posizioni e lo stesso riferimento della tabella 1. Decode e prefill breve sullo
stesso prompt. *Prefill lungo*: SUC a 32k token sul testo lungo condiviso; ds4 / llama.cpp / LM Studio = mediana sui
prompt della hard suite da 4k–31k token (insieme di prompt diverso: indicativo). *Hard suite*: 120 domande valutate
(matematica, conoscenza, ragionamento, contesto lungo), stessi prompt per ogni sistema; Qwen3.8 side8: chunk 64, verifica del 19 ottobre 2026 ([fonte](docs/optimization/QWEN_CHUNK_QUALIFICATION.md)).

| Modello | Sistema | File GB | KL | Decode tok/s | Prefill tok/s · TTFT, breve | Prefill tok/s · TTFT, 2k | Prefill lungo tok/s | Hard suite /120 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | **SUC1** (M3) | 0.32 | 0.092 | 143.9 | — | — | — | — |
| | GGUF Q4_K_M, LM Studio | 0.48 | 0.097 | 219.8 | 594 · 0.02 s | 7,785 · 0.26 s | — | — |
| | GGUF Q8_0, LM Studio | 0.80 | 0.002 | 197.3 | 615 · 0.02 s | 8,151 · 0.25 s | — | — |
| Qwen3-8B | **SUC1** | 4.32 | 0.050 | 76.3 | 128 · 0.11 s | 304 · 6.8 s | 106 | — |
| | GGUF Q4_K_M, LM Studio | 5.03 | 0.040 | 74.8 | 176 · 0.07 s | 884 · 2.3 s | — | — |
| | GGUF Q8_0, LM Studio | 8.71 | 0.001 | 67.9 | 193 · 0.07 s | 1,007 · 2.0 s | — | — |
| | MLX 4-bit, mlx-lm ³ | 4.61 | 0.092 | 95.8 | 588 (128 tok) | — · 2.7 s | — | — |
| Qwen3.8-27B | **SUC1 side8** | 16.36 | 0.023 | 25.3 | 43 · 0.35 s | 93 · 22.0 s | 59 | 93 |
| | GGUF Q8_0, llama.cpp | 29.05 | — | 20.8 ⁴ | — | — | 252 | 92 |
| | MLX 8-bit, LM Studio | ≈ 28.6 | 0.002 | 19.4 ⁴ | — | — | 213 | 89 |
| | MLX 4-bit, mlx-lm ³ | ≈ 15.2 | 0.045 | 26.0 | 169 (128 tok) | 214 · 9.6 s | — | — |
| Qwen3-30B-A3B | **SUC1** | 15.93 | 0.065 | 82.4 | 199 · 0.08 s | 367 · 5.6 s | 101 | — |
| | GGUF Q4_K_M, LM Studio | 18.56 | 0.039 | 93.6 | 192 · 0.07 s | 1,455 · 1.4 s | — | — |
| | GGUF Q8_0, LM Studio | 32.48 | 0.006 | 79.9 | 146 · 0.09 s | 1,642 · 1.2 s | — | — |
| | MLX 4-bit, mlx-lm ³ | 17.2 | 0.081 | 74.8 | 753 (128 tok) | — · 1.5 s | — | — |
| Qwen3.8-Flash-Next | **SUC1 side8** (chunk 64) | 96.32 | 0.060 | 28.3 | 55 · 0.27 s | 366 · 5.6 s | 219 | 100 |
| | GGUF Q4, ds4 ⁵ | 177.28 | 0.061 | 33.6 | 47 · 0.31 s | — | 558 | 99 |
| DeepSeek-V4-Flash | **SUC1 tcq** | 96.82 | 0.235 | 24.0 | 46 · 0.30 s | 182 · 11.3 s | 142 | 103 |
| | GGUF IQ2_XXS mix, ds4 | 97.59 | 0.296 | 24.5 | 42 · 0.33 s | — | 224 | 106 |
| | SUC1 V3, esperti da SSD ⁶ | 155.05 | 0.036 | 13.1 | 43 · 0.36 s | 28 · 74 s | — | — |
| | GGUF MXFP4, ds4, esperti da SSD | 155.98 | 0.011 | 10.0 | 7 · 1.97 s | — | — | — |
| GLM-5.3-Flash | **SUC1 tcq** | 99.00 | 0.304 | 22.2 | 35 · 0.41 s | 128 · 16.1 s | 97 | 99 |
| | GGUF Q2, ds4, esperti da SSD | 96.51 | 0.456 | 24.3 | 41 · 0.36 s | — | 117 | 93 |

³ Sessione storica `suc perf` (2–8 ottobre): decode a contesto 128, prefill a 128 token, TTFT a 2.048. ⁴ Mediana
della hard suite. ⁵ Il GGUF tiene su disco una tabella n-gram BF16 da 102.4 GB; i pesi in memoria sono 74.8 GB (SUC1:
73.8 GB). ⁶ Tutti gli esperti nel formato MXFP4 nativo: il tetto di qualità. Righe ds4: engine ds4 commit `9139e2a`,
misure del 6–7 ottobre (prefill lungo e hard suite: 11 ottobre); LM Studio: runtime llama.cpp 2.41.0, GGUF quantizzati in locale senza imatrix.

**Come leggere le tabelle.**

- **Qualità per bit.** Sui modelli Qwen3 SUC batte la KL di MLX 4-bit con meno bit (8B 0.049 vs 0.092; 30B-A3B 0.064
  vs 0.081); su 27B side8 dimezza la KL con +0.36 bpp (0.023 a 4.87 vs 0.045 a 4.51) e anche il file base da 4.22 bpp
  la batte (0.031). GGUF Q4_K_M ha KL più bassa di SUC su 8B e 30B-A3B con ≈ 0.7 bpp in più (non su 0.6B). A 2.5–2.7 bpp SUC
  diverge meno dei GGUF community a 2 bit di pari dimensione (DeepSeek 0.235 vs 0.296, GLM 0.307 vs 0.456).
- **Decode** entro il 12 % del runtime GGUF più veloce su 8B, 30B-A3B, DeepSeek e GLM; più veloce dei file a 8 bit su
  Qwen3.8-27B (alla pari con MLX 4-bit); 16 % più lento di ds4 su Qwen3.8-Flash-Next; MLX 4-bit era il 26 % più
  veloce su 8B nella sessione storica.
- **Prefill** è il punto debole: 1.2–1.7× più lento di ds4 e 3–4.5× più lento di llama.cpp / LM Studio sui prompt lunghi (TTFT a 2k: 5.6–6.8 s contro
  1.2–2.3 s sui modelli Qwen3). Qwen3.8-Flash e DeepSeek arrivano a circa il 60 % della velocità di ds4 sui prompt
  lunghi, GLM all'80 %.
- **Risposte.** Sulla hard suite da 120 domande nessuna differenza appaiata è significativa (SUC vs il miglior
  concorrente: da −3 a +6 domande).
- **Memoria.** Ogni file sta nel budget di 111 GB dello Studio e gira interamente in memoria; DeepSeek V3 (155 GB) e i
  GGUF ds4 MXFP4 / GLM Q2 leggono gli esperti da SSD.

Glossario: **bpp** bit per parametro dell'intero file · **KL** divergenza sul token successivo rispetto all'originale
(0 = identico; ≈ 0.05 piccola, > 0.25 grande) · **top-1** quota di posizioni con lo stesso token più probabile ·
**PPL** perplessità (più bassa è meglio; confrontabile solo nello stesso modello) · **TTFT** tempo al primo token ·
**prefill** velocità di lettura del prompt · **decode** velocità di generazione · **MoE** mixture of experts ·
**tcq** livello a pochi bit con codifica a traliccio · **side8** attention / head a 8 bit, VQ a 4 bit altrove.

---

## Indice

1. [L'idea](#1-lidea)
2. [La rappresentazione SUC](#2-la-rappresentazione-suc)
3. [Sensibilità e allocazione dei bit](#3-sensibilità-e-allocazione-dei-bit)
4. [Pipeline e formato file](#4-pipeline-e-formato-file)
5. [Inferenza sui codici](#5-inferenza-sui-codici)
6. [Uso](#6-uso)
7. [SUC Studio (app macOS)](#7-suc-studio-app-macos)
8. [Limiti](#8-limiti)
9. [Struttura del repository](#9-struttura-del-repository)
10. [Riferimenti](#10-riferimenti)

## 1. L'idea

Il decode a batch 1 è limitato dalla banda di memoria: ogni peso attivo viene letto una volta per token, quindi
`token/s ≲ banda / byte per token`, e la capacità di memoria decide quali modelli possono girare. SUC si chiede quanto
del *comportamento* di un modello si può conservare per bit memorizzato, e se la GPU può calcolare direttamente su
quella forma compatta.

Invece di arrotondare ogni peso, un gruppo di $G$ pesi diventa un indice in un codebook appreso di $K$ vettori
(quantizzazione vettoriale): $\log_2 K / G$ bit di indice per peso. La qualità si ottimizza come comportamento (KL sul
token successivo), non come errore sui pesi:

$$\min_R \; \text{Size}(R) \quad \text{s.t.} \quad D_{\text{semantic}}(M, M_R) \le \varepsilon$$

SUC combina componenti noti — codebook vettoriali/additivi (AQLM, VPTQ), rotazioni di incoerenza (QuIP#), errori
pesati sulle attivazioni (GPTQ/AWQ), precisione mista (EXL2, imatrix), codifica a traliccio (QTIP), kernel a lookup
table (FLUTE) — attorno a un obiettivo comportamentale, un budget globale di bit e un formato di esecuzione per GPU
Apple.

## 2. La rappresentazione SUC

`Modello = Codebook + Indici + Residui + Metadati`, scelti per tensore (per stack di esperti nei MoE).

- **Rotazione:** il lato di ingresso di ogni matrice viene ruotato con una trasformata di Hadamard randomizzata
  ($W' = W V^\top$, applicata online alle attivazioni, $O(n \log n)$). Elimina gli outlier (curtosi in eccesso
  peggiore 32.4 → 0.057 su Qwen3-30B-A3B); senza rotazione la KL sale del 41 % su Qwen3-8B. Basta il lato di
  ingresso; incorporare la rotazione nel flusso residuo fallisce su Qwen3 (canale ad attivazione enorme).
- **Codebook vettoriali:** $\hat w_b = s_r\, C[i_b]$, addestrati con k-means++ e distanza pesata sulle attivazioni.
  Il migliore misurato: codebook 2-D ($G=2$, $K=256$, 4 bit, 1 KB) — KL più bassa del 19–27 % rispetto ai multi-codebook
  additivi a parità di bit; $G=4$, $K=256$ per 2 bit.
- **Codebook additivi** ($\sum_m C_m[i_m]$): disponibili come opzione; più lenti e peggiori a parità di bit su 8B.
- **Livello tcq:** codici a traliccio stile QTIP con LDLQ per gli esperti a 2–3 bit di DeepSeek / GLM (8B: KL 0.269
  contro IQ2_XXS 0.518 a ≈ 2.5 bpp). [docs/low_tier_codec.md](docs/low_tier_codec.md).
- **Pass-through nativo:** gli esperti pubblicati in MXFP4 possono restare bit-esatti.
- **Residui:** 0.5 % delle voci scelte per impatto pesato sulle attivazioni $\mathbb{E}[x_j^2] R_{ij}^2$, salvate come
  colonne INT8 dense. Necessari: senza, la KL raddoppia su Qwen3-8B; scelti per magnitudine sono quasi altrettanto
  cattivi.
- **Fine-tuning** di codebook e scale blocco per blocco: opzionale (−8 % KL a 3.2 bpp su 8B); non usato nei file finali.
- **Dimensioni oneste:** i bpp includono indici, codebook, scale, residui, embedding, head e metadati.
- **Codifica entropica** (rANS/Rice): solo per l'archiviazione; risparmia < 1 % dei byte di indice su pesi ruotati,
  mai nel percorso di inferenza.

## 3. Sensibilità e allocazione dei bit

- **Sensibilità:** l'errore locale di layer predice male la KL del modello (correlazione di rango 0.09–0.57); la
  Fisher empirica $\tfrac12 \sum F_j \Delta W_j^2$ ordina i tensori a 0.75–0.85. I proxy sono validati con
  perturbazioni iniettate.
- **Allocazione:** un allocatore greedy compra miglioramenti per Δqualità/Δbyte fino a esaurire il budget (per tensore
  o stack di esperti; embedding/head inclusi). Conviene nei MoE (Qwen3-30B-A3B: KL −20 % a 3.0 bpp rispetto
  all'uniforme a 3.2 bpp) e nei modelli frontier (quali esperti restano nativi / a 3 bit / a 2 bit); sui densi un
  codebook uniforme è altrettanto buono; sotto ≈ 3 bpp sull'8B denso è peggiore dell'uniforme (risultato negativo).
- **Calibrazione:** testo naturale, codice, matematica e istruzioni, disgiunti da ogni set di valutazione. I file pubblicati usano
  il mix v1 (matematica 20 %, GSM8K); le nuove configurazioni usano di default v3 (matematica 17 % + matematica da
  competizione 3 %); una quota maggiore di matematica da competizione (v2)
  ha peggiorato la conoscenza generale su DeepSeek.

## 4. Pipeline e formato file

`suc compress CONFIG.yaml` esegue le fasi riprendibili preflight → calibrate → reference → candidates → plan → encode
→ container → evaluate, **un blocco transformer alla volta** (mai l'intero modello in memoria), dal checkpoint
originale a precisione piena (BF16, o FP8/MXFP4 ufficiale dequantizzato esattamente). I file già quantizzati (GGUF,
MLX 4-bit) sono solo baseline. Ogni esecuzione scrive un JSON con revisione e hash del modello, hash del dataset,
commit git, seed, hash della configurazione e hardware.

**SUC1** è un contenitore mappabile in memoria: header da 64 byte, directory JSON canonica (architettura, tokenizer
fissato da SHA-256, tensori, codebook), payload allineati a 64 byte (codebook, indici bit-packed, scale, residui,
tensori grezzi), footer SHA-256. Nessuna sezione a lunghezza variabile. Specifica: [docs/format.md](docs/format.md).

Cosa viene compresso: proiezioni di attention e MLP e ogni esperto (codec SUC); embedding / LM head in affine a 4 o 8
bit; router, norm e piccoli mixer restano BF16; la KV cache è stato di esecuzione.

## 5. Inferenza sui codici

I kernel non materializzano mai $W$: ogni riga di uscita è $y_r = s_r \sum_g \langle C[i_{r,g}], x_g \rangle$, leggendo
pochi bit di indice e una voce di codebook dalla memoria on-chip per gruppo; la rotazione d'ingresso è una Hadamard
veloce per ingresso condiviso (q|k|v, gate|up), il residuo un piccolo passaggio INT8 denso. Le proiezioni con lo
stesso ingresso girano in un solo lancio; nei MoE girano solo gli esperti scelti dal router con kernel raggruppati.
Le varianti dei kernel sono auto-calibrate per GPU e messe in cache (`~/.cache/suc/kernel_tuning.json`). Ogni kernel è
verificato contro il decoder di riferimento Python.

Runtime: decoder di riferimento Python (oracolo), runtime Python MLX/Metal e il **runtime C++** (`cpp/`): mmap del
file SUC1, kernel Metal generati da quelli Python, grafi per Qwen3 denso/MoE, Qwen3.8 (`qwen3_5`, `qwen4_exp`),
DeepSeek-V4 e GLM-5.3, tokenizer BPE, chat template, campionamento, prefill a blocchi, streaming degli esperti da SSD
e il server compatibile OpenAI `suc1-server`. Il decode è bit-identico al runtime Python.

## 6. Uso

### Installazione e test

Apple Silicon, Python 3.12, [uv](https://docs.astral.sh/uv/); l'ambiente virtuale va su un disco locale.

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.venvs/suc-lm UV_LINK_MODE=copy uv sync --python 3.12
```

```bash
.venv/bin/python -m pytest
```

Compilare il runtime C++ (su un disco locale):

```bash
cpp/build.sh ~/suc-local/cpp-build
```

### Dove trovare il modello

**Download:** i file SUC1 compressi e i relativi tokenizer sono pubblicati su Hugging Face in
[huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz) (caricamento in corso). Si scarica la cartella del modello e si
indicano al runtime o a SUC Studio il suo `model.suc1` e il `tokenizer.json`.

**Oppure si comprime in proprio** dal checkpoint originale con `suc compress`, che scarica l'originale nella cache
Hugging Face (`~/.cache/huggingface/hub`, oppure `$HF_HOME/hub`) e scrive il file in
`experiments/artifacts/compress/<nome config>/container/model.suc1`:

```bash
.venv/bin/suc compress experiments/configs/compress_qwen3_8b_final.yaml
```

| Modello | Config (`experiments/configs/`) | Originale (Hugging Face) | Revisione |
|---|---|---|---|
| Qwen3-0.6B | `compress_qwen3_0.6b_final.yaml` | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de2` |
| Qwen3-8B | `compress_qwen3_8b_final.yaml` | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d` |
| Qwen3.8-27B | `compress_qwen3.8_27b_side8.yaml` | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2` |
| Qwen3-30B-A3B | `compress_qwen3_30b_a3b_final.yaml` | [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `ad44e777` |
| Qwen3.8-Flash-Next | `compress_qwen3.8_flash_next_side8.yaml` | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `de4b8e4d` |
| DeepSeek-V4-Flash | `compress_deepseek_v4_flash_tcq.yaml` | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | `7872f01b` |
| GLM-5.3-Flash | `compress_glm5.3_flash_tcq.yaml` | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) | `eb9eb208` |

Per comprimere i modelli frontier serve un Mac Ultra da ≥ 128 GB (i checkpoint pubblicati da 167–354 GB sono letti un
blocco alla volta da disco). Un `model.suc1` è un unico file autosufficiente: basta copiarlo sul Mac che lo eseguirà.

### Dove trovare il tokenizer

Ogni cartella modello su [huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz) contiene il suo `tokenizer.json`. Lo
stesso file si trova nel repository originale della tabella sopra, alla revisione indicata. Il file SUC1 lo fissa come
`Org/Nome@revisione;tokenizer.json:sha256:…` (si legge con `suc1-inspect model.suc1 --no-verify`); i tool C++ lo
cercano automaticamente nella cache Hugging Face e ne verificano lo SHA-256. Per scaricare nella cache solo il
tokenizer dell'originale (pochi MB):

```bash
hf download Qwen/Qwen3-8B tokenizer.json --revision b968826d9c46dd6066d109eabc6255188de91218
```

`hf` è incluso in `huggingface_hub` (in questo repository: `.venv/bin/hf`); usare la revisione completa di 40 caratteri.

Oppure si passa una copia con `--tokenizer percorso/tokenizer.json` (non verificata: deve coincidere con lo SHA-256 del
tokenizer id). I chat template sono integrati nel runtime, quindi `tokenizer.json` è l'unico file necessario.

### Esecuzione

```bash
~/suc-local/cpp-build/suc1-generate model.suc1 --chat --prompt "Spiega la quantizzazione vettoriale." --steps 256
```

```bash
~/suc-local/cpp-build/suc1-server model.suc1 --port 8080 --max-context 32768
```

`suc1-server` espone l'API OpenAI (`/v1/chat/completions`, `/v1/completions`, streaming, `/v1/models`, `/health`;
batch 1); qualunque client OpenAI funziona con `base_url="http://127.0.0.1:8080/v1"`. Streaming degli esperti per file
più grandi della memoria: `--expert-budget 95G --hot-experts auto`. I file grandi richiedono che il proprietario alzi
una volta il limite di memoria GPU (`sudo sysctl iogpu.wired_limit_mb=116000` su 128 GB). Tutte le opzioni:
[cpp/README-IT.md](cpp/README-IT.md).

Valutazione delle risposte e benchmark di velocità di un file SUC1 (`suc eval`, `suc perf`):
[docs/usage_eval_perf.md](docs/usage_eval_perf.md).

## 7. SUC Studio (app macOS)

[SUC Studio](SUCStudio/README.md) è un'app nativa per macOS (Apple Silicon, macOS 15+) costruita sullo stesso motore
C++: client di chat e server locale per i modelli SUC1. Installer: `SUCStudio/dist/SUC-Studio-<versione>-arm64.pkg`
(build locale, firma ad-hoc, senza Developer ID); guida: [SUCStudio/USAGE.md](SUCStudio/USAGE.md); storico delle
versioni: `SUCStudio/dist/CHANGELOG.md`.

- **Modelli:** in Settings si sceglie il file `.suc1` e il suo `tokenizer.json` (preselezionato se si trova accanto al
  modello). Piano di memoria automatico: il modello gira interamente in memoria quando sta nel budget di memoria GPU
  (configurabile), altrimenti gli esperti MoE vengono letti da SSD; i modelli troppo grandi vengono
  rifiutati prima del caricamento.
- **Backend:** il motore locale, un SUC Studio / `suc1-server` remoto, oppure **qualsiasi motore di inferenza con API
  compatibili OpenAI** (ds4, LM Studio, llama.cpp, router LLM, …), con scelta del modello, API key, streaming e
  interruzione. Profili di telemetria per backend mostrano le velocità reali di prefill / decode quando il server le
  fornisce e segnalano esplicitamente le stime.
- **Server integrato:** espone il modello caricato come API compatibile OpenAI, con API key e accesso LAN opzionale;
  stato nella barra dei menu e telemetria del motore.
- **Chat:** sessioni con export / import, rendering Markdown, reasoning separato dalla risposta, modifica e
  rigenerazione, composer ridimensionabile, editor di codice, link aperti nel browser.
- **Allegati:** file di testo con qualsiasi estensione, Word DOC/DOCX, OpenDocument, RTF, PDF testuali; immagini per i
  backend vision. Il testo estratto resta nella sessione.
- **Funzioni agente:** system prompt globale, skill selezionabili, server MCP (stdio locali e HTTP remoti, credenziali
  nel Portachiavi), directory di lavoro per sessione con strumenti per file e comandi, approvazioni manuali o
  automatiche dentro la chat, terminale interattivo, subtask e subagent con contesti separati, hook configurabili,
  artifact persistenti per i file prodotti, registro delle operazioni e limite configurabile di azioni consecutive.
- **Estensioni e routine:** plugin installabili (cartelle locali o marketplace) e routine pianificate con prompt,
  frequenza e storico.
- **Aspetto e sistema:** font e dimensioni indipendenti per interfaccia e chat, avvio al login, Mac mantenuto attivo.

## 8. Limiti

- Con perdita: i pesi cambiano; nessuna affermazione di "lossless" o "stessa intelligenza" senza prove sui task.
- Il prefill è 1.2–1.7× più lento di ds4 e 3–4.5× più lento di llama.cpp / LM Studio sui prompt lunghi; il decode a batch > 1 non esiste nel C++.
- A ≈ 4.4 bpp il GGUF MXFP4 di ds4 batte SUC V3 su DeepSeek in KL (0.011 vs 0.036); GGUF Q4_K_M ha KL più bassa di SUC
  su 8B e 30B-A3B con ≈ 0.7 bpp in più (non su 0.6B). I risultati pubblicati di QuIP#/QTIP/PV-Tuning a ≈ 2 bpw su Qwen3-8B sono migliori di SUC.
- Nessuna esecuzione AQLM/QuIP# con lo stesso protocollo; un solo seed; una sola macchina.
- La codifica entropica riduce solo lo spazio su disco.

## 9. Struttura del repository

```text
suc/          io (safetensors lazy, SUC1), models (adapter, grafi MLX), quant (codec, rotazioni, residui, tcq),
              allocation, kernels (Metal + autotune), runtime, pipeline (`suc compress`), evaluation, cli
cpp/          runtime C++: lettore/decoder SUC1, kernel Metal, grafi dei modelli, tokenizer, suc1-generate, suc1-server
SUCStudio/    app macOS sul motore C++
experiments/  configs/ (YAML) e results/ (JSON, uno per esecuzione)
docs/         specifica, formato, report di compressione, confronti con ds4, note di architettura
scripts/      code di misura e generatori dei report
tests/        test unitari (modelli piccoli casuali) e test su modelli reali (`-m model`)
```

## 10. Riferimenti

AQLM (arXiv 2401.06118) · QuIP (2307.13304) · QuIP# (2402.04396) · QTIP (2406.11235) · PV-Tuning (2405.14852) ·
FLUTE (2407.10960) · CodeGEMM (2512.17970) · [llama.cpp](https://github.com/ggml-org/llama.cpp) ·
[MLX](https://ml-explore.github.io/mlx/) · T. M. Cover, "Universal Portfolios", 1991.
