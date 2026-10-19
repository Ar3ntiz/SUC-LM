# Rapporto di compressione SUC — da Qwen3-0.6B a GLM-5.3-Flash

Versione inglese: [compression_report.md](compression_report.md). Le tabelle generate dagli script (§1b, §1c, §5b, §5c, §7) sono copie di quelle inglesi al 15 ottobre 2026; in caso di differenze fa fede la versione inglese.

Ogni modello compresso del progetto a confronto con il proprio **modello originale a piena precisione** e con altri runtime:
dimensione, memoria, velocità (TTFT, prefill, decode) e qualità. Ogni numero proviene da un JSON di risultato in
`experiments/results/` (citato come `R:<file>`). "—" = non misurato. Stato: 19 ottobre 2026.

**Convenzioni.** *Dimensione originale* = byte equivalenti BF16 di ogni tensore del checkpoint (il modello rispetto a cui si misurano PPL e KL;
per DeepSeek / GLM, pubblicati in FP8 / MXFP4, la dequantizzazione esatta). *bpp* = bit del file / tutti i parametri,
ogni overhead incluso. *Memoria* = file mappato + footprint del processo del runtime C++. *Velocità* = batch 1, C++
`suc1-generate`, greedy, protocolli specificati in §1; prove storiche in §1b–§1c. *Qualità* = KL / top-1 rispetto ai logit di riferimento su 16 × 512
posizioni di validazione tenute da parte e perplessità (PPL) sul test di WikiText-2 (tutte le finestre), con KL/top-1 C++ aggiornati in §1 e PPL storica di `suc compress`. Macchina:
Mac Studio M1 Ultra 128 GB (budget di 111 GB per il modello), tranne Qwen3-0.6B (Apple M3, 16 GB).

## 1. Sintesi

### Dimensione e qualità

Il file SUC1 scelto per ogni modello. Originale = dimensione equivalente BF16 del modello a piena precisione rispetto a cui si misura
la qualità. KL / top-1 = divergenza / concordanza sul token successivo rispetto all'originale su 16 × 512 posizioni tenute da parte; PPL =
test di WikiText-2.

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
¹ Le tabelle n-gram a 4 bit (26.9 GB) vengono lette dal file riga per riga, non tenute in memoria GPU. ² DeepSeek e GLM sono
pubblicati già a bassa precisione (FP8 + MXFP4: 166.88 GB; FP8: 328.33 GB); rispetto a quei file SUC1 è 1.7× e
3.3× più piccolo. Tutti i file girano interamente in memoria; il file DeepSeek di qualità superiore che legge gli expert da SSD (streaming) è nella
terza tabella e in §4.

### Velocità del runtime C++ SUC

`suc1-generate`, greedy: prompt brevi con le misure da 256 token; 2k con la mediana di tre prove da 64 token (8B conserva la prova precedente da 256 token); 8k/32k con una prova limitata a 8 token per lunghezza. *Breve* = un prompt di 12–13 token; *2k / 32k* = i primi 2,048 / 32,768
token dello stesso testo lungo. TTFT = elaborazione del prompt + primo token.

| Modello | Decode tok/s (prompt breve) | Prefill tok/s · TTFT, breve | Prefill tok/s · TTFT, 2k | Prefill tok/s, 8k | Prefill tok/s · TTFT, 32k | Decode tok/s a 32k |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B (M3) | 143.9 | — | — | — | — | — |
| Qwen3-8B | 76.3 | 128 · 0.11 s | 304 · 6.8 s | 231 | 106 · 308 s | 42.5 |
| Qwen3.8-27B (side8) | 25.3 | 43 · 0.35 s | 93 · 22.0 s | 83 | 59 · 558 s | 20.6 |
| Qwen3-30B-A3B | 82.4 | 199 · 0.08 s | 367 · 5.6 s | 243 | 101 · 324 s | 37.6 |
| Qwen3.8-Flash-Next (side8) | 28.3 | 55 · 0.27 s | 366 · 5.6 s | 343 | 219 · 150 s | 29.7 |
| DeepSeek-V4-Flash (tcq) | 24.0 | 46 · 0.30 s | 182 · 11.3 s | 173 | 142 · 231 s | 13.7 |
| GLM-5.3-Flash (tcq) | 22.2 | 35 · 0.41 s | 128 · 16.1 s | 113 | 97 · 339 s | 13.2 |

Il prefill del prompt a batch non è bit-identico al prefill token per token (KL 0.0006–0.03 tra i due,
§1c). Il decode è bit-identico al runtime Python.

### A confronto con altri runtime e formati

Stessa macchina. KL sulle stesse posizioni e con lo stesso riferimento della tabella 1. Decode e prefill con prompt breve sullo stesso prompt.
*Prefill lungo*: SUC a 32k token sul testo lungo condiviso; ds4 / llama.cpp / LM Studio = mediana sui prompt dell'hard suite
da 4k–31k token (un insieme di prompt diverso: indicativo). *Hard suite*: 120 domande valutate (matematica, conoscenze,
ragionamento, contesto lungo), stessi prompt per ogni sistema; Qwen3.8 side8: chunk 64, verifica del 19 ottobre 2026 ([fonte](optimization/QWEN_CHUNK_QUALIFICATION.md)).

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
| | SUC1 V3, expert da SSD ⁶ | 155.05 | 0.036 | 13.1 | 43 · 0.36 s | 28 · 74 s | — | — |
| | GGUF MXFP4, ds4, expert da SSD | 155.98 | 0.011 | 10.0 | 7 · 1.97 s | — | — | — |
| GLM-5.3-Flash | **SUC1 tcq** | 99.00 | 0.304 | 22.2 | 35 · 0.41 s | 128 · 16.1 s | 97 | 99 |
| | GGUF Q2, ds4, expert da SSD | 96.51 | 0.456 | 24.3 | 41 · 0.36 s | — | 117 | 93 |

³ Sessione storica di `suc perf` (2–8 ottobre): decode con contesto di 128 token, prefill a 128 token, TTFT a 2,048.
⁴ Mediana dell'hard suite. ⁵ Il GGUF tiene su disco una tabella n-gram BF16 da 102.4 GB; i suoi pesi in memoria sono 74.8 GB (SUC1:
73.8 GB). ⁶ Ogni expert instradato mantenuto nel suo formato nativo MXFP4: il tetto di qualità. Righe ds4: commit del motore ds4
`9139e2a`, esecuzioni del 6–7 ottobre (prefill lungo e hard suite: 11 ottobre); LM Studio: runtime llama.cpp 2.41.0, GGUF quantizzati localmente senza imatrix.

**Risultati.**

1. **Dimensione.** Da 3.3 a 4.7× più piccolo dell'originale a 4.2–4.9 bpp (famiglia Qwen), 5.9–6.3× a 2.5–2.7 bpp (DeepSeek, GLM;
   1.7× / 3.3× rispetto ai loro file pubblicati FP8 / MXFP4). Ogni file scelto gira interamente in memoria entro 111 GB.
2. **Qualità per bit.** SUC batte la KL di MLX / RTN 4-bit con meno bit ovunque quella baseline potesse girare (0.6B 0.092 vs
   0.190, 8B 0.049 vs 0.092, 30B-A3B 0.064 vs 0.081, Qwen3.8-Flash 0.064 vs 0.202); su 27B side8 dimezza la KL con
   +0.36 bpp (0.023 a 4.87 vs 0.045 a 4.51) e anche il file base da 4.22 bpp la batte (0.031). GGUF Q4_K_M ha KL più
   bassa di SUC su 8B e 30B-A3B con ≈ 0.7 bpp in più (non su 0.6B; §6); a 2.5–2.7 bpp SUC tcq batte i GGUF a 2 bit della community della stessa dimensione (§5).
3. **Decode** entro il 12 % del runtime GGUF più veloce, tranne Qwen3.8-Flash (−16 % vs ds4); 1.5–2.5× più veloce del
   BF16 originale dove questo entra in memoria (§2).
4. **Prefill / TTFT** è il punto debole: 1.2–1.7× più lento di ds4 e 3–4.5× più lento di llama.cpp / LM Studio
   sui prompt lunghi; TTFT 5.6–6.8 s a 2k sui modelli
   Qwen3 contro 1.2–2.3 s di LM Studio. Il prefill a batch non è bit-identico al prefill per token (KL 0.0006–0.03, §1c).
5. **Risposte.** Hard suite da 120 casi: SUC rispetto al miglior concorrente da −3 a +6 domande, nessuna differenza appaiata significativa
   (§7). La core suite da 30 casi è satura (29–30/30 per ogni file da 8B in su).

Fonti. Qualità: R:`20261002T202145Z_compress_qwen3_0.6b_final`, R:`20261002T191301Z_compress_qwen3_8b_final`,
R:`20261007T230047Z_compress_qwen3.8_27b_side8`, R:`20261002T212215Z_compress_qwen3_30b_a3b_final`,
R:`20261007T140252Z_compress_qwen3.8_flash_next_side8`, R:`20261009T181636Z_compress_deepseek_v4_flash_tcq`,
R:`20261010T124951Z_compress_glm5.3_flash_tcq`. Velocità: le righe di §1b / §1c; Qwen3-0.6B
R:`20261002T205833Z_cpp_check_qwen3_0.6b_final_m3`. Concorrenti: §5 (ds4), §6 (LM Studio), §2 (MLX); velocità dei concorrenti sui prompt lunghi
e conteggi dell'hard suite: R:`20261011T011939Z_gguf_eval_hard20_ds_iq2`,
R:`20261011T075042Z_gguf_eval_hard20_glm_q2`, R:`20261011T132617Z_gguf_eval_hard20_q38_q4`,
R:`20261011T213929Z_gguf_eval_hard20_llamacpp_q38_27b_q8_0`, R:`20261012T025455Z_gguf_eval_hard20_lms_q38_27b_8bit`
e i file SUC di §7 (prefill mediano sui prompt ≥ 4k token; decode = mediana di (token − 1) / (tempo − prefill)).

### 1b. Decode C++, build attuale

`scripts/frontier_measure.py generate`: `suc1-generate` con i suoi default al commit indicato, batch 1, greedy,
il prompt frontier (colonna prefill: i suoi token; GLM con una virgola finale), 256 token, singola esecuzione, decode =
mediana dei passi 2…256; righe in streaming con i
`--expert-budget` / `--hot-experts auto` delle loro misure precedenti. La colonna decode di §1 è presa da queste righe
(Qwen3.8-27B: la riga breve di §1c).

| Modello | decode ms/tok | tok/s | GPU ms/tok | prefill s | hit rate degli expert | misura precedente ms/tok | Δ | risultato (commit) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (final) | 13.10 | 76.3 | 13.04 | 0.10 (12 tok) | — | 12.70 (R:`20261010T230512Z_gen_cpp_qwen3_8b_final`) | +3.1 % | R:`20261012T054020Z_gen_cpp_qwen3_8b_final` (258b03f) |
| Qwen3-30B-A3B (final, uniforme) | 12.13 | 82.4 | 12.09 | 0.06 (12 tok) | — | 11.87 (R:`20261010T231250Z_gen_cpp_qwen3_30b_a3b_final`) | +2.2 % | R:`20261012T055423Z_gen_cpp_qwen3_30b_a3b_final` (f04d199) |
| Qwen3.8-Flash-Next side8 (ngram4) | 35.30 | 28.3 | 26.48 | 0.23 (12 tok) | — | 34.40 (R:`20261010T232151Z_gen_cpp_qwen3.8_flash_next_side8_ngram4`) | +2.6 % | R:`20261012T050109Z_gen_cpp_qwen3.8_flash_next_side8_ngram4` (db83734) |
| Qwen3.8-Flash-Next iso (ngram4) | 35.31 | 28.3 | 26.64 | 0.17 (12 tok) | — | 35.18 (R:`20261007T154126Z_gen_cpp_qwen3.8_flash_next_iso_ngram4`) | +0.3 % | R:`20261008T165246Z_gen_cpp_qwen3.8_flash_next_iso_ngram4` (1a2dc24) |
| DeepSeek-V4-Flash V1 (native_max, in memoria) | 39.19 | 25.5 | 36.97 | 0.27 (11 tok) | — | 39.05 (R:`20261007T154239Z_gen_cpp_deepseek_v4_flash_native_max`) | +0.4 % | R:`20261008T165717Z_gen_cpp_deepseek_v4_flash_native_max` (1a2dc24) |
| DeepSeek-V4-Flash V3 (native_stream, expert da SSD) | 76.09 | 13.1 | 35.15 | 0.25 (11 tok) | 0.955 | 78.92 (R:`20261007T154331Z_gen_cpp_deepseek_v4_flash_native_stream`) | -3.6 % | R:`20261008T171204Z_gen_cpp_deepseek_v4_flash_native_stream` (b6fb206) |
| GLM-5.3-Flash (expert da SSD) | 109.59 | 9.1 | 45.36 | 2.12 (13 tok) | 0.911 | 112.97 (R:`20261007T154440Z_gen_cpp_glm5.3_flash_streamed`) | -3.0 % | R:`20261008T180714Z_gen_cpp_glm5.3_flash_streamed` (09f58b8) |
| DeepSeek-V4-Flash tcq (in memoria) | 41.61 | 24.0 | 38.60 | 0.25 (11 tok) | — | 41.59 (R:`20261012T033424Z_gen_cpp_deepseek_v4_flash_tcq`) | +0.1 % | R:`20261012T040303Z_gen_cpp_deepseek_v4_flash_tcq` (a7d2b0f) |
| GLM-5.3-Flash tcq (in memoria) | 45.08 | 22.2 | 41.62 | 0.35 (12 tok) | — | 44.82 (R:`20261011T001020Z_gen_cpp_glm5.3_flash_tcq`) | +0.6 % | R:`20261012T043057Z_gen_cpp_glm5.3_flash_tcq` (f6c001c) |
| DeepSeek-V4-Flash tcq, calib mix v2 (in memoria) | 41.77 | 23.9 | 38.76 | 0.25 (11 tok) | — | — | — | R:`20261012T233316Z_gen_cpp_deepseek_v4_flash_tcq_mathcal` (efce37c) |

### 1c. Prefill e TTFT C++, build attuale (prefill a batch)

`scripts/studio_prefill_measure.sh` (`scripts/frontier_measure.py generate`), batch 1, greedy, 256 token,
singola esecuzione, Mac Studio M1 Ultra, ogni esecuzione dopo `scripts/studio_wait_idle.sh`. Prompt condivisi con le esecuzioni LM Studio
/ llama.cpp (`suc.evaluation.speed_prompts`): breve = la frase frontier (testo grezzo), lungo = i primi
2,048 id della tokenizzazione grezza di `experiments/perf/promessi_sposi_head.txt` con il tokenizer del modello stesso.
Prefill tok/s = token del prompt / tempo reale di prefill (compilazioni delle pipeline escluse); TTFT = prefill + un passo
di decode; decode = mediana dei passi 2…256 dopo quel prompt. Prefill a batch (default, strato per strato sui chunk;
cpp/README-IT.md): grafo Qwen3 / Qwen3-MoE a chunk di 64 token, proiezioni a batch n (lookup simdgroup-matrix).
qwen4_exp / qwen3_5 / deepseek_v4 / glm5_next (dal 2026-10-10, decisione del proprietario: piccole differenze di arrotondamento
ammesse, misurate): GEMM simdgroup-matrix sulle righe del chunk (lookup "mma" SUC, kernel GEMM affine / densi / tcq),
expert instradati in tile MMA da 8-16 righe, compressori e attention di DeepSeek a batch sul chunk;
dal 2026-10-11 (docs/prefill_vs_ds4.md) anche il lightning indexer di DeepSeek oltre i 2,048 token, l'attention QSA di Qwen3.8
e ogni lancio locale alla riga a batch sul chunk, la DSA di GLM solo sui pool selezionati;
chunk di 4,096 (DeepSeek, GLM; 1,024 oltre i prompt da 8,192 token) / 1,024 (qwen4_exp, qwen3_5) token, al massimo
il prompt (1,024 / 512 prima del 2026-10-12). Le righe di un commit più vecchio sono state misurate con
il precedente prefill a batch esatto per riga (ora `--prefill exact`). Prefill per token = `--prefill token` (passi di decode);
bit-identico = gli sha256 dei logit dell'ultimo token del prompt (e, dove registrati, degli stream) coincidono tra i due; l'entità
della differenza è nella seconda tabella.

| Modello | breve: tok prompt | prefill tok/s | TTFT s | decode ms/tok | 2k: tok prompt | prefill tok/s | TTFT s | decode ms/tok | 2k prefill per token tok/s | accelerazione batch | bit-identico | risultati (commit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (final) | 13 | 128.4 | 0.11 | 13.04 | 2048 | 304.2 | 6.75 | 13.72 | 75.9 | 4.0x | **no** | R:`20261012T054125Z_gen_cpp_qwen3_8b_final_short`, R:`20261012T054237Z_gen_cpp_qwen3_8b_final_long2k`, R:`20261010T230902Z_gen_cpp_qwen3_8b_final_long2k_token` (258b03f) |
| Qwen3-30B-A3B (final, uniforme) | 13 | 198.5 | 0.08 | 12.20 | 2048 | 367.4 | 5.59 | 13.81 | 66.9 | 5.5x | **no** | R:`20261012T055530Z_gen_cpp_qwen3_30b_a3b_final_short`, R:`20261012T055643Z_gen_cpp_qwen3_30b_a3b_final_long2k`, R:`20261010T231657Z_gen_cpp_qwen3_30b_a3b_final_long2k_token` (f04d199) |
| Qwen3.8-Flash-Next side8 (ngram4) | 13 | 54.7 | 0.27 | 31.37 | 2048 | 341.2 | 6.04 | 33.82 | 39.9 | 8.6x | **no** | R:`20261012T050239Z_gen_cpp_qwen3.8_flash_next_side8_ngram4_short`, R:`20261012T050422Z_gen_cpp_qwen3.8_flash_next_side8_ngram4_long2k`, R:`20261010T232756Z_gen_cpp_qwen3.8_flash_next_side8_ngram4_long2k_token` (db83734) |
| DeepSeek-V4-Flash V1 (native_max, in memoria) | 12 | 41.4 | 0.33 | 38.66 | 2048 | 40.6 | 50.44 | 43.87 | 27.0 | 1.5x | sì | R:`20261008T170035Z_gen_cpp_deepseek_v4_flash_native_max_short`, R:`20261008T170437Z_gen_cpp_deepseek_v4_flash_native_max_long2k`, R:`20261008T170912Z_gen_cpp_deepseek_v4_flash_native_max_long2k_token` (1a2dc24) |
| DeepSeek-V4-Flash V3 (native_stream, expert da SSD) | 12 | 42.9 | 0.36 | 77.10 | 2048 | 27.7 | 73.93 | 70.65 | 14.2 | 1.9x | sì | R:`20261008T171451Z_gen_cpp_deepseek_v4_flash_native_stream_short`, R:`20261008T173852Z_gen_cpp_deepseek_v4_flash_native_stream_long2k`, R:`20261008T180400Z_gen_cpp_deepseek_v4_flash_native_stream_long2k_token` (db47247) |
| GLM-5.3-Flash (expert da SSD) | 13 | 6.6 | 2.08 | 117.34 | 2048 | 16.3 | 125.84 | 97.53 | 10.2 | 1.6x | sì | R:`20261009T055050Z_gen_cpp_glm5.3_flash_streamed_short`, R:`20261009T074955Z_gen_cpp_glm5.3_flash_streamed_long2k`, R:`20261009T075605Z_gen_cpp_glm5.3_flash_streamed_long2k_token` (28b3ddf) |
| DeepSeek-V4-Flash tcq (in memoria) | 12 | 45.7 | 0.30 | 41.99 | 2048 | 171.4 | 11.99 | 46.74 | 25.7 | 6.7x | **no** | R:`20261012T040518Z_gen_cpp_deepseek_v4_flash_tcq_short`, R:`20261012T040803Z_gen_cpp_deepseek_v4_flash_tcq_long2k`, R:`20261010T235732Z_gen_cpp_deepseek_v4_flash_tcq_long2k_token` (a7d2b0f) |
| GLM-5.3-Flash tcq (in memoria) | 13 | 35.2 | 0.41 | 45.20 | 2048 | 116.3 | 17.66 | 49.50 | 22.4 | 5.2x | **no** | R:`20261012T043319Z_gen_cpp_glm5.3_flash_tcq_short`, R:`20261012T043619Z_gen_cpp_glm5.3_flash_tcq_long2k`, R:`20261011T001942Z_gen_cpp_glm5.3_flash_tcq_long2k_token` (f6c001c) |
| DeepSeek-V4-Flash tcq, calib mix v2 (in memoria) | 12 | 46.1 | 0.30 | 41.78 | 2048 | 170.3 | 12.07 | 46.68 | — | — | — | R:`20261012T233528Z_gen_cpp_deepseek_v4_flash_tcq_mathcal_short`, R:`20261012T233815Z_gen_cpp_deepseek_v4_flash_tcq_mathcal_long2k` (efce37c) |
| Qwen3.8-27B side8 (denso) | 13 | 42.6 | 0.35 | 39.53 | 2048 | 93.0 | 22.05 | 40.55 | 28.4 | 3.3x | **no** | R:`20261012T051802Z_gen_cpp_qwen3.8_27b_side8_short`, R:`20261012T051950Z_gen_cpp_qwen3.8_27b_side8_long2k`, R:`20261010T234016Z_gen_cpp_qwen3.8_27b_side8_long2k_token` (f63b320) |
| Qwen3.8-27B base (denso) | 13 | 65.6 | 0.24 | 39.12 | 2048 | 68.4 | 29.98 | 40.32 | 28.9 | 2.4x | sì | R:`20261009T182408Z_gen_cpp_qwen3.8_27b_base_short`, R:`20261009T195901Z_gen_cpp_qwen3.8_27b_base_long2k`, R:`20261009T200225Z_gen_cpp_qwen3.8_27b_base_long2k_token` (149c389) |

Prompt lunghi (`scripts/studio_prefill_final.sh`): i primi 2,048 / 8,192 / 32,768 id dello stesso prompt
lungo, prefill a batch di default, 256 token greedy dopo di esso; prefill tok/s · TTFT s · decode ms/tok
(decode a quel contesto). Ripartizione del tempo di prefill e confronto con ds4: docs/prefill_vs_ds4.md.

| Modello | 2k prefill tok/s | 2k TTFT s | 2k decode ms/tok | 8k prefill tok/s | 8k TTFT s | 8k decode ms/tok | 32k prefill tok/s | 32k TTFT s | 32k decode ms/tok | risultati (commit) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (final) | 304.2 | 6.75 | 13.72 | 231.3 | 35.43 | 16.16 | 106.1 | 308.94 | 23.51 | R:`20261012T054418Z_gen_cpp_qwen3_8b_final_long8k`, R:`20261012T055035Z_gen_cpp_qwen3_8b_final_long32k` (258b03f) |
| Qwen3-30B-A3B (final, uniforme) | 367.4 | 5.59 | 13.81 | 243.5 | 33.67 | 18.72 | 101.0 | 324.36 | 26.68 | R:`20261012T055826Z_gen_cpp_qwen3_30b_a3b_final_long8k`, R:`20261012T060501Z_gen_cpp_qwen3_30b_a3b_final_long32k` (f04d199) |
| Qwen3.8-Flash-Next side8 (ngram4) | 341.2 | 6.04 | 33.82 | 323.4 | 25.37 | 33.04 | 322.9 | 101.52 | 34.20 | R:`20261012T050627Z_gen_cpp_qwen3.8_flash_next_side8_ngram4_long8k`, R:`20261012T050949Z_gen_cpp_qwen3.8_flash_next_side8_ngram4_long32k` (db83734) |
| DeepSeek-V4-Flash tcq (in memoria) | 171.4 | 11.99 | 46.74 | 163.2 | 50.26 | 54.26 | 137.8 | 237.90 | 73.12 | R:`20261012T041139Z_gen_cpp_deepseek_v4_flash_tcq_long8k`, R:`20261012T041806Z_gen_cpp_deepseek_v4_flash_tcq_long32k` (a7d2b0f) |
| GLM-5.3-Flash tcq (in memoria) | 116.3 | 17.66 | 49.50 | 105.1 | 78.02 | 55.12 | 93.8 | 349.47 | 76.12 | R:`20261012T044027Z_gen_cpp_glm5.3_flash_tcq_long8k`, R:`20261012T044853Z_gen_cpp_glm5.3_flash_tcq_long32k` (f6c001c) |
| DeepSeek-V4-Flash tcq, calib mix v2 (in memoria) | 170.3 | 12.07 | 46.68 | 161.3 | 50.86 | 54.40 | 137.2 | 238.88 | 72.94 | R:`20261012T234153Z_gen_cpp_deepseek_v4_flash_tcq_mathcal_long8k`, R:`20261012T234820Z_gen_cpp_deepseek_v4_flash_tcq_mathcal_long32k` (efce37c) |
| Qwen3.8-27B side8 (denso) | 93.0 | 22.05 | 40.55 | 84.8 | 96.59 | 42.49 | 58.1 | 563.82 | 49.03 | R:`20261012T052251Z_gen_cpp_qwen3.8_27b_side8_long8k`, R:`20261012T053341Z_gen_cpp_qwen3.8_27b_side8_long32k` (d71f978) |

Prefill a batch vs per token (`scripts/prefill_batch_vs_token.py`, il prompt da 2,048 token): logit
di ogni posizione del prompt e di 32 passi di continuazione in teacher forcing sui token greedy dell'esecuzione per token (esecuzione
a batch con `--logits-out`, cioè ogni riga passa per la LM head), KL(token ‖ batch) media / max e concordanza
top-1 (discordanze); l'ultimo token del prompt come lo calcolano le esecuzioni di produzione; il primo passo in cui
le continuazioni greedy libere (256 token) divergono. Origini della differenza (cpp/README-IT.md): grafo Qwen3:
il lookup simdgroup-matrix dei tile di batch, la suddivisione della riduzione del residuo strutturato per tile multi-riga
e l'attention a passata singola delle righe di batch; grafi frontier: l'ordine di accumulazione dei GEMM di ogni
proiezione e tile di expert (float32, un solo arrotondamento), l'attention a chunk di DeepSeek riproduce gli arrotondamenti
del passo di decode.

| Modello | tok prompt | KL prompt media / max | top-1 prompt | KL continuazione media / max | top-1 continuazione | KL ultimo token | top-1 ultimo token | prima divergenza greedy (token) | risultato |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (final) | 2048 | 9.88e-04 / 1.75e-02 | 0.9844 (32) | 7.68e-04 / 3.10e-03 | 0.9677 (1) | 9.94e-04 | 1 | 8 (256) | R:`20261012T055307Z_prefill_batch_vs_token_qwen3_8b_final` |
| Qwen3-30B-A3B (final, uniforme) | 2048 | 3.08e-03 / 7.44e-02 | 0.9712 (59) | 3.46e-03 / 1.04e-02 | 0.9355 (2) | 1.56e-03 | 1 | 17 (256) | R:`20261012T060743Z_prefill_batch_vs_token_qwen3_30b_a3b_final` |
| Qwen3.8-Flash-Next side8 (ngram4) | 2048 | 3.08e-02 / 1.20e+00 | 0.9150 (174) | 3.87e-02 / 2.03e-01 | 0.8065 (6) | 3.84e-02 | 0 | 0 (256) | R:`20261012T051524Z_prefill_batch_vs_token_qwen3.8_flash_next_side8_ngram4` |
| DeepSeek-V4-Flash tcq (in memoria) | 2048 | 2.58e-02 / 1.17e+00 | 0.9438 (115) | 1.83e-02 / 1.40e-01 | 0.9355 (2) | 5.81e-04 | 1 | 22 (256) | R:`20261012T042732Z_prefill_batch_vs_token_deepseek_v4_flash_tcq` |
| GLM-5.3-Flash tcq (in memoria) | 2048 | 1.16e-02 / 4.06e-01 | 0.9380 (127) | 7.12e-03 / 5.90e-02 | 0.9355 (2) | 4.19e-03 | 1 | 4 (256) | R:`20261012T045908Z_prefill_batch_vs_token_glm5.3_flash_tcq` |
| Qwen3.8-27B side8 (denso) | 2048 | 6.25e-04 / 8.50e-03 | 0.9863 (28) | 6.15e-04 / 4.54e-03 | 1.0000 (0) | 3.63e-04 | 1 | 52 (256) | R:`20261012T053911Z_prefill_batch_vs_token_qwen3.8_27b_side8` |

Storia del lavoro sul prefill e dei profili alla sua base: [cpp/README-IT.md](../cpp/README-IT.md) ("Fast batched prefill") e
[prefill_vs_ds4.md](prefill_vs_ds4.md).

## 2. La velocità dei modelli originali

Sessioni `suc perf` appaiate del 2026-10-02/08 (BF16, MLX affine 4-bit e SUC in un'unica sessione per modello; decode a ctx
128, mediana di 3). Le colonne di rapporto confrontano con il decode SUC *in quella sessione*; il prefill SUC è stato riscritto
da allora (§1c), quindi qui fanno da riferimento solo le colonne BF16 / MLX.

| Modello | BF16 originale: decode ms/tok · prefill tok/s · TTFT 128 / 2048 s · memoria | MLX affine 4-bit: stesse colonne · PPL / KL | SUC C++ vs BF16 originale (decode) | SUC C++ vs MLX 4-bit (decode) |
|---|---|---|---|---|
| Qwen3-0.6B (M3) ᵃ | 16.88 · 2,031 · 0.063 / — · 1.35 GiB picco | 6.72 · 1,957 · 0.065 / — · 0.55 GiB · 24.295 / 0.190 | 2.4x più veloce | 0.97x (harness diverso) |
| Qwen3-8B | 27.75 · 479 · 0.27 / 2.18 · 16.5 GB | 10.44 · 588 · 0.22 / 2.66 · 4.9 GB · 10.088 / 0.092 | 2.0x più veloce | 0.75x |
| Qwen3.8-27B | 98.59 · 141 · 0.90 / 8.58 · 54.3 GB picco | 38.46 · 169 · 0.76 / 9.58 · 15.8 GB · 7.212 / 0.045 ᶜ | 2.5x più veloce | 0.99x (side8 39.02, base 38.95) |
| Qwen3-30B-A3B | 19.40 · 799 · 0.16 / 1.11 · 61.2 GB | 13.38 · 753 · 0.17 / 1.53 · 17.4 GB · 9.071 / 0.081 | 1.54x più veloce | **1.06x più veloce** |
| Qwen3.8-Flash-Next | non entra in memoria (354 GB), non misurabile | entra (71.8 GB residenti): velocità — ᵇ · 5.222 / 0.202 | — | — ᵇ |
| DeepSeek-V4-Flash | non entra in memoria (569 GB), non misurabile | non entra in memoria (≈ 160 GB calcolati), non misurabile | — | — |
| GLM-5.3-Flash | non entra in memoria (627 GB), non misurabile | non entra in memoria (≈ 176 GB calcolati), non misurabile | — | — |

ᵃ Harness e2e sull'M3 con decode MLX **sincrono** (R:`20261002T210018Z_e2e_final_baseline_qwen3_0.6b_m3`), C++
   in pipeline: le colonne MLX sono pessimistiche. Lo smoke `suc perf` appaiato sull'M3 (MLX 4-bit 4.96 ms, SUC C++
   7.30 ms, 0.68x; B: "Matched-harness speed") non ha un JSON di risultato committato. Qualità MLX 4-bit: RTN affine 4-bit,
   R:`20261002T204839Z_final_baseline_qwen3_0.6b`.

ᵇ L'MLX 4-bit di Qwen3.8 non è stato cronometrato in `suc perf`. L'unica misura nella stessa sessione è l'harness `suc eval`
   (R:`20261004T111255Z_eval_qwen3.8_flash_next`, mediana sui casi della mediana di decode per caso): MLX 4-bit
   44.3 ms/token, SUC C++ 35.9 ms/token — solo indicativo (eval non è un protocollo di velocità).

ᶜ Qwen3.8-27B: `suc perf` R:`20261008T010229Z_perf_qwen3.8_27b` (BF16, MLX affine 4-bit g64 ed entrambi i file SUC1 in
    un'unica sessione). Qualità MLX 4-bit (4.51 bpp, quantizza i tensori che SUC codifica + embedding / head) e MLX 8-bit
    (8.51 bpp: PPL 7.059, KL 0.0022, top-1 0.986) sul riferimento / sulle posizioni dell'esecuzione side8:
    R:`20261008T004540Z_quality_eval_qwen3.8_27b`. Core suite: MLX 4-bit 30/30, MLX 8-bit 30/30
    (R:`20261008T012317Z_eval_qwen3.8_27b`). Prefill: MLX 4-bit 169 / 214 tok/s (ctx 128 / 2048), SUC C++ 32 / 31.

Righe 8B / 30B-A3B: R:`*_perf_qwen3_8b_final`, R:`*_perf_qwen3_30b_a3b_final` (stessa sessione di SUC); qualità MLX 4-bit
= il piano uniforme RTN affine 4-bit di R:`20261001T072625Z_q7_allocate_qwen3_8b` e
R:`20261002T070150Z_q7_allocate_qwen3_30b_a3b`. Qualità MLX 4-bit di Qwen3.8: R:`20261004T102809Z_quality_eval_qwen3.8_flash_next`
(4.57 bpp sull'insieme codificato, tabelle n-gram lasciate grezze).

## 3. Figure

Disegnate da `scripts/plot_compression_report.py` sul set di dati del 2026-10-07 (i file di riferimento di quella data, prima delle
ricodifiche tcq e del prefill veloce): mantenute per il quadro dimensione / qualità; la velocità attuale e i file scelti sono in §1.

![Dimensione originale vs SUC1](figures/compression_size.svg)

*File originale (equivalente BF16) vs SUC1 per modello, scala logaritmica; il trattino grigio è il checkpoint pubblicato su disco quando
è già a bassa precisione; la linea tratteggiata è il budget di runtime di 111 GB.*

![Velocità di decode vs dimensione del modello](figures/compression_decode.svg)

*Decode batch 1 in ms/token vs parametri totali (log/log). Rombi vuoti: streaming degli expert da SSD. Qwen3.8 e DS-V4:
`suc1-generate` prompt breve, 256 token; gli altri `suc perf` ctx 128. 0.6B sull'M3, tutto il resto sull'M1 Ultra.*

![Qualità vs bpp](figures/compression_quality.svg)

*KL rispetto al riferimento a piena precisione in funzione dei bpp dell'intero file; i segmenti grigi uniscono ogni file SUC alla baseline MLX / RTN affine
4-bit dello stesso modello, dove è stato possibile eseguirla; quadrati viola (vuoti = streaming da SSD): GGUF della community eseguiti
con ds4, collegati da una linea punteggiata al file SUC con cui sono confrontati (§5).*

![Prefill e TTFT](figures/compression_prefill_ttft.svg)

*Sinistra: prefill tok/s a ctx 128 (tratteggiato: `suc1-generate` con prompt breve / verifica del prefill a batch, vedi le note delle tabelle).
Destra: tempo al primo token a ctx 2048 (solo `suc perf`).*

## 4. Altri file misurati

Varianti che non sono il file scelto di §1 (qualità: la loro esecuzione di compress; velocità: righe di §1b–§1c; stessi protocolli).

| Modello | File | GB | bpp | Memoria GB | KL | top-1 | PPL | Decode tok/s | Prefill tok/s · TTFT, 2k | Perché non scelto |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | Fisher greedy | 15.3 | 4.02 | — | 0.064 | — | — | 58.5 ᵃ | — | stessa KL dell'uniforme, decode più lento del 30 % |
| Qwen3.8-27B | base (head a 4 bit) | 14.18 | 4.22 | 14.57 | 0.031 | 0.929 | 7.222 | 25.6 | 68 · 30.0 s ᵇ | KL più alta di side8 per 2.2 GB in meno |
| Qwen3.8-Flash-Next | base (ngram4) | 93.49 | 4.23 | 70.99 | 0.145 | 0.856 | 5.055 | 26.3 ᶜ | — | 2.3× la KL di side8 |
| Qwen3.8-Flash-Next | iso (ngram4) | 100.23 | 4.53 | — | 0.056 | 0.916 | — | 28.3 | — | 3.9 GB più di side8 per −0.008 di KL |
| DeepSeek-V4-Flash | V1 native_max | 108.67 | 3.06 | 113.26 | 0.225 | 0.826 | 5.362 | 25.5 | 41 · 50.4 s ᵇ | 12 GB più di tcq, nessun guadagno nelle risposte |
| DeepSeek-V4-Flash | V3 native_stream (expert da SSD) | 155.05 | 4.36 | 116.75 | 0.036 | 0.916 | 4.840 | 13.1 | 28 · 73.9 s ᵇ | tetto di qualità; streaming da SSD |
| DeepSeek-V4-Flash | s97 (VQ low tier) | 96.82 | 2.72 | 102.38 | 0.387 | 0.787 | 5.792 | 25.8 | — | superato da tcq (stessa dimensione) |
| DeepSeek-V4-Flash | tcq, mix di calibrazione v2 | 96.83 | 2.72 | 102.56 | 0.246 ᵈ | 0.822 | 5.603 | 23.9 | 170 · 12.1 s | MMLU-Pro 10 vs 16/20; 101/119 vs 103/120 |
| GLM-5.3-Flash | base (expert da SSD) | 119.93 | 3.06 | 115.03 | 0.261 | 0.815 | 4.009 | 9.1 | 16 · 125.8 s ᵇ | streaming da SSD: decode 2.4× più lento di tcq |

ᵃ `suc perf` ctx 128 (17.11 ms/token). ᵇ Prefill esatto per riga di una build più vecchia (non rimisurato con il prefill
a batch veloce). ᶜ 38.08 ms/token, build del 2026-10-06. ᵈ Sulle righe di validazione v1 (stesso protocollo di tutte le altre righe; 0.2455 sulle
sue righe v2).

## 5. Confronto con i GGUF ds4 (modelli frontier)

I GGUF della community / ds4 sullo Studio (`~/Projects/ds4/GGUF`), eseguiti con il motore ds4 (Metal, commit `9139e2a`),
coda `scripts/studio_gguf_compare.sh`, strumenti `scripts/gguf_compare.py` + `scripts/gguf/ds4_score.c` (linkato al
motore di ds4, ds4 non modificato), 2026-10-06/07. La KL è misurata esattamente come per le righe SUC1 (vedi Metodo).

| Modello | File GGUF | GB | bpp | Memoria | KL | top-1 | ΔCE | Decode tok/s | Prefill tok/s · TTFT, breve | PPL GGUF, prime finestre |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| DeepSeek-V4-Flash | IQ2_XXS mix | 97.59 | 2.746 | residente | 0.2961 | 0.826 | 0.1253 | 24.5 | 41.6 · 0.33 s | 5.752 (8 × 4096) |
| DeepSeek-V4-Flash | expert MXFP4 | 155.98 | 4.389 | streaming da SSD | 0.0108 | 0.965 | 0.0064 | 10.0 | 6.6 · 1.97 s | 3.526 (5 × 4096) |
| GLM-5.3-Flash | Q2 | 96.51 | 2.464 | streaming da SSD | 0.4563 | 0.755 | 0.2522 | 24.3 | 41.3 · 0.36 s | 4.463 (8 × 4096) |
| Qwen3.8-Flash-Next | Q4 ¹⁰ | 177.28 | 8.015 ¹⁰ | residente | 0.0614 | 0.915 | −0.0086 | 33.6 | 47.2 · 0.31 s | 2.329 (8 × 2048) |

Controparti SUC1: §1 e §4. ΔCE = aumento medio della NLL rispetto al riferimento BF16 sugli stessi token (rapporto di PPL =
e^ΔCE). Velocità: il protocollo frontier con prompt breve per entrambi (stessa frase, 256 token greedy, decode = mediana dei passi
2…n, TTFT = prefill + primo passo; GGUF: 1 riscaldamento + 1 esecuzione in un solo processo). Prompt lunghi: ds4 è misurato solo sui
prompt dell'hard suite (§1, terza tabella; [prefill_vs_ds4.md](prefill_vs_ds4.md)).

10. ds4 tiene la tabella n-gram BF16 di Qwen3.8 (`per_layer_token_embd`, 102.4 GB dei 177.3 GB del file) fuori dalla mappatura
    dei pesi e ne legge le righe dal disco a ogni token; i suoi pesi residenti sono 74.8 GB (4.67 bpw sui 128.3 B
    parametri non n-gram). SUC1 memorizza le tabelle n-gram in affine per riga a 4 bit dentro i suoi 96.32 GB.

**Verifica iso-bit di Qwen3.8** (config `compress_qwen3.8_flash_next_{iso,side8}.yaml`): bit per categoria come nell'header
del GGUF (expert instradati 4.42, attention / expert condiviso / PLE 8.5, head 8.5, embedding 16, tabelle n-gram BF16). Il file iso
con tabelle BF16 grezze (175.75 GB) ha KL 0.054 contro 0.061 del GGUF, top-1 0.921 contro 0.915, PPL 4.682 (BF16
4.695): a parità di bit per categoria i codec SUC sono leggermente migliori. side8 mantiene quella qualità (0.064) al 54 % della
dimensione del GGUF memorizzando le tabelle n-gram a 4 bit. R:`20261007T122502Z_compress_qwen3.8_flash_next_iso`.

**Risultati.**

1. A 2.5–2.7 bpp SUC tcq diverge meno dei GGUF a 2 bit della stessa dimensione (DeepSeek 0.235 vs 0.296 con −0.8 %
   di byte; GLM 0.307 vs 0.456 con +2.6 %), e fa decode al 98 % (DeepSeek) / 91 % (GLM) della velocità di ds4.
2. A ≈ 4.4 bpp il file DeepSeek MXFP4 di ds4 è migliore di SUC V3 (KL 0.011 vs 0.036, stessa dimensione): entrambi tengono gli
   expert nel formato nativo MXFP4, ma il GGUF mantiene F16 / Q8 per i tensori non-expert e la head, dove V3 usa embedding / head
   a 4 bit e attention a 8 bit.
3. Qwen3.8-Flash: stessa KL (0.064 vs 0.061) al 54 % della dimensione del file; ds4 fa decode il 19 % più veloce (33.6 vs 28.3 tok/s).
4. Prompt lunghi: ds4 fa prefill 1.2–1.7× più veloce (DeepSeek 224 vs 138, GLM 117 vs 94, Qwen3.8 558 vs 323 tok/s;
   insiemi di prompt diversi). Analisi del divario nel decode: [decode_speed_vs_ds4.md](decode_speed_vs_ds4.md); prefill:
   [prefill_vs_ds4.md](prefill_vs_ds4.md).

Fonti: R:`20261006T231508Z_gguf_quality_ds_iq2`, R:`20261006T225454Z_gguf_perf_ds_iq2`,
R:`20261007T003712Z_gguf_quality_ds_mxfp4`, R:`20261007T000702Z_gguf_perf_ds_mxfp4`,
R:`20261006T235206Z_gguf_quality_glm_q2`, R:`20261006T232202Z_gguf_perf_glm_q2`,
R:`20261007T082341Z_gguf_quality_q38_q4`, R:`20261007T081505Z_gguf_perf_q38_q4`.

### Metodo

- *File*: DeepSeek-V4-Flash 0731 `…Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf`
  (IQ2_XXS gate/up, Q2_K down, expert Q4_K negli strati 37–42, attention / condivisi / output Q8; residente, 91 GiB mappati),
  `…MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf` (streaming da SSD di ds4,
  obiettivo della cache degli expert 80 GB), `GLM-5.3-Flash-Q2.gguf` (streaming da SSD di ds4, obiettivo 70 GB; ds4 ha adattato la cache a 2,221
  expert), `Qwen3.8-Flash-Next-Q4.gguf` (ds4 residente: righe n-gram lette dal disco, ¹⁰). I file oltre il budget di 111 GB
  non vengono mai aperti senza streaming, a meno che, come per Qwen3.8, il motore tenga la maggior parte del file fuori dalla memoria: il primo tentativo di aprire il file
  da 156 GB in modo residente (una verifica del tokenizer) ha causato un kernel panic da watchdog dello Studio il 2026-10-07; una protezione della memoria ora
  interrompe la coda sotto il 10 % di memoria libera o in caso di crescita dello swap.
- *Token*: ai GGUF vengono passati **gli esatti id di token dello stadio evaluate di `suc compress`** (tokenizer HF; sha256 di WikiText-2
  verificato contro il `reference.json` dell'esecuzione di compress); la tokenizzazione propria di ds4 di un campione di WikiText è identica
  agli id HF (1,387 / 1,402 token per DeepSeek / GLM). Ogni sequenza parte da un contesto vuoto ed è decodificata
  token per token; i logit completi del token successivo vengono letti a ogni posizione.
- *KL / top-1 / ΔCE*: le 16 × 512 sequenze di validazione tenute da parte dell'esecuzione di compress (`calibrate/data.safetensors`)
  contro i logit di riferimento BF16 salvati dal suo stadio reference (`reference/ref_logits.safetensors`), logit GGUF
  arrotondati a bf16 e valutati con `suc.analysis.sensitivity.logit_divergence` — le stesse posizioni, lo stesso riferimento e la stessa
  aritmetica dello stadio evaluate di SUC (8,176 posizioni valutate ciascuno), quindi non serve rimisurare i file SUC1
  (Qwen3.8: `scripts/system_quality.py` sulle stesse posizioni). Le sequenze KL di DeepSeek IQ2 sono state decodificate anche con il
  batch a 4 sessioni di ds4: logit bit-identici al batch 1.
- *PPL*: le prime finestre delle stesse finestre WikiText-2 da 4096 + 1 token, batch a 4 sessioni per DeepSeek IQ2, NLL dai
  logit fp32.

### 5b. Varianti di dimensione / velocità: SUC1 al livello di qualità dei GGUF

Decisione del proprietario 2026-10-08 (dopo l'hard suite: i GGUF ds4 rispondono più o meno bene quanto SUC1): spendere il margine di KL di SUC in dimensione del file e velocità di decode. Ri-pianificazioni degli artefatti esistenti di calibrate / reference / candidates (DeepSeek, GLM: override di `plan.allocation`, nessun nuovo passaggio Fisher; Qwen3.8: la prima esecuzione Fisher-greedy del modello, entrambi i budget da un unico stadio candidates). KL / top-1 / ΔCE: il protocollo 16 × 512 di §5 (stadio evaluate dell'esecuzione di compress; GGUF: `scripts/gguf_compare.py`). Velocità: `scripts/frontier_measure.py generate` (build attuale, prompt breve, 256 token greedy, singola esecuzione, residente salvo dove è indicato un hit rate); GGUF: l'esecuzione con prompt breve di §5. Residente = file mappato + footprint del processo. Coda: `scripts/studio_size_variants.sh`; config `experiments/configs/compress_*_{s97,s94,r245,r235,g94,g91}.yaml` (header: previsioni del solo piano).

| Modello | sistema | GB | bpp | GB residenti | KL | top-1 | ΔCE | PPL | decode ms/tok | GPU ms/tok | prefill s | dimensione vs GGUF | risultato |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | ds4 IQ2_XXS mix (residente) | 97.59 | 2.746 | — | **0.2961** | 0.826 | 0.1253 | — | 40.86 | — | 0.29 | — | R:`20261006T231508Z_gguf_quality_ds_iq2`, R:`20261006T225454Z_gguf_perf_ds_iq2` |
| | SUC1 V1 native_max (riferimento) | 108.67 | 3.056 | 113.26 | 0.2247 | 0.826 | 0.1164 | 5.362 | 39.19 | 36.97 | 0.27 (11 tok) | +11.3 % | R:`20261006T115604Z_compress_deepseek_v4_flash_native_max`, R:`20261008T165717Z_gen_cpp_deepseek_v4_flash_native_max` |
| | **SUC1 s97: native / G4 / M3, budget 96.9 GB** | 96.82 | 2.722 | 102.38 | 0.3871 | 0.787 | 0.2600 | 5.792 | 38.77 | 36.45 | 0.30 (11 tok) | -0.8 % | R:`20261008T155110Z_compress_deepseek_v4_flash_s97`, R:`20261008T155353Z_gen_cpp_deepseek_v4_flash_s97` |
| | **SUC1 s94: native / G4 / M3, budget 94.0 GB** | — | — | — | — | — | — | — | — | — | — | — | non ancora eseguito |
| GLM-5.3-Flash | ds4 Q2 (streaming da SSD) | 96.51 | 2.464 | — | **0.4563** | 0.755 | 0.2522 | — | 41.14 | — | 0.31 | — | R:`20261006T235206Z_gguf_quality_glm_q2`, R:`20261006T232202Z_gguf_perf_glm_q2` |
| | SUC1 base, expert 3.0 bpw, streaming da SSD (riferimento) | 119.93 | 3.060 | 115.03 (hit 0.911) | 0.2605 | 0.815 | 0.1280 | 4.009 | 109.59 | 45.36 | 2.12 (13 tok) | +24.3 % | R:`20261005T022627Z_compress_glm5.3_flash`, R:`20261008T180714Z_gen_cpp_glm5.3_flash_streamed` |
| | **SUC1 r245: expert 2.45 bpw, residente** | — | — | — | — | — | — | — | — | — | — | — | non ancora eseguito |
| | **SUC1 r235: expert 2.35 bpw, residente** | — | — | — | — | — | — | — | — | — | — | — | non ancora eseguito |
| Qwen3.8-Flash-Next | ds4 Q4 (residente, righe n-gram BF16 dal disco) | 177.28 | 8.015 | — | **0.0614** | 0.915 | -0.0086 | — | 29.75 | — | 0.28 | — | R:`20261007T082341Z_gguf_quality_q38_q4`, R:`20261007T081505Z_gguf_perf_q38_q4` |
| | SUC1 side8 (ngram4, riferimento) | 96.32 | 4.348 | 73.82 | 0.0638 | 0.913 | -0.0208 | 4.704 | 35.30 | 26.48 | 0.23 (12 tok) | -45.7 % | R:`20261007T140252Z_compress_qwen3.8_flash_next_side8`, R:`20261012T050109Z_gen_cpp_qwen3.8_flash_next_side8_ngram4` |
| | **SUC1 g94: Fisher greedy, budget 94.0 GB** | — | — | — | — | — | — | — | — | — | — | — | non ancora eseguito |
| | **SUC1 g91: Fisher greedy, budget 91.0 GB** | — | — | — | — | — | — | — | — | — | — | — | non ancora eseguito |

Qwen3.8: il file GGUF tiene la tabella n-gram BF16 (102.4 GB) fuori dalla mappatura dei pesi (§5 nota 10; pesi residenti 74.8 GB), quindi lì la dimensione del file non è la grandezza confrontabile: confrontare i byte residenti e la velocità di decode.

### 5c. Ricodifiche del low tier trellis (tcq)

docs/low_tier_codec.md §6, decisione del proprietario 2026-10-09 (integrazione completa, L=12): gli stessi calibrate / reference / Fisher e gli stessi candidati delle varianti in memoria di §5b (punteggi riusati: `plan.candidates_reuse`), più due candidati per gli expert instradati, tcq 2-bit e 3-bit (3INST, V=2, L=12, BlockLDLQ con il secondo momento gate/up condiviso di ogni strato MoE e momenti down per expert dallo stadio `hessians`, residuo per colonne dello 0.5 %, rotazione dell'input condivisa sullo stack; valutati su 256 righe per tensore expert con bit esatti). Stesso budget della variante sopra (DeepSeek 96.9 GB totali; expert instradati GLM 2.45 bpw, completamente residenti). KL / top-1 / ΔCE: il protocollo 16 × 512 di §5; velocità: `scripts/frontier_measure.py generate` (C++, build attuale, prompt frontier breve, 256 token greedy, residente); hard suite: §7 (30 domande). Coda: `scripts/studio_tcq_frontier.sh`; config `experiments/configs/compress_{deepseek_v4_flash,glm5.3_flash}_tcq.yaml`.

| Modello | sistema | GB | bpp | GB residenti | KL | top-1 | ΔCE | PPL | decode ms/tok | GPU ms/tok | prefill s | dimensione vs GGUF | risultato |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | ds4 IQ2_XXS mix (residente) | 97.59 | 2.746 | — | **0.2961** | 0.826 | 0.1253 | — | 40.86 | — | 0.29 | — | R:`20261006T231508Z_gguf_quality_ds_iq2`, R:`20261006T225454Z_gguf_perf_ds_iq2` |
| | SUC1 V1 native_max (riferimento) | 108.67 | 3.056 | 113.26 | 0.2247 | 0.826 | 0.1164 | 5.362 | 39.19 | 36.97 | 0.27 (11 tok) | +11.3 % | R:`20261006T115604Z_compress_deepseek_v4_flash_native_max`, R:`20261008T165717Z_gen_cpp_deepseek_v4_flash_native_max` |
| | SUC1 s97: native / G4 / M3, 96.9 GB | 96.82 | 2.722 | 102.38 | 0.3871 | 0.787 | 0.2600 | 5.792 | 38.77 | 36.45 | 0.30 (11 tok) | -0.8 % | R:`20261008T155110Z_compress_deepseek_v4_flash_s97`, R:`20261008T155353Z_gen_cpp_deepseek_v4_flash_s97` |
| | **SUC1 tcq: native / G4 / M3 / tcq 2+3-bit, 96.9 GB** | 96.82 | 2.722 | 102.57 | 0.2346 | 0.825 | 0.1386 | 5.587 | 41.61 | 38.60 | 0.25 (11 tok) | -0.8 % | R:`20261009T181636Z_compress_deepseek_v4_flash_tcq`, R:`20261012T040303Z_gen_cpp_deepseek_v4_flash_tcq` |
| | **SUC1 tcq, mix di calibrazione v2 (GSM8K + MATH-train), 96.9 GB** | 96.83 | 2.722 | 102.56 | 0.2455 | 0.819 | 0.1494 | 5.603 | 41.77 | 38.76 | 0.25 (11 tok) | -0.8 % | R:`20261012T233128Z_compress_deepseek_v4_flash_tcq_mathcal`, R:`20261012T233316Z_gen_cpp_deepseek_v4_flash_tcq_mathcal` |
| GLM-5.3-Flash | ds4 Q2 (streaming da SSD) | 96.51 | 2.464 | — | **0.4563** | 0.755 | 0.2522 | — | 41.14 | — | 0.31 | — | R:`20261006T235206Z_gguf_quality_glm_q2`, R:`20261006T232202Z_gguf_perf_glm_q2` |
| | SUC1 base, expert 3.0 bpw, streaming da SSD (riferimento) | 119.93 | 3.060 | 115.03 (hit 0.911) | 0.2605 | 0.815 | 0.1280 | 4.009 | 109.59 | 45.36 | 2.12 (13 tok) | +24.3 % | R:`20261005T022627Z_compress_glm5.3_flash`, R:`20261008T180714Z_gen_cpp_glm5.3_flash_streamed` |
| | **SUC1 tcq: expert 2.45 bpw incl. tcq 2+3-bit, residente** | 99.00 | 2.526 | 106.59 | 0.3073 | 0.793 | 0.1558 | 4.269 | 45.08 | 41.62 | 0.35 (12 tok) | +2.6 % | R:`20261010T124951Z_compress_glm5.3_flash_tcq`, R:`20261012T043057Z_gen_cpp_glm5.3_flash_tcq` |

- DeepSeek-V4-Flash, SUC1 tcq: native / G4 / M3 / tcq 2+3-bit, 96.9 GB: tensori degli expert instradati tcq2 14,316, G4 9,431, tcq3 4,722, native 4,355, M3 200; prefill 45.7 tok/s (breve) / 171.4 tok/s (2048 token); hard suite 25/30.
- DeepSeek-V4-Flash, SUC1 tcq, mix di calibrazione v2 (GSM8K + MATH-train), 96.9 GB: tensori degli expert instradati tcq2 14,392, G4 9,449, tcq3 4,564, native 4,426, M3 193; prefill 46.1 tok/s (breve) / 170.3 tok/s (2048 token); hard suite 101/119 (1 errore).
  - KL a parità di protocollo (decode C++ in teacher forcing, `scripts/suc1_cross_kl.py`): sul protocollo v1 (le righe / il riferimento della precedente esecuzione tcq, = il protocollo delle righe dei GGUF ds4) file calib v2 KL 0.246, top-1 0.822, ΔCE 0.148 contro file calib v1 KL 0.234, top-1 0.826, ΔCE 0.140; sul protocollo v2 (le righe / il riferimento di questa esecuzione) file calib v2 KL 0.248, top-1 0.823, ΔCE 0.158 contro file calib v1 KL 0.241, top-1 0.824, ΔCE 0.144. Ogni file sul proprio protocollo è la verifica del percorso C++ rispetto al suo stadio evaluate (la colonna KL). La colonna KL di questa riga è sul protocollo v2 (righe di validazione diverse da quelle delle righe sopra).
- GLM-5.3-Flash, SUC1 tcq: expert 2.45 bpw incl. tcq 2+3-bit, residente: tensori degli expert instradati tcq2 22,038, tcq3 6,596, G4 5,633, G2 1,937, M3 84; prefill 35.2 tok/s (breve) / 116.3 tok/s (2048 token); hard suite 25/30.

## 6. Modelli piccoli: SUC1 vs GGUF llama.cpp / LM Studio

I modelli che ds4 non esegue (Qwen3-0.6B, Qwen3-8B, Qwen3-30B-A3B) a confronto con i K-quant GGUF che LM Studio distribuisce, coda
`scripts/studio_gguf_llama.sh` (2026-10-07), strumenti `scripts/gguf_llama.py`, `scripts/gguf/llama_score.cpp`,
`scripts/server_speed.py`. **Nessun GGUF è stato scaricato**: ogni file è stato convertito localmente dallo stesso snapshot HF
(`convert_hf_to_gguf.py` → BF16) e quantizzato con `llama-quantize` di un llama.cpp fissato (`b11477`, Metal), **senza
imatrix** (i file Q4_K_M / Q3_K_M della community sono spesso calibrati con imatrix e potrebbero ottenere punteggi migliori). La libreria
di LM Studio sullo Studio non contiene alcun GGUF di questi tre modelli; i file locali vi sono stati collegati con hard link
(`~/.lmstudio/models/suc-local/`) e serviti dal runtime llama.cpp di LM Studio
(`llama.cpp-mac-arm64-apple-metal-advsimd@2.41.0`).

| Modello | file | GB | bpp | KL | KL p95 (media per chunk ¹³) | top-1 | ΔCE | prefill tok/s, breve / 2048 | TTFT s, breve / 2048 | decode ms/tok, breve / 2048 | eval (core) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | GGUF BF16 (soglia del motore) | 1.509 | 16.07 | 0.0009 | 0.002 | 0.985 | 0.0011 | — | — | — | — |
| | GGUF Q8_0 † | 0.805 | 8.57 | 0.0018 | 0.005 | 0.978 | −0.0013 | 615 / 8,151 | 0.021 / 0.25 | 5.07 / 5.26 | 23/30 |
| | GGUF Q4_K_M † | 0.484 | 5.15 | 0.0974 | 0.273 | 0.865 | 0.0880 | 594 / 7,785 | 0.022 / 0.26 | 4.55 / 4.25 | 19/30 |
| | GGUF Q3_K_M | 0.414 | 4.41 | 0.3307 | 1.100 | 0.741 | 0.2550 | — | — | — | — |
| | **SUC1** ¹² | 0.32 | 4.257 | **0.0924** | 0.303 | 0.853 | 0.0602 | — (solo M3) | — | — | — |
| Qwen3-8B | GGUF BF16 (soglia del motore) | 16.388 | 16.01 | 0.0008 | 0.002 | 0.989 | −0.0010 | — | — | — | — |
| | GGUF Q8_0 † | 8.710 | 8.51 | 0.0012 | 0.004 | 0.986 | 0.0029 | 193 / 1,007 | 0.067 / 2.03 | 14.72 / 15.51 | 29/30 |
| | GGUF Q4_K_M † | 5.028 | 4.91 | **0.0399** | 0.135 | 0.928 | 0.0314 | 176 / 884 | 0.074 / 2.32 | 13.38 / 14.08 | 30/30 |
| | GGUF Q3_K_M | 4.124 | 4.03 | 0.1259 | 0.504 | 0.870 | 0.0566 | — | — | — | — |
| | **SUC1** | 4.32 | 4.222 | 0.0494 | 0.149 | 0.902 | 0.0368 | 128 / 304 | 0.11 / 6.75 | 13.04 / 13.72 | 29/30 |
| Qwen3-30B-A3B | GGUF BF16 (soglia del motore) | 61.096 | 16.01 | 0.0049 | 0.012 | 0.978 | −0.0002 | — | — | — | — |
| | GGUF Q8_0 † | 32.484 | 8.51 | 0.0058 | 0.014 | 0.975 | 0.0000 | 146 / 1,642 | 0.089 / 1.25 | 12.51 / 13.55 | 29/30 |
| | GGUF Q4_K_M † | 18.557 | 4.86 | **0.0386** | 0.109 | 0.940 | 0.0025 | 192 / 1,455 | 0.068 / 1.41 | 10.68 / 11.74 | 30/30 |
| | GGUF Q3_K_M | 14.712 | 3.86 | 0.0841 | 0.293 | 0.904 | 0.0782 | — | — | — | — |
| | **SUC1** (uniforme) | 15.93 | 4.174 | 0.0644 | 0.186 | 0.897 | 0.0415 | 199 / 367 | 0.08 / 5.59 | 12.20 / 13.81 | 30/30 |

12. Il file SUC1 da 0.6B è stato prodotto e cronometrato sul portatile M3 (§1); la sua riga KL e le righe KL dei GGUF usano gli stessi
    file di riferimento / token di quell'esecuzione di compress (copiati sullo Studio, `~/suc-local/compress/compress_qwen3_0.6b_final`).
    Non esiste una eval SUC del 0.6B.
13. KL p95 (media per chunk): media sui 16 chunk di valutazione del 95° percentile di ciascun chunk, la definizione di tutti i risultati scritti prima della correzione dell'audit bf86361 (campo `kl_p95` senza `kl_p95_definition`); i risultati più recenti riportano il 95° percentile globale su tutti i token (`kl_p95_definition: global`, valore vecchio come `kl_p95_chunk_mean`). Le due definizioni non sono confrontabili.

† Celle LM Studio rimisurate con l'harness HTTP corretto (audit F03 / F04, schema delle metriche 2; motore lmstudio llama.cpp-mac-arm64-apple-metal-advsimd@2.41.0). Decode = (token − 1) / tempo di streaming: in 36 dei 36 round cronometrati ogni chunk SSE conteneva un token, quindi è un vero valore per token. Variazione massima rispetto ai valori superati: 13.7 %. Eval: rieseguita per Qwen3-8B Q8_0. Valori superati (2026-10-07, file conservati): Qwen3-0.6B Q4_K_M: TTFT 0.024 / 0.27 s, decode 4.64 / 4.93 ms/tok, prefill 549 / 7,537 tok/s; Qwen3-0.6B Q8_0: TTFT 0.020 / 0.25 s, decode 5.07 / 5.37 ms/tok, prefill 643 / 8,230 tok/s; Qwen3-8B Q4_K_M: TTFT 0.075 / 2.30 s, decode 13.35 / 14.08 ms/tok, prefill 174 / 889 tok/s; Qwen3-8B Q8_0: TTFT 0.066 / 2.03 s, decode 14.73 / 15.46 ms/tok, prefill 197 / 1,010 tok/s, eval 29/30 → 29/30; Qwen3-30B-A3B Q4_K_M: TTFT 0.067 / 1.40 s, decode 10.67 / 11.75 ms/tok, prefill 195 / 1,462 tok/s; Qwen3-30B-A3B Q8_0: TTFT 0.089 / 1.25 s, decode 12.48 / 13.52 ms/tok, prefill 146 / 1,643 tok/s. Fonti: R:`20261010T125801Z_server_speed_lms_q06_q4_k_m`, R:`20261010T125928Z_server_speed_lms_q06_q8_0`, R:`20261010T130139Z_server_speed_lms_q8_q4_k_m`, R:`20261010T130359Z_server_speed_lms_q8_q8_0`, R:`20261010T130628Z_server_speed_lms_q30_q4_k_m`, R:`20261010T130929Z_server_speed_lms_q30_q8_0`, R:`20261010T131357Z_gguf_eval_lms_q8_q8_0`; superati: R:`20261007T183414Z_server_speed_lms_q06_q4_k_m`, R:`20261007T200349Z_server_speed_lms_q06_q8_0`, R:`20261007T192714Z_server_speed_lms_q8_q4_k_m`, R:`20261007T194741Z_server_speed_lms_q8_q8_0`, R:`20261007T195404Z_server_speed_lms_q30_q4_k_m`, R:`20261007T195929Z_server_speed_lms_q30_q8_0`.

*Protocollo KL (allineato a quello di SUC).* Ogni GGUF è valutato **esattamente sui 16 × 512 id di token di validazione dell'esecuzione
`suc compress` del modello** (`calibrate/data.safetensors`; tokenizer di llama.cpp verificato identico agli id HF su un campione di 1,955
token per ogni modello), ogni riga decodificata da un contesto vuoto da `llama_score` (libllama, logit completi a ogni
posizione, arrotondati a bf16), contro i logit di riferimento BF16 salvati dallo stadio reference di quell'esecuzione, con la
`logit_divergence` di suc — le stesse posizioni (8,176), lo stesso riferimento e la stessa aritmetica della colonna KL di SUC in §1.
`llama-perplexity --kl-divergence` di llama.cpp **non** è stato usato (tokenizzazione propria, solo la seconda metà di ogni finestra, un GGUF BF16 come
riferimento). La riga GGUF BF16 è la soglia del motore llama.cpp rispetto a MLX: 0.0008–0.0009 sui modelli densi, 0.0049 sul
MoE: un limite inferiore sotto ogni riga GGUF di quel modello, piccolo rispetto alle KL a 4 bit. bpp = byte del file /
numero di parametri del modello, ogni overhead incluso, come per SUC1.

*Protocollo di velocità.* GGUF: server LM Studio (`lms load … --context-length 16384 --gpu max --parallel 1
--no-speculative-draft-mtp`), `/v1/completions` in streaming, greedy, `ignore_eos`, 256 token, lo stesso prompt breve
(13 id) e lo stesso prompt da 2048 token di §1c; per prompt 1 riscaldamento + 3 round cronometrati, ciascuno dopo un ricaricamento a freddo del modello e una
richiesta diversa della stessa lunghezza, così che nessuna cache dei prompt possa essere riusata (`prefix_cache_suspected` false ovunque, 0 discordanze
nei token del prompt); il cronometraggio è lato client via HTTP (il TTFT include HTTP e il campionamento del primo token, prefill tok/s =
token del prompt / TTFT, decode = (token − 1) / tempo di streaming). SUC1: le righe `suc1-generate` di §1c (nel processo, singola
esecuzione, prefill tok/s = prompt / tempo di prefill). I due protocolli sono vicini ma non identici: piccole differenze nel decode
(< 10 %) rientrano nella differenza di protocollo.

*Eval.* Core suite di `suc eval` (30 casi valutati) attraverso il server LM Studio (tipo di sistema `openai_server`) con la
stessa suite, lo stesso testo dei prompt, gli stessi stop, budget e valutatori della config di eval SUC del modello; righe SUC: fonti di §1
(8B C++ 29/30, 30B-A3B uniforme C++ 30/30).

**Risultati.**

1. **A ≈ 4.2 bpp SUC1 si colloca tra i K-quant.** Rispetto a Q3_K_M SUC ha una KL 2.5× (8B), 3.6× (0.6B) e 1.3× (30B-A3B)
   più bassa; rispetto a Q4_K_M (0.7–0.9 bpp in più) è migliore solo sul 0.6B e peggiore sull'8B (0.049 vs 0.040) e sul 30B-A3B
   (0.064 vs 0.039). Non esiste un punto a pari bpp. I K-quant qui non sono calibrati con imatrix.
2. **Prefill / TTFT: LM Studio è 3–4.5× più veloce su 2,048 token** (TTFT 2.3 vs 6.8 s sull'8B, 1.4 vs 5.6 s sul 30B-A3B);
   sul prompt breve il divario è piccolo (0.07 vs 0.11 s, 0.07 vs 0.08 s).
3. **Decode:** alla pari sull'8B (76.3 vs 74.8 tok/s per Q4_K_M), più lento del 12 % sul 30B-A3B (82.4 vs 93.6); cronometraggio nel processo
   vs lato client via HTTP (< 10 % di differenza di protocollo).
4. **La core eval non separa i file** (29–30/30 per SUC1, Q4_K_M e Q8_0 su 8B / 30B-A3B).

Fonti: R:`20261007T180211Z_gguf_llama_quality_q06_bf16`, R:`20261007T180219Z_gguf_llama_quality_q06_q8_0`,
R:`20261007T180227Z_gguf_llama_quality_q06_q4_k_m`, R:`20261007T180236Z_gguf_llama_quality_q06_q3_k_m`,
R:`20261007T192128Z_gguf_llama_quality_q8_bf16`, R:`20261007T192146Z_gguf_llama_quality_q8_q8_0`,
R:`20261007T192204Z_gguf_llama_quality_q8_q4_k_m`, R:`20261007T192222Z_gguf_llama_quality_q8_q3_k_m`,
R:`20261007T192356Z_gguf_llama_quality_q30_bf16`, R:`20261007T192423Z_gguf_llama_quality_q30_q8_0`,
R:`20261007T192443Z_gguf_llama_quality_q30_q4_k_m`, R:`20261007T192502Z_gguf_llama_quality_q30_q3_k_m`;
R:`20261007T183414Z_server_speed_lms_q06_q4_k_m`, R:`20261007T200349Z_server_speed_lms_q06_q8_0`,
R:`20261007T192714Z_server_speed_lms_q8_q4_k_m`, R:`20261007T194741Z_server_speed_lms_q8_q8_0`,
R:`20261007T195404Z_server_speed_lms_q30_q4_k_m`, R:`20261007T195929Z_server_speed_lms_q30_q8_0`;
R:`20261007T183500Z_gguf_eval_lms_q06_q4_k_m`, R:`20261007T200436Z_gguf_eval_lms_q06_q8_0`,
R:`20261007T193020Z_gguf_eval_lms_q8_q4_k_m`, R:`20261007T195135Z_gguf_eval_lms_q8_q8_0`,
R:`20261007T195628Z_gguf_eval_lms_q30_q4_k_m`, R:`20261007T200220Z_gguf_eval_lms_q30_q8_0`; righe SUC: §1, §1c,
R:`20261003T032730Z_eval_qwen3_8b_final`, R:`20261003T034721Z_eval_qwen3_30b_a3b_final`.

## 7. Eval a livello di risposte: core suite, SUC1 vs i GGUF ds4

Core suite di `suc eval` (`suc_core`, 40 casi di cui 30 valutati automaticamente: 10 QA di matematica + 20 SUC-basic a risposta intera / a scelta
multipla; 10 spiegazioni non valutate), greedy, stessi prompt e budget per ogni sistema. GGUF: `scripts/studio_gguf_eval.sh`
(motore ds4, `ds4_score_suc`); SUC1: runtime C++. Decode ms/token = mediana sui casi della mediana di decode per caso
(solo indicativo, eval non è un protocollo di velocità).

| Modello | sistema | bpp | KL | valutati superati | errori | decode ms/tok (eval) | risultato |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | SUC1 V1 native_max, C++ | 3.057 | 0.225 | 30/30 | — | 38.4 | R:`20261007T211615Z_eval_deepseek_v4_flash_v1v3_cpp` |
| | SUC1 V3 native_stream, C++ (SSD) | 4.362 | 0.036 | 30/30 | — | 129.0 | stesso file |
| | SUC1 `native`, C++ (precedente) | 3.022 | 0.248 | 30/30 | — | 80.7 | R:`20261006T001744Z_eval_deepseek_v4_flash` |
| | ds4 IQ2_XXS mix | 2.746 | 0.296 | 30/30 | — | 37.1 | R:`20261007T173042Z_gguf_eval_ds_iq2` |
| | ds4 expert MXFP4 (SSD) | 4.389 | 0.011 | 30/30 | — | 75.0 | R:`20261007T175402Z_gguf_eval_ds_mxfp4` |
| GLM-5.3-Flash | SUC1, C++ (SSD), dopo la correzione dello stop di fine turno | 3.062 | 0.261 | 30/30 | — | 155.1 | R:`20261007T031710Z_eval_glm5.3_flash_cpp` |
| | SUC1, C++ (SSD), prima della correzione | 3.062 | 0.261 | 27/30 | it_math_1 / 2 / 5 (nessuna fine turno) | 153.6 | R:`20261006T200231Z_eval_glm5.3_flash_cpp` |
| | ds4 Q2 | 2.464 | 0.456 | 29/30 | it_math_5 (valutatore: 361 vs 61) | 41.2 | R:`20261007T173814Z_gguf_eval_glm_q2` |
| Qwen3.8-Flash-Next | SUC1 side8 (ngram4), C++ | 4.355 | 0.064 | 30/30 | — | 33.6 | R:`20261007T213545Z_eval_qwen3.8_flash_next_side8_cpp` |
| | SUC1 base (ngram4), C++ | 4.227 | 0.145 | 30/30 | — | 35.9 | R:`20261004T111255Z_eval_qwen3.8_flash_next` |
| | MLX affine 4-bit | 4.57 | 0.202 | 30/30 | — | 44.3 | stesso file |
| | ds4 Q4 | 8.015 | 0.061 | 30/30 | — | 29.4 | R:`20261007T174250Z_gguf_eval_q38_q4` |

- *GLM Q2 it_math_5* ("144 diviso 12, più 7 al quadrato"): il GGUF risponde **61** (corretto) in grassetto, poi aggiunge una
  lettura alternativa "(144 ÷ 12 + 7)² = 361"; il valutatore prende l'ultimo numero in grassetto, 361. È un artefatto della valutazione,
  non una risposta sbagliata: nel contenuto anche il GGUF Q2 fa 30/30.
- **Conclusione: la core suite è satura.** Ogni file SUC1 di questa tabella e ogni GGUF ds4 ottiene 30/30 (GLM Q2 29/30,
  30/30 nel contenuto; Qwen3-8B / 30B-A3B in §6: 29–30/30), dal
  GLM Q2 a 2.46 bpp (KL 0.456) al DeepSeek MXFP4 (KL 0.011) — un intervallo di KL di 40x senza differenze nelle risposte. È
  uno smoke test (il modello compresso risponde ancora, chiude il turno, segue il formato), non una classifica di qualità;
  la KL / ΔCE di §5 e §6 restano le misure che separano i file.
- **Hard suite completate.** I risultati appaiati da 30 casi e quelli estesi da 120 casi sono qui sotto. Sono valutazioni delle risposte datate; vedere i commit registrati e le verifiche numeriche attuali del prefill in §1c.

### Hard suite: il divario di KL si vede nelle risposte? (SUC1 vs GGUF, 30 domande)

Suite `experiments/eval/suite_hard.json` (scripts/build_hard_suite.py): 5 elementi scelti deliberatamente per sottoinsieme (GSM8K test, MATH-500 L3-L5, MMLU-Pro 10 opzioni, ARC-Challenge, BBH 5 task, needle sintetici multi-hop in PG19 a ~8k/8k/16k/16k/32k token), greedy, thinking disattivato, riga `Answer:` in stile ds4, stessi id di token del prompt per entrambi i sistemi di un modello (SUC1 tramite il runtime C++, GGUF tramite il motore ds4). Con 5 elementi per sottoinsieme una sola domanda sposta un sottoinsieme di 20 punti: leggere le colonne appaiate (stesso esito / quale sistema da solo ha risolto cosa), non l'accuratezza come punteggio di benchmark. Coda: scripts/studio_hard_suite.sh.

| modello | sistema | totale | GSM8K | MATH | MMLU-Pro | ARC-Challenge | BBH | LongCtx | risultato |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-V4-Flash | SUC V1 native_max (108.7 GB) | **27/30** | 5/5 | 3/5 | 5/5 | 5/5 | 5/5 | 4/5 | `20261008T105124Z_eval_hard_ds_v1` (ce8c8c01d) |
| DeepSeek-V4-Flash | ds4 IQ2_XXS mix | **29/30** | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | `20261008T014228Z_gguf_eval_hard_ds_iq2` (58ceaa3cb) |
| DeepSeek-V4-Flash | SUC V3 native_stream (155.1 GB) | **28/30** | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | `20261008T031757Z_eval_hard_ds_v3` (0736053b3) |
| DeepSeek-V4-Flash | ds4 expert MXFP4 | **26/30** | 4/5 | 5/5 | 5/5 | 4/5 | 4/5 | 4/5 | `20261008T034912Z_gguf_eval_hard_ds_mxfp4` (c35f0a974) |
| DeepSeek-V4-Flash | SUC tcq (96.9 GB, residente) | **25/30** | 5/5 | 3/5 | 4/5 | 5/5 | 4/5 | 4/5 | `20261009T195431Z_eval_hard_ds_tcq` (d8a90984f) |
| GLM-5.3-Flash | SUC (streaming) | **25/30** | 5/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | `20261008T064059Z_eval_hard_glm` (d7d5ef5b6) |
| GLM-5.3-Flash | ds4 Q2 | **23/30** | 5/5 | 4/5 | 2/5 | 3/5 | 5/5 | 4/5 | `20261008T071222Z_gguf_eval_hard_glm_q2` (9846125b7) |
| GLM-5.3-Flash | SUC tcq (expert 2.45 bpw, residente) | **25/30** | 5/5 | 4/5 | 3/5 | 4/5 | 4/5 | 5/5 | `20261010T164841Z_eval_hard_glm_tcq` (f32720747) |
| Qwen3.8-Flash-Next | SUC side8 | **23/30** | 4/5 | 4/5 | 1/5 | 4/5 | 5/5 | 5/5 | `20261008T080106Z_eval_hard_q38_side8` (92158bae4) |
| Qwen3.8-Flash-Next | ds4 Q4 | **22/30** | 4/5 | 4/5 | 1/5 | 4/5 | 4/5 | 5/5 | `20261008T081738Z_gguf_eval_hard_q38_q4` (8e50e302f) |
| Qwen3.8-Flash-Next | SUC base 4.16 | **25/30** | 5/5 | 4/5 | 2/5 | 4/5 | 5/5 | 5/5 | `20261008T091237Z_eval_hard_q38_base` (bb8222376) |

Concordanza appaiata per modello (casi in cui entrambi i sistemi hanno prodotto una risposta):

| modello | SUC1 | GGUF | stesso superato/fallito | stessa risposta estratta | corretto solo SUC | corretto solo GGUF | casi diversi |
|---|---|---|---:|---:|---:|---:|---|
| DeepSeek-V4-Flash | SUC V1 native_max (108.7 GB) | ds4 IQ2_XXS mix | 28/30 | 28/30 | 0 | 2 | GGUF: math500/counting_and_probability/525; GGUF: math500/intermediate_algebra/1544 |
| DeepSeek-V4-Flash | SUC V3 native_stream (155.1 GB) | ds4 expert MXFP4 | 28/30 | 28/30 | 2 | 0 | SUC: arc_challenge/Mercury_7032428; SUC: bbh/date_understanding-118 |
| DeepSeek-V4-Flash | SUC tcq (96.9 GB, residente) | ds4 IQ2_XXS mix | 26/30 | 26/30 | 0 | 4 | GGUF: math500/intermediate_algebra/1544; GGUF: math500/geometry/627; GGUF: mmlu_pro/11613; GGUF: bbh/date_understanding-118 |
| GLM-5.3-Flash | SUC (streaming) | ds4 Q2 | 22/30 | 20/30 | 5 | 3 | SUC: mmlu_pro/1695; SUC: mmlu_pro/9135; SUC: mmlu_pro/11613; SUC: arc_challenge/TIMSS_2011_8_pg15; SUC: longctx/3hop_reverse_distract-31k; GGUF: mmlu_pro/3817; GGUF: bbh/causal_judgement-18; GGUF: longctx/sum_2x2hop-16k |
| GLM-5.3-Flash | SUC tcq (expert 2.45 bpw, residente) | ds4 Q2 | 24/30 | 22/30 | 4 | 2 | SUC: mmlu_pro/1695; SUC: mmlu_pro/11613; SUC: arc_challenge/TIMSS_2011_8_pg15; SUC: longctx/3hop_reverse_distract-31k; GGUF: mmlu_pro/8195; GGUF: bbh/causal_judgement-18 |
| Qwen3.8-Flash-Next | SUC side8 | ds4 Q4 | 29/30 | 23/30 | 1 | 0 | SUC: bbh/causal_judgement-18 |
| Qwen3.8-Flash-Next | SUC base 4.16 | ds4 Q4 | 27/30 | 23/30 | 3 | 0 | SUC: gsm8k/test-777; SUC: mmlu_pro/9135; SUC: bbh/causal_judgement-18 |

### Hard suite estesa (120 casi): SUC1 vs GGUF / LM Studio per elemento

Suite `experiments/eval/suite_hard20.json` (`scripts/build_hard_suite.py --per-subset 20`, sha256 `a7296a67a4cf`): le regole di selezione della suite da 30 casi applicate a round, 20 elementi per sottoinsieme (round 0 = i 30 casi sopra, stessi id e prompt; regole di selezione e id nel blocco `selection` della suite). Greedy, thinking disattivato, riga `Answer:` in stile ds4, stessi id di token del prompt per ogni sistema di un modello (SUC1 tramite il runtime C++; GGUF tramite il motore ds4; LM Studio MLX e il GGUF Q8_0 di Qwen3.8-27B tramite un server compatibile OpenAI -- LM Studio, rispettivamente llama-server di llama.cpp, il motore che LM Studio usa per i GGUF -- con l'harness dell'audit F03 / F04, prompt = lo stesso testo renderizzato, ctx 40960, parallel 1, nessun decoding speculativo / MTP). Coda: scripts/studio_hard20.sh (+ scripts/studio_llama_server_eval.sh); limite di tempo per job: i casi non avviati prima di esso sono SALTATI ("n/a" sotto) ed esclusi dalle statistiche appaiate. Campioni ancora piccoli: leggere una differenza attraverso la colonna McNemar (p esatto bilaterale sugli elementi discordanti), non l'accuratezza.

| modello | sistema | totale | GSM8K | MATH | MMLU-Pro | ARC-Challenge | BBH | LongCtx | risultato |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-V4-Flash | SUC tcq (96.9 GB, residente) | **103/120** | 19/20 | 12/20 | 16/20 | 19/20 | 19/20 | 18/20 | `20261011T055014Z_eval_hard20_ds_tcq` (091066e70) |
| DeepSeek-V4-Flash | ds4 IQ2_XXS mix | **106/120** | 18/20 | 17/20 | 16/20 | 18/20 | 18/20 | 19/20 | `20261011T011939Z_gguf_eval_hard20_ds_iq2` (5842859bb) |
| DeepSeek-V4-Flash | SUC tcq calib v2 (96.9 GB, residente) | **101/119** (1 errore) | 19/20 | 16/20 | 10/20 | 19/20 | 19/20 | 18/19 (1 n/a) | `20261013T043752Z_eval_hard20_ds_tcq_mathcal` (469a127d7) |
| GLM-5.3-Flash | SUC tcq (expert 2.45 bpw, residente) | **99/120** | 19/20 | 16/20 | 11/20 | 18/20 | 17/20 | 18/20 | `20261011T122746Z_eval_hard20_glm_tcq` (13c165233) |
| GLM-5.3-Flash | ds4 Q2 | **93/120** | 19/20 | 14/20 | 8/20 | 17/20 | 18/20 | 17/20 | `20261011T075042Z_gguf_eval_hard20_glm_q2` (1069527cd) |
| Qwen3.8-Flash-Next | SUC side8 | **97/120** | 17/20 | 15/20 | 6/20 | 20/20 | 19/20 | 20/20 | `20261011T154655Z_eval_hard20_q38_side8` (3f2c71241) |
| Qwen3.8-Flash-Next | ds4 Q4 | **99/120** | 18/20 | 16/20 | 7/20 | 19/20 | 19/20 | 20/20 | `20261011T132617Z_gguf_eval_hard20_q38_q4` (a580cd406) |
| Qwen3.8-27B | SUC side8 (16.4 GB) | **93/120** | 17/20 | 13/20 | 8/20 | 19/20 | 17/20 | 19/20 | `20261011T182537Z_eval_hard20_q38_27b_side8` (f84bafe63) |
| Qwen3.8-27B | GGUF Q8_0 (llama.cpp llama-server) | **92/120** | 18/20 | 13/20 | 6/20 | 18/20 | 17/20 | 20/20 | `20261011T213929Z_gguf_eval_hard20_llamacpp_q38_27b_q8_0` (b377beba5) |
| Qwen3.8-27B | LM Studio MLX 8-bit | **89/120** | 17/20 | 12/20 | 5/20 | 18/20 | 17/20 | 20/20 | `20261012T025455Z_gguf_eval_hard20_lms_q38_27b_8bit` (a223a841f) |

Appaiamento storico per elemento (esecuzioni dell’11–12 ottobre; elementi valutati da entrambi i sistemi): A = il file SUC1, B = l'altro sistema.

| modello | A (SUC1) | B | elementi | entrambi corretti | solo A | solo B | entrambi sbagliati | stessa risposta | p McNemar | esclusi |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | SUC tcq (96.9 GB, residente) | ds4 IQ2_XXS mix | 120 | 98 | 5 | 8 | 9 | 105 | 0.581 | 0 |
| DeepSeek-V4-Flash | SUC tcq calib v2 (96.9 GB, residente) | SUC tcq calib v1 (96.9 GB) | 119 | 95 | 6 | 7 | 11 | 101 | 1 | 1 |
| DeepSeek-V4-Flash | SUC tcq calib v2 (96.9 GB, residente) | ds4 IQ2_XXS mix | 119 | 96 | 5 | 9 | 9 | 102 | 0.424 | 1 |
| GLM-5.3-Flash | SUC tcq (expert 2.45 bpw, residente) | ds4 Q2 | 120 | 87 | 12 | 6 | 15 | 92 | 0.238 | 0 |
| Qwen3.8-Flash-Next | SUC side8 | ds4 Q4 | 120 | 94 | 3 | 5 | 18 | 101 | 0.727 | 0 |
| Qwen3.8-27B | SUC side8 (16.4 GB) | GGUF Q8_0 (llama.cpp llama-server) | 120 | 88 | 5 | 4 | 23 | 100 | 1 | 0 |
| Qwen3.8-27B | SUC side8 (16.4 GB) | LM Studio MLX 8-bit | 120 | 85 | 8 | 4 | 23 | 96 | 0.388 | 0 |

Per sottoinsieme (entrambi corretti / solo A / solo B / entrambi sbagliati):

| modello | B | GSM8K | MATH | MMLU-Pro | ARC-Challenge | BBH | LongCtx |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | ds4 IQ2_XXS mix | 18/1/0/1 | 12/0/5/3 | 14/2/2/2 | 18/1/0/1 | 18/1/0/1 | 18/0/1/1 |
| DeepSeek-V4-Flash | SUC tcq calib v1 (96.9 GB) | 19/0/0/1 | 12/4/0/4 | 10/0/6/4 | 19/0/0/1 | 18/1/1/0 | 17/1/0/1 |
| DeepSeek-V4-Flash | ds4 IQ2_XXS mix | 18/1/0/1 | 15/1/2/2 | 10/0/6/4 | 18/1/0/1 | 17/2/1/0 | 18/0/0/1 |
| GLM-5.3-Flash | ds4 Q2 | 19/0/0/1 | 14/2/0/4 | 5/6/3/6 | 17/1/0/2 | 16/1/2/1 | 16/2/1/1 |
| Qwen3.8-Flash-Next | ds4 Q4 | 17/0/1/2 | 14/1/2/3 | 6/0/1/13 | 19/1/0/0 | 18/1/1/0 | 20/0/0/0 |
| Qwen3.8-27B | GGUF Q8_0 (llama.cpp llama-server) | 17/0/1/2 | 12/1/1/6 | 6/2/0/12 | 18/1/0/1 | 16/1/1/2 | 19/0/1/0 |
| Qwen3.8-27B | LM Studio MLX 8-bit | 17/0/0/3 | 11/2/1/6 | 4/4/1/11 | 18/1/0/1 | 16/1/1/2 | 19/0/1/0 |

Sottoinsiemi evidenziati (la domanda dell'esecuzione; elementi appaiati valutati da entrambi):

- **DeepSeek-V4-Flash, MATH: SUC tcq calib v2 (96.9 GB, residente) 16/20 vs SUC tcq calib v1 (96.9 GB) 12/20** (solo A 4, solo B 0, p McNemar 0.125); solo A: `math500/intermediate_algebra/1350`, `math500/intermediate_algebra/582`, `math500/geometry/283`, `math500/geometry/547`; solo B: nessuno.
- **DeepSeek-V4-Flash, MATH: SUC tcq calib v2 (96.9 GB, residente) 16/20 vs ds4 IQ2_XXS mix 17/20** (solo A 1, solo B 2, p McNemar 1); solo A: `math500/intermediate_algebra/582`; solo B: `math500/counting_and_probability/525`, `math500/intermediate_algebra/1544`.

Elementi discordanti:

- DeepSeek-V4-Flash, SUC tcq (96.9 GB, residente) vs ds4 IQ2_XXS mix: solo A: `gsm8k/test-1165`, `mmlu_pro/9373`, `mmlu_pro/4270`, `arc_challenge/MCAS_2004_8_7`, `bbh/causal_judgement-1`; solo B: `math500/counting_and_probability/525`, `math500/intermediate_algebra/1544`, `math500/intermediate_algebra/1350`, `math500/geometry/283`, `math500/geometry/547`, `mmlu_pro/11613`, `mmlu_pro/4271`, `longctx/3hop_reverse_distract-31k-r2`.
- DeepSeek-V4-Flash, SUC tcq calib v2 (96.9 GB, residente) vs SUC tcq calib v1 (96.9 GB): solo A: `math500/intermediate_algebra/1350`, `math500/intermediate_algebra/582`, `math500/geometry/283`, `math500/geometry/547`, `bbh/causal_judgement-120`, `longctx/3hop_reverse_distract-31k-r2`; solo B: `mmlu_pro/1695`, `mmlu_pro/9135`, `mmlu_pro/3817`, `mmlu_pro/9373`, `mmlu_pro/4270`, `mmlu_pro/1416`, `bbh/date_understanding-118`.
- DeepSeek-V4-Flash, SUC tcq calib v2 (96.9 GB, residente) vs ds4 IQ2_XXS mix: solo A: `gsm8k/test-1165`, `math500/intermediate_algebra/582`, `arc_challenge/MCAS_2004_8_7`, `bbh/causal_judgement-120`, `bbh/causal_judgement-1`; solo B: `math500/counting_and_probability/525`, `math500/intermediate_algebra/1544`, `mmlu_pro/1695`, `mmlu_pro/9135`, `mmlu_pro/3817`, `mmlu_pro/11613`, `mmlu_pro/1416`, `mmlu_pro/4271`, `bbh/date_understanding-118`.
- GLM-5.3-Flash, SUC tcq (expert 2.45 bpw, residente) vs ds4 Q2: solo A: `math500/intermediate_algebra/582`, `math500/geometry/547`, `mmlu_pro/1695`, `mmlu_pro/9135`, `mmlu_pro/9373`, `mmlu_pro/9591`, `mmlu_pro/4270`, `mmlu_pro/8080`, `arc_challenge/TIMSS_2011_8_pg15`, `bbh/causal_judgement-120`, `longctx/3hop_reverse_distract-31k`, `longctx/3hop_reverse_distract-31k-r1`; solo B: `mmlu_pro/8195`, `mmlu_pro/1416`, `mmlu_pro/12204`, `bbh/causal_judgement-18`, `bbh/causal_judgement-118`, `longctx/sum_2x2hop-16k-r3`.
- Qwen3.8-Flash-Next, SUC side8 vs ds4 Q4: solo A: `math500/counting_and_probability/1003`, `arc_challenge/Mercury_7032428`, `bbh/causal_judgement-18`; solo B: `gsm8k/test-1067`, `math500/geometry/465`, `math500/geometry/283`, `mmlu_pro/4270`, `bbh/causal_judgement-118`.
- Qwen3.8-27B, SUC side8 (16.4 GB) vs GGUF Q8_0 (llama.cpp llama-server): solo A: `math500/geometry/283`, `mmlu_pro/3817`, `mmlu_pro/12204`, `arc_challenge/TIMSS_2003_8_pg87`, `bbh/causal_judgement-120`; solo B: `gsm8k/test-353`, `math500/geometry/547`, `bbh/causal_judgement-118`, `longctx/3hop_reverse_distract-31k`.
- Qwen3.8-27B, SUC side8 (16.4 GB) vs LM Studio MLX 8-bit: solo A: `math500/geometry/465`, `math500/geometry/283`, `mmlu_pro/9135`, `mmlu_pro/3817`, `mmlu_pro/1416`, `mmlu_pro/12204`, `arc_challenge/TIMSS_2003_8_pg87`, `bbh/causal_judgement-120`; solo B: `math500/geometry/547`, `mmlu_pro/8625`, `bbh/causal_judgement-118`, `longctx/3hop_reverse_distract-31k`.
