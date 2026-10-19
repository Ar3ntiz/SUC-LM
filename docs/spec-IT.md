# Semantic Universal Compression (SUC)

Versione inglese: [spec.md](spec.md)

**Specifica di implementazione per un agente di programmazione LLM**  
**Hardware di sviluppo:** Mac Studio Apple Silicon con chip Ultra e ≥ 128 GB di memoria unificata — M1 Ultra e successivi (M2 Ultra, M3 Ultra, … fino all'attuale M5 Ultra). Le misure di riferimento del progetto sono state prese su un M1 Ultra 128 GB, la configurazione supportata più vecchia; i chip Ultra più recenti (più banda e più core GPU) dovrebbero essere più veloci, ma qui non sono misurati.  
**Obiettivo primario:** ricercare e prototipare una rappresentazione del modello che minimizzi storage/memoria preservando il comportamento del modello e consentendo l'inferenza diretta dalla rappresentazione compressa, senza prima ricostruire l'intera matrice dei pesi FP16.

---

## 1. Missione

Costruire un sistema di ricerca che verifichi questa ipotesi:

> Un LLM può potenzialmente essere rappresentato in modo molto più compatto rispetto alla quantizzazione scalare convenzionale a pochi bit, combinando sensibilità semantica/data-aware, quantizzazione vettoriale/a codebook, allocazione adattiva dei bit, preservazione di residui/outlier, entropy coding opzionale e inferenza fusa consapevole dell'hardware.

Il sistema deve ottimizzare il **comportamento del modello per bit memorizzato**, non semplicemente l'errore numerico di ricostruzione dei pesi.

Formalmente, dato il modello originale `M` e il modello compresso `Mq`:

\[
\min \; Size(M_q)
\]

soggetto a:

\[
D_{semantic}(M,M_q) \le \epsilon
\]

e, opzionalmente:

\[
Latency(M_q) \le L_{max}
\]

\[
PeakMemory(M_q) \le M_{max}
\]

\[
Energy(M_q) \le E_{max}
\]

Il progetto deve distinguere quattro obiettivi diversi:

1. **Compressione lossless:** ricostruzione esatta dei pesi originali.
2. **Quantizzazione lossy:** i pesi vengono approssimati.
3. **Compressione di storage:** i pesi compressi vengono espansi prima dell'inferenza.
4. **Compressione compute-native:** il kernel di inferenza consuma direttamente la rappresentazione compressa.

L'obiettivo di ricerca primario è il n. 4.

---

# 0. Decisioni di progetto (emendamenti, 2026-09-30)

Queste decisioni sono state concordate con il responsabile del progetto dopo la revisione della specifica originale. **Dove sono in conflitto con una sezione successiva, prevale questa sezione.** Ogni sezione successiva interessata riporta una nota `(vedi D#)`.

### D1 — Due track paralleli invece di un'unica sequenza

L'ordine originale (l'ex §51, poi riscritto secondo D1) richiedeva fasi di compressione dell'intero modello prima della milestone del kernel a singolo layer, in contraddizione con il gate di §43. Il lavoro è diviso in:

- **Quality track** — analisi, calibrazione, codebook, residui, allocazione. Valutato decodificando in Python ed eseguendo i pesi *ricostruiti* in MLX. Perplessità/KL sull'intero modello sono ammesse qui fin dall'inizio; misurano la qualità della rappresentazione, non il runtime.
- **Kernel track** — esecuzione diretta sui dati compressi. Parte non appena esiste un codebook vettoriale di base (Fase 4). Il gate di §43 sul singolo layer lineare si applica a **questo track**: nessuna esecuzione diretta dell'intero transformer finché il benchmark sul singolo layer non è solido.

I due track si incontrano alla Fase 15 (integrazione nel transformer).

### D2 — Runtime C++ rinviato, non abbandonato (completato)

> **Stato al 14/10/2026: consegnato.** Il runtime C++ è stato costruito alla fine del kernel track come previsto ed è
> ora il percorso principale di inferenza: lettore / decoder SUC1, kernel Metal generati da quelli Python, grafi per
> Qwen3 denso / MoE, Qwen3.8 (`qwen3_5`, `qwen4_exp`), DeepSeek-V4 e GLM-5.3, tokenizer, prefill a blocchi, streaming
> degli esperti da SSD e il server compatibile OpenAI `suc1-server` ([cpp/README-IT.md](../cpp/README-IT.md)). È anche
> il motore dell'app macOS SUC Studio (`SUCStudio/`). "Rinviato" qui sotto documenta il piano originale (30/09/2026).

La prototipazione dei kernel usa kernel Metal custom di MLX (`mx.fast.metal_kernel`) chiamati da Python. Il runtime C++ standalone (§18, Fase 11) resta un deliverable obbligatorio ma si sposta alla fine del kernel track, dopo la validazione dei kernel fusi. La baseline forte a 4 bit su Apple Silicon è la quantizzazione affine di MLX stessa (`mx.quantize` / matmul quantizzata fusa), oltre ad AWQ/GPTQ dove gli strumenti lo consentono.

### D3 — Incoherence processing e fine-tuning dei codebook sono componenti di prima classe

I metodi noti sotto i 3 bpw dipendono da due ingredienti che l'architettura originale ometteva:

- **Incoherence processing** — rotazioni Hadamard randomizzate (o ortogonali simili) di pesi/attivazioni prima della quantizzazione (QuIP/QuIP#), per sopprimere gli outlier e rendere i blocchi più "adatti al codebook". La rotazione deve essere fusa nel kernel oppure assorbita nei layer adiacenti; il suo costo a runtime viene misurato.
- **Fine-tuning di codebook / rappresentazione** — dopo il fitting iniziale, si ottimizzano i codebook (e opzionalmente scale/parametri di norm) rispetto a un obiettivo semantico sui dati di calibrazione (fine-tuning a blocchi/globale stile AQLM, aggiornamenti discreti stile PV-tuning).

Entrambi sono componenti opzionali e attivabili, ed entrambi hanno una riga nella tabella di ablation (§26): `SUC - incoherence`, `SUC - fine-tuning`. Per target pari o inferiori a 3 bpw le baseline di riferimento devono includere almeno uno tra **AQLM** e **QuIP#** (numeri pubblicati sullo stesso modello/valutazione, oppure un'esecuzione locale), non solo un metodo a 4 bit.

> **Stato al 14/10/2026:** l'incoherence processing è nella ricetta finale come rotazione di Hadamard randomizzata solo sul lato di ingresso; il fine-tuning dei codebook è stato implementato (Q6) ma è opzionale e non usato nei file scelti; entrambi hanno righe di ablation ([final_report.md](final_report.md) §3.6). AQLM / QuIP#: solo numeri pubblicati (vedi D7).

### D4 — La dimensione del codebook è vincolata dall'hardware fin dal primo giorno

A batch=1 il decode è limitato dalla banda di memoria (M1 / M2 Ultra ≈ 800 GB/s teorici, M3 Ultra ≈ 819 GB/s, di più sui chip successivi). Un codebook che il kernel legge per ogni tile di output deve stare nella threadgroup memory (32 KB per threadgroup sulle GPU Apple). Limite predefinito: **byte del codebook = K × G × bytes_per_entry ≤ 16 KB** per ogni codebook usato attivamente da un threadgroup (es. K=256, G=8, FP16 = 4 KB). Le configurazioni che superano il limite possono essere studiate nel quality track ma sono marcate come "non eseguibili" (not execution-viable), a meno che un kernel non dimostri il contrario.

### D5 — La sensibilità usa prima proxy economici

I test di perturbazione per blocco (§9 T2.3) non scalano a 7B+. La sensibilità viene stimata prima con proxy economici accumulati durante la calibrazione:

- momento secondo dell'input diagonale / a blocchi `diag(XᵀX)` (proxy dell'Hessiana stile GPTQ) per ogni layer lineare;
- opzionalmente Fisher empirica (gradienti al quadrato) della loss rispetto ai pesi.

I test di perturbazione reali vengono eseguiti solo su un **sottoinsieme campionato** di tensori/blocchi, per validare e calibrare il proxy (riportare la correlazione di rango tra proxy e ΔKL misurato). Il proxy viene usato per l'allocazione solo se tale correlazione è riportata.

### D6 — La granularità dell'allocazione segue il kernel

Mescolare rappresentazioni *all'interno* di un tensore con granularità fine a blocchi rompe la regolarità di cui il kernel ha bisogno. Pertanto:

- la **rappresentazione base** (famiglia di codebook, G, K, bit) si sceglie per **tensore** o per **row-group** (righe di output contigue, di dimensione multipla del tile del kernel);
- le decisioni **a grana fine** (per blocco / per elemento) sono limitate a residui e outlier (Fase 6), che il kernel gestisce come passata sparsa separata.

### D7 — Il modello è un input di configurazione, non una costante nel codice

Il modello da comprimere viene selezionato tramite configurazione (repo id Hugging Face o percorso locale + revisione). Il codice non deve fissare nel codice un modello o nomi di layer; i dettagli specifici dell'architettura (quali layer lineari sono idonei, naming, layout dell'attention) stanno in un piccolo adapter per famiglia. Esistono due ruoli:

- **dev model** — piccolo (≈0.5–2B), usato per iterazione rapida e test a scala CI;
- **target model** — il modello ottimizzato seriamente (primo target serio 7B/8B, §7 T0.1).

Piano attuale (2026-09-30), due target in sequenza:

| Fase | Modello | Ruolo |
|---|---|---|
| Sviluppo / test | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` | dev: iterazione nell'ordine dei minuti, test automatici |
| Fondamenta, quality track Q1–Q5, gate kernel §43 | `Qwen/Qwen3-8B` | target 1 (denso) |
| Da Q6–Q7 (fine-tuning, allocatore) in poi | `Qwen/Qwen3-30B-A3B` | target 2 (MoE, ~3B attivi) |
| Dopo il progetto (eseguito dal responsabile) | ≈27B denso, poi modelli densi e MoE più grandi (fino a ≈180B MoE) | fuori dall'ambito del progetto |

La Fase 0 include già il caricamento di `Qwen3-30B-A3B` tramite l'adapter e le sue baseline BF16 / MLX-4-bit, così D8 viene messo alla prova presto su un MoE reale. Per Qwen3 sono pubblicati pochi numeri AQLM/QuIP#, quindi quelle baseline vanno eseguite localmente.

Le famiglie supportate sono quelle caricabili da `mlx-lm`; aggiungere una famiglia significa aggiungere un adapter più un test numerico di conversione identità. Ogni risultato registra l'esatto model id, la revisione e l'hash dei pesi (§7 T0.4).

---

> **Stato al 14/10/2026:** oltre ai due target, la pipeline ha compresso Qwen3.8-27B (denso) e i MoE Qwen3.8-Flash-Next (177 B), DeepSeek-V4-Flash (284 B) e GLM-5.3-Flash (313 B) sullo Studio da 128 GB — file scelti in [compression_report-IT.md](compression_report-IT.md) §1. Qwen3-1.7B non è mai stato usato (modello di sviluppo: Qwen3-0.6B). AQLM / QuIP# non sono stati eseguiti in locale: solo numeri pubblicati ([comparison_published.md](comparison_published.md)).

### D8 — Scalabilità integrata fin dalla Fase 0

> **Stato al 14/10/2026:** la pipeline ha poi compresso Qwen3.8-27B e tre modelli MoE da 177–313 B (Qwen3.8-Flash-Next,
> DeepSeek-V4-Flash, GLM-5.3-Flash) un layer alla volta sullo Studio da 128 GB
> ([compression_report-IT.md](compression_report-IT.md)).

Il progetto ha come target Qwen3-8B (denso) e Qwen3-30B-A3B (MoE) e termina con il runtime C++ e il report finale (decisione 2026-10-01); la scalatura a ≈27B denso e oltre (fino a ≈180B MoE, che in BF16 non sta in 128 GB) viene fatta dal responsabile dopo il progetto con la pipeline finita — quindi la pipeline deve girare senza supervisione dall'inizio alla fine su tali modelli. La pipeline non deve richiedere una riprogettazione per arrivarci:

- **Pipeline in layer streaming.** I pesi vengono letti in modo lazy (mmap degli shard safetensors), e calibrazione, stima della sensibilità e compressione procedono **un blocco transformer alla volta** (passata sequenziale stile GPTQ: propagare le attivazioni di calibrazione attraverso il blocco già processato, poi liberarlo). Nessun passo può richiedere l'intero modello residente in BF16/FP16. Il picco di memoria di ogni stadio della pipeline viene misurato e riportato.
- **Interfaccia adapter di architettura pronta per MoE e attention non standard.** L'adapter enumera i tensori lineari idonei con i loro ruoli (`attn_q/k/v/o`, `mlp_gate/up/down`, `expert[e].*`, `shared_expert.*`, `router`, proiezioni di linear attention, …). I pesi di router/gating sono per default ad alta precisione. I conteggi di attivazione per expert vengono raccolti durante la calibrazione ed esposti all'allocatore; la dimensione della calibrazione deve essere sufficiente per gli expert instradati raramente (riportare il minimo di token visti per expert).
- **Solo sorgente a precisione piena.** L'input è il checkpoint Hugging Face originale (safetensors, BF16/FP16) più il tokenizer. Gli artefatti già quantizzati (GGUF, MLX 4-bit, AWQ/GPTQ, FP8) non sono mai usati come input — solo come baseline di confronto. Gli hash degli shard sorgente vengono verificati e registrati.

---

# 2. Idea di base

L'intuizione originale dell'utente era:

```text
weight value
    ↓
small index
    ↓
lookup table
    ↓
value
```

Una lookup table scalare non batte da sola una quantizzazione forte a 4 bit. Una tabella di 1000 valori scalari richiede circa 10 bit per indice scalare:

\[
\lceil \log_2(1000) \rceil = 10
\]

La generalizzazione promettente è:

```text
block of many weights
        ↓
one compact code/index
        ↓
codebook vector
```

Per esempio, se un indice a 8 bit seleziona un vettore di 16 pesi:

\[
8/16 = 0.5\;bits/weight
\]

prima dell'overhead di codebook, scale, residui, allineamento e metadati.

Questa è quantizzazione vettoriale/a codebook.

La domanda di ricerca cruciale è quindi:

> Quanto comportamento del modello può essere conservato per bit memorizzato, e quella rappresentazione compressa può essere usata direttamente dal kernel di inferenza?

---

# 3. Che cosa è e che cosa non è questo progetto

## 3.1 È

- un prototipo di ricerca;
- un framework di compressione low-bit/vettoriale/a codebook;
- un allocatore di bit consapevole della semantica;
- un esperimento sui kernel di inferenza;
- una piattaforma di benchmark per i compromessi memoria/qualità/latenza.

## 3.2 Non è

- un'affermazione che sia già stata scoperta una nuova primitiva di compressione;
- un'affermazione che tutti gli LLM possano essere compressi a 1–2 bpw senza un degrado significativo;
- un'affermazione che l'entropy coding renda automaticamente l'inferenza più veloce;
- un'eredità diretta del teorema dell'Universal Portfolio di Cover da parte delle reti neurali.

L'Universal Portfolio e i concetti di teoria dell'informazione sono l'ispirazione per il **principio di allocazione e universalizzazione**, mentre la quantizzazione vettoriale, la quantizzazione low-bit degli LLM e l'inferenza centrata sulle LUT hanno già un consistente lavoro precedente.

L'eventuale novità di ricerca deve venire dalla **combinazione e dall'obiettivo**, non dal fingere che i codebook o la quantizzazione a 2 bit siano nuovi.

---

# 4. Scelta tecnologica per Apple Silicon Ultra (M1 Ultra e successivi)

## 4.1 Stack di linguaggi primario

Usare uno stack ibrido.

### Python
Da usare per:

- caricamento del modello;
- dati di calibrazione;
- statistiche di pesi/attivazioni;
- training dei codebook;
- ottimizzazione;
- orchestrazione degli esperimenti;
- reportistica dei benchmark;
- strumenti di serializzazione.

### C++ (rinviato nel piano, consegnato — vedi D2)
Da usare per:

- runtime;
- rappresentazione packed;
- layout di memoria;
- kernel di riferimento a basso livello;
- eventuale integrazione nell'inferenza.

### Metal Shading Language
Da usare per:

- kernel GPU;
- fusione lookup/decode/GEMM;
- esecuzione diretta sui pesi compressi.

### MLX
Da usare come primo framework di ricerca nativo Apple. Apple Silicon ha memoria unificata CPU/GPU e MLX è progettato specificamente attorno a questo modello. Ciò lo rende un ottimo ambiente di prototipazione per rappresentazioni compresse che devono essere visibili a CPU e GPU senza la consueta semantica di copia tra dispositivi.

### llama.cpp / GGML

> **Stato al 14/10/2026:** usato solo come baseline di confronto (llama.cpp, ds4, LM Studio); nessun livello di interoperabilità è stato costruito (§24).
Da usare più avanti come target di interoperabilità/runtime, non come primo ambiente di sviluppo.

L'attuale progetto llama.cpp supporta Apple Silicon tramite Metal e offre un'ampia gamma di formati low-bit, il che lo rende un importante target di confronto finale.

---

# 5. Architettura

```text
                         ORIGINAL MODEL
                              │
                              ▼
                    ┌────────────────────┐
                    │ Model Analyzer     │
                    └─────────┬──────────┘
                              │
               ┌──────────────┼─────────────────┐
               ▼              ▼                 ▼
         weight stats    activation stats   sensitivity
               │              │                 │
               └──────────────┼─────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Semantic Allocator │
                    └─────────┬──────────┘
                              │
                      per-block budgets
                              │
                              ▼
                    ┌────────────────────┐
                    │ Codebook Builder   │
                    └─────────┬──────────┘
                              │
                     codebooks + indices
                              │
                ┌─────────────┼────────────┐
                ▼             ▼            ▼
            residuals      outliers     metadata
                │             │            │
                └─────────────┼────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Optional Entropy   │
                    │ Coding             │
                    └─────────┬──────────┘
                              │
                              ▼
                       SUC Model File
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
        Reference decoder              Direct runtime
        (correctness)                  (C++ / Metal)
                │                            │
                ▼                            ▼
          reconstructed                fused inference
             weights                   on compressed data
```

---

# 6. Struttura del repository

> **Stato al 14/10/2026:** la struttura realizzata è diversa: runtime C++ e sorgenti Metal in `cpp/`, kernel Metal Python in `suc/kernels/`, runtime in `suc/runtime/`, pipeline in `suc/pipeline/`, code in `suc/orchestrate/`; test in `tests/unit` e `tests/model`; i documenti includono `format.md`, `benchmarks.md`, `final_report.md`, `compression_report.md`; l'app macOS è `SUCStudio/`. Nessun file LICENSE per ora.

```text
SUC-LM/
├── README.md
├── LICENSE
├── pyproject.toml
├── CMakeLists.txt
├── docs/
│   ├── architecture.md
│   ├── format.md
│   ├── algorithms.md
│   ├── benchmarks.md
│   └── experiments.md
├── suc/
│   ├── io/
│   ├── models/
│   ├── analysis/
│   ├── quant/
│   ├── allocation/
│   ├── entropy/
│   ├── evaluation/
│   └── cli/
├── runtime/
│   ├── cpp/
│   └── metal/
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── model/
│   ├── runtime/
│   └── regression/
├── experiments/
│   ├── configs/
│   ├── scripts/
│   └── results/
└── benchmarks/
```

---

# 7. Fase 0 — Baseline e riproducibilità

## Obiettivo

Stabilire un riferimento affidabile prima di implementare SUC.

## Attività

### T0.1 — loader del modello

Supportare almeno un transformer decoder-only di Hugging Face.

Iniziare con un modello da 1B–8B, così gli esperimenti sono veloci. Usare un modello 7B/8B come primo target serio. Solo dopo aver raggiunto la stabilità, scalare a modelli di classe 30B/70B.

### T0.2 — runtime di baseline

Misurare:

- numero di parametri;
- dimensioni grezze dei tensori;
- dimensione FP16;
- latenza di inferenza di baseline;
- token/s;
- throughput del prompt;
- picco di memoria unificata;
- tempo di caricamento del modello.

### T0.3 — rappresentazioni di baseline

> **Stato al 14/10/2026:** il riferimento forte a 4 bit è MLX affine 4-bit, più i K-quant di llama.cpp e i GGUF di ds4; AWQ / GPTQ non sono stati eseguiti in locale.

Confrontare come minimo:

- FP16;
- INT8;
- INT4;
- un riferimento a 4 bit esistente e forte, come AWQ o GPTQ, dove architettura/strumenti lo consentono.

### T0.4 — harness di benchmark deterministico

Tutte le esecuzioni devono registrare:

```json
{
  "model": "...",
  "model_hash": "...",
  "git_commit": "...",
  "dataset_hash": "...",
  "seed": 12345,
  "runtime": "...",
  "timestamp": "..."
}
```

## Accettazione

- la conversione identità produce un errore numerico trascurabile;
- l'harness di benchmark produce JSON leggibile da macchina;
- tutti gli esperimenti sono riproducibili da un file di configurazione.

---

# 8. Fase 1 — Analisi della struttura dei pesi

## Obiettivo

Determinare se il modello ha abbastanza struttura locale per la compressione vettoriale/a codebook.

Non darlo per scontato.

## Attività

### T1.1 — statistiche scalari

Per ogni tensore calcolare:

- min/max;
- media/deviazione standard;
- quantili;
- istogramma;
- curtosi;
- tassi di outlier.

### T1.2 — statistiche a blocchi

Testare le dimensioni di blocco:

```text
8
16
32
64
128
```

Per ogni blocco raccogliere:

- norma L1/L2;
- varianza;
- valore assoluto massimo;
- sparsità;
- media;
- direzione normalizzata.

### T1.3 — similarità tra blocchi

Stimare la similarità a coppie/di clustering usando:

- distanza euclidea;
- distanza euclidea normalizzata;
- distanza coseno;
- opzionalmente distanza pesata sulle attivazioni.

### T1.4 — analisi di occupazione

Per i codebook candidati, misurare:

- numero di codici occupati;
- distribuzione a coda lunga;
- ripetizione;
- distanza media dal codebook più vicino;
- distanza p95/p99.

### Deliverable

Generare un report:

```text
layer/tensor
block size
number of blocks
candidate K
entropy of assignments
reconstruction error
```

Questa fase decide se vale la pena perseguire la codifica vettoriale e quali dimensioni di blocco sono plausibili.

---

# 9. Fase 2 — Calibrazione e sensibilità semantica

*(vedi D5: prima i proxy, test di perturbazione su un sottoinsieme campionato)*

## Obiettivo

Misurare quali errori di compressione contano davvero per il modello.

L'MSE sui pesi da solo non è sufficiente.

## Attività

### T2.1 — dataset di calibrazione

Iniziare con circa 1k–10k sequenze, configurabili via YAML/JSON.

Includere più tipi di task:

- linguaggio naturale;
- programmazione;
- matematica/ragionamento;
- instruction following.

Non usare il set di benchmark finale per il tuning.

### T2.2 — statistiche delle attivazioni

Per i layer lineari importanti raccogliere statistiche in streaming dei loro input:

- media;
- varianza;
- magnitudine massima;
- scala per canale;
- proxy del momento secondo/dell'Hessiana dove praticabile.

Non salvare i tensori di attivazione completi a meno che non servano per uno specifico esperimento.

### T2.3 — test di perturbazione locale

Per ogni layer/blocco:

1. creare una versione compressa;
2. eseguire i campioni di calibrazione;
3. misurare la variazione nell'output del layer;
4. misurare la divergenza dei logit;
5. misurare la variazione di cross-entropy/perplessità.

### T2.4 — punteggio di sensibilità

Creare un punteggio come:

\[
S_i = \frac{\Delta D_{semantic}}{\Delta E_i}
\]

dove `Ei` è l'errore di rappresentazione iniettato.

Il punteggio esatto può essere raffinato in seguito. La prima implementazione deve essere trasparente e misurabile.

### Deliverable

Produrre una mappa di sensibilità per tensore/blocco:

```json
{
  "layer.12.mlp.down_proj": {
    "block_size": 16,
    "sensitivity_mean": 0.013,
    "sensitivity_p95": 0.041
  }
}
```

---

# 10. Fase 3 — Baseline con codebook scalare

## Obiettivo

Implementare il concetto originale di lookup table dell'utente come baseline controllata.

Rappresentazione:

```text
codebook = [K scalar values]
indices  = one index per weight
```

Test:

```text
K = 4, 8, 16, 32, 64, 256, 1024
```

## Attività

- implementare k-means e k-means++;
- supportare codebook per tensore e per blocco;
- supportare seed deterministici;
- misurare l'errore di ricostruzione;
- misurare la perplessità;
- misurare l'accuratezza sui task;
- misurare la dimensione compressa incluso l'overhead del codebook;
- opzionalmente applicare entropy coding agli indici.

## Risultato atteso

Si tratta principalmente di un controllo di sanità. Non ci si aspetta che batta in modo drastico un'implementazione forte a 4 bit.

---

# 11. Fase 4 — Quantizzazione con codebook vettoriale

*(vedi D4: limite di dimensione del codebook; D3: incoherence processing)*

## Obiettivo

Rappresentare gruppi di pesi con un unico codice.

Per il blocco `b`:

\[
\hat W_b = C[index_b]
\]

dove `C` contiene voci vettoriali.

## Matrice di test

```text
block size G = 4, 8, 16, 32
K = 16, 32, 64, 128, 256, 512, 1024
codebook dtype = FP16, BF16 where supported
```

## Attività

1. K-means vettoriale.
2. Strategie di inizializzazione.
3. Raffinamento del codebook.
4. Distanza opzionale pesata sulle attivazioni.
5. Calcolatore della dimensione compressa con tutti gli overhead.
6. Implementazione del decoder.

## Output richiesto

Per ogni configurazione riportare:

- bpw nominali degli indici;
- bpw effettivi;
- dimensione del codebook;
- errore di ricostruzione;
- loss di calibrazione;
- loss held-out;
- perplessità del modello.

---

# 12. Fase 5 — Rappresentazione additiva / multi-codebook

> **Stato al 14/10/2026:** implementato e misurato; la VQ a bassa dimensione (G=2, K=256) ha battuto i codebook additivi a ogni rate provato, quindi l'additivo è opzionale e non usato nella ricetta finale ([final_report.md](final_report.md) §2).

## Obiettivo

Evitare di richiedere un unico codebook enorme.

Rappresentare un blocco come:

\[
\hat W_b = C_1[i_1] + C_2[i_2] + ... + C_N[i_N]
\]

Esempio:

```text
codebook 1: 2-bit index
codebook 2: 2-bit index
codebook 3: 2-bit index
```

## Attività

### T5.1 — fitting greedy dei residui

```python
residual = block
approx = 0
for codebook in codebooks:
    idx = nearest(codebook, residual)
    approx += codebook[idx]
    residual -= codebook[idx]
```

### T5.2 — raffinamento congiunto opzionale

Una volta che la versione greedy funziona, ottimizzare tutti i codebook congiuntamente dove il costo computazionale è giustificato.

### T5.3 — studio di scalatura

Misurare:

- 1 codebook;
- 2 codebook;
- 3 codebook;
- 4 codebook.

Tracciare il grafico bit/peso vs qualità semantica.

---

# 13. Fase 6 — Rappresentazione di residui e outlier

## Obiettivo

Spendere bit extra solo dove hanno un alto valore semantico.

Forma generale:

\[
W = W_{base} + R
\]

dove `R` è sparso, low-bit, INT8 o FP16 per eccezioni selezionate.

## Modalità di residuo candidate

```text
none
sparse top-k FP16
sparse top-k INT8
small residual codebook
per-block high-precision exception
```

## Attività

- calcolare il residuo dopo l'approssimazione con codebook;
- ordinare gli elementi del residuo per impatto o magnitudine;
- codificare solo una frazione configurabile;
- fare benchmark del compromesso qualità/dimensione;
- confrontare la selezione degli outlier basata sulla magnitudine con quella basata sulla sensibilità semantica.

Il risultato preferito è qualcosa come:

```text
98–99% of representation → very low-bit base
1–2% → extra precision
```

ma la frazione effettiva deve essere misurata.

---

# 14. Fase 7 — Allocazione adattiva semantica dei bit

*(vedi D6: rappresentazione base per tensore/row-group; grana fine solo per i residui)*

## Obiettivo

Questo è il componente centrale di "Universal Allocation".

Invece di fissare la stessa rappresentazione per ogni blocco, scegliere la rappresentazione che produce la qualità migliore sotto un budget di memoria globale.

## Rappresentazioni candidate per blocco

Esempi:

```text
FP16
INT8
INT4
INT3
INT2
scalar-codebook
vector-codebook
multi-codebook
vector-codebook + residual
```

Ogni candidato deve avere, misurati/stimati:

```text
size
semantic loss
latency
```

## Problema di ottimizzazione

Sia `a_i` la scelta di una rappresentazione per il blocco `i`:

\[
\min_{a_i} \sum_i D_i(a_i)
\]

soggetto a:

\[
\sum_i Size_i(a_i) \le B
\]

e, opzionalmente:

\[
Latency(a) \le L_{max}
\]

## Primo allocatore

Implementare un allocatore greedy a valore marginale.

Per ogni possibile upgrade:

\[
Value = \frac{\Delta Quality}{\Delta Bytes}
\]

Ordinare per valore e spendere il budget disponibile.

## Allocatore avanzato

Sperimentare aggiornamenti in stile multiplicative-weights tra le scelte di rappresentazione candidate.

Si tratta di un'analogia sperimentale con l'Universal Portfolio / aggregazione di expert. **Non** affermare che il teorema di Cover dimostri direttamente l'ottimalità in questo contesto.

---

# 15. Fase 8 — Entropy coding e codifica universale

## Obiettivo

Sfruttare le distribuzioni non uniformi degli indici del codebook.

Calcolare:

\[
H(X) = -\sum_x p(x)\log_2 p(x)
\]

Testare:

- indici a larghezza fissa;
- Huffman;
- ANS/rANS;
- altro entropy coding pratico, se giustificato.

## Importante

Trattare l'entropy coding innanzitutto come **ottimizzazione di storage/banda**.

Non inserire automaticamente un bitstream a lunghezza variabile nel percorso caldo dell'inferenza.

Design raccomandato:

```text
SUC storage stream
       ↓
optional decode/cache
       ↓
fixed/aligned execution blocks
       ↓
Metal kernel
```

Una modalità alternativa di streaming diretto può essere testata in seguito.

---

# 16. Fase 9 — Container binario SUC

> **Stato al 14/10/2026:** implementato come `format_version` 1 — specifica a livello di byte in [format.md](format.md); `architecture_id` / `tokenizer_id` stanno nella directory JSON, non nell'header fisso; nessuna sezione di tabelle entropiche (la codifica entropica è solo per l'archiviazione).

Progettare un formato `SUC1` indipendente prima di toccare GGUF.

## Struttura del file

```text
SUC1
├── header
├── model metadata
├── tensor directory
├── codebook directory
├── codebook payloads
├── index payloads
├── residual payloads
├── optional entropy tables
└── checksum/footer
```

## Campi dell'header

Come minimo:

```text
magic = SUC1
format_version
architecture_id
tokenizer_id
number_of_tensors
number_of_codebooks
global_flags
```

`SUC1` è un identificatore di file fisso di 4 byte e non cambia mai con l'evoluzione del formato; tutta la compatibilità è governata da `format_version` (un nuovo magic solo per una riprogettazione completamente incompatibile). Decisione 2026-09-30.


## Descrittore di tensore

```text
name
shape
layout
representation_type
block_size
codebook_id
index_bits
residual_type
data_offset
compressed_size
logical_size
```

## Requisiti

- output deterministico;
- versionato;
- con checksum;
- offset a 64 bit;
- adatto al memory-mapping;
- endianness definita;
- allineamento definito;
- nessuna assunzione nascosta nei metadati.

---

# 17. Fase 10 — Decoder di riferimento in Python

Implementare un decoder volutamente semplice.

```python
model = SucModel.load("model.suc")
W = model.decode_tensor("layer.0.mlp.down_proj.weight")
```

Scopo:

- oracolo di correttezza;
- validatore del formato;
- riferimento per i test unitari;
- generatore di tensori ricostruiti per la valutazione semantica.

Può essere lento.

Il decoder di riferimento deve essere più semplice del runtime ottimizzato, così da rendere possibili controlli di correttezza indipendenti.

---

# 18. Fase 11 — Strutture dati del runtime C++

*(vedi D2: rinviato alla fine del kernel track; consegnato — [cpp/README-IT.md](../cpp/README-IT.md))*

Strutture suggerite:

```cpp
struct SucCodebook {
    uint32_t entries;
    uint32_t vector_length;
    DataType dtype;
    const void* data;
};

struct SucTensor {
    Shape shape;
    uint32_t block_size;
    RepresentationType representation;
    const uint8_t* indices;
    const uint8_t* residuals;
    const SucCodebook* codebook;
};
```

Obiettivi importanti:

- indici impacchettati in modo compatto;
- dati allineati;
- storage contiguo dei codebook;
- basso overhead di metadati;
- caricamento zero-copy o via mmap dove praticabile.

---

# 19. Fase 12 — Sviluppo dei kernel Metal

Questa è la fase critica del runtime.

## Kernel A — baseline di correttezza

Implementare:

```text
load index
→ lookup vector
→ reconstruct small block
→ dot product
```

all'interno del kernel.

Non materializzare l'intero modello in FP16.

## Kernel B — operazione a blocchi fusa

Target:

```text
activation block
+
compressed index block
+
codebook
        ↓
 fused dot product
        ↓
output
```

## Kernel C — multi-codebook

Calcolare i contributi additivi dei codebook senza materializzare i pesi completi.

## Kernel D — residuo sparso

Aggiungere il contributo del residuo solo per le voci selezionate.

---

# 20. Fase 13 — Eliminare il collo di bottiglia delle LUT

Un kernel a codebook naïf può essere troppo lento a causa dei lookup casuali e della ricostruzione ripetuta.

Testare quindi progressivamente:

### Strategia A

Lookup diretto per elemento.

### Strategia B

Lookup per blocco/vettore.

### Strategia C

Accesso vettorizzato al codebook.

### Strategia D

Precalcolare prodotti scalari parziali derivati dal codebook.

Concettualmente:

```text
activation block
      ↓
partial products for codebook entries
      ↓
index
      ↓
select partial result
```

È analogo alla direzione presa dalla ricerca sulle GEMM centrate sui codebook: l'obiettivo è evitare di ricostruire ripetutamente i valori dei pesi durante l'inferenza.

---

# 21. Fase 14 — Integrazione del layer lineare

Non riscrivere subito l'intero transformer.

Prima sostituire una sola operazione lineare:

\[
Y = XW^T
\]

con l'esecuzione SUC diretta.

Confrontare quattro implementazioni:

```text
A = FP16 GEMM
B = strong INT4 baseline
C = SUC decode-then-GEMM
D = SUC direct Metal GEMM
```

Misurare:

- errore numerico;
- latenza del kernel;
- throughput;
- uso di memoria;
- banda effettiva.

Questa è la prima milestone go/no-go significativa.

---

# 22. Fase 15 — Integrazione nel transformer

Supportare rappresentazioni miste all'interno dello stesso transformer:

```text
Layer A → FP16
Layer B → INT4
Layer C → SUC vector
Layer D → SUC + residual
```

Priorità:

1. proiezioni MLP;
2. proiezioni di attention;
3. proiezione di output;
4. embedding, se vantaggioso.

La normalizzazione e i tensori piccoli possono restare convenzionali.

L'allocatore deve poter selezionare una rappresentazione diversa per tensore/blocco.

---

# 23. Fase 16 — Integrazione con MLX

> **Stato al 14/10/2026:** esposto come comandi CLI (`suc compress`, `suc generate`, `suc eval`, `suc perf`) invece che come API Python; i backend sono il runtime Python / MLX dai codici e il runtime C++ (`suc1-generate`, `suc1-server`); nessun backend llama.cpp.

Esporre un'interfaccia Python di alto livello:

```python
compressed = suc.compress(model, config)
outputs = suc.generate(compressed, prompt, backend="metal")
```

Backend proposti:

```text
reference
mlx
metal
llama_cpp (later)
```

Evitare di duplicare la logica del modello tra i backend. Il formato compresso e il planner devono restare neutrali rispetto al backend.

---

# 24. Fase 17 — Interoperabilità con llama.cpp / GGUF

> **Stato al 14/10/2026:** non realizzato (opzionale). Non esiste un tipo di tensore SUC per GGML / GGUF; llama.cpp, ds4 e LM Studio compaiono solo come baseline di confronto ([compression_report-IT.md](compression_report-IT.md) §5–§6).

Farlo solo dopo che il runtime SUC standalone è stabile.

Possibile percorso futuro:

```text
SUC tensor type
      ↓
GGML/GGUF metadata
      ↓
CPU reference kernel
      ↓
Metal kernel
```

L'attuale modello di sviluppo di llama.cpp è rilevante qui, perché il supporto a nuovi tipi di tensore/runtime richiede conversione, gestione del grafo, strumenti di quantizzazione e test dei backend.

Non usare un'estensione GGUF come prima fonte di verità per SUC.

---

# 25. Fase 18 — Protocollo di benchmark della qualità

Ogni rappresentazione del modello deve usare lo stesso protocollo.

## Compressione

Riportare:

```text
original size
compressed size
actual bits/parameter
codebook overhead
index overhead
residual overhead
entropy metadata overhead
```

Usare:

\[
bpp = \frac{compressed\_bits}{parameter\_count}
\]

Non riportare solo i bit nominali degli indici.

## Qualità

Come minimo:

- perplessità held-out;
- divergenza KL sul token successivo;
- punteggio su benchmark downstream;
- task di programmazione;
- task di matematica/ragionamento.

Il set di calibrazione non deve essere usato come set di test finale.

## Velocità

Misurare separatamente:

- tempo di caricamento;
- latenza del primo token;
- generazione di token a batch=1;
- throughput a batch >1;
- elaborazione del prompt;
- contesto lungo.

## Memoria

Misurare:

- dimensione del file;
- memoria residente;
- picco di memoria unificata;
- opzionalmente banda di memoria effettiva.

## Energia

Se misurabile:

```text
joules/token
energy/model-load
```

Altrimenti dichiarare esplicitamente che l'energia non è stata misurata.

---

# 26. Fase 19 — Ablation obbligatorie

> **Stato al 14/10/2026:** eseguite su Qwen3-8B a 4.22 bpp — [final_report.md](final_report.md) §3.6.

*(vedi D3: aggiungere `SUC - incoherence` e `SUC - fine-tuning`)*

Testare il sistema finale rimuovendo un componente alla volta:

```text
SUC
SUC - semantic sensitivity
SUC - vector quantization
SUC - multi-codebook
SUC - residuals
SUC - adaptive bit allocation
SUC - entropy coding
SUC - fused kernel
```

Questo identifica da dove viene il miglioramento effettivo.

Nessuna affermazione sulle prestazioni è valida senza una tabella di ablation.

---

# 27. Fase 20 — Frontiera di Pareto

> **Stato al 14/10/2026:** frontiera di Qwen3-8B completata (allocazione Fisher-greedy, [final_report.md](final_report.md) §3.5); a ≤ 1.75 bpp il modello si rompe; latenza / memoria per punto dei piani misti non misurate.

Generare modelli attorno alle dimensioni target:

```text
4.0 bpw
3.5 bpw
3.0 bpw
2.5 bpw
2.25 bpw
2.0 bpw
1.75 bpw
1.5 bpw
~1.0 bpw
```

Per ogni punto misurare:

- qualità semantica;
- dimensione;
- latenza;
- picco di memoria;
- energia, se disponibile.

Grafico:

```text
quality
  ^
  |                *
  |            *
  |         *
  |      *
  |   *
  +----------------------> bits/parameter
```

L'obiettivo è spostare la frontiera verso bpw più bassi senza distruggere qualità o runtime.

---

# 28. Fase 21 — esperimenti sull'obiettivo semantico

Confrontare diverse definizioni di similarità tra modelli.

### A. MSE sui pesi

\[
||W-W_q||_2^2
\]

### B. MSE sull'output del layer

Confrontare gli output dei layer sui dati di calibrazione.

### C. MSE sui logit

Confrontare i logit del token successivo.

### D. Divergenza KL

\[
D_{KL}(P||Q)
\]

### E. Perplessità

### F. Obiettivo composito

Per esempio:

\[
L = \lambda_1 KL + \lambda_2 PPL + \lambda_3 TaskLoss + \lambda_4 Memory + \lambda_5 Latency
\]

L'implementazione deve rendere configurabili i pesi.

---

# 29. Fase 22 — esperimenti di allocazione universale/adattiva

Creare due modalità.

## Modalità specifica per il modello

Fortemente ottimizzata per un modello e una distribuzione di calibrazione.

## Modalità adattiva

È disponibile una famiglia di rappresentazioni e l'allocatore regola i pesi delle risorse in base alle prestazioni osservate.

Scopo:

> determinare se l'idea di Universal Allocation produce benefici misurabili rispetto a una quantizzazione statica calibrata a mano.

È una questione empirica, non un teorema ereditato da Cover.

---

# 30. Fase 23 — scalatura a modelli grandi

> **Stato al 14/10/2026:** nessuna fase di scalatura è stata eseguita come tale (D7 / D8), ma la pipeline finita ha compresso Qwen3-30B-A3B, Qwen3.8-27B e modelli MoE da 177–313 B ([compression_report-IT.md](compression_report-IT.md)).

Solo dopo un'implementazione 7B/8B stabile.

Testare, se memoria/tempo lo consentono:

```text
13B/14B
30B-class
70B-class
```

Storage grezzo approssimativo dei parametri per pesi da 70B:

```text
FP16 ≈ 140 GB
8-bit ≈ 70 GB
4-bit ≈ 35 GB
```

Sono approssimazioni dei soli parametri; l'impronta reale in deployment include metadati, scale, buffer di runtime e KV cache.

Un Mac di classe Ultra con ≥ 128 GB di memoria unificata (M1 Ultra o successivo) è un target di ricerca utile perché consente di sperimentare con modelli difficili da tenere residenti nelle configurazioni convenzionali con sola memoria GPU.

---

# 31. Fase 24 — compressione semantica opzionale specifica per il modello

Dopo che il SUC generico funziona, studiare rappresentazioni condizionate dal task.

Esempi:

```text
coding-optimized
math-optimized
general-chat
long-context
```

L'idea non è creare manualmente pesi base diversi, ma misurare se gli obiettivi semantici spostano il punto in cui allocare precisione/capacità del codebook.

Un risultato di successo sarebbe:

```text
same base model
same approximate total size

coding workload:
  preserve blocks important for coding

math workload:
  preserve blocks important for math
```

Ogni miglioramento deve essere validato su dati held-out.

---

# 32. Controlli contro il leakage di dati/modello

L'agente deve prevenire:

- leakage tra calibrazione e test;
- overfitting specifico sui benchmark;
- tuning diretto su esempi di benchmark pubblici che verranno poi riportati;
- conoscenza futura nascosta;
- modifica del protocollo di valutazione dopo aver visto i risultati.

Mantenere tre partizioni di dati:

```text
calibration
validation
final test
```

più un set di valutazione out-of-domain separato.

---

# 33. Test unitari

Implementare test per:

- quantizzatore scalare;
- quantizzatore vettoriale;
- serializzazione dei codebook;
- packing degli indici;
- unpacking degli indici;
- ricostruzione con codebook additivo;
- codifica dei residui;
- codec di entropia;
- serializzazione SUC;
- deserializzazione SUC;
- seed deterministici.

I property test devono verificare che:

```text
encode → decode
```

corrisponda esattamente al riferimento Python per la rappresentazione rappresentabile.

---

# 34. Test numerici/di runtime

Per ogni kernel Metal confrontare con l'implementazione di riferimento:

```text
max_abs_error
mean_abs_error
relative_error
logit KL divergence
```

Usare tolleranze appropriate al formato numerico.

Non accettare la somiglianza visiva/su un singolo prompt come prova sufficiente.

---

# 35. Matrice dei test di prestazione

Come minimo:

```text
batch = 1
batch = 4
batch = 16
```

e diverse lunghezze di sequenza.

Misurare:

```text
prefill tok/s
decode tok/s
first-token latency
kernel latency
load latency
peak memory
```

Mantenere batch=1 come benchmark di prima classe, perché l'uso locale degli LLM è spesso limitato dalla memoria e batch=1 espone chiaramente l'overhead di lookup/latenza.

---

# 36. Modalità di fallimento da investigare attivamente

## Fallimento A — compressione eccellente, qualità scarsa

Cause probabili:

- blocchi troppo grandi;
- codebook troppo piccolo;
- metrica di distanza inadeguata;
- nessuna calibrazione semantica.

## Fallimento B — compressione eccellente, runtime lento

Cause probabili:

- lookup casuali;
- pressione sulla cache;
- overhead di ricostruzione;
- scelta inadeguata di threadgroup/layout.

## Fallimento C — l'entropy coding fa risparmiare storage ma aumenta il runtime

È prevedibile che possa accadere.

Tenere l'entropy coding fuori dal percorso caldo oppure decodificare in una forma packed adatta all'esecuzione.

## Fallimento D — l'allocatore va in overfitting sulla calibrazione

Usare valutazioni held-out e OOD.

## Fallimento E — 2 bpw è attraente numericamente ma non efficiente sull'hardware

Accettare che una rappresentazione leggermente più grande possa essere superiore sulla frontiera di Pareto.

L'obiettivo reale non è "bpw minimi a qualunque costo".

---

# 37. Regola ingegneristica importante: la rappresentazione compressa deve tenere conto dell'esecuzione

Non progettare prima lo storage per scoprire poi che la GPU non riesce a usarlo in modo efficiente.

Mantenere due layout:

## Layout di ricerca

Array/tensori semplici per la sperimentazione.

## Layout di esecuzione

Dati packed e allineati, ottimizzati per l'accesso alla memoria da Metal.

La conversione tra i due deve essere esplicita e sottoposta a benchmark.

---

# 38. Spazio di ricerca per il design dei codebook

Testare:

### Codebook scalare

```text
one value per code
```

### Codebook vettoriale

```text
one vector per code
```

### Codebook additivo

```text
sum of codebook vectors
```

### Codebook gerarchico

```text
global codebook
+
layer codebook
+
block residual
```

### Codebook in stile prodotto

Dividere il vettore in sottospazi e codificare ciascun sottospazio separatamente.

Per ogni famiglia confrontare:

```text
quality
size
lookup complexity
kernel cost
```

---

# 39. Algoritmo di allocazione semantica dei bit — prima versione concreta

Implementare il seguente algoritmo greedy prima di qualsiasi ottimizzatore sofisticato.

### Passo 1

Dividere ogni tensore idoneo in blocchi.

### Passo 2

Per ogni blocco e rappresentazione candidata produrre un record:

```json
{
  "block": 123,
  "candidate": "suc_vq_3bit",
  "bytes": 192,
  "semantic_loss": 0.0017,
  "latency_cost": 0.04
}
```

### Passo 3

Stabilire una rappresentazione minima per ogni blocco.

### Passo 4

Creare i possibili upgrade.

### Passo 5

Ordinare gli upgrade per:

\[
score = \frac{\Delta quality}{\Delta bytes}
\]

oppure per un obiettivo pesato configurabile.

### Passo 6

Spendere il budget rimanente.

### Passo 7

Eseguire la validazione sull'intero modello.

### Passo 8

Iterare se la stima di sensibilità locale diverge dal comportamento dell'intero modello.

Questa prima implementazione deve essere abbastanza semplice da poter essere ispezionata manualmente.

---

# 40. Allocatore avanzato — ricerca con multiplicative weights

> **Stato al 14/10/2026:** non implementato; esiste solo l'allocatore statico Fisher-greedy (§39).

Una volta che l'allocatore greedy funziona, creare un'implementazione di ricerca in cui ogni rappresentazione candidata ha un peso.

Per il candidato `j`:

\[
w_{j,t+1}=w_{j,t}e^{-\eta\ell_{j,t}}
\]

dove `ell` è la loss misurata e `eta` è il learning rate.

Farlo prima in simulazione.

Domande a cui rispondere:

1. L'allocatore converge verso buone rappresentazioni?
2. È stabile con misure rumorose?
3. Quanta esplorazione è necessaria?
4. Supera la selezione greedy?
5. Reagisce in modo eccessivo al rumore di misura?

Questo collega il problema di compressione all'aggregazione di expert senza assumere un'equivalenza non dimostrata con il teorema originale di Cover.

---

# 41. Strategia di ricerca per l'inferenza diretta

Un kernel SUC diretto va esplorato in questo ordine:

```text
1. decode + GEMM
2. fused decode + GEMM
3. vector LUT + dot product
4. additive codebook fused kernel
5. partial-sum/codebook-centric kernel
```

Lo scopo del passo 1 è solo la correttezza.

I passi 2–5 sono quelli in cui vanno dimostrati i miglioramenti di velocità.

---

# 42. Benchmark rispetto a una vera baseline low-bit

Non confrontare mai SUC solo con FP16.

Confrontare come minimo:

```text
FP16
strong 4-bit baseline
SUC 4 bpw
SUC 3 bpw
SUC 2 bpw
```

Una condizione di successo significativa non è:

> "SUC è più piccolo di FP16."

Questo è banale.

Un risultato significativo è:

> "SUC è significativamente più piccolo di una baseline forte a 4 bit, preservando una qualità simile e restando competitivo in latenza."

---

# 43. Prima milestone — singolo layer lineare

*(vedi D1: questo gate si applica al kernel track)*

Questo è il gate di implementazione primario.

Input:

```text
W = FP16 matrix
X = FP16 activations
```

Costruire:

```text
W ≈ codebook[index]
```

Calcolare:

\[
Y=XW^T
\]

senza ricostruire l'intera `W` in FP16 durante il percorso cronometrato.

Confrontare:

```text
FP16
INT4
SUC decode+GEMM
SUC direct Metal
```

Misurare:

- errore sull'output;
- tempo del kernel;
- throughput;
- uso di memoria;
- banda effettiva.

**Non procedere alla compressione dell'intero modello finché questa milestone non è solida.**

---

# 44. Seconda milestone — modello 7B/8B completo

> **Stato al 14/10/2026:** sull'8B denso è stato esplorato ≈ 3 bpp (VQ G=2 K=64 a 3.37 bpp: PPL +9.3 %, KL 0.124) ma il file scelto è a 4.22 bpp (KL 0.049); file scelti sotto i 3 bpp esistono solo per i modelli MoE frontier grazie al livello tcq degli esperti (DeepSeek-V4-Flash 2.72 bpp, GLM-5.3-Flash 2.53 bpp).

Puntare a un modello completo con:

```text
vector codebooks
+
semantic sensitivity
+
mixed precision
+
residual/outlier support
```

Primo target:

```text
~3 bpw
```

Poi investigare:

```text
~2.5
~2.0
<2.0
```

Non ottimizzare aggressivamente per 1 bpw prima di aver dimostrato la stabilità attorno ai 3 bpw.

---

# 45. Terza milestone — allocatore automatico

L'utente deve poter specificare:

```yaml
budget_gb: 20
quality_floor: 0.98
latency_ceiling_ms: 100
```

e ricevere automaticamente un piano di rappresentazione.

Il piano finale deve contenere:

```text
tensor
block
representation
bits
codebook
residual policy
estimated size
```

Nessun valore finale di precisione layer per layer deve essere fissato nel codice dell'allocatore.

---

# 46. Quarta milestone — esecuzione Metal diretta completa

Il modello deve essere eseguibile con una combinazione di:

```text
conventional low-bit tensors
+
SUC tensors
+
FP16 exceptions
```

senza prima ricostruire l'intero modello in FP16.

---

# 47. Livelli di successo

> **Stato al 14/10/2026:** Livello A raggiunto; Livello B solo in parte (qualità ≈ 4 bit raggiunta a ≈ 4.0–4.2 bpp, non a ≈ 3 bpp; esecuzione competitiva nel decode, non nel prefill); Livello C non raggiunto — [final_report.md](final_report.md) §5.1.

## Livello A — proof of concept

Successo se:

- il codebook vettoriale funziona;
- il modello completo può essere ricostruito per la valutazione;
- si dimostra <4 bpw;
- l'errore semantico è misurato;
- l'output è riproducibile.

## Livello B — prototipo di ricerca solido

Successo se:

- circa 3 bpw o meno;
- qualità vicina a una baseline forte a 4 bit;
- l'inferenza Metal diretta funziona;
- nessuna ricostruzione FP16 completa nel percorso caldo;
- runtime competitivo.

## Livello C — risultato di rilievo

Un risultato di rilievo richiederebbe di dimostrare simultaneamente, su almeno un modello serio:

```text
~2 bpw or below
+
small measured quality degradation
+
direct GPU execution
+
competitive throughput
+
substantially lower memory footprint
```

Queste sono aspirazioni, non garanzie.

---

# 48. Che cosa non si deve affermare senza prove

Non affermare mai:

- "compressione semantica lossless" se i pesi sono alterati;
- "stessa intelligenza" senza prove specifiche sui task;
- "più veloce perché più piccolo" senza misure di tempo;
- "universale" nel senso matematico di Cover senza una dimostrazione corrispondente;
- "1 bpw funziona" basandosi solo sull'MSE di ricostruzione;
- "funziona per tutti gli LLM" dopo un solo modello riuscito;
- "l'entropy coding migliora l'inferenza" quando riduce solo la dimensione del file.

---

# 49. Matrice finale degli esperimenti

Alla fine, generare una matrice consolidata come:

| Rappresentazione | bpw | Dimensione | PPL | KL | Punteggio task | tok/s | TTFT | RAM di picco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | | | | | | | | |
| INT8 | | | | | | | | |
| INT4 | | | | | | | | |
| Reference 4-bit | | | | | | | | |
| SUC scalar | | | | | | | | |
| SUC vector | | | | | | | | |
| SUC additive | | | | | | | | |
| SUC + residual | | | | | | | | |
| SUC + semantic allocation | | | | | | | | |
| SUC + entropy storage | | | | | | | | |
| SUC direct Metal | | | | | | | | |

---

# 50. Requisiti del report finale

Creare:

```text
docs/final_report.md
```

Il report deve rispondere a:

1. Quali bpw effettivi minimi sono stati raggiunti?
2. Qual era la vera dimensione compressa, incluso tutto l'overhead?
3. Quanta qualità è stata persa?
4. Quali task sono stati colpiti?
5. Qual era il throughput di inferenza?
6. Qual era la latenza del primo token?
7. Qual era il picco di memoria?
8. L'esecuzione compressa diretta è stata più veloce o più lenta del 4 bit convenzionale?
9. Quali componenti hanno prodotto il miglioramento?
10. Che cosa è fallito?
11. Quali assunzioni non hanno retto?
12. Qual è la prossima direzione di ricerca?

Includere tutti i risultati negativi.

---

# 51. Ordine di esecuzione raccomandato

L'agente di programmazione deve seguire questo ordine, a meno che un esperimento documentato non dimostri un motivo per cambiarlo. Secondo D1 il lavoro procede su due track; all'interno di un track, l'ordine è rigido.

```text
FONDAMENTA CONDIVISE
 1. Inferenza di riferimento (FP16 + MLX a 4 bit + AWQ/GPTQ dove possibile)
 2. Infrastruttura di benchmark riproducibile
 3. Analisi della struttura dei pesi
 4. Calibrazione e stime della sensibilità (D5)

PERCORSO QUALITÀ (pesi ricostruiti, valutati in MLX)
 Q1. Riferimento con codebook scalare
 Q2. Codebook vettoriale (entro il limite D4)
 Q3. Elaborazione per ridurre la coerenza (D3)
 Q4. Codebook multipli
 Q5. Supporto dei residui e dei valori anomali
 Q6. Affinamento del codebook (D3)
 Q7. Allocatore semantico greedy dei bit (granularità secondo D6)
 Q8. Contenitore SUC e decoder Python di riferimento

PERCORSO KERNEL (inizia dopo Q2)
 K1. Kernel Metal per la verifica di correttezza (mx.fast.metal_kernel)
 K2. Kernel Metal fuso
 K3. Benchmark di un singolo strato lineare  <-- verifica §43
 K4. Kernel per codebook multipli, residui e rotazioni

INTEGRAZIONE
 5. Integrazione dell’intero transformer (rappresentazioni miste)
 6. Codifica entropica (solo archiviazione)
 7. Studi di ablazione
 8. Frontiera di Pareto
 9. Esperimenti di allocazione universale/adattiva
10. Scalabilità a modelli grandi (fuori ambito per D7 / D8; realizzata con la pipeline finita, vedi D8)
11. Runtime C++ (D2 — obbligatorio, rinviato; completato)
12. Integrazione opzionale con llama.cpp/GGUF (non realizzata; i GGUF sono solo riferimenti di confronto)
```

Se una fase fallisce, diagnosticarla e correggerla prima di scalare.

---

# 52. Direzione di ricerca approfondita: Semantic Universal Compression

La formulazione di lungo termine più forte è:

\[
Model = Codebooks + Indices + Residuals + Metadata
\]

dove la rappresentazione minimizza la perdita di comportamento del modello anziché il solo errore numerico sui pesi.

Un'ottimizzazione concettuale è:

\[
\min_{R}
\left[
\lambda_s Size(R)
+
\lambda_q D_{semantic}(M,M_R)
+
\lambda_l Latency(R)
+
\lambda_e Energy(R)
\right]
\]

La rappresentazione può contenere:

```text
very-low-bit vector codebooks
+
multiple codebooks
+
layer-specific codebooks
+
semantic-sensitive high precision blocks
+
sparse residuals
+
optional entropy coding
+
hardware-oriented packing
```

L'ipotesi centrale è:

> La rappresentazione utile minima di un LLM è determinata dall'informazione necessaria a preservarne il comportamento, non semplicemente dalla precisione numerica usata per memorizzare ogni peso originale.

L'ipotesi ingegneristica centrale è:

> Una rappresentazione compressa sufficientemente strutturata può essere consumata direttamente da kernel di inferenza ottimizzati senza una passata completa di dequantizzazione.

L'ipotesi di allocazione è:

> Sotto un budget fisso di memoria/bit, un'allocazione adattiva può spendere precisione aggiuntiva dove fornisce la massima preservazione marginale del comportamento del modello.

Tutte e tre le ipotesi devono essere valutate in modo indipendente.

---

# 53. Riferimenti utili

## Apple MLX

Documentazione sulla memoria unificata:

https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html

## llama.cpp

https://github.com/ggml-org/llama.cpp

Strumenti di quantizzazione:

https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize

Build / Metal:

https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

## AQLM

“AQLM: Extreme Compression of Large Language Models via Additive Quantization”

https://arxiv.org/abs/2401.06118

## QuIP

“QuIP: 2-Bit Quantization of Large Language Models With Guarantees”

https://arxiv.org/abs/2307.13304

## QuIP#

“QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Adaptive Rounding”

https://arxiv.org/abs/2402.04396

## PV-Tuning

"PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression"

https://arxiv.org/abs/2405.14852

## Kernel Metal custom di MLX

https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

## FLUTE

“FLUTE: Fast Lookup Table-based Matrix Multiplication for LLM Inference”

https://arxiv.org/abs/2407.10960

## CodeGEMM

“CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs”

https://arxiv.org/abs/2512.17970

---

# 54. Istruzione finale per l'agente di programmazione

Trattare questo repository come **ingegneria di ricerca**.

Non ottimizzare per produrre una grande quantità di codice. Ottimizzare per produrre un piccolo numero di esperimenti verificabili in modo indipendente.

Ogni affermazione deve avere:

```text
baseline
+
change
+
measurement
+
repeatability
+
trade-off
```

Preferire un benchmark del kernel su 1 layer scientificamente corretto a un'implementazione sull'intero modello le cui prestazioni non sono comprese.

Il progetto ha successo anche se l'ipotesi viene smentita, a patto che il risultato negativo sia rigoroso e identifichi il fattore limitante.
