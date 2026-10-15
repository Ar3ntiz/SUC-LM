# Runtime C++ di SUC

English version: [README.md](README.md)

Runtime autonomo per i file SUC1 (spec §0 D2): mappa il file in memoria (zero-copy), esegue direttamente sui codici i kernel
Metal generati da quelli Python, e fornisce tokenizer, chat template, sampling, prefill in batch, streaming degli expert
da SSD e un server compatibile con OpenAI. Nessuna dipendenza oltre all'SDK di macOS.

| Grafo | Modelli | Note |
|---|---|---|
| `qwen3`, `qwen3_moe` | Qwen3-0.6B / 8B / 30B-A3B | milestone 3–4 |
| `qwen3_5` (denso sul grafo `qwen4_exp`) | Qwen3.8-27B | Gated DeltaNet + gated attention |
| `qwen4_exp` | Qwen3.8-Flash-Next | milestone 5; hyper-connection, QSA, tabelle n-gram PLE |
| `deepseek_v4` | DeepSeek-V4-Flash | milestone 6; CSA / HCA, expert stack MXFP4 nativi e misti |
| `glm5_next` | GLM-5.3-Flash | milestone 7; mHC, KDA, MLA + DSA |

Tutti i codec del formato sono decodificati (suc1-decode: raw, bf16, affine, scalar, vq, additive, tcq, MXFP4 nativo); i
grafi dei modelli li eseguono tutti tranne `scalar`. I passi di decode
sono bit-identici al runtime Python dello stesso file (hidden state e logit); il prefill veloce in batch non lo è
(KL misurata 0.0006–0.03, docs/compression_report-IT.md §1c).

## Prestazioni attuali (Mac Studio M1 Ultra, batch 1)

`suc1-generate`, greedy, 256 token; file scelto per modello (docs/compression_report-IT.md §1, dove si trova il confronto con
ds4, LM Studio / llama.cpp e MLX).

| Modello (file) | SUC1 GB | Memoria GB | Decode tok/s | Prefill tok/s · TTFT, breve | Prefill tok/s · TTFT, 2k | Prefill tok/s, 32k | Decode tok/s a 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B (final) | 4.32 | 4.63 | 76.3 | 128 · 0.11 s | 304 · 6.8 s | 106 | 42.5 |
| Qwen3.8-27B (side8) | 16.36 | 16.75 | 25.3 | 43 · 0.35 s | 93 · 22.1 s | 58 | 20.4 |
| Qwen3-30B-A3B (final) | 15.93 | 17.11 | 82.4 | 199 · 0.08 s | 367 · 5.6 s | 101 | 37.5 |
| Qwen3.8-Flash-Next (side8) | 96.32 | 73.82 | 28.3 | 55 · 0.27 s | 341 · 6.0 s | 323 | 29.2 |
| DeepSeek-V4-Flash (tcq) | 96.82 | 102.57 | 24.0 | 46 · 0.30 s | 171 · 12.0 s | 138 | 13.7 |
| GLM-5.3-Flash (tcq) | 99.00 | 106.59 | 22.2 | 35 · 0.41 s | 116 · 17.7 s | 94 | 13.1 |
| DeepSeek-V4-Flash V3 (expert su SSD) | 155.05 | 116.75 | 13.1 | 43 · 0.36 s | 28 · 73.9 s ¹ | — | — |
| GLM-5.3-Flash base (expert su SSD) | 119.93 | 115.03 | 9.1 | 7 · 2.08 s | 16 · 125.8 s ¹ | — | — |

Qwen3-0.6B (Apple M3): 6.95 ms/token (143.9 tok/s). ¹ Prefill riga-esatto di una build precedente.

## Avvio rapido

```bash
cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc1-inspect model.suc1 --no-verify          # header, tokenizer id, byte accounting
~/suc-local/cpp-build/suc1-generate model.suc1 --chat --prompt "Hello" --steps 256
~/suc-local/cpp-build/suc1-server model.suc1 --port 8080 --max-context 32768
```

I file SUC1 e i tokenizer sono pubblicati su [huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz), oppure prodotti da
`suc compress` (`experiments/artifacts/compress/<config>/container/model.suc1`). Il
tokenizer viene risolto a partire dal `tokenizer_id` del SUC1 (`Org/Name@rev;tokenizer.json:sha256:…`) nella cache di Hugging Face
(`$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`) e verificato rispetto al suo SHA-256; lo si scarica con
`hf download ORG/NAME tokenizer.json --revision REV` (REV: l'hash di commit completo di 40 caratteri), oppure si passa `--tokenizer path/to/tokenizer.json`.
Modelli e revisioni: [README-IT.md §6](../README-IT.md#6-uso). I file più grandi del working set GPU predefinito (≈ 75 % della RAM) richiedono di alzare
una volta il limite di memoria wired della GPU (`sudo sysctl iogpu.wired_limit_mb=116000` su 128 GB); i file più grandi della memoria fanno streaming dei loro expert:
`--expert-budget 95G --hot-experts auto`.

## Contenuto di questo documento

Build e test, strumenti, libreria; poi il registro dell'implementazione per milestone (1: reader + decoder di riferimento, 2: un
layer lineare Metal, 3: grafo denso Qwen3, 4: piano fuso / tokenizer / MoE / decode in pipeline, 5: qwen4_exp, streaming
degli expert da SSD, 6: deepseek_v4, 7: glm5_next, tensori tcq), `suc1-server` e il lavoro sul prefill in batch. I
"Risultati (laptop M3)" di ogni milestone sono le verifiche sul modello di sviluppo a quella data; le velocità di riferimento sono quelle della tabella sopra.

## Build

Le directory di build vanno su un disco locale, mai sulla share SMB.

```bash
# CMake (C++20)
cmake -S cpp -B ~/suc-local/cpp-build -DCMAKE_BUILD_TYPE=Release && cmake --build ~/suc-local/cpp-build -j
# or without CMake: same sources and flags
cpp/build.sh ~/suc-local/cpp-build
```

Output: `libsuc.a`, `suc1-inspect`, `suc1-decode`, `suc-tokenize`, `suc-tests`, `suc-request-params-tests`
(limiti int64 degli helper delle richieste sotto `-fsanitize=float-cast-overflow`, audit F10); su macOS anche `libsuc_metal.a`,
`suc1-linear`, `suc-linear-bench`, `suc-mm-bench`, `suc1-generate`, `suc1-server`, `suc1-moe`, `suc-expert-bench`,
`suc-metal-tests` (impostare `SUC_NO_METAL=1` per saltarli). Flag obbligatorio: `-ffp-contract=off`. La bit-esattezza
dipende dal fatto che il compilatore non fonda mai `a*b + c` in una fma, perché numpy arrotonda il prodotto e la somma
separatamente. L'unica operazione fusa è la `std::fma` esplicita nel passo Hadamard `H_m` (vedi sotto).

## Test

```bash
~/suc-local/cpp-build/suc-tests                                   # C++ unit tests
~/suc-local/cpp-build/suc-request-params-tests                    # request int64 boundaries (float-cast sanitizer)
.venv/bin/python -m pytest tests/unit/test_cpp_decoder.py         # bit-exact cross-check vs Python
.venv/bin/python -m pytest tests/unit/test_cpp_decoder.py -m model # + the real dev container
~/suc-local/cpp-build/suc-metal-tests                             # Metal backend unit tests (milestone 2)
.venv/bin/python -m pytest tests/unit/test_cpp_metal.py           # GPU vs CPU reference, bit-exact vs MLX
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py        # model graph vs the Python runtimes (milestones 3-4)
.venv/bin/python -m pytest tests/unit/test_cpp_moe.py             # SwitchLinear / MoeBlock vs Python (milestone 4)
.venv/bin/python -m pytest tests/unit/test_cpp_server.py          # suc1-server with the openai client
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/unit/test_cpp_tokenizer.py   # C++ tokenizer vs transformers
```

I controlli incrociati Python girano con `SUC_AUTOTUNE=0` (lo imposta `tests/unit/conftest.py`), così i test non eseguono mai
l'autotuning implicitamente; `tests/unit/test_autotune.py` copre il tuner.

- `suc-tests` (senza framework) copre: vettori SHA-256 (FIPS, 1M × 'a', output di hashlib), sia il percorso
  portabile sia quello con SHA-2 ARMv8, ogni suddivisione in chunk e ogni lunghezza 0..200. Unpacking dei bit a larghezze 1..32
  rispetto a un packer indipendente bit per bit, più valori di ancoraggio da `suc.quant.packing.pack_indices`. Il flusso di segni
  PCG64 di numpy rispetto a `np.random.default_rng(seed).integers(0, 2)` per 7 seed, inclusi seed ≥ 2³².
  Conversione fp16 → fp32 esaustiva. Parsing JSON e i suoi casi di errore. Esattezza e ortonormalità
  di Hadamard. Validazione di header, footer e directory su file sintetici: magic errato, versione, flag sconosciuti
  o incoerenti, conteggi, allineamento, troncamento, byte in coda, campi del footer, checksum non corrispondente
  (e `--no-verify`), dimensioni, dtype, sezioni e JSON corrotto.
- `tests/unit/test_cpp_decoder.py` (saltato quando gli strumenti non sono compilati; impostare `SUC_CPP_BUILD` per puntare
  altrove) scrive file SUC1 con il writer Python e verifica che ogni pattern di bit float32 coincida con
  `SucModel.decode_tensor`. Il file contiene 38 tensori codec e 8 tensori raw. Codec coperti: bf16;
  affine a 2/3/4/8 bit; scalar per tensore, per gruppo di righe e gruppi di righe irregolari, con k = 512; vq con
  larghezze di indice di 1, 3, 4, 5, 9 e 10 bit; additive con M = 2/3/4; rotazione in_out, in e out; dimensioni di Hadamard
  2^k, 12·2^k, 20·2^k, 28·2^k (incl. 5120, 7168) e 16384 (split di Kronecker); residui sparsi fp16/int8 (magnitudine/importanza, n = 0),
  colonne int8/fp16 e colonne con coda sparsa. Dtype raw coperti: bf16, f16, f32, i8, i32, u8,
  bool e 0-d.

## Strumenti

- `suc1-inspect FILE [--no-verify] [--codebook-hashes] [--tensors] [--filter S]`: stampa i campi dell'header,
  i flag, gli id di architettura e tokenizer, lo SHA-256 del footer e il suo stato di verifica (con i tempi),
  i conteggi per (rappresentazione, residuo, rotazione), parametri, byte e bpw, e il conteggio dei byte dell'intero
  file. Il conteggio coincide con `SucModel.bits()` e somma esattamente alla dimensione del file. Con
  `--tensors` aggiunge una riga per tensore: shape, bit di indice, byte, bpw dei suoi array e bpw
  inclusi i codebook che possiede (ogni codebook deduplicato è attribuito al primo tensore che lo usa,
  in ordine canonico).
- `suc1-decode FILE (--all | --tensor NAME ...) [--out DIR] [--format npy|raw] [--digest] [--threads N]`:
  decodifica in `.npy` float32 (o `.f32` raw). `--digest` stampa lo SHA-256 di ciascun buffer decodificato, il che
  permette il confronto senza scrivere gigabyte. Il risultato non dipende mai da `--threads`. Ogni rappresentazione
  SUC1 viene decodificata (raw, bf16, affine, scalar, vq, additive, tcq, `mxfp4` nativo: E2M1 con il nibble basso per primo,
  scale di gruppo E8M0, codice 8 = +0, scala 255 rifiutata), bit-esatta con `SucModel.decode_tensor`
  (tests/unit/test_cpp_decoder.py); la tabella delle capacità dei codec è `suc/capabilities.py`.
- `suc1-server`: i campi numerici delle richieste sono validati prima di qualsiasi conversione con restringimento (HTTP 400 con il nome del campo;
  `cpp/include/suc/request_params.hpp`); un errore durante una generazione in streaming viene inviato come evento SSE `error`
  senza finish_reason, che l'eval runner riporta come ERROR.

## Libreria

- `suc/suc1.hpp`, `Suc1File`: mappa il file in memoria in sola lettura e valida header e footer (le
  regole di format.md §9). Verifica lo SHA-256 se non disabilitato; può anche verificare l'hash di dedup di ciascun
  codebook. Analizza la directory in JSON canonico con il parser minimale incluso (`suc/json.hpp`), che mantiene
  esatti i valori int64. Valida il layout: allineamento a 64 byte, payload all'interno della regione dei payload e non
  sovrapposti, `nbytes = prod(shape) · itemsize`, shape impacchettata `[rows, ceil(n·b/32)]`, `data_offset`,
  `compressed_size`, ordine canonico dei tensori, id dei codebook e `global_flags` coerenti con i
  descrittori. Espone `TensorDesc` / `CodebookDesc` / `ArrayRecord` e viste zero-copy `std::span` sui payload.
- `suc/decode.hpp`, `decode_tensor`: segue l'ordine delle operazioni di format.md §10, elemento per elemento:
  - raw: qualsiasi dtype memorizzato convertito in f32 (bf16 tramite shift di bit);
  - bf16;
  - affine: `q·s + b`, due arrotondamenti;
  - scalar: `C_g[i] · s_r`;
  - vq / additive: `acc = C_0[i_0]`, poi `acc + C_m[i_m]` per ogni m in ordine, poi `· s_r`;
  - rotazione: `rht_inverse` sulle righe (seed_in), poi sulle colonne (seed_out);
  - residuo: prima le colonne (`colval · colscale`, poi somma), poi la coda sparsa (`val · res_scale`, poi somma).
- `suc/hadamard.hpp`: le trasformate di `suc.quant.hadamard` con il suo ordine delle operazioni in float32.
  - Il fattore 2^k è la FWHT in place con distanze di butterfly 1, 2, 4, …, scalata una sola volta per
    `float(1/sqrt(2^k))`. Coincide con il kernel Metal `hadamard_n` di MLX e con il percorso CPU di MLX. Sopra 8192
    usa lo split di Kronecker `_pow2`.
  - Il fattore `m ∈ {12, 20, 28}` usa le tabelle ±1 di MLX scalate per `float(1/sqrt(m))`, applicate come
    catena sequenziale di fma su k (la stessa definizione usata anche da `suc.quant.hadamard._mix`).
- `suc/numpy_random.hpp`: i segni di Hadamard non sono memorizzati. Vengono ricavati da `seed_in` / `seed_out`
  con una reimplementazione bit-esatta di `np.random.default_rng(seed).integers(0, 2, n)`: SeedSequence
  (pool di 4 word, hashmix/mix), PCG64 (LCG a 128 bit, output XSL-RR) e l'estrazione Lemire a 32 bit bufferizzata.
- `suc/sha256.hpp`: SHA-256 portabile, più le istruzioni SHA-2 ARMv8 quando disponibili (sempre su Apple Silicon).

## Risultati (laptop M3, 2026-10-01)

Container di sviluppo `compress_qwen3_0.6b/container/model.suc1` (320 MB: 310 tensori, 784 codebook; 196 tensori additive
M=4 K=256 G=8 con un residuo sparso int8 e rotazione in_out, 1 embedding affine a 4 bit, 113 tensori
raw):

- Tutti i 310 tensori sono bit-identici a `SucModel.decode_tensor`, confrontati tramite SHA-256 dei buffer float32.
- Apertura (header, directory, validazione del layout): ≈ 5 ms. La verifica SHA-256 completa richiede 0.16 s a caldo (≈ 2 GB/s)
  e 0.6–0.7 s a freddo.
- Decodificare tutti i 596 M elementi in float32 richiede 1.43 s su 8 thread e 4.3 s su 1 thread. Il decoder di
  riferimento Python richiede ≈ 9.9 s per caricamento + decodifica.

## Deviazioni note / risultati

1. **Risolto: i fattori di Hadamard m = 20 e m = 28 sono ora bit-esatti.** Il riferimento Python applicava
   `H_m` con `mx.matmul`, il cui ordine di accumulo dipende dal dispositivo e dalla shape (per m = 20 / 28 GPU, CPU e
   la catena di fma differivano fino a 2.1e-7 in relativo). `suc.quant.hadamard._mix` ora definisce il passo come la stessa
   catena sequenziale di fma di `hadamard_transform` qui (`o = 0; for k: o = fma(H[i, k], y[k, c], o)`): un kernel
   Metal sulla GPU, numpy (float64 con arrotondamento round-to-odd, poi float32) sulla CPU. MLX GPU, MLX CPU e C++ concordano bit
   per bit, incluse 5120 = 20·256 e 7168 = 28·256 (`vq_rot_5120`, `vq_rot_7168` in `test_bit_exact_every_codec`).
   I risultati per m = 12 sono invariati (il matmul coincideva già con la catena per le dimensioni di Qwen3), quindi i
   container esistenti si decodificano ancora in modo bit-esatto. I kernel di rotazione del runtime (`suc.kernels.rotation`) coincidono solo entro
   una tolleranza: sull'M3 il layout "row" è bit-esatto, "blocks" differisce di ≤ 2.3e-7 in relativo.
2. **Risolto: rotazione solo in output** (`rotate_in = False, rotate_out = True`). Il decoder di riferimento Python
   chiamava `unrotate_weight(x, seed_in=None, seed_out)` (segni casuali) e il writer etichettava tali tensori
   `rotation = "in_out"`. `suc/io/suc1.py` ora rispecchia `_finish` in `suc.quant.codecs`, decodificando a partire dai parametri,
   quindi i file con la vecchia etichetta si decodificano ancora correttamente. Il writer emette la nuova etichetta `rotation = "out"`
   (format.md §5.2). Il decoder C++ usava già i parametri; `test_out_only_rotation_matches_codec` verifica
   C++ == SucModel == codec bit per bit.
3. CMake: `CMakeLists.txt` rispecchia `build.sh` (inclusi i target Metal), ma non è stato eseguito sul laptop
   di sviluppo (lì CMake non è installato). La build verificata è `build.sh`.

## Milestone 2: esecuzione Metal di un layer lineare SUC

`y = x Ŵᵀ` per un tensore vq / additive di un file SUC1, sulla GPU, con la rappresentazione letta **in place** dal
file mappato. Libreria `libsuc_metal.a` (`suc/metal.hpp`, `src/metal/metal.cpp`), metal-cpp di Apple (incluso
senza modifiche in `third_party/metal-cpp`, Apache-2.0).

### Progettazione

- **Kernel = i kernel Python/MLX.** `metal/suc_kernels.metal` è *generato* da `metal/gen_suc_kernels.py`
  a partire dai sorgenti Python: gli header di supporto e i corpi dei kernel sono copie testuali di
  `suc.kernels.vq_matmul._HEADER / _LOOKUP_SRC / _MMA_REG_SRC`, `suc.kernels.rotation._HEADER` più il corpo che
  `rotation._blocks_kernel(1, False)` costruisce, e `suc.runtime.compressed_model._STRUCT_HEADER / _STRUCT_SRC`. Solo
  le firme sono scritte a mano, nello stile in cui le genera `mx.fast.metal_kernel` (parametri template nello
  stesso ordine, stessi nomi degli attributi). `test_kernel_source_in_sync` fallisce quando un kernel Python cambia e
  il file non è stato rigenerato: `.venv/bin/python cpp/metal/gen_suc_kernels.py`, poi ricompilare.
- **Compilazione a run-time.** I Command Line Tools non includono un compilatore `metal` offline, quindi la build incorpora il
  testo `.metal` come stringa (`build.sh` / CMake generano `gen/suc_kernels_msl.cpp`), e il `Context` lo compila
  con `newLibraryWithSource`. Ogni istanziazione template è una riga `template [[host_name(..)]] [[kernel]]
  decltype(k<..>) k<..>;`, come in MLX, compilata una volta e messa in cache (≈ 5–60 ms ciascuna; la shader cache del sistema operativo rende
  economiche le ripetizioni). Le opzioni di compilazione sono quelle di MLX: fast math disattivato, la versione del linguaggio predefinita del sistema operativo (il corpo MMA
  usa una lambda, che MSL 3.1 rifiuta). `$SUC_METAL_SOURCE=path.metal` sostituisce il sorgente incorporato.
- **Mappatura zero-copy.** `MappedFile` avvolge l'intero `mmap` del file (base allineata alla pagina) in buffer
  `newBufferWithBytesNoCopy`. La lunghezza è arrotondata per eccesso alla dimensione di pagina (16 KB; `mmap` mappa pagine intere,
  quindi la coda dell'ultima pagina è indirizzabile). Oltre `maxBufferLength` il file è coperto da più viste tagliate
  solo tra *unità di mappatura* (`mapping_units`: per layer transformer e sezione di payload la sequenza dei segmenti di quel
  layer, così gli expert stack e le proiezioni fuse indirizzati come un unico intervallo restano in una sola vista; più l'intervallo dei codebook
  di ciascun layer), che condividono al massimo una pagina. La residenza è contabilizzata per vista: un command buffer che collega una vista la rende
  interamente residente, e il working set della GPU (`recommendedMaxWorkingSetSize`, ~ `iogpu.wired_limit_mb`) viene addebitato per ogni
  vista anche dove due condividono pagine. La precedente suddivisione in viste sovrapposte per metà addebitava ~1.4x il file
  (DeepSeek-V4-Flash: 86.6 + 64.1 GB per 107.4 GB sull'M1 Ultra, oltre il working set di 115.3 GB) e il primo
  command buffer restava in attesa in `waitUntilCompleted` per sempre. `MappedFile` ora avvisa quando le viste superano il working
  set; `--max-view-bytes N` limita le viste (test). I payload SUC1 sono
  allineati a 64 byte, quindi ogni array è semplicemente (buffer, offset in byte). Letti in place: indici impacchettati, scale di riga f16
  (`float(half)` è esatto, quindi leggerle direttamente equivale alla copia float32 che `SucLinear` mantiene), valori dei residui,
  valori delle colonne, e i codebook quando gli M segmenti di un tensore sono adiacenti nel file (sempre per i file
  scritti da `write_suc1`). Altrimenti i codebook da ≤ 16 KB sono impacchettati in un unico piccolo buffer.
- **Layout di esecuzione costruito al caricamento (solo array piccoli).** Segni di Hadamard (dai seed, RNG compatibile con numpy),
  puntatori di riga CSR e id di colonna u16/u32 dalle posizioni impacchettate a bit, id di colonna spacchettati, scale di colonna float32,
  meta dei segmenti: qualche decina di KB per tensore (vedi Risultati).
- **Piano di esecuzione** (`Linear`, = `SucLinear` / `SucResidualLinear` con `FUSE_RESIDUAL`), in un unico compute
  encoder (dispatch concorrente, memory barrier tra stadi dipendenti):
  1. `x̃ = V x`: kernel di rotazione, arrotondato al dtype delle attivazioni (solo con `rotate_in`); layout "blocks", oppure
     "row" (`rotation._kernel`, un threadgroup per riga) per le dimensioni in cui la cache di tuning lo seleziona (sull'M1
     Ultra n = 1024; prima di 2026-10 il port eseguiva lì "blocks" e non era bit-identico a Python su quella macchina).
  2. `ỹ = W' x̃`: kernel di lookup per tile di batch di ≤ 16 righe. Il kernel "vec" gira sotto `mma_min_batch`, il
     kernel simdgroup-matrix (frammenti "reg") da lì in su. L'output è float32 quando segue uno stadio successivo, altrimenti il
     dtype delle attivazioni.
  3. `y = Uᵀ ỹ`: rotazione inversa (solo con `rotate_out`).
  4. `+ R x`: kernel del residuo strutturato (parte a colonne e/o coda CSR, un solo lancio, arrotondamento dentro il kernel).
- **Tuning = il tuning Python.** `codebook_in`, `mma_min_batch`, il layout di rotazione e la forma dei blocchi provengono dalla stessa
  cache per dispositivo (`$SUC_TUNING_CACHE` o `~/.cache/suc/kernel_tuning.json`, chiave `architecture|device name`), con
  la stessa regola dell'M più vicino e gli stessi fallback (`applegpu_g13*` → codebook in threadgroup, MMA da B = 5,
  `default_block_config`). Ogni costante di modulo (`LOOKUP_SG`, `MMA_RT`, …) è un campo di `Tuning`.
- **Dtype delle attivazioni**: f16, bf16 (MSL `bfloat`), f32.

Non portati (non usati dalla configurazione Python predefinita): le strategie K1 / "table", `MMA_KIND = "tg"`,
`LOOKUP_HALF_MATH`, i prologhi element-wise (RMSNorm / SwiGLU + rotazione), le
proiezioni fuse multi-segmento (`SucFusedLinear`, `SucFusedResidual`) e `suc_switch` (MoE). Questi appartengono alla
milestone 3. Le dimensioni di rotazione il cui fattore 2^k è < 32 (Python: fallback su `mx.hadamard_transform`) usano un semplice kernel denso
per elemento (`suc_rot_dense`, solo C++).

### Strumenti

- `suc1-linear FILE (--all | --tensor NAME ...) [--batches 1,4,16] [--dtype f16|bf16|f32] [--path auto|vec|mma]
  [--json] [--x X.npy --y-out Y.npy] ...`: per x casuale (normale, arrotondata al dtype), esegue ogni tensore supportato sulla
  GPU. Confronta con il riferimento CPU: `decode_tensor` → Ŵ float32, poi `x Ŵᵀ` in float64. Riportati:
  max abs, mean abs, L2 relativa, max abs / max |ref|, e la L2 relativa dell'arrotondamento del risultato esatto stesso al
  dtype (il limite inferiore che ha qualsiasi output f16 / bf16). `--x/--y-out` esegue un tensore su una x data e scrive y; il
  pytest lo usa per il confronto bit-esatto con MLX. I tensori senza kernel GPU (affine, scalar, bf16, raw)
  vengono saltati.
- `suc-linear-bench [--shape 12288x4096] [--batches 1,4,16] [--gb 2] [--min-mb 512] [--samples 5] [--json OUT]`:
  la controparte C++ della modalità "layers" di `benchmarks/k3_cold_cache.py`: stessi metodi (`vq_G8_K256`,
  `vq_G4_K256`, `add2_G8_K256`, `add3_G8_K256` × `_kernel`, il percorso A/B `_vec` / `_mma` a B ≥ 4, `_full` con
  rotazioni), le stesse rappresentazioni casuali e lo stesso numero di copie (N copie distinte così che un ciclo legga in streaming ≥ max(gb byte
  FP16, min_mb) dalla DRAM), e la stessa regola di misura (un command buffer = N chiamate indipendenti, encode +
  esecuzione, mediana / minimo per chiamata sui campioni). Riporta anche il tempo della sola GPU.

### Test

- `suc-metal-tests` (C++, richiede una GPU; 696 verifiche): kernel di lookup su layer sintetici lato host rispetto a riferimenti CPU
  float64:
  - larghezze di indice 1/4/5/6/8/10, allineate e non allineate;
  - M = 1–4, G = 2/3/4/8 (G dispari prende il percorso MMA in ordine naturale);
  - righe di coda; B = 1, 3, 4, 5, 8, 16, 17, 33 (tiling); percorsi auto / vec / mma; f32 / f16 / bf16;
  - rotazioni in / out con dimensioni 2^k, 12·2^k, 2^14 e il fallback denso (448, 40, 24, 2);
  - residui CSR int8 / fp16, colonne int8 / fp16, entrambi, `out % 4 ≠ 0`;
  - conversione f16 / bf16 round-to-nearest-even.
- `tests/unit/test_cpp_metal.py` (saltato se non compilato):
  - ogni variante vq / additive scritta con `write_suc1` (24 tensori) rispetto al riferimento CPU a B = 1/3/4/5/16/17,
    per f32, f16 e bf16;
  - **bit-esatto rispetto a MLX** `module_from_encoded` (f16, B = 1/4/16) per ogni tensore le cui dimensioni di rotazione Python esegue
    con il proprio kernel; le dimensioni con fallback denso sono verificate a 5e-3;
  - la mappatura multi-vista (forzata con `--max-view-bytes`: stessi risultati di una sola vista);
  - la verifica di sincronizzazione del sorgente dei kernel, e i test unitari C++;
  - `-m model`: tutti i 196 tensori SUC del container 0.6B rispetto al riferimento, più 4 tensori bit-esatti rispetto a MLX.

### Risultati (laptop M3, 2026-10-02)

Accordo numerico con il riferimento CPU (L2 relativa di `y`, caso peggiore su tensori e B = 1/4/16; "floor" = la
L2 relativa dell'arrotondamento del risultato esatto al dtype):

| file / dtype | f32 | f16 (floor) | bf16 (floor) |
|---|---:|---:|---:|
| container 0.6B, 196 tensori (additive M=4 K=256 G=8, residuo CSR int8, rotazione in_out 1024/2048/3072) | 1.0e-6 | 3.1e-4 (2.3e-4) | 2.7e-3 (1.8e-3) |
| sintetico, 24 varianti di codec (variante peggiore) | 2.2e-6 | 3.6e-4 | 3.3e-3 |

- f32: percorso vec ≈ 2e-7, percorso MMA ≤ 1e-6 (MMA simdgroup float32). f16 / bf16 stanno a ~1.3× il floor: x̃ e y sono
  arrotondati ciascuno una volta al dtype, come nel runtime Python.
- **Bit-esatto con MLX**: ogni bit di output è uguale a quello di `SucLinear` / `SucResidualLinear` in Python. Vale per f16
  su tutte le varianti sintetiche tranne le dimensioni con fallback denso, e per f16 / bf16 / f32 sui tensori del container
  di sviluppo che sono stati verificati.
- Zero-copy: per un tensore 0.6B (3072×1024, M = 4), 1.61 MB sono letti in place dalla mappatura, e 52 KB sono
  costruiti al caricamento (segni, puntatori di riga CSR e colonne u16).

Tempi indicativi, 12288×4096 (gate/up dell'8B), x f16, µs per chiamata, cicli a cache fredda. Misurati sul laptop M3
mentre i job di altri agenti condividevano la GPU. Due esecuzioni C++ / MLX alternate, la migliore delle due mediane; i numeri del Mac Studio
sono quelli di riferimento:

| method | B=1 C++ | B=1 MLX | B=4 C++ | B=4 MLX | B=16 C++ | B=16 MLX |
|---|---:|---:|---:|---:|---:|---:|
| vq_G8_K256_kernel (1.0 bpw) | 127 | 126 | 302 | 306 | 620 | 630 |
| vq_G4_K256_kernel (2.0 bpw) | 154 | 157 | 309 | 306 | 584 | 594 |
| add2_G8_K256_kernel (2.0 bpw) | 479 | 495 | 553 | 622 | 841 | 984 |
| add3_G8_K256_kernel (3.0 bpw) | 341 | 386 | 604 | 582 | 1128 | 1144 |
| vq_G8_K256_full (+ rotazioni) | 144 | 147 | 325 | 325 | 744 | 674 |
| add3_G8_K256_full | 403 | 403 | 651 | 678 | 1188 | 1175 |

Lettura:
- I kernel sono gli stessi, e lo sono anche i tempi (entro il rumore di ±10–30 % di una GPU condivisa).
- Il lato host C++ non aggiunge nulla di misurabile a queste dimensioni.
- `_full` a B = 16 è ≈ 10 % più lento in C++ per vq. Una causa plausibile è l'ordine di dispatch: il C++ codifica rotazione →
  kernel → rotazione per ogni copia, con una barrier a ogni passo, mentre MLX potrebbe raggruppare prima le N rotazioni indipendenti.
  Non è verificato.
- Il percorso vec forzato a B = 16 (non scelto da auto) era 2–3× più lento sotto MLX in entrambe le esecuzioni. Non è spiegato
  e forse è termico: quelle righe vengono eseguite per ultime nello script Python.

Output grezzi: eseguire `suc-linear-bench` e `benchmarks/k3_cold_cache.py` (comandi sotto); i JSON del laptop non sono stati
committati (i tempi sono riportabili solo dallo Studio).

### Comandi per il Mac Studio (benchmark di riferimento)

```bash
cd ~/Projects/SUC-LM
cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_metal.py tests/unit/test_cpp_decoder.py -q
mkdir -p logs && TS=$(date -u +%Y%m%dT%H%M%SZ) && nohup sh -c "
  for s in 12288x4096 4096x12288 4096x4096; do
    ~/suc-local/cpp-build/suc-linear-bench --shape \$s --batches 1,4,16 --gb 16 --min-mb 2048 --samples 7 \
      --json experiments/results/${TS}_cpp_metal_linear_\$s.json; done
  HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/k3_cold_cache.py --shapes 8b --methods _K256 --no-decode \
    --gb 16 --min-mb 2048 --samples 7 --json experiments/results/${TS}_k3_cold_cache_suc_8b.json" \
  > logs/m2_bench_${TS}.log 2>&1 &
```

In sequenza, mai in parallelo con altri benchmark. Stessi `--gb / --min-mb / --samples` su entrambi i lati. Entrambi leggono
la stessa cache di tuning, quindi le esecuzioni C++ usano gli stessi `codebook_in` / `mma_min_batch` di MLX: codebook in threadgroup
sull'M1 Ultra. Confrontare il `time_us` delle righe con uguali (`method`, `batch`, shape).

## Milestone 3: grafo del modello denso Qwen3 e generazione greedy

Un decoder denso Qwen3 (`suc/model.hpp`, `src/metal/model.cpp`) sul backend della milestone 2, costruito a partire dal solo file
SUC1, e lo strumento `suc1-generate` (token id in ingresso, token id greedy in uscita).

### Progettazione

- **Descrizione del modello** da `directory.architecture` (il `config.json` HF): hidden / intermediate size, layer,
  head, KV head, head_dim, vocab, `rms_norm_eps`, `rope_theta`, `tie_word_embeddings`, `torch_dtype` (il dtype delle
  attivazioni, bf16 per Qwen3). Le impostazioni non supportate falliscono in modo esplicito: MoE (`qwen3_moe`; aggiunto nella milestone 4), `rope_scaling`, bias dell'attention,
  sliding window, `hidden_act` diverso da silu, head_dim non multiplo di 32. I ruoli dei tensori provengono da una tabella dei nomi
  per famiglia (`NameTable`, nomi del checkpoint come in `suc.models.adapters`).
- **I pesi restano nella mappatura.** Le proiezioni SUC sono `metal::Linear` (vq / additive, rotazioni, residui). L'embedding
  / LM head (affine 4-bit nei container attuali) e le proiezioni non SUC (affine, raw / bf16) sono letti
  in place da un kernel di gather e da uno di matvec. I pesi delle norm sono letti in place quando memorizzati nel dtype delle attivazioni, altrimenti
  convertiti una volta (qualche KB). Nulla viene decodificato in pesi densi.
- **Op = le op del runtime Python, nell'ordine delle operazioni di MLX** (`metal/suc_model_kernels.metal`, scritto a mano; accodato
  al `suc_kernels.metal` generato al momento della build):

  | op | runtime Python | kernel C++ | bit uguali |
  |---|---|---|---|
  | embedding | peso decodificato (`q·s`, `+ b` in float32) convertito in bf16, gather di riga | `suc_embed_affine` / `_dense` | sì |
  | RMSNorm, q/k-norm | `mx.fast.rms_norm` (ordine `rms_single_row`, N ≤ 4096) | `suc_rms_norm` | sì |
  | RoPE | `mx.fast.rope` (`rope_single`, non tradizionale, `std::log2(base)`) | `suc_rope` | sì |
  | attention | `mx.fast.scaled_dot_product_attention` → MLX `sdpa_vector` (1 passata) | `suc_sdpa_vector` (sorgente MLX, mask / sink rimossi) | sì |
  | SwiGLU | `swiglu` di `mlx_lm` (`silu(g)·u` compilato, op in bf16) | `suc_swiglu` | sì (bf16, f32; non f16) |
  | somma residua | `+` in bf16 | `suc_add` | sì |
  | q / k / v, o, gate / up, down | `SucLinear` / `SucResidualLinear` (un modulo per proiezione) | stadi di `metal::Linear` | sì |
  | RMSNorm + rotazione di input condivisa | `rotation.rms_norm_rht` (K6, `fuse_elementwise("norm")` per dispositivo) | `suc_rot_prologue_blocks` (generato) | sì |
  | LM head | `embed_tokens.as_linear` = matmul MLX del peso bf16 decodificato | `suc_matvec_affine` / `_dense` (stessi pesi per elemento, somma float32) | no: ordine di somma |
  | argmax | `mx.argmax` (primo indice del massimo) | `suc_argmax` | sì |

  I kernel RoPE e SwiGLU sono stati verificati bit-esatti rispetto a MLX su 1M valori bf16 casuali prima del porting
  (SwiGLU in float16 non lo è: il 3.6 % degli output differisce; nel perimetro rientrano solo modelli bf16).
- **Esecuzione**: un command buffer per passo di decode, un compute encoder concorrente, una memory barrier su ogni vera
  dipendenza, nessuna sincronizzazione con l'host all'interno di un passo. Un'arena di attivazioni preallocata (dimensionata per `--chunk` righe) e la KV cache
  (`[position][kv head][head_dim]` per layer, così la proiezione v e la RoPE di k scrivono le loro righe direttamente nella
  cache; `sdpa_vector` prende gli stride). q|k|v e gate|up condividono UNA sola rotazione di input quando il loro `seed_in` è
  uguale (file della pipeline di compressione: sempre) e vengono eseguiti stadio per stadio (prima i kernel di lookup insieme, poi le
  rotazioni inverse, poi i residui); con la scelta tarata `fuse_elementwise("norm")` (come il runtime Python) la norm
  e quella rotazione sono un solo lancio. Entrambi sono switch A/B (`--no-shared-rotations`, `--fuse-norm auto|on|off`) e
  non cambiano un bit.
- **Prefill**: `--prefill batch` (predefinito) esegue il prompt a chunk di `--chunk` (64) token per command buffer: le
  proiezioni a batch n (il percorso di lookup "mma" da `mma_min_batch` righe), attention causale tramite `sdpa_vector` con
  un threadgroup per (head, riga di query) e `n_keys = N − n + t + 1` (la regola `do_causal` di MLX). `--prefill token`
  esegue il prompt come passi di decode, cioè esattamente ciò che calcola un'esecuzione Python token per token.
  Il prefill in batch **non** è bit-identico ai passi di decode, per scelta progettuale (mantiene i kernel di batch; i token
  greedy dopo un prompt di 2,048 token differiscono per la prima volta dopo pochi passi su Qwen3-8B / 30B-A3B). Diagnosi (Qwen3-8B final,
  M1 Ultra, `prefill_logits_sha256` dell'ultimo token del prompt ora in `--json`): esattamente tre fonti, ciascuna confermata
  rimuovendola: (1) il lookup simdgroup-matrix ("mma") dei tile da `mma_min_batch` righe (`$SUC_MMA_MIN_BATCH=100000`
  mantiene ogni tile sul percorso vec); (2) lo split JS della riduzione del residuo strutturato, che si riduce con le righe per
  threadgroup (`$SUC_RESIDUAL_EXACT=1` mantiene il JS a una riga); (3) attention: un passo di decode da 1,024 chiavi esegue il
  kernel a due passate sui dispositivi 'd', una riga di batch quello a passata singola. Su un prompt di 900 token, rimuovendo (1)+(2) si ottiene
  l'hash del prefill per token; su 2,048 token è uguale a `--prefill token` con `SUC_SDPA_2PASS=off`. Rimuoverle costa
  lo speed-up (2,048 token: 6.7 s in batch, 19.8 s con (1)+(2) rimossi, 27.2 s token per token), quindi la differenza è
  accettata e invece misurata: `scripts/prefill_batch_vs_token.py` (KL / top-1 per posizione rispetto al prefill per token;
  report §1c).
- **Riferimento Python** (`suc/runtime/suc1_model.py`): `load_suc1_model(path, group=False)` costruisce il modello mlx-lm
  dell'architettura del file senza alcun checkpoint: tensori raw come memorizzati, altri tensori non SUC come la loro decodifica di
  riferimento convertita nel dtype del modello (= la rappresentazione che la pipeline di compressione valuta), tensori SUC tramite
  `convert_to_compressed` sui codici memorizzati. `group=False` è il piano di esecuzione C++ (un `SucLinear` per
  proiezione); `group=True, fuse=True` è il runtime Python predefinito (fuso). `step_hidden` / `run_tokens` /
  `greedy` eseguono il passo di decode di mlx-lm (mask None, KV cache) e restituiscono gli hidden state per layer.

### Strumento

`suc1-generate FILE (--tokens 1,2,3 | --tokens-file F) [--steps N] [--prefill batch|token] [--chunk N]
[--logits-out L.npy] [--hidden-out H.npy] [--nll-out N.npy] [--force 4,5,6] [--json OUT.json] [--stop-eos]
[--max-context N] [--dtype bf16|f16|f32] [--no-shared-rotations] [--fuse-norm auto|on|off] [--tuning-cache P|-]`

Stampa `generated: ids`, poi (su stderr) tempo di caricamento / compilazione, ms/token di prefill e decode (wall, tempo GPU del
command buffer, tempo di encode sull'host) e la memoria: file mappato (e quanta parte di esso è residente nella page cache,
`mincore`), array di esecuzione costruiti al caricamento, KV cache, arena, workspace, footprint / dimensione residente del processo (le
letture della GPU dalla mappatura non compaiono nella dimensione residente del processo). `--logits-out` / `--hidden-out` scrivono i logit
`(positions, vocab)` e gli hidden state `(positions, layers + 1, hidden)` di ogni posizione; `--nll-out` la NLL di ciascun
token successivo del prompt (`--steps 0`: solo scoring); `--force` fornisce token dati invece dell'argmax (teacher forcing).

### Test

`tests/unit/test_cpp_generate.py` (saltato quando `suc1-generate` non è compilato):
- file Qwen3 sintetici scritti con `write_suc1` (embedding affine 4-bit; per ruolo un codec diverso: additive
  M = 2 / 3, vq K = 16 / 64 / 256, G = 4 / 8, rotazioni in_out e su un solo lato, residui CSR int8 / fp16 e a colonne
  int8), tied e untied (LM head raw bf16), con la fusione della norm e le rotazioni condivise attive / disattive, e un
  blocco alle larghezze di Qwen3-8B (hidden 4096: RMSNorm a 1024 thread e il prologo 4096; intermediate 3072 = 12·256:
  H_12): ogni hidden state di ogni posizione bit-identico al runtime Python (`group=False`), KL dei logit
  < 1e-4, accordo greedy 1.0;
- prefill in batch (chunk 3 e 64) rispetto al prefill per token: prima posizione identica, L2 relativa degli hidden state
  < 2e-2, KL < 1e-3, stessi token greedy; teacher forcing; id fuori vocabolario rifiutati;
- `-m model`: il container di sviluppo, prompt di 13 token + 64 token greedy: tutti i 77 × 29 hidden state bit-identici, e la
  sequenza greedy libera di Python è uguale a quella C++.

`benchmarks/cpp_generate_check.py` (non è un test) produce i numeri sotto: tre prompt fissi, confronti teacher-forced e
liberi rispetto a entrambi i runtime Python, tempi di decode / prefill, perplexity su WikiText-2.

### Risultati (laptop M3, 2026-10-02)

Container di sviluppo `compress_qwen3_0.6b/container/model.suc1` (196 proiezioni additive M=4 K=256 G=8 con residui CSR int8
e rotazioni in_out, embedding / LM head tied affine 4-bit). Attivazioni bf16; `--prefill token` per le righe di
accordo (il passo di decode per ogni posizione); Python = `load_suc1_model(group=False)` salvo "default".

| prompt (token) | posizioni | hidden state bit-identici | logit bit-identici | KL media / max | top-1 (teacher-forced) | greedy libero, 64 token |
|---|---:|---:|---:|---:|---:|---:|
| history (13) | 76 | 100 % (76 × 29) | 99.988 % | 1.4e-9 / 1.0e-7 | 1.000 | 64 / 64 identici |
| code (4) | 67 | 100 % | 99.986 % | 1.1e-10 / 7.2e-9 | 1.000 | 64 / 64 |
| short (5) | 68 | 100 % | 99.982 % | 2.6e-10 / 1.6e-8 | 1.000 | 64 / 64 |

- Un'esecuzione libera più lunga (prompt code, 400 token greedy, 404 posizioni): 400 / 400 token identici al runtime Python
  del piano.
- Gli unici valori diversi sono i logit (max |Δ| 0.0625, un ulp bf16 per logit di grandezza 8 – 16), dovuti
  all'ordine di somma dell'LM head (deviazione 1).
- Rispetto al **runtime Python predefinito (fuso)**: KL media 5.9e-4 – 1.0e-3 (max 1.0e-2), top-1 teacher-forced 1.000,
  greedy libero 64 / 64 identici su tutti e tre i prompt. Le sue proiezioni fuse (e, dove attivato dal tuning, il prologo
  SwiGLU) arrotondano diversamente dal piano per proiezione, quindi questa è la dispersione attesa degli arrotondamenti bf16 tra due
  piani di somma, non un errore del C++.
- **Perplexity** (test WikiText-2, 4 finestre × 512 token, prefill C++ in batch con chunk da 128 rispetto a una chiamata mlx-lm
  sull'intera finestra): C++ 33.36, runtime Python del piano 33.31, runtime Python predefinito 33.33 (+0.15 % / +0.11 %); |ΔNLL| medio
  0.036 nat per token, l'entità di un arrotondamento bf16 dei logit. Prefill in batch rispetto a prefill per token in C++ (260
  token): KL media 1.2e-4, top-1 1.000.

Tempi (indicativi: laptop M3, job di altri agenti sulla stessa GPU; B = 1, sincrono per token, mediane):

| | C++ `suc1-generate` | runtime Python del piano | runtime Python predefinito |
|---|---:|---:|---:|
| decode ms/token (prompt di 13 token, 64 token) | 9.4 (GPU 8.2, encode host 0.7) | 16.1 | 14.4 |
| prefill ms/token (prompt di 256 token) | 1.05 (batch, chunk da 64) | 1.10 | 1.08 |

- Il decode C++ è 1.5 – 1.7× più veloce con gli stessi kernel. La causa probabile è l'overhead di costruzione del grafo per op e di lancio
  del passo Python (non profilato); il passo C++ è un unico command buffer codificato in ≈ 0.7 ms. Il tempo di prefill
  è lo stesso su entrambi i lati.
- Le rotazioni di input condivise fanno risparmiare ≈ 1.2 ms di GPU per token (A/B, 4 esecuzioni alternate, miglior tempo GPU: 8.4 vs 9.6 ms); il
  lancio fuso norm + rotazione è neutro sull'M3 (migliore 8.40 vs 8.44 ms). Chunk di prefill 16 / 64 / 128: 1.49 / 1.11 /
  1.06 ms/token; prefill token per token 12 ms/token.

Memoria (decode, contesto di 79 posizioni): file mappato 320.4 MB (tutto nella page cache, `mincore`);
array di esecuzione costruiti al caricamento 8.3 MB (segni di Hadamard, puntatori di riga CSR e colonne u16); KV cache 9.1 MB
(115 KB per posizione); arena 6.9 MB (chunk da 64 righe); workspace 0.7 MB. Footprint del processo 94 MB, residente 57 MB: le letture
della GPU dalla mappatura no-copy non sono contate nella dimensione residente del processo, quindi il totale onesto è mappatura + footprint
≈ 414 MB. Il runtime Python mantiene l'embedding bf16 decodificato (311 MB) in aggiunta ai codici.

### Deviazioni note / limiti

1. **LM head** (chiusa nella milestone 4: l'LM head ora esegue l'ordine gemv di MLX, logit bit-identici; `--legacy-matvec`
   ripristina questo comportamento): il matvec C++ somma ogni riga in un ordine diverso dal matmul di MLX sul peso bf16 decodificato
   (99.98 % dei logit bit-identici, KL ≤ 1e-7). Portare il tiling `gemv` di MLX chiuderebbe la differenza; MLX sceglie il
   tiling in codice host che non è distribuito con la wheel.
2. **L'attention a due passate non è portata** (portata nella milestone 4, verificata sull'M3; vedi lì). MLX passa a `sdpa_vector_2pass` da 1024 chiavi sui dispositivi la cui
   architettura termina in `d` / `s` (l'M1 Ultra: `applegpu_g13d`), e da 4096 chiavi con GQA ovunque. Il C++
   esegue sempre il kernel a passata singola, quindi oltre quelle lunghezze i risultati concordano solo a meno di arrotondamenti. Mantenere le esecuzioni di validazione
   sul Mac Studio sotto le 1024 posizioni.
3. **Prefill in batch ≠ il prefill Python bit per bit**: mlx-lm esegue il prefill con il kernel di attention steel e una mask
   causale; il C++ usa `sdpa_vector` per riga di query. Proiezioni, norm e op element-wise sono identiche riga per riga.
4. RMSNorm sopra 4096 (MLX `rms_looped`, es. hidden 5120) segue l'ordine a ciclo di MLX ricavato dalla struttura del suo sorgente ma
   non è verificata bit-esatta.
5. Le proiezioni non SUC (affine / raw mantenute da un piano) girano sui kernel matvec C++: corrette, non bit-identiche a
   `QuantizedLinear` / `nn.Linear` di MLX.
6. I risultati dipendono dalla cache di tuning esattamente come in Python (forma dei blocchi di rotazione, `codebook_in`, `mma_min_batch`,
   `fuse_elementwise`): entrambi leggono `~/.cache/suc/kernel_tuning.json`, quindi concordano se eseguiti sulla stessa macchina con
   la stessa cache. Il C++ non fa mai tuning: su una nuova macchina, caricare prima il file una volta in Python (convert_to_compressed
   esegue l'autotune). Sul laptop, un'esecuzione C++ fatta prima che l'autotune Python avesse scritto le sue voci ha prodotto una
   continuazione greedy diversa dal 7° token, molto probabilmente a causa della forma dei blocchi di rotazione predefinita.
7. CMake rispecchia `build.sh` (nuovi sorgenti, il secondo file di kernel incorporato, `suc1-generate`) ma non è stato eseguito (niente CMake
   sul laptop di sviluppo).

### Comandi per il Mac Studio (Qwen3-8B)

```bash
cd ~/Projects/SUC-LM
cpp/build.sh ~/suc-local/cpp-build
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py tests/unit/test_cpp_metal.py -q
F=experiments/artifacts/compress/compress_qwen3_8b/container/model.suc1
~/suc-local/cpp-build/suc1-generate $F --tokens 785,3840,315,279,18484,3493,12033,304,279,11128,45387,9294,11 --steps 64
mkdir -p logs && TS=$(date -u +%Y%m%dT%H%M%SZ) && HF_HUB_OFFLINE=1 nohup .venv/bin/python \
  benchmarks/cpp_generate_check.py --file $F --steps 64 --ppl-windows 8 --ppl-seq-len 512 \
  --out-dir experiments/results > logs/m3_check_8b_${TS}.log 2>&1 &
```

In sequenza, mai in parallelo con altri job GPU. Il file 8B è untied: il suo `lm_head.weight` gira sul matvec affine o
denso, a seconda di ciò che il piano ha memorizzato. Tutte le posizioni restano sotto 1024 (deviazione 2). Il lato Python carica due
copie del modello (runtime del piano e predefinito, ≈ 2 × la dimensione del file).

## Milestone 4: piano fuso, tokenizer, MoE, decode in pipeline

### Gruppi di proiezioni fuse (il piano del runtime Python predefinito)

Il runtime Python predefinito (`convert_to_compressed(group=True, fuse=True)`, cioè `load_suc1_model(path, group=True,
fuse=True)`) raggruppa q|k|v e gate|up in un `SucInputGroup`: una rotazione di input condivisa, e i membri con lo stesso layout
dei codici concatenati per righe in UN solo `SucFusedLinear` / `SucFusedResidual`. Il default C++ è ora quel piano
(`--projections fused`; `--projections separate` mantiene il piano per proiezione della milestone 3 = `group=False`):

- **Cluster** = `SucInputGroup.__init__`: i membri con `FusedLinear::key` uguale (= `group_fusable_key`: in, bit,
  (M, K, G), seed_in, rotate_out, dimensioni di rotazione del layout blocks) formano un `FusedLinear` (≥ 2 membri); gli altri girano
  da soli sulla x̃ condivisa. I gruppi richiedono un seed_in comune (file della pipeline di compressione: `rotation_seeds`).
- **Stadi** (un lancio ciascuno per l'intero gruppo, una barrier tra l'uno e l'altro):
  1. `suc_fused_lookup_vec / _mma`: i corpi di vq_matmul con `suc.kernels.suc_linear._PATCHES` (la riga r legge l'insieme di codebook
     `SUC_SEG(r)`; confini dei segmenti SB1..SB3 come costanti template; codebook in staging (CBTG) solo quando ogni
     confine è multiplo delle righe del threadgroup, come `fused_matmul_metal`);
  2. `suc_rot_blocks{2,3}[_split]`: `rotation._blocks_kernel(nseg, split)`, la rotazione inversa segmentata con il
     piano di `rotation._blocks_plan` (forma dei blocchi da `block_config(max size)`, limitata dal segmento più piccolo) — il
     motivo per cui i piani fuso e separato differiscono per arrotondamento sulle shape reali;
  3. `suc_structured_residual` con S segmenti (`SucFusedResidual.residual`: righe CSR concatenate, blocchi di colonne,
     una scala int8 per membro, `row_quads` = le righe di ogni membro % 4 == 0), arrotondamento dentro il kernel.
  Senza residuo la rotazione di output scrive ciascun membro direttamente nel suo buffer (split); altrimenti l'output fuso
  (B, Σ out) viene distribuito come viste per una riga (q / k / gate / up; v viene copiato nella V cache da
  `suc_copy_cols`) o tramite copie con stride (`suc_copy_cols`) per le righe di batch.
- **In place** (predefinito; `--fused-layout copy` per A/B): il layout Python concatena gli indici dei membri per
  flusso di codebook ((M, Σ out, words)), che il file non ha. I kernel solo C++ `suc_fused_lookup_*_tab`
  (generati: gli stessi corpi, con solo `packed + (m·OUT + row)·WORDS` e `scales[row]` sostituiti da una tabella di offset per (segmento,
  flusso)) leggono gli indici e le scale f16 di ogni membro dove si trovano nella mappatura. I codebook
  (S, M, K, G), i segni di output e gli array di residuo concatenati sono costruiti al caricamento (piccoli).
- **SwiGLU + rotazione** (`SucMLP`, `rotation.swiglu_rht`): `suc_rot_prologue_blocks` con PRO = 2, usato con il piano
  fuso quando la cache di tuning sceglie `fuse_elementwise("swiglu")` (`--fuse-swiglu auto|on|off`); il runtime Python del piano
  non lo fonde mai. (Disattivato sull'M3 per la sua scelta di tuning.)

### LM head nell'ordine gemv di MLX

L'LM head tied / untied (e qualsiasi proiezione non SUC mantenuta affine / raw) ora esegue `suc_gemv_affine` /
`suc_moe_gemv`: l'ordine di accumulo del `gemv` non trasposto di MLX (`GEMVKernel`, TN = 4; SN = 4 per in ≤ 64, BN = 8 per
in ≥ 16·out, altrimenti SN = 32, BN = 1 — determinato rispetto a MLX 0.32.3 per il router MoE, vedi sotto) sull'elemento del peso
decodificato `T(q·s + b)`. I logit sono quindi bit-identici al runtime Python per una riga (deviazione 1 della milestone 3
chiusa; `--legacy-matvec` mantiene i vecchi kernel per A/B). Le righe del prefill in batch differiscono ancora (lì MLX esegue una gemm).

### Attention di decode a due passate

MLX passa la sua attention di decode da `sdpa_vector` a `sdpa_vector_2pass_1` + `_2` sui contesti lunghi; il numero di blocchi
è scelto in codice host che la wheel non distribuisce. I kernel sono ora portati (`suc_sdpa_2pass_1`, anche la versione
specializzata per GQA `suc_sdpa_2pass_1_gqa`, e `suc_sdpa_2pass_2`: sorgente MLX, mask / sink rimossi, `blocks` come costante
template) e `Model::Impl::sdpa2_rule` riproduce il dispatch: due passate quando l'architettura termina in `d` e
n ≥ 1024 chiavi, oppure con GQA e n ≥ 4096; blocks = 64 se il fattore GQA è ≥ 4, altrimenti 32 sugli altri dispositivi, le tabelle `d` / `s`
di MLX negli altri casi. Trovato e verificato sull'M3 da `benchmarks/cpp_sdpa_2pass_probe.py` (candidati C++ rispetto a MLX, hidden state
di ogni posizione): fattori GQA 1 / 2 / 4 / 8 fino a 4160 chiavi, fattore 4 fino a 8400 chiavi, 0 posizioni diverse;
passata singola, altri numeri di blocchi e la variante GQA differiscono tutti da 4096 chiavi in poi. I rami `d` (M1 Ultra) e `s`
sono il codice host di MLX come ricordato, **non verificato** — eseguire la probe sullo Studio. `--single-pass-attention` ripristina il
comportamento della milestone 3. Le righe del prefill in batch usano sempre il kernel a passata singola.

### Decode in pipeline

`Model::generate(first, steps, depth)`: l'argmax di ogni passo scrive il token successivo nel buffer di token della GPU che il
gather dell'embedding del passo successivo legge, così l'host codifica il passo t + 1 mentre la GPU esegue il passo t (fino a `--depth` command
buffer in volo, predefinito 2; l'hazard tracking di Metal ordina l'arena condivisa / KV cache tra command buffer).
Stessi kernel, stessi token (verificato rispetto a `--sync` e al greedy Python). Usato da `suc1-generate` salvo quando si richiede un
output per passo (`--logits-out`, `--hidden-out`, `--nll-out`, `--force`) o si passa `--sync`. Questo nasconde il
tempo di encode dell'host (≈ 0.6–0.9 ms per token sull'M3) dietro la GPU.

Percorso di encode: le risposte del tuning (`codebook_in`, `mma_min_batch`, forma dei blocchi di rotazione) sono memoizzate per M / dimensione (erano
lookup JSON a ogni dispatch), e i call site caldi (kernel di lookup, rotazioni, residui, norm, RoPE,
attention, op element-wise) cercano la loro pipeline tramite una chiave intera (`PsoKey`) invece di formattare la stringa
template a ogni dispatch. A/B misurato (decode sincrono, 5 esecuzioni ciascuno): encode 0.58 ms/token prima e dopo — nessuna
variazione misurabile, quindi il tempo host restante sono le chiamate stesse all'encoder Metal (≈ 1 µs per dispatch). Barrier: la copia di
v nella V cache (piano fuso, una riga) gira in parallelo alle norm di q / k invece che dopo una propria barrier (una barrier
in meno per layer; ≈ 0.7 % del passo di decode, entro il rumore).

### Tokenizer (`suc/tokenizer.hpp`, `src/tokenizer.cpp`, strumento `suc-tokenize`)

Il `tokenizer.json` di Qwen2/Qwen3 in C++ (senza dipendenze): token aggiunti / speciali separati per primi (match letterale
leftmost-longest; flag lstrip / rstrip / single_word / normalized), NFC, la regex del pre-tokenizer Qwen come matcher
scritto a mano (ordine di alternanza di Oniguruma), ByteLevel, BPE con il ciclo di merge di HF (heap dei rank; cache delle parole), decode ByteLevel
con la semantica di `from_utf8_lossy` di Rust. Le impostazioni non supportate (altri normalizer / regex / modelli, dropout,
byte_fallback, …) falliscono in modo esplicito. `resolve_tokenizer_path(tokenizer_id)` mappa il `tokenizer_id` del SUC1
(`Org/Name@rev;tokenizer.json:sha256:…`) sulla cache HF (`$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`)
e verifica lo sha256. Le tabelle Unicode (`src/unicode_tables.inc`, generate da `tools/gen_unicode_tables.py`, `--check`
rileva se sono obsolete) prendono `\p{L}` / `\p{N}` / `\s` dal motore regex di HF `tokenizers` stesso (Unicode 16; `unicodedata`
di Python differisce su 4,924 lettere e 80 numeri) e NFC da `unicodedata` meno i 109 caratteri che i dati NFC più vecchi di HF
non normalizzano. La libreria non ha chat template (il `--chat` dello strumento genera il template Qwen3
a turno singolo); l'input UTF-8 non valido viene rifiutato.

`suc1-generate --prompt "text" | --prompt-file F [--tokenizer tokenizer.json] [--text]`: codifica il prompt, stampa
`prompt: ids`, `generated: ids` e `text: <decoded generation>` (l'ultimo output, può occupare più righe). `--chat [--system
TEXT] [--no-think]` avvolge il prompt nel chat template Qwen3 per un turno utente (+ turno di sistema) con il generation
prompt, come lo genera `apply_chat_template(..., add_generation_prompt=True, enable_thinking=...)` (id verificati rispetto a
transformers); `--stop-eos` poi si ferma a `<|im_end|>`.

### Sampling (`suc/sampling.hpp`, `src/sampling.cpp`)

`--temperature T [--top-k K] [--top-p P] [--min-p P] [--seed S]`: sull'host, dall'ultima riga di logit: top-k, poi
top-p (il più piccolo prefisso decrescente che raggiunge P), poi min-p, poi un'estrazione da `std::mt19937_64(seed)` (softmax float64 di
logit / T). Riproducibile per seed, NON le estrazioni di mlx-lm (l'RNG di MLX non è reimplementato); temperatura 0 o top-k 1
è l'argmax greedy. Il sampling esegue il ciclo di decode sincrono (la riga di logit viene riletta a ogni passo).

### MoE: qwen3_moe (`suc/switch.hpp`, `src/metal/switch.cpp`, strumento `suc1-moe`)

- `ModelConfig` legge `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `decoder_sparse_step`,
  `mlp_only_layers`, `norm_topk_prob`; un layer è sparso quando `num_experts > 0`, `(l + 1) % decoder_sparse_step == 0`
  e l ∉ `mlp_only_layers` (mlx-lm `Qwen3MoeDecoderLayer`); il suo MLP è una semplice RMSNorm, poi `MoeBlock`.
- `SwitchLinear` = percorso "gather" di `SucSwitchLinear` (BT = 1, un tile per riga): rotazione con segni per riga
  (`suc_rot_blocks_rows` = `rotation._blocks_kernel(1, False, row_signs=True)`, oppure il lancio semplice quando tutti gli expert
  condividono seed_in), `suc_switch_lookup_vec / _mma` (`suc_switch._PROLOGUE` + corpi di vq_matmul; CBE = 0 quando ogni expert
  usa gli stessi codebook: il layout con expert stack condiviso), rotazione inversa, `suc_switch_residual` (CSR impilato),
  un solo arrotondamento. I residui strutturati a colonne (`columns_int8 / columns_fp16`, es. i file finali di Qwen3-30B-A3B) passano
  per `suc_switch_structured_residual` (`suc_switch._COLRES_SRC`): insiemi di colonne per expert concatenati con una
  tabella di puntatori alle colonne (i conteggi possono differire per expert), int8 con scale per colonna o fp16, più la coda CSR opzionale,
  un lancio per proiezione per tutti gli expert selezionati, arrotondato al dtype delle attivazioni dentro il kernel (copiati al
  caricamento, come gli array CSR). Gli array degli expert sono letti in place: `StackLayout::Table` (prologo solo C++ `suc_switch_lookup_*_tab`
  con una tabella di offset per expert) a meno che gli expert non siano già consecutivi (`Stacked`, il kernel testuale).
- `MoeBlock` = `Qwen3MoeSparseMoeBlock` di mlx-lm con quegli expert, senza sincronizzazione con l'host: router `suc_moe_gemv` (ordine gemv di MLX,
  bit-esatto), `suc_moe_route` (softmax(precise) = `softmax_single_row` di MLX; top-k con la semantica di `argpartition(g, -k)[..., -k:]`
  = gli ultimi k di un argsort stabile crescente, cioè ordine crescente dei gate, parità risolte per indice di expert (misurato: 0
  discrepanze su 800 righe incl. parità); take_along_axis; somma `norm_topk_prob` sequenziale in bf16 a partire da 0, come la riduzione
  per riga di MLX), `suc_repeat_rows`, gate / up / swiglu / down, `suc_moe_combine` (prodotti bf16, somma bf16 sequenziale su k
  a partire da 0 — MLX non accumula questa riduzione in float32).

- **Default del lookup vec degli expert (decode, sotto `mma_min_batch`).** I codebook sono messi in staging nella memoria del threadgroup quando
  il `codebook_in` tarato del dispositivo per quell'M è "threadgroup" (la scelta per dispositivo del lookup denso da
  `kernel_tuning.json`; fallback incorporato `applegpu_g13*` → threadgroup) e stanno in 32 KB; 2 righe per simdgroup
  (Python `lookup_rows(1)` = 4), `LOOKUP_SG` = 8 simdgroup. Nessuno dei due cambia l'aritmetica di alcuna riga: i logit sono
  identici byte per byte al layout Python (CBTG = 0, 4 righe) su ogni modello C++ e nei piccoli controlli incrociati
  ([decode_speed_vs_ds4.md](../docs/decode_speed_vs_ds4.md) §8). Override: `SUC_SWITCH_CBTG=0|1`, `SUC_SWITCH_ROWS`,
  `SUC_SWITCH_SG`.

### Modifiche agli strumenti

Nuove opzioni di `suc1-generate`: `--projections fused|separate` (predefinito fused), `--fused-layout inplace|copy`,
`--fuse-swiglu auto|on|off`, `--prompt TEXT | --prompt-file F`, `--tokenizer PATH`, `--text`, `--sync`, `--depth N`,
`--legacy-matvec`, `--single-pass-attention`, `--chat [--system TEXT] [--no-think]`, `--temperature / --top-k / --top-p / --min-p / --seed`; il
riepilogo su stderr e `--json` riportano i gruppi fusi / la fusione SwiGLU. I file qwen3_moe girano con
lo stesso strumento. `suc-tokenize (--tokenizer F | --suc1 FILE) [--encode-lines F] [--encode-json F] [--decode-json F]
[--text T] [--decode IDS] [--skip-special] [--add-special-tokens] [--json]`. `suc1-moe FILE --layer L --x X.npy
--y-out Y.npy [--stack PROJ --eidx E.npy] [--routing R.npy] [--route-out R.npy] [--layout auto|stacked|table]`: un blocco
MoE (o un expert stack con id di expert dati) su righe date (il controllo incrociato del pytest).

### Test

- `tests/unit/test_cpp_generate.py` (47 test al 2026-10-15, 9 dei quali `-m model`, incl. il test a due passate su contesto lungo: fattori
  GQA 2 / 4 / 8, 24 posizioni oltre il punto di passaggio di MLX): ogni confronto verifica hidden state E logit bit-identici.
  Nuovi: piano fuso rispetto al runtime Python predefinito su tre layout di codici sintetici (q|k|v additive M = 2 con residui CSR / a
  colonne e gate|up vq con un residuo fp16; rotazione split senza residuo con q|v fusi + k da solo; rotazioni su un solo
  lato) × fusione della norm auto / on / off (incl. il prologo SwiGLU forzato attivo, Python `FUSE_ELEMENTWISE = True`);
  seed condivisi con il piano separato; prefill in batch fuso = prefill in batch separato bit per bit; in pipeline = `--sync`
  = greedy Python; qwen3_moe (top-2, top-4 con un blocco denso `mlp_only_layers`, `norm_topk_prob` False) × runtime del piano / predefinito;
  prefill in batch MoE; `-m model`: il container di sviluppo con il piano fuso rispetto al runtime predefinito (77 × 29 hidden
  state, 64 token greedy liberi) e testo `--prompt` (id = transformers, testo stampato = decode di transformers).
- `tests/unit/test_cpp_moe.py` (20, 2026-10-15): SwitchLinear rispetto a `SucSwitchLinear.rows(x, eidx, "gather")` bit per bit (quattro varianti
  MoE incl. un file con expert stack condiviso scritto con `suc.quant.stacks`, 3 proiezioni, N = 1/3/8/40 con expert
  ripetuti, f16 / bf16, layout auto / stacked / table); MoeBlock rispetto a `layers[L].mlp(x)` (id di routing, punteggi e output
  bit-identici, n = 1/3/32/k; percorso con routing dato); un blocco alle larghezze di 30B-A3B (hidden 2048, E = 128, k = 8,
  moe_intermediate 768 = 12·64 con rotazioni H_12).
- `tests/unit/test_cpp_tokenizer.py` (23, 2026-10-15): id esatti rispetto a `transformers` (tokenizer di Qwen3-0.6B, offline) su 2,891
  righe di test di WikiText-2 (298,938 token), le prime 2,000 righe come un unico documento (216,940 token), 59 casi limite
  scritti a mano (NFC / NFD, CJK, jamo, emoji ZWJ / toni della pelle / bandiere, arabo, devanagari, sequenze di spazi, tab, \r\n,
  contrazioni in entrambi i casi, cifre, codice, token speciali incorporati, stringa vuota) e 2,330 stringhe casuali: 0
  discrepanze, con `add_special_tokens` False / True; decode (con / senza token speciali, 300 sequenze di id casuali
  incl. UTF-8 parziale) = transformers; round trip su input NFC; flag dei token aggiunti su un tokenizer.json modificato; obsolescenza
  delle tabelle; risoluzione del tokenizer_id e sha256 non corrispondente.
- Inoltre: id del prompt `--chat` = `apply_chat_template` (`-m model`; con / senza turno di sistema, enable_thinking attivo /
  disattivo); sampling sul file piccolo: stesso seed → stessi token, temperatura 0 / top-k 1 → greedy, seed diversi →
  estrazioni diverse.
- C++: `suc-tests` 1078 verifiche (build del 2026-10-13; parti del tokenizer: mappa dei byte, UTF-8, split della regex, NFC, BPE su un file giocattolo; sampler:
  casi greedy, supporti top-k / top-p / min-p, riproducibilità, frequenze entro 4σ della softmax);
  `suc-metal-tests` 705 verifiche (+ gemv / combine / parità del top-k MoE).

### Risultati (laptop M3, 2026-10-02)

Container di sviluppo `compress_qwen3_0.6b/container/model.suc1` (come nella milestone 3), `benchmarks/cpp_generate_check.py`
(tre prompt, `--prefill token`, 64 token greedy, confronto teacher-forced a ogni posizione):

| piano C++ | runtime Python | hidden state bit-identici | logit bit-identici | KL max | greedy libero (64) |
|---|---|---:|---:|---:|---:|
| `--projections separate` | piano (`group=False`) | 100 % (3 prompt, 67–77 × 29) | 100 % | 0 | 64 / 64 identici |
| fused (predefinito) | predefinito (`group=True, fuse=True`) | 100 % | 100 % | 0 | 64 / 64 identici |

- I due piani differiscono tra loro (C++ fuso rispetto a C++ separato sul prompt di 13 token + 64 token: L2 relativa degli
  hidden state 3.7e-3; le rotazioni segmentate usano altre forme di blocchi), e ciascun piano C++ è uguale alla sua controparte
  Python bit per bit, quindi il confronto è discriminante. I layout fusi in place e copiati danno bit identici.
- Perplexity (test WikiText-2, 4 × 512 token, prefill C++ in batch, piano fuso): C++ 33.369, Python piano 33.312, Python
  predefinito 33.326 (invariata rispetto alla milestone 3: le righe di batch differiscono ancora per i kernel di attention / gemm).
- Tokenizer: 0 discrepanze rispetto a transformers su 2,891 righe di test di WikiText-2 e sui corpora di casi limite / casuali
  (vedi Test); `--prompt` sul container di sviluppo dà i 13 id del prompt della milestone 3. Encode ≈ 10–19 MB/s
  (2.3–4.5 M token/s) su un core dell'M3, contro 3 MB/s di HF `tokenizers` da Python con un thread.
- MoE: ogni piccolo file qwen3_moe (top-2 / top-4 / layer denso / senza `norm_topk_prob`, runtime del piano e predefinito) e il
  blocco alle larghezze di 30B-A3B: hidden state e logit bit-identici.

Tempi (indicativi: laptop M3, B = 1, mediane di 63–127 passi di decode; due esecuzioni del benchmark a un'ora di distanza, la seconda
più lenta in tutto, quindi intervalli):

| | ms/token | GPU ms/token |
|---|---:|---:|
| C++ fuso, in pipeline (predefinito) | 8.2 – 10.1 | 8.2 – 10.2 |
| C++ fuso, `--sync` | 10.8 – 11.8 | 8.2 – 10.4 |
| C++ separato, in pipeline | 8.2 – 9.5 | 8.3 – 9.5 |
| C++ separato, `--sync` | 9.5 – 11.0 | 8.2 – 9.6 |
| runtime Python predefinito | 14.4 – 15.8 | |
| runtime Python del piano | 14.5 – 16.7 | |

- Il decode in pipeline elimina la parte host del passo: tempo wall = tempo GPU (sync: GPU + ≈ 1.4–1.6 ms di encode,
  commit e attesa). C++ predefinito rispetto al runtime Python predefinito: 1.5–1.75× meno ms per token con gli stessi kernel.
- Il piano fuso **non** riduce il tempo GPU sull'M3 (fuso rispetto a separato entro ±10 %, nella seconda esecuzione il fuso è 8 %
  più lento). È sull'M1 Ultra, per i cui kernel con codebook in staging la fusione è stata progettata, che va misurato
  (comandi per lo Studio sotto); nessuna affermazione dal laptop.
- Memoria (decode, 79 posizioni): file mappato 320.4 MB; array di esecuzione costruiti al caricamento 17.3 MB (layout fuso
  in place; il layout copiato richiede 164.7 MB e ha alzato la dimensione residente di picco da 194 a 380 MB in un'esecuzione A/B); KV
  cache 9.1 MB; arena 7.6 MB; workspace 0.6 MB; footprint del processo 111 MB, residente 74 MB.

MoE, indicativo: un file sintetico a 2 layer alle larghezze di Qwen3-30B-A3B (hidden 2048, 32 / 4 head da 128, 128 expert,
top-8, moe_intermediate 768; vq K = 256 G = 8 ≈ 1 bpw + 0.5 % di residuo int8 su attention e up_proj; vocab 4096;
176 MB): C++ 0.73 ms/token in pipeline (1.03 ms `--sync`, GPU 0.73 ms), runtime Python 3.30 ms/token (piano e predefinito
uguali): ≈ 0.36 ms per layer MoE in C++. Non è una misura di Qwen3-30B-A3B (pesi sintetici, nessuna vera LM head).

### Deviazioni note / limiti

1. **Prefill in batch MoE sopra 32 righe di expert**: `SucSwitchLinear.rows` di Python passa al suo percorso "tiled" (raggruppamento
   per expert lato host, poi tile da BT righe) e `SwitchGLU` ordina le righe da 64 indici in su; il C++ esegue sempre il
   percorso gather senza sincronizzazione. Misurato: uno stack a N = 128 coincide sul 99.98–99.99 % dei valori (L2 relativa da 7e-9 a
   3e-5); un blocco a n = 32, k = 2 coincide sul 98.9–99.7 % (L2 relativa 2–4e-4). Il decode (N = top_k righe) è esatto.
2. Attention a due passate (deviazione 2 della milestone 3): portata e verificata sull'M3; la regola del numero di blocchi dell'M1 Ultra è
   non verificata finché `benchmarks/cpp_sdpa_2pass_probe.py` non è stato eseguito lì (mantenere la validazione bit-esatta sotto 1024 posizioni
   fino ad allora).
3. Il piano fuso richiede che gli array dei membri stiano in un'unica vista mappata per il layout in place; altrimenti (file oltre
   `maxBufferLength / 2` per vista a cavallo di un confine di vista, o layer lato host) ripiega sul layout copiato.
4. La fusione RMSNorm + rotazione è usata solo per N ≤ 4096 (l'ordine MLX bit-esatto); Python fonde anche dimensioni maggiori
   (arrotondamento diverso per hidden 5120+, non una dimensione di Qwen3-8B / 30B-A3B).
5. Il ramo MoE `mma` a BT = 1 (solo se una cache di tuning imposta `mma_min_batch` 1) e le rotazioni per riga con 2^k < 32
   non sono testati / sono rifiutati.

### Comandi per il Mac Studio (Qwen3-8B; Qwen3-30B-A3B quando il suo container esiste)

```bash
cd ~/Projects/SUC-LM
cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-tests && ~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py tests/unit/test_cpp_moe.py tests/unit/test_cpp_metal.py -q
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/unit/test_cpp_tokenizer.py -q    # needs the Qwen3-0.6B tokenizer cached
# M1 Ultra two-pass attention rule (expect "auto ... 0"; if not, the candidate with 0 is MLX's choice -> fix sdpa2_rule)
for kv in 2 1 4; do .venv/bin/python benchmarks/cpp_sdpa_2pass_probe.py --heads 8 --kv-heads $kv \
  --candidates auto,off,std:64,std:128,std:256,gqa:2:128,gqa:4:128; done
.venv/bin/python benchmarks/cpp_sdpa_2pass_probe.py --heads 8 --kv-heads 2 --positions 9300 --start 8192 \
  --candidates auto,std:128,std:256
.venv/bin/python -m pytest tests/unit/test_cpp_generate.py -q -m model -k long_context
F=experiments/artifacts/compress/compress_qwen3_8b/container/model.suc1
~/suc-local/cpp-build/suc1-generate $F --prompt "The history of the printing press begins in the fifteenth century," --steps 64
for p in fused separate; do ~/suc-local/cpp-build/suc1-generate $F --tokens 785,3840,315,279,18484,3493,12033,304,279,11128,45387,9294,11 \
  --steps 128 --projections $p; done            # decode ms/token (pipelined), memory line
mkdir -p logs && TS=$(date -u +%Y%m%dT%H%M%SZ) && HF_HUB_OFFLINE=1 nohup .venv/bin/python \
  benchmarks/cpp_generate_check.py --file $F --steps 64 --ppl-windows 8 --ppl-seq-len 512 \
  --out-dir experiments/results > logs/m4_check_8b_${TS}.log 2>&1 &
```

In sequenza, mai in parallelo con altri job GPU; tutte le posizioni di validazione sotto 1024 (deviazione 2). Il benchmark
ora confronta il piano separato C++ con `group=False` e il piano fuso C++ con il runtime predefinito (frazioni di hidden state
e logit bit-identici, KL, greedy), e misura fuso / separato × in pipeline / sync. Caricare il file 8B una volta
in Python sullo Studio prima della prima esecuzione C++ se la sua cache di tuning non ha le dimensioni dell'8B (il C++ non fa mai tuning).

## Milestone 5: qwen4_exp (Qwen3.8-Flash-Next)

`suc/qwen4.hpp`, `src/metal/qwen4.cpp`, kernel `metal/suc_qwen4_kernels.metal` (accodati al sorgente incorporato).
`suc1-generate` smista su `model_type == "qwen4_exp"`. Riferimento: `suc/models/mlx_qwen4_exp.py` sul runtime SUC1
Python (`load_suc1_model(path, group=False, affine_modules=False)`, `step_hidden` per token); architettura:
[docs/arch/qwen4_exp.md](../docs/arch/qwen4_exp.md).

### Progettazione

- **Grafo** (un token per passo, come `Model.step_hidden`): embedding raccolto in `hc` stream identici; per layer
  l'aggiornamento PLE (layer PLE), il residuo con gate dell'attention (`hc_norm` → mixer BF16 `input_mix_weight_down / up` → `u`,
  pesi di iniezione `2 σ(W_inj x̂ / hc)`), Gated DeltaNet o gated attention su `u`, iniezione in ogni stream, il residuo con gate
  dell'MLP e il MoE; il mixer finale alimenta l'LM head (nessuna norm finale).
- **Pesi in place**: le proiezioni SUC sono `metal::Linear` (una per proiezione: il runtime Python del piano), gli expert
  stack `metal::SwitchLinear` tramite `MoeBlock::encode_experts`; i tensori BF16 (mixer, router, `in_proj_a / b`,
  `index_qk_proj`, `shared_expert_gate`) usano l'ordine gemv di MLX (`suc_moe_gemv`), l'LM head affine `suc_gemv_affine`,
  norm / pesi delle conv / `A_log` / `dt_bias` sono letti in place nel loro dtype memorizzato.
- **Op** (`suc_qwen4_kernels.metal`): ogni tensore che il modello Python materializza in bf16 viene arrotondato nello stesso punto;
  gli scalari float Python incontrano gli array bf16 come valori bf16 (weak typing di MLX: `/ hc`, `/ sqrt(H)`, le scale di q / k della
  delta rule, `1e-6`); le somme per colonna (media degli stream, combine degli expert) accumulano in bf16; il gate PLE `<k_i, q_i>` segue
  la riduzione per riga bf16 di MLX (misurato: N ≤ 64 sequenziale; ≤ 512 un simdgroup; ≤ 1024 128 thread; sopra 1024 thread, 4
  elementi contigui per thread per passo). Gated delta rule = il kernel di mlx-lm (stato float32, in place); attention =
  `sdpa_vector` / `sdpa_vector_2pass` di MLX con il ramo a mask booleana di MLX alimentato dalla selezione QSA (head dim 64 / 96 / 128 /
  256, la regola a due passate della milestone 4), il fallback non fuso di MLX per altre head dim; softmax del router in float32, top-k in
  ordine argpartition, rinormalizzato, punteggi arrotondati a bf16.
- **Indexer QSA**: chiavi grezze dell'indexer messe in cache per token; una chiave di blocco (RoPE(k_layernorm(media di 4 chiavi grezze)) all'inizio
  del blocco) viene calcolata una volta quando il suo blocco è completo; da `budget / ratio` blocchi completi in poi, i punteggi dei blocchi
  `Σ_h relu(q_h · k_b) / √d` e un top-k esatto (ricerca della soglia radix; parità risolte con l'indice di blocco più basso — Python usa
  argpartition, quindi le parità esatte possono essere risolte diversamente) danno la selezione, con la coda incompleta sempre visibile.
- **Tabelle n-gram (mai residenti)**: per token l'host calcola l'hash della storia dei token (contesto di 2 token consapevole dell'EOS, xor dei
  prodotti con i moltiplicatori del layer, modulo i primi per head; i buffer di hash I64 memorizzati sono verificati rispetto al
  ricalcolo al caricamento) e legge le 16 righe dalla mappatura: righe BF16 raw copiate, record affine per riga
  (`suc repack`, `group_size` = larghezza della riga, `scheme: row_minmax`) decodificati `q · s + b` in float32 sulla CPU. **Le viste GPU
  della mappatura escludono le tabelle** (`MappedFile(ctx, f, qwen4_table_ranges(f))`; `Qwen4Model` rifiuta una
  mappatura le cui viste le coprono): un command buffer che usa un buffer no-copy rende residenti tutte le sue pagine — misurato
  sull'M3, una lettura float ogni 128 MB di un file mappato da 8 GB ha caricato tutti gli 8 GB.
- **Cache**: K / V (12 layer sul modello reale), chiavi grezze e chiavi di blocco dell'indexer, stato della conv di Gated DeltaNet (3 righe) e
  stato ricorrente float32, stato della conv PLE (9 righe, dilatazione 3), contesto n-gram; `reset()` le svuota.
- **Prefill esatto per riga** (`--prefill exact` / `$SUC_PREFILL_EXACT=1` dal 2026-10-10; prima era il predefinito; in
  batch, layer-major, `Qwen4Options::layer_prefill`): il prompt in chunk di `--chunk` (64)
  token, ogni chunk layer per layer. Su tutte le righe del chunk insieme, ogni riga con l'aritmetica del suo passo di decode:
  le proiezioni affine (`suc_gemv_affine_rows`: ogni elemento del peso decodificato una volta e applicato a un gruppo di righe,
  4 di default, `$SUC_AFFINE_ROWS_NB`; aritmetica e ordine di somma di ogni riga sono quelli di
  `suc_gemv_affine` a una riga), le proiezioni dense (raw) (`suc_moe_gemv_rows`, stessa idea per `suc_moe_gemv`, 4 righe,
  `$SUC_GEMV_ROWS_NB`), le proiezioni SUC (`Linear::encode_rows_exact`: rotazioni e residuo su tutte le righe, il
  residuo con lo split della riduzione a una riga, il lookup in tile di fino a `max_exact_rows` righe sul percorso vec —
  2 di default, `$SUC_EXACT_TILE`: il lookup vec è limitato dall'ALU oltre ~2 righe per lettura di peso sull'M1 Ultra —
  il cui switch CHUNK è quello del lancio a una riga; `$SUC_PREFILL_ROW_PROJ=1` = una riga alla volta, A/B), il
  passo di Gated DeltaNet (`q4_conv_rows`, `q4_delta_prep_rows`, `q4_gated_delta_rows`: conv, preparazione dei gate e ricorrenza di
  tutti i token del chunk con un lancio ciascuno, lo stato tenuto nei registri; `$SUC_PREFILL_LIN_TOKENS=1` = il ciclo
  per token), il router / shared expert, e gli expert instradati in **tile di expert** (le righe instradate del chunk
  ordinate per expert in tile di `bt` righe, un expert per tile; `suc_switch_lookup_vec` con BT = bt legge i
  codici dell'expert una volta per tile; bt è una potenza di due fino alla massima altezza di tile per cui il lookup vec mantiene la sua
  aritmetica per riga (`SwitchLinear::max_tile_rows`: il suo switch CHUNK), scelta per layer in base a
  Σ tile · (1 + 0.3 (bt − 1)), `$SUC_TILE_BT` la forza); token per token in ordine di posizione sui buffer a un token:
  PLE, KV cache, indexer e attention (i mixer delle hyper-connection girano stadio per stadio sulle righe).
  `$SUC_Q4_PROFILE=1` divide ogni chunk in un command buffer per stadio e stampa il tempo GPU per stadio. Una sincronizzazione con l'host per layer MoE per
  chunk (il piano dei tile; in streaming: il fetch degli expert, in gruppi di token i cui expert stanno negli slot). Stessi bit dei passi
  di decode: i logit e lo sha256 degli stream dell'ultimo token del prompt (`prefill_logits_sha256` / `prefill_streams_sha256` in
  `--json`) sono confrontati con `--prefill token` (passi di decode, 16 per command buffer) in
  `test_batched_prefill_bit_identical` (file con proiezioni raw / repacked / affine 8 bit, chunk 5 / 7 / 64, tile
  forzati da 2 e 16 righe con padding, expert in streaming con top-k + 1 slot). Le righe n-gram del prompt sono
  raccolte in anticipo per chunk. Con output per posizione (`--logits-out` ...) un passo per command buffer. I file densi
  qwen3_5 prendono lo stesso percorso (proiezioni e MLP sulle righe). Il predefinito è ora il prefill veloce in batch
  (sezione 'Prefill veloce in batch').
- **Tokenizer / chat**: il `tokenizer.json` di Qwen3.8 usa la regex Split di Qwen3.5 (sequenze di lettere `[\p{L}\p{M}]+`, la
  classe della punteggiatura esclude anche `\p{M}`): implementata come seconda modalità del matcher, `\p{M}` aggiunto alle tabelle Unicode
  (`tools/gen_unicode_tables.py`, bit di classe 8). `--chat` su un file qwen4_exp genera il `chat_template.jinja` del modello
  (turno utente singolo, turno di sistema opzionale; con thinking un turno di sistema per il reasoning effort, `--reasoning-effort
  xhigh|medium|low`, predefinito xhigh, e il generation prompt `<think>\n`; `--no-think`: `<think>\n\n</think>\n\n`).
  `--stop-eos` si ferma al testo `eos_token_id` o ai token di fine del tokenizer (`<|im_end|>`, `<|endoftext|>`).
  `--print-prompt` stampa gli id del prompt senza caricare il modello.

### Test

- `tests/unit/test_cpp_qwen4.py` (CTest `suc-qwen4-crosscheck`; saltato se non compilato): piccoli checkpoint qwen4_exp
  (`tiny_models.write_tiny_qwen4_exp`) passati per `suc compress` (VQ K = 16 G = 2, rotazione di input su un solo lato, residui a colonne
  int8, codebook degli expert stack, embedding / head affine), tabelle raw e passate per `suc repack` (affine per riga), due shape:
  quella piccola (hidden 64, head_dim 16 → fallback dell'attention di MLX) e una alle larghezze reali in cui i kernel si diramano
  (hidden 2560 = stream larghi 10240, head_dim 256, linear key dim 128, 16 head n-gram × 160). Prompt + 30 token
  greedy (33 posizioni, QSA sparsa dalla posizione 11), esecuzione Python teacher-forced sulla sequenza C++: errore di logit / stream,
  KL, argmax e i token greedy liberi; prefill a chunk = prefill per passo = greedy Python (EOS dentro
  il prompt); teacher forcing; controllo del vocabolario; le viste GPU escludono le tabelle; il chat template di Qwen3.8 rispetto a
  jinja2 + HF `tokenizers` (quando lo snapshot è nella cache HF).
- `tests/unit/test_cpp_tokenizer.py`: `test_qwen38_*` — il tokenizer.json di Qwen3.8 rispetto a HF `tokenizers` su WikiText-2
  (2,891 righe, 297,053 token), i casi limite e il corpus casuale più stringhe con segni combinanti: 0 discrepanze
  (encode con / senza token speciali, decode, round trip). Richiede lo snapshot nella cache HF o
  `$SUC_QWEN38_TOKENIZER`.

### Risultati (laptop M3, 2026-10-03)

Ogni file di test, entrambi i formati di tabella, entrambe le shape: **logit e ogni stream bit-identici** al runtime Python del piano
in tutte le 33 posizioni (KL 0), token greedy identici; rispetto a `affine_modules=True` (head con matmul quantizzato di MLX) i logit
differiscono di ≤ 1 ulp bf16 (KL ≤ 2e-6), stessi token. Non confrontato: il runtime Python predefinito (`group=True, fuse=True`; le sue
rotazioni segmentate arrotondano diversamente, come per Qwen3). Decode del file piccolo ≈ 1.8 ms/token (sincrono, un round trip con l'host
per token per le righe n-gram; GPU 1.3 ms) — nessuna affermazione sui tempi; i numeri del modello reale vengono solo dallo Studio.

### Non supportato / limiti

- Head MTP e vision tower (ignorati, come nel runtime Python); embedding tied, bias dell'attention, altri tipi di layer.
- Nessun decode in pipeline (le righe n-gram del token successivo richiedono l'host).
- Contesti lunghi: il percorso QSA è esatto sull'M3 entro il dispatch MLX misurato; la regola a due passate dell'M1 Ultra è quella
  non verificata della milestone 4. Parità esatte tra punteggi dei blocchi possono selezionare blocchi diversi da argpartition.
- File reale misurato sullo Studio: vedere la tabella in alto (73.82 GB, 28.3 tok/s).

### Mac Studio: il file reale

```bash
cd ~/Projects/SUC-LM && cpp/build.sh ~/suc-local/cpp-build
.venv/bin/python -m pytest tests/unit/test_cpp_qwen4.py tests/unit/test_cpp_tokenizer.py -q -k "qwen4 or qwen38"
F=experiments/artifacts/compress/compress_qwen3.8_flash_next_side8/container/model.suc1   # chosen file (ngram4)
~/suc-local/cpp-build/suc1-generate $F --prompt "The history of the printing press begins" --steps 64 --text
~/suc-local/cpp-build/suc1-generate $F --prompt "Explain QSA attention briefly." --chat --no-think --stop-eos \
  --steps 256 --max-context 4096 --json logs/q4_cpp_$(date -u +%Y%m%dT%H%M%SZ).json
```

In sequenza, mai insieme ad altri job GPU. Il C++ non fa mai tuning: se la cache di tuning non ha le dimensioni di qwen4_exp (2560,
640, 6144), eseguire prima una volta `suc generate $F --tokens 1 --steps 1` (Python fa l'autotune al caricamento).

Memoria attesa (stima da docs/arch/qwen4_exp.md §6; la riga `memory:` su stderr riporta i valori misurati): viste
GPU = il file meno le tabelle n-gram ≈ 67 GB (codici idonei ≈ 65 GB + BF16 mantenuti ≈ 1.5 GB + embedding / head affine
≈ 0.7 GB), tutte rese residenti al primo passo (≈ 15–20 s da SSD); array di esecuzione costruiti al caricamento (tabelle dei segni
per expert, residui a colonne) ≈ 1 GB; KV + chiavi dell'indexer ≈ 28 KB per posizione (115 MB a 4,096); stati di Gated DeltaNet
≈ 115 MB; arena < 50 MB. Totale ≈ 68–70 GB residenti (misurato: 73.82 GB); le tabelle n-gram (102 GB raw / 27 GB affine per riga) restano nella page
cache solo per le righe toccate (16 righe per token). Sta nello Studio da 128 GB, non in un Mac da 32 GB.

## Streaming degli expert da SSD (`suc/expert_stream.hpp`, `src/metal/expert_stream.cpp`, strumento `suc-expert-bench`)

Per i file MoE più grandi della memoria che il runtime può usare (docs/expert_streaming.md). `suc1-generate FILE
--expert-budget BYTES[K|M|G] [--hot-experts FILE|auto] [--expert-policy lru|slru] [--expert-io-threads N]
[--expert-pin-fraction F] [--expert-prefetch] [--expert-cached-io]`; il JSON riceve un record `expert_streaming`
(pool, insieme pinnato, richieste / hit / miss / eviction / hit e spreco del prefetch, byte e chiamate di lettura, tempo di IO e di
blocco; warm-up escluso) e `excluded_from_gpu_views`.

- **Pool.** Un MTLBuffer con storage condiviso per layer MoE di S_l slot; uno slot contiene gli array in streaming di un expert nel
  layout del file stesso (sequenze di array adiacenti, allineati a 64 byte), così un miss è una lettura diretta nello slot: i byte
  memorizzati sono gli operandi dei kernel, nessuna conversione. S_l = la stessa frazione di capacità per ogni layer, almeno top-k,
  il budget avanzato distribuito slot per slot; Σ pool ≤ budget (verificato nei test). Un budget inferiore agli slot top-k di ogni layer
  è un errore.
- **Letture.** `pread` su un unico descrittore con `F_NOCACHE` (lo slot è l'unica copia) su un pool di thread di IO (predefinito
  8); tutti i miss di un layer sono inviati insieme, sequenze suddivise in letture da ≤ 1 MiB.
- **Politica.** Per layer LRU o LRU segmentata (probation + protected 80 %, il default). Insieme caldo: i conteggi di routing di
  `hot_experts.json` (`suc compress` con preflight.expert_streaming; chiavi `"<prefix>.mlp"`), pinnati in modo greedy per conteggio
  su tutti i layer fino a `--expert-pin-fraction` del budget (predefinito 0.5), lasciando top-k slot non pinnati per layer.
- **Prefetch** (`--expert-prefetch`, disattivato di default): il router del layer MoE successivo applicato all'input MoE del layer corrente
  (euristica di pre-gating), il suo top-k letto in modo asincrono mentre la GPU esegue gli expert correnti; il fetch di un layer
  attende prima le sue letture di prefetch in sospeso.
- **Viste GPU.** I payload degli expert instradati (`routed_expert_ranges(file)`: ogni array di ogni tensore `….experts.<e>.…`)
  sono esclusi dalle viste di MappedFile (pagine intere; i file di test piccoli non ne hanno), come le tabelle n-gram di qwen4_exp.
- **Hook indipendente dal codec.** `ExpertStream::add_layer(name, per-expert file ranges, min_slots, bind)`; all'installazione lo
  stream chiama `bind(expert, pool, byte offsets of the ranges)`. `SwitchLinear` ha acquisito `StackLayout::Streamed` (i
  kernel `_tab`, indici impacchettati / scale di riga con base nel pool, voci della tabella degli offset riscritte da `bind_expert`;
  codebook, tabelle dei segni e residui restano residenti come in `Table`), `MoeBlock::attach_stream / stream_in /
  stream_prefetch`. Un altro codec degli expert (es. MXFP4 nativo) registra i propri array allo stesso modo.
- **Grafi.** `ModelOptions::expert_stream` / `Qwen4Options::expert_stream`. Per layer in streaming: router, chiusura del
  command buffer e attesa (`split_command_buffer`), fetch degli id instradati sull'host, encode degli expert. Il decode
  in streaming mantiene un solo command buffer in volo (`generate` forza depth 1); il prefill in batch di qwen3_moe esegue chunk di al massimo
  ⌊min slot non pinnati / top-k⌋ token. Streaming e residente sono bit-identici (stessi kernel, stessi byte degli operandi).

- **deepseek_v4** (`Ds4Options::expert_stream`): i tre `ExpertStack` di ogni layer sono costruiti in streaming (sub-stack a codec
  misto = `SwitchLinear::from_file_slots(..., streamed)` sugli id globali degli expert, expert nativi =
  `Mxfp4Switch` con le basi dei codici / delle scale nel pool e la sua tabella degli offset in byte riscritta all'installazione); per expert 6
  intervalli ({gate, up, down} x {indici impacchettati o codici MXFP4, scale di riga o scale E8M0}), `bind` smistato al
  sub-stack che contiene l'expert. Per layer: router + gate sulla GPU, split, fetch, expert; prefetch = il gate del layer successivo
  (router + bias, o `tid2eid[token]` sui layer hash, cioè esatto) sull'input MoE di questo layer. `--mixed-experts all`
  viene rifiutato con un budget (legge il primo expert di ogni sub-stack per le righe estranee, che non è detto siano residenti).
  Il prefill del prompt a chunk continua a funzionare (ogni layer MoE di ogni passo divide il command buffer).

- **Residency set** (`Context::make_resident`, macOS 15+, `SUC_RESIDENCY_SET=0` lo disattiva): i pool, le viste del file
  e ogni `new_buffer` di almeno 64 KiB entrano in un unico `MTLResidencySet` collegato alla coda (fino a
  max(recommendedMaxWorkingSetSize, `iogpu.wired_limit_mb`); il JSON riporta `residency_set_bytes` /
  `residency_skipped_bytes`). Senza di esso ogni command buffer per layer rendeva di nuovo residente il suo pool, e vicino al limite
  wired il driver rilasciava (unwire) i pool inattivi tra un command buffer e l'altro: GLM-5.3-Flash sullo Studio faceva decode a 2.58 s/token
  con 0.14 s di tempo GPU e 0.07 s bloccato in letture (≈ 55 ms per split, 42 split). `suc-expert-bench --gpu`
  (un command buffer per layer che legge gli slot instradati; `--experts E` fa ricominciare gli intervalli dall'inizio del file così che i pool possano
  superarlo, `--prefetch` = predizione perfetta del layer successivo) sull'M3, shape di GLM (expert da 9.4 MB, 42 layer, top-8,
  27 expert per layer, 8.85 GB di pool), 8 thread di IO: 968 ms/token senza il set (409 ms/token fuori da GPU e
  letture), 567 ms/token con esso (37 ms/token fuori da GPU e letture); per commit 23 ms p50 / 335 ms p90 → 0.2 ms in un
  test autonomo 8 x 1.2 GB. Studio dopo la correzione: GLM-5.3-Flash base in streaming 9.1 tok/s, hit rate 91 % (tabella in alto).

Non in streaming (residenti, costruiti al caricamento come in `Table`): array dei residui degli expert (CSR / colonne; ≈ 1–3 % di un expert),
tabelle dei segni e codebook per expert. Non fatto: numeri di accuratezza del prefetch su una traccia di routing reale.

Percorso di lettura sul laptop M3 (2026-10-04; `suc-expert-bench ~/suc-local/qwen3_8b_final/model.suc1`: expert sintetici
su un file da 4.3 GB, 48 layer, top-10 casuale uniforme, 16 token, SLRU, F_NOCACHE, nessun calcolo tra i layer; numeri del
laptop, da rieseguire sullo Studio e sui Mac di destinazione):

| MB per expert | budget | thread IO | IO ms / token | MB letti / token | GB/s durante il blocco | miss per layer p50 / p90 ms |
|---|---|---|---|---|---|---|
| 2.58 | 35 % (hit 0.32) | 1 | 546 | 836 | 1.53 | 11.0 / 14.8 |
| 2.58 | 35 % | 4 | 259 | 835 | 3.23 | 5.3 / 7.2 |
| 2.58 | 35 % | 8 | 224 | 839 | 3.77 | 4.6 / 6.0 |
| 1.54 | 25 % (hit 0.23) | 4 | 203 | 573 | 2.83 | 4.1 / 5.3 |
| 1.54 | 25 % | 8 | 187 | 569 | 3.06 | 3.9 / 4.8 |

I ~7 miss di un layer raggiungono 3–3.8 GB/s, sotto i ~5 GB/s di `scripts/pread_bench.py` con una coda profonda: per layer le
letture dipendono dal router, quindi la coda si svuota a ogni layer (il prefetch è ciò che può renderla più profonda).

Test: `tests/unit/test_cpp_expert_stream.py` (codec per expert di Qwen3-MoE con residui CSR e residui a colonne, LRU /
SLRU / prefetch, entrambi i piani di proiezione, insieme caldo, errore di budget, prefill in batch, decode in pipeline, esclusione dalle viste GPU;
qwen4_exp con e senza prefetch): logit, hidden state e token bit-identici al runtime C++ residente con
budget di top-k + 1 slot per layer (eviction a ogni passo). `tests/unit/test_cpp_deepseek_v4.py::test_streamed_experts_equal_resident`:
shape tiny / wide / real di deepseek_v4, base (stack misti VQ / additive) e native (+ expert MXFP4, un layer interamente
nativo), SLRU e LRU + prefetch, prefill a chunk: logit, stream e token bit-identici al residente con top-k + 1
slot per layer; `--mixed-experts all` con un budget fallisce.

## Milestone 6: deepseek_v4 (DeepSeek-V4-Flash)

`suc/deepseek_v4.hpp`, `src/metal/deepseek_v4.cpp`; expert stack `suc/expert_mixed.hpp`, `src/metal/expert_mixed.cpp`;
kernel `metal/suc_ds4_kernels.metal` (accodati al sorgente incorporato). `suc1-generate` smista su
`model_type == "deepseek_v4"`. Riferimento: `suc/models/mlx_deepseek_v4.py` sul runtime SUC1 Python
(`load_suc1_model(path, group=False, affine_modules=False)`, `step_hidden` per token); architettura:
[docs/arch/deepseek_v4.md](../docs/arch/deepseek_v4.md).

### Progettazione

- **Grafo** (un token per passo): l'embedding raccolto nei 4 stream delle hyper-connection; per layer
  `hc_attn.pre` (scala RMS float32 dello stream largo 16384, i 24 logit dei mixer nell'ordine gemv di MLX, gate sigmoidi,
  softmax per riga + 20 normalizzazioni di Sinkhorn in un solo thread, il collasso degli stream), attention, `hc_attn.post` (il matmul
  di mixing 4 x 4 come catena di fma di MLX), lo stesso attorno al MoE; `hc_head` + norm + LM head. `--hidden-out` scrive il
  tensore degli stream Python (stream + i 3 canali delle cifre del token id del routing hash).
- **Attention**: wq_a / wq_b / wkv / wo_a / wo_b sono `metal::Linear` (wo_a come il modulo SUC Python: ogni gruppo di righe su
  ogni gruppo di input, presi i blocchi diagonali; wo_a denso: un gemv per gruppo); norm RMS float32 per head di q, RoPE
  sugli ultimi 64 canali (theta base sui layer a finestra, YaRN + compress_rope_theta sui layer compressi; le frequenze
  inverse nell'aritmetica float32 di numpy), il round trip FP8 dei canali KV non RoPE (la conversione E4M3 di MLX),
  un ring delle ultime `sliding_window` chiavi. I compressori (ratio 4 con il gruppo precedente sovrapposto, ratio 128) mantengono
  le righe float32 grezze del gruppo corrente e di quello precedente ed emettono una voce quando un gruppo è completo (softmax sulle
  righe, pooling, norm, RoPE all'inizio del gruppo, FP8 o Hadamard + FP4). Il lightning indexer (con il proprio compressore
  ratio 4) assegna un punteggio a ogni voce una volta che ne esistono più di `index_topk` e seleziona le prime k (ricerca esatta della soglia,
  parità risolte con l'indice più basso = l'ordine di argpartition). L'attention segue il dispatch di MLX: `sdpa_vector` con la mask booleana e
  i sink per head dim 64 / 96 / 128 / 256 con al massimo 32 query head, altrimenti il fallback non fuso di MLX — la head_dim reale
  512 prende quest'ultimo (una mask di tipo array esclude sempre il kernel vettoriale di MLX per head-dim 512). MLX ripiega le H query head
  dell'unica KV head in M = H righe di ciascun matmul, quindi il C++ segue il dispatch dei matmul di MLX: una head: punteggi in ordine gemv,
  p @ V in ordine gemv_t; H > 1: q' Kᵀ tramite `gemv_wide` (`ds4_fb_scores_wide`; generazione GPU Apple >= 15, 2-15
  head, bf16 / f16, >= 2 chiavi; 32 lane per chiave con una passata o <= 64 chiavi, altrimenti 16) o l'ordine della GEMM steel di MLX
  (`ds4_mma`: fusa con prima la coda K % BK, split-k quando K >= max(M, N) e K / 16 >= 8; l'ordine e il piano di
  `glm_mma` / `gemm_plan` del grafo GLM), una chiave: gemv sulle H righe (BN in base a H); p[1:] @ V sempre la GEMM steel
  (split-k da 512 chiavi in poi). Le chiavi sono lette in place (ring della finestra + voci); softmax(precise) a riga singola fino a 4096
  valori, il kernel a ciclo di MLX con 1024 thread oltre.
- **MoE**: logit del router in float32 (gemv), punteggi sqrt(softplus) (un thread per expert), layer hash da
  `tid2eid[token]` (qualsiasi dtype intero), gli altri i primi k di punteggi + bias (decrescente, parità per indice), pesi
  normalizzati e scalati; SwiGLU con clamp in float32 con il peso di routing prima di w2; le k righe degli expert sommate in
  float32 nell'ordine per colonne di MLX; shared expert sommato in float32.
- **Expert stack** (`ExpertStack`, = `switch_from_encodings`): un solo `SwitchLinear` quando ogni expert ha un unico layout
  dei codici; `Mxfp4Switch` quando tutti sono MXFP4 nativi (record SUC1 `mxfp4` letti in place: un port di
  `fp_gather_qmv` / `fp_gather_qmv_fast` di MLX per `gather_qmm(mode="mxfp4")`, il kernel veloce quando in % 512 == 0 e out %
  8 == 0); altrimenti uno **stack misto** = `SucSwitchMixed`: un sub-stack per layout (gli expert nativi ne formano uno), ciascuno
  sugli id GLOBALI degli expert (`SwitchLinear::from_file_slots`: gli slot vuoti portano un offset impacchettato ~0 che il prologo di lookup `_tab`
  controlla e salta), poi un lancio di selezione prende la riga n dal sub-stack del suo expert.
  `--mixed-experts routed` (predefinito): le righe instradate ad altri sub-stack vengono saltate (nessuna lettura di pesi); `all`: ogni
  sub-stack calcola ogni riga (la semantica Python, A/B). Stessi bit in entrambi i casi (testato).
- **Numerica** — misurata rispetto a MLX 0.32.3 con delle probe (candidati `mx.fast.metal_kernel` rispetto alle op di MLX):
  MLX arrotonda ogni op element-wise, quindi i kernel multi-op disattivano la contrazione in fma (`#pragma clang fp
  contract(off)`); exp / log / sqrt / rsqrt / sin / cos precise tranne dove MLX scrive `fast::exp` (softmax,
  attention); somme per riga (`mean(x²)`) con i thread della riduzione per riga di MLX (32 fino a 512 valori, 128 a 1024, 512 a 2048,
  1024 oltre, 4 contigui per thread, somme simd); somme per colonna (stream, righe del compressore, head dell'indexer, righe degli expert)
  come 8 parziali sequenziali interlacciati fino a 16 righe, 32 parziali + `simd_sum` da 32 righe; una softmax non sull'ultimo asse è
  max, exp precisa, somma per colonna, divisione; `mx.hadamard_transform` su bf16 è `hadamard_n` con radix 16 e
  intermedi bf16; il matmul bf16 dell'indexer concorda dopo l'arrotondamento (fma sequenziale).
- **Prefill esatto per riga** (`--prefill exact` / `$SUC_PREFILL_EXACT=1` dal 2026-10-10; prima era il predefinito; in
  batch, layer-major, `Ds4Options::layer_prefill`): come il grafo qwen4_exp sopra, chunk di
  `--chunk` (64) token layer per layer: `wq_a`, `wkv`, `wq_b`, `wo_a` (per gruppo), `wo_b` e lo shared expert sulle
  righe del chunk (affine / denso / SUC esatto per riga come per qwen4_exp), norm di q / kv sulle righe,
  gli expert instradati in tile di expert (sub-stack SUC: il lookup vec con BT righe; MXFP4 nativo:
  `ds4_mxfp4_qmv_tile`, i corpi `fp_qmv` / `fp_qmv_fast` di MLX su BT righe che condividono i codici di un expert, l'aritmetica di ogni
  riga invariata; tile pianificati nell'ordine dei sub-stack (gate, up, down) e ogni sub-stack di uno stack misto eseguito
  solo sull'intervallo di tile dei propri expert, `$SUC_MIXED_ALL_TILES=1` = tutti i tile), hyper-connection, compressori
  (righe + voci emesse), KV della finestra, indexer e attention token per token in ordine di posizione; la rope di kv e la norm
  + rope di q di ogni riga girano prima di quel ciclo e la rope inversa degli output dopo, senza barrier
  tra i token (`$SUC_DS4_SEQ_TOKENS=1` = dentro il ciclo, A/B). Una sincronizzazione con l'host per layer MoE per chunk; gli expert in streaming sono recuperati per chunk in
  gruppi di token i cui expert distinti stanno negli slot. Bit-identità con `--prefill token`
  (`test_batched_prefill_bit_identical`: proiezioni base / native / affine 8 bit x shape tiny / wide / real, chunk
  5 / 7 / 64 attraverso i gruppi del compressore e il regime top-k dell'indexer, tile forzati da 2 e 16 righe, streaming). Il predefinito è ora il prefill veloce in
  batch (sezione 'Prefill veloce in batch').
- **Tokenizer / chat**: il `tokenizer.json` di DeepSeek (normalizer `Sequence` vuoto, tre pre-tokenizer `Split`:
  gruppi di cifre fino a 3, sequenze CJK, la regex punteggiatura / lettere / spazi con `\p{P}` e `\p{S}` — nuove classi delle tabelle
  Unicode dal motore di HF `tokenizers`, `tools/gen_unicode_tables.py`). `--chat` genera l'`encode_messages` a turno singolo
  di `encoding/encoding_dsv4.py` del checkpoint (DeepSeek-V4-Flash-0731 rev 7872f01b): BOS, in
  modalità thinking il testo `REASONING_EFFORT_PROMPTS` di `--reasoning-effort low|high|max` (predefinito low = nessuno), il
  testo di sistema, `<｜User｜>`, il prompt, `<｜Assistant｜>`, poi `<think>` (thinking) o `</think>` (`--no-think`);
  tokenizzato come fa `inference/generate.py` (`tokenizer.encode`, `add_bos_token` false). `--stop-eos` si ferma a
  `<｜end▁of▁sentence｜>`. Tool, task e `latest_reminder` non sono generati; la cronologia multi-turno (suc1-server) usa il
  markup per turno, non verificato rispetto a `encode_messages`.
- **Router**: lo `scoring_func` / `score_func` del checkpoint è `sqrtsoftplus` (config.json e inference/config.json alla
  revisione fissata); `Gate` di `inference/model.py` = `F.softplus(x @ Wᵀ).sqrt()` in float32, top-k di punteggi + bias,
  pesi = punteggi senza bias normalizzati x `route_scale`; `suc/models/mlx_deepseek_v4.py` e il `ds4_route` C++
  implementano esattamente questo (verificato 2026-10-04; `test_gate_scores_follow_the_reference_score_func`), quindi le esecuzioni di
  compressione sullo Studio non vanno rifatte per il router.

### Test

- `tests/unit/test_cpp_deepseek_v4.py` (CTest `suc-deepseek-v4-crosscheck`; saltato se non compilato): piccoli checkpoint
  DeepSeek-V4 nel layout originale FP8 / MXFP4 codificati tensore per tensore (attention: VQ / additive, rotazioni, residui int8
  a colonne e CSR fp16; expert: VQ K=16 G=2 + residuo a colonne, VQ K=64 G=4, additive M=3 per expert = stack misti
  in ogni layer; la variante "native" aggiunge expert MXFP4 testuali (4 codec in uno stack) e un ultimo layer interamente
  nativo), tre shape: tiny (head_dim 128: sdpa_vector; fp_gather_qmv), wide (head_dim 512: fallback di MLX;
  fp_gather_qmv_fast) e real (hidden 4096 = stream larghi 16384, head_dim 512, 64 head dell'indexer, expert 2048 x 4096,
  un compressore ratio 32). 40 posizioni (top-k dell'indexer attivo da 20), esecuzione Python teacher-forced sulla sequenza C++:
  si verifica **ogni stream e logit bit-identico**, greedy libero = greedy Python; `--mixed-experts all` = routed;
  prefill a chunk = per token = greedy Python; teacher forcing; controllo del vocabolario; gli id del prompt chat rispetto
  all'`encode_messages` del checkpoint (scaricato nella cache HF alla revisione fissata; thinking / chat, turno di sistema,
  reasoning effort low / high / max); expert in streaming = residenti (sopra).
- `tests/unit/test_cpp_tokenizer.py`: `test_dsv4_*` — tokenizer di DeepSeek-V4-Flash rispetto a HF `tokenizers` su WikiText-2
  (2,891 righe, 287,597 token), i casi limite e i corpora casuali con CJK / cifre / punteggiatura / simboli / token
  speciali: 0 discrepanze (encode con / senza token speciali, decode, round trip). Richiede lo snapshot nella cache HF
  o `$SUC_DSV4_TOKENIZER`.

### Risultati (laptop M3, 2026-10-04)

Tutti i 15 test di controllo incrociato: logit e stream bit-identici in tutte le 39 posizioni confrontate per i 6 file (tiny / wide /
real x base / native; tutti con 2 head di attention; 64 head e sequenze lunghe: la voce sull'attention ripiegata sotto). Tempi indicativi (M3, file sintetico da 433 MB alle larghezze reali, 2 layer, 16 expert top-6,
layer 0 uno stack misto di 4 codec (native + VQ G=2 + VQ G=4 + additive M=3), layer 1 interamente nativo; 64 passi greedy,
3 esecuzioni): C++ `--mixed-experts routed` 3.9 ms/token (GPU 3.5), `all` 6.4 - 7.2 ms/token (GPU 6.1 - 6.9): instradare le righe
per codec è 1.65 - 1.85x più veloce su questo file; runtime Python 12.3 ms/token. Nessuna affermazione sul modello reale da questi numeri.

### Non supportato / limiti

- Head MTP / DSpark (scartata, come in Python), `simulate_act_fp8` (disattivato anche nel modello Python), embedding tied,
  n_shared_experts != 1, num_key_value_heads != 1, `--mixed-experts all` insieme a `--expert-budget`.
- **Attention di fallback ripiegata (corretta il 2026-10-04).** Fino ad allora il C++ eseguiva gemv / gemv_t per head per H > 1 head
  mentre MLX esegue i matmul ripiegati descritti sopra; le somme float32 differiscono nell'ordine e ogni tanto invertono un arrotondamento bf16 dell'output
  dell'attention: probe (`mx.fast.scaled_dot_product_attention` rispetto alle stesse op per head, dati casuali) 0-1 di
  4 x 2 x 512 output differivano con 2 head, 3-346 di 4 x 64 x 512 con 64 head (il modello reale), già a 40 chiavi;
  sui file piccoli i logit differivano per la prima volta alla posizione 182 (2 head) / 20 (64 head), poi KL ~1e-3 a ogni posizione.
  I 15 test di controllo incrociato passavano solo perché i loro file hanno 2 head e 40 posizioni. Ora bit-identici:
  `test_folded_attention_heads` (8 e 64 head, 600 posizioni: gemv_wide con 32 e 16 lane, GEMM, split-k su entrambi
  i matmul) e `test_long_context_bit_exact` (`SUC_LONG_CONTEXT=1`, ~4 min per caso; 2 e 64 head, 8,400
  posizioni teacher-forced con una sliding window di 8192, cioè >= 8k chiavi dalla posizione 8,191 in poi e ~10,300 chiavi sul
  layer ratio 4): logit identici in tutte le 8,399 posizioni. La tabella BK 'd' (M1 Ultra) di `gemm_plan` è quella del grafo
  GLM ("come ricordata"); conta solo per le GEMM fuse il cui K non è multiplo di 16, cioè p V sotto 512 chiavi.
- Verificato: la softmax a ciclo di MLX (righe > 4096 chiavi) usa 1024 thread sull'M3 (una copia `mx.fast.metal_kernel` di
  `softmax_looped` con 1024 thread è uguale a `mx.softmax(precise=True)` su righe bf16 di 4,097-16,385 valori, 0 su
  4.2 M elementi differiscono; con 512 / 256 thread differiscono), che è ciò che esegue `ds4_softmax_looped`.
- L'attention di decode a due passate di MLX per il percorso vettoriale (head dim <= 256, > 1024 chiavi sui dispositivi 'd' di classe M1 Ultra,
  > 4096 chiavi con GQA, cioè sempre per l'unica KV head di deepseek_v4): non portata (la head_dim reale 512 non la usa
  mai), quindi i file piccoli con head_dim 128 non sono bit-esatti oltre quei numeri di chiavi.
- Lo scoring softmax (`scoring_func: softmax`) gira in un solo thread (non nell'ordine della softmax di MLX); i checkpoint DeepSeek-V4
  usano sqrtsoftplus.
- Nessun decode in pipeline (un command buffer per passo).

### Mac Studio: il file reale

```bash
cd ~/Projects/SUC-LM && cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-tests && ~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_deepseek_v4.py -q
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/unit/test_cpp_tokenizer.py -q -k dsv4   # needs the V4 tokenizer cached
F=experiments/artifacts/compress/compress_deepseek_v4_flash_tcq/container/model.suc1   # chosen file (resident)
~/suc-local/cpp-build/suc1-generate $F --prompt "The history of the printing press begins" --steps 64 --text \
  --json logs/ds4_cpp_$(date -u +%Y%m%dT%H%M%SZ).json
~/suc-local/cpp-build/suc1-generate $F --prompt "Explain sparse attention briefly." --chat --no-think --stop-eos \
  --steps 256 --max-context 4096
for m in routed all; do ~/suc-local/cpp-build/suc1-generate $F --tokens 0,128803,2058,128804 --steps 64 --mixed-experts $m; done
```

In sequenza, mai insieme ad altri job GPU. Il C++ non fa mai tuning: se la cache di tuning non ha le dimensioni di deepseek_v4
(4096, 1024, 2048, 512, 8192, 32768 / 8 gruppi), eseguire prima una volta `suc generate $F --tokens 1 --steps 1`. Accordo
con Python sul file reale: eseguire il confronto teacher-forced di `tests/unit/test_cpp_deepseek_v4.py`
(`--logits-out` / `--hidden-out` rispetto a `run_tokens`); la tabella dei tile GEMM dell'M1 Ultra ('d') non è verificata, quindi una discrepanza
lì punta per prima cosa a `gemm_plan`. Streaming: aggiungere `--expert-budget BYTES [--hot-experts auto] [--expert-prefetch]`.
Il gemv affine a 8 bit (deepseek_v4 e glm5_next, colonne non suddivise) esegue 4 simdgroup di righe per threadgroup di default
(bit-identico per qualsiasi valore; da 1 a 4 si passa da 49.2 a 45.6 ms/token solo su V1, §8 di
[decode_speed_vs_ds4.md](../docs/decode_speed_vs_ds4.md)); `SUC_AFFINE_SG=N` lo sovrascrive.

Memoria attesa (dalla directory; la riga `memory:` su stderr riporta i valori misurati): le viste GPU mappano
l'intero file (107.5 GB al budget configurato) e ogni pagina diventa residente al primo passo; array di esecuzione
costruiti al caricamento (tabelle degli offset degli expert, tabelle dei segni per expert, tabelle dei residui) ~1-2 GB; KV + stato del compressore
~6.9 KB per posizione + 5.6 MB di chiavi della finestra + 22 MB di righe dei gruppi del compressore (0.23 GB a 32k posizioni); punteggi
dell'attention di fallback 64 head x (128 + posizioni / 4) x 4 B; arena < 100 MB. Totale ~110 GB: sta nello Studio da 128 GB (il
runtime_budget della config di compressione è 111 GB), non in un Mac da 32 GB senza lo streaming degli expert da SSD (`--expert-budget`,
vedere la sezione sullo streaming degli expert da SSD). Misurato (file tcq): 102.57 GB.
Velocità misurata: la tabella "Prestazioni attuali" in cima (V1 native_max: 25.5 tok/s; tcq: 24.0 tok/s).

## Milestone 7: glm5_next (GLM-5.3-Flash)

`suc/glm5.hpp`, `src/metal/glm5.cpp`, kernel `metal/suc_glm5_kernels.metal` (accodati al sorgente incorporato; riusa
i kernel ds4 / qwen4 / model dove il modello Python ha la stessa op). `suc1-generate` smista su
`model_type == "glm5_next"` (`tools/suc1_generate_glm5.inc`, incluso da `suc1_generate.cpp`). Riferimento:
`suc/models/mlx_glm5_next.py` sul runtime SUC1 Python (`load_suc1_model(path, group=False, affine_modules=False)`,
`step_hidden` per token); architettura: [docs/arch/glm5_next.md](../docs/arch/glm5_next.md).

### Progettazione

- **Grafo** (un token per passo): embedding raccolto in 4 stream identici; per layer mHC pre (riduzione per riga di MLX di
  mean(x²) sullo stream 4 · hidden, lo stream normalizzato PRIMA dei 24 logit dei mixer (gemv float32), gate sigmoidi,
  softmax + 20 normalizzazioni di Sinkhorn in un solo thread, collasso degli stream), RMSNorm di input, KDA o MLA + DSA, mHC post nel
  dtype del modello (`T(T(post) f) + T(combᵀ res)`, il matmul bf16 4 x 4 come catena di fma float32), lo stesso attorno all'MLP denso /
  MoE; head: media degli stream (float32), norm, LM head (codici affine in place o denso).
- **KDA**: proiezioni q / k / v; conv4 depthwise causale + SiLU in float32 su una coda bf16 scorrevole (3 righe); q / k
  `x / sqrt(Σx² + 1e-6)` (somme per riga di <= 64 valori sequenziali in un thread = row_reduce_small di MLX, righe più lunghe
  row_reduce_simple di MLX); il decadimento safe-gate `exp(-5 σ(exp(A_log) (f_b f_a x + dt_bias)))` per canale, beta = σ_bf16(b x);
  il passo gated-delta di mlx-lm con il gate vettorizzato (stato float32 in place); o arrotondato a bf16, RMSNorm float32
  (peso `o_norm` convertito verso l'alto), `· σ(g_b g_a x)`, o_proj.
- **MLA + DSA**: q_a → RMSNorm → q_b; il latent kv_a (RMSNorm) accodato a una cache dei latent per layer; kv_b applicato
  in forma assorbita per head (`q'_h = q_h W_k,h`: gemv_t di MLX per head; `o_h = (P c) W_v,hᵀ`: gemv di MLX per head). Indexer: chiave
  `LayerNorm(wk x)` (layer_norm_single_row di MLX) e punteggio di gate `kpool_gate x` mantenuti come righe float32 del pool incompleto;
  ogni 4 token una chiave aggregata (softmax sul pool di gate + ape, `ds4_comp_pool`); una volta che esistono più di index_topk / 4
  pool completi, punteggi dei pool `Σ_h w_h relu(q_h · K_p / √d)`, top-k esatto (`q4_index_select`, parità per indice
  più basso = argpartition di MLX, un ordinamento stabile sulla GPU), e i token dei pool selezionati più la coda incompleta formano la
  mask dell'attention. Attention: `sdpa_vector` di MLX (kv_lora_rank 64 / 96 / 128 / 256 e <= 32 head) o il fallback non fuso
  di MLX (il rank reale 512).
- **MLX ripiega le head del fallback in GEMM.** Le H query head condividono l'unica KV head latente, quindi `q' Kᵀ`, `P V` e
  il `q poolsᵀ` dell'indexer sono singole GEMM steel in MLX (B in broadcast sul batch -> M = head), non gemv per head:
  `glm_mma` riproduce il loro ordine float32 con MMA simdgroup 8 x 8 x 8: split-k quando K >= max(M, N) e K / 16 >= 8
  (P = min(max(2, next_pow2((K/16) / (⌈M/32⌉ ⌈N/32⌉))), 32) partizioni di `((K/16)/P) · 16`, parziali sommati da 0),
  altrimenti il kernel fuso con prima la coda di K non allineata (BK dalla tabella per classe di dispositivo di MLX). Misurato con operandi float32
  rispetto a MLX 0.32.3 sull'M3 (ogni shape bit per bit); una sola chiave (primo token) o una sola head prende il gemv di MLX.
  L'attention di fallback di deepseek_v4 (64 head su una KV head, head_dim 512) usa lo stesso ordine GEMM ripiegato
  (corretto il 2026-10-04, milestone 6).
- **MoE**: logit del router float32 (gemv), sigmoide, top-k di punteggi + bias di correzione (decrescente, parità per indice),
  `w / (Σw + 1e-20) · 2.5`; expert instradati = `ExpertStack` (codici SUC; codec misti per stack = sub-stack + select);
  SwiGLU con clamp (senza peso), down, `T(Σ_k float(y_k) w_k)` (ordine per colonne di MLX) `+ shared` in bf16. I layer MLP densi e
  lo shared expert: proiezioni SUC / affine / dense + la SwiGLU con clamp.
- **Pesi in place**: proiezioni SUC (`metal::Linear`), proiezioni affine (`suc_gemv_affine` = la decodifica bf16 Python
  + gemv), tensori raw nel loro dtype memorizzato; copie float32 solo dove il parametro Python è float32 e il
  file no (pesi delle conv, `o_norm` dopo la conversione bf16, A_log / dt_bias / bias del router / base e scala di mHC).
- **Streaming degli expert da SSD** (`--expert-budget`, come gli stack della milestone 6): `ExpertStack::from_file(..., streamed)`, per
  expert gli intervalli {gate, up, down} x {indici, scale} registrati con `ExpertStream::add_layer("<prefix>.mlp")`
  (= le chiavi di `hot_experts.json` di `suc compress`); per layer MoE: router, `split_command_buffer`, `fetch`, expert;
  `--expert-prefetch` applica il router del layer MoE successivo all'input MoE di questo layer. Stessi bit del residente.
- **Chat**: `--chat` genera il chat_template.jinja di GLM-5.3-Flash (rev eb9eb208):
  `[gMASK]<sop><|system|>Reasoning Effort: Max` (`--reasoning-effort low|high|max`, predefinito max),
  `[<|system|>TEXT]<|user|>PROMPT<|assistant|><think>`. Il template ignora enable_thinking, quindi `--no-think` non
  cambia il prompt. Gli id del prompt sono verificati rispetto a transformers da `test_chat_prompt_format`. Il
  tokenizer è il BPE byte-level generico; un `tokenizer.json` con una regex del pre-tokenizer che il tokenizer C++ non conosce
  fallisce in modo esplicito al caricamento.

### Test

`tests/unit/test_cpp_glm5.py` (CTest `suc-glm5-crosscheck`; saltato se non compilato): piccoli checkpoint con la struttura di
GLM-5.3-Flash nel layout FP8 codificati tensore per tensore (proiezioni KDA / MLA / MLP / shared: VQ / additive con
rotazioni, residui int8 a colonne e CSR fp16, affine 8 bit; expert instradati: VQ K=16 G=2 + residuo a colonne, VQ K=64 G=4,
additive M=3 per expert = stack misti; embedding / head affine 4 bit), shape tiny (hidden 128, head_dim KDA 32, rank MLA
128: sdpa_vector), larghezze reali (hidden 4096 = stream larghi 16384, head_dim KDA 128, rank 512 / nope 256 / v 256,
indexer 32 x 128, expert 2048 x 4096) e 64 head MLA su 564 posizioni; 40 posizioni con la selezione dei pool attiva
dalla posizione 11: **ogni stream e logit bit-identico** al runtime Python (teacher-forced), token greedy uguali;
attivazioni float32 bit-identiche su ogni shape; prefill a chunk = per token; streaming (LRU / SLRU / prefetch, top-k + 1
slot per layer, un insieme caldo pinnato) = residente; un container di `suc compress` (codebook degli expert stack, `hot_experts.json`
`auto`) residente = Python e streaming = residente.

### Risultati (laptop M3, 2026-10-04)

Tutti i controlli incrociati bit-identici (logit e stream, bf16 e float32). Solo indicativo: il file di test a 2 layer con larghezze reali
fa decode a 2.4 ms/token (GPU 2.0 ms); nessuna affermazione per il modello reale.

### Non supportato / limiti

- RoPE (qk_rope_head_dim > 0), top-k dell'indexer condiviso, `index_kpool_always_select_tail = false`, router diversi da
  sigmoide / un gruppo, n_shared_experts != 1, bias dell'attention, embedding tied (tutti falliscono in modo esplicito); MTP e vision scartati.
- Contesti lunghi: la softmax del fallback sopra 4096 chiavi usa la softmax a ciclo di MLX (1024 thread, verificato sull'M3 nella
  milestone 6); lo sdpa_vector a due passate di MLX (rank <= 256 su contesti lunghi) non è portato (il rank reale 512 non lo usa
  mai). La tabella dei tile GEMM di classe 'd' (M1 Ultra) conta solo per attivazioni float32 con K % 16 != 0; le shape bf16 del
  modello reale sono indipendenti dal dispositivo.
- Nessun decode in pipeline (un command buffer per passo, più una sincronizzazione con l'host per layer MoE in streaming).
  Prefill in batch: vedere 'Prefill veloce in batch' sotto.

### Mac Studio: il file reale

```bash
cd ~/Projects/SUC-LM && cpp/build.sh ~/suc-local/cpp-build
~/suc-local/cpp-build/suc-tests && ~/suc-local/cpp-build/suc-metal-tests
.venv/bin/python -m pytest tests/unit/test_cpp_glm5.py -q
# owner, once: sudo sysctl iogpu.wired_limit_mb=116000
F=experiments/artifacts/compress/compress_glm5.3_flash/container/model.suc1   # 119.9-GB streamed file; the chosen tcq file is resident (no --expert-budget)
~/suc-local/cpp-build/suc1-generate $F --tokens 5,6,7,8 --steps 64 --expert-budget 95G --hot-experts auto \
  --json logs/glm53_cpp_$(date -u +%Y%m%dT%H%M%SZ).json
~/suc-local/cpp-build/suc1-generate $F --prompt "Explain sparse attention briefly." --chat --no-think --stop-eos \
  --steps 256 --max-context 4096 --expert-budget 95G --hot-experts auto --expert-prefetch
```

In sequenza, mai insieme ad altri job GPU. Il C++ non fa mai tuning: se la cache di tuning non ha le dimensioni di glm5_next, eseguire
prima una volta `suc generate $F --tokens 1 --steps 1 --expert-budget 1.02e11` (la CLI Python accetta byte). Bit-esattezza
rispetto a Python sul file reale: il confronto teacher-forced di `tests/unit/test_cpp_glm5.py` (`--logits-out` /
`--hidden-out` rispetto a `run_tokens` di `load_streamed_suc1_model`) con prompt sotto ~4,000 token.

Memoria attesa (stima dal piano di compressione, file da ~120 GB: expert instradati 114.2 GB, il resto ~5.8 GB; le righe su stderr
`memory:` / `expert streaming:` riportano i valori misurati): viste GPU = il file meno gli expert instradati
≈ 5.8 GB (tutte residenti dopo il primo passo); array di esecuzione costruiti al caricamento ≈ 2 GB (tabelle dei segni per expert ≈ 0.5 GB,
i residui a colonne degli expert ≈ 1.5 GB, che restano residenti); KV ≈ 11.7 KB per posizione (0.38 GB a 32k) + stati KDA
0.15 GB; pool degli expert = `--expert-budget` (95 GiB = 102 GB ≈ 89 % degli expert instradati, metà dei quali insieme caldo pinnato).
Totale ≈ 110 GB: dentro il runtime budget di 111 GB della config di compressione, solo con il limite wired alzato; non un
Mac da 32 GB. Misurato: 9.1 tok/s con un hit rate degli expert del 91 % (il
file tcq residente: 22.2 tok/s; tabella in cima).

## Tensori trellis-coded (tcq)

Rappresentazione SUC1 `tcq` (docs/format.md §5.5, docs/low_tier_codec.md; Python: `suc.kernels.tcq_matmul`,
`SucLinear` / `SucSwitchLinear` in modalità tcq): per riga, sequenze di T pesi codificate come bitstream little-endian
tail-biting di NW = T·k/32 word (codici `1mad`, `3inst`, `pair3`; V = 1 o 2 pesi per passo; stati a L bit), una scala f16
per riga, la rotazione di input su un solo lato, residuo opzionale a colonne / CSR.

- **Kernel**: `cpp/metal/gen_suc_kernels.py` emette `tcq_matmul._TCQ_HEADER`, i corpi `vec` (`_SRC`) e `mma` (`_MMA_SRC`)
  testualmente come `suc_tcq_vec` / `suc_tcq_mma`, dietro `_SWITCH_PROLOGUE` come `suc_tcq_switch_{vec,mma}` (array
  degli expert impilati) e dietro un prologo con tabella degli offset solo C++ come `suc_tcq_switch_{vec,mma}_tab` (expert letti in place,
  slot vuoti dei sub-stack a codec misto, pool di slot in streaming). Le scale di riga f16 sono collegate direttamente dal file
  (`float(half)` è esatto, quindi equivale alla copia float32 di SucLinear); `cst` = float32 [mu_0, c_0, mu_1, c_1] è calcolato
  sull'host dalla directory (`TcqLayout::from_params`, = `tcq_matmul.kernel_consts`: c_j = float(gain / code_sd[j])
  con la divisione in double).
- **Decoder di riferimento** (`src/decode.cpp`): stati dalle finestre dello stream, valori grezzi dei codici esatti in double,
  float32((raw - mu) / sd) · float32(gain) · scala di riga, poi l'inversa di Hadamard condivisa: bit-identico a
  `SucModel.decode_tensor`.
- **`metal::Linear`**: gli stream sono il payload `chunks` memorizzato visto come (out, in/T · NW) u32 (zero-copy). Dispatch =
  `tcq_matmul.launch_plan`: tile di batch di <= 16 righe; un tile di bt righe esegue `vec` sotto 5 righe (SG = 8, R = 4 per
  bt <= 2 altrimenti 2), `mma` da 5 (SG = 4, RT = 4, NB = ceil(bt / 8)). Rotazioni, residuo e dtype di output sono gli stadi
  del Linear vq. Mai fuso (`FusedLinear::key` = ""; i membri tcq condividono comunque la rotazione di input del loro gruppo).
  `encode_rows_exact` esegue tile vec fino a 4 righe: l'aritmetica di nessuna riga dipende dalle altre righe del tile, quindi
  ogni riga è uguale al suo lancio a una riga (decode) bit per bit (`suc1-linear --rows-exact`, verificato dai test).
- **Expert stack**: `SwitchLinear` accetta stack tcq uniformi (Stacked quando gli stream / le scale degli expert sono consecutivi
  nel file, altrimenti Table; Streamed per lo streaming degli expert da SSD: gli stessi due intervalli di file per expert del vq), segni
  di input condivisi (1, in) quando ogni expert ha lo stesso `seed_in` (ciò che `suc compress` scrive per gli expert tcq), altrimenti tabelle
  dei segni per riga. Decode = il percorso gather (BT = 1, vec); tile di expert del prefill in batch fino a 4 righe (vec, esatto per riga).
  Stack a codec misto (`ExpertStack`): gli expert tcq formano i propri sub-stack (uno per chiave di layout tcq) accanto a quelli vq /
  additive / MXFP4 nativi, come `switch_from_encodings`.
- **Grafi**: qwen3 / qwen3_moe, qwen4_exp / qwen3_5, deepseek_v4 e glm5_next accettano proiezioni ed expert tcq.

Test: `tests/unit/test_cpp_tcq.py` (decoder; Linear rispetto al riferimento CPU e bit-esatto rispetto a Python in f16 / bf16 su entrambi i
kernel e sui percorsi forzati; batch esatti per riga; stack in ogni layout, tile, stack misti vq + tcq, un blocco MoE; le sei
famiglie di grafi teacher-forced rispetto a Python, prefill in batch di deepseek_v4 ed expert in streaming, expert in streaming di
glm5_next). Strumenti: `suc1-linear --rows-exact`, `suc1-moe --stack P --mixed routed|all --tiles BT`.

Limiti: i kernel accettano V in {1, 2}, T multiplo di 128 che divide in, kv <= L <= 24 (i codici di calcolo SUC1; `hyb` non è
una rappresentazione SUC1). Nessun tuning per dispositivo (costanti di lancio fisse dei kernel Python). Velocità: le righe tcq di DeepSeek / GLM della
tabella in alto.

## suc1-server: server HTTP compatibile con OpenAI

`suc1-server FILE [options]` (`tools/suc1_server.cpp`) carica un file SUC1 una sola volta e lo serve tramite un server HTTP/1.1 minimale
e senza dipendenze (socket POSIX; keep-alive; corpi delle richieste con `Content-Length`; le risposte in streaming usano
il chunked transfer encoding). Gira ogni famiglia di modelli di `suc1-generate` (qwen3, qwen3_moe, qwen3_5, qwen4_exp,
deepseek_v4, glm5_next), con tutti i suoi flag di runtime e di streaming degli expert (`--expert-budget`, `--hot-experts`, ...).

```bash
~/suc-local/cpp-build/suc1-server model.suc1 [--host 127.0.0.1] [--port 8080] [--model-name NAME] \
    [--max-context 4096] [--max-tokens 256] [--temperature T --top-k K --top-p P --min-p P] \
    [--no-think] [--reasoning-effort LEVEL] [--tokenizer tokenizer.json] [--quiet] [suc1-generate runtime flags]
```

- Endpoint: `POST /v1/chat/completions`, `POST /v1/completions` (`"stream": true` = server-sent event, un
  `chat.completion.chunk` / `text_completion` per delta di testo, un chunk finale con `finish_reason`, un chunk di usage con
  `stream_options.include_usage`, poi `data: [DONE]`), `GET /v1/models`, `GET /health`. Gli errori usano il corpo OpenAI
  `{"error": {...}}` (400 per parametri non supportati).
- Parametri: `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`
  (stringa o lista; il testo di stop è escluso), `n` = 1; chat: `messages` (system / developer / user / assistant, contenuto stringa
  o a parti di testo), `reasoning_effort`, `chat_template_kwargs.enable_thinking`. Le completion accettano un prompt testuale o una
  lista di token id (un prompt per richiesta). I parametri di sampling non impostati prendono i flag del server (predefinito: greedy).
  Non supportati: `n` > 1, logprobs, echo, suffix, penalità, tool.
- Generazione: le richieste sono serializzate (batch 1, un mutex attorno al modello; ogni connessione ha il proprio thread, quindi
  `/health` risponde durante una generazione), la KV cache / gli stati vengono azzerati per richiesta. Le richieste greedy qwen3 / qwen3_moe
  eseguono il decode in pipeline di `suc1-generate` (`Model::generate`, `--depth` command buffer in volo, la sua
  callback `on_token` consuma ogni token e invia il chunk SSE mentre il passo successivo gira sulla GPU); il sampling,
  `--sync` e le altre famiglie fanno decode in modo sincrono (un command buffer per token; il sampler di
  `suc/sampling.hpp`, con seed per richiesta, un seed casuale quando `seed` non è impostato), come fa `suc1-generate`. Si ferma all'`eos_token_id` dell'architettura e ai token di fine del tokenizer (`finish_reason: "stop"`, il
  token di fine conta in `usage.completion_tokens` ma non è nel testo), a una stringa di stop, oppure a `max_tokens` / al
  contesto (`--max-context`, predefinito 4096: la KV cache viene allocata per esso al caricamento; `finish_reason: "length"`).
  I delta in streaming trattengono una coda UTF-8 incompleta e un possibile prefisso di una stringa di stop, così si ricompongono nel
  testo non in streaming.
- Chat template: il template per famiglia di `suc1-generate --chat` (ora `chat_prompt_text` in `tools/tool_common.hpp`,
  condiviso da entrambi gli strumenti). Un turno utente (eventualmente dopo un turno di sistema) viene generato esattamente come prima; i turni precedenti di una
  storia multi-turno usano lo stesso markup per turno, con il ragionamento `<think>...</think>` di un turno assistant scartato (come
  fa il template di Qwen3; per qwen4_exp, deepseek_v4 e glm5_next il markup della storia non è verificato rispetto ai loro
  template di riferimento).
- Tokenizer: obbligatorio (`--tokenizer`, altrimenti il tokenizer.json fissato dal `tokenizer_id` del SUC1). Gli id del prompt >= il vocab del
  modello vengono rifiutati.
- Velocità: i ms/token del decode greedy (registrati per richiesta) sono uguali a quelli di `suc1-generate` sul piccolo file di test (M3: 0.18 vs
  0.18 ms/token non in streaming e in streaming; 0.83 ms/token con `--sync`, il precedente ciclo sincrono); il test verifica
  <= 1.1x.

Test: `tests/unit/test_cpp_server.py` (anche una voce CTest) avvia il server su un piccolo file Qwen3 casuale con un
tokenizer byte-level sintetico e lo pilota con il client Python `openai` (dipendenza di sviluppo): il testo greedy di
`/v1/completions` (prompt a token id e testuali) e di `/v1/chat/completions` (con / senza turno di sistema, `enable_thinking`
false) è uguale ai token di `suc1-generate` troncati al primo token di fine, i delta in streaming si ricompongono nello stesso testo, i conteggi dei token
del prompt sono uguali al prompt di `suc1-generate --chat`, il sampling con seed è uguale a `suc1-generate --temperature/--top-k/--top-p
--seed`, stringhe di stop, il limite di contesto e gli errori delle richieste.

## Prefill veloce in batch dei grafi frontier (2026-10-10)

Decisione del proprietario: il prefill in batch può differire dal decode token per token per piccole differenze di arrotondamento (nessun
requisito di bit-identità); l'entità della differenza viene invece misurata (`scripts/prefill_batch_vs_token.py`:
KL / top-1 per posizione di ogni riga del prompt e di una continuazione teacher-forced, report §1c). I grafi deepseek_v4,
glm5_next, qwen4_exp e qwen3_5 eseguono quindi, di default, un prefill in batch costruito su kernel
simdgroup-matrix; il precedente prefill in batch esatto per riga resta disponibile come `--prefill exact` (o `$SUC_PREFILL_EXACT=1`,
`*Options::exact_prefill`), `--prefill token` = passi di decode. Il grafo Qwen3 / Qwen3-MoE è invariato (già non
esatto per riga).

- **Kernel GEMM** (`cpp/metal/suc_model_kernels.metal`, `suc_mm_*`; host `encode_mm_affine` / `encode_mm_dense` /
  `encode_tcq_mm` in `suc/metal.hpp`): Y (n, out) = X (n, in) · Ŵᵀ su tutte le righe del chunk, MMA simdgroup 8 x 8,
  accumulo float32, un solo arrotondamento. Tile del threadgroup BM (8 / 16 / 32 / 64) righe x 64 output, passi di k da 32; il
  tile dei pesi viene decodificato nella memoria del threadgroup (affine: T(q·s + b) come il gemv di decode; denso: T(w), o il valore
  memorizzato per output float32 come i gemv del router / compressore; tcq: decodifica ad accesso casuale della finestra trellis
  di ciascun peso). Usato per ogni proiezione affine / densa di un chunk (n >= 8, in % 32 == 0); le proiezioni SUC eseguono
  `Linear::encode(..., Path::Prefill)` (il lookup "mma" in tile da 16 righe; tensori tcq: `suc_tcq_mm` su tutte le righe).
- **Tile di expert** (`Tiles::exact = false`): fino a 16 righe per tile (i sub-stack tcq eseguono ogni tile da 16 righe come due
  tile mma tcq_matmul da 8 righe, poiché il tile da 16 righe di quest'ultimo era 3.5x più lento, moltiplicando matrici float32 invece di quelle nel
  dtype delle attivazioni di tcq_matmul: con matrici bf16 il piccolo file tcq glm5 raggiungeva una KL per posizione di 0.245), il lookup vq / additive sul suo kernel mma dalle `mma_min_batch` righe tarate (il kernel mma dello switch
  ora riceve NB = ceil(bt / 8): con NB fisso a 1, i tile da 16 righe calcolavano solo metà di ciascun tile), i sub-stack MXFP4
  nativi su `ds4_mxfp4_mm_tile` da 8 righe (unendo tile consecutivi di un expert in una riga di threadgroup di
  <= 64 righe quando il piano dell'host è noto). `plan_tiles` mantiene il suo modello di costo (α = 0.3: un α più piccolo, cioè più tile
  da 16 righe, era più lento).
- **Attention di deepseek_v4** (`attn_chunk_fast`): proiezioni q / kv come GEMM, norm di q + RoPE e RoPE di kv su tutte le righe
  (`ds4_qnorm_rope_rows`, `ds4_rope_rows`), chiavi in un buffer lineare (la finestra prima del chunk raccolta dal
  ring, poi le chiavi del chunk con FP8 simulato, `ds4_kv_store_rows`), entrambi i compressori una volta per chunk (`comp_chunk`: una
  GEMM per peso del compressore in righe float32, i gruppi completati dentro il chunk aggregati in un unico lancio
  `ds4_comp_pool_rows`, normalizzati, con rope, Hadamard + FP4 / FP8, buffer dei gruppi lasciati come il passo di decode si aspetta), UN
  solo lancio `ds4_attn_chunk` per layer (un simdgroup per riga e head sulle sue chiavi della finestra e voci visibili), il
  ring aggiornato con le ultime chiavi della finestra del chunk, RoPE inversa su tutte le righe, `wo_a` / `wo_b` come GEMM. Il kernel
  mantiene gli arrotondamenti del passo di decode (sink in T; per il percorso non fuso a 64 head q' = T(T(scale) q), punteggi in T e le
  probabilità arrotondate a T prima di p·V, due passate): con essi l'attention del chunk è bit-identica ai passi di decode
  sui piccoli file di test; senza l'arrotondamento del sink la KL raggiungeva 4e-2 in singole posizioni, senza l'arrotondamento delle probabilità
  1.2e-2 sul file wide. I layer il cui indexer selezionerebbe (voci > index_topk, cioè contesti oltre
  4 · index_topk token) mantengono l'attention sequenziale per token.
- **qwen4_exp / qwen3_5**: proiezioni del chunk come GEMM / lookup mma, tile di expert come sopra; le parti ricorrenti come
  prima (Gated DeltaNet con un lancio per chunk, full attention per token).
- **glm5_next** (`encode_chunk_fast`, residente e in streaming): proiezioni come GEMM / lookup mma, il router come una
  GEMM float32, expert instradati in tile MMA (passate di <= 2048 righe di tile, `$SUC_GLM_EXPERT_PASS_ROWS`), kernel mHC
  a righe in batch, conv causale KDA sul chunk con un solo lancio, la ricorrenza gated-delta dell'intero chunk con un lancio
  per layer, MLA / DSA con latent in batch, LayerNorm e pool dell'indexer, le query assorbite e le proiezioni dei valori
  per head come GEMM (W_k trasposta per layer DSA al primo uso, ~184 MB sul file reale) e un lancio `glm_attn_chunk`
  per layer con la selezione DSA di ogni riga. Con `SUC_FAST_OFF=ptra` (lasciati solo questi kernel a righe in batch) il
  percorso in batch è bit-identico al prefill per token sul file reale (512 posizioni). `$SUC_GLM_PROFILE=1`: profilo per stadio.
  Server: in batch di default con chunk da 512.
- Chunk predefiniti: sostituiti il 2026-10-12 (sezione successiva: 4,096 / 1,024 token); `--prefill exact` mantiene i
  chunk precedenti.
- Entità della differenza (esecuzioni di sviluppo sull'M1 Ultra, `prefill_batch_vs_token.py`): i modelli frontier MoE sono molto
  più sensibili di Qwen3-8B (KL 1e-3): DeepSeek-V4-Flash tcq 512 token KL media 0.031 / top-1 94.1 %, GLM-5.3-Flash
  tcq 0.011-0.013 / 93.9 %. Qualsiasi singola parte veloce da sola (anche solo la GEMM del router, i cui punteggi differiscono da quelli del gemv
  negli ultimi bit float) dà KL ~5e-3 su DeepSeek e ~1.2e-2 su GLM: il routing degli expert quasi in parità su 43 / 42 layer
  MoE amplifica variazioni a livello di float; con tutte le parti veloci disattivate i percorsi sono bit-identici. La coda
  (`scripts/studio_fast_prefill_queue.sh`) registra i valori a 2,048 token per ogni modello (report §1c).
- Logit per riga: `--logits-out` insieme a un `--prefill batch|exact` esplicito calcola i logit di ogni riga del prompt
  dentro i chunk in batch (`*Model::forward_rows`), che `scripts/prefill_batch_vs_token.py` confronta con
  `--prefill token`. Senza `--prefill`, `--logits-out` esegue ancora passi di decode (i controlli incrociati rispetto a Python).
- Diagnostica: `$SUC_FAST_OFF` (lettere delle parti da mantenere esatte per riga: p proiezioni, a attention, t tile di expert, r router),
  `$SUC_MM_BM`, `$SUC_TCQ_MM=0`, `$SUC_TCQ_RUNS=1`, `$SUC_MXFP4_RUNS=0`, `$SUC_DS4_ATTN_RND`.

### Prefill dopo il confronto con ds4 (2026-10-11, docs/prefill_vs_ds4.md)

- **deepseek_v4: lightning indexer in batch.** I chunk le cui righe CSA superano `index_topk` voci (contesti oltre
  4 · 512 token) non ripiegano più sull'attention token per token.
  - `attn_chunk_fast` proietta le query / i pesi dell'indexer di tutte le righe come GEMM, poi esegue rope / Hadamard / FP4
    su tutte le righe.
  - Punteggi: `ds4_index_scores_mm`, tile simdgroup-matrix da 32 voci × 64 head. `$SUC_IDX_MM=0` ripiega su
    `ds4_index_scores_rows`, l'aritmetica di decode per voce.
  - Selezione: `ds4_index_select_rows`, la ricerca esatta della soglia e la regola delle parità di `q4_index_select`, scritta come
    lista crescente di voci per riga.
  - `ds4_attn_chunk` fa attention sulla finestra più quella lista.
  - `$SUC_FAST_OFF=i` ripristina il percorso sequenziale per quei layer.
- **Lanci a righe in batch (`$SUC_FAST_OFF=h` = i lanci per riga, stessi bit).** Ogni kernel locale alla riga che girava una volta
  per riga del chunk ora gira una volta per chunk:
  - deepseek_v4: pre e post delle hyper-connection, routing, output del MoE, embedding.
  - qwen4_exp / qwen3_5: norm dell'hyper-mix, act e mix low-rank; inject; routing; somma dello shared expert; norm / somma dense;
    le norm di q / k di Gated DeltaNet.
  - I kernel prendono la riga dalla griglia. L'ultima riga del chunk ora vive nel ring come le altre e viene copiata
    in `hs` per layer.
- **qwen4_exp: full attention in batch** (`full_attn_core_rows`; `$SUC_FAST_OFF=a` = per token). Su tutte le righe del chunk:
  norm di q / k e RoPE con posizioni per riga, le chiavi dell'indexer, i blocchi completati nel chunk con un unico lancio
  `q4_block_key`, `q4_index_scores_rows`, i blocchi top-k di ogni riga come liste crescenti di blocchi
  (`ds4_index_select_rows`), e un lancio `q4_sdpa_vector_sel` su (head, righe) con il conteggio causale delle chiavi di ogni riga.
  Le righe oltre il budget visitano solo i loro blocchi selezionati più la coda incompleta (modalità lista), non ogni chiave con un
  test della mask. Questo è sempre il kernel a passata singola; il decode
  passa a due passate da 1,024 chiavi sull'M1 Ultra. Anche il gate sigmoide gira su tutte le righe.
- **glm5_next: attention DSA solo sui pool selezionati.**
  - `glm_attn_chunk` percorre la lista crescente dei pool di ogni riga più la coda incompleta, invece di tutte le `p + 1` posizioni
    con un test della mask. Stesse chiavi, stesso ordine, stessi bit.
  - I lanci `q4_index_select` per riga sono diventati un unico `ds4_index_select_rows`.
  - I punteggi dei pool usano `ds4_index_scores_mm` (MODE 1).
- **Punteggi mantenuti nella memoria del threadgroup.** `ds4_attn_chunk` e `glm_attn_chunk` (`RND = 1`) calcolano ogni punteggio una volta e
  lo mantengono, come valore T, nella memoria del threadgroup: fino a 2,048 / 2,560 chiavi per head, 16 / 20 KB. Prima lo
  ricalcolavano a ogni passata. Stessi bit.
- Profilo per stadio: `$SUC_DS4_PROFILE=1` ora suddivide anche `attn_proj_comp`, `attn_indexer`, `attn_kernel` e `attn_wo`.
- **Chunk predefiniti** (suc1-generate, quando `--chunk` non è dato, al massimo il prompt): 4,096 token per deepseek_v4 e
  glm5_next, e 1,024 oltre i prompt da 8,192 token. Un prompt da 32k in chunk da 4,096 ha esaurito la memoria GPU sull'
  M1 Ultra con il file DeepSeek da 97 GB. qwen4_exp / qwen3_5 usano 1,024. Questi valori vengono dagli sweep a 8k token sull'M1 Ultra: DeepSeek tcq 134.5 →
  147.9 tok/s, GLM tcq 99.8 → 105.9, Qwen3.8 side8 GPU 3.03 → 2.80 ms/token.
  `suc1-server`: chunk di 64 (qwen4_exp, qwen3_5, deepseek_v4) e 512 (glm5_next) se `--chunk` non è indicato.
- **Righe n-gram di qwen4_exp**: i gather dell'host per i token di un chunk girano in parallelo. Con la page cache fredda sono
  letture casuali da SSD; il ciclo seriale costava fino a ~1.9 ms di tempo host per token del prompt a 8k sullo Studio.
  `prefill_ngram_s` è ora in `--json`.

Test: `test_fast_batched_prefill_close_to_token` in `tests/unit/test_cpp_{deepseek_v4,qwen4,qwen3_5,glm5}.py`
(KL / top-1 di ogni posizione rispetto al prefill per token sui piccoli file casuali, chunk che dividono gruppi del compressore /
pool, altezze dei tile forzate, expert in streaming); i test di bit-identità ora passano `--prefill exact`.

## Punti aperti

- Decode con batch > 1 (oggi una richiesta alla volta).
- Prefill: i tile degli expert instradati (8–16 righe contro i tile dequantizzati da 32 token di ds4) e l'attention scalare del chunk
  sono il divario rimanente rispetto a ds4 (docs/prefill_vs_ds4.md).
- Decode: Qwen3.8-Flash-Next è il 16 % più lento di ds4 (docs/decode_speed_vs_ds4.md).
- CMake rispecchia `build.sh` ma non fa parte della build verificata.
