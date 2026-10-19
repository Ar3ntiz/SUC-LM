# Cache dei turni Qwen — 1.4.7.03, build 30

- [x] Analisi del renderer e del mancato riuso del prefisso completo.
- [x] Checkpoint ibrido limitato, reset/errore/cancellazione sicuri e contratto condiviso app/server.
- [x] Regressioni CPU e Metal; suite Universal complete Intel/M1.
- [x] Confronti appaiati sul Qwen reale: logits, continuazioni, thinking, modifica e cambio sistema.
- [x] Benchmark HTTP e helper app reale: chat, cancellazione, API autenticata, unload.
- [x] Conteggi indipendenti HF, tempi osservati e parser Swift dei tok/s.
- [ ] Confezionamento e verifica finale di installer, archivio CLI/server, checksum e sincronizzazione.

## Funzionamento

Un solo checkpoint per modello, allineato al chunk e almeno 64 token prima della fine del prompt. Si copiano stati Gated DeltaNet, convoluzioni, PLE e contesto n-gram CPU, più la posizione; KV e chiavi QSA del prefisso rimangono in sede perché append-only. Il ripristino invalida gli ultimi logits e rielabora una coda non vuota. Le righe successive vengono ignorate dai limiti causali e sovrascritte. Nessuna modifica al renderer SUC esistente, ai pesi, ai kernel o alla memory policy.

Le copie sono limitate a 128 MiB. Sul Qwen3.8-Flash-Next side8 misurato occupano 115.642.384 byte, circa 110,3 MiB. Prefisso incompatibile, checkpoint indisponibile, errore, annullamento o reload comportano fallback/invalidazione sicuri. `SUC_PREFIX_CHECKPOINTS=0` conserva il solo prefisso live precedente; `SUC_PREFIX_CACHE=0` disattiva entrambi. Non sono introdotte cache distinte per ogni sessione o salvataggi SSD.

## Prestazioni reali

M1 Ultra, modello side8 locale selezionato, BF16, batch 1, chunk 64, massimo 16 token generati, dialoghi sintetici con identici input fra le varianti. Processi sequenziali, checkpoint disattivi/attivi, nessun altro modello caricato. Tempi alla prima porzione di testo ricevuta via HTTP, non tok/s né tempi del parser UI.

| Caso | Token del prompt | Senza checkpoint | Con checkpoint | Token riusati |
|---|---:|---:|---:|---:|
| Turno successivo, circa 2k | 2165 | 20.991 s | 1.126 s | 2048 |
| Modifica dopo completamento | 2165 | 19.497 s | 1.079 s | 2048 |
| Rigenerazione dopo completamento | 2165 | 19.499 s | 1.082 s | 2048 |
| Turno con thinking | 2156 | 19.490 s | 1.043 s | 2048 |
| Turno successivo, circa 11,7k | 11679 | 108.024 s | 0.936 s | 11584 |

Tutte le risposte e i motivi di fine coincidono fra varianti (hash verificati). Il primo prompt lungo, senza prefisso disponibile, richiede ancora circa 108–110 secondi; il cambio del system prompt forza correttamente il ricalcolo. I casi modifica/rigenerazione sono eseguiti dopo il completamento; l’annullamento invalida la cache e la richiesta seguente rielabora il prompt completo. Sequenza fissa baseline/candidato, non una stima statistica universale: la differenza di lavoro è confermata dai conteggi `cached_tokens`. Il caso denominato `thinking_initial` riusa un prefisso condiviso della sequenza e non rappresenta una partenza a freddo.

Dati: [CHAT_PREFIX_REAL_QWEN.json](CHAT_PREFIX_REAL_QWEN.json). Questi tempi non sostituiscono le misure di throughput a prefisso vuoto delle model card.

## Correttezza e limiti della qualifica

Sul Qwen reale: otto cronologie, thinking acceso/spento, modifica e cambio sistema. Tutti i **235.904.000 logits di 950 posizioni identiche** coincidono bit per bit: errore massimo 0 e KL massima 0 fra ripristino e ricalcolo completo. Coincidono l’ultima distribuzione normale e quattro token greedy di continuazione per caso. È un confronto sul medesimo modello/protocollo, non una nuova KL contro una reference esterna. I tempi del runner numerico includono head per riga e compilazione iniziale: non sono benchmark di velocità utente.

Le fixture tiny/wide verificano QSA sparso; il caso aggiuntivo con chunk 63 ripristina il checkpoint alla posizione 126, interna a un blocco QSA: 361 posizioni identiche. Il Qwen3.8-27B selezionato ha esercitato il fallback senza checkpoint: 991 posizioni identiche, senza attribuirgli un guadagno di cache. L’accelerazione qui qualificata riguarda il Flash-Next side8; il grafo condiviso non garantisce uno snapshot entro il limite per ogni modello.

CPU M1: 1.220 controlli generali e 52 sui parametri, zero errori; regressioni prefisso anche su Intel. Metal: 705 controlli, zero errori. Suite Universal Intel/M1 passate. Sette smoke reali sui binari finali CLI/server/helper: generazione, SSE, caricamento, chat, autenticazione e cancellazione. L’helper sul Qwen reale passa anche stop/ricalcolo e unload senza mappature trattenute. Il vecchio comportamento di unload è riprodotto su fixture; il nuovo helper rilascia le funzioni catturate dai task già conclusi.

Evidenze: [logits reali](CHAT_PREFIX_REAL_LOGITS.json), [fixture](CHAT_PREFIX_FIXTURE_RESULTS.json), [controlli finali](CHAT_PREFIX_FINAL_CHECKS.json), [helper Qwen](CHAT_PREFIX_APP_QWEN.json), [smoke](RELEASE_SMOKE_1.4.7.03.json). CLI/server ricompilati identici byte per byte a quelli del benchmark; l’helper ricompilato è stato nuovamente collaudato. L’input C++ parentale è comune alle distribuzioni, documentato nei manifest.

## Verifica dei tok/s

Il tokenizer Rust HF riconta il prompt emesso dall’esatta funzione C++ condivisa dal server, senza duplicare il renderer nel test. Confermati 2.136 token iniziali e 2.165 nel seguito; dopo 2.048 token riusati il lavoro è di 117 token. I tempi di prefill del motore sono confrontati con l’arrivo dell’evento misurato indipendentemente dal client (scarti di pochi millisecondi). La ricostruzione Swift usa i sorgenti di produzione.

Il frame API reale riporta 117 token elaborati in 1.072738 s: il parser Swift restituisce **109.0667 tok/s**, escludendo i token in cache. La misura indipendente dopo annullamento è 110.7893 tok/s interna contro 110.7872 osservata.

Il benchmark storico della model card usa invece CLI chunk 1024: 2.048 / 5,5945 s = 366,0738 tok/s, TTFT 5,6239785 s. I 64 token di quel benchmark sono l’output generato, non il chunk di prefill. App/server chunk 64 misurano circa 100–110 tok/s. I due profili non vanno presentati come equivalenti. Nel test hard precedente chunk 64 ha ottenuto 100/120, chunk 1024 98/120: è un indizio di differenza numerica, non una dimostrazione universale di superiorità qualitativa del 64. La selezione di profili veloce/prudente nell’app e la qualifica dei checkpoint a 1024 restano un lavoro separato, non incluso nella 1.4.7.03.

Il renderer SUC ereditato non è pienamente equivalente al Jinja HF nei turni assistant storici: nel caso sintetico con `<think>` incorporato HF produce 2.176 token, contro i 2.165 effettivamente elaborati da SUC. Il conteggio indipendente sul rendering C++ coincide col server; i tok/s non sono falsati. Questa versione conserva il formato precedente per isolare la cache. Una futura armonizzazione del template richiede una qualifica separata, non viene fatta silenziosamente durante questa correzione.

Dati: [tempi e tokenizer](CHAT_PREFIX_PREFILL_TELEMETRY.json), [parser Swift](CHAT_PREFIX_SWIFT_TELEMETRY.json). Nessuna chiave API o conversazione privata è registrata.

## Distribuzione

SUC Studio Universal 1.4.7.03 build 30, app/helper shell arm64+x86_64, motore arm64. Archivio CLI/server ARM64 dalla stessa base C++. Template SUC, file modello e memory policy invariati; MTP escluso. Nessuna installazione o pubblicazione automatica. Il collaudo fisico dell’interfaccia resta manuale; non sono cambiate le view oltre all’identità di versione.

Installer e archivio CLI/server verificati senza installazione: firma ad hoc, payload, architetture, documentazione inclusa e hash condivisi. Evidenza: [verifica dei pacchetti](PACKAGE_1.4.7.03.json).
