# Engine 1.4.7 qualification status

Completed on 19 October 2026. All six selected models passed full same-reference
KL equality and all twelve 8k/32k paired checks. Actual CLI/server/bundled-engine
smoke, bounded session/cache/cancellation tests and Universal suites passed.
Model files and memory policy are unchanged. No main-branch merge or physical
build-28 GUI acceptance is implied. MTP is a separate post-release project.

See [release and setup](RELEASE_1.4.7.md),
[final qualification](FINAL_QUALIFICATION_1.4.7.md),
[reference KL](REFERENCE_KL.md), [performance](COMBINED_RESULTS.md) and
[inference behavior](INFERENCE_1.4.7.md). Internal host/queue logs and raw
machine-specific result JSONs are excluded from this public source snapshot.

## Studio 1.4.7.03 chat checkpoints

Qualified Qwen hybrid prefix checkpoints retain bounded recurrent state without duplicating KV rows or changing the existing SUC template. The first request and earlier-prefix changes still require full prefill. See [checkpoint evidence](CHAT_PREFIX_CHECKPOINTS.md) and [release/setup](RELEASE_1.4.7.03.md).

## Studio 1.4.7.04 progress and live telemetry

Actual helper/SSE progress, model-loading phases, cancellation/replay and server-token live metrics passed. Universal build and extracted installer were verified; GUI acceptance remains manual. See [release/setup](RELEASE_1.4.7.04.md).

## Studio 1.4.7.05 candidate and Android 0.1.4

Loading activity follows engine phases; resident-memory readings are separate, without completion percentages. Adaptive history prefill 1024/64 is opt-in; conservative Batch remains the default. Qwen 8k prefill measured 67.673 to 28.834 seconds in one pass per profile. Six fixed hard cases scored 5/6 for both profiles; full hard-suite and reference-KL qualification of the adaptive profile remain pending. Android Verify refreshes stale model IDs. MTP is shelved. See [release and limits](RELEASE_1.4.7.05.md).

## Studio 1.4.8 and 1.4.8.02

Persistent per-session message queues, context indicators, manual/automatic
compaction, chat commands and explicit plugin invocation are implemented.
Long transcripts retain stable previews and bounded history pagination.
Universal packaging and Intel/M1 context regressions passed. No new GPU/KL
measurements or real-model summary-quality qualification are implied.
See [1.4.8](RELEASE_1.4.8.md), [1.4.8.02](RELEASE_1.4.8.02.md) and
[verification evidence](RELEASE_1.4.8.02_QA.json).

## Studio 1.4.8.03

Compaction uses a dynamic output budget and rejects truncated or reasoning-only
summaries, with one bounded retry and original history preserved. Scheduled
routines reserve output headroom, require a final answer and remain running
during preflight compaction. Long-chat rendering uses bounded pages, collapsible
historical reasoning, background context sampling and throttled live metrics.
Intel UI/functional tests and native M1 context/helper checks passed. The engine
and memory policy are unchanged; external LM Studio latency is documented
separately from client rendering costs. See [release](RELEASE_1.4.8.03.md) and
[verification evidence](RELEASE_1.4.8.03_QA.json).

### SUC Studio 1.4.8.03 — build 36

Completata la risoluzione dei nomi e degli alias MCP non ambigui: le chiamate usano il server connesso corretto. L’avvio del browser per la ricerca web rispetta l’approvazione automatica della sessione; le approvazioni manuali restano attive quando selezionate. Verificati catalogo MCP, limiti degli schemi e dei risultati, ricerca web e installer Universal. Motore e memory policy invariati.

### SUC Studio 1.4.9 · build 37 / Android 0.1.5 · codice 6

Interfacce IT/EN con inglese predefinito e lingua salvata negli archivi esistenti. Prerequisiti shell espliciti su macOS: directory e azioni del modello necessarie; controllo disabilitato quando mancano. Guide, menu, dialoghi e suggerimenti localizzati. Verificati archivio macOS reale, catalogo, 14 test Kotlin, suite UI Android e pacchetti. Motore, memory policy e metriche dei modelli invariati. Dettagli in `RELEASE_1.4.9_IT.md` e `RELEASE_1.4.9_QA.json`.

## Studio 1.4.9.02 · managed shell commands

Universal build 38 adds per-session command jobs, incremental bounded output and status/wait/stop tools. Waiting no longer kills a command at 30 seconds. Real subprocess tests passed on Intel and M1 Ultra, together with queue, context and localization regressions. Default context is 16000 tokens; saved limits remain intact. Engine, memory policy and Android 0.1.5 unchanged. See `RELEASE_1.4.9.02_QA.json`.

### Studio 1.4.9.02 · build 39

Fixed complete tool envelopes rejected when a remote model emits numeric wait/timeout/offset arguments. Numeric normalization preserves tool validation and approvals; booleans, null, nested structures and multiple envelopes remain rejected. Real subprocess regression passed on Intel and M1 Ultra; Universal installer verified. User sessions and engine unchanged.

## Studio 1.4.9.03 · build 40

Conservative IT/EN recovery for promises without tool actions and malformed action envelopes shares two attempts per turn within the action budget. Corrected requests still require normal permissions/approvals; no guessed JSON executes. Exhaustion ends in error and pauses queued work. Streaming phases no longer regress on incomplete usage counts. Real streamed fixture/file/artifact tests passed on Intel and M1 Ultra; Universal installer verified. Engine, memory policy and Android unchanged. See `RELEASE_1.4.9.03_IT.md` and `RELEASE_1.4.9.03_QA.json`.

### Studio 1.4.9.03 · build 41

Tool-call wire markers stay hidden across all streaming prefixes while commentary, original history and code examples remain intact. Automatic compaction uses 99% of projected input without output reservation or overriding valid server counts with larger byte estimates. A sole missing outer object delimiter is tolerated only after normal stop with unchanged complete action/arguments and full validation. Protocol/file/artifact and full context suites passed on Intel and M1 Ultra; Universal installer verified. Engine and memory policy unchanged.

### SUC Studio 1.4.9.03 · build 42

Rimosso il pannello fisso Comandi della sessione e resi espandibili i riquadri cliccando tutta l’intestazione. Registro azioni con schede più grandi, stato/ora/anteprima e dettagli richiesta/risultato separati; consultabile durante l’esecuzione. Collaudo UI isolato superato su Intel e M1 Ultra, installer Universal verificato. Motore, memory policy e Android invariati.

### SUC Studio 1.4.9.03 · build 43

Budget della compattazione coerenti in token, senza sottrarre byte dal limite. Memorie precedenti troppo grandi ricompattate senza tagli, retry entro finestra. Risultato operativo conservato una sola volta dopo errore o Stop, senza riesecuzione. Intera suite contesto/plugin HTTP offline superata su Intel e M1 Ultra, incluso il caso memoria da circa 38 KB/finestra 40000. Installer Universal verificato; soglia 99%, motore, memory policy e Android invariati.

### SUC Studio 1.4.9.03 · build 44

Reasoning non reinviato escluso dal contesto futuro con conteggi server o stime esplicite. Controlli thinking per LM Studio/router e override persistenti. Sintesi breve con ricompattazione della memoria e retry su metà Unicode dopo troncamenti, senza commit parziali. Budget effettivo e avvisi di contesto corretti, soglia 99% invariata. Suite contesto/remota/telemetria/IT-EN su Intel e M1 Ultra; prova reale del Router su dati neutri conclusa senza reasoning. Installer Universal verificato; motore, memory policy e Android invariati.

## SUC Studio 1.4.9.04 · build 45

Chiamate MCP gestite come job: stato, attesa e annullamento per chat, nessun termine fisso a 30 secondi, scadenza totale facoltativa, approvazioni e Stop preservati. Verificati stdio/HTTP oltre 30 secondi e assenza di duplicazioni su Intel e M1. Dichiarazioni Apple Events IT/EN ed entitlement dell’app. Motore e Android invariati. Esiti e limiti nel rapporto `RELEASE_1.4.9.04_QA.json`.

## SUC Studio 1.4.9.05 · build 46

Aggiornamenti inviati durante il lavoro recepiti tra le operazioni: una proposta ancora da eseguire viene riconsiderata, mentre il risultato di uno strumento già avviato precede la correzione. Approvazioni superate ritirate senza concedere permessi. Test Intel/M1 su FIFO, hook, limiti, allegati, Stop e compattazione. Decode, motore e Android invariati. Verifiche e limiti in `RELEASE_1.4.9.05_QA.json`.
