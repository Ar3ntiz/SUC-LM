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
