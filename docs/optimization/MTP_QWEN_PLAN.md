# MTP — Qwen3.8-Flash-Next side8

Stato: pianificazione; nessun predictor implementato o benchmark MTP completato. Si applicano tutti i gate del [piano generale](MTP_PLAN.md), in particolare rollback, sampling e confronto numerico del verificatore.

## Base conosciuta

Target: Qwen3.8-Flash-Next-SUC1-side8 già selezionato, locale sul Mac Studio. L'adapter attuale scarta `mtp.*`. L'inventario storico identifica 31 tensori su 28 shard e stima 2,607 miliardi di parametri dedicati, circa 5,21 GB BF16. Sono dati dell'inventario originale, non misure di un sidecar già costruito; forme e norme dedotte devono essere confermate.

Reference e architettura: [Qwen SUC](../arch/qwen4_exp.md), [implementazione pubblica da fissare a commit](https://github.com/ggml-org/llama.cpp/pull/28243). La PR è chiusa e non dimostra il supporto nella release corrente. Embedding/output condivisi e semantica del mixer richiedono revisione.

## Task e avanzamento

- [x] Q0 — Individuare i tensori e le fonti preliminari.
- [ ] Q1 — Fissare revisione/hash del modello e della reference; verificare header, `fc_hidden`, `fc_embedding`, assegnazione `pre_fc_norm_*`, mixer e layer predictor. Diagramma del forward con forme per passaggio.
- [ ] Q2 — Sidecar separato BF16, manifest di compatibilità e checksum; embedding/head condivisi senza copie. Calcolare e misurare RAM totale, Metal working set e margine per contesto/stati.
- [ ] Q3 — Esporre lo hidden state target nella posizione corretta e implementare predictor a una proposta. Confronto degli intermedi/logits con reference equivalente; distinguere differenze dovute alle proiezioni SUC quantizzate.
- [ ] Q4 — Transazioni per ricorrenze DeltaNet, convoluzioni, PLE, n-gram, KV/indexer/QSA, posizioni/logits e predictor. Snapshot transazionale distinto dal checkpoint chat 128 MiB; non assumere che il limite e l'API attuali bastino per MTP.
- [ ] Q5 — Rifiuto totale/parziale, accettazione completa, EOS, confini QSA e cancellazione: stato dopo commit/rollback equivalente al replay sui token accettati. Controllare anche turno successivo, edit e cambio system prompt.
- [ ] Q6 — Verifica greedy con token identici al target seriale; controllo della traiettoria numerica della verifica batch. Sampling solo dopo accettazione/correzione probabilistica qualificata.
- [ ] Q7 — KL sugli stessi token/reference e hard 120 nel protocollo originale; nessuna nuova metrica attribuita ai SUC1 prima dei risultati.
- [ ] Q8 — A/B senza/con MTP, contesti 2k/8k/32k compatibili con memoria, codice/prosa/thinking; profondità 1 e poi 2/4 se il grafo lo consente. Warmup e almeno tre coppie alternate.
- [ ] Q9 — CLI/server/helper condivisi, Settings sidecar e capacità/metriche; smoke reali e pacchetti dopo il superamento dei gate comuni.

## Gate specifici

Il vantaggio della cache chat già misurato non è un vantaggio MTP. Usare chunk 64 per la baseline app/server: il profilo 1024 rimane separato. Misurare costo delle copie di stato per ciclo: il checkpoint persistente da circa 110,3 MiB non può essere copiato ripetutamente senza contabilizzarne il costo. Se il target batched non conserva il comportamento qualificato, correggere il verificatore prima di valutarne la velocità.

Il primo risultato atteso è l'inventario esatto Q1 e il budget Q2, non una previsione di speedup. Archiviare risultati sotto una cartella dedicata Qwen con revisioni, hash e protocollo; aggiornare le caselle con i relativi report.
