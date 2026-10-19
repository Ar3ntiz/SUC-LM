# Piano MTP reale — avanzamento

Aggiornato il 19 ottobre 2026. La consegna del motore e dei pacchetti 1.4.7.03 è completata: il rinvio concordato è terminato e può iniziare la progettazione MTP. **MTP non è ancora implementato né misurato in SUC.** Prompt lookup e checkpoint della chat non costituiscono MTP.

Perimetro richiesto: i tre modelli grandi selezionati, Qwen3.8-Flash-Next side8, DeepSeek-V4-Flash tcq e GLM-5.3-Flash tcq, residenti sul Mac Studio M1 Ultra. Nessun modello segnaposto o su disco di rete, nessun cambio di modello senza accordo. Main rimane la base di lavoro; ogni fase deve avere risultati e commit separati. Predictor e qualifica restano specifici di ciascuna famiglia.

## Tre percorsi e ordine di esecuzione

| Modello | Predictor inventariato | Primo rischio da risolvere | Piano dedicato |
|---|---|---|---|
| Qwen3.8-Flash-Next side8 | Un layer, circa 2,61 B parametri | Mapping delle norme e rollback DeltaNet/PLE/n-gram | [Q1–Q9](MTP_QWEN_PLAN.md) |
| DeepSeek-V4-Flash tcq | DSpark, tre stadi, circa 19,85 B parametri | Grafo a blocchi, memoria e hidden target intermedi | [D1–D11](MTP_DEEPSEEK_PLAN.md) |
| GLM-5.3-Flash tcq | Layer 45, circa 7,43 B parametri | Reference del forward e interazione mHC/DSA | [G1–G11](MTP_GLM_PLAN.md) |

Le dimensioni sono inventari storici da riconfermare, non misure di sidecar già prodotti. Esaminare prima reference, header e budget di **tutti e tre**; quindi implementare l'infrastruttura comune e il predictor Qwen come primo percorso completo. Seguono GLM e DeepSeek, rivedendo l'ordine se l'audit evidenzia un diverso rischio o una reference mancante. Il completamento Qwen non qualifica gli altri modelli. Tre esiti indipendenti, anche se uno non supera memoria o prestazioni.

Condividere solo ciò che è realmente comune: contenitore/manifest, capacità, telemetria, algoritmo di accettazione, test e API transazionali. Le implementazioni del predictor e del salvataggio/ripristino restano specifiche; DSpark non è un predictor Qwen con più layer. Benchmark e carichi reali sul Mac Studio sempre sequenziali.

## Decisioni e vincoli

- Conservare i tre SUC1 principali: ogni predictor sarà un sidecar opzionale, associato tramite architettura, revisione, tokenizer e hash. Non occorre rigenerare il modello principale per il primo prototipo.
- L'indice originale inventariato contiene 31 tensori `mtp.*` su 28 shard; i conteggi storici stimano 2,61 miliardi di parametri dedicati, circa 5,21 GB BF16. Dimensioni, nomi e assegnazione delle norme restano da confermare direttamente dagli header. Non sono una misura del futuro sidecar SUC né della RAM aggiuntiva.
- Riutilizzare embedding e proiezione output del target soltanto dopo avere verificato la semantica della reference. Considerare anche la loro quantizzazione SUC quando si confronta il predictor.
- Preservare memory policy e limite Metal reale: aggiungere esplicitamente pesi, stati e buffer MTP al preventivo. Nessun aumento silenzioso dei limiti o passaggio a SSD streaming. Se manca memoria, rifiuto motivato o fallback senza MTP.
- MTP disattivo per impostazione predefinita fino alla qualifica. Il percorso disattivo deve conservare risultati, cache e comportamento del rilascio attuale.
- Test GPU e build sequenziali sul Mac Studio. Prima di un carico reale verificare i processi attivi; non interrompere il server o il modello dell'utente senza accordo.

## Task in ordine di esecuzione

### M0 — base e fattibilità

- [x] M0.1 — Chiudere il rilascio precedente, con artefatti e misure verificati.
- [x] M0.2 — Inventario preliminare dei nomi MTP e individuazione di implementazioni pubbliche.
- [x] M0.3 — Definire sequenza di lavoro, prove e criteri di accettazione in questo file.
- [ ] M0.4 — Fissare commit della reference, licenza e hash del checkpoint originale; rileggere il grafo predictor e verificare dagli header tutti i tensori, dtype e dimensioni, incluse le norme finora dedotte.
- [ ] M0.5 — Registrare baseline senza MTP: prompt/token/posizioni, template, chunk, sampling, memoria, decode e TTFT; stabilire se il grafo target può verificare più proposte in modo utile senza cambiare la traiettoria numerica qualificata.

**Uscita:** inventario riproducibile e schema del grafo. Se la verifica richiede lo stesso costo seriale del decode più il costo del draft, segnalare il limite prima di promettere accelerazione.

### M1 — sidecar e reference

- [ ] M1.1 — Estrarre dai soli shard necessari il predictor del checkpoint selezionato, su storage locale. Niente download dell'intero modello o conversione del target esistente.
- [ ] M1.2 — Definire contenitore versionato, manifest e controlli di compatibilità; loader con errori chiari per tensori mancanti, forme errate, tokenizer/revisione incompatibili o file corrotti.
- [ ] M1.3 — Qwen: prima versione BF16 per isolare gli errori. DeepSeek/GLM: reference dequantizzata per blocco e scelta esplicita del formato runtime prima dell’estrazione, evitando espansioni BF16 residenti non sostenibili. Qualsiasi transcodifica o quantizzazione del sidecar richiede confronto numerico separato.
- [ ] M1.4 — Preventivo e misura memoria con sidecar, buffer di verifica, KV e stato predictor. Caricamento/scaricamento senza mappature trattenute.

**Uscita:** sidecar separato, hash e byte effettivi; nessuna duplicazione involontaria di embedding/head e nessuna modifica ai SUC1 esistenti.

### M2 — predictor C++/Metal

- [ ] M2.1 — Esporre lo hidden state target necessario; definire precisamente posizione e allineamento fra token corrente, token proposto e hidden state.
- [ ] M2.2 — Implementare fusioni, norme, attenzione/MoE e stato predictor secondo il grafo fissato; cominciare con una proposta per ciclo.
- [ ] M2.3 — Confrontare hidden intermedi e logits del predictor su token/posizioni identici, prima con pesi e proiezioni equivalenti alla reference, poi con le proiezioni condivise SUC. Separare errore di porting da differenze di quantizzazione.

**Uscita:** report numerico completo e tolleranze motivate prima dei test. Nessun benchmark di velocità qualificato se il predictor non supera questa fase.

### M3 — transazioni, verifica e rollback

- [ ] M3.1 — Inventariare tutti gli stati modificati: ricorrenze DeltaNet, convoluzioni, PLE, n-gram CPU, KV, indexer/QSA, logits, posizioni e stato predictor.
- [ ] M3.2 — Implementare begin/commit/rollback del ciclo speculativo; accettazione parziale deve conservare solo il prefisso accettato e il token correttivo. Non basta troncare la KV.
- [ ] M3.3 — Separare lo snapshot transazionale dal checkpoint persistente della chat, oppure serializzarne esplicitamente l'uso. Nessuna sovrascrittura silenziosa del punto di riuso della conversazione.
- [ ] M3.4 — Test deterministici: rifiuto del primo token, accettazione parziale/totale, EOS/stop, confine QSA, contesto pieno, cancellazione, errore e turno successivo. Confrontare logits dopo rollback con replay senza MTP sugli stessi token.
- [ ] M3.5 — Verificare se il batch di controllo altera i logits rispetto al decode seriale. Non usare il prefill veloce come verificatore equivalente senza dimostrarlo; se serve una diversa traiettoria numerica, documentare e qualificare separatamente o mantenere il prototipo disattivo.

**Uscita:** nessuno stato residuo dopo rifiuto/cancellazione; corretta interazione con la cache 1.4.7.03 e con il suo fallback per snapshot non disponibili.

### M4 — accettazione e qualità

- [ ] M4.1 — Greedy: token ed EOS identici al target senza MTP, con stesso prompt, template, chunk e configurazione. Un cambio di token non può essere giustificato da un miglior punteggio medio.
- [ ] M4.2 — Sampling: implementare accettazione con rapporto fra probabilità target e draft e distribuzione residua corretta, usando le distribuzioni effettive dopo temperature, penalità, top-k/top-p e vincoli. Gestire massa nulla e RNG; il medesimo seed non implica la stessa sequenza.
- [ ] M4.3 — Test statistici su distribuzioni piccole note, incluso draft volutamente errato; poi verifica sul modello reale. Se il sampling non è qualificato, rendere MTP disponibile soltanto in greedy sperimentale, senza chiamarlo supporto completo.
- [ ] M4.4 — Confronto KL su identici token, posizioni e reference già usata, senza sostituirlo con KL dell'ultima posizione di un chunk. Registrare anche il confronto target verificato/target seriale. Reference mancante significa qualifica pendente.
- [ ] M4.5 — Hard suite completa 120 quesiti nel protocollo precedente, report degli scostamenti e controllo di risposte, tool, thinking, stop e chat multi-turno. Non confondere questo risultato con la KL.

**Uscita:** correttezza greedy e, se esposto, sampling qualificato; nessuna regressione nascosta dal guadagno di velocità.

### M5 — prestazioni e decisione

- [ ] M5.1 — A/B sullo stesso target locale, profondità draft 1 iniziale, poi 2/4 solo se sostenibili; contesti 2k/8k/32k o massimo effettivamente disponibile, output e token di input fissati. Thinking e codice/prosa separati.
- [ ] M5.2 — Scartare warmup e alternare almeno tre coppie A/B; riportare tempi e dispersione, costo draft/verifica/rollback, accettazione, token per ciclo, decode end-to-end, TTFT e picco memoria. Niente GPU concorrenti.
- [ ] M5.3 — Distinguere prefill iniziale, turno con cache e decode: MTP mira al decode e non sostituisce la cache dei turni. Chunk 64 resta la base app/server; 1024 è un asse di qualifica separato.
- [ ] M5.4 — Promuovere solo se il guadagno end-to-end è ripetibile e supera la variabilità. Obiettivo preliminare: almeno +10% mediano nella matrice dichiarata, senza regressioni sistematiche maggiori del 5%; soglie da fissare nel protocollo prima delle misure, senza correggerle a posteriori per far passare il test.

**Uscita:** tabella misurata con esito positivo o negativo. Nessuna promessa di moltiplicatore; costo predictor e costo di verifica possono annullare il guadagno su M1 Ultra.

### M6 — integrazione e consegna

- [ ] M6.1 — Parametri espliciti sidecar/on-off/profondità in CLI e server, capacità effettive e telemetria distinte; stessa implementazione nel motore parentale incluso nell'app.
- [ ] M6.2 — Settings macOS con selezione sidecar, compatibilità, stima memoria e stato attivo/fallback. Nessun uso locale su Intel. Android resta client remoto: adeguare il parsing solo se il protocollo lo richiede.
- [ ] M6.3 — Regressioni CLI, API OpenAI, SSE, tool, cancellazione, cache, unload e suite Universal; binari e payload realmente collaudati, non soltanto compilati.
- [ ] M6.4 — Aggiornare report, istruzioni, changelog, manifest e checksum; model card soltanto con metriche effettivamente misurate e protocollo chiaro. Nuova versione da decidere alla qualifica, non assegnata dal solo piano.
- [ ] M6.5 — Commit, allineamento cartella sorgenti/Mac Studio ed export pubblico; nessuna installazione o pubblicazione esterna automatica.

## Avanzamento per famiglia

| Percorso | Audit/reference/memoria | Sidecar/predictor | Rollback/qualità | Prestazioni | Integrazione |
|---|---|---|---|---|---|
| Qwen | Inventario preliminare; Q1–Q2 da completare | Da iniziare | Da iniziare | Non misurate | Da iniziare |
| DeepSeek | Inventario storico DSpark; D1–D4 da completare | Da iniziare | Da iniziare | Non misurate | Da iniziare |
| GLM | Inventario storico; forward G1–G4 da confermare | Da iniziare | Da iniziare | Non misurate | Da iniziare |

Aggiornare questo riepilogo insieme alle caselle del piano dedicato. Per ciascun modello registrare target e sidecar esatti; non aggregare KL, accettazione o speedup tra famiglie.

## Evidenze da produrre

Una cartella di risultati per fase con revisioni target/predictor/reference, hash, config e token di prova; inventario dei tensori, contratto del sidecar, confronto numerico predictor, matrice rollback, qualifica sampling, confronto KL/hard e benchmark appaiati. Aggiornare le caselle solo dopo aver ottenuto l'evidenza corrispondente; un processo avviato non significa task completato.

## Riferimenti verificati e limiti

- Inventario SUC: [architettura Qwen](../arch/qwen4_exp.md). Le dimensioni storicamente dedotte vanno ancora confermate.
- [PR llama.cpp 28243](https://github.com/ggml-org/llama.cpp/pull/28243), consultata il 19 ottobre 2026: chiusa. È una fonte di implementazione da fissare a commit e revisionare, non prova che il supporto sia integrato nella versione corrente o equivalente a SUC.
- [Documentazione sidecar del convertitore](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/raw/main/MTP/README.md), consultata il 19 ottobre 2026. Il formato GGUF non è caricabile dal runtime SUC1 corrente. Prestazioni pubblicate su altri hardware non sono risultati M1/SUC.

Prossimo lavoro eseguibile: **M0.4, audit di Q1, D1–D2 e G1–G2 per tutte e tre le famiglie**, seguito da budget memoria e baseline M0.5. Non estrarre o caricare pesi finché non è definito e verificato il contratto del predictor.
