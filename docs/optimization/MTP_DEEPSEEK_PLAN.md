# MTP/DSpark — DeepSeek-V4-Flash

Stato: pianificazione; DSpark non implementato in SUC. Si applicano i gate comuni del [piano generale](MTP_PLAN.md). Questa famiglia non va trattata come una copia del predictor Qwen.

## Base conosciuta e punti da risolvere

Target: DeepSeek-V4-Flash-SUC1-tcq selezionato, derivato dal checkpoint Flash-0731 già analizzato. [Inventario e forward DSpark](../arch/deepseek_v4.md): l'adapter scarta `mtp.*`; gli originali descrivono tre stadi `mtp.0/1/2`, mentre `num_nextn_predict_layers` riporta 1. La configurazione di inferenza indica tre stadi. Risolvere esplicitamente la discrepanza fissando codice e revisione.

L'inventario stima 19,846 miliardi di parametri dedicati, 10,86 GB nella rappresentazione originale mista e 39,7 GB espansi BF16. Non sono dimensioni del futuro sidecar SUC o RAM misurata. Convertire tutto in BF16 insieme al target potrebbe superare il budget: prima il preventivo, poi l'estrazione.

Il grafo inventariato usa hidden dai layer target 40/41/42, tre blocchi con attenzione a finestra e MoE, embedding/head condivisi, markov head e confidence head. Il blocco draft comprende il token corrente più quattro noise token. Non presumere che numero di stadi, numero di proposte e profondità draft siano intercambiabili.

## Task e avanzamento

- [ ] D1 — Fissare reference originale, licenza e hash; verificare nomi, shape, scale FP8/FP4, noise token, stadi e configurazioni effettive. Confermare compatibilità del checkpoint scelto e disponibilità dei soli shard necessari su storage locale.
- [ ] D2 — Formalizzare il grafo: riduzione dei flussi HC target, `main_proj/main_norm`, costruzione della finestra, proiezione delle KV target con i pesi draft, dipendenze fra stadi, markov bias, confidence e proiezione finale condivisa.
- [ ] D3 — Definire sidecar versionato con scale e metadati. Reference di correttezza dai pesi originali dequantizzati per blocco; non obbligare un'espansione residente di 39,7 GB. Definire il formato runtime iniziale prima della conversione, con test di ogni transcodifica e di eventuale quantizzazione separata.
- [ ] D4 — Budget RAM/Metal per target + predictor + finestre e buffer, senza alterare memory policy. Se non entra residente, dichiarare il limite e valutare formato/contesto; niente benchmark su disco di rete o streaming SSD attivato implicitamente.
- [ ] D5 — API hidden/intermedi target dei layer necessari, predictor DSpark e confronto numerico su intermedi/logits con la reference. Solo dopo questa prova consentire un sottoinsieme di proposte del blocco nativo, se semanticamente valido.
- [ ] D6 — Transazioni su KV finestra, cache latenti, compressori/accumulatori, indexer, posizioni, HC/intermedi richiesti e stato draft/markov. Individuare tutti gli stati realmente mutati dal runtime; non limitarsi a troncare un contatore KV.
- [ ] D7 — Test rifiuto/accettazione parziale e totale, confini finestra/compressione, EOS, stop e cancellazione; ricostruzione dello stato equivalente al replay e nessuna invalidazione nascosta del riuso della chat.
- [ ] D8 — Verificatore target fedele al decode seriale; greedy identico e sampling corretto. Confidence/markov possono guidare le proposte, ma non sostituire l'accettazione target che preserva la distribuzione.
- [ ] D9 — KL sulla reference dequantizzata originale e sugli stessi token/posizioni del rilascio; hard suite completa e regressioni tool/thinking/multi-turno.
- [ ] D10 — Benchmark appaiati 2k/8k/32k compatibili con memoria, lunghezze proposta ammesse dalla reference; costo di ogni stadio e finestra, accettazione, draft/verifica/rollback, decode end-to-end e memoria.
- [ ] D11 — Integrazione CLI/server/helper/app e pacchetti, soltanto dopo i gate comuni; report separato con esito anche negativo se il costo DSpark annulla il vantaggio.

## Gate specifici

Il rischio principale è costo e memoria di tre stadi con 768 esperti complessivi, oltre all'esposizione degli hidden target intermedi. Occorre provare sul grafo reale che la verifica di più proposte ammortizzi i pesi senza alterare il comportamento qualificato. Non trasferire a DSpark tassi di accettazione o guadagni misurati su Qwen o altre GPU.

Produrre prima D1–D4, con stima documentata di disco/RAM e schema del grafo; autorizzazione a procedere sui modelli selezionati non autorizza a scaricare modelli alternativi o sostituire i file SUC1.
