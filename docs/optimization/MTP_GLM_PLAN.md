# MTP — GLM-5.3-Flash

Stato: pianificazione; predictor non implementato e forward esatto ancora da verificare. Si applicano i gate comuni del [piano generale](MTP_PLAN.md).

## Base conosciuta e incertezze

Target: GLM-5.3-Flash-SUC1-tcq già selezionato. L'[inventario SUC](../arch/glm5_next.md) distingue 45 layer target (0–44) e il layer MTP 45, escluso dall'adapter. Il predictor contiene `eh_proj` di forma [4096,8192], `enorm/hnorm`, un blocco DSA+MoE e `shared_head.norm`; i pesi output sono indicati come condivisi.

Sono stimati 7,433 miliardi di parametri MTP: circa 14,87 GB BF16 o 7,49 GB nella rappresentazione originale mista. Sono numeri di inventario, non byte del futuro sidecar né footprint runtime. Le note esistenti indicano forward non implementato nel codice Transformers allora consultato, assenza di parametri HC nel predictor e possibile riuso indexer con `index_share_for_mtp_iteration`. Questi elementi non bastano per dedurre il grafo: serve una reference verificabile.

## Task e avanzamento

- [ ] G1 — Fissare revisioni del checkpoint e di una reference eseguibile; verificare stato attuale del supporto, licenza, header, scale FP8, numero layer e tensori condivisi. Se il forward resta sconosciuto, fermare la conversione e registrare il blocco, senza inventarne uno DeepSeek-style.
- [ ] G2 — Derivare grafo completo e allineamento token/hidden: concatenazione embedding/hidden, norme, `eh_proj`, mHC target verso stream predictor, attenzione DSA, indexer/pool, MoE e norma/proiezione output.
- [ ] G3 — Sidecar separato per il layer 45, versionato e compatibile con target/tokenizer/revisione; conservare le scale necessarie. Confronto di correttezza con dequantizzazione originale; eventuale compressione runtime separatamente qualificata. Nessuna modifica alla visione o al SUC1 target.
- [ ] G4 — Budget e misura di RAM/Metal con KV target, KDA ricorrente, cache DSA e predictor; niente aumento dei limiti o SSD implicito se i 14,87 GB BF16 non entrano con il contesto richiesto.
- [ ] G5 — Esporre hidden target e implementare predictor a una proposta secondo G2. Confrontare intermedi e logits con la reference, rendendo esplicite le differenze introdotte dalle proiezioni SUC condivise.
- [ ] G6 — Transazioni per tutti gli stati KDA/conv target, MLA/DSA/indexer/pool, compressione, posizioni/logits e predictor. Verificare il riuso indexer fra iterazioni: solo dati del prefisso accettato possono sopravvivere al rollback.
- [ ] G7 — Rifiuto totale/parziale, accettazione completa, confini dei pool, EOS/stop, contesto pieno, cancellazione e successivo turno; equivalenza dello stato con replay seriale senza MTP.
- [ ] G8 — Greedy identico, verificatore target qualificato e sampling con correzione probabilistica; nessuna promozione se il controllo batch cambia la traiettoria del target oltre i criteri dichiarati.
- [ ] G9 — KL su token/posizioni/reference dequantizzata originali; hard suite completa e regressioni chat/tool/thinking/cancellazione.
- [ ] G10 — A/B 2k/8k/32k compatibili con memoria, profondità 1 iniziale e poi 2/4 solo se previste dal grafo; accettazione, tempi draft/verifica/rollback, decode end-to-end e memoria. Warmup e tre coppie alternate, senza altri carichi GPU.
- [ ] G11 — Integrazione nelle stesse CLI/API/helper/app, controlli sidecar e fallback, suite Universal e pacchetti soltanto dopo i gate comuni.

## Gate specifici

L'incertezza del forward G1–G2 è il primo rischio, prima ancora della velocità. Il target GLM ha stati ricorrenti KDA: il rollback non è soltanto gestione della KV, anche se il layer predictor usa DSA. Un eventuale riferimento che ignora il layer 45 non può validare i logits del predictor.

Prima consegna: G1–G4, con fonte del grafo, inventario preciso e budget. Nessuno speedup atteso viene attribuito a GLM prima della misura reale M1 Ultra.
