# Correzioni successive a SUC Studio 1.4.7.04

MTP rimane accantonato. Nessuna ripresa del worker remoto.

- [x] Diagnosi: Android conserva un ID non più presente anche dopo Verifica; il service C++ usa chunk fissi 64 (GLM 512).
- [x] Android UI: scegliere il modello corrente se unico o sostituire un ID assente; conservare una scelta ancora valida fra più modelli. Verificare UI e generare APK.
- [x] Load implementato e test CPU: misurare baseline e residenza delle pagine del file in memoria unificata; mostrare bytes/percentuale insieme alla fase. Le allocazioni Metal non provano da sole la residenza; compilazione e warm-up restano fasi distinte.
- [x] Prefill implementato e test scheduler/cache CPU: studiare DS4 e implementare chunk adattivi separando capacità dei buffer e dimensione effettiva delle chiamate. Rispettare cache, richieste esplicite, modalità exact, streaming expert e limiti memoria.
- [ ] Qualifica: test selezione chunk e checkpoint, confronto numerico/risposte su input identici e misura separata di cold history e turni con cache. Nessuna promessa di equivalenza dal solo ritorno a chunk 64.
- [x] Build macOS Universal 1.4.7.05 e Android 0.1.4, collaudi funzionali, documentazione e pacchetti.
- [ ] Qualifica completa hard 120/KL per promuovere Automatico a default: separata e ancora pendente.

DS4 locale: `ds4_prefill_cap_for_prompt` sceglie il limite dal prompt, opzione esplicita ed environment; i default Metal di quella reference non sono trasferibili automaticamente a SUC. I benchmark storici Qwen SUC chunk 64/1024 hanno mostrato 100/120 contro 98/120: il nuovo percorso deve dichiarare il profilo e superare verifiche dedicate.

## Stato della qualifica

Candidate macOS 1.4.7.05 build 32, Android 0.1.4 build 5. Il prefill Automatico è una modalità distinta nei Settings; Batch prudente rimane il valore salvato preesistente e il default fino al confronto sul modello reale. Automatico prealloca 1024 righe e invia blocchi da 1024, poi al più 64; i checkpoint rimangono allineati a 64. Token, exact, chunk espliciti e streaming expert non vengono promossi al percorso adattivo. Il decode continua a elaborare un token alla volta.

Il monitor usa una seconda mappatura read-only del file e i contatori VM macOS, senza leggere i pesi o forzare il page-in. Conta pagine residenti (anche già in cache prima del load), registra baseline/residenza/footprint e aggiorna con intervallo adattivo 1–5 secondi. Non misura la percentuale di compilazione GPU né afferma che tutto il file debba essere residente prima di rispondere. Sono invariati memory policy e scelta residente/SSD.

Mac Studio: helper SUC Studio PID 29770 ancora caricato all'inizio dei lavori. Richiesta chiusura completa per evitare due istanze del Qwen nei confronti. Nessun modello altrui terminato; confronto reale, KL e hard suite nuovi ancora pendenti.

Correzione durante la revisione: blocchi grandi solo nel primo prefill dopo reset, coda e suffissi con cache sempre al massimo 64; checkpoint mantenuto vicino alla fine a multipli di 64. Evita di ampliare inutilmente il replay dei turni caldi a circa 1024 token. Test CPU aggiuntivo verifica anche un'estensione in cache di oltre 1024 token senza nuovi blocchi grandi. La cache è esatta rispetto allo stato salvato; equivalenza con un ricalcolo completo a blocchi diversi rimane da qualificare.

## Primo confronto reale completato

Qwen Flash side8 locale, stesso helper impacchettato, profili off/auto in processi sequenziali, prompt sintetici identici, max 16 token, nessuna inferenza concorrente. Prefill 2.032 token: 20,920→12,751 s; 8.032 token: 68,064→28,839 s. A 8k i turni in cache restano 0,93–1,00 s, 96/111 token nuovi. Tutte e sei le brevi risposte sintetiche coincidono. Non sono misure KL, punteggi hard o medie su più ripetizioni. I primi cold possono comprendere compilazione delle pipeline.

La scansione mincore dell'intero file da 96,3 GB costa 1,8106 s sul M1. VM_REGION_EXTENDED_INFO restituisce lo stesso conteggio (2.842.763 pagine nella prova) in 0,1818 s. Il monitor passa a questo contatore e attende 1–5 s fra osservazioni, in funzione del costo della misura; riporta anche query_s. Occorre ricontrollare il binario definitivo dopo questa ottimizzazione.

## Chiusura della candidate

Monitor VM leggero verificato su Intel/M1 e nel binario finale. Confronto sintetico finale: 8.032 token 67,673→28,834 s, due nuove domande con cache intorno a 1 s; tutte le risposte sintetiche coincidono. Controllo hard fisso a sei casi: 5/6 per entrambi, stesso answer estratto in 6/6, testo identico in 5/6. Il testo del caso lungo cambia pur mantenendo la risposta corretta. Nessuna nuova metrica KL o score completo 120 attribuiti alla release. Batch prudente resta il default; Automatico viene scelto esplicitamente nei Settings. Tutti i processi di test terminati e il modello rilasciato sul Mac Studio.

Risultati e istruzioni: [RELEASE_1.4.7.05.md](RELEASE_1.4.7.05.md). Worker MTP e automazioni precedenti non riattivati.
