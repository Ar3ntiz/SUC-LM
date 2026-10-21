# QA — SUC Studio Universal 1.4.6.02

## Correzioni stessa versione — build interna 25

- Generatore: tutte e sette le combinazioni Skill/Hook/MCP esportate e rilette dall’importatore reale. Testi indipendenti, esempio limitato all’editor corrente, nome skill tradotto nel percorso corretto.
- Errori selezionati impediscono la pubblicazione; componenti non selezionati non vengono validati. Provati errore tardivo di trasporto MCP, pulizia dello staging e protezione della cartella esistente.
- [Anteprima combinata](UI_1.4.6.02_PLUGIN_COMBINED.png) renderizzata e ispezionata.
- Riprodotta la risposta segnalata: premessa + envelope web_search su riga propria. JSON raw/fenced riconosciuto e nascosto a ogni prefisso dello streaming; JSON ordinario, esempi inline e codice non interpretati come azioni. Azioni multiple rifiutate.
- Test SSE + WebKit locale: routine in chat non selezionata → premessa + JSON → ricerca reale → risultato al modello → risposta finale. Verificati cronologia, non letti, selezione e bozza conservate, approvazione manuale e caricamento skill ancora funzionanti.
- Le esecuzioni già archiviate non vengono ripetute automaticamente e l’archivio personale usato per diagnosticare il problema non è stato modificato.


## Personalizzazione 1.4.6.02 — verifiche superate

- Typecheck di tutte le view e runtime Swift; test del runtime con sorgenti definitivi, inclusa cancellazione MCP stdio prima della risposta e senza attendere il timeout.
- Pacchetti portabili reali: skill, frontmatter multilinea, metadati openai.yaml, MCP stdio/HTTP, placeholder di codice/dati e onboarding.
- Catalogo progressivo, richiamo esplicito, elenco/lettura risorse; traversal e symlink esterni rifiutati.
- Contesto SessionStart/UserPromptSubmit unito a un solo system prompt, snapshot della generazione coerente e registro I/O della richiesta effettiva; matcher startup verificato.
- MCP HTTP/SSE con header dichiarati nel plugin, autenticazione, protocollo/sessione, errori e redirect rifiutati.
- Hook: hash revocato dalla modifica, matcher regex, template JSON tipizzati, decisioni deny/allow, feedback, stdin JSON e stdout/stderr distinti.
- Stop durante un hook MCP PermissionRequest: cancellazione immediata, nessuna approvazione residua; task già annullati non aprono nuove conferme.
- Timeout reale con helper e gruppo processi; hook asincrono che accoda contesto senza avviare un turno.
- Integrazione: PreToolUse riscrive una vera write_file, pubblicando solo safe.txt; PermissionRequest consente senza popup; hook MCP chiama il server stdio di prova e riceve il risultato.
- Gestione plugin: scelte dei componenti conservate dopo disattivazione/riattivazione; marketplace NOT_AVAILABLE e percorsi confinati.
- Esempi shell/Python eseguiti e output interpretati; pacchetto portable-review incluso e verificato.
- Regressioni UniversalPlatformTests native/remote-only e RemoteTests; Release146Tests end-to-end con SSE/WebKit locale, routine, risultati, non letti, JSON nascosto e approvazioni inline.
- Editor renderizzati e ispezionati: [hook](UI_1.4.6.02_HOOKS.png), [skill](UI_1.4.6.02_SKILL.png), [plugin](UI_1.4.6.02_PLUGIN.png).

Test e screenshot eseguiti in runtime x86_64; nessun servizio pubblico o segreto reale necessario. La clonazione di repository remoti non è stata collaudata contro un marketplace pubblico; parser e confini dei percorsi sono coperti da fixture locali. Restano il collaudo interattivo della nuova versione su entrambi i Mac e le differenze esplicite descritte in CUSTOMIZATION_1.4.6.02_IT.md.

Compilazione finale ottimizzata riuscita per arm64 e x86_64, con verifica della firma ad hoc e delle architetture di app/helper; motore arm64 ricompilato dal padre, content SHA256 `3009335a65540cfa9c74074c8417b6662840f0ad655433a3624f7ae6b3ee774d`.

Installer 1.4.6.02 estratto e verificato: identità separata, versione/build corretti, firma ad hoc integra, app/helper Universal e motore del padre; guida HTML, esempi e licenze inclusi. Nessuna installazione sulla macchina di sviluppo è stata effettuata.

## Base 1.4.6 conservata

## Regressioni superate

- UniversalPlatformTests: backend remoto obbligatorio su x86_64; nessun avvio del
  motore arm64; identità, dati e credenziali separati dalla versione stabile.
- RemoteTests: autenticazione, streaming, reasoning, immagini, cancellazione,
  health check e telemetria compatibile SUC, DS4, LM Studio e llama.cpp.
- Release146Tests (sorgenti definitivi): JSON tool raw/fenced (anche JSON maiuscolo),
  prefissi incompleti in streaming e thinking esplicito/implicito nascosti;
  JSON ordinario e codice conservati. Riepiloghi precedenti normalizzati inline.
- Default nuove chat 4000 token; parametri delle chat esistenti conservati.
- Routine una tantum senza replay, ricorrenza numerica in minuti/ore/giorni/mesi,
  fine mese (31 gennaio → 28 febbraio), cambio ora legale e migrazione dei vecchi intervalli.
- Test end-to-end con un server SSE locale e WebKit reale: routine in una chat
  non selezionata → JSON fenced → web_search → risultati HTTP estratti → invio
  al modello → risposta finale e riepilogo. Bozza e selezione conservate.
- Puntino non letti salvato nell’archivio; rimosso aprendo la chat senza cambiare
  la data dell’ultima attività. Le approvazioni manuali notificano la chat corretta.
- Skill: import frontmatter, catalogo delle sole descrizioni attive, corpo caricato
  tramite load_skill anche senza web/workspace, skill disattivate rifiutate e
  compatibilità dei vecchi archivi senza descrizione.
- Routine manuale: approvazione inline, stato di attesa esplicito e rifiuto senza blocchi.

Fixture usate: tests/remote_fixture.py e tests/release146_fixture.py su localhost;
nessuna ricerca web pubblica o credenziale reale è necessaria per i test.

## Build e distribuzione

Anteprima della chat renderizzata e ispezionata in x86_64: riepilogo su una sola
riga, font piccolo, box grigio tenue, nessun JSON tecnico visibile.
Snapshot: [UI_1.4.6.png](UI_1.4.6.png).

Compilazione ottimizzata riuscita per arm64 e x86_64. App e helper shell/terminale
sono Universal; il motore SUC è arm64. Firma ad hoc verificata con
codesign --verify --deep --strict; dipendenze dinamiche di sistema verificate.
Installer: dist/SUC-Studio-1.4.6-universal.pkg, per aggiornare soltanto
/Applications/SUC Studio Universal Preview.app, preservandone dati e credenziali.
Distribuzione locale senza Developer ID o notarizzazione.
Il motore è ricompilato dai sorgenti cpp correnti del progetto padre, senza
modificare kernel o memory policy. La cartella SUCStudio originale resta invariata.

## Limiti di collaudo

I test sono eseguiti in ambiente x86_64. L’esecuzione del motore locale arm64 e
l’interfaccia su un Mac Apple Silicon richiedono collaudo manuale. Le fixture non
sostituiscono prove con modelli/server reali, inclusa la loro capacità di scegliere
le skill. Le routine richiedono l’app aperta e un server disponibile; le approvazioni
manuali attendono l’utente nella chat indicata. La versione stabile resta separata.

## Consolidamento 1.4.6.04 — 18 ottobre 2026

Il collaudo utente della versione Universal è positivo su Intel, M3 e M1 Ultra. Universal è ora l’unica implementazione in SUCStudio. Il nome visibile diventa SUC Studio, mentre bundle ID, servizi Keychain, percorso dati e chiave del layout finestra mantengono la precedente identità per preservare gli archivi. Il motore viene compilato dal cpp padre aggiornato dai sorgenti locali; nessuna modifica alla memory policy. Il postinstall rimuove soltanto la vecchia app Preview con identificativo corrispondente, dopo aver verificato quella nuova; non tocca i dati utente.

Verifica finale 1.4.6.04: suite test-universal.sh completata con successo sul Mac Intel di build, incluse regressioni reali SSE/WebKit delle routine, plugin combinati, lifecycle hook, MCP HTTP e cancellazione. Test dedicati di identità e installer superati. Build app/helper arm64+x86_64 e motore arm64 superata, firma ad hoc e dipendenze verificate. Hash degli input C++ identico al progetto locale: 3009335a65540cfa9c74074c8417b6662840f0ad655433a3624f7ae6b3ee774d. Il runtime Apple Silicon di questa nuova build non è stato eseguito sul Mac Intel; il collaudo utente su M3 e M1 Ultra riguarda la Universal precedente, il cui codice funzionale resta invariato.
# Qualified 1.4.7 — build 28, 19 October 2026

Universal bundle verified on M1 Ultra: app/helpers arm64+x86_64, engine arm64, macOS 15 deployment target, ad-hoc signature and system-library dependencies. Passed Swift native/simulated remote-only, HTTP/SSE/cancellation, scheduled web, routine, plugin/skill/hook and MCP HTTP suites. Final engine: 1220 CPU checks, 52 request-parameter checks, 84 Python regressions; one strict expected failure retains the rejected half-TCQ gate. Seven real selected-model checks passed on final CLI/server/bundled engine.

All six models passed complete same-reference KL equality (8176 positions each), and all twelve 8k/32k final-logit/continuation comparisons passed. Actual bundled-engine sessions passed cache hits/misses, cancellation, reload and eight replay cycles; maximum observed replay-footprint growth 0.20 MiB, not total unified/model memory. DeepSeek/GLM service profiles were separately tested. Final engine commit `24b2ad7`; manifests record the shared C++ inputs. The standalone CLI archive targets macOS 26.0+, unlike the macOS-15 app bundle.

Installer/archive are in `dist/`, with SHA256 checksums. See [release](../../docs/optimization/RELEASE_1.4.7.md) and [final evidence](../../docs/optimization/FINAL_QUALIFICATION_1.4.7.md). No physical-machine manual GUI acceptance of this build, automatic installation or main merge is claimed. Earlier sections are historical QA records, not current release status.

## 1.4.7.02 — build 29, 19 ottobre 2026

Suite Universal completa superata su Intel e M1 Ultra: instradamento nativo ARM e simulato solo remoto, SSE/auth/cancellazione, UTF-8, EOF senza separatore finale, risposta JSON, limiti 4/16 MiB e metriche con cache. Superate anche le regressioni reali WebKit delle routine, skill progressive, plugin combinati, hook e MCP HTTP. I log TEST_RESULTS_1.4.7.02_INTEL.txt e TEST_RESULTS_1.4.7.02_ARM.txt riportano le esecuzioni.

Il vero client Swift ottimizzato, senza finestra SwiftUI, elabora lo stesso stream sintetico da 1 MB in 0,837 s contro 20,125 s della 1.4.7; a 4 MB in 3,410 s contro 66,237 s. Hash del testo identici, anche con registro I/O attivo. Sul server reale il prefill di 11692 token ha richiesto 108,91 s e il successivo turno ha riportato zero token in cache: questa attesa non è eliminata dalla correzione del lettore. Diagnosi e dati in docs/optimization/LONG_CHAT_DIAGNOSTICS.md.

App/helper Universal, motore ARM, metadati e firma ad hoc verificati; installer 1.4.7.02, build 29. Il motore incluso è identico byte per byte alla 1.4.7 installata (SHA256 fa3dc9e896261203b66bb21189a05c3e2925b4a99983d97a0879d2f42226d6e4). Modelli, template, memoria, archivi e credenziali conservati. Nessuna nuova esecuzione delle hard suite su tutti i modelli viene attribuita a questa modifica del client; il punteggio Qwen3.8 documentato è 100/120 con chunk 64. L’app in uso non è stata sostituita automaticamente; collaudo interattivo della nuova build da eseguire dopo l’installazione.

## 1.4.7.03 — build 30, 19 ottobre 2026

Suite complete Universal Intel/M1 superate, incluse auth/SSE/cancellazione, tool web reali, routine, skill/plugin/hook/MCP. Nuovo checkpoint condiviso: CPU 1.220 controlli, parametri 52, Metal 705; fixture QSA con chunk 63 e checkpoint alla posizione 126. Qwen Flash side8 reale: 950 posizioni/235,9 milioni di logits identici; Qwen27B verifica il fallback senza snapshot. Chat helper, stop/ricalcolo, API con chiave e unload reale superati; sette smoke CLI/server/helper. I binari CLI/server finali coincidono byte per byte con quelli del benchmark, l’helper ricompilato è collaudato sul modello reale.

Il tokenizer Rust HF conferma i conteggi del rendering effettivo C++; tempi interni e osservati coincidono entro pochi millisecondi. Il parser Swift di produzione ricostruisce 109,0667 tok/s da 117 token / 1,072738 s, senza contare i 2.048 token riusati. La differenza ereditata con il Jinja HF è documentata, non corretta alterando il protocollo durante questa versione.

Bundle Universal compilato e firma ad hoc/architetture verificati. Il collaudo fisico della nuova versione rimane manuale. Piano, JSON e limiti in `docs/optimization/CHAT_PREFIX_CHECKPOINTS.md`; log suite in TEST_RESULTS_1.4.7.03_INTEL.txt e TEST_RESULTS_1.4.7.03_ARM.txt. Nessuna modifica di memory policy, modelli o kernel e nessuna installazione automatica.

Pacchetti 1.4.7.03 verificati senza installazione: payload estratto, firma ad hoc, architetture Universal, motore identico a quello collaudato, guide aggiornate e hash CLI/server coerenti con il manifest del motore padre. Evidenza: `docs/optimization/PACKAGE_1.4.7.03.json`.

## 1.4.7.04 — build 31

Suite Universal su M1 superata, compresi modalità remota, regressioni dei tool e SSE. Test mirati del client reale superati su Intel e M1: avanzamento, token riusati/nuovi, rate live dai conteggi server e precedenza sulle stime. Callback C++ verificato per conteggi, throttle, isolamento thread e ripristino dopo errore.

Helper finale: fasi load, progressione IPC/SSE e metriche decode live verificate sul Qwen3-8B locale selezionato; testo greedy identico alla 1.4.7.03. Verificati fallback cache zero, annullamento durante prefill e successivo replay identico. Nessun test hard/KL completo o nuovo benchmark di throughput attribuito a questa modifica; memory policy e kernel invariati. Prova reale: `docs/optimization/PROGRESS_HELPER_1.4.7.04.json`. Collaudo visuale dell'interfaccia dopo installazione resta manuale.

Installer 1.4.7.04 estratto e verificato senza installazione: versione/build, firma, architetture, guide e motore identico al collaudo. Manifest degli input C++ coerente con i sorgenti. Evidenza: `docs/optimization/PACKAGE_1.4.7.04.json`.

## Release 1.4.8.03 — build 35

Regressioni Intel e M1 del contesto superate: budget dinamico, secondo tentativo dopo thinking-only o sintesi troncata, rollback, coda, protocollo locale e cancellazione. Verificati thinking-only come errore, routine con risultato di strumento e routine in compattazione prima dell’invio senza falso fallimento. Suite chat/skill/MCP fino a 4 MiB/web schedulato/approvazioni/coda superate.

Test SSE: 1500 frammenti con usage per token integri in 0,128 s; 3 eventi metriche, 5 delta alla UI. Test grafico Intel con 100 messaggi e 1,67 MB: 150 aggiornamenti integri in 7,48 s, intervallo UI massimo 1,60 s sotto carico sintetico. Cache di presentazione: 0,154 s senza cache e 0,00052 s con cache per lo stesso ciclo di lettura; non misura la velocità complessiva dell’app. La soglia 8000 caratteri mantiene l’altezza (1514 → 1560 pt). Ragionamenti storici richiusi e paginazione limitata evitano di renderizzare l’intero storico a ogni aggiornamento.

Diagnosi diretta LM Studio: 269 token input → primo frammento 1,775 s, ricezione 42,19 tok/s; 78.341 token input → primo frammento 453,484 s, ricezione 19,90 tok/s, confermati dai tempi server. Singola prova sintetica, senza UI, non valutazione della qualità o benchmark ripetuto. Motore SUC e memory policy invariati. Verifica degli artefatti e checksum in docs/optimization/RELEASE_1.4.8.03_QA.json del progetto padre.

### Hotfix build 36

Test identità MCP superati: nomi canonici senza distinzione di maiuscole/spazi esterni, alias senza suffisso MCP, separatori, UUID, precedenza del nome esatto, rifiuto delle ambiguità e indicazione dei server disponibili per nomi sconosciuti. Fixture stdio/HTTP conservano i limiti di schema 3 MiB e risultati 4 MiB. La catena catalogo→schema→echo da 4 MiB funziona usando `fixture` per un server configurato come `Fixture MCP`; nessun tool reale REA viene eseguito.

Su Intel e M1, la chat interattiva automatica avvia il browser a freddo, esegue web_search su localhost e riceve la risposta finale senza approvazioni; il web disabilitato resta bloccato. Approvazioni MCP e web manuali e web schedulato conservano il comportamento atteso. Versione 1.4.8.03 invariata, build 36; misure grafiche e prestazionali precedenti restano riferite alla build 35.

## 1.4.9 · build 37

Localizzazione IT/EN e prerequisiti shell. Piano e verifiche definitive: `docs/optimization/PROGRESS_1.4.9_IT.md` e `RELEASE_1.4.9_QA.json`. Nessun nuovo benchmark del motore.

Verificati: catalogo e interpolazioni IT/EN, migrazione e salvataggio lingua con store reale, prerequisiti shell, identità installer, guide IT/EN. Android: 14 test Kotlin e suite UI, codice 6 e certificato invariato. Hash del motore padre e binario helper identici alla build precedente. Nessuna nuova misura GPU/KL né collaudo fisico su Apple Silicon o Android. Le verifiche definitive di bundle e installer sono nel rapporto JSON della release.
