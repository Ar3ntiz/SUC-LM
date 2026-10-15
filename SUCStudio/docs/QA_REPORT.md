# SUC Studio 0.9.0 — verifiche e limiti

Data: 13 ottobre 2026. Build destinata ad Apple Silicon, macOS 15+. Ambiente di compilazione/esecuzione test: macOS 26.6.2 x86_64, Command Line Tools, Swift 6.3.3, Apple clang 21. Nessun modello reale o GPU Apple Silicon disponibile in questo ambiente.

## Eseguito

| Verifica | Esito / evidenza |
|---|---|
| Motore condiviso | Build diretta da `../cpp`; snapshot duplicato rimosso, input identificati da hash |
| Runtime C++ CPU | **1.078 controlli, zero fallimenti**: codec, SHA, JSON, tokenizer e altre primitive della suite del motore padre |
| Servizio C++ | Auth corretta/assente, Origin/Host, chiavi duplicate, limite body, profondità JSON, epoch del listener, nomi modello, limite contesto e cancellazione in coda |
| HTTP/SSE | Renderer effettivo del server su socketpair, con grafo deterministico esclusivamente di test: output SSE, usage e `[DONE]` |
| Helper reale, senza modello | Handshake, frame spezzati, errore senza modello, caricamento file inesistente, recupero successivo, chiusura su EOF |
| Archivio Swift | Scrittura atomica, bozza Unicode, risposta interrotta, backup, corruzione preservata, export/import e formule 0/1/N token |
| Sessioni Swift | Cambio sessione durante delta, bozza indipendente, metriche al messaggio originario, duplicazione con nuovi ID, eventi idle fuori ordine |
| Build app/helper | Compilazione e link arm64 riusciti; codice SwiftUI/AppKit e C++/Metal reali, senza mock nel prodotto |
| Bundle | Info.plist valido, Mach-O arm64, firma ad-hoc verificata, dipendenze soltanto di sistema/Swift |
| Packaging | Installer locale `.pkg` generato; app in `/Applications`, relocazione disabilitata, nessun daemon/script privilegiato |
| UI statica nativa | Render NSHostingView verificato: Chat 1180×780, minimo 820×560, Server, Settings scuri. Screenshot allegati con **contenuto sintetico di test**, non inferenza reale |
| Script | Sintassi Bash verificata; build e test usano percorsi relativi a SUCStudio e al motore padre |

I log dei test sono in `TEST_RESULTS.txt`; input del motore in `dist/ENGINE_BUILD_MANIFEST.json`; artefatti e checksum in `dist/`. Il backend dei test di routing HTTP è esplicitamente finto per verificare trasporto e scheduler: non costituisce prova della numerica GPU. Il prodotto non include quel grafo di test.


## Verifica hardware riferita dall’utente

L’utente ha verificato **tutte le funzionalità di Settings e Chat su Apple Silicon M3**, senza problemi, prima della modifica di build 0.4. Successivamente ha testato anche il server e ha riferito un decode più veloce. Questa evidenza aggiorna le limitazioni dei report precedenti; non implica una prova già eseguita delle nuove funzioni 0.5 o confronti numerici/performance sistematici su ogni famiglia.

## Verifiche specialistiche e Server ancora aperti

- Confronto numerico app/CLI per ogni famiglia e sampling; regressione hardware della nuova build 0.5.
- Prestazioni GPU, memoria sotto carico, overhead telemetria rispetto alla baseline.
- Collegamento API da un secondo dispositivo LAN, client SDK esterno e autorizzazioni firewall/rete locale reali.
- Installazione/aggiornamento su Mac pulito senza strumenti developer; comportamento download/quarantena/Gatekeeper.
- Interazione manuale completa con Keychain, menu bar, finestre, input IME, VoiceOver, monitor scollegato, sleep/wake e disco esterno.

La compilazione arm64, i test CPU e il rendering nativo non sostituiscono queste prove. La consegna è una prima build locale installabile, non una release certificata su tutti i modelli supportati dal runtime.

## Limiti implementativi espliciti

- Prefill/decode sono misure a fine fase/richiesta, riusate dal motore. Non viene mostrato un token/s live artificiale. Il primo testo è misurato lato UI.
- Cancel durante un forward frontier opaco può attendere la fine del prefill. I kernel non sono interrotti forzatamente; chiudere l’app termina il proprio helper con timeout.
- Persistenza JSON atomica con backup; migrazione fra future versioni e storage SQLite/Core Data non implementati. Per archivi enormi i costi di riscrittura vanno misurati prima di estendere l’uso.
- Le varianti sono sessioni derivate, non rami interni con selettore. Modelli locali selezionati per percorso/bookmark; nessun catalogo avanzato o download.
- UI italiana; Markdown e highlighting basilari, nessun LaTeX/HTML attivo o contenuto remoto automatico.
- API IPv4 HTTP con key, senza TLS/CORS e senza keep-alive. Tools, embeddings, Responses e multimodale esclusi.
- Firma ad-hoc tecnica, nessun Developer ID/notarizzazione; aggiornamento manuale.

## Prova finale sul Mac Apple Silicon

1. Installare il PKG, aprire Settings, selezionare modello/tokenizer e caricare.
2. Scambiare più turni; verificare token/s a fine fase, stop e una seconda risposta.
3. Creare una seconda sessione durante streaming; controllare isolamento, bozza e recupero al riavvio.
4. Avviare API LAN; verificare 401 senza chiave e risposta valida con chiave dal secondo dispositivo.
5. Usare chat e API insieme; fermare API e verificare che chat/modello restino disponibili.
6. Minimizzare, chiudere/ripristinare dalla menu bar, uscire e verificare assenza del helper.
7. Ripetere installazione come aggiornamento e verificare conservazione delle sessioni.

## Verifiche aggiunte 0.2.0

Policy memoria testata senza GPU: residente denso, MoE automatico/manuale, superamento limite, assenza expert, riserva eccessiva, limite non disponibile, soglie esatte e budget malformati. Migrazione settings 0.1 verificata con/senza budget e persistenza modalità 0.2. Telemetria SSD letta sul thread proprietario della generazione, a fine fase, senza accesso concorrente dai timer di stato.

Restano NON ESEGUITE su questo host x86_64: inferenza e pressione memoria su Apple Silicon, validazione della stima per tutte le famiglie/dimensioni, throughput SSD e confronto con CLI. Le schermate precedenti documentano 0.1. UI_SETTINGS_0.2.png è un render nativo della nuova pagina Settings (1180×780, tema scuro), verificato visivamente; non sostituisce una prova interattiva. La policy è una protezione euristica, non una garanzia di caricabilità.

## Verifiche aggiunte 0.3.0

Test estrattore: testo Unicode con estensione sconosciuta, UTF-16 e Windows-1252; rifiuto directory, binari, vuoti e dimensioni eccessive; DOCX e ODT ZIP/XML con prefissi namespace alternativi, paragrafi e tabelle; DOC e RTF tramite textutil; PDF reale con testo e pagina vuota; XML malformato/DTD; output subprocess limitato. Test sessioni: importazione asincrona mentre si cambia chat, invio di soli allegati, ricostruzione del contenuto, rimozione e persistenza. Campioni sintetici rappresentativi: non validano ogni variante prodotta da Word/LibreOffice.

Restano da verificare interattivamente su Apple Silicon: drag-and-drop dal Finder (compreso campo NSTextView), documenti reali complessi e inferenza con allegati. I render nativi controllano la disposizione dei controlli, non simulano il trascinamento. Estrazione e test CPU eseguiti sul Mac Intel disponibile.

Render 0.3 verificati visivamente: UI_ATTACHMENTS_0.3.png (1180×780) e UI_ATTACHMENTS_MINIMUM_0.3.png (820×560), con documenti allegati alla bozza e alla cronologia.

## Verifiche aggiunte 0.4.0

Build e suite reali compilate da ../cpp, senza runtime locale. Test degli script con compilatore fittizio: errore senza padre anche in presenza di runtime locale, riuso cache invariata, modifica del padre con timestamp più vecchio, rimozione sorgente senza oggetti residui. Gli input del motore vengono controllati prima e dopo compilazione. Nuova icona Dock/Finder disegnata tramite lo stesso SF Symbol `sparkles.rectangle.stack` della barra menu a server spento, su fondo arrotondato; il simbolo server.rack durante il servizio resta un indicatore di stato.


## Verifiche aggiunte 0.5.0

Test automatici: composizione prompt globale/skill abilitate/extra sessione, esclusione skill disabilitate, snapshot della generazione effettiva, persistenza e migrazione dei settings precedenti. MCP con fixture locali: processo stdio reale, inizializzazione, elenco paginato, chiamata ed errore strumento, chiusura connessione; risultato associato alla chat di origine anche cambiando sessione. Server HTTP localhost reale: JSON e SSE, header Authorization/MCP-Session-Id/MCP-Protocol-Version verificati dal server, errore 401 e redirect 307 rifiutati. Il test HTTP richiede il permesso di ascolto localhost (la sandbox di sviluppo lo impedisce).

Render nativi 0.5 con dati sintetici: chat 820×560 con sezioni espanse, Settings con barra compressa e prompt globale. Guida HTML generata offline da USAGE.md e inclusa nel bundle. Build arm64 e installer verificati. Nessuna modifica numerica al motore padre.

Restano da provare sul Mac dell’utente: nuove funzioni con modello reale, server MCP di terze parti, accesso interattivo al Portachiavi e apertura del browser dai Settings. Gli strumenti sono invocati manualmente e i risultati sono allegati alla chat; nessun tool calling autonomo del modello. Supporto di trasporto e limiti descritti nella guida. Le fixture non attestano la compatibilità con ogni server MCP.


## Verifiche aggiunte 0.6.0

Client remoto provato contro server HTTP localhost reale: elenco modelli, autenticazione Bearer e 401, SSE con CRLF, ricostruzione delta, usage senza velocità inventate, rimozione dei parametri non standard, annullamento e validazione endpoint. Suite Swift di archivio, routing, allegati, personalizzazione e MCP superata. Build arm64 e installer verificati. App Transport Security consente endpoint HTTP esplicitamente configurati per la LAN; usare HTTPS su reti non fidate.

Restano da verificare su hardware utente la rete tra due Mac, i permessi macOS e le peculiarità dei server terzi. Il client usa Chat Completions; non promette la compatibilità con endpoint Responses o con tutti i parametri dei diversi fornitori. I contatori richiedono usage; le velocità prefill/decode remote non sono stimate.


## Verifiche aggiunte 0.7.0

Test reali con directory temporanee: protocollo d’azione JSON esatto (Markdown escluso), traversal e symlink esterni rifiutati, scrittura/lettura, shell disabilitata, comando tramite helper con processo dedicato, annullamento, approvazione manuale rifiutata e automatica, azione associata alla sessione di origine durante cambio chat. Le regressioni archivio, allegati, prompt, MCP e client remoto passano. Il test ha rilevato e corretto il caso macOS in cui Process crea già un gruppo isolato: il runner evita setsid ridondante.

La capacità del modello di usare il protocollo richiede una prova con il modello scelto. I controlli file sono dell’harness, non una sandbox macOS resistente a modifiche concorrenti malevole del filesystem. La shell usa i permessi dell’utente. Build arm64 e installer locali; nuova UI da verificare interattivamente su M3.


## Verifiche aggiunte 0.8.0

Suite Swift completa superata. Test subagent: sessione e contesto separati, selezione del principale preservata, ereditarietà della directory/approvazioni, restituzione del risultato, rifiuto delega annidata e limite di quattro subtask. Hook reali tramite helper: variabili d’ambiente, exit non zero che blocca e disattivazione. App/helper arm64 compilati e pacchetto verificato.

L’orchestrazione è testata con eventi deterministici; la capacità dei modelli reali di pianificare/delegare tramite il protocollo JSON resta da valutare sul Mac dell’utente. Gli hook sono comandi locali espliciti, non una compatibilità completa con ogni evento o formato Claude. Non sono eseguiti al di fuori delle sessioni con directory.


## Verifiche aggiunte 0.9.0 e preservazione 0.8.1

Suite completa superata: 1.078 controlli CPU, test della policy memoria dell’utente, servizio/protocollo e suite Swift. Plugin: manifest e checksum, identifier non valido, installazione disattivata, attivazione/disattivazione e rimozione. Routine: invio alla sessione, bozza preservata, prossima data senza backlog, storico e recupero interruzioni. Build arm64 e installer verificati. Marketplace usa HTTPS/checksum ma non include un catalogo pubblico; i test deterministici validano il formato, non la disponibilità di server terzi.

Importate integralmente dalla 0.8.1 utente memory_policy.hpp, stima KV/riserva in inference_service.hpp e relativi test. Uguaglianza byte per byte verificata prima della release. Il successo su Mac Studio 128 GB con limite aumentato è una prova riferita dall’utente; non è stato replicato su questo host Intel.

## 1.0.0
Regressioni Swift, plugin/routine e test preferenze superati. Verificati persistenza, famiglie indipendenti, limiti dimensioni e assertion inattiva. Render nativo a larghezza minima con Helvetica Neue 16 e Georgia 22 ispezionato. Build arm64 e firma ad hoc verificate. Avvio al login e mantenimento attivo richiedono verifica interattiva sul Mac destinatario; non alterati dal processo di test.

## 1.1.0
Fixture Markdown superate: titoli, liste numerate/checklist, tabelle/allineamento/pipe con escape, citazioni, fence incompleto in streaming, enfasi, link espliciti e URL in chiaro. Schemi non web esclusi. Apertura browser usa NSWorkspace solo al click.
Render nativo verificato a 820 px, tema scuro e tema chiaro con Helvetica Neue 16 / Georgia 22; tabelle e blocchi codice leggibili.

## 1.2.0
Intera suite Swift superata: archivio/routing, DOCX/ODT/DOC/RTF/PDF, immagini, MCP stdio/HTTP, remoto SSE e payload vision, workspace, subagent/hook, plugin/routine, preferenze e Markdown. Verificati conversione JPEG, validazione, persistenza, blocco backend locale, edit/reload nella stessa sessione ed esclusione eventi tardivi. Corrette due regressioni del riconoscimento ImageIO (file testuali e PDF). Render pannello remoto verificato a 820 px: pannello locale assente. Build arm64 e pacchetto verificati; comprensione vision da validare con un modello remoto reale.

## 1.3.0
Test C++: 1078 controlli CPU senza errori; policy memoria, protocollo e servizio HTTP/SSE superati, incluso il nuovo campo suc_metrics. Diff del servizio limitato alla serializzazione dei tempi esistenti, nessuna modifica alle formule di memoria/KV. Test RemoteTelemetry: tempi SUC, timing compatibili, priorità misure server, stima decode etichettata e prefill indisponibile senza misura. Render nativi verificati per approvazione inline e loader globale in Chat/Server, con accesso alla chat da un’altra scheda. Test su GPU reale e precisione delle misure con server terzi restano da verificare sul Mac destinatario.
Intera suite Swift e fixture remota temporizzata superate; bundle arm64 verificato.

## 1.4.0
Intera suite Swift superata sui sorgenti finali: archivio, routing, allegati, MCP, remoto, workspace, subagent/hook, plugin/routine, preferenze, Markdown, immagini, telemetria e report agentici. Test specifici: limite 1–100/default 12 e persistenza, prompt coerente, blocco della seconda azione con limite 1, risultati operativi mantenuti ma nascosti, errore immediato, riepilogo finale e annullamento resistente a delta/finished tardivi. Render nativo a 820 px verificato, senza JSON o risultati tecnici nel transcript.

Il worktree è allineato al commit padre cc8762ee59575cf46a06773449347325b1b14ee1; conservate le modifiche locali ai documenti. Policy memoria e test 0.8.1 invariati. Il servizio API differisce solo per l’estensione di telemetria introdotta nella 1.3; nessuna modifica all’inferenza o alle formule KV/riserva.
Build finale arm64 completata; app/helper firmati ad hoc e verificati. Installer e consegna verificati con SHA-256.

## 1.4.1
Build arm64 e render nativo del registro verificati. Test AppKit con 120 sessioni: scorrimento della lista, selezione di una nuova chat, conversazione lunga, seguito del fondo dopo aggiornamento in streaming, spostamento nativo e ripresa al cambio sessione. Test della policy di pausa/ripresa, limite di 200 operazioni, stati terminali, cancellazione e routing delle azioni.
Il test sintetico del gesto wheel non viene recepito dalla finestra di prova: la pausa tramite onScrollPhaseChange richiede una prova interattiva con mouse/trackpad sul Mac destinatario. Non viene dichiarata validata dal test nativo. Motore e policy memoria identici alla 1.4.0 consegnata.

## 1.4.2
Test con PTY reale: riconoscimento TTY, richiesta/consegna input, password con stty -echo, mascheramento dell’input riservato anche quando il comando lo stampa, completamento e cancellazione. Controlli Ctrl-C (exit 130) e Ctrl-D (EOF) verificati sul nuovo helper. Artifact: copia indipendente dall’originale, metadati Codable, pubblicazione automatica dopo write_file, rifiuto di traversal, symlink esterni, nomi non validi e file oltre 100 MiB. Verificato rifiuto di run_interactive con comandi disabilitati.
Render nativo del terminale e della scheda artifact ispezionato a 1040 × 820 e 820 × 560; altezza della console adattiva per mantenere accessibile la conversazione. I test non eseguono sudo né usano credenziali reali; restano da provare interattivamente sul Mac destinatario i programmi specifici dell’utente e le finestre Salva/Finder. La console è a righe, non un emulatore VT completo. Il C++ del motore e la policy memoria sono invariati; il nuovo helper PTY appartiene all’app.

Suite Swift completa superata per la 1.4.2. Dopo la revisione del flusso di output, ripetuti i test terminale/artifact: limite input 4 KiB con errore visibile, backpressure e buffer UTF-8 da 128 KiB sotto output continuo, nessuna corruzione interna dei caratteri. Build finale arm64 completata con helper PTY incluso.

## 1.4.3

Test di presentazione: separazione dei tag thinking completi e parziali, reasoning remoto separato, fence sicure e preparazione limitata di 355.028 byte HTML in 0,000108 secondi. Test AppKit in un’app host: tre backtick, invio senza alterazioni di 288.139 byte HTML e successiva digitazione, completati in 1,99 secondi incluse attese del run loop. Il testo lungo resta integrale nell’archivio e si apre in un NSTextView dedicato.

Fixture SSE verificate per SUC Studio, DS4, llama.cpp, LM Studio e campi personalizzati: reasoning_content, usage, suc_metrics, timings, stats, prompt_progress, aggiornamenti live disattivabili e stime etichettate. Ispezionati `ds4-server` al commit 54b36ed9ba42da31b24f2d1a5feb075c2475dbb1 e LLM_Router al commit 030eef3e18d1f6d53c0b8842fe28a6c1e871b63c. Un test ASGI reale conferma che LLM_Router inoltra invariati eventi reasoning/usage/timings e le opzioni telemetria.

DS4 calcola prefill/decode tok/s ma li scrive nei log e non nell’API Chat Completions verificata: senza modificare il server l’app non può ottenere quelle misure reali. Il profilo client conserva usage/reasoning e può fornire solo una stima decode esplicitamente marcata. LM Studio e server compatibili possono variare i campi tra versioni; i profili leggono i dati quando presenti, senza trasformare il tempo di rete in prefill.

Build arm64 eseguita da zero e bundle verificato. Il manifest contiene 161 input del motore padre, commit cc8762ee59575cf46a06773449347325b1b14ee1 e hash contenuto 96ca9685c94b56176afaf40619537e3f16365d5e9dbd4088a1e503f64b1f505a. Gli script usano esplicitamente il Python di sistema e rifiutano manifest/guida vuoti, evitando successi silenziosi causati da interpreti esterni danneggiati nel PATH. Motore e policy memoria 0.8.1 invariati; il wrapper server conserva la telemetria aggiunta nella 1.3.

## 1.4.4

Suite Swift completa superata, comprese fixture HTTP reali per health check, autenticazione, streaming SSE, cancellazione e server senza telemetria. Il client usa limiti di rete superiori a dieci minuti e un probe health autonomo; il probe non interrompe né ritenta la richiesta. ModelIOLog verifica aggregazione dello streaming, tetto in memoria e redazione dei data URL. Il Registro log è spento per default: i test verificano la migrazione, lo svuotamento e l’assenza di nuove entry quando è disabilitato. Il flag raggiunge il C++, dove impedisce la costruzione degli eventi input/output per i client API esterni.

Test C++: 1.078 controlli del motore, policy memoria, servizio e protocollo superati. La policy usa `min(working set Metal, 96% RAM fisica)` e la soglia esatta è coperta dai test, compreso il limite Metal da 116.000 MiB. La riserva di caricamento è KV specifica dell’architettura più 1 GiB. Come riscontro empirico, la colonna `Memory at run time` del README del motore riporta Qwen3.8-Flash-Next residente a circa 69 GB, DeepSeek residente con 107,4 GB di file più 7,9 GB, e GLM da 119,9 GB con expert in streaming SSD. Sono misure di esecuzioni specifiche, non garanzie in presenza di altra pressione memoria.

PTY reale verificato con input, password no-echo, mascheramento, Ctrl-C/Ctrl-D, output UTF-8 limitato e cancellazione. Il pulsante shell manuale è assente; la scheda osserva direttamente lo stato interattivo, compare solo durante un `run_interactive` ancora attivo e scompare alla conclusione. Il render nativo conferma prompt e controlli inline; un secondo render verifica Registro log, flag e tail monospaziata. La barra inferiore mostra la work directory accanto alle metriche.

Build 1.4.4 arm64 eseguita da zero e bundle verificato con firma ad hoc. Il manifest contiene 161 file del motore padre, commit `cc8762ee59575cf46a06773449347325b1b14ee1`, hash contenuto `96ca9685c94b56176afaf40619537e3f16365d5e9dbd4088a1e503f64b1f505a`. L’installer locale resta non firmato come richiesto; il suo SHA-256 è scritto in `dist/SHA256SUMS.txt` dopo il packaging.

## 1.4.5

`scripts/test.sh` completo superato (28 programmi e controlli, 1.078 controlli del motore), più i test nativi di editor e scorrimento. Il nuovo `tests/WebToolsTests.swift` viene compilato ed eseguito da WorkspaceTests, quindi da `scripts/test-swift.sh` senza modificare gli script; usa solo la fixture locale `tests/web_fixture.py` su 127.0.0.1. Verifica hash SHA-256 dei cinque script ds4, codifica della query, formati `<markdown>` e `<head -100 …>` (anche con limite di 8 KiB su confine UTF-8), rifiuto di `file:`, `javascript:`, `about:`, `data:` e `ftp:`, estrazione esatta dei risultati di ricerca, consenso cliccato con nuova attesa, scroll pigro rilevato tramite lo shim `getEventListeners`, pagina `/sorry/` senza clic, annullamento con `interrupted` e WebView rimossa, `allowWeb`, conferma di avvio negata, routine senza utente, approvazione per azione rifiutata e percorso agente completo con profilo dedicato persistente, riapertura senza nuova conferma e cancellazione dei dati.

Non eseguiti in questa sessione (richiedono rete o interazione): ricerca reale su google.com con modello locale e remoto, banner cookie reale di Google, pagine reali con commenti e SPA, CAPTCHA reale e verifica visiva della finestra.

## 1.4.5_Final

Browser web nascosto verificato prima del resto con un probe dedicato sulla fixture locale (app in secondo piano, nessuna rete). Finestra visibile coperta, mai mostrata, ritirata o assente: `visibilityState` hidden, rAF fermo, timer annidati a 1 s (900 → 1.000 ms, uno 1,2–1,7 s). Pannello fuori schermo con rilevamento occlusione WebKit disattivato: visible, rAF ~60 Hz, timer 250–253 / 900–905 ms, scroll 903 ms per passo; scroll pigro, IntersectionObserver, fetch/XHR e consenso superati in tutti i casi.

`WebToolsTests` esteso: durante ogni azione nessuna finestra dell'app su alcuno schermo (CGWindowList), pannello mai key/main, app mai attivata; timer, rAF e passi di scroll misurati nella pagina nascosta; pagine IntersectionObserver e fetch/XHR; transizioni dello stato del pulsante fino a nil a fine azione, errore e annullamento; verifica Google con pagina tenuta nascosta e stato arancione; apertura dal pulsante durante l'azione e in verifica, chiusura dell'utente senza effetti sullo strumento; nessuna finestra residua. `scripts/test.sh` completo superato (28 programmi e controlli, 1.078 controlli del motore; misura nascosta in `WebToolsTests`: visible, rAF 60, `setTimeout` 250/900 ms max 251/901 ms, scroll 902 ms per passo su 5), più i test nativi di editor e scorrimento. Verifica visiva con `StudioRoot` reale e fixture locale: pulsante globo «Scorrimento · n s» durante `visit_page`, pulsante arancione in verifica (`docs/UI_WEB_BUTTON_1.4.5_Final.png`, `docs/UI_WEB_BUTTON_VERIFICA_1.4.5_Final.png`); in CGWindowList solo la finestra dell'app sullo schermo, pannello del browser fuori schermo con alpha 0, nessuna finestra residua a fine azione. `pkgbuild` accetta la versione `1.4.5_Final`; bundle 1.4.5_Final (build 21) verificato con `codesign --verify --deep --strict` e `shasum -c`.

Non eseguiti (richiedono rete o interazione): Google reale, CAPTCHA reale, cambio di Spazio o di schermi durante un'azione.
