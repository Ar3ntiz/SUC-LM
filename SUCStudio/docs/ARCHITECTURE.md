# SUC Studio 1.4.6.04

Versione unica **SUC Studio**, Universal per Intel e Apple Silicon, macOS 15 o superiore.

- Apple Silicon: motore SUC C++/Metal locale oppure server remoto.
- Intel: solo client OpenAI compatibile; nessun porting del motore SUC.
- Sessioni/dati: `~/Library/Application Support/SUC Studio Universal Preview/`.
- Bundle ID: `local.sucstudio.universal.app`. Credenziali Keychain separate.
- Installazione: aprire `dist/SUC-Studio-1.4.6.04-universal.pkg`.
  Installa `/Applications/SUC Studio.app` come unica versione del prodotto.
- Il pacchetto usa firma ad hoc, senza Developer ID/notarizzazione.
- Non importa automaticamente dati, plugin o credenziali dalla precedente versione non Universal.
  Esportare/importare le sessioni dall’interfaccia; reinserire le API key e riconfigurare gli strumenti.

## Setup e server remoto

Aprire Server, scegliere Server remoto (obbligatorio su Intel), impostare URL base
con prefisso `/v1`, API key se necessaria e ID modello. Usare Verifica / modelli,
poi Salva server remoto. Aprire Chat e creare una sessione.
Esempi: `http://192.0.2.20:8080/v1` oppure `http://127.0.0.1:8080/v1` per un
motore esterno in esecuzione sullo stesso Mac. Modelli, vision, strumenti e metriche
dipendono dalle capacità del server. Nessun modello/tokenizer locale è richiesto su Intel.

## Build e verifica

Mantenere `SUCStudio` dentro SUC-LM accanto a `cpp`.
Il motore viene compilato dai sorgenti del padre, incluse modifiche non committate.
Non usare il motore del vecchio worktree. App e helper shell/terminale sono
compilati per arm64 e x86_64 e uniti con lipo; il motore integrato rimane arm64
ed è bloccato su Intel. Per compilare entrambe le architetture eseguire:

```sh
./scripts/build.sh
./scripts/test-universal.sh
./scripts/package.sh
```

La build seleziona SDK 26.5 se presente (oppure SDKROOT esplicito).
Un singolo installer contiene entrambe le architetture. Le build per entrambe le architetture e i test automatici non sostituiscono
il collaudo dell’interfaccia su Mac Intel e Apple Silicon.
Collaudo utente completato su Intel, M3 e M1 Ultra.

---

# Architettura implementata — SUC Studio 0.9.0

## Layout con motore condiviso

- `../cpp/`: unica sorgente del motore nel repository padre; sorgenti C++, kernel Metal e header sono compilati direttamente. Nessuna directory `runtime/` locale. `ENGINE_BUILD_MANIFEST.json` registra gli input effettivi di ogni build.
- `Engine/inference_service.hpp`: estrazione del server di riferimento, con stesso caricamento e percorsi di generazione. Il parsing richieste è condiviso da chat e HTTP. Aggiunge controlli contesto/modello, inoltro misure esistenti e punti di cancellazione nel wrapper.
- `Engine/studio_support.hpp`: framing, scheduler FIFO, stato richieste, auth, telemetria e limiti JSON.
- `Engine/studio_engine.cpp`: eseguibile figlio, caricamento singolo, comandi privati e listener opzionale.
- `Sources/`: SwiftUI/AppKit, client pipe, store osservabile sul MainActor, persistenza e Keychain.
- `scripts/`: build CLT, test, verifica bundle, installer e generazione dell’icona.

La libreria Metal e i kernel sono incorporati nel helper, che viene cercato nel bundle. Chat → pipe → helper; il server HTTP apre una socket solo quando l’utente lo avvia. Non viene usato un server loopback nascosto per la chat. Un solo modello/contesto Metal è condiviso dalla coda chat/API. Il caricamento avviene su un worker; l’app non tokenizza o carica pesi sul main thread.

## IPC v1

JSON UTF-8 per riga, massimo 4 MiB/frame; profondità JSON 64. La app invia oggetti con `command`, `id` e parametri. Il helper restituisce `version`, `sequence`, `event`, `id`, `payload`. Le richieste chat hanno UUID univoco; quelle API hanno un ID locale `api-N`, identico nell’output HTTP e negli eventi. La correlazione sessione/messaggio risiede nella app.

Comandi implementati: `hello`, `load`, `unload`, `generate`, `cancel`, `start_server`, `stop_server`, `status`, `shutdown`.

Eventi: `ready`, `loading`, `loaded`, `unloaded`, `server`, `queued`, `started`, `prefill`, `delta`, `metrics`, `finished`, `cancelled`, `error`, `idle`, `status`.

Esempio di richiesta privata:

```json
{"command":"generate","id":"UUID","body":{"model":"model.suc1","messages":[{"role":"user","content":"Ciao"}],"max_tokens":128,"temperature":0.7}}
```

Le chiavi API arrivano solo via pipe privata e vengono conservate nel Keychain lato app. Il controllo amministrativo non ha route HTTP. Stdout contiene solo eventi; stderr diagnostica locale. La app legge entrambe le pipe in asincrono, raggruppa i delta testuali a circa 30 Hz e richiede footprint del helper ogni secondo.

## Concorrenza e cancellazione

Una sola richiesta possiede il modello/tokenizer; fino a otto ticket FIFO in attesa, al massimo 32 connessioni HTTP. Un listener fermato invalida il proprio epoch: connessioni precedenti non possono avviare nuove richieste dopo Stop/riavvio. Stop API richiede cancellazione delle richieste API e lascia disponibile la chat.

Cancellazione prima/dopo prefill e durante decode; nel wrapper dense anche tra chunk/token del prefill. I forward opachi delle famiglie frontier non vengono interrotti nel mezzo: la cancellazione resta pendente fino al ritorno. Non sono stati aggiunti callback ai kernel. EOF/shutdown ferma il listener; un watchdog termina il figlio dopo cinque secondi se un’operazione opaca non restituisce. La app non termina processi per nome.

## Telemetria riusata

`prefill_s`, `decode_s`, conteggi prompt/completion sono gli stessi campi del server. Prefill viene inoltrato quando la fase termina; decode alla fine della richiesta. Le formule e le differenze con la mediana della hard suite sono documentate in `TELEMETRY_AUDIT_IT.md`. Nessuna stima dai chunk SSE. Token 0/1 → decode non disponibile.

Le statistiche per messaggio vengono salvate con modello e snapshot dei parametri. Menu bar e tab leggono lo stesso store. La memoria mostrata è il footprint del processo helper, non una somma duplicata di RAM/VRAM.

## Sessioni e persistenza

Per questa prima versione la persistenza è **JSON atomico versionato con backup**, invece del Core Data/SQLite proposto nel piano originario. Questo mantiene il formato autonomo e ispezionabile; il protocollo del motore non dipende dalla scelta di storage. Le scritture sono serializzate in un actor, raggruppate a intervalli di circa 400 ms anche durante streaming continuo. Directory privata (0700), archivio e backup 0600. Un archivio illeggibile blocca il salvataggio per preservare l’originale; il recupero backup è esplicito.

Ogni sessione conserva UUID, titolo, date, bozza, archivio/preferita, system prompt, sampling e messaggi. Modifica/rigenera crea una **sessione derivata**, anziché un albero di varianti interno allo stesso thread. Le richieste congelano il contesto all’invio e mantengono la coppia session ID/message ID anche quando l’utente cambia selezione. Il backend resetta la KV cache per ogni richiesta.

## Differenze dalla specifica progettuale

La specifica in `SUC_STUDIO_PLAN_IT.md` resta il riferimento originario; questo documento descrive il codice consegnato. Storage atomico JSON e varianti come sessioni derivate sono le scelte v0.1. Non sono implementati catalogo multi-modello con metadati completi, migrazioni fra versioni future del database, localizzazione inglese, rendering LaTeX o avanzamento per layer. La UI è italiana; Markdown include titoli, liste, tabelle semplici, blocchi codice e highlighting basilare. Modelli e tokenizer sono riferimenti a file locali con bookmark, senza copie dei pesi.

## Rete

Bearer key obbligatoria tranne `/health` minimale; confronto a tempo indipendente dai prefissi. Origin browser rifiutato, Host limitato a localhost/IPv4. API LAN solo IPv4 HTTP, senza CORS o TLS. Header duplicati rifiutati; body massimo 2 MiB, header 64 KiB; timeout IO socket 15 s. Il listener non offre keep-alive, pur rispondendo HTTP/1.1 e SSE validi: i client possono riconnettersi. Il timeout socket non limita la durata del calcolo GPU.

## Memoria 0.2.0

Il wrapper LoadedEngine legge i metadati e riusa i parser architettura originali. Prima di creare MappedFile/ExpertStream applica memory_policy.hpp: limite `min(working set Metal, 96% RAM fisica)`, stima pesi e riserva KV specifica dell’architettura più 1 GiB, modalità auto/resident/manual. La cache SSD è il percorso ExpertStream originale. Dopo costruzione del modello controlla viste/pool effettivi prima del warm-up. Le stime non sono un admission controller del sistema e possono rifiutare modelli che entrerebbero in RAM o non prevedere tutte le allocazioni temporanee. Gli errori passano attraverso il normale evento error e il modello parziale viene distrutto tramite RAII.

memory_plan è un evento del caricamento; expert_stats è emesso solo dal proprietario della generazione seriale a fine fase, con UUID della richiesta. Lo stato SSD viene azzerato a inizio richiesta senza svuotare la cache. Il timer di status non legge contatori mutabili del runtime. AppSettings.memoryMode è opzionale per migrazione compatibile con 0.1.

## Harness allegati 0.3.0

Attachments.swift estrae localmente il testo prima dell’invio. Il contenuto decide l’estrattore, senza whitelist di estensioni. DOCX/ODT: unzip di sistema legge selettivamente le parti XML in stdout, senza estrarre percorsi dall’archivio sul disco; XMLParser senza entità esterne, DTD rifiutati. DOC/RTF: textutil. Sottoprocessi lanciati direttamente con argv (nessuna shell), output limitato e timeout; temporanei privati eliminati dopo la conversione. PDFKit per PDF; decodifica testuale con controlli per il resto. Nessuna dipendenza scaricata.

ChatAttachment conserva nome e testo, non URL/bookmark. Campi Codable opzionali per mantenere compatibilità con archivi precedenti. L’import cattura UUID sessione prima del lavoro asincrono; invio disabilitato finché la lettura non termina. modelContent combina domanda e allegati in un unico messaggio user e viene riutilizzato per tutta la cronologia. Il motore applica il limite token reale senza troncamento. Il C++ non cambia salvo stringa versione dell’helper.

Dalla 0.4 gli script condividono `engine-path.sh`; la cache usa hash del contenuto e la suite CPU proviene da `../cpp/tests`. I paragrafi 0.2/0.3 descrivono le funzionalità aggiunte in quelle versioni, non un requisito di vendorizzazione attuale.


## Personalizzazione e MCP 0.5.0

PromptComposer compone un singolo messaggio system (globale, skill attive, extra sessione) prima dell’invio nativo. GenerationSnapshot conserva il testo composto. AppSettings usa campi nuovi opzionali per leggere archivi precedenti; l’API del motore continua a rispettare il prompt inviato dal proprio client.

MCPClient.swift gestisce JSON-RPC stdio con Process/pipe e Streamable HTTP con URLSession effimera, session ID e parser SSE limitato a 32 MiB, con massimo 4 MiB di testo UTF-8 complessivo per risultato. Le credenziali risiedono nel Portachiavi; nessun redirect HTTP viene seguito. Il client non espone sampling/elicitation o accesso implicito ai file. Customization.swift mantiene connessioni, stato e UUID della sessione che ha avviato lo strumento. Le chiamate del modello passano dal coordinatore delle azioni e dalle approvazioni della sessione. I risultati vengono conservati nel registro azioni e inviati come messaggi tool_result; il limite è indipendente da quello degli allegati. La UI mantiene anche le chiamate manuali. Chat locale, client remoto e server integrato accettano richieste JSON fino a 32 MiB; la pipe helper fino a 64 MiB include il comando. Non occorrono modifiche al tool calling del C++.

ExtensionViews.swift contiene editor skill/MCP e selettore strumenti. La navigazione mantiene una colonna laterale di 58 o 154 punti con preferenza persistente. generate-help.py genera HTML escapato con CSS incorporato, senza script o risorse remote, e build.sh lo include nel bundle. Il pulsante Settings apre il file HTML con il browser predefinito.


## Backend remoto 0.6.0

RemoteInference possiede una URLSession effimera per generazione, senza retry o redirect, con parser SSE limitato e aggiornamenti aggregati. Il trasporto riusa gli eventi dello store con ID richiesta: il cambio sessione non sposta i risultati. RemoteConfiguration conserva endpoint, modello e opzione parametri estesi; la chiave usa un UUID del Portachiavi. La selezione della chat è indipendente da modello locale e listener API. La chiusura del motore locale non cancella richieste remote.


## Workspace e approvazioni 0.7.0

WorkspaceAction accetta un solo oggetto suc_action completo, con nome e argomenti stringa. PromptComposer pubblica il protocollo solo per sessioni abilitate con directory. WorkspaceCoordinator riconosce le risposte completate, accoda approvazioni, conserva ActionRecord e inoltra al modello il risultato reale, catturando l’UUID della sessione. Limite di 12 azioni per turno e arresto esplicito. Le autorizzazioni di sessioni importate sono azzerate.

WorkspaceTools verifica percorsi canonicalizzati rispetto al root e limita file/output. studio-command.cpp è un piccolo helper dell’app, distinto dal motore: avvia zsh in un gruppo di processi proprio per consentire timeout e cancellazione del gruppo. Non implementa una sandbox di filesystem/rete. Nessuna modifica numerica al motore condiviso.


## Delega e hook 0.8.0

AutomationHarness collega i risultati delle sessioni figlie a continuation del principale. Delega sequenziale, quattro figli per turno e un livello; le sessioni conservano parentSessionID e SubtaskRecord. La generazione locale resta nello scheduler del singolo motore. Stop propaga cancellazioni a richieste, strumenti e approvazioni dei figli.

Gli hook sono configurazioni archiviate, eseguite tramite WorkspaceCommand con ambiente esplicito, stesso root e approvazioni della sessione. Prima della richiesta vengono eseguiti prima dell’invio al backend; a fine richiesta precedono la scelta di nuove azioni. Non richiamano ricorsivamente altri hook. Eventi ed esiti sono persistiti nel registro.


## Plugin e routine 0.9.0

PluginManifest v1 contiene contributi dichiarativi; l’installazione crea ID locali e non attiva nulla. Attivazione esplicita aggiunge skill/MCP/hook e disattivazione rimuove soltanto gli ID appartenenti al plugin. Marketplace pubblico HTTPS, manifest massimo 1 MiB, SHA-256 obbligatorio e redirect rifiutati. Nessuna installazione di binari o dipendenze.

StudioRoutine conserva pianificazione, backend, sessione e 50 esiti. Il timer dello store avvia un solo lavoro quando il backend è disponibile e lo store è libero; la data successiva viene aggiornata prima dell’invio. Invii schedulati preservano bozze e selezione. Dopo sleep non c’è replay della coda; dopo crash le esecuzioni correnti diventano interrotte. Il fix memoria 0.8.1 dell’utente resta invariato.

## Preferenze 1.0
StudioTypography applica separatamente font UI/chat persistiti; il chrome nativo resta gestito da macOS. SMAppService.mainApp gestisce il login. PowerController mantiene una assertion NoIdleSleep e la rilascia allo stop/uscita. Le istanze Store dei test con archivio personalizzato non toccano queste preferenze del sistema.

## Markdown 1.1
ChatMarkdown separa parsing dei blocchi e attributi inline dal renderer SwiftUI. Le tabelle usano una Grid con colonne comuni. MarkdownBody installa OpenURLAction con validazione HTTP/HTTPS e NSWorkspace per il browser predefinito. Nessun HTML o contenuto remoto viene eseguito o caricato dal renderer.

## Immagini e revisione 1.2
ImageIO decodifica e normalizza gli allegati nell’harness Swift; il motore C++ resta testuale. ChatAttachment aggiunge campi opzionali compatibili con archivi precedenti. ChatMessage.remoteContent produce parti text/image_url. La revisione ritira gli ID prima della cancellazione, attende il task agentico e riutilizza l’UUID della sessione; receive ignora gli eventi degli ID ritirati.

## Attività e telemetria 1.3
GlobalActivityView usa stato condiviso per loader e accesso alle approvazioni di altre chat; le schede ActionApprovalView sono nel transcript. GenerationActivity rispetta accessibilityReduceMotion. RemoteTelemetry mantiene separati conteggi, tempi server e stima client; la misura server ha priorità. L’estensione JSON suc_metrics serializza i tempi già misurati dal servizio C++, senza modificare il motore padre o il piano memoria.

## Limiti e report 1.4
AppSettings.actionLimit normalizza il limite 1–100 con default compatibile 12. Lo stesso valore alimenta il prompt strumenti e il controllo prima di eseguire un’azione. ChatMessage.kind distingue tool_call/tool_result dai messaggi visibili e da action_notice/action_report; questi ultimi sono esclusi dal contesto. I risultati degli strumenti restano invece nel contesto originale. Il riepilogo usa i record effettivi del turno, inclusi errori e cancellazioni; le routine preservano il loro prompt visibile e il draft dell’utente.

## Registro e scorrimento (1.4.1–1.4.4)
OperationJournal conserva internamente le ultime 200 operazioni, aggiornate per identificatore; gli esiti terminali non vengono sovrascritti da eventi idle tardivi. ModelIOLog alimenta il pannello visibile **Registro log** con input, reasoning, output, stato ed errori della chat. È disattivato di default. Il flag viene propagato al processo C++ con `set_model_log`: spento, evita sia la serializzazione lato Swift sia gli eventi `model_input`/`model_output` lato server. Le entry contigue vengono aggregate, ciascuna è limitata a 64 KiB e la coda a 256 KiB; i data URL immagine sono redatti. Il pannello è chiuso di default e non persiste.

## Richieste lunghe e terminale inline (1.4.4)
RemoteInference usa un limite d’inattività di 24 ore e di risorsa di sette giorni, quindi un prefill di dieci minuti non scade. Un task parallelo interroga `GET /health` ogni 15 secondi con timeout autonomo di otto secondi; il risultato aggiorna lo stato UI ma non cancella, non duplica e non ritenta la generazione. Stop invalida URLSession e cancella anche il probe.

InteractiveTerminal non espone più una shell manuale. Viene creato soltanto da un’azione `run_interactive` approvata; TerminalPanel è renderizzato nel transcript solo mentre la richiesta interattiva è attiva, con un breve filtro che evita la comparsa per processi già conclusi. `run_command` non crea terminali. Il modello attende l’exit status; input, password e output PTY restano fuori dal contesto. Il terminale usa la directory della sessione e mantiene gli stessi limiti di input/output della 1.4.2.
La chat usa le fasi di scroll SwiftUI per sospendere il seguito automatico durante l’interazione; cambi di dimensione del contenuto richiedono un posizionamento al fondo dopo il layout, solo quando il seguito è attivo. Invio, cambio sessione e Vai alla fine lo riattivano. La lista sessioni ha un’area scrollabile vincolata allo spazio disponibile e rende visibile la nuova selezione.

## Ricerca web (1.4.5)
`Sources/WebTools.swift` porta il modulo web di ds4 (commit `9139e2a`, MIT) su WebKit; specifica in `docs/WEB_TOOLS_SPEC_IT.md`. `WebTools` valida argomenti, schema URL e `ChatSession.allowWeb`, codifica la query come ds4, salva il Markdown di `visit_page` in `.suc-web/` e produce i formati `<markdown>` / `<head -100 …>`. `WebBrowser` (MainActor) possiede un pannello nascosto fuori schermo, non attivante e non throttled (da 1.4.5_Final; misura e scelta in `docs/WEB_TOOLS_SPEC_IT.md` §2bis), la finestra «Browser SUC Studio» aperta solo dal pulsante globo e lo stato `WebActivity` pubblicato in `StudioStore.webActivity`, e una `WKWebViewConfiguration` condivisa con `WKWebsiteDataStore(forIdentifier:)` (UUID in `AppSettings.webDataStoreID`) e uno shim `getEventListeners` iniettato a inizio documento; ogni chiamata crea una `WKWebView` 1365 × 900, serializzata, rimossa a fine chiamata. Gli script ds4 sono stringhe raw identiche byte per byte (test SHA-256) eseguite con `callAsyncJavaScript` nel world `.page`, timeout 20 s per valutazione e cancellazione cooperativa. `handleAgentResponse` instrada i nomi web a `StudioStore.runWebTool`, che chiede la conferma di avvio (una volta per processo, anche in modalità automatica, errore nelle routine) prima di `WebTools.execute`. Il motore C++ non cambia.

## Terminale e artifact (1.4.2)
Il nuovo helper app-owned suc-studio-terminal crea una PTY con forkpty e inoltra byte tra shell e app. È separato dal motore C++, che resta invariato. InteractiveTerminal gestisce input, output limitato, stato e attesa asincrona; ogni chat mantiene una sola PTY. L’azione run_interactive riusa autorizzazioni e hook, ma restituisce solo exit status al modello. Il processo viene interrotto alla cancellazione della sessione o all’uscita dall’app.
ChatArtifact è un campo opzionale dell’archivio per compatibilità con le chat precedenti. ArtifactFiles risolve percorsi relativi alla directory di lavoro, controlla file regolari/dimensioni e copia il contenuto in un identificatore indipendente. La UI usa esclusivamente quella copia per il salvataggio; non interpreta il testo del modello come un URL locale attendibile. I metadati esportati non includono i file.


## Estensioni 1.4.6.02

`ExtensionPackages.swift` confina e copia i pacchetti; `SkillCatalog.swift` interpreta SKILL.md. Nome/descrizione sono il catalogo iniziale, istruzioni e risorse vengono caricate su richiesta. `PluginPackages.swift` risolve pacchetti portabili/legacy, MCP, hook, onboarding e marketplace; i dati scrivibili hanno una radice separata dal codice.

`HookRuntime.swift` gestisce schema JSON, hash della definizione autorizzata, matcher, input degli eventi e risultati strutturati. Le esecuzioni shell sono separate da WorkspaceCommand: stdin JSON, stdout/stderr distinti, timeout e cancellazione del gruppo processi tramite helper. Gli handler MCP riusano connessioni attive senza ricorsione. Gli hook async accodano contesto al successivo invio, senza avviare turni. I punti di integrazione sono StudioStore, AutomationHarness e WorkspaceCoordinator.

Gli argomenti dell’harness restano stringhe: updatedInput deve rispettare questo schema e i normali controlli di percorso/strumento/approvazione restano applicati. Gli hook non possono cambiare il nome dell’azione. Una sola continuazione Stop evita cicli; le restanti azioni rispettano il limite configurato. MCP stdio annulla le richieste singolarmente senza bloccare l’attesa fino al timeout. Compatibilità e limiti: CUSTOMIZATION_1.4.6.02_IT.md.

## Engine 1.4.7

The app retains one parent C++/Metal implementation shared with CLI/server. Fast categorical sampling avoids sorting only when filters are inactive; the seed-to-text sequence can change. Live-prefix reuse owns one actual engine state per loaded model, not one KV snapshot per conversation. Different/equal/shorter token histories replay safely; cancellation/reload invalidates modified state. App/server retain automatic chunks of 64 for Qwen/DeepSeek and 512 for GLM; CLI large-prompt policy remains distinct. DeepSeek/GLM use qualified FP32 TCQ register blocking, with a historical-path override. Runtime flags, numerical limits and the evidence protocol are in [INFERENCE_1.4.7.md](../../docs/optimization/INFERENCE_1.4.7.md). Memory policy, stored weights and model formats are unchanged. MTP is deferred.

## Cache dei turni Qwen 1.4.7.03

Il contratto `cpp/include/suc/inference_engine.hpp` è condiviso dal server standalone e dall’helper dell’app. Il wrapper `LivePrefixEngine` può ripristinare un checkpoint del grafo Qwen4 quando il nuovo prompt contiene il prefisso salvato, anche se il template elimina il ragionamento della risposta precedente. Il checkpoint mantiene la posizione, gli stati ricorrenti/convoluzioni e il contesto n-gram; KV e chiavi QSA precedenti restano in sede. Un solo checkpoint per modello, copie CPU limitate a 128 MiB, fallback al prefill completo in caso di incompatibilità o mancata allocazione. Cancellazione e reload invalidano lo stato. Il primo turno lungo resta da elaborare per intero.

L’helper azzera la funzione catturata da ogni task asincrono concluso: il future in archivio non deve conservare il modello dopo un comando unload. Il collaudo confronta anche le mappature effettive prima/dopo unload. Template, pesi e memory policy restano invariati.


## Coda della chat (1.4.8)

ChatSession conserva queuedMessages separatamente da messages; gli allegati sono congelati al momento dell’invio. ChatQueue coordina l’invio FIFO sul MainActor dopo finishAgentReport, senza anticipare il messaggio nel contesto del turno corrente e senza consumare la bozza attuale. La verifica di occupazione include richieste, azioni, MCP, hook e revisione del turno. Errori, stop e uscita del motore sospendono la coda; riavvio e ripristino backup richiedono una ripresa esplicita. ChatQueuePanel permette pausa/ripresa, rimozione e ritorno in bozza. RemoteInference cancella i task senza invalidare la sessione prima che una generazione cancellata sia entrata in bytes(for:); l’invalidazione definitiva avviene alla deallocazione.


## Contesto, comandi e transcript (1.4.8.02)

ContextManagement separa activeContextMessages dallo storico completo con contextThroughID e contextSummary. Usa prima i conteggi effettivi delle generazioni compatibili, altrimenti una stima dichiarata; il limite è quello locale caricato, la metadata remota rilevata o un override per sessione. Il riassunto procede a segmenti con il backend fissato, fuori dal ciclo strumenti, e viene applicato soltanto dopo successo completo. Il vecchio contesto sopravvive a errore/cancellazione. La coda e i risultati degli strumenti attendono la sintesi; l’output è limitato allo spazio disponibile. Il comando locale /compact o la soglia del 99% fra generazioni avviano l’operazione. I conteggi delle immagini sono stimati finché non li espone il server.

ChatCommands intercetta i comandi prima dell’invio o dell’accodamento, escludendo help e report locali dal contesto. PluginCatalog carica le skill abilitate dei plugin attivi, rispettando allowImplicitInvocation nei richiami autonomi. Il protocollo load_plugin non attiva componenti disabilitati o hook non revisionati.

Il transcript usa VStack e una finestra iniziale di 20 messaggi; le pagine hanno massimo 40 righe e avanzano/arretrano di 20 messaggi, mantenendo il contenuto integrale nell’archivio. TranscriptWindow non sposta l’indice durante la lettura manuale. BoundedMessageBody conserva la stessa MarkdownBody attraversando la soglia degli 8000 caratteri, senza ridurla a 12 righe; il testo completo resta apribile. MarkdownBody mantiene il testo visibile durante la preparazione asincrona. La prova AppKit di regressione ha misurato 1514 → 1560 punti attraversando la soglia.

## 1.4.8.03 completion and rendering fixes

Compaction budgets grow with historical material (10% estimated memory target, doubled output reservation for reasoning, capped at 16k and one third of the window). A single retry may double the reservation up to 32k within conservative input/output headroom. Streaming finish reasons are retained across remote and local summary protocols; length-truncated or empty summaries cannot replace the checkpoint. DS4 compatibility no longer drops standard chat-template thinking flags when routing to other backends.

Scheduled turns reserve at least 4000 output tokens within context constraints. Thinking-only responses fail explicitly; routine completion waits for tools/compaction and requires a nonempty final answer. Context estimation in the composer runs on sampled background snapshots rather than per-token view evaluation. Stable historical presentation is cached within 16 MiB/256 entries; unchanged rows and unchanged 8k Markdown previews avoid redundant rendering. Live metrics are throttled to 5 Hz; text and final usage remain lossless. These changes remove client overhead without claiming to reduce an external backend’s attention or prefill cost.

Historical reasoning is collapsed on first display and can be expanded. Live reasoning opens automatically and does not auto-collapse when decoding finishes, preserving scroll height. Transcript pagination renders at most 40 rows with a 20-row initial/following window; manual reading keeps the page start stable, and explicit previous/next controls navigate the preserved history.

## Command jobs · 1.4.9.02

`CommandJobs` owns session-scoped, runtime-only shell jobs. `WorkspaceCommand` drains nonblocking pipes off the main thread and isolates cancellation to the helper process group. Expiring a tool wait does not terminate or restart a command. Optional total deadlines are explicit and use monotonic time. A bounded output tail and incremental cursors avoid repeated full logs.

The coordinator authorizes execution once, keeps management tools scoped to the owning session and exempts observations from the execution action limit. Hooks still run. The store integrates active jobs with FIFO queuing, Stop, normal shutdown, unread state and persisted action records. UI snapshots refresh once per second; process output does not publish a view update per byte. Restored archives cannot adopt previous process handles or automatically rerun commands.

New app configurations use 16000 context tokens. Explicit session limits and API-reported remote limits take precedence; an unreported remote window uses the visible 16000-token assumption. The engine and its memory policy are unchanged.

## Agent recovery · 1.4.9.03

`AgentContinuation.swift` adds a conservative terminal-promise heuristic and malformed-envelope feedback. Both paths share two runtime repair attempts per user turn and consume the configured action budget. Feedback goes through the normal request pipeline; corrected actions still pass permission checks, hooks, approvals and tool validation. No guessed JSON or command executes. Exhaustion marks the reply as an error and pauses queued work; counters reset on new user turns, not tool results. Valid normal finals remain final. Runtime streaming phases no longer regress from decode/reasoning on incomplete usage metrics.

Build 41: standalone tool_call wrappers are part of protocol suffixes, including streaming prefixes, so presentation excludes markers while preserving raw history and literal code examples. Optional outer-close tolerance is enabled only by the coordinator for complete replies with finish_reason stop; the scanner requires exactly one unmatched outer object, all inner containers/strings complete, followed by normal JSON/action validation. Context preflight and output budgeting share the indicator/server-anchor basis plus incoming input and extra system content. The 99% trigger excludes reserved output; output is capped to remaining room.

Build 42 removes the fixed command-jobs footer without changing process ownership/lifecycle. CommandDisclosureHeader is a single full-width button with a 36pt minimum height. ActionLogView is outside the disabled configuration subtree, so read-only inspection remains usable during work. Lazy newest-first cards bound previews to 16000 characters, preserve full recorded results for copying/opening, and refresh a running job snapshot only for expanded entries.

Build 43: compaction budgets use estimated tokens throughout, with UTF-8 conversion only for lossless chunk boundaries. Oversized prior memory is folded back through a bounded reducer; failed/stopped compaction archives pending completed tool output once. No model/engine or archive-schema change.

Build 44 separates generated reasoning tokens from resent context, retains optional server counts with legacy estimates, and adds persisted backend thinking-control overrides. Compaction reduces prior memory, asks for a bounded-length summary, bisects unusable blocks to two levels, and commits only after all material succeeds. Token/error diagnostics contain counts only.

## Job MCP · 1.4.9.04

`MCPJobs` conserva in memoria una singola chiamata attiva per chat, fino a otto nell’app, e un insieme limitato di risultati. `mcp_call` restituisce uno stato dopo l’attesa richiesta; `mcp_status`, `mcp_wait` e `mcp_cancel` richiedono un identificativo appartenente alla stessa sessione. Il coordinatore mantiene l’approvazione sull’avvio; le osservazioni dei propri job non consumano altre azioni. Una risposta finale prematura viene rinviata al risultato reale. Coda, Stop, invocazioni manuali, scollegamento e ripristino degli archivi usano lo stesso ciclo di vita. Stop chiude il turno anche per la consegna dei risultati: un vecchio job interrotto non viene reinserito in un turno successivo.

Il client distingue la scadenza totale facoltativa dalla durata dell’osservazione. Le chiamate stdio senza scadenza non creano timer; HTTP usa una durata di trasporto praticamente illimitata, con scadenza totale esplicita gestita separatamente dalla ricezione dei byte. Connessione e catalogo mantengono il termine di 30 secondi. Annullamento e scadenza inviano `notifications/cancelled` best effort; nessun retry o riavvio del provider. L’esito può restare incerto se il server ha già compiuto effetti. I job non sono persistiti né riadottati dopo il riavvio; i record operativi restano archiviati. I risultati, inclusi quelli di errore, mantengono il limite di 4 MiB.

Il bundle dichiara `NSAppleEventsUsageDescription` con traduzioni IT/EN ed è firmato con `com.apple.security.automation.apple-events`. L’entitlement è applicato soltanto all’app; i binari del motore non cambiano. Il consenso TCC e la firma dei programmi esterni restano indipendenti. Le credenziali/variabili MCP continuano a provenire dal Portachiavi e non vengono incluse negli export.

## Aggiornamenti tra operazioni · 1.4.9.05

`forwardQueuedAtBoundary` intercetta il primo aggiornamento FIFO a generazione conclusa, prima della scelta dell’azione e nuovamente dopo un’eventuale attesa di approvazione. Una proposta superata ha `kind=superseded_action`: il testo originale resta archiviato, il contesto indica esplicitamente che non è stata eseguita e la UI non mostra un comando come già avviato. Nessuna modifica a richieste o KV del decode attivo.

Il percorso di `sendMessage` per i risultati operativi aggiunge il primo aggiornamento pendente come messaggio utente distinto, dopo il risultato reale. La rimozione dalla coda avviene insieme all’archiviazione della nuova richiesta, dopo le verifiche di allegati, finestra e dimensione del payload. La compattazione trattiene separatamente risultato e aggiornamento; contesto insufficiente senza compattazione e payload troppo grande conservano il risultato già ottenuto e sospendono la coda. Le correzioni non azzerano i contatori del ciclo; invocano UserPromptSubmit sul proprio testo e mantengono le skill esplicite precedenti. Le normali richieste dopo la fine del ciclo continuano ad avviare un nuovo turno. Pausa, Stop e ripristino rimangono vincolanti.

L’inserimento di un aggiornamento non sospeso risolve negativamente una proposta in attesa di approvazione. Il coordinatore verifica il nuovo messaggio prima della decisione di esecuzione: propone un nuovo passo senza interpretare il ritiro come un permesso. Se la Task è stata annullata da Stop, questa deviazione resta disabilitata.

### Build 47 · riconoscimento e limite dei cicli

Il rilevatore inline di `WorkspaceAction` è separato dal parser eseguibile: classifica wrapper di funzione non canonici come richieste da correggere, non come azioni da convertire. La proiezione UI elimina il suffisso tecnico mantenendo la prosa precedente; l’archivio resta invariato. Fences, code span, stringhe tra virgolette e blockquote sono esclusi. Prefissi streaming e marcatori completi condividono il rilevatore; il recupero resta entro due tentativi, budget azioni, permessi e Stop. Una risposta conclusa per lunghezza non viene completata.

`ActionLoopGuard` conserva solo digest di azione e riferimenti ai record del ciclo corrente, fino a 100 elementi. Prima dell’esecuzione confronta gli ultimi tre tentativi non gestionali e legge gli esiti terminali aggiornati dai job. Per shell/MCP esclude metadati volatili e considera uscita/output o risultato/errore. Non confonde uno stato running o un record assente con un esito identico. Blocca la quarta richiesta, registra `blocked`, termina il ciclo e sospende l’eventuale coda. Polling/attesa/cancellazione sono esenti. Nuovo turno, Stop, fine del ciclo ed eliminazione della chat liberano lo stato. Il controllo è conservativo: non misura progresso semantico né prova assenza di effetti.
