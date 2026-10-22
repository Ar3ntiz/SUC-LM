# SUC Studio 1.4.9.03

Versione unica **SUC Studio**, Universal per Intel e Apple Silicon, macOS 15 o superiore.

- Apple Silicon: motore SUC C++/Metal locale oppure server remoto.
- Intel: solo client OpenAI compatibile; nessun porting del motore SUC.
- Sessioni/dati: `~/Library/Application Support/SUC Studio Universal Preview/`.
- Bundle ID: `local.sucstudio.universal.app`. Identità e credenziali della versione Universal conservate.
- Installazione: aprire `dist/SUC-Studio-1.4.9.03-universal.pkg`.
  Installa `/Applications/SUC Studio.app` come unica versione del prodotto.
- Il pacchetto usa firma ad hoc, senza Developer ID/notarizzazione.
- Non importa automaticamente dati, plugin o credenziali dalla precedente versione non Universal.
  Esportare/importare le sessioni dall’interfaccia; reinserire le API key e riconfigurare gli strumenti.

## Novità della 1.4.6

Le richieste JSON dei tool e i risultati tecnici restano nel registro azioni;
la chat mostra la risposta finale e un piccolo riepilogo grigio, su una riga
quando possibile. Gli errori vengono segnalati durante l’esecuzione.

In Gestisci routine puoi scegliere la prima esecuzione e attivare **Ripeti nel tempo**,
con numero intero e unità minuti, ore, giorni o mesi. Senza il flag la routine
si disattiva prima dell’invio e viene eseguita una sola volta. Le vecchie routine
mantengono il loro intervallo. Giorni e mesi seguono il calendario locale; dopo
stop/riavvio non viene eseguita una raffica di recuperi arretrati.
L’app deve restare aperta e il server deve essere disponibile. Nelle sessioni
con approvazione automatica una routine autorizzata può avviare anche il browser
integrato; con approvazione manuale attende conferma inline nella chat indicata.
Un puntino giallo accanto al titolo segnala risposte o approvazioni non lette;
si spegne aprendo la chat e viene conservato al riavvio.

Le skill hanno titolo, descrizione e istruzioni. Il prompt iniziale elenca soltanto
le descrizioni delle skill attive, con ID; il modello carica il corpo tramite
`load_skill(id)` quando serve. Il caricamento non abilita file/comandi o altri
permessi. Import SKILL.md supporta `name`/`title`, `description` e corpo; le
vecchie skill restano utilizzabili con una descrizione generica modificabile.
Limiti: descrizione 4 KiB e istruzioni 64 KiB. I plugin possono fornire una
descrizione opzionale per ogni skill.

Le nuove chat usano Output max **4000 token**; i valori già salvati restano invariati.

## Aggiornamento alla 1.4.7.03

Chiudere le copie dell’app prima dell’installazione. Il pacchetto installa `SUC Studio.app` e rimuove la precedente `SUC Studio Universal Preview.app` solo se ha l’identificativo previsto. Le chat, impostazioni, credenziali e risorse della Universal restano disponibili: il percorso dati storico e il bundle ID sono mantenuti intenzionalmente. La cartella sorgenti unica è ora `SUCStudio/`; il motore rimane quello di `../cpp`.

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

## Guida ereditata dalla versione stabile

Le sezioni sul motore locale valgono solo su Apple Silicon.

# SUC Studio — setup e utilizzo

App locale per macOS 15+ e Apple Silicon. Modelli SUC1 e tokenizer non sono inclusi. Il pacchetto è locale, privo di firma Developer ID/notarizzazione; la firma ad-hoc serve all’integrità tecnica del bundle.

## Compilare

Installare i Command Line Tools Apple (`xcode-select --install`), con SDK macOS 15 o successivo, e Python 3 per generare il file sorgente che incorpora i kernel. Non occorrono CMake, Homebrew o dipendenze scaricate. Da questa cartella:

```sh
./scripts/build.sh
./scripts/package.sh
```

Output: `dist/SUC Studio.app` e `dist/SUC-Studio-1.4.9.03-universal.pkg`. Tutti i percorsi vengono calcolati rispetto agli script: la cartella deve restare all’interno di SUC-LM, accanto a `cpp/`. Per ricompilare da zero: `CLEAN=1 ./scripts/build.sh`. La app finale non richiede Python, Xcode, CLT o il repository.

## Installare e iniziare

Aprire il `.pkg` locale e seguire Installer; installa in `/Applications`. In alternativa copiare `SUC Studio.app` in Applicazioni. Il pacchetto non installa servizi privilegiati o daemon. Una copia ricevuta come download può essere bloccata dalle normali verifiche macOS: questa release non è notarizzata. Non disabilitare Gatekeeper o SIP.

1. Aprire Settings, scegliere il file `.suc1` e il relativo `tokenizer.json` locale.
2. Impostare contesto e gestione memoria (Automatico per default). Premere **Carica modello**. Un modello solo viene mantenuto in memoria; il primo caricamento compila le pipeline Metal.
3. Aprire Chat e scrivere. Invio invia, Shift+Invio va a capo; Interrompi annulla la richiesta. Durante prefill opaco l’interruzione viene applicata al ritorno della chiamata GPU, non istantaneamente.
4. La chat funziona senza aprire una porta TCP. Chiudere la finestra la lascia nella menu bar; minimizzazione macOS standard. **Esci** chiude anche il motore.

## Sessioni

Nuova chat (`⌘N`), selezione dalla sidebar, ricerca su titoli/testo, rinomina, preferite, archivio/ripristino, duplicazione ed eliminazione dal menu contestuale. Le bozze e i messaggi vengono salvati automaticamente. Cambiare sessione non ferma una risposta: rimane associata alla sessione di origine.

Modifica e rigenera mantengono la stessa sessione, interrompono la richiesta precedente e sostituiscono la parte di conversazione dal messaggio scelto. I controlli sessione permettono system prompt, temperatura, output massimo e thinking. Una sessione riaperta usa il modello attualmente caricato per la nuova risposta; il modello usato è registrato su ogni risposta. Nessuna cache KV viene condivisa tra conversazioni.

Il menu della chat esporta Markdown o JSON e importa JSON SUC Studio. I dati si trovano in `~/Library/Application Support/SUC Studio Universal Preview/sessions.json`, con backup. Dopo crash le risposte parziali risultano interrotte; non vengono reinviate automaticamente. Un contesto troppo grande produce un errore, senza troncamento silenzioso.

## Server LAN

Caricare il modello, aprire Server, scegliere Questo Mac o Rete locale, porta e **Avvia server**. API key obbligatoria, generata e conservata nel Keychain. Copiarla dal pulsante dedicato. Per ruotarla arrestare il server; la vecchia chiave non sarà più accettata al riavvio.

Dalla LAN usare l’indirizzo IPv4 del Mac al posto di `0.0.0.0`, ad esempio `http://192.0.2.10:8080/v1`. HTTP non è cifrato: usare una rete fidata. Il sistema può richiedere accesso alla rete locale/firewall. Non viene configurato alcun inoltro porte Internet.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.suc1","messages":[{"role":"user","content":"Ciao"}],"max_tokens":128,"stream":true}'
```

Usare il nome modello mostrato dalla app. Disponibili `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`. Non disponibili tools, embeddings, Responses, audio o immagini. API e chat condividono modello e coda (massimo otto richieste in attesa); nessuna seconda copia dei pesi. Stop server chiude il listener e richiede cancellazione delle richieste API, conservando la chat.

## Statistiche

Prefill = token prompt / durata già misurata nel runtime; disponibile a fine prefill. Decode = passi successivi al primo token / durata decode; disponibile a fine richiesta. Non è la mediana riportata da alcuni benchmark della hard suite. Nessuna velocità viene dedotta dai chunk di testo. Durante l’attesa sono mostrati stato e tempo; `—` indica misura non disponibile. Il tempo al primo testo include il trasporto verso la UI.

## Problemi comuni

- Modello/tokenizer spostato o disco scollegato: riselezionare i file.
- Porta occupata: cambiare porta; l’app non termina altri processi.
- Memoria insufficiente: ridurre contesto/modello; Automatico abilita lo streaming solo per expert instradati. Un budget manuale troppo basso può fallire il requisito minimo top-k per layer del runtime.
- Archivio illeggibile: i dati originali vengono preservati; usare Ripristina backup in Settings dopo averne fatto una copia.
- Motore terminato: ricaricare il modello; le risposte interrotte non vengono riprese automaticamente.

Vedere `docs/QA_REPORT.md` per verifiche eseguite e limiti della release.

## Novità 0.2.0: memoria e SSD

**Automatico** usa il percorso residente se la stima rientra nel limite prudenziale; altrimenti abilita il percorso SSD già esistente per gli expert instradati. La cache riceve il budget restante dopo pesi non instradati e riserva. Il minimo necessario per gli slot/top-k di ogni layer viene verificato dal runtime: un budget insufficiente produce un errore di caricamento, non un ripiego sullo swap.

**Residente** richiede che la stima rientri nel limite. **Streaming expert · budget manuale** accetta un valore positivo come `8G` o `1.5G` (K/M/G in base 1024), sempre entro il limite stimato. Il budget riguarda soltanto la cache expert, non tutta la memoria del processo.

Dalla 1.4.4 il limite è `min(working set Metal reale, 96% della RAM fisica)`: il 96% riguarda l’uso complessivo previsto, sistema operativo compreso, mentre Metal resta un limite rigido per le allocazioni GPU. La riserva comprende la KV stimata per architettura più 1 GiB di buffer; non aggiunge più una percentuale del file, perché avrebbe contato due volte i pesi mappati. Per GLM si stimano i latenti MLA, non una cache per ogni testa. La policy resta una stima preventiva: l’app non modifica il limite Metal e non garantisce assenza di OOM in presenza di altra pressione memoria.

La sezione Settings mostra l’ultimo piano accettato (anche se il caricamento successivo fallisce). Le statistiche SSD sono snapshot della richiesta corrente/ultima, aggiornati a fine prefill e fine generazione, anche nel menu della barra: byte letti, cache hit e attesa I/O. A fine richiesta sono cumulative dal suo inizio, non solo del decode; la cache dei pesi rimane riutilizzabile tra richieste. In caso di errore/cancellazione anticipata l’ultimo snapshot può fermarsi al prefill.

**Modelli senza expert instradati:** funzionano in modalità residente se appartenenti alle architetture già supportate dal motore (per esempio Qwen3 denso) e compatibili con la stima memoria. Se troppo grandi, il caricamento viene rifiutato. Il motore non implementa streaming SSD dei layer densi: servirebbero caricamento/evizione dei pesi per layer, aggiornamento dei binding Metal e sincronizzazione dei command buffer. `mmap` e swap macOS non realizzano quel percorso. La nuova versione non amplia l’elenco di architetture o formati supportati.

Nelle versioni 0.1–0.3 il codice vendorizzato in `runtime/` era invariato; dalla 0.4 quella copia è rimossa e la build usa `../cpp`. Cambiano SwiftUI e il wrapper d’integrazione `Engine/`, scritto anch’esso in C++. Gli archivi 0.1 restano leggibili: budget preesistente → manuale; budget vuoto → automatico. Nessuna sessione viene cancellata.

## Dove trovare tokenizer.json

Usa preferibilmente il file della cartella originale impiegata per comprimere il modello, oppure lo stesso snapshot del repository Hugging Face. Il tokenizer deve corrispondere esattamente al SUC1; un tokenizer di un'altra variante può avere vocabolario/token speciali diversi anche se il nome della famiglia coincide.

Link verificati alle pagine ufficiali (apri **Download file**, non salvare la pagina HTML):

| Modello originale | File |
| --- | --- |
| Qwen3-0.6B | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json) |
| Qwen3-8B | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer.json) |
| Qwen3-30B-A3B | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/tokenizer.json) |

Per esempio, scaricare solo il tokenizer Qwen3-8B in una cartella dedicata:

```sh
mkdir -p ~/Models/Qwen3-8B
curl --fail --location 'https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json' --output ~/Models/Qwen3-8B/tokenizer.json
```

`main` può cambiare: quando conosci la revisione usata in compressione, sostituiscilo con il suo commit. Se il modello è già stato scaricato con Hugging Face, cerca in `~/.cache/huggingface/hub/models--ORGANIZZAZIONE--MODELLO/snapshots/REVISIONE/tokenizer.json`; il percorso può cambiare con HF_HOME/HF_HUB_CACHE o download in directory personalizzate. Conserva una copia effettiva, non un collegamento a una cache che potresti cancellare.

Per Qwen3.8-Flash-Next, DeepSeek-V4-Flash e GLM-5.3-Flash usa il tokenizer della sorgente/snapshot impiegata nella compressione; in questa sessione non è stato possibile verificare le relative pagine pubbliche, quindi non vengono presentati link come confermati. Il progetto non contiene copie di questi tokenizer. Il tokenizer_id del SUC1, visibile con l’inspector del progetto, può identificare repository e SHA-256; in caso di hash errato recupera lo snapshot corretto senza aggirare la verifica.

Nell’app seleziona il file in **Settings → Tokenizer JSON**. Se metti `tokenizer.json` accanto al `.suc1`, viene preselezionato alla scelta del modello. L’app non scarica tokenizer automaticamente e non converte `tokenizer.model`/GGUF in JSON.

## Novità 0.3.0: trascina documenti nella chat

Trascina uno o più file nella conversazione o direttamente nel campo di scrittura, oppure usa la graffetta. L’harness dell’app riconosce il contenuto e lo legge in background, senza chiedere conversioni o estrazioni manuali. Gli allegati compaiono sopra il campo di scrittura: clic sul nome per vedere esattamente il testo estratto, × per rimuoverli. Scrivi la domanda e premi Invio; puoi anche inviare soltanto gli allegati.

- **Testo con qualsiasi estensione, anche senza estensione:** sorgenti, configurazioni, log, Markdown, CSV, JSON, XML, HTML e altri file testuali. Si provano UTF-8, UTF-16/32 con BOM e Windows-1252. L’estensione non è una whitelist.
- **Word moderno (.docx e contenitori equivalenti):** lettura automatica dell’XML del documento, tabelle testuali, intestazioni/piè di pagina e note. I prefissi XML possono variare. Non vengono eseguite macro né caricati oggetti incorporati.
- **OpenDocument (.odt, .ott e contenitori equivalenti):** lettura dell’XML content.xml, paragrafi, titoli e testo nelle tabelle. L’estrazione privilegia il testo, non la riproduzione dell’impaginazione.
- **Word binario (.doc) e RTF:** conversione trasparente tramite il textutil incluso in macOS, senza richiedere Word/LibreOffice.
- **PDF testuali:** estrazione locale tramite PDFKit, con numeri di pagina. Pagine senza testo vengono segnalate; immagini/scansioni non ricevono OCR.

Il riconoscimento avviene dai byte del contenuto: un DOCX o PDF rinominato con un’estensione diversa continua a essere riconosciuto. “Qualsiasi estensione” non significa decodificare qualsiasi formato binario: audio, eseguibili, archivi generici e formati proprietari senza estrattore vengono rifiutati, anche se contengono qualche stringa ASCII. Documenti cifrati/protetti o corrotti producono un errore esplicito. Per PDF/Word viene letto il testo estraibile, non grafici, formule come immagini o impaginazione visiva.

Limiti: 5 file per messaggio, 10 MiB per file originale, 256 KiB di testo estratto per file e 512 KiB complessivi per messaggio; PDF fino a 200 pagine. Il limite del contesto del modello si applica comunque a **cronologia + allegati + output**. Il testo non viene troncato silenziosamente: riduci/dividi il documento o aumenta il contesto se la memoria lo consente. Non c’è un indice RAG: il testo completo accettato entra nel messaggio utente, senza cambiare il motore C++.

Allegati pronti e relativo testo vengono salvati nelle bozze/sessioni, mantenuti nelle domande successive e inclusi negli export JSON/Markdown dei messaggi. Modifica e rigenerazione li conservano. Spostare o cancellare il file originale dopo l’importazione non li rende illeggibili: la sessione conserva il testo, non dipende dal percorso originale. Chi riceve un export riceve anche il contenuto degli allegati. Se cambi sessione durante la lettura, i file restano associati alla chat di origine. L’importazione in corso non è ancora salvata: attendi che gli allegati compaiano prima di chiudere l’app.

Tutto avviene sul Mac; l’app non carica file su servizi esterni. La nuova importazione riguarda la chat nativa: l’API OpenAI compatibile continua ad accettare messaggi testuali e non aggiunge un endpoint /files.

## Novità 0.4.0: motore unico nel progetto padre

La build richiede `SUC-LM/cpp/`: sia app sia test compilano direttamente sorgenti, header, strumenti comuni e kernel Metal di quella cartella. `SUCStudio/runtime/` è stato eliminato. Non viene usata una copia di riserva; se manca il padre, gli script terminano con un messaggio esplicito. Non è necessario committare una modifica al motore per provarla nell’app: basta ricompilare.

Gli hash degli input del motore e la toolchain determinano la validità della cache; cambiare sorgenti con timestamp meno recenti o rimuovere file non riutilizza oggetti obsoleti. Se gli input cambiano durante la build, la build viene rifiutata e va ripetuta. `dist/ENGINE_BUILD_MANIFEST.json` registra commit del repository e hash effettivi: questi ultimi descrivono anche modifiche locali non committate. Il manifest è incluso nell’app. `Engine/` resta codice specifico dell’app, non una copia dei kernel/runtime.

Dopo la build, `.app` e `.pkg` includono il motore compilato e restano distribuibili senza `cpp/`, Python o strumenti developer sul Mac destinatario. Il pacchetto include la versione del motore presente al momento della build, non aggiorna automaticamente un’app già installata quando si modificano i sorgenti.

**Verifica M3 comunicata dall’utente:** tutte le funzionalità Settings e Chat della versione precedente funzionano senza problemi. Successivamente l’utente ha provato anche il server, riferendo un decode più veloce. Non è una validazione sistematica di LAN, sampling o delle nuove funzioni 0.5.

L’icona 0.4 nel Dock/Finder usa lo stesso simbolo `sparkles.rectangle.stack` della barra menu a server spento, su fondo arrotondato. Gli asset sono inclusi; per rigenerarli: `python3 scripts/make-icon.py` (macOS con CLT), poi ricompilare l’app. Quando il server è acceso, l’icona della barra menu continua a indicarlo con `server.rack`.

## Prompt globale e skill

Nei Settings puoi scrivere il **System prompt generale**. Ogni nuova richiesta della chat nativa combina, nell’ordine, le istruzioni generali, le skill attive nell’ordine dell’elenco e il prompt extra della sessione. Le modifiche valgono anche per i successivi messaggi delle chat esistenti; non cambiano le risposte già salvate. Ogni risposta conserva uno snapshot del prompt effettivo. Le richieste ricevute dall’API continuano a usare i messaggi system inviati dal client API.

Nella sezione Skill puoi creare, modificare, eliminare o importare un file di istruzioni (anche SKILL.md), poi attivarlo con l’interruttore. Ogni skill ha titolo, descrizione (fino a 4 KiB) e istruzioni (fino a 64 KiB). All’inizio viene letto il catalogo; il corpo viene caricato con load_skill(id) quando pertinente. Si importa il testo: script e file citati nella skill non sono eseguiti o caricati automaticamente. Le istruzioni concorrono al limite di contesto del modello.

## MCP locali e remoti

In Settings → Server MCP scegli **Aggiungi MCP**:

- **Locale · stdio**: percorso assoluto dell’eseguibile, argomenti come array JSON e variabili d’ambiente come oggetto JSON. Esempio eseguibile `/opt/homebrew/bin/node`, argomenti `["/percorso/server.js"]`. Il relativo server e il suo runtime devono essere installati sul Mac; non sono inclusi nell’app.
- **Remoto · HTTP**: endpoint Streamable HTTP, per esempio `https://server.example/mcp`, ed eventuali header come `{"Authorization":"Bearer TOKEN"}`. Usa HTTPS per credenziali e dati su reti non fidate. Sono gestite risposte JSON e SSE sullo stesso endpoint HTTP.

Variabili e header vengono salvati nel Portachiavi macOS; nome, endpoint, eseguibile e argomenti nell’archivio locale. Metti i segreti nei campi variabili/header, non nell’URL o negli argomenti. I server attivi si riconnettono all’apertura dell’app. Spegnere un server ne chiude la connessione e interrompe le chiamate in corso; **Riconnetti** aggiorna anche l’elenco degli strumenti. Per modificare un server, prima disattivalo.

Nella chat premi il pulsante degli strumenti, scegli uno strumento, consulta lo schema, inserisci gli argomenti JSON e premi **Esegui e allega risultato**. La chiamata esegue davvero lo strumento, che può modificare dati secondo le proprie funzioni. Il risultato testuale viene aggiunto alla bozza della sessione di origine: puoi leggerlo, rimuoverlo o inviarlo al modello insieme a una domanda. Non occorre che il modello supporti tool calling. In questa versione la scelta dello strumento è manuale; il modello non avvia strumenti autonomamente.

Sono supportati testo, risorse testuali incorporate e structuredContent; immagini e audio non sono inviati al modello. Valgono i limiti degli allegati (5 file, 256 KiB per risultato, 512 KiB totali), 64 KiB per gli argomenti e 256 strumenti per server. Gli errori e i risultati troppo grandi sono espliciti. Nessuna chiamata viene ripetuta automaticamente: in caso di timeout/disconnessione l’esito di un’azione remota può essere incerto.

Supporto MCP: inizializzazione, elenco strumenti con paginazione e chiamata strumenti; trasporti stdio e Streamable HTTP. OAuth interattivo, SSE legacy con endpoint separati, ripresa degli stream, sottoscrizioni, sampling/elicitation, risorse e prompt MCP come funzioni autonome non sono inclusi. Usa gli header per autenticare server che accettano token già disponibili. I redirect HTTP vengono rifiutati; configura l’endpoint finale. Dopo una sessione scaduta riconnetti manualmente. Alla chiusura viene terminata la connessione locale; la sessione HTTP lato server scade secondo la politica del server.

Riferimenti: [trasporti MCP](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports), [lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle), [strumenti](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

## Navigazione e guida

La barra delle sezioni rimane sul lato sinistro. Il pulsante in alto espande o comprime le etichette lasciando visibili le icone di Chat, Server e Settings; la preferenza viene salvata. La finestra conserva il minimo di 820 × 560 punti e i normali controlli macOS.

Il pulsante **Setup e utilizzo** apre questa guida HTML nel browser predefinito, anche offline. La pagina è inclusa nel bundle; i link esterni richiedono Internet. `scripts/generate-help.py` la genera da USAGE.md durante la build.

## Chat remota — 0.6

In **Server → Motore per la chat** scegli Questo Mac oppure Server remoto. Inserisci la base API (per esempio `http://192.0.2.20:8080/v1`), l’ID del modello e l’API key. Per un’altra SUC Studio usa l’indirizzo LAN del Mac server e la chiave copiata sul server. **Verifica / modelli** controlla GET /models e propone i modelli; se il server non offre questo endpoint puoi inserire l’ID manualmente. Premi **Salva server remoto** prima di tornare in chat.

La modalità remota usa POST /chat/completions con streaming SSE e non richiede un modello locale. Prompt globale, skill, cronologia e testo degli allegati vengono inviati al server scelto. Le credenziali sono nel Portachiavi; non inserire segreti nella URL. Il servizio API locale può continuare a funzionare indipendentemente. Il backend non si cambia durante una risposta.

Per compatibilità, top_k, min_p e chat_template_kwargs sono inviati solo attivando i parametri estesi; temperature, top_p, max_tokens, stop e seed valido restano parametri standard. Il server deve supportare Chat Completions e lo streaming; nessun adattatore Responses o OAuth è incluso. Eventuali differenze del server sono riportate come errori, senza ripetere automaticamente una richiesta.

**Interrompi** chiude la richiesta remota; l’effettivo arresto del calcolo dipende dal server. Le generazioni non hanno più il precedente timeout breve: possono restare senza token durante prefill lunghi. Ogni 15 secondi l’app prova `GET /health` sulla base API e mostra se il server è raggiungibile, non supporta l’endpoint o non risponde; un health check fallito non interrompe né ripete la generazione. Rimangono solo limiti di rete di 24 ore senza dati e sette giorni complessivi, oltre alla cancellazione manuale. Il primo testo è misurato dal client e include la rete. Il limite di contesto effettivo è gestito dal server remoto.

## Directory e azioni del modello — 0.7

Apri i controlli della sessione nella chat, scegli **Directory di lavoro** e abilita le azioni. La scelta resta associata a quella sessione. Il prompt dell’harness descrive gli strumenti al modello: elenco directory, lettura, scrittura, creazione directory e comandi shell opzionali. Il modello deve saper emettere il protocollo JSON descritto nel prompt; non sono interpretati come comandi i normali messaggi o blocchi Markdown. Funziona con backend locale o remoto, senza richiedere tool calling nativo.

La modalità iniziale è **Manuali**: prima di ogni azione compare una scheda con sessione, directory, strumento e argomenti completi. Approva soltanto quella singola azione oppure rifiutala. Con **Automatiche**, gli strumenti abilitati vengono eseguiti senza dialoghi. Il registro nella sessione conserva richiesta, esito e risultato. Il risultato reale viene restituito al modello; limite predefinito di 12 azioni consecutive, configurabile nei Settings da 1 a 100, poi occorre un nuovo messaggio.

Gli strumenti file accettano percorsi relativi, controllano la directory effettiva e rifiutano symlink diretti all’esterno. Lettura fino a 256 KiB, scrittura fino a 128 KiB, elenco fino a 2.000 elementi. Nessuna cancellazione implicita. I file importati come sessione non ricevono autorizzazioni: directory e abilitazioni vanno selezionate nuovamente.

**Consenti comandi shell** è separato e disabilitato inizialmente. I comandi partono nella directory scelta con zsh e i permessi del tuo account: possono accedere ad altri percorsi e alla rete, quindi la directory non costituisce una sandbox del sistema operativo. Ogni comando ha output massimo 256 KiB e durata massima 30 secondi. L’harness termina il gruppo di processi del comando al termine o all’interruzione; non usare questo strumento per avviare daemon permanenti. Processi che si staccano autonomamente dal gruppo non sono una modalità supportata.

Interrompere non annulla modifiche già effettuate. Conserva il lavoro sotto controllo versione quando opportuno. Le azioni non vengono riprese automaticamente dopo un riavvio dell’app. Il motore C++ numerico resta invariato; l’harness Swift e il piccolo helper di esecuzione gestiscono strumenti e approvazioni.

## Subagent e hook — 0.8

Nei controlli della sessione puoi attivare **Modello capace di delega: abilita subagent**. È una dichiarazione esplicita: il modello deve seguire il protocollo strumenti JSON dell’harness. Quando richiede `delegate_task` con argomento `task`, l’app crea una sessione separata e restituisce il risultato al modello principale. Puoi aprire le sessioni dei subagent dalla cronologia o dall’elenco Subtask.

Sono consentiti quattro subtask per turno, uno alla volta, senza ulteriore delega dai subagent. Ogni subagent eredita directory, prompt della sessione, sampling e approvazioni, ma riceve solo il proprio obiettivo: non tutta la cronologia del principale. Usa lo stesso backend/modello selezionato. Il motore locale rimane unico e seriale. Interrompere il principale interrompe anche le richieste dei figli. Il risultato completo rimane nella sessione figlia; risultati molto lunghi sono abbreviati esplicitamente nel riepilogo restituito al principale. Le attività interrotte non ripartono al riavvio.

In **Settings → Automazione e hook → Gestisci hook** puoi aggiungere script zsh per `request_start`, `request_end`, `before_tool` e `after_tool`. Gli hook si applicano alle sessioni con directory di lavoro, inclusi i subagent, e seguono le approvazioni della sessione. Sono distinti dal permesso per i comandi proposti dal modello: attivare un hook configura uno script scelto da te.

Gli script ricevono le variabili `SUC_HOOK_EVENT`, `SUC_SESSION_ID`, `SUC_WORKSPACE`, `SUC_TOOL_NAME` e `SUC_TOOL_ARGUMENTS` (oggetto JSON). Non concatenare questi dati non attendibili come codice shell. Gli hook hanno timeout di 30 secondi e output massimo 256 KiB; esito e output sono nel registro azioni. Un errore prima della richiesta o dello strumento ne impedisce l’esecuzione; un errore dopo lo strumento viene registrato senza annullare ciò che è già accaduto. Un errore a fine richiesta ferma le azioni successive. Non è un importatore dei file di configurazione Claude: gli eventi supportati e il contratto sono quelli descritti qui.


## Plugin e marketplace — 0.9

Apri Settings → Plugin e routine → Gestisci plugin. Puoi installare un file `plugin.json` o una cartella che lo contiene. I plugin vengono installati **disattivati**; consulta Dettagli (provenienza e manifest completo) prima di attivarli. Attivare un plugin aggiunge skill, avvia i suoi MCP e abilita gli hook dichiarati. Disattivarlo rimuove i contributi attivi e chiude i suoi MCP; rimuoverlo elimina anche le relative credenziali. Le sessioni esistenti restano conservate. Non è un importatore universale di plugin Claude/Codex.

Il formato SUC Studio v1 richiede `formatVersion: 1`, `identifier`, `name` e `version`; `skills`, `mcpServers` e `hooks` sono array opzionali. Un esempio completo è in `examples/plugin/plugin.json`. Gli MCP usano percorsi assoluti degli eseguibili già installati o endpoint HTTP. Il plugin non installa dipendenze o binari e non copia file di supporto: gli script hook sono contenuti nel manifest. Credenziali MCP da configurare nei Settings, mai nel manifest. Limiti: 1 MiB per manifest, 64 skill/16 MCP/16 hook per plugin, 32 plugin installati.

Un marketplace è un indice JSON HTTPS con `entries`: ogni voce contiene `name`, `manifestURL` HTTPS e `sha256` del manifest esatto. Esempio in `examples/marketplace.json` (URL dimostrativo da sostituire). Carica l’indice, seleziona una voce, verifica la provenienza e installa disattivato. Download limitati, checksum obbligatorio e redirect rifiutati. HTTPS e checksum verificano trasporto/integrità, non l’affidabilità dell’autore: l’indice deve provenire da una fonte scelta da te. Non è incluso un marketplace pubblico gestito da SUC Studio.

Per cambiare versione rimuovi esplicitamente il plugin e installa il nuovo manifest. Modifiche manuali ai contributi vengono sostituite alla successiva riattivazione del plugin; conserva le personalizzazioni come skill/hook separati.

## Routine schedulabili — 0.9

Da Settings → Gestisci routine imposta nome, prompt, sessione, backend richiesto, prima data e, facoltativamente, ripetizione con numero e unità (minuti, ore, giorni o mesi). La routine usa il modello configurato nel backend selezionato, preserva la bozza manuale e conserva uno storico di 50 esecuzioni. Eredita directory, strumenti e approvazioni della sessione: in modalità manuale le azioni attendono l’utente.

Lo scheduler lavora mentre l’app è aperta, anche con finestra chiusa nella menu bar. Attende che il backend richiesto sia selezionato, il modello sia disponibile e non ci siano richieste in corso. Non carica automaticamente modelli locali. Dopo sleep o riapertura recupera al massimo un’esecuzione, poi calcola la prossima data; non esegue tutta la coda arretrata. Le esecuzioni interrotte da un riavvio sono segnate come tali, senza reinvio automatico. L’app non installa daemon, non accende il Mac e non esegue routine quando è chiusa.

## Aspetto, avvio ed energia — 1.0

Nella sezione Aspetto puoi scegliere famiglia e dimensione separatamente per interfaccia e chat. Interfaccia: 10–18 pt; chat: 12–30 pt. Titoli e metadati mantengono proporzioni leggibili; il codice resta monospaziato e segue la dimensione chat. Il composer usa lo stesso font della chat. Le preferenze sono persistenti e il pulsante di ripristino torna ai valori di sistema. La barra del titolo e i menu nativi gestiti da macOS conservano la tipografia di sistema.

La sezione **Avvio ed energia**, sopra Aspetto, permette di registrare l’app per l’avvio al login tramite il servizio macOS. Installa l’app in Applicazioni; eventuale approvazione aggiuntiva si gestisce dal pulsante Impostazioni macOS. La UI mostra lo stato restituito dal sistema, compresi errori o approvazioni pendenti. L’app non avvia il computer prima del login.

**Mantieni il computer attivo ed evita lo standby** crea un’assertion macOS contro lo stop automatico del sistema, mantenuta finché l’opzione è attiva e l’app aperta. Viene rilasciata quando disattivi l’opzione o esci. Lo schermo può spegnersi e lo stop esplicito/chiusura del portatile segue le regole macOS. Non viene modificata la configurazione energetica permanente del computer.

## Link e Markdown (1.1)
I link HTTP/HTTPS nel testo della chat si aprono con un click nel browser di sistema. Anche gli URL web scritti in chiaro diventano link. Schemi file, javascript e comandi non vengono aperti dal transcript. Il testo non carica immagini o pagine esterne automaticamente.

Il renderer supporta titoli con #, enfasi e codice inline, liste puntate/numerate e checklist, citazioni, separatori, tabelle con allineamento e blocchi di codice con copia. I blocchi di codice parziali restano leggibili durante lo streaming. HTML, formule LaTeX e l’intera specifica CommonMark non sono implementati; il contenuto rimane testuale.

## Immagini e correzioni chat (1.2)
Trascina immagini nella chat o usa la graffetta: riconoscimento dal contenuto, anche con estensioni arbitrarie. I formati decodificabili da ImageIO (fra cui PNG, JPEG, HEIC; per immagini animate o multipagina viene usato solo il primo fotogramma/pagina) vengono orientati e convertiti in JPEG senza metadati, lato massimo 2048 pixel. Anteprima e miniatura mostrano il contenuto inviato. Limiti: originale 10 MiB/40 megapixel, JPEG 2 MiB, immagini complessive per messaggio 8 MiB e massimo 5 allegati totali. Non si estraggono automaticamente immagini interne ai PDF o ai documenti Office.

Serve **Server remoto** con modello vision che accetti `content` multimodale e `image_url` data URL. Anche la cronologia conserva le immagini per i turni successivi. Il motore locale SUC è testuale: la richiesta con immagini viene bloccata esplicitamente, senza perdere la bozza. L’harness non può aggiungere capacità vision a un modello testuale. La richiesta remota complessiva è limitata a 20 MiB; oltre il limite aprire una nuova sessione. L’export JSON conserva le immagini; l’export Markdown contiene solo il riferimento descrittivo.

**Modifica** interrompe il lavoro in corso e rimette testo e allegati nel composer della stessa sessione. **Rigenera** interrompe e reinvia il messaggio utente precedente nella stessa sessione. In entrambi i casi i messaggi successivi al punto scelto vengono rimossi; le azioni sui file già eseguite non sono annullate. La duplicazione esplicita resta disponibile dal menu sessione.

La sezione Server alterna i pannelli Server locale/Server remoto. Cambiare backend non arresta un server locale già avviato: rimane controllabile dalla barra menu. La gestione hook include Chiudi/Esc e quattro esempi modificabili, inizialmente disattivati. Scegliere un esempio non esegue alcun comando.

## Approvazioni, attività e metriche remote (1.3)
Le approvazioni sono schede nella chat interessata, con directory e argomenti completi. Approva o rifiuta lì: non si apre un popup. Se la richiesta riguarda un’altra sessione o sei in Server/Settings, una barra permette di aprire la chat corretta. Le approvazioni dei subagent rimangono nella loro sessione.

Durante il caricamento del modello una barra globale con loader è visibile in tutte le schede. Prefill e decode hanno animazioni distinte; con Riduci movimento macOS diventano indicatori statici. Prima del primo testo remoto l’app segnala l’attesa: un server può includere coda e prefill nello stesso intervallo.

Il server SUC Studio 1.3 aggiunge `suc_metrics.prefill_s` e `suc_metrics.decode_s` alle risposte JSON e al frame SSE usage (`stream_options.include_usage=true`). Per ottenere entrambe le velocità reali fra due Mac, aggiornare anche l’app che espone il server. La policy memoria 0.8.1 è invariata. Si leggono inoltre i campi `timings.prompt_per_second`, `predicted_per_second`, `prompt_ms`, `predicted_ms` dei server che li espongono.

Se sono disponibili solo i conteggi token, il prefill resta «—»: il tempo al primo testo comprende anche rete e coda. Il decode può essere stimato dal conteggio finale e dall’intervallo fra primo e ultimo testo ricevuto; è indicato con **≈ / stima client** ed è influenzato da rete e buffering. Una risposta in un solo blocco non consente questa stima. Non si contano i chunk SSE come token.

## Limite e report delle azioni (1.4)
Nei Settings, **Azioni del modello → Azioni consecutive**, imposta il limite per turno da 1 a 100 (predefinito 12). Si applica anche a ciascun subagent e alle routine, dalla prossima azione; non cambia le approvazioni manuali/automatiche. Un nuovo messaggio avvia un nuovo conteggio.

La chat nasconde i messaggi operativi del protocollo: JSON delle azioni e risultati tecnici inviati al modello. Il contesto del modello e il registro azioni continuano a conservarli; gli export JSON mantengono i dati completi. Una risposta ordinaria del modello, anche in JSON, resta visibile.

A fine turno compare un riepilogo con azioni completate, non completate e tipologie di lavoro. Errori, rifiuti e raggiungimento del limite vengono segnalati subito nella conversazione. I dettagli restano nel **Registro azioni** delle impostazioni della sessione. I report prodotti dall’interfaccia non vengono reinviati al modello come se fossero sue risposte. Le azioni già eseguite sul filesystem non vengono annullate da Stop, Modifica o Rigenera.

## Registro e scorrimento (1.4.1, aggiornato in 1.4.4)
In Server, sopra Registro eventi, il **Registro log** può mostrare una tail in tempo reale degli input, del reasoning e degli output elaborati dalla chat. Il flag **Abilita Registro log I/O** è spento di default: in questo stato l’app non serializza le richieste per il log e il motore C++ non produce gli eventi I/O del server. Attivalo solo per una diagnosi. Il pannello è chiuso di default, non viene salvato e conserva al massimo 256 KiB; le immagini data URL vengono sostituite da un riferimento. **Svuota** cancella la coda visualizzata.

La chat segue il fondo anche quando cambia l’impaginazione Markdown, arriva un’approvazione o compare un report. Uno scroll manuale sospende l’inseguimento: i nuovi token non ti riportano in basso. **Vai alla fine**, un nuovo invio o il cambio di sessione riprendono l’inseguimento. L’elenco laterale ha una propria area scrollabile; creando una chat, la selezione viene portata in vista.

## Terminale interattivo e artifact (1.4.2)

Seleziona una directory di lavoro e abilita le azioni della sessione. Non esiste più una shell manuale scollegata: la scheda terminale compare nella conversazione solo mentre è attivo un `run_interactive(command)` richiesto dal modello per conferme, password o altro input. I normali `run_command` e i processi già conclusi non mostrano il terminale. **Riservato** maschera il campo. Invio manda una riga, anche vuota (massimo 4 KiB); Ctrl-C interrompe, Ctrl-D invia EOF e Termina chiude il processo.

L’input non viene conservato in una cronologia dell’app. L’output, limitato agli ultimi 128 KiB, resta in memoria e non viene salvato nell’archivio o inviato al modello. Le normali richieste password usano il no-echo del terminale; il campo Riservato maschera inoltre i valori nell’output visualizzato. Non inserire credenziali nel testo della chat. Il terminale è una console a righe con PTY (100 colonne / 24 righe), non un emulatore completo per vim, top o altre UI a schermo intero.

La richiesta segue le approvazioni manuali/automatiche e attende la conclusione del processo; non ha il timeout di 30 secondi di run_command. Il modello riceve soltanto il codice di uscita. Per leggere un risultato usa read_file; per offrirlo come file usa publish_artifact. È ammesso un solo terminale inline attivo per sessione. I comandi operano con i permessi dell’utente e non costituiscono una sandbox del filesystem.

Gli artifact compaiono in fondo alla conversazione: **Salva…** esporta una copia nella posizione scelta, **Mostra nel Finder** mostra la copia conservata dall’app. write_file pubblica automaticamente il file; per file prodotti da shell o altri strumenti il modello deve chiamare `publish_artifact(path)` con un percorso relativo alla directory della sessione. Sono ammessi file regolari di qualsiasi tipo fino a 100 MiB; percorsi esterni e symlink che escono dalla directory vengono rifiutati. Ogni pubblicazione è una nuova copia persistente sotto `Application Support/SUC Studio Universal Preview/artifacts`, indipendente dalle modifiche successive all’originale.

L’export JSON/Markdown della chat non incorpora i byte degli artifact: salva separatamente i file da trasferire a un altro Mac. Le copie possono occupare spazio su disco e restano disponibili alle chat duplicate; non vengono eliminate automaticamente cancellando una sessione.

## Reasoning, editor e telemetria (1.4.3)

Le risposte dei modelli thinking mostrano il ragionamento in un box distinto, con fondo leggero e font un punto più piccolo. La risposta finale rimane sotto il box con il font scelto per la chat. Sono riconosciuti sia i tag `<think>…</think>` sia i campi streaming `reasoning_content`/`reasoning` dei server remoti. Il testo completo resta archiviato nella sessione.

Digitando tre backtick consecutivi nel composer si apre un editor dedicato in cui incollare il codice e indicare facoltativamente il linguaggio. **Inserisci codice** crea una fence Markdown sicura anche se il testo contiene già backtick. La maniglia sopra il composer permette di estenderlo verso l’alto; l’altezza minima rimane quella standard e l’altezza massima preserva lo spazio per la conversazione.

Messaggi oltre 8.000 caratteri mostrano un’anteprima leggera e un comando per aprire il testo integrale. Il parsing Markdown viene preparato fuori dal thread dell’interfaccia, con limiti al numero di blocchi; HTML grezzo è mostrato come codice e non viene interpretato. I controlli automatici di link, dati e ortografia del composer sono disattivati per evitare scansioni costose durante l’incolla. Il contenuto inviato e persistito non viene troncato.

In **Server → Profilo telemetria** scegli Automatico, SUC Studio, DS4, LM Studio, llama.cpp, LLM Router oppure campi JSON personalizzati. Le misure fornite dal server hanno priorità; le stime sono sempre marcate con `≈`. llama.cpp riceve `timings_per_token` e `return_progress` quando l’aggiornamento live è attivo. DS4 espone reasoning e conteggi usage ma, nella versione verificata, lascia i tok/s soltanto nei log del server: l’app può stimare il decode, mentre non inventa il prefill. LM Studio usa i campi `stats` quando presenti. LLM Router conserva la telemetria del backend; per inoltrare le estensioni llama.cpp seleziona il profilo llama.cpp.

Il profilo personalizzato accetta percorsi JSON puntati, per esempio `metrics.prefill_tps` e `metrics.decode_tps`, con profondità massima controllata. **Aggiorna le metriche durante lo streaming** può essere disattivato; **Stima decode live** usa una stima approssimativa di quattro caratteri per token quando il server non pubblica una velocità. Tempi di rete, batching e buffering possono alterarla. Senza una misura server il prefill resta `—`.

## Richieste lunghe e controlli chat (1.4.4)

La riga sotto il composer mostra `work directory: percorso` accanto a prefill e decode quando la sessione ha una directory di lavoro. Il percorso completo compare anche come suggerimento al passaggio del mouse. Nelle impostazioni della sessione, **Sampling avanzato** include una legenda per Top-p, Top-k, Min-p, seed e stop. I filtri si combinano: per valutare l’effetto conviene modificarne uno alla volta.

## Ricerca web (1.4.5, rivista in 1.4.5_01 e 1.4.5_Final)

In ogni sessione con **Consenti strumenti web** attivo il modello può usare due strumenti web, anche senza directory di lavoro e con **Consenti azioni del modello** spento: `web_search(query)` cerca su Google e restituisce fino a 20 link in Markdown più 1.200 caratteri del testo della pagina; `visit_page(url)` apre un indirizzo `http` o `https` noto e ne estrae titolo, contenuti e fino a 80 link. È accettato anche il nome `google_search`. Gli strumenti funzionano con ogni backend (motore locale, SUC Studio remoto, LM Studio, ds4 e altri server compatibili OpenAI) perché passano dal normale protocollo `suc_action`. Senza directory di lavoro (o con le azioni del modello spente) il prompt elenca solo questi due strumenti e invita il modello a usarli invece di dichiarare che non può navigare; le altre azioni restano ignorate. Una richiesta racchiusa in un unico blocco di codice (anche marcato `json`) viene accettata; JSON inserito in un testo no.

Le pagine si aprono nel browser integrato, con WebKit di macOS e il suo user agent standard. Il browser lavora **nascosto**: non compare sopra le tue finestre, non prende il focus e non porta l’app in primo piano. Mentre una ricerca o una pagina è in corso nella sessione selezionata, nella barra in alto della chat, accanto all’icona delle impostazioni della sessione, compare un **pulsante globo** con indicatore di attività: mostra la fase («Navigazione», «Attesa caricamento», «Consenso cookie», «Scorrimento», «Estrazione») e i secondi trascorsi; il suggerimento riporta anche «Ricerca Google: …» o «Apertura pagina: …». Facendo clic si apre la finestra **Browser SUC Studio** con la pagina dal vivo e una barra di stato (fase, URL, tempo): puoi guardare cosa succede o dove si è fermato. Chiuderla non interrompe lo strumento. A fine azione (completata, con errore o interrotta) il pulsante scompare e la finestra si chiude da sola: il contenuto estratto è già in chat o nel file salvato, e una finestra vuota non resta mai sullo schermo. Alla prima azione web dopo l’avvio dell’app compare in chat la richiesta «Lo strumento web vuole usare il browser integrato (nascosto; puoi guardarlo dal pulsante globo). Consenti?», anche con approvazioni automatiche. La risposta vale fino all’uscita dall’app. Nelle routine pianificate con approvazione automatica il browser può avviarsi senza conferma aggiuntiva; con approvazione manuale la conferma resta inline nella chat indicata, segnalata come non letta. Ogni ricerca o pagina segue poi le approvazioni manuali o automatiche della sessione, mostrando la query o l’URL.

Nelle impostazioni della sessione **Consenti strumenti web** è attivo di default; spento, gli strumenti rispondono «Strumenti web disabilitati per questa sessione.» e il prompt non li elenca. `visit_page` rifiuta `file:`, `javascript:`, `data:`, `about:` e ogni altro schema. Il browser usa un profilo dedicato e persistente, separato da Safari: cookie e consenso restano tra le chiamate e i riavvii. **Settings → Ricerca web → Cancella dati del browser web** chiude il browser e cancella cookie, consenso e cache.

Il comportamento replica il modulo web di ds4: una pagina nuova per chiamata in una vista 1365 × 900, attesa del caricamento (controlli ogni 250 ms, fino a 25 s), accettazione automatica del banner cookie di Google e, per `visit_page`, scorrimento progressivo per caricare commenti e contenuti pigri. Il Markdown completo di `visit_page` viene salvato in `.suc-web/visit-<data>-<n>.md` nella directory di lavoro; al modello arrivano il percorso relativo e le prime 100 righe (al massimo 8 KiB). Per leggere il resto usa `read_file`, che legge file fino a 256 KiB. Senza directory di lavoro la pagina va in `~/Library/Application Support/SUC Studio Universal Preview/web/visit-<data>-<n>.md`: il modello riceve il percorso assoluto e le stesse prime 100 righe, ma non può leggere il resto perché `read_file` non è disponibile; il file resta a disposizione dell’utente. Nel registro operazioni le chiamate compaiono come «Ricerca Google: …» e «Apertura pagina: …».

Se Google chiede una verifica (pagina `/sorry/` o reCAPTCHA), l’app non la aggira e non porta il browser in primo piano: la pagina resta caricata e il pulsante globo diventa arancione con «Verifica Google richiesta: clicca per completarla». Lo strumento restituisce un errore che chiede di aprire il browser dal pulsante, completare la verifica e ripetere la ricerca. Completata la verifica il pulsante scompare; la finestra resta aperta finché non la chiudi. La pagina in attesa viene sostituita dalla successiva azione web. I contenuti web sono dati non attendibili: non concedono autorizzazioni e non modificano le regole della sessione. Interrompi annulla anche l’attesa della pagina.

Gli script di estrazione e la logica di navigazione derivano da ds4 (https://github.com/antirez/ds4), licenza MIT, Copyright (c) 2026 The ds4.c authors; il testo della licenza è in `licenses/ds4-MIT.txt`.

## Verifiche del motore 1.4.7

La versione nel branch di test conserva i file SUC1 esistenti e la politica di memoria. DeepSeek e GLM usano il percorso TCQ FP32 verificato; per confrontarlo con quello storico imposta `SUC_TCQ_REUSE=0` all’avvio della CLI o del server. Le altre famiglie mantengono il kernel predefinito. Nella CLI il prefill automatico di Qwen usa fino a 1024 token per blocco; app e server conservano il proprio profilo prudente. Il valore esplicito di `--chunk` viene rispettato. Il campionamento può produrre testi diversi con lo stesso seed, mantenendo la distribuzione prevista. Il riuso del prefisso accelera l’elaborazione di contesti ripetuti corrispondenti. MTP è rinviato.

CLI e server autonomo per Apple Silicon: `dist/SUC-CLI-Server-1.4.7.03-arm64.tar.gz`, con esempi di configurazione e hash dei binari inclusi. L’app e gli strumenti ausiliari Universal supportano il funzionamento come client remoto su Intel e l’inferenza locale su Apple Silicon. I risultati della verifica KL sono in `../docs/optimization/REFERENCE_KL.md`; i collaudi dei pacchetti sono registrati in `../docs/optimization/RELEASE_SMOKE_1.4.7.json`.

### Profili di prefill e qualità

App e server mantengono blocchi automatici di 64 righe per Qwen e DeepSeek e di 512 righe per GLM; la CLI conserva il profilo separato per i prompt lunghi. Il valore esplicito di `--chunk` per il prefill in batch viene rispettato. Cambiare i confini dei blocchi può cambiare gli arrotondamenti: la sola velocità non giustifica quindi una modifica automatica. La cache conserva un unico stato attivo del motore, senza salvare una KV cache separata per ogni chat. Qwen conserva anche un solo checkpoint degli stati ricorrenti e delle convoluzioni, allineato ai blocchi e poco prima della fine del prompt. Se il nuovo turno coincide con il prefisso salvato, rielabora solo la coda; conserva il formato SUC esistente che rimuove il ragionamento dai turni precedenti. La copia degli stati è limitata a 128 MiB e la KV cache non viene duplicata. Cambiamenti anteriori al checkpoint richiedono il prefill completo. Non occorre scaricare nuovi file SUC1.

### Consegna qualificata 1.4.7 — build 28

Installer Universal: macOS 15 o successivo; su Intel usa Server remoto. L’archivio CLI/server ARM64 distribuito richiede invece macOS 26.0 o successivo (target verificato nei binari). Dopo l’estrazione segui USAGE.md nell’archivio. Verifica i pacchetti dalla cartella dist con `shasum -a 256 -c SHA256SUMS-1.4.7.txt`. Il server standalone non include autenticazione: per la LAN con API key usa il server dell’app. Nessun nuovo download del modello è necessario.

KL invariata su sei modelli; superati tutti i dodici confronti 8k/32k e i collaudi reali CLI/server/motore integrato. Consulta `docs/optimization/RELEASE_1.4.7.md` nel progetto e i documenti tecnici inclusi nel pacchetto per protocolli, impostazioni e limiti.

### Streaming remoto nella 1.4.7.02

La lettura dello stream remoto avviene fuori dal thread dell’interfaccia; restano attivi cancellazione, controlli di dimensione e gestione UTF-8. Le metriche di prefill derivate dai tempi contano solo i token nuovi, mentre le velocità esplicite del server hanno precedenza. Con cache completa senza una velocità esplicita compare `—`. Le impostazioni, le sessioni e le API key esistenti vengono conservate.

La 1.4.7.02 riduce il ritardo di ricezione nel client. La 1.4.7.03 aggiunge il checkpoint dei turni Qwen descritto sopra. Il primo turno lungo e le modifiche anteriori al prefisso salvato richiedono ancora il prefill completo. Template e kernel restano invariati. Per aggiornare chiudi l’app e avvia `dist/SUC-Studio-1.4.9.03-universal.pkg`; non occorre ricaricare o convertire i file SUC1 per usare il nuovo client remoto.

### Avanzamento e velocità nella 1.4.7.04

Durante il prefill la barra mostra token nuovi elaborati e prefisso riusato, quando il motore invia questi dati. Nelle chat lunghe con cronologia, zero token riusati produce l’avviso di rilettura completa: il comportamento è normale dopo un reset o un prefisso incompatibile. Senza telemetria la barra rimane indeterminata. Durante il caricamento del modello la barra globale mostra la fase effettiva, senza simulare percentuali.

SUC invia i conteggi reali durante il decode; la media live e quella finale usano la stessa formula. Negli altri server, «Ricezione ≈» indica una stima dai caratteri ricevuti e dalla rete, non la velocità del motore. Il dato finale può quindi differire. Le metriche reali del server hanno sempre precedenza.

### Candidate 1.4.7.05: caricamento e prefill automatico

Durante il caricamento la barra globale mostra un indicatore di attività e la fase comunicata dal motore, fino alla conferma di modello pronto. I GiB delle pagine del file residenti e l'allocazione Metal sono dati separati: possono aumentare o diminuire e non rappresentano una percentuale di completamento. Il dettaglio indica la residenza iniziale. Il monitor non forza letture, page-in o variazioni della politica di memoria.

Nei Settings, Prefill → **Automatico (1024 / 64)** abilita la rilettura in blocchi da 1024 e la coda in blocchi al massimo da 64. Richiede ricaricare il modello. Cache e checkpoint vengono riusati prima di rileggere il contesto; il decode resta a un token per passo. Con exact/token, chunk esplicito o streaming degli expert, rimane il percorso previsto da quei profili. **Batch prudente** rimane predefinito: il confronto sul modello grande è pendente e non viene promessa equivalenza numerica fra profili. CLI server: `--history-prefill auto` abilita la stessa schedulazione, `off` conserva il profilo precedente.


### Passaggi intermedi e comandi nella chat

Le premesse e i ragionamenti del modello tra due azioni rimangono visibili, anche riaprendo le sessioni precedenti. Sono nascosti soltanto il protocollo JSON e i risultati tecnici interni degli strumenti. I comandi shell e interattivi hanno un dettaglio «Comando» chiuso di default: espandilo per leggere il comando, poi richiudilo. Per le nuove azioni viene conservato il comando effettivo dopo eventuali modifiche degli hook. Password e altri input riservati restano gestiti dalla scheda terminale.


### Schemi MCP estesi

I cataloghi MCP accettano schemi completi fino a 3 MiB per strumento, conservati come JSON compatto senza troncamenti. Rimane il limite di 256 strumenti per server. I risultati testuali MCP possono raggiungere 4 MiB senza troncamenti; il trasporto JSON dispone di 32 MiB per escaping e metadati. Se uno schema supera il limite, il messaggio indica lo strumento e la dimensione. Connessione REA verificata tramite initialize e tools/list; la connessione non esegue strumenti.


### Pannello artefatti

Gli artefatti non compaiono più in fondo alla chat. L’icona a forma di pacco nella barra della chat, con il numero dei file disponibili, apre e chiude il pannello laterale. Il pulsante × nel pannello lo nasconde completamente. Il pannello segue la sessione selezionata, è ridimensionabile e mantiene i pulsanti Salva e Finder. Rimane chiuso finché non lo apri, anche quando viene prodotto un nuovo file.


### MCP su richiesta e continuità delle skill

Il prompt iniziale contiene i server MCP connessi e il numero di strumenti, senza copiarne tutti gli schemi. Il modello cerca gli strumenti con mcp_list (query e paginazione), consulta mcp_schema e poi usa mcp_call con le normali approvazioni. Gli schemi grandi si consultano tramite path JSON Pointer e paginazione: lo schema completo resta conservato nel pannello MCP. Le skill richiamate con @nome o $nome rimangono attive durante tutte le azioni dello stesso turno. La lettura del catalogo non esegue strumenti remoti.

Il health check prova anche /health se il percorso sotto /v1 non è disponibile. Un server raggiungibile non dimostra che stia elaborando: lo stato indica l’attesa della risposta, senza simulare avanzamento. Il pulsante × degli artefatti è un controllo AppKit con binding diretto alla chat e area di clic di 28 pt. Il pannello laterale usa una vista divisa ridimensionabile; nelle finestre strette l’elenco delle sessioni viene nascosto temporaneamente per lasciare spazio alla conversazione.


### Risultati MCP fino a 4 MiB

Il limite vale sul testo UTF-8 complessivo restituito dallo strumento, inclusi eventuali contenuti strutturati. Risultati superiori vengono rifiutati con un errore esplicito; nessun taglio automatico. Chat locale, client remoto e server integrato accettano richieste JSON fino a 32 MiB. Restano i limiti di contesto del modello e gli eventuali limiti del server remoto: 4 MiB di testo possono contenere molti più token di quelli disponibili. Il limite degli allegati e dei comandi shell resta invariato.


## Messaggi in coda — 1.4.8

Puoi scrivere e premere Invio anche durante la generazione: il messaggio e i suoi allegati vengono messi in coda nella sessione corrente. Partono in ordine quando finisce l’intero turno, comprese le azioni degli strumenti. Il pannello sopra il compositore permette di mettere in pausa, riprendere, rimuovere o riportare un messaggio in bozza. Per modificarlo devi avere una bozza vuota; il nuovo invio lo rimette in fondo alla coda. Interrompi ferma la richiesta attuale e mette in pausa la coda. Anche un errore di generazione la sospende. I messaggi sono salvati localmente e dopo un riavvio aspettano Riprendi. Puoi cambiare sessione mentre la coda continua; la bozza che stai scrivendo non viene inviata automaticamente.


## Contesto, comandi e plugin — 1.4.8.02

L’anello accanto al compositore mostra la percentuale di contesto utilizzata; ≈ indica una stima. Cliccalo per vedere token, limite e auto-compact. Il server locale usa il limite caricato; per un server remoto premi Verifica / modelli e salva la configurazione, oppure usa `/context 32768` per la sessione. `/context 0` torna al rilevamento automatico. Senza limite noto compare ? e l’autocompact non parte. Le immagini hanno un costo variabile, stimato finché non arriva il conteggio server.

`/compact` sintetizza la conversazione col modello corrente; puoi aggiungere priorità dopo il comando. Il transcript e gli allegati originali rimangono salvati, mentre le richieste successive usano la memoria sintetica. La sintesi può omettere dettagli: consulta l’originale quando serve. La compattazione automatica al 99% interviene fra le richieste, considerando anche una riserva per l’output. Non interrompe il decode. Per disabilitarla usa `/auto off`; `/auto on` la ripristina. I messaggi in coda attendono la fine della sintesi. Se viene interrotta o fallisce, il vecchio contesto resta intatto e la coda si mette in pausa.

Il menu / e `/help` mostrano i comandi: `/status`, `/new`, `/clear`, `/stop`, `/queue`, `/pause`, `/resume`, `/skills`, `/skill nome richiesta`, `/plugins`, `/plugin identificatore richiesta`, `/model`, `/workspace`, `/export`, `/retry`, oltre a quelli del contesto. `/clear` conserva lo storico e riparte dalle istruzioni di sistema. Per testo letterale che inizia con / scrivi //. Un comando va inviato senza allegati.

Per richiamare un plugin attivo puoi usare anche `@plugin:identificatore richiesta` o il menu delle skill. Il modello può selezionarlo tramite il catalogo e `load_plugin`; skill solo esplicite, MCP e hook conservano le rispettive impostazioni e approvazioni. Il richiamo non attiva plugin disabilitati.

Le chat lunghe caricano inizialmente gli ultimi 80 messaggi; Carica messaggi precedenti mostra il resto. Durante lo scroll manuale i messaggi visibili restano montati. Oltre 8000 caratteri l’anteprima non collassa più a 12 righe: Apri testo completo mantiene accessibile tutto il contenuto.

### Correzioni 1.4.8.03

La compattazione adatta il budget allo storico: da 2048 a 16384 token di output, ridotti se la finestra è piccola; un solo secondo tentativo può arrivare a 32768 token, se c’è spazio. Il budget comprende l’eventuale thinking del backend e viene mostrato durante la sintesi. Sintesi vuote o troncate non sostituiscono la memoria precedente. Alcuni backend possono ignorare la richiesta di disattivare il thinking.

Le routine riservano almeno 4000 token di output, entro il contesto disponibile. Una risposta contenente solo thinking non è un risultato completato: la chat mostra il motivo e la routine registra l’errore. L’indicatore del contesto si aggiorna circa ogni secondo; i conteggi prima dell’invio vengono calcolati al momento dell’invio. La velocità reale dei server esterni dipende anche dal loro modello e dal contesto.

Lo storico grafico usa pagine limitate con pulsanti per messaggi precedenti/successivi; tutti i messaggi rimangono salvati. I ragionamenti storici partono richiusi e sono espandibili. Il ragionamento in streaming si apre e non si richiude automaticamente a fine risposta, evitando salti di altezza.

### Correzioni build 36 (stessa versione 1.4.8.03)

Le azioni MCP accettano il nome configurato, il suo UUID e un alias univoco senza il suffisso MCP: ad esempio `rea` riconosce `REA mcp`. In presenza di ambiguità viene richiesto il nome esatto o l’UUID; non viene scelto un server arbitrario. Un nome sconosciuto mostra i server disponibili e non viene più descritto come una disconnessione.

Se gli strumenti web sono abilitati e le approvazioni sono automatiche, anche il primo avvio del browser segue la stessa impostazione, sia nelle chat interattive sia nelle routine. La modalità manuale conserva le conferme; gli strumenti web disabilitati restano bloccati.

## Lingua dell’interfaccia · 1.4.9

Il primo avvio usa l’inglese, anche con macOS in italiano. Da Settings → Language scegli English o Italiano; la preferenza è salvata nell’archivio delle sessioni e riletta al caricamento. Anche gli archivi precedenti senza preferenza usano l’inglese. Il cambio aggiorna l’interfaccia senza tradurre conversazioni, nomi, codice, skill o risultati degli strumenti e senza imporre la lingua delle risposte del modello. I dialoghi di sistema e i messaggi provenienti dai server possono seguire la propria lingua. La guida HTML segue la lingua selezionata.

Il controllo «Consenti comandi shell» è disabilitato finché non sono selezionati una directory di lavoro e «Consenti azioni del modello». La modalità di approvazione automatica non abilita da sola gli strumenti. Il fix 1.4.8.04 è incluso senza una build intermedia.

Android 0.1.5 offre la stessa scelta IT/EN, con inglese predefinito e preferenza salvata nell’archivio del dispositivo. Nessuna sincronizzazione automatica con il Mac.

## Comandi prolungati · 1.4.9.02

Il modello avvia un job con `run_command`, poi ne legge lo stato con `command_status`, attende con `wait_command` e lo interrompe con `stop_command`. Il superamento dell’attesa non termina il processo; non esiste più la scadenza fissa di 30 secondi. La durata totale è facoltativa (`timeout_seconds`). L’avvio segue i permessi della sessione; le consultazioni dello stesso job non richiedono una nuova approvazione né consumano il limite delle azioni.

I riquadri dei comandi restano nella conversazione: tutta la riga permette di aprire e chiudere i dettagli. Il pannello fisso in fondo alla chat è rimosso; dettagli e output conclusi sono consultabili nel registro azioni della sessione. Interrompi ferma il turno e il comando; i nuovi messaggi restano in coda fino alla conclusione dei job. La coda di output è limitata a 256 KiB per job, fino a 64 KiB per lettura, con omissioni esplicitamente indicate; per log integrali occorre un file. I job non vengono ricollegati dopo il riavvio. I comandi che richiedono input continuano a usare il terminale inline.

Il contesto predefinito delle nuove configurazioni è 16000 token. I limiti già salvati o dichiarati dal server remoto hanno precedenza; per i server che non dichiarano una finestra, il pannello del contesto indica il valore assunto di 16000 token, modificabile.

## Recupero dell’agente · 1.4.9.03

Le brevi promesse IT/EN senza azioni e le richieste di strumento malformate possono attivare fino a due tentativi di correzione condivisi per turno, entro il limite azioni. Il modello deve richiedere un’azione valida già autorizzata oppure spiegare risultato o impedimento. L’harness non inventa contenuti o argomenti delle azioni. Approvazioni e Stop restano applicati; al limite, errore esplicito e coda sospesa. Risposte finali normali e domande restano conclusive. Metriche streaming senza conteggi di output non riportano più decode a prefill. Il controllo euristico conservativo non garantisce il completamento del compito.

Build 41: i marcatori tecnici tool_call non compaiono durante lo streaming o nello storico. L’auto-compattazione usa il 99% dell’input proiettato senza sommare una riserva output; i conteggi server validi restano l’ancora della stima. La risposta viene limitata allo spazio residuo. Solo dopo uno stop normale è tollerata una singola parentesi JSON esterna mancante, con azione e argomenti già completi e validati senza modifiche. Stringhe/argomenti incompleti e richieste troncate per lunghezza non vengono completati.

Build 42: rimosso il pannello fisso Comandi della sessione. Il clic su tutta la riga del comando (freccia, etichetta o spazio vuoto) apre e chiude i dettagli. Il testo del comando aperto resta selezionabile. Stop e terminale interattivo sono invariati; gli output conclusi restano nel registro azioni della sessione.

Il registro azioni mostra schede più spaziose, dalla più recente, con nome leggibile, stato e ora. Il clic su tutta la riga apre richiesta e risultato separati. Resta consultabile durante il lavoro. Le anteprime sono limitate; copia e apertura del risultato completo archiviato restano disponibili.

## Compattazione · build 43

Corretta la mescolanza tra byte UTF-8 e token: memoria precedente, istruzioni, blocchi di storico e spazio per la risposta sono ora conteggiati nella stessa unità stimata in token. La conversione in byte avviene soltanto per dividere il testo in blocchi Unicode integri. La soglia automatica resta al 99% e la finestra impostata non viene aumentata.

Se la memoria precedente non entra dopo una riduzione della finestra, viene ricompattata prima dei nuovi dati, senza tagli automatici; i tentativi di riduzione sono limitati. Il secondo tentativo di sintesi rispetta lo spazio residuo. Conteggi stimati e tokenizer del server possono differire: non si tratta di una tokenizzazione esatta universale.

Se la compattazione fallisce o viene fermata, il risultato dello strumento già completato viene aggiunto integralmente allo storico come normale risultato operativo, senza rieseguire il comando. Lo storico e la memoria precedente restano conservati. Si può riprovare `/compact`, poi continuare la chat; il fallimento non riprende automaticamente le azioni. Le priorità manuali troppo grandi possono ancora richiedere di ridurre le istruzioni o aumentare una finestra effettivamente supportata dal server.

## Contesto e thinking remoto · build 44

Il contesto futuro esclude il reasoning separato che non viene reinviato al modello. Se il server espone `completion_tokens_details.reasoning_tokens`, si usa il conteggio; negli archivi precedenti si stima soltanto il contenuto effettivamente reinviato, indicando il valore come stimato. Il reasoning incorporato nel testo resta conteggiato. Durante la generazione il reasoning continua a consumare la finestra e le metriche decode includono tutti i token prodotti. La soglia di auto-compattazione resta al 99%.

In Server remoto, con parametri estesi attivi, il nuovo Controllo thinking remoto permette Automatico per profilo, `chat_template_kwargs`, `reasoning_effort = none`, `think` oppure Predefinito del server. In automatico, LM Studio usa `reasoning_effort=none` quando il thinking è spento; Router aggiunge anche `think`, DS4 usa `think`. Gli override consentono di scegliere il parametro effettivamente supportato dal backend. I server che rifiutano il parametro restituiscono un errore senza reinvii automatici. Con estensioni disattivate non vengono aggiunti questi controlli.

La compattazione ricompatta anche la memoria precedente, richiede una sintesi breve (256–2048 token come obiettivo, distinto dal budget di generazione), poi elabora i nuovi dati. Se due tentativi non producono una sintesi completa, il blocco viene diviso in due metà Unicode esatte e riprovato, fino a due livelli e con un limite complessivo proporzionale ai blocchi iniziali. Nessuna sintesi troncata o solo thinking viene salvata. Gli errori riportano il budget finale, i conteggi disponibili e le dimensioni di risposta/ragionamento senza mostrarne il contenuto. Stop, storico, memoria precedente e risultati degli strumenti sono conservati come nella build 43. Non è garantito il successo di modelli che ignorano tutti i controlli o non producono mai una risposta finale.

La generazione registra il massimo output effettivamente richiesto. Se il contesto limita la risposta, l’avviso indica il budget reale e propone `/compact` o input più breve: alzare Output max non allarga la finestra. Nessuna compattazione anticipata viene aggiunta.
