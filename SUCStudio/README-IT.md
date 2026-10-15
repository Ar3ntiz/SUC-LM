# SUC Studio

English version: [README.md](README.md)

Versione corrente: **1.4.5_Final** — installer `dist/SUC-Studio-1.4.5_Final-arm64.pkg`.

App macOS nativa (SwiftUI) per usare i modelli SUC1 con il runtime SUC C++/Metal: chat locali, server compatibile OpenAI, collegamento a server remoti, strumenti per il modello e ricerca web. Destinazione: Apple Silicon, macOS 15+.

- **[USAGE-IT.md](USAGE-IT.md)** — installazione, configurazione, utilizzo e risoluzione dei problemi.
- **[docs/QA_REPORT.md](docs/QA_REPORT.md)** — verifiche eseguite e limiti effettivi.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — struttura e protocollo privato tra app e motore.
- **[docs/WEB_TOOLS_SPEC_IT.md](docs/WEB_TOOLS_SPEC_IT.md)** — specifica degli strumenti web.

## Funzionalità principali

- **Modelli locali** SUC1 con piano di memoria automatico: residenti quando entrano nella memoria GPU, altrimenti con streaming degli expert MoE da SSD.
- **Chat** con sessioni, rami, export / import, Markdown, reasoning separato, modifica e rigenerazione, allegati di testo, documenti, PDF e immagini (backend vision).
- **Server** compatibile OpenAI integrato (API key, accesso LAN opzionale) e **client remoto** verso SUC Studio, `suc1-server` o qualsiasi motore con API compatibili OpenAI (ds4, LM Studio, llama.cpp, router LLM), con profili di telemetria.
- **Strumenti per il modello**: directory di lavoro, file e comandi, terminale interattivo, artifact, MCP, skill, subagent, hook, plugin e routine, con approvazioni manuali o automatiche e registro delle operazioni.
- **Ricerca web** (`web_search`, `visit_page`) con browser integrato nascosto e pulsante globo per seguirla dal vivo.

## Build

```sh
./scripts/build.sh
./scripts/test.sh
./scripts/package.sh
```

La build usa **un'unica sorgente del motore: `../cpp` del progetto padre SUC-LM**; `SUCStudio/` deve restare dentro il repository. `Sources/` contiene l'app, `Engine/` il wrapper per IPC, memoria e HTTP, `dist/` app e installer, `.build/` una cache ricreabile. Ogni build verifica gli hash dei sorgenti del motore, ricompila quelli cambiati (anche non committati) e registra gli input in `dist/ENGINE_BUILD_MANIFEST.json`, incluso nel bundle. L'app e il pacchetto sono autonomi: sul Mac di destinazione non serve il repository. Su questo Mac la build richiede `SDKROOT=…/MacOSX26.5.sdk` (l'SDK 27 dei Command Line Tools non ha il plugin delle macro SwiftUI).

Pesi e tokenizer dei modelli non sono inclusi (vedi [huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz)). Distribuzione locale con firma ad-hoc, senza Developer ID né notarizzazione.

## Release (dalla più recente)

### 1.4.5_Final
- Browser web **nascosto**: nessuna finestra sopra il lavoro e nessun furto del focus; tempi identici a ds4 (pannello fuori schermo non rallentato da WebKit, misurato).
- **Pulsante globo** accanto alle impostazioni della chat, solo durante una ricerca: fase, URL e secondi; al clic apre il browser con la pagina dal vivo e una barra di stato.
- Verifica Google segnalata dal pulsante arancione, mai portata in primo piano; nessuna finestra vuota a fine azione.

### 1.4.5_Beta
- Strumenti web in **ogni sessione** che li consente, anche senza directory di lavoro e con le azioni del modello spente (prompt dedicato con solo `web_search` e `visit_page`).
- Senza directory le pagine sono salvate in `~/Library/Application Support/SUC Studio/web`.
- Richieste d'azione accettate anche dentro un blocco di codice `json`.

### 1.4.5
- Strumenti web per il modello: `web_search` (alias `google_search`) e `visit_page`, con la logica del modulo web di ds4 su WebKit.
- Profilo browser dedicato e persistente, cancellabile dai Settings; conferma di avvio una volta per esecuzione e interruttore per sessione.
- Pagina completa in `.suc-web`; verifiche Google mai aggirate; solo URL http e https.

### 1.4.4
- Generazioni remote lunghe con health check.
- Budget di memoria fino al 96 % e riserva KV più precisa.
- Terminale inline solo per le richieste interattive del modello; registro I/O diagnostico disattivato di default; legenda del sampling e directory di lavoro sotto la chat.

### 1.4.3
- Reasoning separato dalla risposta, editor di codice aperto da tre backtick, composer ridimensionabile.
- Rendering asincrono e anteprime per input lunghi reattivi.
- Profili di telemetria per SUC Studio, DS4, LM Studio, llama.cpp, LLM Router e JSON personalizzato, con misure e stime distinte.

### 1.4.2
- Terminale interattivo nella chat per comandi, conferme e input riservati, usabile anche dal modello con i permessi della sessione.
- Artifact persistenti per i file prodotti, con apertura e salvataggio sul Mac.

### 1.4.1
- Registro operazioni espandibile (coda, stato ed esito delle ultime 200 operazioni).
- La chat segue il fondo e si ferma allo scroll manuale; elenco sessioni scorrevole.

### 1.4.0
- Limite di azioni consecutive configurabile.
- JSON delle azioni nascosto dal transcript ma mantenuto nel contesto e nel registro; riepilogo finale delle attività e segnalazione immediata degli errori.

### 1.3.0
- Approvazioni dentro la chat; indicatore di caricamento modello in tutte le sezioni; animazioni distinte per prefill e decode.
- Telemetria dei server remoti con velocità reali quando disponibili.

### 1.2.0
- Allegati immagine per backend vision, con anteprima e persistenza.
- Modifica e rigenerazione nella stessa chat; pannelli Server locale e Server remoto alternati; gestione hook con esempi.

### 1.1.0
- Link della chat aperti nel browser di sistema; rendering Markdown rivisto (streaming, codice, tabelle, liste, citazioni).

### 1.0.0
- Sezione Aspetto: font e dimensioni separati per interfaccia e chat.
- Avvio al login e Mac mantenuto attivo.

### 0.9.0
- Plugin installabili da cartelle locali o marketplace remoto, con attivazione e rimozione.
- Routine pianificate con prompt, frequenza, stato e storico.

### 0.8.1
- Piano di memoria dal working set Metal reale con margine per il sistema: DeepSeek e GLM tornano residenti invece che in streaming da SSD.
- Stima KV di GLM sui latenti MLA; riserva KV + 2 GiB + 3 % del file.

### 0.8.0
- Subtask e subagent con contesti separati; limiti di delega e autorizzazioni ereditate.
- Hook configurabili per gli eventi del ciclo di lavoro.

### 0.7.0
- Directory di lavoro per sessione con strumenti per file e comandi.
- Approvazione manuale o automatica, registro delle azioni e ciclo strumenti guidato dal modello.

### 0.6.0
- Chat collegabile al motore locale o a un server remoto compatibile OpenAI (modello, API key, streaming, interruzione).

### 0.5.0
- System prompt globale, skill selezionabili, MCP locali (stdio) e remoti (HTTP); credenziali nel Portachiavi.
- Barra delle sezioni comprimibile e guida HTML offline.

### 0.4.0
- Motore C++ compilato direttamente dal progetto padre, con cache di build basata su hash e manifest; nuova icona.

### 0.3.0
- Allegati: testo con qualsiasi estensione, Word DOC/DOCX, OpenDocument, RTF e PDF testuali, con anteprima ed export.

### 0.2.0
- Gestione automatica della memoria, residente o con budget expert; streaming SSD degli expert MoE; controlli preventivi sui modelli troppo grandi.

### 0.1.0
- Prima versione: chat e sessioni locali, impostazioni modello / tokenizer, server compatibile OpenAI con API key e accesso LAN, barra dei menu e telemetria del motore.

Storico completo: `dist/CHANGELOG.md`; i pacchetti precedenti restano in `dist/`.

## Crediti

Script di estrazione web e logica di navigazione adattati da ds4 (https://github.com/antirez/ds4), licenza MIT, Copyright (c) 2026 The ds4.c authors. Testo completo in [licenses/ds4-MIT.txt](licenses/ds4-MIT.txt).
