# Changelog SUC Studio

## 1.4.8.03 — build 36

Compattazione con budget dinamico e rollback delle sintesi troncate. Routine con riserva output e verifica del risultato. Chat lunghe alleggerite con paginazione, ragionamenti richiudibili e metriche limitate. Build 36: alias MCP univoci, errori di identità distinti dalle disconnessioni e avvio browser coerente con le approvazioni automatiche. Motore e memory policy invariati.

## 1.4.8.02 — build 34

Indicatore circolare del contesto, /compact e auto-compact al 99%, con storico originale conservato e limite remoto rilevabile o impostabile. Comandi della chat, richiamo esplicito dei plugin e caricamento autonomo dal modello. Corrette sparizioni e salti nelle chat lunghe: anteprima stabile oltre 8000 caratteri e cronologia caricata progressivamente. Coda, permessi e memory policy preservati.

## 1.4.8 — build 33

Messaggi inviabili durante prefill, decode e azioni, con coda persistente per sessione. Invio sequenziale dopo il turno completo, allegati e bozze preservati. Pannello con pausa/ripresa, rimozione e ritorno in bozza. Stop/errori sospendono la coda; al riavvio serve ripresa esplicita. Motore e memory policy invariati.

## 1.4.7.05 — build 32 · candidate

Loader per fase. Corretta la × degli artefatti. MCP su richiesta, schemi fino a 3 MiB, risultati fino a 4 MiB e skill esplicite mantenute durante le azioni. Health check compatibile col router. Passaggi intermedi visibili e comandi espandibili. Prefill Automatico 1024/64 opzionale; Batch prudente predefinito. Android 0.1.4 aggiorna il modello dopo Verifica. Versione e build invariate.

## 1.4.7.04 — build 31

Prefill con avanzamento dai token realmente elaborati e avviso di rilettura completa nelle chat lunghe senza riuso della cache. Caricamento visibile da tutte le tab, con barra e fase effettiva. SUC invia velocità decode reali durante lo streaming; le stime client sono indicate come ricezione. Versione dei Settings centralizzata. Modelli, kernel e memory policy invariati.

## 1.4.7.03 — build 30

Cache dei turni Qwen: checkpoint ricorrenti limitati a 128 MiB, senza duplicare la KV, con riuso del prefisso anche quando il template rimuove il ragionamento precedente. Ricalcolo se incompatibile; rilascio dei riferimenti nei task conclusi. Versione nei Settings collegata all’identità centrale. Template, file SUC1 e memory policy invariati.

## 1.4.7.02 — build 29

Corretto il ritardo dello streaming remoto su macOS: lettura fuori dal thread dell’interfaccia, mantenendo cancellazione e limiti di risposta. Prefill calcolato sui token nuovi, con compatibilità llama.cpp e cache completa. Punteggio Qwen3.8 aggiornato a 100/120 con chunk 64. Motore C++/Metal e memory policy invariati.

## 1.4.7 — build 28

Motore condiviso più rapido: sampling lineare, riuso sicuro del prefisso e TCQ FP32 per DeepSeek/GLM. KL invariata su sei modelli; superati dodici confronti a 8k/32k e collaudi reali CLI/server/app. Profili prefill e memory policy preservati. Installer Universal e CLI/server ARM64 verificati. MTP rinviato.

## 1.4.6.04

Consolidata la versione Universal, verificata su Intel, M3 e M1 Ultra, come unico SUCStudio. Eliminata la biforcazione dei sorgenti e rimossa «Universal Preview» dal nome visibile. Conservati dati, impostazioni e credenziali della Universal. Installer aggiornato con rimozione della vecchia app Preview. Inferenza locale Apple Silicon e client remoto Intel invariati.

## 1.4.6.02

Skill con risorse, hook a eventi e plugin portabili/Codex/Claude con MCP e marketplace. «Crea con esempi» esporta Skill + Hook + MCP insieme, con editor indipendenti e validazione completa. Corrette le ricerche schedulate quando il modello antepone testo al JSON: azione eseguita, protocollo nascosto anche in streaming, risultato inviato al modello. Gestite cancellazioni e registro I/O. Installer Universal aggiornato.

## 1.4.6

Nasconde il JSON dei tool anche durante lo streaming; riepiloghi attività piccoli e compatti su sfondo grigio. Routine una tantum o ricorrenti in minuti, ore, giorni e mesi; corretto l’avvio web automatico e segnalati risultati/approvazioni non letti con un puntino giallo. Skill con titolo, descrizione e caricamento su richiesta. Output predefinito delle nuove chat: 4000 token.

## 1.4.5_Universal

Variante parallela della 1.4.5_Final: app e helper Universal per Apple Silicon e Intel, macOS 15+. Inferenza SUC locale su Apple Silicon; server OpenAI compatibili su Intel, anche localhost. Identità, sessioni e credenziali separate dalla versione stabile. Conserva strumenti, web search e telemetria. Il motore C++ resta unico nel progetto padre. Validazione su hardware Intel da completare.

## Storico precedente al consolidamento

## 0.1.0 — rilasciata

Prima app macOS Apple Silicon: chat e sessioni locali, impostazioni modello/tokenizer, server API compatibile OpenAI con chiave e accesso LAN, menu bar e telemetria del motore. Installer locale senza firma Developer ID.

## 0.2.0 — rilasciata

Gestione memoria automatica, residente o con budget expert. Streaming SSD degli expert MoE tramite il motore esistente, statistiche a fine fase e controlli preventivi sui modelli troppo grandi. Nessuno streaming SSD dei pesi densi.

## 0.3.0 — rilasciata

Allegati tramite trascinamento e selezione: testo con qualsiasi estensione, Word DOC/DOCX, OpenDocument, RTF e PDF testuali. Estrazione trasparente nell’harness, anteprima, persistenza ed export. Limiti espliciti senza troncamento silenzioso.

## 0.4.0 — rilasciata

Motore C++ compilato direttamente dal progetto padre, eliminando la copia duplicata. Cache di build basata su hash e manifest degli input. Nuova icona desktop coerente con il simbolo della barra menu.

## 0.5.0 — rilasciata

System prompt globale, skill selezionabili, MCP locali stdio e remoti HTTP. Strumenti eseguiti manualmente dalla chat con risultati allegati. Credenziali nel Portachiavi, barra sezioni comprimibile e guida HTML offline aperta nel browser.

## 0.6.0 — rilasciata

Chat collegabile al motore locale oppure a un server remoto compatibile OpenAI. Configurazione nella sezione Server, selezione modello, API key, streaming e interruzione. Stessi prompt, skill e allegati; metriche solo quando disponibili.

## 0.7.0 — rilasciata

Directory di lavoro per sessione e strumenti dell’harness per file e comandi. Modalità di approvazione manuale o automatica, registro delle azioni e ciclo strumenti guidato dal modello, con limiti e cancellazione.

## 0.8.0 — rilasciata

Subtask definiti dal modello e subagent con contesti separati sui backend capaci. Limiti di delega e autorizzazioni ereditate. Hook configurabili dai Settings per gli eventi del ciclo di lavoro, con timeout e gestione degli errori.

## 0.8.1 — rilasciata

Fix memoria dell’utente: limite basato sul working set Metal reale e margine OS, stima KV GLM sui latenti MLA, riserva KV + 2 GiB + 3% del file. Validato su Mac Studio 128 GB con limite aumentato: caricamento residente dei modelli grandi, senza streaming SSD degli expert.

## 0.9.0 — rilasciata

Plugin SUC Studio installabili da cartelle locali o marketplace remoto documentato. Gestione di attivazione e rimozione. Routine schedulabili con prompt, frequenza, stato e storico; esecuzione quando l’app è aperta.

## 1.0.0 — rilasciata

Font e dimensioni indipendenti per interfaccia e chat nella sezione Aspetto. Nuova sezione con avvio al login e mantenimento del Mac attivo, con stato esplicito e integrazione nelle API macOS.

## 1.1.0 — rilasciata

Link della chat apribili nel browser di sistema. Revisione e test del rendering Markdown, comprese risposte in streaming, codice, tabelle, liste, enfasi e citazioni, con attenzione a leggibilità e dimensioni minime.


## 1.2.0 — rilasciata

Allegati immagine per backend vision, anteprima e persistenza. Modifica e rigenerazione nella stessa chat con annullamento della richiesta precedente. Pannelli Server locale e Server remoto alternati. Gestione hook con pulsante Chiudi ed esempi selezionabili.

## 1.3.0 — rilasciata

Approvazioni dentro la chat, indicatore di caricamento modello visibile in tutte le sezioni e animazioni distinte per prefill e decode. Telemetria dei server remoti con velocità reali quando disponibili e indicazione esplicita delle misure mancanti o stimate.

## 1.4.0 — rilasciata

Limite delle azioni consecutive configurabile nei Settings. JSON delle azioni e risultati tecnici nascosti dal transcript, mantenuti nel contesto del modello e nel registro. Riepilogo finale delle attività svolte e segnalazione immediata degli errori.

## 1.4.1 — rilasciata

Registro operazioni espandibile, chiuso di default, sopra gli eventi: coda, stato ed esito delle ultime 200 operazioni. La chat segue il fondo anche dopo variazioni di layout e si ferma allo scroll manuale. Elenco sessioni scrollabile e nuova selezione portata in vista.

## 1.4.2 — rilasciata

Terminale interattivo nella chat per comandi, conferme e input riservati, utilizzabile anche dal modello con le autorizzazioni della sessione. Artifact persistenti per i file prodotti, con apertura e salvataggio sul Mac.

## 1.4.3 — rilasciata

Reasoning distinto dalla risposta, editor codice aperto da tre backtick e composer ridimensionabile. Rendering asincrono e anteprime rendono reattivi gli input lunghi. Profili telemetria per SUC Studio, DS4, LM Studio, llama.cpp, LLM Router e JSON personalizzato, con misure e stime chiaramente distinte.

## 1.4.4 — rilasciata

Generazioni remote lunghe con health check. Budget memoria fino al 96% e riserva KV più precisa. Terminale inline solo per richieste interattive del modello. Registro I/O diagnostico disattivato di default, legenda sampling e work directory sotto la chat.

## 1.4.5 — rilasciata

Strumenti web per il modello: `web_search` (alias `google_search`) e `visit_page`, con la logica del modulo web di ds4 su WebKit. Finestra Browser SUC Studio visibile, profilo dedicato persistente cancellabile dai Settings, conferma di avvio una volta per esecuzione e opzione per sessione. Pagina completa in `.suc-web`, verifiche Google mai aggirate, solo URL http e https.

## 1.4.5_01 — rilasciata

Strumenti web disponibili in ogni sessione che li consente, anche senza directory di lavoro e con le azioni del modello spente: prompt dedicato con solo `web_search` e `visit_page`, approvazioni e limite azioni invariati. Senza directory le pagine sono salvate in `~/Library/Application Support/SUC Studio/web`. Richieste d’azione accettate anche in un unico blocco di codice `json`.

## 1.4.5_Final — rilasciata

Browser web nascosto: niente finestra sopra il lavoro né furto del focus, temporizzazioni ds4 invariate (pannello fuori schermo non throttled, misurato). Pulsante globo nella barra della chat con fase, URL e secondi; apre il browser su richiesta. Verifica Google dal pulsante arancione, mai in primo piano. Nessuna finestra vuota a fine azione.
