# SUC Studio 1.4.7.05 — build 32

Candidate Universal, senza installazione automatica. MTP rimane accantonato nello stash.

## Caricamento del modello

Durante il caricamento la barra globale mostra un indicatore di attività e la fase comunicata dal motore, fino alla conferma di modello pronto. I GiB delle pagine del file residenti e l'allocazione Metal sono dati separati: possono aumentare o diminuire e non rappresentano una percentuale di completamento. Il dettaglio indica la residenza iniziale. Il monitor non forza letture, page-in o variazioni della politica di memoria.

Il monitor non legge i pesi e non forza page-in, SSD streaming o variazioni della memory policy. Usa VM_REGION_EXTENDED_INFO su una mappatura read-only e attende 1–5 secondi fra osservazioni, adattandosi al costo. Nel controllo M1 lo stesso conteggio di mincore è ottenuto in 0,182 contro 1,811 secondi. Nel collaudo finale le cinque misure per load costano mediamente 0,171–0,210 s ciascuna: la frequenza ridotta limita il lavoro del monitor. Non attribuire la differenza di tempo totale fra load con cache diverse al solo monitor.

## Prefill automatico

Settings → Prefill → **Automatico (1024 / 64)**; ricaricare il modello dopo la selezione. Il primo prefill da zero usa blocchi completi da 1024, poi blocchi al massimo da 64. Coda e suffissi con cache usano sempre al massimo 64; il checkpoint resta allineato a 64 e vicino alla fine del prompt. Il decode rimane a un token per passo. Si rispettano exact/token, chunk espliciti e streaming expert, che mantengono il proprio percorso.

Batch prudente resta predefinito. Il nuovo profilo cambia i confini dei calcoli e può cambiare gli arrotondamenti. Il ritorno a 64 non annulla differenze precedenti. Il confronto completo hard/KL resta da eseguire prima di promuoverlo a default.

### Confronto reale sul M1 Ultra

Qwen3.8 Flash-Next side8 locale, modalità residente forzata e confermata dal piano memoria, contesto 16.384, nessun altro modello caricato. Stesso helper del pacchetto per entrambi i profili, processi sequenziali, prompt sintetici identici e al massimo 16 token di risposta. Un passaggio per profilo; i primi cold possono includere compilazione delle pipeline. Risposte brevi coincidenti nei sei casi.

| Caso | Prudente | Automatico |
|---|---:|---:|
| Prefill completo, 2.032 token | 17,614 s | 12,432 s |
| Prefill completo, 8.032 token | 67,673 s | 28,834 s |
| Nuovo turno con cache, 127 token nuovi | 1,049 s | 1,096 s |
| Nuovo turno con cache, 111 token nuovi | 0,994 s | 0,996 s |

A 8k la prova osserva circa **2,35×** nel prefill (118,7→278,6 token/s), senza ampliare la coda rielaborata dei turni caldi. Non è una media di benchmark ripetuti né un risultato KL/hard. Nessun aggiornamento alle metriche delle model card sulla base di questo controllo. [Evidenze complete](STUDIO_14705_QWEN_PREFILL.json).

## Android 0.1.4

Verifica / modelli sostituisce un ID non più disponibile e seleziona il modello se unico, anche con il campo già compilato. Fra più modelli mantiene una selezione valida. Salva server conferma la configurazione. APK build 5 con la stessa firma debug della 0.1.3; suite UI, web challenge e scrolling superate.

## Verifiche

- Test CPU di scheduler/checkpoint/prefisso e osservazione della memoria superati su Intel e M1.
- Parser Swift della telemetria e suite Universal su Intel superati; interfaccia/helper Universal arm64+x86_64, motore arm64, verificati tramite firme e architetture.
- Server standalone verificato in compilazione ARM64, senza nuova qualifica runtime separata.
- Controllo reale del caricamento, progressi, sei dialoghi sintetici per profilo, cache e rilascio del modello completato sul helper impacchettato.
- Controllo hard circoscritto, primo caso per fonte della suite originale: **5/6 corretti in entrambi**, tutte le sei risposte estratte coincidono. I cinque casi brevi hanno testo identico; nel caso lungo cambia il testo, ma la risposta resta corretta. L’errore GSM8K è presente in entrambi. Stessi token, budget, renderer e grader; non è il punteggio su 120 casi né una misura KL. [Evidenza](STUDIO_14705_HARD_SMOKE.json).

Installer: `SUCStudio/dist/SUC-Studio-1.4.7.05-universal.pkg`. Android: `SUCStudioAndroid/dist/SUC-Studio-Android-0.1.4.apk`.


## Passaggi intermedi e dettagli dei comandi

Corretta la presentazione delle risposte che terminano con una richiesta di strumento: premesse e ragionamenti restano nel transcript mentre il solo envelope JSON è nascosto. I dati originali del messaggio e la cronologia inviata al modello sono preservati. La correzione si applica anche alle sessioni già salvate senza migrazioni o riscritture. Comandi shell e interattivi consultabili in dropdown chiusi di default; nuove azioni conservano il comando aggiornato dagli hook. I risultati tecnici restano nel registro e gli input riservati non vengono copiati nei dettagli dei comandi. Versione 1.4.7.05 build 32 invariata.


## Cataloghi MCP con schemi estesi

Rimosso il limite di 128 KiB applicato alla rappresentazione indentata dello schema. Il client conserva ora JSON compatto completo, fino a 3 MiB per strumento, senza eliminare descrizioni o proprietà. Rimane il limite di 256 strumenti per server; i nuovi limiti dei risultati sono descritti sotto. Gli errori distinguono un nome eccessivo da uno schema eccessivo e riportano nome dello strumento e byte dello schema. Test stdio e HTTP su schemi sopra 128 KiB e rifiuto sopra 3 MiB; verifica aggiuntiva su REA locale tramite il client Swift, senza tools/call. Versione e build restano invariate.


### Pannello artefatti

Gli artefatti non compaiono più in fondo alla chat. L’icona a forma di pacco nella barra della chat, con il numero dei file disponibili, apre e chiude il pannello laterale. Il pulsante × nel pannello lo nasconde completamente. Il pannello segue la sessione selezionata, è ridimensionabile e mantiene i pulsanti Salva e Finder. Rimane chiuso finché non lo apri, anche quando viene prodotto un nuovo file.


### MCP su richiesta e continuità delle skill

Il prompt iniziale contiene i server MCP connessi e il numero di strumenti, senza copiarne tutti gli schemi. Il modello cerca gli strumenti con mcp_list (query e paginazione), consulta mcp_schema e poi usa mcp_call con le normali approvazioni. Gli schemi grandi si consultano tramite path JSON Pointer e paginazione: lo schema completo resta conservato nel pannello MCP. Le skill richiamate con @nome o $nome rimangono attive durante tutte le azioni dello stesso turno. La lettura del catalogo non esegue strumenti remoti.

Il health check prova anche /health se il percorso sotto /v1 non è disponibile. Un server raggiungibile non dimostra che stia elaborando: lo stato indica l’attesa della risposta, senza simulare avanzamento. Il pulsante × degli artefatti è un controllo AppKit con binding diretto alla chat e area di clic di 28 pt. Il pannello laterale usa una vista divisa ridimensionabile; nelle finestre strette l’elenco delle sessioni viene nascosto temporaneamente per lasciare spazio alla conversazione.


### Risultati MCP estesi

Risultati fino a 4 MiB di testo UTF-8 complessivo, senza troncamenti. Trasporto MCP fino a 32 MiB per escaping JSON e metadati; richieste chat fino a 32 MiB anche dopo hook e nel server integrato. Pipe del helper fino a 64 MiB per includere il comando. Nessuna modifica al motore padre, ai kernel o alle memory policy. Rimangono il limite di input utente, degli allegati e il contesto del modello. Versione 1.4.7.05 build 32 invariata.

Il parser delle risposte HTTP MCP lavora fuori dal MainActor: evita un passaggio al thread UI per ogni byte. Credenziali, sessioni e risposte ai ping rimangono gestite dalla connessione; limiti di trasporto e cancellazione conservati.
