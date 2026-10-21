# SUC Studio 1.4.8.03 — build 36

## Correzioni

- Compattazione: budget proporzionale allo storico (target 10% stimato, riserva doppia per il thinking), 2048–16384 token salvo finestre piccole. Un secondo tentativo può raddoppiare il budget fino a 32768 token, senza superare lo spazio prudenziale disponibile. Input suddiviso in blocchi UTF-8 fino a 24 KiB. Sintesi vuote, JSON di azioni o risposte con finish_reason=length vengono rifiutate; storico e checkpoint precedente restano intatti.
- Il profilo DS4 conserva chat_template_kwargs oltre a think, consentendo al router di inoltrare entrambi. Il backend può comunque ignorare la richiesta di disattivare il thinking.
- Routine: riserva minima di 4000 token di output, nei limiti del contesto; attesa di strumenti/compattazione prima del completamento. Thinking senza risposta finale produce errore visibile e non uno stato completato. I limiti manuali delle chat ordinarie restano impostabili.
- Chat lunghe: indicatore contesto calcolato su snapshot in background circa ogni secondo, invece di ricalcolare più volte tutto lo storico a ogni delta; caching limitato delle presentazioni storiche, righe invarianti escluse dal ridisegno e Markdown non rigenerato quando l’anteprima è invariata. Metriche live aggiornate al massimo circa cinque volte al secondo; testo, usage finale e motivo di arresto restano integri.

## Diagnosi e limiti

Le sessioni segnalate usavano Qwen 27B tramite LLM Router e LM Studio, non il motore SUC. La routine aveva max_tokens=512 e nessun testo finale: tutti i token erano consumati dal thinking. La compattazione precedente era limitata a 1024 token. Una prova innocua sul backend reale con budget 4096 ha restituito sintesi valida, finish_reason=stop, 166 token complessivi (132 di ragionamento) in circa 5,60 secondi. Questa verifica breve non misura le prestazioni su 100k token né garantisce la qualità semantica di ogni sintesi. Le conversazioni reali sono state consultate in sola lettura e non vengono incluse nei report.

Confronto indicativo diretto sul router, senza UI e senza riusare lo storico reale (stesso modello già caricato, due prompt sintetici, uscita limitata a 64 token):

| Token input dichiarati | Token output | Primo frammento | Intervallo ricezione output | Ricezione tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 269 | 44 | 1,775 s | 1,019 s | 42,19 |
| 78.341 | 56 | 453,484 s | 2,764 s | 19,90 |

Il log LM Studio conferma per il caso lungo 453,328 s di prefill (172,81 tok/s) e circa 19,89 tok/s di decode. Il prefill lungo è quindi effettivamente eseguito dal backend, mentre una velocità apparente molto inferiore durante la visualizzazione può aggiungere overhead client. Il confronto è una singola prova diagnostica, non un benchmark ripetuto o una valutazione della qualità delle risposte; il limite di 64 token serve soltanto a misurare la latenza. Nessuna risposta di questa prova viene usata come sintesi o salvata nelle chat.

Il costo di attenzione e la cache di un server esterno rimangono responsabilità del backend. Nessuna modifica al motore C++/Metal, ai modelli o alla memory policy; nessuna nuova misura KL.

## Verifica e installazione

Evidenze finali in RELEASE_1.4.8.03_QA.json. Installer Universal in SUCStudio/dist/SUC-Studio-1.4.8.03-universal.pkg; checksum nella stessa cartella. Chiudere l’app prima di installare.

Lo storico grafico usa pagine limitate con pulsanti per messaggi precedenti/successivi; tutti i messaggi rimangono salvati. I ragionamenti storici partono richiusi e sono espandibili. Il ragionamento in streaming si apre e non si richiude automaticamente a fine risposta, evitando salti di altezza.

## Hotfix build 36 — identità MCP e approvazioni web

La sessione segnalata usava il nome `rea`, mentre il catalogo conteneva `REA mcp`. Il confronto esatto provocava errori di connessione nonostante il server fosse disponibile. Discovery, schema e chiamata usano ora lo stesso resolver: nome esatto senza differenze di maiuscole/spazi esterni, UUID, alias univoco senza suffisso MCP e separatori normalizzati. Se più identità corrispondono, l’azione viene rifiutata con elenco dei nomi/UUID; nessun server disabilitato viene abilitato. Un nome sconosciuto viene distinto da un catalogo non disponibile.

La conferma web aggiuntiva era relativa all’avvio del browser: veniva forzata anche dopo l’approvazione automatica dell’azione, salvo per le routine. Ora la medesima modalità automatica vale anche nelle chat interattive con web abilitato. Conferme manuali, controlli sui tool disabilitati e gestione CAPTCHA restano separati.

Le sessioni sono consultate in sola lettura. I test usano strumenti echo e pagine localhost; nessun comando della sessione reale viene eseguito o esportato. Le misure UI/backend precedenti sono quelle raccolte nella build 35; questo hotfix non introduce nuovi benchmark di inferenza.
