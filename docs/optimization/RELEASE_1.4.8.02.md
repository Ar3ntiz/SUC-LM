# SUC Studio 1.4.8.02 · build 34

## Contesto

Indicatore circolare nella chat con percentuale e dettagli dei token usati. Il prefisso ≈ distingue le stime dai conteggi effettivi del server. Il limite locale è quello caricato nel motore; quello remoto viene letto da /v1/models quando presente (max_model_len, context_length, context_window, max_context_length). Il popover o /context TOKEN consentono un limite per sessione; 0 ripristina il rilevamento. Senza limite noto non viene inventata una percentuale e l’autocompact resta sospeso. I costi delle immagini restano stimati finché il server non li riporta.

/compact [priorità] produce una sintesi incrementale usando il modello corrente. Lo storico originale e gli allegati restano nell’archivio e nel transcript; soltanto il contesto delle richieste successive usa la memoria sintetica. La sintesi è una trasformazione del modello e può omettere dettagli: l’originale rimane consultabile ed esportabile. I dati vengono suddivisi per non inviare l’intera chat già piena al riassuntore. L’operazione non offre strumenti e non esegue comandi contenuti nei dati. Backend e modello restano fissati durante la compattazione. Fallimenti e cancellazioni non sostituiscono il contesto precedente.

Auto-compact è attivo di default al 99% del contesto disponibile. Si esegue fra le generazioni: una riserva per l’output può anticiparlo prima dell’invio; i conteggi finali possono avviarlo alla fine del turno. Non interrompe una risposta in decode. Con tokenizzazione non esposta dal server la decisione usa una stima dichiarata. Il massimo output richiesto rimane un tetto e può essere ridotto allo spazio disponibile. I messaggi in coda ripartono solo dopo la sintesi completata. Stop/errori mantengono la coda in pausa.

## Comandi e plugin

/help elenca /compact, /context, /auto, /status, /new, /clear, /stop, /queue, /pause, /resume, /skills, /skill, /plugins, /plugin, /model, /workspace, /export e /retry. Sono comandi dell’harness e non testo da eseguire dal modello. // invia un messaggio letterale che inizia con /. /clear riparte con un nuovo contesto conservando la cronologia. I comandi con allegati richiedono di inviarli separatamente.

/plugin IDENTIFICATORE richiesta e @plugin:IDENTIFICATORE richiesta richiamano un plugin già attivo. Il menu skill include anche i plugin. Il modello vede titoli e descrizioni e può caricare i componenti pertinenti con load_plugin. Le skill solo esplicite restano escluse dal caricamento autonomo. Nessuna attivazione implicita di plugin disabilitati, MCP scollegati o hook non revisionati. Le approvazioni restano quelle della sessione.

## Chat lunghe

Rimosso il collasso da Markdown a 12 righe quando il testo supera 8000 caratteri. La stessa vista e la stessa anteprima restano montate; il testo completo si apre dal pulsante dedicato. Il transcript usa una finestra iniziale di 80 messaggi con caricamento progressivo dei precedenti, senza perdere dati. Durante lo scroll manuale l’indice iniziale rimane fermo; solo seguendo il fondo può essere ridotta la finestra fuori schermo. Le risposte ai comandi non sostituiscono le metriche della generazione in corso.

## Verifica

Test deterministici con vere richieste HTTP/SSE per compattazione manuale/automatica, soglia 99%, ripresa della coda, conservazione dell’archivio, conteggi reali, rollback da errore/stop, comandi e invocazione esplicita/autonoma dei plugin. Test grafico dedicato alla stabilità dell’altezza oltre 8000 caratteri. I report QA distinguono queste prove dai benchmark del motore, che non sono stati ripetuti: C++/Metal, formati modello e memory policy sono invariati.
