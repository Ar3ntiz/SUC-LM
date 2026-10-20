# SUC Studio 1.4.8 · build 33

## Coda dei messaggi per sessione

Durante prefill, decode o esecuzione degli strumenti si può continuare a scrivere e inviare. Il messaggio con i suoi allegati entra in una coda separata dallo storico: il modello lo vede soltanto quando arriva il suo turno. Alla conclusione dell’intera risposta, comprese le azioni e gli hook, parte il primo messaggio in coda. La coda segue la sessione anche quando si cambia chat; le bozze non inviate restano intatte.

Un pannello sopra il compositore mostra i messaggi in ordine, con pausa/ripresa, rimozione e ritorno in bozza. Lo stop della generazione o un errore di generazione mette in pausa i messaggi rimanenti. La coda è salvata nell’archivio locale; dopo un riavvio si riprende solo dal pulsante dedicato. La duplicazione della chat non duplica richieste in coda.

Il motore padre C++/Metal, la memory policy e i formati dei modelli restano invariati. Universal: inferenza locale Apple Silicon e client remoto Intel/Apple Silicon.

## Verifiche

- Fixture HTTP/SSE: ordine FIFO, invio durante lo streaming e risposta precedente presente nel contesto successivo.
- Una skill viene caricata e la risposta finale completata prima del turno seguente.
- Allegati congelati all’invio, bozza corrente e altra sessione preservate.
- Stop ed errore: pausa senza perdita dei messaggi, ripresa esplicita.
- Riavvio: coda conservata in pausa; ritorno in bozza senza sovrascrivere testo esistente.
- Build, pacchetto e verifica delle firme/architetture registrati nel report QA della release.

Corretto anche un crash del client remoto quando Stop precede l’avvio della richiesta HTTP: il client rifiuta nuove richieste dopo lo stop e cancella i task senza invalidare prematuramente la sessione URLSession.
