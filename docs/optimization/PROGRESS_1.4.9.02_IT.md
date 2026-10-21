# SUC Studio 1.4.9.02 · comandi prolungati

- [x] Esecuzione non bloccante con identificativo, output limitato in memoria e stato reale.
- [x] Strumenti run_command, command_status, wait_command e stop_command; attesa distinta dalla durata del processo.
- [x] Isolamento per sessione, approvazioni all’avvio, stop manuale, chiusura e protezione della coda.
- [x] Pannello in chat e testi IT/EN; protocollo aggiornato per il modello.
- [x] Contesto predefinito 16000 token, con limiti già salvati o rilevati preservati.
- [x] Test con processi reali oltre 30 secondi, polling, output, cancellazione e sessioni separate.
- [x] Build Universal, installer, guida, changelog e QA.
- [x] Commit e sincronizzazione sorgenti/Mac Studio/export pubblico.

Motore C++/Metal, memory policy e APK Android invariati. Nessuna installazione automatica.

## Verifiche completate

- Comando reale di 31 secondi concluso senza riavvio; scadenza dell’attesa e annullamento di una lettura non terminano il processo.
- Separazione tra sessioni, blocco dei duplicati, output oltre 256 KiB con coda e offset corretti, stderr e codice di uscita 7 verificati.
- Durata totale esplicita, cancellazione di figli che ignorano SIGTERM, validazione degli argomenti e assenza di riavvii automatici.
- Approvazione iniziale manuale; due consultazioni oltre il limite di una sola azione senza nuove approvazioni.
- Stop, coda sospesa, aggiornamento dei record e recupero dell’archivio dopo riavvio. Risposte premature del modello seguite da attesa reale.
- Contesto 16000 predefinito; valore salvato 4096, limite di sessione 8192 e limite API 32000 preservati secondo precedenza.
- Suite di regressione coda, compattazione/contesto e localizzazione superate. Nessun modello reale caricato.

Controllo aggiuntivo dell’output UTF-8: caratteri multibyte tra letture e ai bordi della coda, senza spezzare le sequenze complete.

Collaudo nativo sul Mac Studio M1 Ultra superato: stessa suite di processi reali, compreso il comando di 31 secondi. Bundle della dist aggiornati senza installazione.

## Correzione build 39

- [x] Diagnosi in sola lettura della sessione attiva: JSON completo con argomenti numerici rifiutato dal decoder di stringhe.
- [x] Normalizzazione dei numeri JSON nel parser comune; validazioni e approvazioni invariate.
- [x] Regressione Intel: formato osservato, parametri attesa/scadenza/offset, rifiuto di booleani/null/oggetti e azioni multiple; esecuzione reale fixture e approvazione.
- [x] Collaudo M1, installer Universal build 39 e sincronizzazione sorgenti/pubblico.

Non sono stati modificati gli archivi o rilanciati i comandi della sessione dell’utente.
