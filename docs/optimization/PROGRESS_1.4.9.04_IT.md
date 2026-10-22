# Avanzamento SUC Studio 1.4.9.04 · build 45

- [x] Analizzare gli errori MCP della sessione: Apple Events/Hopper, configurazione Ghidra e scadenza MCP a 30 secondi.
- [x] Aggiungere dichiarazione Apple Events IT/EN e entitlement di automazione alla firma dell’app.
- [x] Separare la durata delle chiamate MCP dall’attesa del modello, con job isolati per sessione, stato, attesa e annullamento.
- [x] Verificare trasporti stdio e HTTP, chiamate oltre 30 secondi, scadenze esplicite, annullamento, permessi e coda.
- [x] Eseguire le regressioni dell’harness e la verifica dell’identità della release.
- [x] Compilare e verificare app Universal e pacchetto 1.4.9.04.
- [x] Aggiornare guide IT/EN, changelog e rapporto QA con risultati effettivi.
- [x] Committare e sincronizzare sorgenti, Mac Studio e repository pubblico.

Nessuna modifica a motore, memory policy, modelli, stash MTP o Android. Nessuna installazione automatica. Le autorizzazioni TCC e la firma del processo Hopper rimangono sotto il controllo di macOS e del fornitore; la dichiarazione dell’app non può correggere un eseguibile esterno non autorizzato.
