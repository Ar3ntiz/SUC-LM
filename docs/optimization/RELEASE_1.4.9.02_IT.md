# SUC Studio 1.4.9.02 · build 38

## Comandi prolungati

I comandi shell dell’harness diventano job gestiti per sessione. La scadenza dell’attesa non termina il processo: il modello riceve un identificativo e lo stato effettivo, poi può attendere, leggere altro output o interrompere il comando. La vecchia scadenza fissa di 30 secondi è rimossa.

- `run_command(command, wait_seconds?, timeout_seconds?)`: avvia il comando nella directory di lavoro. Attesa iniziale predefinita 1 secondo, impostabile tra 0 e 60. `timeout_seconds`, se presente, impone una durata totale tra 1 e 86400 secondi; se omesso non c’è una durata totale fissa.
- `command_status(job_id, output_offset?)`: legge stato e output senza attendere.
- `wait_command(job_id, wait_seconds?, output_offset?)`: attende fino a 30 secondi per default, massimo 60, senza terminare o riavviare il processo.
- `stop_command(job_id)`: interrompe il comando e il gruppo di processi avviato dall’helper.

Gli argomenti rimangono stringhe come nel protocollo esistente. La risposta JSON contiene `job_id`, `state`, `elapsed_seconds`, `exit_code` quando disponibile e output incrementale. Gli stati sono `running`, `completed`, `failed`, `cancelled`, `timed_out`. I risultati tecnici vengono passati al modello senza stamparli come JSON nella chat.

La lettura successiva riparte automaticamente dall’ultimo offset consegnato; `output_offset` permette di richiedere un intervallo ancora conservato. Si conserva una coda di 256 KiB per job, con massimo 64 KiB per risposta. `omitted_bytes`, `total_output_bytes` e `output_remaining_bytes` rendono esplicite omissioni e dati ancora leggibili. Un output molto lungo non termina il processo: i dati precedenti non conservati non sono recuperabili dall’harness; per log integrali il modello deve scrivere su file.

Una sola esecuzione contemporanea per sessione, massimo 8 complessive; si mantengono fino a 32 job, eliminando prima quelli conclusi. Le consultazioni di job della stessa sessione non richiedono nuove approvazioni e non consumano il limite delle azioni consecutive. L’avvio richiede sempre directory, azioni e shell abilitate e segue la modalità di approvazione; gli hook restano applicati. Identificativi di altre sessioni non sono accessibili.

Se il modello produce una risposta finale mentre un comando è ancora attivo, l’harness attende e gli restituisce lo stato reale per proseguire la verifica.

Il pannello «Comandi della sessione» mostra stato, tempo, comando e output espandibile. Interrompi ferma il turno e i job della sessione. La coda attende i comandi attivi. La chiusura normale dell’app li interrompe prima di salvare l’archivio. I job non vengono ripresi o ricollegati dopo un riavvio: una chiusura forzata può lasciare effetti già prodotti e richiede di verificare lo stato reale prima di ripetere il lavoro. I comandi interattivi continuano a usare la scheda terminale dedicata.

## Contesto predefinito

Nuove configurazioni locali: 16000 token. I valori già salvati restano invariati. Per i server remoti hanno precedenza il limite della sessione e quello rilevato dall’API; se mancano entrambi si usa un limite assunto di 16000 token, indicato nel pannello del contesto e modificabile dall’utente. Il valore non aumenta la finestra realmente supportata dal server.

## Ambito e verifiche

Interfaccia IT/EN conservata. Motore C++/Metal e memory policy invariati. Nessuna modifica all’APK Android 0.1.5. I test, i checksum e le verifiche dei pacchetti sono registrati in `RELEASE_1.4.9.02_QA.json` e nel piano `PROGRESS_1.4.9.02_IT.md` al completamento.
