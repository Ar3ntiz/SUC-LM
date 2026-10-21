# SUC Studio 1.4.9.03 · build 40

## Continuità dell’agente

Un controllo euristico conservativo IT/EN riconosce brevi risposte che promettono operazioni senza richiedere strumenti. Con strumenti disponibili invia un feedback: compiere il prossimo passo necessario già autorizzato oppure spiegare risultato, impedimento o decisione necessaria. Non sceglie né esegue comandi al posto del modello. Risposte finali normali, domande, richieste di conferma, citazioni e codice non attivano questo controllo. I job realmente attivi continuano a seguire il percorso di attesa esistente.

Le azioni `suc_action` riconoscibili ma malformate, come un `write_file` con la parentesi finale mancante, ricevono una richiesta di correzione. Il modello deve rigenerare una richiesta completa e valida: nessun JSON viene ricostruito ed eseguito automaticamente. Si conservano integralmente gli originali nello storico. Risposte terminate per limite di token e azioni oltre il limite del parser non vengono ritentate automaticamente.

I due percorsi condividono un massimo di due tentativi per turno, conteggiati nel limite azioni. I risultati degli strumenti non azzerano il contatore; un nuovo messaggio dell’utente sì. Al limite, il turno passa in errore e la coda viene sospesa, evitando di segnare una routine come completata. Permessi, hook, approvazioni e Stop restano applicati. Nessuna ripresa automatica degli archivi al riavvio. L’euristica può non riconoscere formulazioni diverse e non certifica il completamento del compito.

## Fase dello streaming

Le metriche che omettono i token di output o riportano zero non fanno retrocedere l’indicatore da decode/ragionamento a prefill. Le metriche di altre richieste non modificano la fase corrente. Ogni nuova richiesta riparte dalla propria attesa/prefill; non viene forzato decode prima di iniziare la generazione. I valori tok/s non vengono alterati per correggere l’icona.

## Verifica e distribuzione

Fixture HTTP in streaming, file e archivi temporanei: promesse seguite da lettura reale, JSON corretto dal modello seguito da un solo file Markdown e artifact, limite condiviso, conferme/Stop, FIFO e fasi con conteggi mancanti. Nessuna esecuzione dei comandi o produzione dei documenti della sessione utente. Esiti e checksum nel report `RELEASE_1.4.9.03_QA.json`.

Contesto predefinito 16000 token e valori salvati preservati. Interfaccia IT/EN, motore C++/Metal, memory policy e APK Android 0.1.5 invariati. Installer Universal locale non notarizzato; nessuna installazione automatica.
