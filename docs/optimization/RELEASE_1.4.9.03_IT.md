# SUC Studio 1.4.9.03 · build 44

## Continuità dell’agente

Un controllo euristico conservativo IT/EN riconosce brevi risposte che promettono operazioni senza richiedere strumenti. Con strumenti disponibili invia un feedback: compiere il prossimo passo necessario già autorizzato oppure spiegare risultato, impedimento o decisione necessaria. Non sceglie né esegue comandi al posto del modello. Risposte finali normali, domande, richieste di conferma, citazioni e codice non attivano questo controllo. I job realmente attivi continuano a seguire il percorso di attesa esistente.

Le azioni `suc_action` riconoscibili ma malformate, come un `write_file` con la parentesi finale mancante, ricevono una richiesta di correzione. Il modello deve rigenerare una richiesta completa e valida: non vengono ricostruiti contenuti o argomenti mancanti (la sola tolleranza della chiusura esterna, introdotta nella build 41, è descritta sotto). Si conservano integralmente gli originali nello storico. Risposte terminate per limite di token e azioni oltre il limite del parser non vengono ritentate automaticamente.

I due percorsi condividono un massimo di due tentativi per turno, conteggiati nel limite azioni. I risultati degli strumenti non azzerano il contatore; un nuovo messaggio dell’utente sì. Al limite, il turno passa in errore e la coda viene sospesa, evitando di segnare una routine come completata. Permessi, hook, approvazioni e Stop restano applicati. Nessuna ripresa automatica degli archivi al riavvio. L’euristica può non riconoscere formulazioni diverse e non certifica il completamento del compito.

## Fase dello streaming

Le metriche che omettono i token di output o riportano zero non fanno retrocedere l’indicatore da decode/ragionamento a prefill. Le metriche di altre richieste non modificano la fase corrente. Ogni nuova richiesta riparte dalla propria attesa/prefill; non viene forzato decode prima di iniziare la generazione. I valori tok/s non vengono alterati per correggere l’icona.

## Verifica e distribuzione

Fixture HTTP in streaming, file e archivi temporanei: promesse seguite da lettura reale, JSON corretto dal modello seguito da un solo file Markdown e artifact, limite condiviso, conferme/Stop, FIFO e fasi con conteggi mancanti. Nessuna esecuzione dei comandi o produzione dei documenti della sessione utente. Esiti e checksum nel report `RELEASE_1.4.9.03_QA.json`.

Contesto predefinito 16000 token e valori salvati preservati. Interfaccia IT/EN, motore C++/Metal, memory policy e APK Android 0.1.5 invariati. Installer Universal locale non notarizzato; nessuna installazione automatica.

## Correzioni build 41

- I marcatori di protocollo `<tool_call>` su una riga propria appartengono all’involucro tecnico: non vengono mostrati nella chat, neanche durante la ricezione parziale. Sono supportati involucro aperto, chiuso e JSON recintato al suo interno. Commenti precedenti, ragionamento separato, archivio originale ed esempi espliciti in blocchi codice restano conservati. Il payload deve comunque essere un’unica azione SUC valida: nessuna conversione implicita di protocolli nativi diversi.
- La soglia di auto-compattazione usa l’input corrente/proiettato al 99%, senza aggiungere una riserva di output. Si usa la stessa base dell’indicatore, con l’ancora dei conteggi del server quando valida, più nuovo input e istruzioni aggiuntive. Non si preferisce una stima integrale in byte più alta del conteggio reale. Prima della soglia si riduce il budget della risposta allo spazio residuo; input nuovi o risultati di tool che raggiungono il 99% fanno partire la compattazione con un avviso del conteggio proiettato. `/compact` resta manuale.
- Solo dopo `finish_reason=stop` e risposta completa si tollera una singola `}` esterna mancante, quando tutti gli oggetti interni, stringhe e argomenti sono già completi e la validazione JSON/azione riesce. Non si modificano contenuti, non si completano stringhe, array o argomenti, non si accettano più documenti e non si applica la tolleranza a risposte in streaming o troncate per lunghezza. L’azione segue le normali approvazioni/validazioni. Questo caso strutturale evita di chiedere inutilmente al modello lo stesso documento già completo. Gli altri errori restano nel recupero limitato della build 40.

## Interfaccia build 42

Il pannello fisso «Comandi della sessione» è stato rimosso dal fondo della chat. I comandi restano nei riquadri della conversazione; la riga di intestazione è un singolo pulsante largo quanto il riquadro, con altezza minima di 36 punti e stato espanso/compresso accessibile. Si può cliccare sulla freccia, sul testo o nello spazio vuoto, senza doppie attivazioni. Il corpo del comando resta selezionabile e non viene usato come area di toggle.

La gestione dei processi non cambia: la rimozione del pannello non ferma comandi attivi, non cancella record e non modifica code, permessi o Stop. Gli output conclusi restano nel registro azioni della sessione; gli avvisi di errore indirizzano a quel registro. I terminali interattivi inline restano disponibili.

Il registro azioni ora usa schede con nome leggibile, icona/stato, ora e anteprima della richiesta; le ultime 30 azioni sono mostrate dalla più recente. La superficie scrollabile cresce a 240–380 punti, con testo del corpo e riquadri monospaziati selezionabili per richiesta e risultato. Ogni intestazione è cliccabile per intero. Per i job, i metadati JSON vengono resi come codice di uscita, durata e output. Le anteprime sono limitate a 16000 caratteri; copia e apertura integrale conservano il risultato completo archiviato. Le voci attive aperte mostrano un aggiornamento periodico dell’output. Il registro resta consultabile mentre i controlli di configurazione sono disabilitati durante il lavoro.

## Compattazione · build 43

Corretta la mescolanza tra byte UTF-8 e token: memoria precedente, istruzioni, blocchi di storico e spazio per la risposta sono ora conteggiati nella stessa unità stimata in token. La conversione in byte avviene soltanto per dividere il testo in blocchi Unicode integri. La soglia automatica resta al 99% e la finestra impostata non viene aumentata.

Se la memoria precedente non entra dopo una riduzione della finestra, viene ricompattata prima dei nuovi dati, senza tagli automatici; i tentativi di riduzione sono limitati. Il secondo tentativo di sintesi rispetta lo spazio residuo. Conteggi stimati e tokenizer del server possono differire: non si tratta di una tokenizzazione esatta universale.

Se la compattazione fallisce o viene fermata, il risultato dello strumento già completato viene aggiunto integralmente allo storico come normale risultato operativo, senza rieseguire il comando. Lo storico e la memoria precedente restano conservati. Si può riprovare `/compact`, poi continuare la chat; il fallimento non riprende automaticamente le azioni. Le priorità manuali troppo grandi possono ancora richiedere di ridurre le istruzioni o aumentare una finestra effettivamente supportata dal server.

## Contesto e thinking remoto · build 44

Il contesto futuro esclude il reasoning separato che non viene reinviato al modello. Se il server espone `completion_tokens_details.reasoning_tokens`, si usa il conteggio; negli archivi precedenti si stima soltanto il contenuto effettivamente reinviato, indicando il valore come stimato. Il reasoning incorporato nel testo resta conteggiato. Durante la generazione il reasoning continua a consumare la finestra e le metriche decode includono tutti i token prodotti. La soglia di auto-compattazione resta al 99%.

In Server remoto, con parametri estesi attivi, il nuovo Controllo thinking remoto permette Automatico per profilo, `chat_template_kwargs`, `reasoning_effort = none`, `think` oppure Predefinito del server. In automatico, LM Studio usa `reasoning_effort=none` quando il thinking è spento; Router aggiunge anche `think`, DS4 usa `think`. Gli override consentono di scegliere il parametro effettivamente supportato dal backend. I server che rifiutano il parametro restituiscono un errore senza reinvii automatici. Con estensioni disattivate non vengono aggiunti questi controlli.

La compattazione ricompatta anche la memoria precedente, richiede una sintesi breve (256–2048 token come obiettivo, distinto dal budget di generazione), poi elabora i nuovi dati. Se due tentativi non producono una sintesi completa, il blocco viene diviso in due metà Unicode esatte e riprovato, fino a due livelli e con un limite complessivo proporzionale ai blocchi iniziali. Nessuna sintesi troncata o solo thinking viene salvata. Gli errori riportano il budget finale, i conteggi disponibili e le dimensioni di risposta/ragionamento senza mostrarne il contenuto. Stop, storico, memoria precedente e risultati degli strumenti sono conservati come nella build 43. Non è garantito il successo di modelli che ignorano tutti i controlli o non producono mai una risposta finale.

La generazione registra il massimo output effettivamente richiesto. Se il contesto limita la risposta, l’avviso indica il budget reale e propone `/compact` o input più breve: alzare Output max non allarga la finestra. Nessuna compattazione anticipata viene aggiunta.
