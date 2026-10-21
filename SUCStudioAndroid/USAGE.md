# SUC Studio Android — setup e utilizzo

## Server remoto

In **Server** imposta la base API, per esempio `http://192.0.2.20:8080/v1` per SUC Studio in LAN oppure un URL HTTPS. `localhost` sul telefono indica il telefono stesso, non il Mac. Sul computer il server deve accettare connessioni LAN; verifica la relativa configurazione e l’accesso alla porta. Inserisci API key e ID modello; **Verifica / modelli** interroga `/models`.

HTTP richiede la casella **Consenti HTTP sulla rete fidata**: non cifra chiavi e conversazioni. Per Internet usa HTTPS. I redirect delle API sono rifiutati, quindi indica l’URL finale corretto. Per cambiare la chiave compilane una nuova; vuoto conserva quella salvata, la casella dedicata la elimina. Le chiavi sono cifrate con AES-GCM e una chiave Android Keystore, separate dall’archivio chat.

La compatibilità usa Chat Completions JSON/SSE. Non è prevista l’API Responses. Il server deve supportare i contenuti inviati: immagini richiedono vision e alcuni parametri sampling possono essere specifici del motore. Non vengono effettuati retry automatici delle generazioni, evitando richieste duplicate. Prefill lunghi non sono interrotti da un timeout breve; ogni 30 secondi una richiesta `/models` verifica la raggiungibilità, senza pretendere di dimostrare che la generazione sia avanzata. **Ferma** interrompe esplicitamente la richiesta.

## Conversazioni

Nuova chat crea una sessione locale. Cerca nell’elenco, usa **Opzioni chat** per titolo, system prompt aggiuntivo, export e cancellazione. Il prompt generale si imposta nelle Impostazioni e viene anteposto a quello di sessione. Modifica e Rigenera interrompono la richiesta precedente e mantengono la sessione, sostituendo la prosecuzione del dialogo dal punto interessato.

Il thinking è separato dalla risposta in un box leggero. Markdown include titoli, liste, link, citazioni, tabelle e blocchi di codice; HTML viene visualizzato come testo. I link HTTP/HTTPS si aprono nel browser del sistema. Tre backtick aprono l’editor di codice; puoi aprirlo anche con il pulsante `</>`. Il grip sopra il composer ne regola l’altezza. Ctrl/⌘ + Invio invia il messaggio.

La chat segue la generazione mentre rimani in fondo. Quando risali, la posizione di lettura resta ferma anche durante lo streaming e al termine della risposta. Tornando manualmente in fondo, oppure toccando la freccia verso il basso, riattivi lo scorrimento automatico. L’elenco delle sessioni ha uno scorrimento indipendente. Un pallino giallo segnala il risultato o un’approvazione di una chat diversa da quella selezionata.

## Allegati e artifact

Il pulsante **＋** apre il selettore file Android. Non servono autorizzazioni globali alla memoria. Limiti espliciti: 8 allegati per invio, 10 MiB per file, testo estratto entro 256 KiB. DOC 97+, DOCX e ODT sono letti localmente; PDF fino a 300 pagine devono contenere testo. Non è incluso OCR. File binari non decodificabili vengono rifiutati, mentre il testo UTF-8/UTF-16 è ammesso indipendentemente dall’estensione. PPTX/ODP e stringhe XLSX sono estratti come testo; non viene ricostruito il layout dei fogli di calcolo.

Le immagini sono ricodificate JPEG e ridotte per limitare memoria e upload. Il modello deve supportare vision; il nome di un modello non consente di verificarlo automaticamente. Gli allegati vengono inviati al server insieme alla richiesta. Tocca il nome prima dell’invio per vedere il testo estratto.

Puoi salvare codice e risposte dai relativi pulsanti. Il tool `publish_artifact(name,content)` permette inoltre al modello di produrre file testuali entro 1 MiB: compaiono come file salvabili alla fine della risposta, restano nell’area privata dell’app e sono inclusi negli export JSON Android. Non si tratta di esecuzione di codice o di file creati sul Mac.

## Skill, MCP e plugin

Le skill hanno titolo, descrizione e istruzioni. Abilita quelle desiderate: il prompt elenca solo le descrizioni, e il modello richiede `load_skill` per leggere il contenuto. Puoi importare SKILL.md o usare gli esempi dell’editor. Frontmatter semplice con `name` e `description` su una riga; YAML avanzato non è implementato.

Gli MCP supportano Streamable HTTP, risposte JSON/SSE, session ID e API key Bearer. Configura l’URL completo `/mcp`, verifica gli strumenti e abilita il server. Non sono supportati trasporto stdio, vecchio trasporto SSE a due endpoint, OAuth interattivo o header personalizzati. I risultati dei tool ritornano al modello come dati, mentre il JSON operativo resta nascosto dalla chat.

Le conferme per web/MCP e creazione artifact compaiono nella chat. Puoi scegliere approvazione automatica e limite di azioni consecutive nelle Impostazioni. Le skill di testo non richiedono una conferma aggiuntiva. Lo stato degli strumenti viene ricostruito per ogni richiesta; nessun processo MCP locale viene avviato.

Plugin portabile ZIP:

```text
plugin.json
skills/revisore/SKILL.md
skills/revisore/references/stile.md   # facoltativo
mcp.json                            # facoltativo
```

```json
{"name":"assistente-testi","version":"1.0.0","description":"Revisione e strumenti remoti"}
```

```json
{"mcpServers":{"ricerca":{"type":"http","url":"https://example.org/mcp"}}}
```

Skill e MCP possono coesistere. Plugin installati spenti; abilitarli aggiunge i componenti agli elenchi. Non sono eseguiti script né gestiti hook. Si rifiutano pacchetti con hook, stdio, app o overlay di altri runtime: il formato Android è un sottoinsieme portabile, non l’intero runtime desktop. Limiti ZIP: 8 MiB, 256 file, 64 skill e 16 MCP; riferimenti testuali delle skill entro 64 KiB. Le credenziali si configurano dopo l’importazione.

Marketplace HTTPS minimale:

```json
{"plugins":[{"name":"Assistente","url":"https://example.org/assistente.zip","sha256":"HASH_SHA256_DEL_FILE"}]}
```

L’app mostra l’elenco, scarica solo il pacchetto scelto e chiede conferma prima di installarlo. Il checksum è verificato quando presente. Non sono inclusi marketplace Git/npm del desktop.

## Ricerca web

Abilita **Ricerca web e lettura pagine**. Il modello può richiedere `web_search`/`google_search` o `visit_page`, indipendentemente dal server di inferenza. Dopo la conferma, un browser integrato a schermo intero carica la pagina Google o l’URL. Il pulsante **Torna alla chat** nasconde la pagina senza annullare: nella chat compare **Risolvi CAPTCHA / apri pagina** per riaprirla. Il controllo periodico riconosce anche una verifica completata senza nuova navigazione; durante l’attesa non c’è una scadenza automatica. **Usa questa pagina** non invia al modello una pagina CAPTCHA ancora attiva. **Annulla ricerca** o **Ferma** interrompono l’operazione. Puoi completare verifiche e cookie, annullare o usare **Usa questa pagina**. Se il sito non accetta la WebView, scegli **Browser esterno**, completa la verifica nel browser di sistema, copia il testo utile e torna all’app: **Incolla testo → Usa testo** riprende il lavoro del modello. I cookie e il contenuto del browser esterno non vengono importati automaticamente. Nessun CAPTCHA viene aggirato. Il tool restituisce URL, titolo, testo fino a 48.000 caratteri e collegamenti; non garantisce l’accesso a pagine protette o che ogni modello scelga di usare gli strumenti correttamente.

## Dati e limiti mobile

Le sessioni sono nell’area privata dell’app, con scrittura atomica. Il backup Android automatico è disabilitato; esporta le chat importanti. La disinstallazione elimina i dati locali. Import/export JSON riguarda il formato Android; gli export macOS non sono convertiti automaticamente.

La versione 0.1.1 mantiene sessioni e chiavi della 0.1.0: installa il nuovo APK come aggiornamento, senza disinstallare.

L’app non usa notifiche, routine o servizi permanenti. Una richiesta può continuare mentre l’Activity resta viva, ma Android può sospendere o terminare il processo in background. Al riavvio le risposte incomplete vengono segnate come interrotte e non reinviate. Non è garantito il completamento con telefono bloccato o app chiusa.

Le metriche usano valori forniti da `timings`, `metrics`, `telemetry` o usage compatibili; quando disponibili solo i conteggi finali, prefill e decode sono stime comprensive di overhead di rete/coda e indicate con `≈`. Nessuna velocità viene inventata quando i dati mancano.

## Build e test

```sh
./scripts/build.sh
./scripts/test.sh
./scripts/package.sh
```

`build.sh` genera la UI ES2017 offline e l’APK debug. `test.sh` esegue test JVM e DOM. `test-emulator.sh SERIAL PORTA_ADB` usa un emulatore già avviato, installa APK e runner, avvia solo una fixture localhost e verifica SSE, web, parser, Keystore e sessioni; rifiuta seriali che non iniziano con `emulator-`. Le schermate di QA si trovano in docs. Il collaudo con un server/modello reale e su dispositivi fisici resta necessario prima della distribuzione pubblica.

## 0.1.3 — cache e metriche del motore remoto

Con SUC Studio 1.4.7 il riuso del prefisso e i miglioramenti del motore funzionano sul server, senza attivazioni aggiuntive in Android. Il vantaggio dipende dal modello, dal campionamento e dalla corrispondenza del prefisso. Non serve scaricare nuovamente i modelli né cambiare i prompt.

Il prefill calcolato dai tempi SUC usa soltanto i token nuovi: `prompt_tokens - prompt_tokens_details.cached_tokens`. Per llama.cpp, `timings.prompt_n` è già il numero di token elaborati e non viene ridotto una seconda volta. Le velocità esplicite fornite dal server hanno precedenza. Se tutto il prompt è in cache e non c’è una velocità esplicita, compare `—`. Quando presenti, i token riusati vengono mostrati nella barra delle metriche. Le stime da tempo al primo token rimangono indicate con `≈` e includono gli effetti della rete.

Installa `dist/SUC-Studio-Android-0.1.3.apk` come aggiornamento, senza disinstallare, usando la stessa firma delle versioni precedenti. Sessioni, API key e impostazioni rimangono conservate. La compatibilità del protocollo con server senza il campo cache resta invariata.

## Android 0.1.4

Verifica / modelli sostituisce il modello salvato se non è più disponibile e seleziona automaticamente l'unico modello esposto dal server. Fra più modelli mantiene una selezione ancora valida. Premi Salva server per confermare la configurazione.

## 0.1.5 · interfaccia bilingue

Avvio in inglese; selezione English/Italiano nelle impostazioni, salvata in `settings.language` nello stesso archivio locale e riletta al caricamento. Gli archivi precedenti senza preferenza usano EN. Il cambio non traduce chat, codice o titoli salvati. APK: `dist/SUC-Studio-Android-0.1.5.apk`; codice versione 6. La distribuzione locale conserva la firma debug precedente.
