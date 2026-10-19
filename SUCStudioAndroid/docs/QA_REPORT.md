# QA — SUC Studio Android 0.1.1

Data: 18 ottobre 2026. Build su Mac Intel con JDK 11, SDK Android 34, Gradle 7.6.3 / AGP 7.4.2. APK senza librerie native legate a una singola architettura.

## Verifiche automatiche

- 7 test JVM superati: endpoint e policy HTTP; SSE multilinea/CRLF/[DONE]; thinking/usage/metriche SUC Studio, LM Studio e llama.cpp; JSON azioni nascosto per tutti i prefissi di streaming e rifiuto di ambiguità; prompt con immagini e documenti; plugin combinati e limiti ZIP; trasporto HTTP/MCP reale, session ID, SSE e redirect rifiutati.
- Test DOM superati: Markdown, tabelle, codice salvabile, HTML/XSS non eseguito, thinking separato, approvazioni inline, modifica nella stessa chat, esempi e asset ES2017.
- Emulatore Android 11/API 30 x86_64 isolato, con fixture localhost e nessun modello esterno: SSE reale fino alla UI, azione web → conferma inline → WebView della pagina → risultato al modello → risposta finale. JSON operativo non visibile.
- Sul runtime Android superati modifica/rigenerazione senza creare sessioni, cancellazione, metriche reali della fixture, creazione artifact testuale e lista sessioni scrollabile dopo 35 nuove chat.
- Sul runtime Android superati DOC (fixture Apache POI), DOCX, ODT, PDF testuale, file testuale con estensione arbitraria e immagini JPEG/base64. Nessuna API key in chiaro nell’archivio; round-trip Keystore verificato.
- Schermata telefono verificata visivamente; compatibilità flex gap aggiunta per le WebView meno recenti. Layout tablet verificato su viewport emulata; schermate in questa cartella.

Log: EMULATOR_TEST_RESULTS.txt, TEST_RESULTS.txt. Il server usato è una fixture di test e non misura prestazioni di inferenza. Non è stato eseguito un collaudo su smartphone/tablet fisici o su tutti i server/modelli nominati. Questi test restano da eseguire prima della distribuzione pubblica.

## Distribuzione

APK di collaudo locale, firmato con chiave debug della macchina di build. Firma e manifest verificati; SHA256 in dist. Build release con chiave propria supportata dagli script, non prodotta in questa consegna. Nessuna chiave privata è inclusa nei sorgenti.

## Limiti intenzionali

Niente server/inferenza locale, stdio, hook, terminale, routine, notifiche o servizi permanenti. Background soggetto al ciclo di vita Android. Plugin portabili e marketplace JSON sono un sottoinsieme documentato del desktop. OCR, OAuth MCP e sincronizzazione macOS non sono inclusi. I tool dipendono dalla capacità del modello di seguire il protocollo.

SHA256 APK 0.1.0 archiviato: `3debf2ca435fcfe4795378b618d1149889e6e34805f092555183b736c0141580`. Firma APK debug verificata con apksigner; min SDK 26, target SDK 34. Nessuna libreria nativa o permission per notifiche/servizi foreground nel pacchetto.

## Correzione CAPTCHA 0.1.1 — 18 ottobre 2026

Il browser web usa ora una finestra a schermo intero con altezza assegnata, senza il dialog WRAP_CONTENT con WebView a peso zero. Tornare alla chat nasconde la finestra mantenendo la pagina; il banner della richiesta la riapre. Un controllo ogni secondo rileva anche verifiche completate tramite aggiornamenti DOM, senza attendere onPageFinished. Il CAPTCHA non è acquisibile come risultato del tool. Il timeout fisso di 180 secondi è rimosso: restano Annulla ricerca e Ferma.

Fallback esplicito: Browser esterno, poi Incolla testo. Non si promette condivisione di cookie né acquisizione automatica dal browser esterno. Alcuni provider possono rifiutare una WebView; la verifica reale dipende dal sito. Le pagine web non hanno accesso al JavascriptInterface della chat.

Test JVM: 8 superati. Test DOM: banner riapribile, rilevazione della verifica visibile, falso positivo su articoli che parlano di CAPTCHA, iframe nascosti e aggiornamento DOM dopo una verifica simulata. Il collaudo Android usa esclusivamente una pagina localhost che simula la verifica: nessun CAPTCHA reale viene automatizzato o aggirato.

Colllaudo 0.1.1 completato: viewport CAPTCHA effettivamente visibile e interattiva, ritorno alla chat e riapertura, verifica simulata con aggiornamento DOM senza navigazione, nuova risposta del modello verificata. Anche Incolla testo restituisce il contenuto manuale al tool e il modello completa la risposta. Regressioni chat/allegati/Keystore superate. Schermata: QA_CAPTCHA.png.
SHA256 APK 0.1.1: `2385a763debdd85a6d1d5ea51e93a521a1f6a2b8b7042fae2070d503a0f98bb1`. Firma identica alla 0.1.0, versionCode 2.

## Scorrimento 0.1.2 — 18 ottobre 2026

Eliminato il riaggancio automatico causato da eventi scroll di layout e frame già accodati. I gesti verso l’alto sospendono immediatamente il seguito; solo un movimento utente verso il fondo o la freccia lo riattivano. Un semplice tap non lo disabilita. Durante aggiornamenti completi e delta viene conservata l’ancora di lettura; l’apertura del thinking viene mantenuta anche a risposta conclusa.

Test DOM deterministici superati: frame accodati, touch/tap, wheel, streaming sospeso, eventi programmatici, riaggancio manuale, temporanea riduzione del contenuto, snapshot finale, handler delta reale, resize e cambio sessione. Test UI/CAPTCHA e 8 test JVM superati. Build APK riuscita, asset incluso verificato, versionCode 3 e stessa firma debug delle versioni precedenti. Questa correzione non è stata collaudata nuovamente su dispositivo fisico o emulatore.

SHA256 APK 0.1.2: `19f478abb1e7dbc1af74ffbf463ff40c9454e4f49c0c2eeae347c2caffe57a3f`.

## 0.1.3 — metriche cache, 19 ottobre 2026

- 13 test JVM superati, nessun errore o test saltato. Cinque nuove regressioni coprono usage/tempi in frame SSE distinti, cached_tokens prima del conteggio totale, campi parziali, token totali/nuovi, cache completa, contatori fuori intervallo, prompt_n di llama.cpp già netto e precedenza delle velocità esplicite. I server senza il campo cache conservano il comportamento precedente.
- Tre script DOM superati: UI, estrazione web/CAPTCHA e scorrimento. La barra mostra i token riusati, la velocità corretta e `—` con cache completa senza una misura esplicita; non inventa una stima quando manca il lavoro di prefill.
- APK 0.1.3 compilato, versionCode 4, min SDK 26 e target SDK 34 verificati. apksigner conferma la stessa firma debug della 0.1.2, consentendo l’aggiornamento senza disinstallazione. Hash nel file dist/SHA256SUMS.txt.
- Nessuna modifica al protocollo delle richieste, al motore C++ o all’archivio delle chat. Nessun nuovo collaudo su smartphone/tablet fisico, emulatore o server con modello reale è dichiarato per questa versione. Le verifiche storiche sopra restano riferite alle rispettive build.
