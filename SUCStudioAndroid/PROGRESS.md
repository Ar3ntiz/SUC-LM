# SUC Studio Android — avanzamento

Versione iniziale 0.1.0. Client remoto per smartphone e tablet, senza motore locale, MCP stdio, hook shell, terminale, routine o notifiche.

- [x] A1 — Verifica ambiente Android e definizione architettura separata dal progetto macOS.
- [x] A2 — Progetto compilabile, identità Android, icona, UI adattiva Chat/Server/Impostazioni.
- [x] A3 — Sessioni persistenti, streaming remoto, stop/edit/reload, thinking, Markdown, metriche.
- [x] A4 — Allegati immagini/documenti, import/export chat e salvataggio artifact.
- [x] A5 — Prompt, skill progressive, MCP HTTP, plugin compatibili e strumenti web con approvazioni inline.
- [x] A6 — Test automatici di protocollo/persistenza/sicurezza e collaudo su emulatore.
- [x] A7 — APK, guida setup/utilizzo, changelog e consegna nei sorgenti locali.

Le sessioni Android sono locali al dispositivo; nessuna sincronizzazione automatica con il Mac. In background Android può sospendere il processo: nessun servizio permanente o notifica. Gli strumenti operano sul telefono, salvo MCP remoto esplicitamente configurato.

## 0.1.1 — ricerca web e CAPTCHA

- [x] B1 — Diagnosi: dialog con WebView a peso/altezza ambigui, verifica solo a fine navigazione, nessun comando per riaprire e scadenza a 180 secondi.
- [x] B2 — Browser a schermo intero, riapertura dalla chat, rilevazione dinamica e attesa della verifica senza timeout.
- [x] B3 — Browser esterno e incolla testo come alternativa esplicita, senza trasferimento automatico dei cookie.
- [x] B4 — Test regressione CAPTCHA interattivo, build e consegna 0.1.1.

## 0.1.2 — scorrimento durante lo streaming

- [x] C1 — Diagnosi dei frame accodati e riaggancio causato da eventi di layout.
- [x] C2 — Sospensione con gesto verso l’alto, riaggancio solo tornando in fondo o con freccia, conservazione dell’ancora e del thinking.
- [x] C3 — Test regressione, build APK e consegna ai sorgenti locali.

## 0.1.3 — conteggio cache e prefill

- [x] D1 — Distinguere token totali, riusati e nuovi; gestire usage/tempi in frame SSE separati.
- [x] D2 — Conservare le velocità esplicite, trattare prompt_n di llama.cpp come token già elaborati, evitare velocità fittizie con cache completa.
- [x] D3 — Mostrare i token in cache e aggiungere regressioni JVM/DOM per conteggi, priorità e compatibilità.
- [x] D4 — Eseguire test, produrre e verificare APK 0.1.3, aggiornare guida/changelog e sincronizzare nei sorgenti.

Verifiche 0.1.3: 13 test JVM e tre script DOM superati; compilazione APK riuscita, manifest/versionCode 4 e min SDK 26 verificati. Firma debug identica alla 0.1.2 e SHA256 verificato. Nessun nuovo collaudo su dispositivo fisico o benchmark con modello reale; per questa correzione la verifica copre il parser SSE, il calcolo delle metriche e la UI con dati di protocollo controllati.

## 0.1.4 — selezione modello remoto

- [x] Verifica sostituisce un ID obsoleto con quello disponibile e seleziona il modello se unico. Fra più modelli mantiene una selezione ancora valida. Una lista vuota non cancella il campo.
- [x] Rigenerato asset ES2017; suite UI, web challenge e scrolling superate.
- [x] APK debug 0.1.4 generato in dist; build Gradle riuscita.

## 0.1.5 · interfaccia bilingue

Avvio in inglese; selezione English/Italiano nelle impostazioni, salvata in `settings.language` nello stesso archivio locale e riletta al caricamento. Gli archivi precedenti senza preferenza usano EN. Il cambio non traduce chat, codice o titoli salvati. APK: `dist/SUC-Studio-Android-0.1.5.apk`; codice versione 6. La distribuzione locale conserva la firma debug precedente.
