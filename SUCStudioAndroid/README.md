# SUC Studio Android 0.1.3

Client di chat per smartphone e tablet Android **8.0 o superiore**, separato da SUCStudio per macOS. Usa esclusivamente server remoti compatibili OpenAI: SUC Studio, DS4, LM Studio, llama.cpp e altri endpoint `/v1/chat/completions`.

## Funzioni

- Chat persistenti, ricerca, rinomina/eliminazione, export JSON/Markdown e import JSON Android con immagini incorporate.
- Streaming, stop, modifica e rigenerazione nella stessa sessione; thinking separato, Markdown, codice copiabile/salvabile e composer ridimensionabile.
- Immagini per modelli vision, testo con qualsiasi estensione, DOC 97+, DOCX, ODT, RTF, PDF testuali. Estrazione locale trasparente e anteprima prima dell’invio.
- System prompt globale e di sessione, sampling e aspetto configurabili. Output predefinito: 4000 token.
- Skill progressive con titolo/descrizione/istruzioni ed esempi; MCP Streamable HTTP; plugin portabili ZIP o SUC v1 JSON con skill e/o MCP HTTP. Import locale e marketplace HTTPS JSON.
- Ricerca web e lettura pagine dal dispositivo, con browser separato e conferme inline. Artifact testuali persistenti e salvataggio tramite selettore Android.
- Metriche remote quando esposte, token riusati in cache e prefill sui soli token nuovi; stime indicate con `≈`, valori mancanti mostrati con `—`.

**Esclusi:** motore/server locale, MCP stdio, hook shell, terminale, routine e notifiche. Non esistono sincronizzazione automatica con il Mac o accesso implicito ai suoi file.

## Installazione e utilizzo

Apri `dist/SUC-Studio-Android-0.1.3.apk` sul dispositivo e autorizza l’installazione per il gestore file/browser usato. È un APK per collaudo locale, firmato con la chiave Android di sviluppo della macchina di build; non è una pubblicazione Play Store.

Configura URL, eventuale API key e modello nella tab **Server**. Usa **Verifica / modelli**, poi **Salva server**. Apri **Chat** e invia un messaggio. Su telefono l’elenco chat è a scomparsa, su tablet affiancato.

La guida completa è [USAGE.md](USAGE.md). Stato del lavoro: [PROGRESS.md](PROGRESS.md). Verifiche: [docs/QA_REPORT.md](docs/QA_REPORT.md). Release: [dist/CHANGELOG.md](dist/CHANGELOG.md).

## Compilazione

Servono JDK 11, Android SDK con piattaforma 34 e Node.js 18+ per rigenerare la UI. Gradle wrapper e lock npm sono inclusi; le dipendenze hanno versioni fissate. Questo progetto non dipende da SUCStudio o dal motore C++.

```sh
export ANDROID_HOME="$HOME/Library/Android/sdk"
./scripts/build.sh
./scripts/test.sh
./scripts/package.sh
```

Il Gradle plugin 7.4.2 e Gradle 7.6.3 sono scelti per l’ambiente JDK 11 disponibile; sono strumenti di build, non dipendenze richieste sul telefono. Per importare in Android Studio apri questa cartella e seleziona JDK 11 per Gradle.

Per una release con una chiave propria configura le variabili `SUC_ANDROID_KEYSTORE`, `SUC_ANDROID_STORE_PASSWORD`, `SUC_ANDROID_KEY_ALIAS` e `SUC_ANDROID_KEY_PASSWORD`; `package.sh` compilerà la variante release. La chiave non va inclusa nel repository. Usa la stessa firma per gli aggiornamenti.

## 0.1.5 · interfaccia bilingue

Avvio in inglese; selezione English/Italiano nelle impostazioni, salvata in `settings.language` nello stesso archivio locale e riletta al caricamento. Gli archivi precedenti senza preferenza usano EN. Il cambio non traduce chat, codice o titoli salvati. APK: `dist/SUC-Studio-Android-0.1.5.apk`; codice versione 6. La distribuzione locale conserva la firma debug precedente.
