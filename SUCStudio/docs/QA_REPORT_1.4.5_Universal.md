# QA — SUC Studio 1.4.5_Universal

## Base e isolamento

- Sorgenti e risorse derivati dalla cartella locale SUCStudio 1.4.5_Final.
- Individuato un disallineamento fra UI finale e sette file Swift intermedi locali.
  Recuperati dalla release registrata nel repository, esclusivamente nella variante
  Universal. Dettagli: SOURCE_BASELINE.json e UNIVERSAL_DESIGN.md.
- Engine/memory_policy.hpp e il resto del wrapper C++ conservati dalla cartella locale.
- Motore compilato dal cpp padre corrente, fingerprint:
  `3009335a65540cfa9c74074c8417b6662840f0ad655433a3624f7ae6b3ee774d`.
- Nessuna modifica a SUCStudio originale: confronto SHA-256 con la situazione iniziale.
- Archivio, credenziali Keychain, profilo WebKit, pagine salvate, bundle ID e installer separati.

## Test eseguiti

- Test piattaforma eseguiti in ambiente x86_64: backend remoto obbligatorio anche
  con impostazione locale preesistente; loadModel/startServer/EngineClient.start
  bloccati senza avviare il motore arm64. Ripetuti con SUC_REMOTE_ONLY.
- Test HTTP su fixture localhost: elenco modelli, API key e 401, health check,
  SSE CRLF, testo e usage, reasoning, immagini vision, cancellazione, URL,
  parametri compatibili e telemetria SUC/DS4/LM Studio/llama.cpp/custom.
- La fixture HTTP richiede l'apertura di una porta localhost; eseguita fuori dalla
  sandbox di sviluppo dopo il blocco iniziale del bind.
- Sintassi bash e Info.plist verificati.

## Build e distribuzione

- Compilazione app riuscita per arm64 e x86_64; unione con lipo.
- App e helper shell/terminale verificati Universal; motore verificato arm64.
- Firma ad hoc verificata con codesign --verify --deep --strict.
- Dipendenze dinamiche limitate ai framework/librerie di sistema e runtime Swift.
- Helper Universal eseguiti in x86_64: shell, Ctrl-C (uscita 130), Ctrl-D (EOF).
- Installer dedicato: dist/SUC-Studio-1.4.5_Universal-universal.pkg.
  Installazione affiancata in /Applications/SUC Studio Universal Preview.app.
- Installer senza Developer ID o notarizzazione, destinato a prove locali.
- Pacchetto espanso con pkgutil: identificativo, versione, percorso affiancato,
  binario Universal e firma del payload verificati; checksum SHA-256 verificato.

## Collaudo ancora necessario

- Interfaccia completa, WebKit, import/export e installazione su Mac Intel e Apple Silicon.
- Caricamento di un modello e inferenza SUC locale su Apple Silicon. Il runtime
  arm64 non è eseguibile nell'ambiente x86_64 usato per questo lavoro.
- I test HTTP usano una fixture; non equivalgono a benchmark con server/modelli reali.
- Requisito mantenuto: macOS 15+. La preview non sostituisce la versione stabile.
