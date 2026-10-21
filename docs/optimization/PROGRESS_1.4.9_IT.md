# SUC Studio 1.4.8.04 → 1.4.9

- [x] 1.4.8.04: disabilitare il controllo shell senza directory e azioni abilitate; spiegare entrambi i prerequisiti. Nessuna build intermedia.
- [x] 1.4.9: selettore IT/EN nei Settings, inglese predefinito e persistenza nell’archivio esistente.
- [x] Localizzare interfaccia, pannelli, menu, messaggi e dialoghi; preservare contenuti utente e protocollo strumenti.
- [x] Verificare migrazione, persistenza, cambio lingua e dipendenze shell.
- [x] Aggiornare guide, changelog e identità; compilare e verificare installer Universal 1.4.9.
- [x] Sincronizzare sorgenti e artefatti nella cartella del progetto.

Motore C++/Metal e memory policy invariati. Nessuna installazione automatica.

## Verifiche completate

- Test del catalogo Swift: inglese predefinito, migrazione, italiano, interpolazioni e prerequisiti shell.
- Store macOS reale: cambio lingua, salvataggio su ArchiveDisk, ricaricamento e ripristino della lingua. Preferenze font e limiti preservati.
- Android: 14 test Kotlin senza errori, suite UI/Markdown/CAPTCHA/scorrimento e cambio IT/EN con messaggi, codice e titoli invariati.
- APK 0.1.5, codice 6: firma verificata e certificato uguale alla 0.1.4; catalogo e selettore inclusi.
- Motore padre: contenuto e binario helper identici alla build precedente; nessun nuovo benchmark GPU/KL.

Installer macOS 1.4.9 build 37 e APK 0.1.5 verificati. Checksum e dettagli: `RELEASE_1.4.9_QA.json`. Nessun nuovo collaudo fisico su Apple Silicon/Android; compilazione Universal e test locali.

## Consegna

Sorgenti e pacchetti allineati su main nella cartella sorgenti e sul Mac Studio. Aggiornato anche il bundle dist di entrambi; le copie precedenti sono conservate nella cartella .build. Nessuna installazione o modifica delle sessioni utente. Installer macOS e APK hanno checksum uguali nelle copie verificate. Esportazione pubblica tramite lo script ufficiale, conservando la cronologia del repository.
