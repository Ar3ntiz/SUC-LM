# SUC Studio 1.4.9 · build 37

La versione include il fix sorgente 1.4.8.04, senza installer intermedio: «Consenti comandi shell» è disabilitato se non sono presenti una directory di lavoro e l’abilitazione delle azioni del modello. Una nota rende espliciti i prerequisiti. Il valore precedentemente salvato non viene cancellato e i controlli di esecuzione restano attivi.

L’interfaccia macOS supporta italiano e inglese. L’inglese è il valore predefinito anche su sistemi italiani e per archivi precedenti privi della preferenza. La selezione nei Settings viene salvata in `AppSettings.language`, nello stesso archivio delle sessioni, e riletta al caricamento e al ripristino del backup. Valori sconosciuti ricadono sull’inglese. Il cambio lingua aggiorna anche menu e pannelli; richieste e stato operativo risiedono nello store e non vengono riavviati.

La localizzazione riguarda il testo dell’app: etichette, descrizioni, suggerimenti, pannelli, dialoghi, messaggi diagnostici conosciuti e guida HTML. Il catalogo è incorporato nell’app e non richiede rete. Non traduce chat, codice, titoli già salvati, definizioni delle estensioni, contenuti web, output dei tool o messaggi prodotti da server esterni. I nomi dei tool, gli eventi hook e le chiavi dei protocolli restano invariati. I dialoghi del sistema seguono le impostazioni del sistema operativo.

Android 0.1.5 · codice 6 offre la stessa selezione IT/EN nell’archivio del dispositivo, con inglese predefinito. L’asset web e i dialoghi del browser sono localizzati; non vengono aggiunte shell, inferenza locale, routine o notifiche. L’APK locale conserva la firma debug delle versioni precedenti; una release firmata richiede il keystore del proprietario.

## Verifiche

Le verifiche definitive, i checksum e i percorsi degli artefatti sono registrati nel piano `PROGRESS_1.4.9_IT.md` e nel rapporto `RELEASE_1.4.9_QA.json` al completamento della build. Non sono previste nuove misure di velocità o KL: il motore C++/Metal, il formato SUC1 e la memory policy non cambiano.
