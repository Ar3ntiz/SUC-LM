# Architettura Android

Applicazione Kotlin, API 26+, senza codice C++ o dipendenza dai sorgenti macOS. MainActivity ospita una WebView con HTML/CSS/JavaScript locale. UI sorgente in ui/index.html; esbuild produce l’asset ES2017 incorporato. Non esistono server web locale, CDN o download di codice UI a runtime.

Core.kt gestisce endpoint, richieste, SSE, thinking/telemetria e classificazione delle azioni. Network.kt implementa HttpURLConnection con redirect disabilitati, limiti di payload, cancellazione e client MCP HTTP JSON-RPC. MainActivity orchestra una generazione attiva, controlli di identità della richiesta, approvazioni inline e passaggi dei tool. Il lavoro di rete/estrazione è fuori dal thread UI; aggiornamenti DOM limitati a circa 8–10/s. Le risposte grandi hanno un’anteprima Markdown limitata a 64 KiB; l’originale resta salvabile.

Storage.kt usa AtomicFile per le sessioni e Android Keystore AES-GCM per le API key. Attachments.kt estrae documenti localmente, ricodifica immagini e memorizza copie private. Extensions.kt valida pacchetti senza estrarre percorsi arbitrari; non esegue risorse. WebView degli strumenti web separata e senza JavascriptInterface.

La WebView chat ha CSP restrittiva, file/content access disabilitati, nessuna risorsa remota e rendering HTML escapato. Solo il contenuto locale può invocare il bridge. I link esterni passano al browser e non caricano pagine nel renderer dotato di bridge. I parametri sensibili non vengono inseriti nell’archivio o nei prompt.

MCP e skill usano lo stesso ciclo a JSON suc_action del desktop, così anche un modello privo di function calling nativo può richiedere strumenti se segue le istruzioni. Vengono accettati anche tool_calls OpenAI ricevuti dal server, senza pubblicizzarli automaticamente a un backend che potrebbe non supportarli. Una sola azione per passaggio; limiti configurabili e risultati tenuti fuori dal transcript pubblico.

Nessun accesso al filesystem del server di inferenza tramite la sola API OpenAI. Nessuna shell, terminale, hook, stdio, scheduling o notifica Android. Il ciclo di vita segue i limiti del sistema mobile.

Browser web 0.1.1: WebToolBrowser possiede la finestra a schermo intero e la WebView interattiva senza bridge; WebPagePolicy classifica la pagina e contiene lo script ES5 di estrazione. MainActivity espone solo riapertura/stato/annullamento alla UI locale. La pagina rimane aperta nascondendo la finestra, finché l’utente risolve la verifica, fornisce testo manuale o annulla. Ogni timer e finestra viene rilasciato alla conclusione/cancellazione.

Telemetria 0.1.3: StreamAccumulator conserva il conteggio cache anche quando usage e tempi arrivano in frame SSE distinti. Le velocità esplicite del server hanno precedenza; dai tempi SUC si calcola il prefill sui soli token nuovi. timings.prompt_n di llama.cpp è già netto e non viene sottratto nuovamente. I contatori cache sono limitati all’intervallo valido; con cache completa e nessuna misura esplicita la velocità resta assente. Il totale del prompt resta separato dai token effettivamente elaborati. La UI riceve cachedTokens/uncachedPromptTokens senza cambiare l’archivio o il contenuto inviato al modello.
