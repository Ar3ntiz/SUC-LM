# SUC Studio 1.4.7.04 — build 31

## Modifiche

- Prefill: barra dai token effettivamente elaborati a fine chunk GPU. Mostra token nuovi e riusati; il denominatore della barra esclude il prefisso già in cache. Gli eventi arrivano a inizio/fine fase e al massimo ogni 250 ms durante l'elaborazione, senza modificare chunk o kernel.
- Nelle chat con almeno 2.048 token e una risposta precedente, se il motore conferma zero token riusati: «Cache non riutilizzata: rilettura completa della sessione». L'avviso descrive un fallback normale, non un errore. Senza informazioni sulla cache non si deduce un mancato riuso.
- Caricamento: barra indeterminata globale, visibile in tutte le tab, con la fase effettiva (verifica file/tokenizer, preparazione Metal, mappatura dei pesi, grafo, warm-up). Non viene simulata una percentuale delle operazioni opache.
- Decode live SUC: conteggio dei token reali e tempo cumulativo dopo il prefill, coerenti con la formula finale (token generati meno il primo, divisi per il tempo decode). Un aggiornamento intermedio è una media parziale e può oscillare; non include il tempo di prefill.
- Server esterni: se manca la telemetria reale, l'opzione di stima mantiene la dicitura «Ricezione ≈». Usa circa quattro caratteri per token e il tempo di ricezione; non è confrontabile direttamente con i tok/s del motore. Le metriche server hanno precedenza.
- Settings: versione collegata all'identità centrale, 1.4.7.04.

I frame aggiuntivi `prompt_progress` e `suc_metrics` non contengono delta di testo o azioni. Il parser supporta lo stesso avanzamento dei server llama.cpp che lo espongono; negli altri casi resta una barra indeterminata. La nuova telemetria è per richiesta/thread e non lascia un callback attivo durante decode o dopo errori/cancellazione.

## Limiti e distribuzione

Memory policy, formato SUC1, pesi, chunk e kernel numerici invariati. Nessun nuovo punteggio hard/KL o speedup viene attribuito a questa modifica di osservabilità. MTP rimane nel piano separato per Qwen, DeepSeek e GLM.

Installer atteso: `SUCStudio/dist/SUC-Studio-1.4.7.04-universal.pkg`; versione 1.4.7.04, build 31, firma ad hoc, macOS 15+. Verifica: `shasum -a 256 -c SHA256SUMS-1.4.7.04.txt` dalla cartella dist. Nessuna installazione automatica.

## Avanzamento della qualifica

- [x] Test CPU del callback: conteggio cache/nuovi, throttle, isolamento thread e ripristino dopo eccezione.
- [x] Regressioni Swift su M1 e client remoto Intel; progresso SSE e coerenza delle formule live/finale verificati.
- [x] Build Universal e test helper/SSE reali: sette fasi load, sei eventi prefill e tre aggiornamenti metriche live, cancellazione e replay; testo greedy identico alla 1.4.7.03 sul Qwen3-8B locale selezionato.
- [x] Pacchetto estratto verificato: versione/build, firma ad hoc, architetture, guide e identità del motore collaudato.

Aggiornare questa lista solo dopo la verifica dei risultati.

Evidenza del motore reale: [PROGRESS_HELPER_1.4.7.04.json](PROGRESS_HELPER_1.4.7.04.json). Il seguito Qwen3-8B ha esercitato il fallback con cache zero; non si attribuisce a questo test una nuova qualifica del checkpoint del modello grande. Nessuna app installata è stata sostituita.
