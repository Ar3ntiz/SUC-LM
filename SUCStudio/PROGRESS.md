# SUC Studio — avanzamento

Dal 18 ottobre 2026 la versione Universal collaudata sostituisce la biforcazione: unica cartella SUCStudio, versione 1.4.6.04. Segue lo storico della realizzazione iniziale.

- [x] U1 — Copiare sorgenti e risorse dalla versione locale finale, escludendo cache e distribuzioni precedenti.
- [x] U2 — Separare nome app, bundle ID, sessioni, preferenze, credenziali e installer.
- [x] U3 — Forzare server remoto su Intel; proteggere avvio motore/caricamento modello/server locale e adattare Settings, Server e menu bar.
- [x] U4 — Build arm64 + x86_64 per app e helper, motore arm64 dal solo cpp padre; verifica Universal e pacchetto dedicato.
- [x] U5 — Compilare entrambe le architetture e testare routing, streaming remoto e isolamento dati.
- [x] U6 — Creare e verificare installer, guida e changelog; consegnare nella cartella sorgente locale.
- [x] U7 — Collaudo utente della Universal positivo su Intel, M3 e M1 Ultra, confermato il 18 ottobre 2026.

## 1.4.6 — chat, routine e skill

- [x] F1 — Nascondere richieste tool complete/parziali (anche JSON fenced e thinking); mantenere i dettagli nel registro.
- [x] F2 — Riepiloghi e avvisi attività compatti, piccoli e con sfondo grigio.
- [x] F3 — Routine una tantum o ricorrenti con numero/unità (minuti, ore, giorni, mesi), compatibilità delle routine salvate e test calendario.
- [x] F4 — Completare il ciclo tool delle routine, gestire l’avvio web automatico e le approvazioni manuali; puntino giallo persistente per risultati non letti.
- [x] F5 — Skill con titolo, descrizione e corpo; catalogo nel prompt e caricamento delle istruzioni su richiesta, import SKILL.md e plugin compatibili.
- [x] F6 — Output max predefinito 4000 nelle nuove chat, conservando le impostazioni delle sessioni esistenti.
- [x] F7 — Regressioni, build Universal, QA/guida/changelog, installer 1.4.6 e consegna nella cartella locale.

## 1.4.6.02 — skill, hook e plugin compatibili

- [x] E1 — Leggere le quattro pagine richieste e le specifiche ufficiali collegate; confrontare il runtime SUC e definire la compatibilità.
- [x] E2 — Skill a cartella con SKILL.md, risorse progressive, metadati openai.yaml, invocazione esplicita/implicita e discovery user/progetto.
- [x] E3 — Hook JSON con eventi lifecycle, matcher, stdin/stdout strutturati, timeout/async, trust per definizione e decisioni/context/rewrite.
- [x] E4 — Integrare hook nei turni, strumenti, approvazioni, MCP, subagent, interruzioni e chiusura; preservare i vecchi hook.
- [x] E5 — Plugin portable e compatibilità .codex-plugin, copia risorse, MCP/hook/skill, marketplace locali/remoti e gestione componenti.
- [x] E6 — Editor guidati, suggerimenti e sample per skill/hook/plugin; selezione esplicita dalla chat.
- [x] E7 — Test unitari/integrazione/regressioni, anteprime UI, guida e matrice di compatibilità.
- [x] E8 — Build Universal, installer 1.4.6.02, changelog compatto e consegna verificata nei sorgenti locali.

### Consegna 1.4.6.02 — 17 ottobre 2026

- Sorgenti e distribuzione copiati in `/Users/username/PycharmProjects/LLM/SUC-LM/SUCStudioUniversal`.
- Test delle estensioni, cancellazione MCP/approvazioni, prompt e registro I/O; regressioni Universal/chat/routine e MCP HTTP/SSE superati.
- Build ottimizzata arm64/x86_64; installer estratto, firma/architetture/risorse verificati. Binary del payload identici all’app consegnata.
- 204 file confrontati con la copia locale, inclusi permessi eseguibili. Il progetto stabile SUCStudio e il motore padre cpp hanno hash invariati.
- Changelog nella dist: tutte le descrizioni entro 60 parole.
- Installer SHA256: `c30c14bce48a1c92eb1c60ea909dcc6717e1756493c48d8915e470e3f0e7a3e9`.
- Restano il collaudo interattivo dell’app sui due Mac e le limitazioni di compatibilità documentate nella guida; nessuna installazione automatica eseguita.

## Correzione generatore plugin — stessa versione 1.4.6.02

- [x] P1 — Selettori indipendenti Skill / Hook / MCP ed editor che conservano i rispettivi testi.
- [x] P2 — Esportazione combinata, validazione dell’intero pacchetto e protezione delle cartelle esistenti.
- [x] P3 — Test delle combinazioni, verifica UI e aggiornamento guida/changelog.
- [x] P4 — Ricompilazione Universal, nuovo installer 1.4.6.02 e consegna nei sorgenti locali.

- [x] R1 — Riprodotta nell’archivio la routine interrotta: introduzione testuale seguita dal JSON, non riconosciuto dal parser.
- [x] R2 — Riconoscere l’envelope finale su una propria riga e nasconderlo anche durante lo streaming; preservare JSON ordinario/codice e rifiutare azioni multiple ambigue.
- [x] R3 — Verificare il ciclo schedulato completo con intro + JSON raw/fenced, risultato web, risposta finale e notifiche.

### Consegna correzioni stessa versione

Versione visibile **1.4.6.02**, build interna **25** per aggiornare l’installer precedente. Generatore combinato e routine con premessa + JSON verificati; protocollo operativo nascosto anche durante lo streaming. Build Universal e installer verificati, file copiati nei sorgenti locali. Le vecchie routine senza risultato vanno rilanciate dopo installazione/riavvio; nessun replay automatico o modifica all’archivio personale.

Installer SHA256: `f8f86aa96be27b6fad0d2624e1c4cb01209b8ae39e934260687458f86d68712c`.

## 1.4.6.04 — consolidamento, 18 ottobre 2026

- [x] U1 — Verifica dei sorgenti locali e del collaudo utente Intel, M3, M1 Ultra.
- [x] U2 — Rimozione della biforcazione: Universal rinominata SUCStudio, sorgenti precedenti conservati fuori dal progetto.
- [x] U3 — Nome visibile SUC Studio; identità, dati e Keychain Universal conservati.
- [x] U4 — Aggiornamento versione, installer, guida e changelog.
- [x] U5 — Test di regressione e build arm64/x86_64 con motore padre aggiornato.
- [x] U6 — Verifica pacchetto e consegna nella cartella locale dei sorgenti.

## 1.4.7.03 — cache dei turni Qwen, 19 ottobre 2026

- [x] C1 — Checkpoint ibrido limitato a 128 MiB, senza duplicare KV/indexer; stesso contratto parentale per helper e server.
- [x] C2 — Regressioni CPU, Metal e suite Universal complete Intel/M1.
- [x] C3 — Qwen side8 reale: 950 posizioni e 235.904.000 logits identici, KL 0, continuazioni identiche; thinking, modifica e cambio sistema verificati.
- [x] C4 — Riprodotto e corretto il trattenimento della mappatura dopo unload nei task dell’helper.
- [x] C5 — Completare benchmark HTTP 2k/11,7k con checkpoint attivi/disattivi e collaudo del nuovo helper sul Qwen reale.
- [x] C6 — Pacchetto Universal e archivio CLI/server, verifica dei payload, manifest/checksum e sincronizzazione dei sorgenti.

Piano ed evidenze: `../docs/optimization/CHAT_PREFIX_CHECKPOINTS.md`. Qualifica del motore e verifica dei pacchetti completate. La 1.4.7.02 rimane disponibile; nessuna installazione automatica.

## 1.4.7.04 — avanzamento e metriche live

- [x] P1 — Avanzamento prefill dai chunk GPU completati, contatori cache e frame SSE/IPC.
- [x] P2 — Barre prefill e caricamento; avviso nelle chat da almeno 2k token con cronologia precedente e cache non riutilizzata.
- [x] P3 — Decode live SUC su token/tempi reali; stime client chiamate ricezione.
- [x] P4 — Test regressione Swift/C++, collaudo helper e SSE, build Universal e verifica installer.
- [x] P5 — Documentazione, pacchetto e allineamento locale/Mac Studio/public.

## 1.4.9.04 · build 45 — chiamate MCP prolungate

Dichiarazione Apple Events IT/EN ed entitlement di automazione nell’app. Chiamate MCP gestite come job con stato, attesa, annullamento e scadenza totale facoltativa; approvazioni, isolamento delle chat e Stop conservati. Nessuna ripetizione automatica dopo esito incerto. Piano e verifiche: `docs/optimization/PROGRESS_1.4.9.04_IT.md`; dettagli: `docs/optimization/RELEASE_1.4.9.04_IT.md`. Motore, memory policy e APK invariati.

## 1.4.9.05 · build 46 — aggiornamenti tra le operazioni

La coda consegna le correzioni ai confini sicuri del ciclo: prima di una proposta ancora da eseguire oppure dopo un risultato reale. Decode, operazioni avviate, bozze, allegati e limiti restano conservati. Piano: `docs/optimization/PROGRESS_1.4.9.05_IT.md`; verifiche e limiti nel rapporto `RELEASE_1.4.9.05_QA.json`.
