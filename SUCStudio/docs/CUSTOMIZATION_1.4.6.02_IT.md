# Personalizzazione — SUC Studio Universal 1.4.6.02

Questa versione adotta i formati pubblici di skill, hook e plugin. SUC è un harness indipendente: non usa il runtime o gli account di ChatGPT. Le estensioni funzionano con qualsiasi modello/server che sappia seguire il protocollo degli strumenti; i modelli meno capaci possono richiedere un richiamo esplicito o l’esecuzione manuale degli MCP.

## Skill

Una skill è una cartella con `SKILL.md`:

```markdown
---
name: review-code
description: Usa quando viene richiesta una revisione del codice.
---
Leggi i file pertinenti. Riporta bug concreti, file, righe e motivazione.
```

Aggiungi facoltativamente `references/`, `scripts/`, `assets/` e `agents/openai.yaml`. Importa la cartella o SKILL.md dai Settings. Il pacchetto viene copiato nello spazio dati separato dell’app; successive modifiche all’originale richiedono una nuova importazione. Gli esempi nell’editor sono modificabili e inizialmente disattivati.

Il modello riceve solo nome, ID e descrizione delle skill attive. `load_skill` carica le istruzioni; `skill_resources` elenca i file e `skill_resource` legge una risorsa UTF-8, fino a 64 KiB. I riferimenti `@review-code` e `$review-code` nel prompt caricano esplicitamente le istruzioni. Il pulsante con le scintille nel composer inserisce il richiamo.

Metadati scalari letti da `agents/openai.yaml`: `display_name`, `default_prompt`, `allow_implicit_invocation`. Quest’ultimo è un’indicazione al modello, non un confine di sicurezza. Dipendenze MCP dichiarative, icone, stringhe YAML multilinea in openai.yaml e configurazioni di provider non sono interpretate: configura gli MCP nella loro sezione. Il frontmatter SKILL.md supporta descrizioni semplici e multilinea.

«Scopri skill locali» cerca `~/.agents/skills`, `/etc/codex/skills` e `.agents/skills` dalla directory di lavoro fino alla radice Git. Non attiva né esegue ciò che trova. I nomi duplicati mantengono ID e origini distinti. Le risorse devono rimanere dentro la cartella della skill; script, letture di progetto e modifiche usano gli ordinari strumenti e permessi della sessione.

## Hook

Apri la gestione Hook dai Settings. Puoi scrivere un comando o importare `hooks.json`:

```json
{"hooks":{"PreToolUse":[{"matcher":"^Bash$","hooks":[{"type":"command","command":"printf '%s\\n' '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Comandi disabilitati dal progetto.\"}}'","timeout":10}]}]}}
```

Revisiona il comando e seleziona «Ho letto questa definizione…», poi «Autorizza e attiva». L’autorizzazione copre la definizione esatta: una modifica invalida il suo hash. Installare o attivare un plugin non autorizza i suoi hook. I vecchi hook SUC restano leggibili, ma richiedono questa revisione prima di funzionare.

Eventi emessi: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, `Stop`, `Interrupt`, `SubagentStart`, `SubagentStop`, `SessionEnd`. `PreCompact` e `PostCompact` sono accettati nei file ma non vengono emessi: non esiste ancora un compattatore del contesto. SessionStart viene emesso al primo invio dopo apertura/ripresa, SessionEnd all’eliminazione della sessione o alla chiusura dell’app. Stop riguarda la risposta finale, non ogni passaggio intermedio.

Ogni comando riceve JSON su stdin: evento, session_id, cwd, model, turn_id, permission_mode; per gli strumenti anche tool_name, tool_input e tool_use_id. `Bash` identifica i comandi; `apply_patch` identifica le scritture. Matcher regex: nome dello strumento, source per SessionStart, agent_type per SubagentStart, trigger per la compattazione o reason per SessionEnd (SubagentStop non espone un discriminante di matcher in questa versione). Vuoto o `*` per tutti; ignorato su UserPromptSubmit, Stop, Interrupt. Restano disponibili le variabili SUC_HOOK_* delle vecchie versioni. `transcript_path` è null: l’app non esporta automaticamente conversazioni su disco.

Output supportato: testo come contesto per SessionStart/UserPromptSubmit; JSON `hookSpecificOutput.additionalContext`, blocco di prompt/tool, `PreToolUse` deny o allow con updatedInput (argomenti stringa), `PermissionRequest` allow/deny, feedback sostitutivo PostToolUse, `Stop` block/reason per una continuazione. Un deny prevale sugli allow. Exit 2 segue il comportamento di blocco/feedback dell’evento; gli altri errori sono registrati e non interrompono il lavoro. PreToolUse allow non sostituisce l’approvazione dell’utente. Decisioni non supportate non sono applicate e vengono registrate.

Timeout predefinito 600 s, massimo 3600; SessionEnd/Interrupt 1 s, massimo 3. Output limitato a 256 KiB per stream. Il contesto viene unito al system prompt esistente, senza messaggi system duplicati; snapshot e registro I/O riflettono la richiesta inviata. Il contesto oltre additionalContextLimit (2500 token stimati, 4 caratteri/token) viene abbreviato conservando un file completo. Gli hook asincroni sono limitati a 8 per sessione: il contesto arriva al successivo invio, senza avviare un turno autonomamente; non possono autorizzare/bloccare/riscrivere azioni. SessionEnd è sincrono.

Handler `mcp_tool`: specifica server, tool e input; il server deve essere già attivo. Le stringhe `${tool_input.path}` e `${prompt}` vengono risolte; un riferimento che occupa tutto il valore conserva il tipo JSON. Non vengono invocati altri hook ricorsivamente. SessionEnd non usa MCP. Gli errori non concedono autorizzazioni.

Differenze dal runtime OpenAI: handler sequenziali per decisioni deterministiche; massimo una continuazione Stop per turno; nessun handler prompt/agent, configurazione TOML amministrativa o sandbox di sistema. Gli script hanno i permessi dell’utente: revisionarli prima di attivarli.

## Plugin e marketplace

Formati importabili: `plugin.json` portabile, `.codex-plugin/plugin.json`, `.claude-plugin/plugin.json`, manifest SUC v1. Nei pacchetti portabili: skill in `skills/`, MCP in `mcp.json`, hook in `hooks/hooks.json` o nell’overlay `extensions.com.openai.hooks`. L’overlay portabile sostituisce quello `.codex-plugin`, senza fusione dei campi. Hook espliciti possono essere un percorso, più percorsi o oggetti inline.

MCP stdio e Streamable HTTP; variabili `${PLUGIN_ROOT}`, `${PLUGIN_DATA}` e alias CLAUDE nei campi dei pacchetti. Il codice viene copiato in `Plugins/`; i dati scrivibili sono separati in `PluginData/`. Non inserire segreti nei manifest: usa il Portachiavi nel gestore MCP. OAuth interattivo, account/app OpenAI, npm e marketplace privati ChatGPT non sono inclusi; componenti app e handler non supportati causano un errore esplicito. I metadati OAuth non attivano automaticamente l’autenticazione: occorrono credenziali utilizzabili configurate manualmente.

«Crea con esempi» esporta una cartella verificata con esempi di skill, hook o MCP. «Marketplace pacchetti» apre un indice locale o un repository Git HTTPS; cerca `.agents/plugins/marketplace.json`, `.claude-plugin/marketplace.json` o `marketplace.json`. Sorgenti local, url e git-subdir; path locali confinati alla radice del marketplace; policy NOT_AVAILABLE rispettata. I plugin vengono installati disattivati, senza installazione silenziosa imposta dall’indice. Il vecchio marketplace HTTPS SUC con checksum resta disponibile.

Dettagli consente di gestire skill e MCP del plugin attivo; gli hook si revisionano nella loro sezione. `onboardingSkill` deve puntare a una skill inclusa e offre il pulsante di configurazione guidata. L’attivazione può avviare gli MCP; non esegue automaticamente la skill di onboarding. La rimozione toglie i componenti e le credenziali, conservando il pacchetto/dati per recupero. Import massimo 64 MiB e 2048 file; percorsi fuori dal pacchetto rifiutati; `.git`, `.build`, node_modules esclusi (predisporre quindi dipendenze di runtime esterne).

## Fonti e compatibilità

Formati e semantica derivano dalla documentazione pubblica consultata il 17 ottobre 2026:

- [Panoramica personalizzazione](https://learn.chatgpt.com/docs/customization/overview#skills)
- [Creare skill](https://learn.chatgpt.com/docs/build-skills)
- [Hook](https://learn.chatgpt.com/docs/hooks)
- [Plugin](https://learn.chatgpt.com/docs/plugins?surface=app)
- [Creare plugin](https://learn.chatgpt.com/docs/build-plugins)
- [Specifica pacchetti](https://developers.openai.com/plugins/build/plugins)

Non è una replica completa del runtime ChatGPT/Codex. Questa guida identifica le parti implementate e le differenze; il motore C++ e le memory policy non vengono modificati.


### Generatore con componenti combinati (correzione 1.4.6.02)

In «Crea con esempi» seleziona le caselle Skill, Hook e MCP: puoi includere qualsiasi combinazione, anche tutte e tre nello stesso plugin. Le schede editor cambiano soltanto il componente visibile; il testo degli altri resta conservato anche quando togli e rimetti una selezione. «Carica esempio» sostituisce solo il testo dell’editor corrente. «Esporta plugin» crea un unico plugin.json e i file dei componenti selezionati, verificando il pacchetto completo prima di pubblicarlo. Una cartella esistente non viene sovrascritta; un componente selezionato non valido impedisce l’esportazione. Importa poi la cartella nel gestore Plugin; gli hook richiedono ancora revisione e autorizzazione.


### Ricerche schedulate con testo introduttivo (correzione 1.4.6.02)

Se il modello scrive una premessa e poi, su una propria riga, l’envelope suc_action, l’harness riconosce l’azione e continua il ciclo: ricerca reale, risultato inviato al modello e risposta finale. Il protocollo viene nascosto anche mentre arriva in streaming e nelle risposte già archiviate. Sono accettati JSON raw o racchiusi in una singola fence json; codice ordinario e esempi inline non vengono eseguiti. Più azioni nello stesso messaggio vengono rifiutate come ambigue. Le vecchie esecuzioni terminate senza risultato non vengono ripetute automaticamente: rilancia la routine dopo avere installato e riavviato l’app aggiornata.
