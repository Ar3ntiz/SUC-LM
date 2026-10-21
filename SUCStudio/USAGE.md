# SUC Studio 1.4.9.02

Versione unica **SUC Studio**, Universal per Intel e Apple Silicon, macOS 15 o superiore.

- Apple Silicon: motore SUC C++/Metal locale oppure server remoto.
- Intel: solo client OpenAI compatibile; nessun porting del motore SUC.
- Sessioni/dati: `~/Library/Application Support/SUC Studio Universal Preview/`.
- Bundle ID: `local.sucstudio.universal.app`. Identità e credenziali della versione Universal conservate.
- Installazione: aprire `dist/SUC-Studio-1.4.9.02-universal.pkg`.
  Installa `/Applications/SUC Studio.app` come unica versione del prodotto.
- Il pacchetto usa firma ad hoc, senza Developer ID/notarizzazione.
- Non importa automaticamente dati, plugin o credenziali dalla precedente versione non Universal.
  Esportare/importare le sessioni dall’interfaccia; reinserire le API key e riconfigurare gli strumenti.

## Novità della 1.4.6

Le richieste JSON dei tool e i risultati tecnici restano nel registro azioni;
la chat mostra la risposta finale e un piccolo riepilogo grigio, su una riga
quando possibile. Gli errori vengono segnalati durante l’esecuzione.

In Gestisci routine puoi scegliere la prima esecuzione e attivare **Ripeti nel tempo**,
con numero intero e unità minuti, ore, giorni o mesi. Senza il flag la routine
si disattiva prima dell’invio e viene eseguita una sola volta. Le vecchie routine
mantengono il loro intervallo. Giorni e mesi seguono il calendario locale; dopo
stop/riavvio non viene eseguita una raffica di recuperi arretrati.
L’app deve restare aperta e il server deve essere disponibile. Nelle sessioni
con approvazione automatica una routine autorizzata può avviare anche il browser
integrato; con approvazione manuale attende conferma inline nella chat indicata.
Un puntino giallo accanto al titolo segnala risposte o approvazioni non lette;
si spegne aprendo la chat e viene conservato al riavvio.

Le skill hanno titolo, descrizione e istruzioni. Il prompt iniziale elenca soltanto
le descrizioni delle skill attive, con ID; il modello carica il corpo tramite
`load_skill(id)` quando serve. Il caricamento non abilita file/comandi o altri
permessi. Import SKILL.md supporta `name`/`title`, `description` e corpo; le
vecchie skill restano utilizzabili con una descrizione generica modificabile.
Limiti: descrizione 4 KiB e istruzioni 64 KiB. I plugin possono fornire una
descrizione opzionale per ogni skill.

Le nuove chat usano Output max **4000 token**; i valori già salvati restano invariati.

## Aggiornamento alla 1.4.7.03

Chiudere le copie dell’app prima dell’installazione. Il pacchetto installa `SUC Studio.app` e rimuove la precedente `SUC Studio Universal Preview.app` solo se ha l’identificativo previsto. Le chat, impostazioni, credenziali e risorse della Universal restano disponibili: il percorso dati storico e il bundle ID sono mantenuti intenzionalmente. La cartella sorgenti unica è ora `SUCStudio/`; il motore rimane quello di `../cpp`.

## Setup e server remoto

Aprire Server, scegliere Server remoto (obbligatorio su Intel), impostare URL base
con prefisso `/v1`, API key se necessaria e ID modello. Usare Verifica / modelli,
poi Salva server remoto. Aprire Chat e creare una sessione.
Esempi: `http://192.0.2.20:8080/v1` oppure `http://127.0.0.1:8080/v1` per un
motore esterno in esecuzione sullo stesso Mac. Modelli, vision, strumenti e metriche
dipendono dalle capacità del server. Nessun modello/tokenizer locale è richiesto su Intel.

## Build e verifica

Mantenere `SUCStudio` dentro SUC-LM accanto a `cpp`.
Il motore viene compilato dai sorgenti del padre, incluse modifiche non committate.
Non usare il motore del vecchio worktree. App e helper shell/terminale sono
compilati per arm64 e x86_64 e uniti con lipo; il motore integrato rimane arm64
ed è bloccato su Intel. Per compilare entrambe le architetture eseguire:

```sh
./scripts/build.sh
./scripts/test-universal.sh
./scripts/package.sh
```

La build seleziona SDK 26.5 se presente (oppure SDKROOT esplicito).
Un singolo installer contiene entrambe le architetture. Le build per entrambe le architetture e i test automatici non sostituiscono
il collaudo dell’interfaccia su Mac Intel e Apple Silicon.
Collaudo utente completato su Intel, M3 e M1 Ultra.

---

## Guida ereditata dalla versione stabile

Le sezioni sul motore locale valgono solo su Apple Silicon.

# SUC Studio — setup and usage

Italian version: [USAGE-IT.md](USAGE-IT.md)

Local app for macOS 15+ and Apple Silicon. SUC1 models and tokenizers are not included. The package is local and has no Developer ID signature or notarization; the ad-hoc signature only guarantees the technical integrity of the bundle.

## Building

Install the Apple Command Line Tools (`xcode-select --install`) with the macOS 15 SDK or later, and Python 3 to generate the source file that embeds the kernels. No CMake, Homebrew or downloaded dependencies are needed. From this folder:

```sh
./scripts/build.sh
./scripts/package.sh
```

Output: `dist/SUC Studio.app` and `dist/SUC-Studio-1.4.9.02-universal.pkg`. All paths are computed relative to the scripts: the folder must stay inside SUC-LM, next to `cpp/`. To rebuild from scratch: `CLEAN=1 ./scripts/build.sh`. The final app does not require Python, Xcode, the CLT or the repository.

## Installing and getting started

Open the local `.pkg` and follow Installer; it installs into `/Applications`. Alternatively, copy `SUC Studio.app` into Applications. The package does not install privileged services or daemons. A copy received as a download may be blocked by the normal macOS checks: this release is not notarized. Do not disable Gatekeeper or SIP.

1. Open Settings and choose the `.suc1` file and its local `tokenizer.json`.
2. Set the context and memory management ("Automatico" (Automatic) by default). Press **Carica modello** (Load model). Only one model is kept in memory; the first load compiles the Metal pipelines.
3. Open Chat and type. Return sends, Shift+Return inserts a new line; "Interrompi" (Stop) cancels the request. During opaque prefill, the stop is applied when the GPU call returns, not instantly.
4. Chat works without opening a TCP port. Closing the window keeps the app in the menu bar; minimizing is standard macOS. **Esci** (Quit) also shuts down the engine.

## Sessions

New chat (`⌘N`), selection from the sidebar, search in titles/text, rename, favorites, archive/restore, duplicate and delete from the context menu. Drafts and messages are saved automatically. Switching session does not stop a response: it stays attached to the session it started in.

Edit and regenerate keep the same session, stop the previous request and replace the conversation from the chosen message onwards. The session controls set the system prompt, temperature, maximum output and thinking. A reopened session uses the currently loaded model for the new response; the model used is recorded on every response. No KV cache is shared between conversations.

The chat menu exports Markdown or JSON and imports SUC Studio JSON. Data is stored in `~/Library/Application Support/SUC Studio Universal Preview/sessions.json`, with a backup. After a crash, partial responses are marked as interrupted; they are not resent automatically. A context that is too large produces an error, with no silent truncation.

## LAN server

Load the model, open Server, choose "Questo Mac" (This Mac) or "Rete locale" (Local network), the port, and **Avvia server** (Start server). An API key is mandatory; it is generated and stored in the Keychain. Copy it with the dedicated button. To rotate it, stop the server; the old key will no longer be accepted after the restart.

From the LAN, use the Mac's IPv4 address instead of `0.0.0.0`, for example `http://192.0.2.10:8080/v1`. HTTP is not encrypted: use a trusted network. The system may ask for local network/firewall access. No Internet port forwarding is configured.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.suc1","messages":[{"role":"user","content":"Ciao"}],"max_tokens":128,"stream":true}'
```

Use the model name shown by the app. Available: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`. Not available: tools, embeddings, Responses, audio or images. API and chat share the model and the queue (at most eight pending requests); there is no second copy of the weights. Stopping the server closes the listener and requests cancellation of the API requests, keeping the chat.

## Statistics

Prefill = prompt tokens / duration already measured in the runtime; available at the end of prefill. Decode = steps after the first token / decode duration; available at the end of the request. This is not the median reported by some hard-suite benchmarks. No speed is inferred from text chunks. While waiting, status and time are shown; `—` means the measurement is not available. Time to first text includes the transport to the UI.

## Common problems

- Model/tokenizer moved or disk disconnected: select the files again.
- Port in use: change the port; the app does not terminate other processes.
- Insufficient memory: reduce the context/model; "Automatico" (Automatic) enables streaming only for routed experts. A manual budget that is too low can fail the runtime's minimum per-layer top-k requirement.
- Unreadable archive: the original data is preserved; use "Ripristina backup" (Restore backup) in Settings after making a copy of it.
- Engine terminated: reload the model; interrupted responses are not resumed automatically.

See `docs/QA_REPORT.md` for the checks performed and the release limits.

## New in 0.2.0: memory and SSD

**Automatico** (Automatic) uses the resident path if the estimate fits within the conservative limit; otherwise it enables the existing SSD path for routed experts. The cache receives the budget left after the non-routed weights and the reserve. The minimum needed for the slots/top-k of each layer is checked by the runtime: an insufficient budget produces a load error, not a fallback to swap.

**Residente** (Resident) requires the estimate to fit within the limit. **Streaming expert · budget manuale** (Expert streaming · manual budget) accepts a positive value such as `8G` or `1.5G` (K/M/G in base 1024), always within the estimated limit. The budget covers only the expert cache, not the whole process memory.

Since 1.4.4 the limit is `min(working set Metal reale, 96% della RAM fisica)` (min(actual Metal working set, 96% of physical RAM)): the 96% refers to the total expected usage, operating system included, while Metal remains a hard limit for GPU allocations. The reserve includes the KV estimated per architecture plus a 1 GiB buffer; it no longer adds a percentage of the file, because that would have counted the mapped weights twice. For GLM the MLA latents are estimated, not a per-head cache. The policy remains a preventive estimate: the app does not change the Metal limit and does not guarantee the absence of OOM under other memory pressure.

The Settings section shows the last accepted plan (even if the following load fails). SSD statistics are snapshots of the current/last request, updated at the end of prefill and at the end of generation, also in the menu bar menu: bytes read, cache hits and I/O wait. At the end of a request they are cumulative from its start, not only for decode; the weight cache remains reusable across requests. On an error/early cancellation the last snapshot may stop at prefill.

**Models without routed experts:** they work in resident mode if they belong to architectures already supported by the engine (for example dense Qwen3) and fit the memory estimate. If too large, loading is refused. The engine does not implement SSD streaming of dense layers: that would require per-layer loading/eviction of weights, updating Metal bindings and synchronizing command buffers. `mmap` and macOS swap do not provide that path. The new version does not extend the list of supported architectures or formats.

In versions 0.1–0.3 the vendored code in `runtime/` was unchanged; from 0.4 that copy is removed and the build uses `../cpp`. What changes is SwiftUI and the `Engine/` integration wrapper, also written in C++. 0.1 archives remain readable: existing budget → manual; empty budget → automatic. No session is deleted.

## Where to find tokenizer.json

Preferably use the file from the original folder used to compress the model, or the same snapshot of the Hugging Face repository. The tokenizer must match the SUC1 exactly; a tokenizer from another variant may have a different vocabulary/special tokens even if the family name is the same.

Verified links to the official pages (open **Download file**, do not save the HTML page):

| Original model | File |
| --- | --- |
| Qwen3-0.6B | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json) |
| Qwen3-8B | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer.json) |
| Qwen3-30B-A3B | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/tokenizer.json) |

For example, to download only the Qwen3-8B tokenizer into a dedicated folder:

```sh
mkdir -p ~/Models/Qwen3-8B
curl --fail --location 'https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json' --output ~/Models/Qwen3-8B/tokenizer.json
```

`main` can change: when you know the revision used for compression, replace it with its commit. If the model has already been downloaded with Hugging Face, look in `~/.cache/huggingface/hub/models--ORGANIZATION--MODEL/snapshots/REVISION/tokenizer.json`; the path can change with HF_HOME/HF_HUB_CACHE or downloads to custom directories. Keep an actual copy, not a link to a cache you might delete.

For Qwen3.8-Flash-Next, DeepSeek-V4-Flash and GLM-5.3-Flash, use the tokenizer from the source/snapshot used for compression; it was not possible to verify their public pages in this session, so no links are presented as confirmed. The project does not contain copies of these tokenizers. The SUC1 tokenizer_id, visible with the project's inspector, can identify the repository and SHA-256; if the hash is wrong, retrieve the correct snapshot without bypassing the check.

In the app, select the file in **Settings → Tokenizer JSON**. If you place `tokenizer.json` next to the `.suc1`, it is preselected when you choose the model. The app does not download tokenizers automatically and does not convert `tokenizer.model`/GGUF to JSON.

## New in 0.3.0: drag documents into the chat

Drag one or more files into the conversation or directly into the input field, or use the paper clip. The app's harness recognizes the content and reads it in the background, without asking for manual conversions or extractions. Attachments appear above the input field: click the name to see exactly the extracted text, × to remove them. Type your question and press Return; you can also send only the attachments.

- **Text with any extension, or no extension at all:** source code, configurations, logs, Markdown, CSV, JSON, XML, HTML and other text files. UTF-8, UTF-16/32 with BOM and Windows-1252 are tried. The extension is not a whitelist.
- **Modern Word (.docx and equivalent containers):** automatic reading of the document XML, text tables, headers/footers and notes. XML prefixes may vary. No macros are executed and no embedded objects are loaded.
- **OpenDocument (.odt, .ott and equivalent containers):** reading of the content.xml XML, paragraphs, headings and text in tables. Extraction favors the text, not reproduction of the layout.
- **Binary Word (.doc) and RTF:** transparent conversion through the textutil included in macOS, without requiring Word/LibreOffice.
- **Text PDFs:** local extraction through PDFKit, with page numbers. Pages without text are flagged; images/scans do not get OCR.

Recognition is based on the content bytes: a DOCX or PDF renamed with a different extension is still recognized. "Any extension" does not mean decoding any binary format: audio, executables, generic archives and proprietary formats without an extractor are rejected, even if they contain some ASCII strings. Encrypted/protected or corrupted documents produce an explicit error. For PDF/Word the extractable text is read, not charts, formulas as images or the visual layout.

Limits: 5 files per message, 10 MiB per original file, 256 KiB of extracted text per file and 512 KiB in total per message; PDFs up to 200 pages. The model's context limit still applies to **history + attachments + output**. Text is not silently truncated: reduce/split the document or increase the context if memory allows. There is no RAG index: the full accepted text goes into the user message, without changing the C++ engine.

Ready attachments and their text are saved in drafts/sessions, kept for follow-up questions and included in the JSON/Markdown exports of the messages. Edit and regenerate keep them. Moving or deleting the original file after import does not make them unreadable: the session keeps the text and does not depend on the original path. Whoever receives an export also receives the attachments' content. If you switch session while reading, the files stay attached to the originating chat. An import in progress is not saved yet: wait for the attachments to appear before closing the app.

Everything happens on the Mac; the app does not upload files to external services. The new import applies to the native chat: the OpenAI-compatible API still accepts text messages and does not add a /files endpoint.

## New in 0.4.0: single engine in the parent project

The build requires `SUC-LM/cpp/`: both the app and the tests compile the sources, headers, common tools and Metal kernels of that folder directly. `SUCStudio/runtime/` has been deleted. No fallback copy is used; if the parent is missing, the scripts exit with an explicit message. You do not need to commit an engine change to try it in the app: just rebuild.

The hashes of the engine inputs and the toolchain determine cache validity; changing sources to ones with older timestamps or removing files does not reuse stale objects. If the inputs change during the build, the build is rejected and must be repeated. `dist/ENGINE_BUILD_MANIFEST.json` records the repository commit and the actual hashes: the latter also describe uncommitted local changes. The manifest is included in the app. `Engine/` remains app-specific code, not a copy of the kernels/runtime.

After the build, the `.app` and `.pkg` include the compiled engine and remain distributable without `cpp/`, Python or developer tools on the target Mac. The package includes the engine version present at build time; it does not automatically update an already installed app when the sources change.

**M3 check reported by the user:** all Settings and Chat features of the previous version work without problems. The user later also tried the server, reporting faster decode. This is not a systematic validation of LAN, sampling or the new 0.5 features.

The 0.4 icon in the Dock/Finder uses the same `sparkles.rectangle.stack` symbol as the menu bar with the server off, on a rounded background. The assets are included; to regenerate them: `python3 scripts/make-icon.py` (macOS with CLT), then rebuild the app. When the server is on, the menu bar icon keeps indicating it with `server.rack`.

## Global prompt and skills

In Settings you can write the **System prompt generale** (General system prompt). Every new native chat request combines, in this order, the general instructions, the titles/descriptions of active skills in list order (full instructions are loaded on demand with load_skill) and the session's extra prompt. Changes also apply to subsequent messages of existing chats; they do not change responses already saved. Every response keeps a snapshot of the effective prompt. Requests received from the API keep using the system messages sent by the API client.

In the Skill section you can create, edit, delete or import an instruction file (including SKILL.md), then enable it with the toggle. Each skill has a title, a description (up to 4 KiB) and instructions (up to 64 KiB). Only the catalog is included initially; the body is loaded by load_skill(id) when relevant. The text is imported: scripts and files mentioned in the skill are not executed or loaded automatically. The instructions count toward the model's context limit.

## Local and remote MCP

In **Settings → Server MCP** (MCP servers) choose **Aggiungi MCP** (Add MCP):

- **Locale · stdio** (Local · stdio): absolute path of the executable, arguments as a JSON array and environment variables as a JSON object. Example executable `/opt/homebrew/bin/node`, arguments `["/percorso/server.js"]`. The server and its runtime must be installed on the Mac; they are not included in the app.
- **Remoto · HTTP** (Remote · HTTP): Streamable HTTP endpoint, for example `https://server.example/mcp`, and optional headers such as `{"Authorization":"Bearer TOKEN"}`. Use HTTPS for credentials and data on untrusted networks. JSON and SSE responses on the same HTTP endpoint are handled.

Variables and headers are saved in the macOS Keychain; name, endpoint, executable and arguments in the local archive. Put secrets in the variable/header fields, not in the URL or the arguments. Active servers reconnect when the app opens. Turning off a server closes its connection and stops calls in progress; **Riconnetti** (Reconnect) also refreshes the tool list. To edit a server, disable it first.

In the chat, press the tools button, choose a tool, review the schema, enter the JSON arguments and press **Esegui e allega risultato** (Run and attach result). The call really executes the tool, which may modify data according to its own functions. The text result is added to the draft of the originating session: you can read it, remove it or send it to the model together with a question. The model does not need to support tool calling. In this version tool selection is manual; the model does not start tools on its own.

Text, embedded text resources and structuredContent are supported; images and audio are not sent to the model. The attachment limits apply (5 files, 256 KiB per result, 512 KiB in total), 64 KiB for the arguments and 256 tools per server. Errors and results that are too large are explicit. No call is repeated automatically: on timeout/disconnection the outcome of a remote action may be uncertain.

MCP support: initialization, tool listing with pagination and tool calls; stdio and Streamable HTTP transports. Interactive OAuth, legacy SSE with separate endpoints, stream resumption, subscriptions, sampling/elicitation, and MCP resources and prompts as standalone features are not included. Use headers to authenticate servers that accept tokens you already have. HTTP redirects are rejected; configure the final endpoint. After an expired session, reconnect manually. On close the local connection is terminated; the server-side HTTP session expires according to the server's policy.

References: [MCP transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports), [lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle), [tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

## Navigation and help

The section bar stays on the left side. The button at the top expands or collapses the labels, leaving the Chat, Server and Settings icons visible; the preference is saved. The window keeps its minimum of 820 × 560 points and the normal macOS controls.

The **Setup e utilizzo** (Setup and usage) button opens this HTML guide in the default browser, even offline. The page is included in the bundle; external links require Internet. `scripts/generate-help.py` generates it from USAGE.md during the build.

## Remote chat — 0.6

In **Server → Motore per la chat** (Chat engine) choose "Questo Mac" (This Mac) or "Server remoto" (Remote server). Enter the API base (for example `http://192.0.2.20:8080/v1`), the model ID and the API key. For another SUC Studio, use the LAN address of the server Mac and the key copied on the server. **Verifica / modelli** (Check / models) checks GET /models and suggests the models; if the server does not offer this endpoint you can enter the ID manually. Press **Salva server remoto** (Save remote server) before going back to the chat.

Remote mode uses POST /chat/completions with SSE streaming and does not require a local model. Global prompt, skills, history and attachment text are sent to the chosen server. The credentials are in the Keychain; do not put secrets in the URL. The local API service can keep running independently. The backend cannot be changed during a response.

For compatibility, top_k, min_p and chat_template_kwargs are sent only when the extended parameters are enabled; temperature, top_p, max_tokens, stop and a valid seed remain standard parameters. The server must support Chat Completions and streaming; no Responses or OAuth adapter is included. Any server differences are reported as errors, without automatically repeating a request.

**Interrompi** (Stop) closes the remote request; whether computation actually stops depends on the server. Generations no longer have the previous short timeout: they can stay without tokens during long prefills. Every 15 seconds the app tries `GET /health` on the API base and shows whether the server is reachable, does not support the endpoint or does not respond; a failed health check neither stops nor repeats the generation. The only remaining limits are network limits of 24 hours without data and seven days in total, plus manual cancellation. The first text is measured by the client and includes the network. The effective context limit is handled by the remote server.

## Directory and model actions — 0.7

Open the session controls in the chat, choose **Directory di lavoro** (Working directory) and enable the actions. The choice stays associated with that session. The harness prompt describes the tools to the model: directory listing, read, write, directory creation and optional shell commands. The model must be able to emit the JSON protocol described in the prompt; normal messages or Markdown blocks are not interpreted as commands. It works with the local or remote backend, without requiring native tool calling.

The initial mode is **Manuali** (Manual): before every action a card appears with the session, directory, tool and full arguments. Approve only that single action or reject it. With **Automatiche** (Automatic), enabled tools run without dialogs. The log in the session keeps the request, outcome and result. The actual result is returned to the model; default limit of 12 consecutive actions, configurable in Settings from 1 to 100, after which a new message is needed.

File tools accept relative paths, check the effective directory and reject direct symlinks pointing outside. Read up to 256 KiB, write up to 128 KiB, listing up to 2,000 entries. No implicit deletion. Files imported as a session do not receive permissions: the directory and enablements must be selected again.

**Consenti comandi shell** (Allow shell commands) is separate and disabled initially. Commands start in the chosen directory with zsh and your account's permissions: they can access other paths and the network, so the directory is not an operating-system sandbox. Each command has a maximum output of 256 KiB and a maximum duration of 30 seconds. The harness terminates the command's process group when it ends or is stopped; do not use this tool to start permanent daemons. Processes that detach themselves from the group are not a supported mode.

Stopping does not undo changes already made. Keep the work under version control when appropriate. Actions are not resumed automatically after an app restart. The numerical C++ engine is unchanged; the Swift harness and the small execution helper handle tools and approvals.

## Subagents and hooks — 0.8

In the session controls you can enable **Modello capace di delega: abilita subagent** (Delegation-capable model: enable subagents). It is an explicit declaration: the model must follow the harness's JSON tool protocol. When it requests `delegate_task` with a `task` argument, the app creates a separate session and returns the result to the main model. You can open the subagent sessions from the history or from the Subtask list.

Four subtasks per turn are allowed, one at a time, with no further delegation from subagents. Each subagent inherits the directory, session prompt, sampling and approvals, but receives only its own goal: not the main session's whole history. It uses the same selected backend/model. The local engine remains single and serial. Stopping the main session also stops the children's requests. The full result stays in the child session; very long results are explicitly shortened in the summary returned to the main session. Interrupted tasks do not restart on relaunch.

In **Settings → Automazione e hook → Gestisci hook** (Automation and hooks → Manage hooks) you can add zsh scripts for `request_start`, `request_end`, `before_tool` and `after_tool`. Hooks apply to sessions with a working directory, subagents included, and follow the session's approvals. They are distinct from the permission for commands proposed by the model: enabling a hook configures a script you chose.

Scripts receive the variables `SUC_HOOK_EVENT`, `SUC_SESSION_ID`, `SUC_WORKSPACE`, `SUC_TOOL_NAME` and `SUC_TOOL_ARGUMENTS` (JSON object). Do not concatenate this untrusted data as shell code. Hooks have a 30-second timeout and a maximum output of 256 KiB; outcome and output are in the action log. An error before the request or the tool prevents its execution; an error after the tool is logged without undoing what already happened. An error at the end of the request stops subsequent actions. This is not an importer for Claude configuration files: the supported events and the contract are those described here.


## Plugins and marketplace — 0.9

Open **Settings → Plugin e routine → Gestisci plugin** (Plugins and routines → Manage plugins). You can install a `plugin.json` file or a folder containing it. Plugins are installed **disattivati** (disabled); review "Dettagli" (Details: provenance and full manifest) before enabling them. Enabling a plugin adds skills, starts its MCP servers and enables the declared hooks. Disabling it removes the active contributions and closes its MCP servers; removing it also deletes the related credentials. Existing sessions are kept. This is not a universal importer of Claude/Codex plugins.

The SUC Studio v1 format requires `formatVersion: 1`, `identifier`, `name` and `version`; `skills`, `mcpServers` and `hooks` are optional arrays. A complete example is in `examples/plugin/plugin.json`. MCP servers use absolute paths of already installed executables or HTTP endpoints. The plugin does not install dependencies or binaries and does not copy support files: hook scripts are contained in the manifest. MCP credentials are configured in Settings, never in the manifest. Limits: 1 MiB per manifest, 64 skills/16 MCP/16 hooks per plugin, 32 installed plugins.

A marketplace is an HTTPS JSON index with `entries`: each entry contains `name`, an HTTPS `manifestURL` and the `sha256` of the exact manifest. Example in `examples/marketplace.json` (demo URL to be replaced). Load the index, select an entry, check the provenance and install it disabled. Downloads are limited, the checksum is mandatory and redirects are rejected. HTTPS and checksum verify transport/integrity, not the author's trustworthiness: the index must come from a source you chose. No public marketplace run by SUC Studio is included.

To change version, explicitly remove the plugin and install the new manifest. Manual changes to the contributions are replaced on the next re-enabling of the plugin; keep customizations as separate skills/hooks.

## Schedulable routines — 0.9

From **Settings → Gestisci routine** (Manage routines) set the name, prompt, session, required backend, first date and, optionally, a repeat count and unit (minutes, hours, days or months). The routine uses the model configured in the selected backend, preserves the manual draft and keeps a history of 50 runs. It inherits the session's directory, tools and approvals: in manual mode the actions wait for the user.

The scheduler works while the app is open, even with the window closed in the menu bar. It waits until the required backend is selected, the model is available and no requests are in progress. It does not load local models automatically. After sleep or reopening it recovers at most one run, then computes the next date; it does not run the whole backlog. Runs interrupted by a restart are marked as such, with no automatic resend. The app does not install daemons, does not wake the Mac and does not run routines when it is closed.

## Appearance, startup and energy — 1.0

In the "Aspetto" (Appearance) section you can choose the font family and size separately for the interface and the chat. Interface: 10–18 pt; chat: 12–30 pt. Headings and metadata keep readable proportions; code stays monospaced and follows the chat size. The composer uses the same font as the chat. Preferences are persistent and the reset button returns to the system values. The title bar and the native menus managed by macOS keep the system typography.

The **Avvio ed energia** (Startup and energy) section, above "Aspetto" (Appearance), lets you register the app to launch at login through the macOS service. Install the app in Applications; any additional approval is handled from the "Impostazioni macOS" (macOS Settings) button. The UI shows the status returned by the system, including errors or pending approvals. The app does not start the computer before login.

**Mantieni il computer attivo ed evita lo standby** (Keep the computer awake and prevent standby) creates a macOS assertion against automatic system sleep, held while the option is on and the app is open. It is released when you turn off the option or quit. The display can turn off, and explicit sleep/closing the laptop lid follow the macOS rules. The computer's permanent energy configuration is not changed.

## Links and Markdown (1.1)
HTTP/HTTPS links in the chat text open with one click in the system browser. Web URLs written as plain text also become links. File and javascript schemes and commands are not opened from the transcript. The text does not load external images or pages automatically.

The renderer supports headings with #, emphasis and inline code, bulleted/numbered lists and checklists, quotes, separators, tables with alignment and code blocks with copy. Partial code blocks remain readable during streaming. HTML, LaTeX formulas and the full CommonMark specification are not implemented; the content stays as text.

## Images and chat fixes (1.2)
Drag images into the chat or use the paper clip: recognition is based on content, even with arbitrary extensions. Formats decodable by ImageIO (including PNG, JPEG, HEIC; for animated or multi-page images only the first frame/page is used) are oriented and converted to JPEG without metadata, longest side 2048 pixels. Preview and thumbnail show the content that is sent. Limits: original 10 MiB/40 megapixels, JPEG 2 MiB, total images per message 8 MiB and at most 5 attachments in total. Images embedded in PDFs or Office documents are not extracted automatically.

You need **Server remoto** (Remote server) with a vision model that accepts multimodal `content` and `image_url` data URLs. The history also keeps the images for subsequent turns. The local SUC engine is text-only: a request with images is explicitly blocked, without losing the draft. The harness cannot add vision capability to a text model. The total remote request is limited to 20 MiB; beyond the limit, open a new session. The JSON export keeps the images; the Markdown export contains only the descriptive reference.

**Modifica** (Edit) stops the work in progress and puts the text and attachments back into the composer of the same session. **Rigenera** (Regenerate) stops and resends the previous user message in the same session. In both cases the messages after the chosen point are removed; file actions already executed are not undone. Explicit duplication remains available from the session menu.

The Server section switches between the "Server locale" (Local server) and "Server remoto" (Remote server) panels. Changing backend does not stop a local server already started: it remains controllable from the menu bar. Hook management includes "Chiudi" (Close)/Esc and four editable examples, initially disabled. Choosing an example does not run any command.

## Approvals, activity and remote metrics (1.3)
Approvals are cards in the relevant chat, with the full directory and arguments. Approve or reject there: no popup opens. If the request concerns another session or you are in Server/Settings, a bar lets you open the right chat. Subagent approvals stay in their own session.

While the model is loading, a global bar with a loader is visible in all tabs. Prefill and decode have distinct animations; with the macOS Reduce motion setting they become static indicators. Before the first remote text the app signals the wait: a server can include queue and prefill in the same interval.

The SUC Studio 1.3 server adds `suc_metrics.prefill_s` and `suc_metrics.decode_s` to JSON responses and to the SSE usage frame (`stream_options.include_usage=true`). To get both real speeds between two Macs, also update the app that exposes the server. The 0.8.1 memory policy is unchanged. The fields `timings.prompt_per_second`, `predicted_per_second`, `prompt_ms`, `predicted_ms` are also read from servers that expose them.

If only token counts are available, prefill stays "—": time to first text also includes network and queue. Decode can be estimated from the final count and the interval between the first and last text received; it is marked **≈ / stima client** (≈ / client estimate) and is affected by network and buffering. A response in a single block does not allow this estimate. SSE chunks are not counted as tokens.

## Action limit and reports (1.4)
In Settings, **Azioni del modello → Azioni consecutive** (Model actions → Consecutive actions) sets the per-turn limit from 1 to 100 (default 12). It also applies to each subagent and to routines, from the next action; it does not change manual/automatic approvals. A new message starts a new count.

The chat hides the protocol's operational messages: action JSON and technical results sent to the model. The model context and the action log keep them; JSON exports keep the full data. An ordinary model response, even in JSON, stays visible.

At the end of a turn a summary appears with completed and not completed actions and the kinds of work. Errors, rejections and reaching the limit are reported immediately in the conversation. The details stay in the **Registro azioni** (Action log) of the session settings. Reports produced by the interface are not resent to the model as if they were its own responses. Actions already executed on the filesystem are not undone by Stop, "Modifica" (Edit) or "Rigenera" (Regenerate).

## Log and scrolling (1.4.1, updated in 1.4.4)
In Server, above "Registro eventi" (Event log), the **Registro log** (Log viewer) can show a real-time tail of the inputs, reasoning and outputs processed by the chat. The **Abilita Registro log I/O** (Enable I/O log viewer) flag is off by default: in this state the app does not serialize requests for the log and the C++ engine does not produce the server's I/O events. Turn it on only for diagnosis. The panel is closed by default, is not saved and keeps at most 256 KiB; data URL images are replaced by a reference. **Svuota** (Clear) clears the displayed tail.

The chat follows the bottom even when the Markdown layout changes, an approval arrives or a report appears. A manual scroll suspends following: new tokens do not bring you back down. **Vai alla fine** (Go to end), a new send or a session switch resume following. The side list has its own scrollable area; when you create a chat, the selection is scrolled into view.

## Interactive terminal and artifacts (1.4.2)

Select a working directory and enable the session's actions. There is no longer a detached manual shell: the terminal card appears in the conversation only while a `run_interactive(command)` requested by the model is active, for confirmations, passwords or other input. Normal `run_command` calls and processes that have already finished do not show the terminal. **Riservato** (Private) masks the field. Return sends a line, even an empty one (at most 4 KiB); Ctrl-C interrupts, Ctrl-D sends EOF and "Termina" (Terminate) closes the process.

The input is not kept in an app history. The output, limited to the last 128 KiB, stays in memory and is not saved in the archive or sent to the model. Normal password prompts use the terminal's no-echo; the "Riservato" (Private) field also masks the values in the displayed output. Do not put credentials in the chat text. The terminal is a line console with a PTY (100 columns / 24 rows), not a full emulator for vim, top or other full-screen UIs.

The request follows manual/automatic approvals and waits for the process to finish; it does not have the 30-second timeout of run_command. The model receives only the exit code. To read a result use read_file; to offer it as a file use publish_artifact. Only one active inline terminal per session is allowed. Commands run with the user's permissions and do not constitute a filesystem sandbox.

Artifacts appear at the bottom of the conversation: **Salva…** (Save…) exports a copy to the chosen location, **Mostra nel Finder** (Show in Finder) shows the copy kept by the app. write_file publishes the file automatically; for files produced by the shell or other tools the model must call `publish_artifact(path)` with a path relative to the session directory. Regular files of any type up to 100 MiB are allowed; external paths and symlinks leaving the directory are rejected. Each publication is a new persistent copy under `Application Support/SUC Studio Universal Preview/artifacts`, independent of later changes to the original.

The JSON/Markdown export of the chat does not embed the artifact bytes: save the files to transfer to another Mac separately. The copies can take up disk space and remain available to duplicated chats; they are not deleted automatically when a session is deleted.

## Reasoning, editor and telemetry (1.4.3)

Responses from thinking models show the reasoning in a separate box, with a light background and a font one point smaller. The final answer stays below the box in the font chosen for the chat. Both the `<think>…</think>` tags and the `reasoning_content`/`reasoning` streaming fields of remote servers are recognized. The full text stays archived in the session.

Typing three consecutive backticks in the composer opens a dedicated editor where you can paste code and optionally specify the language. **Inserisci codice** (Insert code) creates a safe Markdown fence even if the text already contains backticks. The handle above the composer lets you extend it upwards; the minimum height stays the standard one and the maximum height preserves space for the conversation.

Messages over 8,000 characters show a lightweight preview and a command to open the full text. Markdown parsing is prepared off the interface thread, with limits on the number of blocks; raw HTML is shown as code and not interpreted. The composer's automatic link, data and spelling checks are disabled to avoid expensive scans while pasting. The content sent and persisted is not truncated.

In **Server → Profilo telemetria** (Telemetry profile) choose "Automatico" (Automatic), SUC Studio, DS4, LM Studio, llama.cpp, LLM Router or custom JSON fields. Measurements provided by the server take priority; estimates are always marked with `≈`. llama.cpp receives `timings_per_token` and `return_progress` when live updating is on. DS4 exposes reasoning and usage counts but, in the verified version, leaves tok/s only in the server logs: the app can estimate decode, but does not invent prefill. LM Studio uses the `stats` fields when present. LLM Router keeps the backend's telemetry; to forward the llama.cpp extensions, select the llama.cpp profile.

The custom profile accepts dotted JSON paths, for example `metrics.prefill_tps` and `metrics.decode_tps`, with a controlled maximum depth. **Aggiorna le metriche durante lo streaming** (Update metrics during streaming) can be turned off; **Stima decode live** (Live decode estimate) uses a rough estimate of four characters per token when the server does not publish a speed. Network times, batching and buffering can skew it. Without a server measurement, prefill stays `—`.

## Long requests and chat controls (1.4.4)

The line below the composer shows `work directory: percorso` (the directory path) next to prefill and decode when the session has a working directory. The full path also appears as a tooltip on mouse hover. In the session settings, **Sampling avanzato** (Advanced sampling) includes a legend for Top-p, Top-k, Min-p, seed and stop. The filters combine: to assess the effect it is best to change one at a time.

## Web search (1.4.5, revised in 1.4.5_01 and 1.4.5_Final)

In every session with **Consenti strumenti web** (Allow web tools) on, the model can use two web tools, even without a working directory and with **Consenti azioni del modello** (Allow model actions) off: `web_search(query)` searches Google and returns up to 20 links in Markdown plus 1,200 characters of the page text; `visit_page(url)` opens a known `http` or `https` address and extracts its title, content and up to 80 links. The name `google_search` is also accepted. The tools work with every backend (local engine, remote SUC Studio, LM Studio, ds4 and other OpenAI-compatible servers) because they go through the normal `suc_action` protocol. Without a working directory (or with model actions off) the prompt lists only these two tools and tells the model to use them instead of stating that it cannot browse; the other actions remain ignored. A request enclosed in a single code block (even one marked `json`) is accepted; JSON embedded in running text is not.

Pages open in the built-in browser, with macOS WebKit and its standard user agent. The browser works **hidden**: it does not appear over your windows, does not take focus and does not bring the app to the front. While a search or page is in progress in the selected session, a **globe button** with an activity indicator appears in the chat's top bar, next to the session settings icon: it shows the phase («Navigazione» (Navigating), «Attesa caricamento» (Waiting for load), «Consenso cookie» (Cookie consent), «Scorrimento» (Scrolling), «Estrazione» (Extracting)) and the elapsed seconds; the tooltip also shows «Ricerca Google: …» (Google search: …) or «Apertura pagina: …» (Opening page: …). Clicking it opens the **Browser SUC Studio** (SUC Studio Browser) window with the live page and a status bar (phase, URL, time): you can watch what is happening or where it stopped. Closing it does not stop the tool. When the action ends (completed, with an error or stopped), the button disappears and the window closes by itself: the extracted content is already in the chat or in the saved file, and an empty window never remains on screen. On the first web action after the app starts, the chat shows the prompt «Lo strumento web vuole usare il browser integrato (nascosto; puoi guardarlo dal pulsante globo). Consenti?» (The web tool wants to use the built-in browser (hidden; you can watch it from the globe button). Allow?), even with automatic approvals. The answer is valid until you quit the app. In scheduled routines, automatic approval also authorizes the initial browser start; manual approval remains inline in the target chat, which is marked unread while waiting. Each search or page then follows the session's manual or automatic approvals, showing the query or the URL.

In the session settings **Consenti strumenti web** (Allow web tools) is on by default; when off, the tools reply «Strumenti web disabilitati per questa sessione.» (Web tools disabled for this session.) and the prompt does not list them. `visit_page` rejects `file:`, `javascript:`, `data:`, `about:` and every other scheme. The browser uses a dedicated, persistent profile, separate from Safari: cookies and consent persist across calls and restarts. **Settings → Ricerca web → Cancella dati del browser web** (Web search → Clear web browser data) closes the browser and deletes cookies, consent and cache.

The behavior replicates the ds4 web module: a new page per call in a 1365 × 900 view, waiting for the load (checks every 250 ms, up to 25 s), automatic acceptance of Google's cookie banner and, for `visit_page`, progressive scrolling to load comments and lazy content. The full Markdown of `visit_page` is saved to `.suc-web/visit-<data>-<n>.md` (`<data>` = date) in the working directory; the model receives the relative path and the first 100 lines (at most 8 KiB). To read the rest use `read_file`, which reads files up to 256 KiB. Without a working directory the page goes to `~/Library/Application Support/SUC Studio Universal Preview/web/visit-<data>-<n>.md`: the model receives the absolute path and the same first 100 lines, but cannot read the rest because `read_file` is not available; the file remains available to the user. In the operation log the calls appear as «Ricerca Google: …» (Google search: …) and «Apertura pagina: …» (Opening page: …).

If Google asks for a verification (a `/sorry/` page or reCAPTCHA), the app does not bypass it and does not bring the browser to the front: the page stays loaded and the globe button turns orange with «Verifica Google richiesta: clicca per completarla» (Google verification required: click to complete it). The tool returns an error asking to open the browser from the button, complete the verification and repeat the search. Once the verification is complete the button disappears; the window stays open until you close it. The pending page is replaced by the next web action. Web content is untrusted data: it grants no permissions and does not change the session rules. "Interrompi" (Stop) also cancels the wait for the page.

The extraction scripts and the navigation logic derive from ds4 (https://github.com/antirez/ds4), MIT license, Copyright (c) 2026 The ds4.c authors; the license text is in `licenses/ds4-MIT.txt`.


## Skill, hook e plugin (1.4.6.02)

Nei Settings puoi importare una cartella skill con SKILL.md e risorse, scoprire le skill locali oppure partire dagli esempi nell’editor. Attiva quelle desiderate; richiama una skill con @nome-skill, $nome-skill o il pulsante scintille nella chat. Il modello riceve inizialmente le descrizioni e legge le risorse quando servono.

La gestione Hook include eventi, matcher, timeout, background, comandi e strumenti MCP. Puoi importare hooks.json; prima di eseguirlo devi revisionare e autorizzare la definizione esatta. Cambiare la definizione revoca la fiducia. Installare un plugin non autorizza automaticamente i suoi hook.

Plugin accetta pacchetti portabili, .codex-plugin, .claude-plugin e SUC v1. Crea con esempi esporta un pacchetto modificabile; Marketplace pacchetti importa indici locali o repository HTTPS. I pacchetti rimangono disattivati fino alla tua scelta. La sezione Dettagli gestisce i componenti; le credenziali MCP vanno inserite nel Portachiavi. L’onboarding viene avviato solo su richiesta.

La compatibilità è documentata in docs/CUSTOMIZATION_1.4.6.02_IT.md, inclusi esempi, struttura dei pacchetti e differenze dal runtime OpenAI. Non sono disponibili app/account privati OpenAI, OAuth interattivo o handler hook prompt/agent. Gli script vengono eseguiti con i permessi dell’utente e vanno revisionati prima dell’attivazione.


Per una prima prova: apri Plugin → Crea con esempi, scegli «Skill · revisione codice», esporta una nuova cartella e importala con «Installa da locale». Revisiona Dettagli e attiva il plugin; nella chat digita @review-code seguito dalla tua domanda. Il generatore propone anche un hook di contesto e un MCP remoto. Nei sorgenti trovi inoltre examples/portable-review, un pacchetto completo già verificato con skill, checklist e hook inizialmente da autorizzare.


### Generatore con componenti combinati (correzione 1.4.6.02)

In «Crea con esempi» seleziona le caselle Skill, Hook e MCP: puoi includere qualsiasi combinazione, anche tutte e tre nello stesso plugin. Le schede editor cambiano soltanto il componente visibile; il testo degli altri resta conservato anche quando togli e rimetti una selezione. «Carica esempio» sostituisce solo il testo dell’editor corrente. «Esporta plugin» crea un unico plugin.json e i file dei componenti selezionati, verificando il pacchetto completo prima di pubblicarlo. Una cartella esistente non viene sovrascritta; un componente selezionato non valido impedisce l’esportazione. Importa poi la cartella nel gestore Plugin; gli hook richiedono ancora revisione e autorizzazione.


### Ricerche schedulate con testo introduttivo (correzione 1.4.6.02)

Se il modello scrive una premessa e poi, su una propria riga, l’envelope suc_action, l’harness riconosce l’azione e continua il ciclo: ricerca reale, risultato inviato al modello e risposta finale. Il protocollo viene nascosto anche mentre arriva in streaming e nelle risposte già archiviate. Sono accettati JSON raw o racchiusi in una singola fence json; codice ordinario e esempi inline non vengono eseguiti. Più azioni nello stesso messaggio vengono rifiutate come ambigue. Le vecchie esecuzioni terminate senza risultato non vengono ripetute automaticamente: rilancia la routine dopo avere installato e riavviato l’app aggiornata.

## Verifiche del motore 1.4.7

La versione nel branch di test conserva i file SUC1 esistenti e la politica di memoria. DeepSeek e GLM usano il percorso TCQ FP32 verificato; per confrontarlo con quello storico imposta `SUC_TCQ_REUSE=0` all’avvio della CLI o del server. Le altre famiglie mantengono il kernel predefinito. Nella CLI il prefill automatico di Qwen usa fino a 1024 token per blocco; app e server conservano il proprio profilo prudente. Il valore esplicito di `--chunk` viene rispettato. Il campionamento può produrre testi diversi con lo stesso seed, mantenendo la distribuzione prevista. Il riuso del prefisso accelera l’elaborazione di contesti ripetuti corrispondenti. MTP è rinviato.

CLI e server autonomo per Apple Silicon: `dist/SUC-CLI-Server-1.4.7.03-arm64.tar.gz`, con esempi di configurazione e hash dei binari inclusi. L’app e gli strumenti ausiliari Universal supportano il funzionamento come client remoto su Intel e l’inferenza locale su Apple Silicon. I risultati della verifica KL sono in `../docs/optimization/REFERENCE_KL.md`; i collaudi dei pacchetti sono registrati in `../docs/optimization/RELEASE_SMOKE_1.4.7.json`.

### Profili di prefill e qualità

App e server mantengono blocchi automatici di 64 righe per Qwen e DeepSeek e di 512 righe per GLM; la CLI conserva il profilo separato per i prompt lunghi. Il valore esplicito di `--chunk` per il prefill in batch viene rispettato. Cambiare i confini dei blocchi può cambiare gli arrotondamenti: la sola velocità non giustifica quindi una modifica automatica. La cache conserva un unico stato attivo del motore, senza salvare una KV cache separata per ogni chat. Al cambio di sessione il prompt viene rielaborato, salvo corrispondenza dell’intero prefisso di token effettivamente elaborato. Non occorre scaricare nuovi file SUC1.

### Consegna qualificata 1.4.7 — build 28

Installer Universal: macOS 15 o successivo; su Intel usa Server remoto. L’archivio CLI/server ARM64 distribuito richiede invece macOS 26.0 o successivo (target verificato nei binari). Dopo l’estrazione segui USAGE.md nell’archivio. Verifica i pacchetti dalla cartella dist con `shasum -a 256 -c SHA256SUMS-1.4.7.txt`. Il server standalone non include autenticazione: per la LAN con API key usa il server dell’app. Nessun nuovo download del modello è necessario.

KL invariata su sei modelli; superati tutti i dodici confronti 8k/32k e i collaudi reali CLI/server/motore integrato. Consulta `docs/optimization/RELEASE_1.4.7.md` nel progetto e i documenti tecnici inclusi nel pacchetto per protocolli, impostazioni e limiti.

### Streaming remoto nella 1.4.7.02

La lettura dello stream remoto avviene fuori dal thread dell’interfaccia; restano attivi cancellazione, controlli di dimensione e gestione UTF-8. Le metriche di prefill derivate dai tempi contano solo i token nuovi, mentre le velocità esplicite del server hanno precedenza. Con cache completa senza una velocità esplicita compare `—`. Le impostazioni, le sessioni e le API key esistenti vengono conservate.

La 1.4.7.02 riduce il ritardo di ricezione nel client. La 1.4.7.03 aggiunge il checkpoint dei turni Qwen descritto sopra. Il primo turno lungo e le modifiche anteriori al prefisso salvato richiedono ancora il prefill completo. Template e kernel restano invariati. Per aggiornare chiudi l’app e avvia `dist/SUC-Studio-1.4.9.02-universal.pkg`; non occorre ricaricare o convertire i file SUC1 per usare il nuovo client remoto.

### Qwen conversation cache in 1.4.7.03

The server keeps one bounded checkpoint of recurrent/convolution/n-gram state shortly before the end of the prompt. Matching subsequent turns restore that prefix and evaluate only the suffix, preserving the existing SUC Qwen rendering. CPU state copies are capped at 128 MiB; KV rows are retained in place. The first request, changes before the saved prefix and unavailable checkpoints still require full prefill. `SUC_PREFIX_CHECKPOINTS=0` retains the earlier strict live-prefix policy; `SUC_PREFIX_CACHE=0` disables both policies.

### Avanzamento e velocità nella 1.4.7.04

Durante il prefill la barra mostra token nuovi elaborati e prefisso riusato, quando il motore invia questi dati. Nelle chat lunghe con cronologia, zero token riusati produce l’avviso di rilettura completa: il comportamento è normale dopo un reset o un prefisso incompatibile. Senza telemetria la barra rimane indeterminata. Durante il caricamento del modello la barra globale mostra la fase effettiva, senza simulare percentuali.

SUC invia i conteggi reali durante il decode; la media live e quella finale usano la stessa formula. Negli altri server, «Ricezione ≈» indica una stima dai caratteri ricevuti e dalla rete, non la velocità del motore. Il dato finale può quindi differire. Le metriche reali del server hanno sempre precedenza.

### 1.4.7.05 candidate: model residency and adaptive prefill

The global loading indicator shows activity and the current engine phase until the engine confirms readiness. Resident model-file GiB and Metal allocation are separate memory readings: they may rise or fall and do not represent completion percentages. The tooltip reports initial residency. Monitoring does not force page-in or change the memory policy.

Settings → Prefill → **Automatico (1024 / 64)** enables 1024-token blocks followed by chunks of at most 64; reload the model after changing the setting. Prefix reuse and aligned checkpoints are retained; decoding remains single-token. Exact/token modes, explicit chunks and expert streaming retain their existing profiles. Conservative Batch remains the default while real-model qualification is pending; cross-profile numerical equivalence is not claimed. Standalone server: `--history-prefill auto|off`.


### Passaggi intermedi e comandi nella chat

Le premesse e i ragionamenti del modello tra due azioni rimangono visibili, anche riaprendo le sessioni precedenti. Sono nascosti soltanto il protocollo JSON e i risultati tecnici interni degli strumenti. I comandi shell e interattivi hanno un dettaglio «Comando» chiuso di default: espandilo per leggere il comando, poi richiudilo. Per le nuove azioni viene conservato il comando effettivo dopo eventuali modifiche degli hook. Password e altri input riservati restano gestiti dalla scheda terminale.


### Schemi MCP estesi

MCP catalogs accept complete compact JSON schemas up to 3 MiB per tool without truncation, and up to 256 tools per server. MCP text results allow 4 MiB, with 32 MiB of JSON transport headroom for escaping and metadata. Oversized-schema errors identify the tool and its schema size. REA connectivity was verified with initialize and tools/list, without executing tools.


### Pannello artefatti

Gli artefatti non compaiono più in fondo alla chat. L’icona a forma di pacco nella barra della chat, con il numero dei file disponibili, apre e chiude il pannello laterale. Il pulsante × nel pannello lo nasconde completamente. Il pannello segue la sessione selezionata, è ridimensionabile e mantiene i pulsanti Salva e Finder. Rimane chiuso finché non lo apri, anche quando viene prodotto un nuovo file.


### MCP su richiesta e continuità delle skill

Il prompt iniziale contiene i server MCP connessi e il numero di strumenti, senza copiarne tutti gli schemi. Il modello cerca gli strumenti con mcp_list (query e paginazione), consulta mcp_schema e poi usa mcp_call con le normali approvazioni. Gli schemi grandi si consultano tramite path JSON Pointer e paginazione: lo schema completo resta conservato nel pannello MCP. Le skill richiamate con @nome o $nome rimangono attive durante tutte le azioni dello stesso turno. La lettura del catalogo non esegue strumenti remoti.

Il health check prova anche /health se il percorso sotto /v1 non è disponibile. Un server raggiungibile non dimostra che stia elaborando: lo stato indica l’attesa della risposta, senza simulare avanzamento. Il pulsante × degli artefatti è un controllo AppKit con binding diretto alla chat e area di clic di 28 pt. Il pannello laterale usa una vista divisa ridimensionabile; nelle finestre strette l’elenco delle sessioni viene nascosto temporaneamente per lasciare spazio alla conversazione.


### MCP results up to 4 MiB

The limit applies to the combined UTF-8 text returned by the tool, including structured content. Larger results fail explicitly and are never silently truncated. Local chat, the remote client and the embedded server accept JSON requests up to 32 MiB. Model context and remote server limits still apply; 4 MiB of text can exceed the available token budget. Attachment and shell output limits are unchanged.


## Queued messages — 1.4.8

Write and press Return while generation is running to queue a message with its attachments in the current session. Messages run in order after the complete turn, including tools and hooks. The panel above the composer offers pause, resume, removal and return to draft. Editing requires an empty draft; resending places it at the end of the queue. Stop and generation errors pause the queue. Queued inputs are saved locally and require Resume after relaunch. Switching sessions does not stop the queue or send your unsent draft.


## Context, chat commands and plugins — 1.4.8.02

The ring beside the composer shows context usage. ≈ marks estimates; click it for token counts, capacity and auto-compact settings. Local capacity follows the loaded engine. For remote servers, verify the model and save its reported capacity or set `/context 32768` per session. `/context 0` restores detection. Unknown capacity shows ? and disables automatic compaction. Image costs remain approximate until the server reports usage.

`/compact [priorities]` summarizes the active conversation with the current model, preserving the original transcript and attachments. Summaries may omit details; originals remain accessible. Automatic compaction runs between generations at 99%, with output headroom potentially triggering it earlier. It never interrupts decode. `/auto off|on` controls it. Queued messages wait; cancellation or failure preserves the old context and pauses the queue.

Use `/help` or the / menu for `/status`, `/new`, `/clear`, `/stop`, `/queue`, `/pause`, `/resume`, `/skills`, `/skill name request`, `/plugins`, `/plugin identifier request`, `/model`, `/workspace`, `/export`, `/retry` and the context commands. `/clear` keeps history but resets the active context. // escapes an initial slash. Commands cannot include attachments.

Invoke enabled plugins with `/plugin identifier request`, `@plugin:identifier request` or the skill menu. The model can discover them and request `load_plugin`. Explicit-only skills, MCP connections, hook review and tool approvals remain enforced. Disabled plugins are never enabled implicitly.

Long chats initially render the latest 80 messages, with earlier history available on demand. Manual scrolling keeps the visible window stable. Crossing 8000 characters no longer collapses a message to 12 lines; complete text remains accessible.

### 1.4.8.03 fixes

Compaction adapts its output budget to historical material: 2048–16384 tokens, reduced for small context windows; one bounded retry may use up to 32768 tokens when space permits. This budget includes backend reasoning and is shown during compaction. Empty or truncated summaries never replace existing memory. Some backends may ignore requests to disable thinking.

Scheduled routines reserve at least 4000 output tokens within the available context. Reasoning-only responses are reported as failures rather than completed results. The context indicator refreshes roughly once per second; pre-send budget checks remain current. External inference speed also depends on the server model and its active context.

Transcript pages render a bounded number of rows with previous/next controls; every message remains saved. Historical reasoning starts collapsed and can be expanded. Live reasoning opens and does not auto-collapse at the end of a response, avoiding height jumps.

### Build 36 fixes (version remains 1.4.8.03)

MCP actions accept configured names, UUIDs and unambiguous aliases without the MCP suffix: `rea` resolves to `REA mcp`. Ambiguous aliases require an exact name or UUID; no arbitrary server is selected. Unknown names report available servers instead of claiming a disconnection.

When web tools are enabled and approvals are automatic, the first browser launch follows that policy in interactive chats as well as scheduled routines. Manual approvals remain in place and disabled web tools remain blocked.

## Lingua dell’interfaccia · 1.4.9

Il primo avvio usa l’inglese, anche con macOS in italiano. Da Settings → Language scegli English o Italiano; la preferenza è salvata nell’archivio delle sessioni e riletta al caricamento. Anche gli archivi precedenti senza preferenza usano l’inglese. Il cambio aggiorna l’interfaccia senza tradurre conversazioni, nomi, codice, skill o risultati degli strumenti e senza imporre la lingua delle risposte del modello. I dialoghi di sistema e i messaggi provenienti dai server possono seguire la propria lingua. La guida HTML segue la lingua selezionata.

Il controllo «Consenti comandi shell» è disabilitato finché non sono selezionati una directory di lavoro e «Consenti azioni del modello». La modalità di approvazione automatica non abilita da sola gli strumenti. Il fix 1.4.8.04 è incluso senza una build intermedia.

Android 0.1.5 offre la stessa scelta IT/EN, con inglese predefinito e preferenza salvata nell’archivio del dispositivo. Nessuna sincronizzazione automatica con il Mac.

## Comandi prolungati · 1.4.9.02

Il modello avvia un job con `run_command`, poi ne legge lo stato con `command_status`, attende con `wait_command` e lo interrompe con `stop_command`. Il superamento dell’attesa non termina il processo; non esiste più la scadenza fissa di 30 secondi. La durata totale è facoltativa (`timeout_seconds`). L’avvio segue i permessi della sessione; le consultazioni dello stesso job non richiedono una nuova approvazione né consumano il limite delle azioni.

Il pannello «Comandi della sessione» mostra stato e output espandibile. Interrompi ferma il turno e il comando; i nuovi messaggi restano in coda fino alla conclusione dei job. La coda di output è limitata a 256 KiB per job, fino a 64 KiB per lettura, con omissioni esplicitamente indicate; per log integrali occorre un file. I job non vengono ricollegati dopo il riavvio. I comandi che richiedono input continuano a usare il terminale inline.

Il contesto predefinito delle nuove configurazioni è 16000 token. I limiti già salvati o dichiarati dal server remoto hanno precedenza; per i server che non dichiarano una finestra, il pannello del contesto indica il valore assunto di 16000 token, modificabile.
