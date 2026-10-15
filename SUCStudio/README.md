# SUC Studio

Italian version: [README-IT.md](README-IT.md)

Current version: **1.4.5_Final** — installer `dist/SUC-Studio-1.4.5_Final-arm64.pkg`.

Native macOS app (SwiftUI) for running SUC1 models with the SUC C++ / Metal runtime: local chats, an OpenAI-compatible server, remote servers, model tools and web search. Target: Apple Silicon, macOS 15+. The app's interface is in Italian.

- **[USAGE.md](USAGE.md)** — installation, setup, usage and troubleshooting.
- **[docs/QA_REPORT.md](docs/QA_REPORT.md)** — checks performed and known limits.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — structure and the private app ↔ engine protocol.
- **[docs/WEB_TOOLS_SPEC_IT.md](docs/WEB_TOOLS_SPEC_IT.md)** — web tools specification (Italian).

## Main features

- **Local models** in SUC1 format with an automatic memory plan: fully resident when they fit the GPU memory, otherwise MoE experts streamed from the SSD.
- **Chat** with sessions, branches, export / import, Markdown, separate reasoning, edit and regenerate, attachments (text, documents, PDF, images for vision backends).
- Built-in **OpenAI-compatible server** (API key, optional LAN access) and a **remote client** for SUC Studio, `suc1-server` or any engine with an OpenAI-compatible API (ds4, LM Studio, llama.cpp, LLM routers), with telemetry profiles.
- **Model tools**: work directory, file and shell tools, interactive terminal, artifacts, MCP, skills, subagents, hooks, plugins and routines, with manual or automatic approvals and an operation log.
- **Web search** (`web_search`, `visit_page`) through a hidden built-in browser, with a globe button to watch it live.

## Build

```sh
./scripts/build.sh
./scripts/test.sh
./scripts/package.sh
```

The build uses **a single engine source: `../cpp` of the parent SUC-LM project**, so `SUCStudio/` must stay inside the repository. `Sources/` holds the app, `Engine/` the IPC / memory / HTTP wrapper, `dist/` the app and installers, `.build/` a disposable cache. Every build hashes the engine sources, recompiles what changed (uncommitted changes included) and records the inputs in `dist/ENGINE_BUILD_MANIFEST.json`, which ships in the bundle. The built app and package are self-contained: the target Mac does not need the repository. On the development Mac the build needs `SDKROOT=…/MacOSX26.5.sdk` (the SDK 27 of the Command Line Tools lacks the SwiftUI macro plugin).

Model weights and tokenizers are not included (see [huggingface.co/Ar3ntiz](https://huggingface.co/Ar3ntiz)). Local distribution, ad-hoc signed, no Developer ID or notarization.

## Releases (newest first)

### 1.4.5_Final
- **Hidden** web browser: no window over your work and no focus stealing; timings identical to ds4 (off-screen panel not throttled by WebKit, measured).
- **Globe button** next to the chat settings, shown only during a web action: phase, URL and seconds; clicking it opens the browser with the live page and a status bar.
- Google verification signalled by the orange button, never brought to the front; no empty window left when the action ends.

### 1.4.5_Beta
- Web tools in **every session** that allows them, even without a work directory and with model actions off (dedicated prompt listing only `web_search` and `visit_page`).
- Without a work directory, pages are saved in `~/Library/Application Support/SUC Studio/web`.
- Action requests are also accepted inside a single `json` code block.

### 1.4.5
- Web tools for the model: `web_search` (alias `google_search`) and `visit_page`, with the ds4 web-module logic on WebKit.
- Dedicated, persistent browser profile that can be cleared from Settings; start confirmation once per run and a per-session switch.
- Full page saved in `.suc-web`; Google verifications never bypassed; only http and https URLs.

### 1.4.4
- Long remote generations with a health check.
- Memory budget up to 96 % with a more precise KV reserve.
- Inline terminal only for interactive model requests; diagnostic I/O log off by default; sampling legend and work directory below the chat.

### 1.4.3
- Reasoning separated from the answer, code editor opened by three backticks, resizable composer.
- Asynchronous rendering and previews keep long inputs responsive.
- Telemetry profiles for SUC Studio, DS4, LM Studio, llama.cpp, LLM Router and custom JSON, with measured and estimated values kept apart.

### 1.4.2
- Interactive terminal in the chat for commands, confirmations and secret input, also usable by the model with the session's permissions.
- Persistent artifacts for produced files, opened and saved on the Mac.

### 1.4.1
- Expandable operation log (queue, state and outcome of the last 200 operations).
- The chat follows the bottom and stops on manual scrolling; scrollable session list.

### 1.4.0
- Configurable limit of consecutive actions.
- Action JSON hidden from the transcript but kept in the model context and the log; final activity summary and immediate error reporting.

### 1.3.0
- Approvals inside the chat; model-loading indicator in every section; distinct animations for prefill and decode.
- Remote-server telemetry with real speeds when available.

### 1.2.0
- Image attachments for vision backends, with preview and persistence.
- Edit and regenerate in the same chat; alternating Local server / Remote server panels; hook management with examples.

### 1.1.0
- Chat links open in the system browser; revised Markdown rendering (streaming, code, tables, lists, quotes).

### 1.0.0
- Appearance section: separate fonts and sizes for interface and chat.
- Launch at login and keep the Mac awake.

### 0.9.0
- Plugins installable from local folders or a remote marketplace, with enable and remove.
- Scheduled routines with prompt, frequency, state and history.

### 0.8.1
- Memory plan from the real Metal working set minus an OS margin: DeepSeek and GLM run resident again instead of streaming from SSD.
- GLM KV estimate on the MLA latents; reserve KV + 2 GiB + 3 % of the file.

### 0.8.0
- Subtasks and subagents with separate contexts; delegation limits and inherited permissions.
- Configurable hooks for work-cycle events.

### 0.7.0
- Per-session work directory with file and command tools.
- Manual or automatic approval, action log and model-driven tool loop.

### 0.6.0
- Chat connectable to the local engine or to a remote OpenAI-compatible server (model, API key, streaming, stop).

### 0.5.0
- Global system prompt, selectable skills, local (stdio) and remote (HTTP) MCP servers; credentials in the Keychain.
- Collapsible section bar and offline HTML guide.

### 0.4.0
- C++ engine built directly from the parent project, with a hash-based build cache and manifest; new icon.

### 0.3.0
- Attachments: text files of any extension, Word DOC/DOCX, OpenDocument, RTF and text PDFs, with preview and export.

### 0.2.0
- Automatic memory management, resident or with an expert budget; SSD streaming of MoE experts; pre-load checks for models that are too large.

### 0.1.0
- First version: local chats and sessions, model / tokenizer settings, OpenAI-compatible server with API key and LAN access, menu bar and engine telemetry.

Full history: `dist/CHANGELOG.md` (Italian); older installers stay in `dist/`.

## Credits

Web extraction scripts and browsing logic adapted from ds4 (https://github.com/antirez/ds4), MIT License, Copyright (c) 2026 The ds4.c authors. Full text in [licenses/ds4-MIT.txt](licenses/ds4-MIT.txt).
