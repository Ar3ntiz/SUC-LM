# SUC Studio 1.4.9 — Setup and usage

SUC Studio is a Universal macOS app for Apple Silicon and Intel Macs, requiring macOS 15 or later.
Apple Silicon supports the SUC C++/Metal engine and remote servers. Intel Macs use remote OpenAI-compatible servers.

## Installation and updates

Open `SUC-Studio-1.4.9-universal.pkg` and follow the installer. The app is installed in Applications. This local distribution uses ad-hoc app signing without Developer ID or notarization. Existing sessions and settings remain in the same data directory; do not delete that directory during an update. The installer does not start the app or load a model automatically.

To build from source, keep `SUCStudio` inside the parent SUC-LM project and run `SUCStudio/scripts/build.sh`, then `SUCStudio/scripts/package.sh`. The app uses the parent `cpp` engine, without a duplicated engine tree. Python can be selected with `PYTHON_BIN`.

## Language

The app starts in **English**, independently of the macOS language. In **Settings → Language**, choose **English** or **Italiano**. The change applies immediately and is saved in the same session archive as other settings, with the usual backup behavior. Older archives without a language preference also default to English.

Changing language does not translate existing messages, session names, instructions, skills, tool results or code. It does not force the model's response language. Native system dialogs and server-generated messages may follow their own language settings.

## Choose a backend

In **Server → Chat engine**, select **Local server** or **Remote server**. The corresponding configuration is shown.

For local inference on Apple Silicon, choose a local SUC1 model and its matching `tokenizer.json` in Settings, set the context size and load the model. Files remain in their original locations. The loading indicator shows the engine phase; memory residency is an observation, not a completion percentage. The model is ready only when confirmed by the engine.

For a remote server, enter its API base URL, model ID and API key if required. Use **Verify / models**, select the model and save. SUC Studio, DS4, LM Studio, llama.cpp and OpenAI-compatible routers can be used. HTTP is suitable only for trusted networks; prefer HTTPS for external servers. Prompts and attachments are sent to the selected server. API keys are stored in Keychain and excluded from chat exports.

The telemetry profile controls supported server extensions. Real prefill/decode rates require server timings. **Reception ≈** is a client reception estimate, not engine throughput. Long generations remain cancellable; independent health checks do not repeat the request.

## Chat and sessions

Create, rename, pin, archive, search, import or export sessions from the chat sidebar. Chats are saved locally with backups. Drag documents or images into the composer, or use the attachment button. Text extraction is automatic for supported documents, including Word, OpenDocument and PDF. Vision requires a compatible remote model/server.

You can resize the composer upward and paste code through the code editor. Reasoning is shown separately from the answer. Scrolling upward detaches automatic following; use **Jump to bottom** to reattach. Long histories load progressively. Edit and regenerate reuse the current session.

Messages sent during a response enter the session queue. Pause/resume, remove or return queued items to draft from its panel. On restart, queues require explicit resumption. Stop or errors pause the queue.

## Working directory and shell

In session settings, choose a working directory and enable **Allow model actions**. **Allow shell commands** is disabled until both prerequisites are present. Automatic approvals do not enable tools by themselves; manual approvals appear inline in chat.

File tools are constrained to the working directory. Shell commands run with your account permissions and may access other locations. An interactive terminal appears inline only when the model requests a command requiring input. Sensitive input is masked and is not sent to the model. Commands can be expanded in chat; published files are available in the Artifacts side panel.

## Context and commands

The context indicator shows usage and its limit. Remote limits are detected when available or may be set per session; approximate counts are marked. `/compact` summarizes active context while preserving history; auto-compact runs at 99% or earlier when a response reserve is needed. Failed compaction retains the previous context.

Use `/help` to see commands, including `/context`, `/status`, `/new`, `/clear`, `/stop`, `/queue`, `/pause`, `/resume`, `/skills`, `/plugins`, `/model`, `/workspace`, `/export` and `/retry`. Use `//` to send text beginning with `/`.

## Skills, MCP, plugins and web

Enable skills in Settings and invoke them with `@name` or `$name`; the model initially sees their descriptions and loads instructions on demand. MCP servers may use local stdio or remote Streamable HTTP. Select the correct executable/arguments or endpoint, then connect. Schemas are limited to 3 MiB and results to 4 MiB, without silent truncation. Unambiguous server aliases are accepted.

Plugins can include skills, hooks and MCP together. Imported components are disabled until reviewed. Hooks need explicit trust; editing revokes it. Shell hooks and commands retain your account permissions. Routines run only while the app is open and inherit session approvals; they are not background system jobs.

Web search is orchestrated by the app independently of the inference backend. Enable web tools in session settings. The globe button opens the integrated browser, including when a site requires a CAPTCHA. Automatic approval permits browser startup; manual approval waits inline. Pages are untrusted data.

## Local API server

Load a local model, then start the server in the Server tab. Choose access from this Mac or the local network and configure the port. LAN clients require the displayed API key. Supported routes include `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`, with SSE streaming. Chat and API share one loaded model.

## Diagnostics and data

The **Log** panel is disabled by default. Enable it only for diagnostics to see a bounded in-memory tail of model input/output. Server/tool output and raw diagnostic content are not translated. Settings also provide data-folder access, backup recovery, font/theme controls, launch at login and keep-awake options. No engine, model format or memory-policy changes are introduced in 1.4.9.

## Android 0.1.5

Android is a remote chat client. In Settings, choose English or Italiano; the default is English and the preference is saved in the device archive. It supports sessions, documents/images, web search, text skills and remote HTTP MCP. It does not include a local engine, shell, terminal, hooks, routines or notifications. Android data is not automatically synced with the Mac.
