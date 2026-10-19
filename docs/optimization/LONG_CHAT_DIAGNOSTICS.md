# Long-chat latency investigation — 19 October 2026

Reported setup: macOS client, SUC Studio 1.4.7 build 28 server on M1 Ultra, Qwen3.8-Flash-Next side8. Installed engine manifest matches the qualified revision. Current server settings: context 115000, batched prefill. Model and server remain running; no reload, memory-policy change or concurrent GPU benchmark is authorized by this investigation.

- [x] L1 — Verify actual installed build, selected model, health and operational settings without reading private conversations.
- [x] L2 — Correct current model-card/README hard score from historical 97/120 to 100/120 at chunk 64 (98/120 at chunk 1024). Preserve historical paired statistics.
- [x] L3 — Measure raw loopback HTTP vs actual optimized macOS RemoteInference SSE parser, with and without the I/O log. Record server-send end, client completion, event counts and main-actor heartbeat. This isolates parser/log cost and does not emulate a real SwiftUI window.
- [x] L4 — Measure the running authenticated server sequentially with synthetic short/long chat inputs, bounded output and matching follow-up turns. Record TTFT, received-last-token time, completion time, server prefill/decode times and cached-token counts. Credentials must not enter command lines, files or results. The owner supplied the API key for the tests after SSH Keychain access returned interaction-not-allowed. It was entered through a hidden terminal prompt, kept in process memory only and omitted from results. The probe process has exited.
- [x] L5 — Inspect the old 32k anomaly using per-stage traces and make the safety decision: do not load a second 96-GB model while the owner’s server is running. A controlled warm/cold baseline rerun is deferred, not claimed passed. The single earlier pair was 96.6 s baseline vs 149.7 s candidate, both chunk 1024; this is not a repeated throughput estimate and must not be dismissed as qualified performance.
- [x] L6 — Report the measured bottleneck and separate client, server, cache/profile and network effects. No inference-code change or unmeasured performance claim is part of the diagnostic stage.

Initial code findings are hypotheses: RemoteInference runs its per-byte SSE loop on MainActor; archive mutation, Markdown display and optional model I/O logging can compete for UI time. The live cache only reuses strict extensions of the entire actually-fed prefix. Qwen service chunk 64 differs from the CLI's historical 1024 profile; token/s from those paths cannot be compared as identical configurations. Health currently reports reachability, not proof of active request progress.

## Measured findings

- Optimized Intel macOS client, identical synthetic SSE text hashes: 256 KB / 1 MB / 4 MB old-reader wall times 4.676 / 20.125 / 66.237 s; line-reader outside MainActor 0.264 / 0.837 / 3.410 s. These are parser/log probes, not GPU throughput or visual GUI acceptance. I/O logging was not the primary bottleneck in the isolated tests.
- Running M1 server, unchanged SUC Studio 1.4.7 build 28 and context 115000: synthetic 2178-token chat TTFT 20.31 s; 2206-token follow-up 20.00 s; 11692-token chat 108.95 s; 11720-token follow-up 109.91 s. All reported cached_tokens=0. Raw HTTP completion followed the last text by at most about 0.03 s. Raw evidence: LONG_CHAT_LIVE_SERVER.json, without request/response text or credentials.
- Qwen template inspection explains a systematic obstacle to strict-prefix reuse: generation starts with a thinking marker (an empty closed thinking block when disabled), while past assistant turns are rendered with reasoning stripped. These token sequences need not extend the actually-fed prefix. The raw aligned-token cache test from 1.4.7 did not qualify natural templated-chat cache hits. No template or hybrid-state change is made silently in this client release.
- The old 32k anomaly is CPU n-gram preparation, not slower Metal: baseline/candidate GPU prefill 94.3446/94.4977 s, n-gram prefill 0.255473/48.7044 s. Candidate ran first at this length, so cold file/page-cache effects are confounded with the changed gather scheduling. Both logits and continuations matched. A controlled warm/cold rerun remains separate work; loading a second 96-GB model while the user's server is active is excluded.
- macOS RemoteTelemetry also lacked cached-token subtraction. Version 1.4.7.02 corrects it while retaining explicit server-rate priority, llama.cpp evaluated-token semantics, missing-field compatibility and the archived full prompt count. New optional Metrics fields decode existing archives without migration.

## 1.4.7.02 delivery

- [x] F1 — Replace per-byte UI-actor reads with bounded non-UI line/body reading; retain EOF, UTF-8, cancellation and byte limits.
- [x] F2 — Correct cache-aware prefill in RemoteTelemetry and Metrics fallback.
- [x] F3 — Complete Universal functional/security regressions and verify source/bundled-client identity.
- [x] F4 — Build app/helpers Universal plus unchanged ARM engine, package build 29, verify signatures/metadata/checksums, and sync main/source/M1. Do not install over the running app automatically.

The inference C++/Metal graph, official chat template, model files and memory policy are unchanged. This client release fixes stream-drain overhead and metric reporting; it does not claim to remove the measured full-prefill cost of templated Qwen chat or complete natural-chat cache reuse.

Final validation: the complete Universal suite passed on Intel and M1 Ultra (native ARM and simulated remote-only guards; auth/SSE/cancel/EOF/UTF-8/limits/cache metrics; real scheduled WebKit tools; skill/hook/plugin/MCP HTTP regressions). The M1 burst probe additionally verified the corrected ARM client. The rebuilt bundled inference helper has SHA256 fa3dc9e896261203b66bb21189a05c3e2925b4a99983d97a0879d2f42226d6e4, identical to the installed 1.4.7 helper; parent source content SHA256 remains f6bb844d6d5d03e35e2cffc673a500b3a1c68bacbd41abd58037feebc3af577f. No model-quality rerun is attributed to the client-only correction. Physical GUI acceptance of the new build remains manual.
