# SUC Studio 1.4.7.02 — build 29

macOS Universal client correction. The parent inference graph, model files, official chat template and memory policy are unchanged from qualified engine 1.4.7. No new CLI/server engine release or new compression is implied.

## Changes

- Read response bytes on a non-UI executor and deliver bounded lines to the MainActor parser. Preserve cumulative 16-MiB stream and 4-MiB line/body limits, UTF-8, CRLF, multiline events, final EOF, authentication, health probes and user cancellation.
- Derive prefill work from uncached tokens. Explicit server rates retain priority; llama.cpp prompt_n already denotes evaluated work. Complete cache reuse does not invent a rate. Full prompt usage is retained separately; new optional archived Metrics fields are backward compatible.
- Correct current Qwen3.8 model-card and summary hard score to 100/120 at chunk 64, with subset counts from the recorded 19 October evaluation. Chunk 1024 scored 98/120. This is not a new full hard-suite run on all models.

## Measurements

Optimized actual RemoteInference client on Intel, identical synthetic burst text hashes; no GPU/model or SwiftUI window:

| SSE size | 1.4.7 reader wall time | 1.4.7.02 reader wall time |
|---|---:|---:|
| 256 KB | 4.676 s | 0.264 s |
| 1 MB | 20.125 s | 0.837 s |
| 4 MB | 66.237 s | 3.410 s |

These single diagnostic runs establish a large parser improvement in the controlled setup, not universal end-to-end model speed. Both with/without-I/O-log variants are recorded. Baseline/fixed JSONs retain byte counts, hashes, server-send timestamps and heartbeat measurements.

Running M1 Ultra server, original 1.4.7 build 28, Qwen3.8 side8, configured context 115000: HTTP TTFT was 20.31 s for 2178 input tokens and 108.95 s for 11692. Follow-ups (2206/11720 input tokens) took 20.00/109.91 s, reporting cached_tokens=0. Stream completion followed the last text within about 0.03 s. Input/output are synthetic; the API key and conversation text are not stored in results.

The canonical Qwen chat template strips reasoning in past assistant turns while generation opens a thinking marker. Natural templated chat can therefore miss the strict live-prefix cache. This release does not change the template or claim to eliminate full-history prefill. The earlier 32k anomaly was n-gram preparation (0.255/48.704 s), with GPU prefill essentially equal (94.345/94.498 s); cache/order and worker scheduling require a separate controlled investigation.

See [diagnostic plan/evidence](LONG_CHAT_DIAGNOSTICS.md), [unchanged engine qualification](FINAL_QUALIFICATION_1.4.7.md) and [reference KL](REFERENCE_KL.md).

## Validation and installation

Installer: SUCStudio/dist/SUC-Studio-1.4.7.02-universal.pkg. Version 1.4.7.02, build 29, macOS 15+, Universal app/helpers and ARM-only local engine; Intel remains remote-only. Existing bundle ID, user data and Keychain identity are retained. Locally ad-hoc signed, no Developer ID/notarization. Install manually after closing the app; the running server was not replaced automatically.

Final Intel/ARM regression, manifest/signature and package-checksum status is recorded in SUCStudio/docs/QA_REPORT.md and LONG_CHAT_DIAGNOSTICS.md. No physical-machine interactive GUI acceptance or repeat of all model hard suites is claimed by automated parser tests.
