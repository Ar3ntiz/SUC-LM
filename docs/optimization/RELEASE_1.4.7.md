# SUC Studio / CLI / server 1.4.7 — qualified test-branch delivery

Completed 19 October 2026 on M1 Ultra 128 GiB. All final evidence gates passed. Delivery remains on `test`; no merge to `main`, installation or external publication was performed.

## Packages and setup

- `SUCStudio/dist/SUC-Studio-1.4.7-universal.pkg`: SUC Studio 1.4.7 build 28, macOS 15+, app/helpers arm64+x86_64, local engine arm64. Intel uses a remote OpenAI-compatible server. Open the installer manually to install/update the app; no model or tokenizer is included. Existing data/credentials retain their identity.
- `SUCStudio/dist/SUC-CLI-Server-1.4.7-arm64.tar.gz`: standalone Apple Silicon/Metal tools. These particular binaries target **macOS 26.0+**, verified from their Mach-O load commands; rebuild from source for a different supported deployment target. Extract the archive and follow its `USAGE.md`; `BUILD.json` records binary hashes and minimum OS.
- Verify both downloaded/copied files from `SUCStudio/dist` with `shasum -a 256 -c SHA256SUMS-1.4.7.txt`. Packages are locally distributed, ad-hoc signed where applicable, without Developer ID/notarization. No automatic installation is part of validation.

In the app, select the matching local SUC1/tokenizer on Apple Silicon, or configure Server remoto for Intel/remote use. The app's server provides API-key authentication for LAN use. The standalone server defaults to loopback and has no API-key layer; use the app's server for authenticated LAN access. Existing SUC1 files and matching tokenizers are reusable.

## Provenance

Standalone and bundled engines were compiled from `24b2ad7ff7aed65b7b41160ab35f6df8738e3c88`; shared parent C++/Metal input SHA256 `f6bb844d6d5d03e35e2cffc673a500b3a1c68bacbd41abd58037feebc3af577f`. Final generation CLI, inspector and tokenizer are byte-identical to the `6693b2c` binaries used for the combined matrix/long-context qualification. Later documentation and packaging commits do not change those compiled inputs. See `CLI_BINARY_CONTINUITY.json`, `SUCStudio/dist/ENGINE_BUILD_MANIFEST.json` and `CLI_SERVER_BUILD.json`.

## Acceptance

- 1220 CPU checks, 52 request-parameter checks and 84 Python regressions passed; one strict expected failure preserves the rejected half-TCQ quality gate.
- Swift native and simulated remote-only platform, HTTP/SSE/cancellation, scheduled real web extraction, routine, plugin/skill/hook and MCP HTTP suites passed.
- Seven real-model checks passed on final shipped CLI/server/bundled app engine: generation, loading, completion/usage, streaming, authenticated app server and cancellation.
- All six models have identical complete same-reference KL and raw logits on 16×512 validation rows (8176 scored positions each). [REFERENCE_KL.md](REFERENCE_KL.md) records both mean and global p95.
- All twelve 8192/32768-token paired checks passed final-logit and greedy-continuation equality, with no increase in the compared engine allocation fields. These bounded checks are not full-context BF16 reference evaluation.
- Actual bundled-engine Qwen8B/Qwen3.8 sessions passed strict prefix reuse, divergent/equal/shorter requests, cancellation, eight replay cycles and unload/reload. Maximum observed replay-footprint growth was 0.20 MiB; process footprint is not total unified memory.
- DeepSeek/GLM default service chunks 64/512 match historical-kernel CLI oracles. DeepSeek service-64 additionally passed the full reference protocol with identical raw logits.
- Universal architecture slices, ad-hoc signature, metadata and system-library dependencies verified.

## Measured performance and scope

[COMBINED_RESULTS.md](COMBINED_RESULTS.md) contains the complete six-model matrix, three measured alternating pairs after warmup. Qwen8B unfiltered T=0.7 decode: 36.71→67.51 tok/s. Qwen3.8 2k greedy prefill: 341.57→366.07 tok/s; decode: 29.23→33.92. DeepSeek/GLM 2k prefill: 169.37→182.19 and 116.73→127.83 tok/s. Other gains/regressions are small and may be noise. No universal speedup, improved compression or new MLX/GGUF comparison is claimed.

[FINAL_QUALIFICATION_1.4.7.md](FINAL_QUALIFICATION_1.4.7.md) records memory/context/session results; [INFERENCE_1.4.7.md](INFERENCE_1.4.7.md) explains sampler semantics, service/CLI profiles, live-prefix ownership, TCQ defaults and A/B flags. Sampling distribution is preserved, but token-ID CDF order can change seeded sampled text. Memory policy and model payloads remain unchanged. Manual build-28 GUI acceptance on physical Intel/other Apple Silicon machines is not claimed. N-gram parallelism, half TCQ and prompt lookup were rejected as defaults; query tiling remains opt-in. MTP is deferred.
