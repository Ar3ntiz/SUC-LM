# Real MTP qualification (separate from prompt lookup)

Status: investigated, not implemented or benchmarked. The `--lookup-window` experiment is not an MTP implementation.

Scheduling decision (owner, 2026-10-19): deferred until the current optimization release has completed qualification and delivered final macOS app, CLI and standalone server builds, with updated documentation. No MTP implementation, head conversion or MTP benchmark is required to close that release. Resume this plan afterwards, separately for each compatible model.

## Selected target

Use the existing selected Qwen3.8-Flash-Next side8 SUC1 as the target. Do not substitute a network-placeholder model or benchmark another checkpoint without the owner's agreement.

Its original checkpoint index contains 31 `mtp.*` tensors spread across 28 shards. Existing project architecture notes estimate about 2.61B MTP-specific parameters (5.21 GB in BF16). Current SUC compression drops these tensors; the selected SUC1 therefore cannot enable MTP with a runtime flag.

## Required stages

- [x] Identify MTP keys in the selected original checkpoint metadata without loading or benchmarking checkpoint weights.
- [x] Identify a public implementation to resolve the previously uncertain architecture mapping.
- [ ] Pin and review the predictor graph and shared embedding/head semantics; verify exact tensor dimensions from checkpoint headers.
- [ ] Extract only the selected model's MTP tensors to local M1 storage, preserve hashes/revision and build a separate sidecar. Do not duplicate or replace the selected main SUC1.
- [ ] Implement the predictor with target hidden-state input and explicit shared embedding/output projection ownership.
- [ ] Implement verification and rollback for Qwen's recurrent states, convolution state, KV/indexer state, n-gram context and predictor state. KV truncation alone is insufficient.
- [ ] Validate predictor logits against the reference before throughput tests.
- [ ] Validate greedy token identity under the qualified verifier; for temperature sampling implement the actual probability-ratio acceptance/correction algorithm. Greedy agreement alone does not qualify sampled MTP.
- [ ] Measure draft cost, verification cost, accepted tokens, tokens per cycle, end-to-end tok/s, TTFT, RAM and KV overhead on M1 at several context lengths and draft depths.
- [ ] Run held-out quality/hard-suite checks, cancellation, streaming and multi-turn regressions. Enable only when gains exceed noise and memory/quality gates pass.

## References and limits

- Project inventory and architecture uncertainties: `docs/arch/qwen4_exp.md` (MTP sections).
- Predictor implementation reference: https://github.com/ggml-org/llama.cpp/pull/28243 (closed PR; pin code before reuse, do not assume it is merged).
- Converter author's sidecar documentation: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/raw/main/MTP/README.md . Its reported speedups are from a B200, not an M1, and are not SUC results. GGUF sidecars cannot be loaded by the current SUC1 runtime without a new loader; no such download or benchmark has been performed.

MTP is a separate implementation milestone. Do not label prompt lookup or multiple command buffers in flight as MTP, and do not report the MTP stage as completed until the predictor and state rollback are actually exercised on M1.
