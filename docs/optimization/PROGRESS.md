# Engine 1.4.7 qualification status

Completed on 19 October 2026. All six selected models passed full same-reference
KL equality and all twelve 8k/32k paired checks. Actual CLI/server/bundled-engine
smoke, bounded session/cache/cancellation tests and Universal suites passed.
Model files and memory policy are unchanged. No main-branch merge or physical
build-28 GUI acceptance is implied. MTP is a separate post-release project.

See [release and setup](RELEASE_1.4.7.md),
[final qualification](FINAL_QUALIFICATION_1.4.7.md),
[reference KL](REFERENCE_KL.md), [performance](COMBINED_RESULTS.md) and
[inference behavior](INFERENCE_1.4.7.md). Internal host/queue logs and raw
machine-specific result JSONs are excluded from this public source snapshot.

## Studio 1.4.7.03 chat checkpoints

Qualified Qwen hybrid prefix checkpoints retain bounded recurrent state without duplicating KV rows or changing the existing SUC template. The first request and earlier-prefix changes still require full prefill. See [checkpoint evidence](CHAT_PREFIX_CHECKPOINTS.md) and [release/setup](RELEASE_1.4.7.03.md).

## Studio 1.4.7.04 progress and live telemetry

Actual helper/SSE progress, model-loading phases, cancellation/replay and server-token live metrics passed. Universal build and extracted installer were verified; GUI acceptance remains manual. See [release/setup](RELEASE_1.4.7.04.md).
