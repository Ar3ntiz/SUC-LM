"""`suc` command-line entry point."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="suc")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="run a Phase 0 baseline benchmark from a YAML config")
    bench.add_argument("config")
    analyze = sub.add_parser("analyze", help="run the Phase 1 weight-structure analysis from a YAML config")
    analyze.add_argument("config")
    calibrate = sub.add_parser("calibrate", help="run Phase 2 calibration and sensitivity from a YAML config")
    calibrate.add_argument("config")
    quantize = sub.add_parser("quantize", help="quality track: encode a model with each codec and evaluate it")
    quantize.add_argument("config")
    allocate = sub.add_parser("allocate", help="Phase 7: Fisher-guided greedy allocation, validated on the full model")
    allocate.add_argument("config")
    compress = sub.add_parser("compress", help="unattended pipeline: HF checkpoint -> SUC1 file + quality report "
                                               "(resumable; rerun the same config to continue)")
    compress.add_argument("config")
    ev = sub.add_parser("eval", help="answer a test suite with SUC1 (C++ / Python) and reference systems, grade it "
                                     "and compare with a reference system (JSON + markdown)")
    ev.add_argument("config", nargs="?")
    ev.add_argument("--regrade", metavar="RESULT_JSON", help="re-grade a stored suc eval result (no model)")
    perf = sub.add_parser("perf", help="prefill / TTFT / decode speed and memory of several systems in one session "
                                       "(JSON + markdown)")
    perf.add_argument("config")
    perf.add_argument("--compare-to", metavar="RESULT_JSON", help="previous suc perf result to compare row by row")
    queue = sub.add_parser("queue", help="unattended multi-model queue: HF check -> disk space -> download -> verify "
                                         "-> suc compress -> report, resumable (docs/queue.md)")
    queue.add_argument("queue_file")
    queue.add_argument("--once", action="store_true", help="exit when only models awaiting a decision remain")
    queue.add_argument("--dry-run", action="store_true", help="HF check + space estimate + config per model, no download")
    queue.add_argument("--status", action="store_true", help="print SUMMARY.md of the queue")
    queue.add_argument("--stop", action="store_true", help="SIGTERM the running orchestrator of this queue")
    repack = sub.add_parser("repack", help="rewrite a SUC1 file with its raw lookup tables (qwen4_exp n-gram shards) "
                                           "as row-wise affine; every other tensor copied byte for byte")
    repack.add_argument("source")
    repack.add_argument("output")
    repack.add_argument("--bits", type=int, default=4, choices=(2, 4, 8))
    repack.add_argument("--scale-dtype", default="f16", choices=("f16", "bf16"))
    repack.add_argument("--no-verify", action="store_true", help="skip the checksum / byte-identity / row checks")
    repack.add_argument("--output-dir", default="experiments/results", help="where the result JSON is written")
    gen = sub.add_parser("generate", help="greedy generation from a SUC1 file (Python/MLX runtime) with load time, "
                                          "memory and speed")
    gen.add_argument("file")
    gen.add_argument("--prompt")
    gen.add_argument("--tokens", help="comma-separated prompt token ids (no tokenizer needed)")
    gen.add_argument("--steps", type=int, default=32)
    gen.add_argument("--tokenizer", help="tokenizer directory or HF id (default: the model named in the file)")
    gen.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    gen.add_argument("--decoded-head", action="store_true",
                     help="decode an affine embedding / LM head to the model dtype instead of running from codes")
    gen.add_argument("--separate", action="store_true", help="one module per projection (no grouping / fusion)")
    gen.add_argument("--expert-budget", type=float, default=None,
                     help="stream the routed experts from the file through an expert cache of this many bytes "
                          "(suc.runtime.expert_streaming; e.g. 100e9)")
    gen.add_argument("--hot-experts", default="auto",
                     help="hot_experts.json of the compress run to pin (default: the one next to the SUC1 file)")
    gen.add_argument("--json", dest="json_out")
    args = parser.parse_args(argv)

    if args.command == "bench":
        from suc.cli.bench import run

        print(run(args.config))
    elif args.command == "analyze":
        from suc.analysis.weights import run_analysis

        for path in run_analysis(args.config):
            print(path)
    elif args.command == "calibrate":
        from suc.analysis.calibrate import run_calibration

        print(run_calibration(args.config))
    elif args.command == "quantize":
        from suc.quant.run import run_quantize

        print(run_quantize(args.config))
    elif args.command == "allocate":
        from suc.allocation.run import run_allocate

        print(run_allocate(args.config))
    elif args.command == "compress":
        from suc.pipeline.compress import run_compress

        print(run_compress(args.config))
    elif args.command == "eval":
        if args.regrade:
            from suc.cli.eval import regrade

            raise SystemExit(regrade(args.regrade))
        if not args.config:
            parser.error("suc eval: a config (or --regrade RESULT_JSON) is required")
        from suc.cli.eval import run as run_eval

        print(run_eval(args.config))
    elif args.command == "perf":
        from suc.cli.perf import run as run_perf

        print(run_perf(args.config, compare_to=args.compare_to))
    elif args.command == "repack":
        from suc.io.repack import repack_cli

        print(repack_cli(args.source, args.output, bits=args.bits, scale_dtype=args.scale_dtype,
                         verify=not args.no_verify, output_dir=args.output_dir))
    elif args.command == "generate":
        import json

        from suc.cli.generate import run as run_generate

        rec = run_generate(args.file, prompt=args.prompt, tokens=args.tokens, steps=args.steps,
                           tokenizer=args.tokenizer, affine_modules=not args.decoded_head, group=not args.separate,
                           chat=args.chat, json_out=args.json_out, expert_budget=args.expert_budget,
                           hot_experts=None if args.hot_experts == "none" else args.hot_experts)
        print(json.dumps(rec, indent=2, sort_keys=True))
    elif args.command == "queue":
        from suc.orchestrate.cli import main as run_queue

        raise SystemExit(run_queue(args.queue_file, once=args.once, dry=args.dry_run, show_status=args.status,
                                   stop=args.stop))


if __name__ == "__main__":
    main()
