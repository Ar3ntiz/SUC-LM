"""Cross-check of the C++/Metal deepseek_v4 graph (cpp/src/metal/deepseek_v4.cpp, `suc1-generate` on a deepseek_v4 file)
with the Python/MLX SUC1 runtime of the same file (suc.runtime.suc1_model.load_suc1_model(group=False,
affine_modules=False): one SucLinear per projection, SucSwitchLinear / MXFP4SwitchLinear / SucSwitchMixed expert stacks,
decoded embedding / LM head = the C++ execution plan).

Files: tiny random DeepSeek-V4 checkpoints in the original FP8 / MXFP4 layout (tests/unit/tiny_models.py: hidden 256,
4 layers with compress ratios [0, 4, 16, 4] = window / CSA + indexer / HCA / CSA, one hash-routed layer, 8 experts
top-2 + shared expert, 4 hyper-connection streams), encoded tensor by tensor and written with write_suc1:
  attention projections  vq / additive with rotations, column int8 and CSR fp16 residuals (wo_a as a SUC module: every
                         row group on every input group, the diagonal taken);
  routed experts          a different codec per expert in every stack: VQ K=16 G=2 (+ int8 column residual), VQ K=64
                         G=4, additive M=3 K=16 G=4 -> mixed-codec stacks (SucSwitchMixed);
  "native" variant        additionally every 4th expert as the verbatim checkpoint MXFP4 record (representation mxfp4:
                         4 codecs in one stack), and the last layer entirely native (a pure MXFP4 stack);
  embedding / LM head     affine 4-bit; kept tensors (norms, hyper-connection F32 parameters, sinks, ape, compressor /
                         indexer projections, router, tid2eid) raw as the source store returns them.
Three shapes: "tiny" (head_dim 128: MLX sdpa_vector; experts K = 256 / 128: MLX fp_gather_qmv), "wide" (head_dim 512,
hidden / moe_intermediate 512: MLX's unfused head-dim-512 attention and fp_gather_qmv_fast) and "real" (the
DeepSeek-V4-Flash widths where kernels branch: hidden 4096 = 16384-wide streams, head_dim 512, 64 indexer heads of 128,
experts 2048 x 4096, a ratio-32 compressor (pooling over 32 rows); 2 layers, 4 experts, mostly native experts).

Per position (token-by-token decode, teacher-forced on the C++ token sequence, prompt + generated tokens well into the
indexer's top-k regime): logits and every stream after the embedding and after every block are compared, the
free-running greedy tokens must be identical.

Skipped unless suc1-generate is built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.io.safetensors_store import SafetensorsStore
from suc.io.suc1 import write_suc1
from suc.models.adapters import get_adapter
from suc.quant.codecs import AdditiveCodebook, AffineRTN, MXFP4Native, VectorCodebook
from suc.quant.residual import ResidualCodec
from suc.runtime.suc1_model import greedy, kl_rows, load_suc1_model, run_tokens
from tiny_models import tiny_deepseek_v4_config, write_tiny_deepseek_v4

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
GENERATE = BUILD / "suc1-generate"
HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
DSV4 = HUB / "models--deepseek-ai--DeepSeek-V4-Flash" / "snapshots"

pytestmark = pytest.mark.skipif(not GENERATE.exists(), reason=f"suc1-generate not built in {BUILD} (run cpp/build.sh)")

# 8 indexer heads: with 4, an all-ReLU-zero indexer score (exact ties broken differently by argpartition) is frequent.
TINY = {"index_n_heads": 8}
# Widths where the kernels branch: head_dim 512 (the real one), expert in-features 512 (fp_gather_qmv_fast).
WIDE = {"index_n_heads": 8, "hidden_size": 512, "head_dim": 512, "num_attention_heads": 2, "q_lora_rank": 256,
        "o_lora_rank": 128, "moe_intermediate_size": 512, "index_head_dim": 128}
# The DeepSeek-V4-Flash widths (2 layers: ratio 4 with the indexer, ratio 32; 4 experts top-2; 2 attention heads).
REAL = {"hidden_size": 4096, "head_dim": 512, "num_attention_heads": 2, "q_lora_rank": 256, "o_groups": 2,
        "o_lora_rank": 128, "moe_intermediate_size": 2048, "index_n_heads": 64, "index_head_dim": 128,
        "num_hidden_layers": 2, "compress_ratios": [4, 32, 0], "n_routed_experts": 4, "num_experts_per_tok": 2}
PROMPT = [5, 6, 7, 8]
STEPS = 36            # 40 positions: the CSA indexer selects top-4 of its entries from 20 tokens on


def _vq(k, g, rot_out=True, **kw):
    return VectorCodebook(k, block_size=g, rotate=True, rotate_out=rot_out, kmeans_iters=2, **kw)


# per-projection codecs of the attention / shared expert (role -> codec)
ATTN = {
    "self_attn.wq_a": lambda: ResidualCodec(_vq(16, 2), 0.02, "int8", structure="columns"),
    "self_attn.wq_b": lambda: AdditiveCodebook(16, block_size=4, num_codebooks=2, rotate=True, kmeans_iters=2),
    "self_attn.wkv": lambda: ResidualCodec(_vq(16, 4, rot_out=False), 0.01, "fp16"),
    "self_attn.wo_a": lambda: _vq(16, 2),
    "self_attn.wo_b": lambda: _vq(64, 4),
    "mlp.shared_experts.gate_proj": lambda: _vq(16, 2, rot_out=False),
    "mlp.shared_experts.up_proj": lambda: _vq(16, 2, rot_out=False),
    "mlp.shared_experts.down_proj": lambda: ResidualCodec(_vq(16, 2, rot_out=False), 0.02, "int8", structure="columns"),
}
# routed experts: codec by expert (mixed-codec stacks in every layer)
EXPERT = [
    lambda: ResidualCodec(_vq(16, 2, rot_out=False), 0.02, "int8", structure="columns"),
    lambda: _vq(64, 4),
    lambda: AdditiveCodebook(16, block_size=4, num_codebooks=3, rotate=True, rotate_out=False, kmeans_iters=2),
]


def ds4_file(path: Path, *, native: bool, over: dict, seed: int = 2, native_every: int = 4,
             affine_attn: bool = False) -> Path:
    """SUC1 file of a tiny deepseek_v4 checkpoint (see the module docstring). affine_attn: the attention and shared
    expert projections as affine 8-bit codes (group 64), as in the real DeepSeek-V4-Flash SUC1 files."""
    model_dir = write_tiny_deepseek_v4(path.parent / (path.stem + "_ckpt"), seed=seed, mtp=False, **over)
    cfg = json.loads((model_dir / "config.json").read_text())
    st = get_adapter(cfg).source_store(SafetensorsStore.from_dir(model_dir))
    L = cfg["num_hidden_layers"]
    t: dict = {}
    s = 1
    for n in st.names():
        s += 1
        role = n.split(".", 3)[3].removesuffix(".weight") if n.startswith("model.layers.") else n
        if n in ("model.embed_tokens.weight", "lm_head.weight"):
            t[n] = AffineRTN(4, 64).encode(st.load(n).astype(mx.float32), seed=s)
        elif ".mlp.experts." in n:
            layer, e = int(n.split(".")[2]), int(n.split(".")[5])
            if native and (e % native_every == 0 or layer == L - 1):
                kind, w8, s8 = st.load_source(n)
                assert kind == "mxfp4"
                t[n] = MXFP4Native().encode_source(w8, s8)
            else:
                t[n] = EXPERT[e % 3]().encode(st.load(n).astype(mx.float32), seed=s)
        elif role in ATTN and affine_attn:
            t[n] = AffineRTN(8, 64).encode(st.load(n).astype(mx.float32), seed=s)
        elif role in ATTN:
            t[n] = ATTN[role]().encode(st.load(n).astype(mx.float32), seed=s)
        else:
            a = st.load(n)
            t[n] = a
    write_suc1(path, architecture=cfg, tokenizer_id="none", tensors=t)
    return path


@pytest.fixture(scope="module", params=["tiny", "wide", "real"])
def ds4_files(request, tmp_path_factory):
    """{"base": SUC1 path (SUC codecs only), "native": the same with native MXFP4 experts}."""
    tmp = tmp_path_factory.mktemp(f"ds4cpp_{request.param}")
    over = {"tiny": TINY, "wide": WIDE, "real": REAL}[request.param]
    every = 2 if request.param == "real" else 4
    return {"base": ds4_file(tmp / "base.suc1", native=False, over=over),
            "native": ds4_file(tmp / "native.suc1", native=True, over=over, native_every=every),
            "affine": ds4_file(tmp / "affine.suc1", native=True, over=over, native_every=every, affine_attn=True),
            "kind": request.param}


def _cpp(path: Path, tmp: Path, *args: str, tokens=PROMPT, steps=STEPS) -> tuple[list[int], dict, str]:
    j = tmp / "out.json"
    r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, tokens)), "--steps", str(steps),
                        "--json", str(j), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    gen = [int(x) for x in r.stdout.split("generated:")[1].split()[0].split(",")]
    return gen, json.loads(j.read_text()), r.stderr


@pytest.mark.parametrize("variant", ["base", "native"])
def test_graph_matches_python_runtime(ds4_files, variant, tmp_path):
    path = ds4_files[variant]
    gen, rec, _ = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "l.npy"), "--hidden-out", str(tmp_path / "h.npy"))
    assert rec["model_type"] == "deepseek_v4"
    assert rec["expert_stacks"]["mixed"] > 0
    if variant == "native":
        assert rec["expert_stacks"]["native_experts"] > 0 and rec["expert_stacks"]["native_only"] > 0
    lc, hc = np.load(tmp_path / "l.npy"), np.load(tmp_path / "h.npy")
    seq = PROMPT + gen[:-1]
    assert lc.shape[0] == hc.shape[0] == len(seq)
    model, _, rep = load_suc1_model(path, group=False, affine_modules=False)
    kinds = {c["kind"] for c in rep["converted"]}
    assert "SucSwitchMixed" in kinds
    lp, hp = run_tokens(model, seq)
    kl = kl_rows(lp, lc)
    err_l, err_h = np.abs(lp - lc), np.abs(hp - hc)
    same_l, same_h = (lp == lc).mean(), (hp == hc).mean()
    print(f"\n[{ds4_files['kind']}/{variant}] positions {len(seq)}: logits max {err_l.max():.3g} mean "
          f"{err_l.mean():.3g} bit-identical {same_l:.4f}; KL max {kl.max():.3g} mean {kl.mean():.3g}; streams max "
          f"{err_h.max():.3g} mean {err_h.mean():.3g} bit-identical {same_h:.4f}")
    # Every stream after every block and the logits are bit-identical (the kernels follow MLX's operation order and
    # rounding, cpp/README.md milestone 6), so any difference is a bug, not a tolerance question.
    assert same_l == 1.0 and same_h == 1.0
    assert (lp.argmax(-1) == lc.argmax(-1)).all()
    assert greedy(model, PROMPT, STEPS) == gen


def test_row_rotation_layout(ds4_files, tmp_path, monkeypatch):
    """Tuning cache with the rotation "row" layout (the M1 Ultra's choice for n = 1024: wide/native failed there with
    one flipped bf16 element of wq_b's 1024-row output while the port ran "blocks"): bit-identical to Python."""
    from tiny_models import row_rotation_cache
    row_rotation_cache(monkeypatch, tmp_path / "tuning.json")
    path = ds4_files["native"]
    gen, _, _ = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "l.npy"), "--hidden-out", str(tmp_path / "h.npy"))
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    lp, hp = run_tokens(model, PROMPT + gen[:-1])
    assert np.array_equal(hp, np.load(tmp_path / "h.npy")) and np.array_equal(lp, np.load(tmp_path / "l.npy"))


def test_mixed_routing_all_rows_equals_routed(ds4_files, tmp_path):
    """--mixed-experts all (every sub-stack on every routed row, the Python SucSwitchMixed semantics) gives the same
    bits as the default (each row only through the sub-stack holding its expert)."""
    path = ds4_files["native"]
    a, _, _ = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "a.npy"), steps=12)
    b, _, _ = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "b.npy"), "--mixed-experts", "all", steps=12)
    assert a == b and np.array_equal(np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"))


@pytest.mark.parametrize("variant", ["base", "native"])
def test_streamed_experts_equal_resident(ds4_files, variant, tmp_path):
    """--expert-budget on a deepseek_v4 file (mixed-codec stacks; native: + MXFP4 experts and an all-native layer): the
    routed experts read into slot pools of top-k + 1 slots per layer (evictions on every step), with and without the
    next-layer prefetch and with chunked prefill: logits, streams and tokens bit-identical to the resident run."""
    path = ds4_files[variant]
    a, _, _ = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "a.npy"), "--hidden-out", str(tmp_path / "ha.npy"),
                   steps=16)
    _, rec, _ = _cpp(path, tmp_path, "--expert-budget", str(1 << 34), steps=2)
    st = rec["expert_streaming"]
    k = 2   # num_experts_per_tok of every shape
    slot, layers = st["slot_bytes_max"], st["layers"]
    assert layers == {"tiny": 4, "wide": 4, "real": 2}[ds4_files["kind"]]
    budget = layers * (k + 1) * slot
    for extra in ([], ["--expert-prefetch", "--expert-policy", "lru"]):
        b, rec, _ = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "b.npy"), "--hidden-out",
                         str(tmp_path / "hb.npy"), "--expert-budget", str(budget), *extra, steps=16)
        st = rec["expert_streaming"]
        assert a == b
        assert np.array_equal(np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"))
        assert np.array_equal(np.load(tmp_path / "ha.npy"), np.load(tmp_path / "hb.npy"))
        assert st["pool_bytes"] <= budget and st["slots"] == layers * (k + 1)
        assert st["misses"] > 0 and st["evictions"] > 0 and st["bytes_read"] > 0 and st["hit_rate"] < 1.0
        if extra:
            assert st["prefetch_issued"] > 0
    prompt = [3, 9, 1, 4, 4, 17, 2, 1, 1, 30, 5, 6]
    c, _, _ = _cpp(path, tmp_path, "--chunk", "5", tokens=prompt, steps=6)
    d, rec, _ = _cpp(path, tmp_path, "--chunk", "5", "--expert-budget", str(budget), tokens=prompt, steps=6)
    assert c == d and rec["expert_streaming"]["evictions"] > 0
    r = subprocess.run([str(GENERATE), str(path), "--tokens", "5,6", "--steps", "2", "--expert-budget", str(budget),
                        "--mixed-experts", "all"], capture_output=True, text=True)
    assert r.returncode != 0 and "routed" in r.stderr


@pytest.mark.parametrize("variant", ["base", "native"])
def test_split_gpu_views_bit_identical(ds4_files, variant, tmp_path):
    """Files above maxBufferLength (DeepSeek-V4-Flash: 107.4 GB > 86.6 GB on the M1 Ultra) are mapped as several no-copy
    views, forced here with --max-view-bytes. The views are cut only between mapping units (a layer's segments of one
    payload section, so expert stacks and fused projections stay inside one view) and share at most one page: residency
    is charged per view, and the former half-overlapping views (1.4x the file, above the GPU working set) left the first
    command buffer of the real file unscheduled forever. Tokens and logits are bit-identical to one view."""
    path = ds4_files[variant]
    a, rec, _ = _cpp(path, tmp_path, "--logits-out", str(tmp_path / "a.npy"), steps=8)
    assert rec["gpu_views"] == 1
    page = resource.getpagesize()
    size = -(-path.stat().st_size // page) * page
    split = 0
    for frac in (0.6, 0.5, 1 / 3, 1 / 4, 1 / 6):
        cap = int(size * frac) // page * page
        r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, PROMPT)), "--steps", "8",
                            "--json", str(tmp_path / "v.json"), "--logits-out", str(tmp_path / "b.npy"),
                            "--max-view-bytes", str(cap)], capture_output=True, text=True)
        if r.returncode != 0:           # a mapping unit larger than the cap: a clear error, never a wrong mapping
            assert "does not fit in one GPU view" in r.stderr, r.stderr
            break
        rec = json.loads((tmp_path / "v.json").read_text())
        b = [int(x) for x in r.stdout.split("generated:")[1].split()[0].split(",")]
        assert rec["gpu_views"] >= 2
        assert rec["gpu_view_bytes"] <= size + (rec["gpu_views"] - 1) * page
        assert a == b
        assert np.array_equal(np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"))
        split += 1
    assert split >= 1


def test_chunked_prefill_and_teacher_forcing(ds4_files, tmp_path):
    path = ds4_files["base"]
    prompt = [3, 9, 1, 4, 4, 17, 2, 1, 1, 30, 5, 6, 7, 8, 11, 12, 40, 41, 42, 43, 44, 45]
    a, rec, _ = _cpp(path, tmp_path, "--chunk", "5", tokens=prompt, steps=8)
    b, _, _ = _cpp(path, tmp_path, "--prefill", "token", tokens=prompt, steps=8)
    assert rec["chunk"] == 5 and a == b
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    assert greedy(model, prompt, 8) == a
    gen, rec, _ = _cpp(path, tmp_path, "--force", "1,2,3,4,5", steps=5)
    assert rec["teacher_forced"] and len(gen) == 5
    r = subprocess.run([str(GENERATE), str(path), "--tokens", "5,999"], capture_output=True, text=True)
    assert r.returncode != 0 and "vocab" in r.stderr


LONG_PROMPT = [int(x) for x in np.random.default_rng(7).integers(2, 300, size=45)]


@pytest.mark.parametrize("variant", ["base", "native", "affine"])
def test_batched_prefill_bit_identical(ds4_files, variant, tmp_path, monkeypatch):
    """Row-exact batched (layer-major) prefill (--prefill exact) = token-by-token prefill: the last prompt token's logits and streams (sha256), and
    the greedy continuation, for chunks that split compressor groups (7), the whole prompt (64) and forced 16-row
    expert tiles (padding rows)."""
    path = ds4_files[variant]
    ref, rr, _ = _cpp(path, tmp_path, "--prefill", "token", tokens=LONG_PROMPT, steps=6)
    assert rr["prefill"] == "token"
    runs = [("7", None), ("64", None), ("64", "16"), ("5", "2")]
    for chunk, bt in runs:
        if bt:
            monkeypatch.setenv("SUC_TILE_BT", bt)
        else:
            monkeypatch.delenv("SUC_TILE_BT", raising=False)
        gen, rec, _ = _cpp(path, tmp_path, "--prefill", "exact", "--chunk", chunk, tokens=LONG_PROMPT, steps=6)
        assert rec["prefill"] == "batch" and rec["chunk"] == int(chunk)
        assert rec["prefill_logits_sha256"] == rr["prefill_logits_sha256"], (chunk, bt)
        assert rec["prefill_streams_sha256"] == rr["prefill_streams_sha256"], (chunk, bt)
        assert gen == ref
    monkeypatch.delenv("SUC_TILE_BT", raising=False)
    # streamed experts, top-k + 1 slots per layer: a chunk needs several fetch groups per layer
    _, rec, _ = _cpp(path, tmp_path, "--expert-budget", str(1 << 34), steps=2)
    st = rec["expert_streaming"]
    budget = st["layers"] * 3 * st["slot_bytes_max"]
    gen, rec, _ = _cpp(path, tmp_path, "--prefill", "exact", "--chunk", "16", "--expert-budget", str(budget),
                       tokens=LONG_PROMPT, steps=6)
    assert rec["prefill_logits_sha256"] == rr["prefill_logits_sha256"]
    assert rec["prefill_streams_sha256"] == rr["prefill_streams_sha256"]
    assert gen == ref and rec["prefill_chunk_fetches"] > st["layers"] * 3


def _row_kl(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    def ls(x):
        x = x.astype(np.float64)
        m = x.max(-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
    a, b = ls(ref), ls(got)
    return (np.exp(a) * (a - b)).sum(-1)


@pytest.mark.parametrize("variant", ["base", "native", "affine"])
def test_fast_batched_prefill_close_to_token(ds4_files, variant, tmp_path, monkeypatch):
    """Default batched prefill (simdgroup-matrix GEMMs, MMA expert tiles, batched compressors and chunk attention; not
    bit-identical): the logits of every prompt position (--prefill batch --logits-out: every chunk row through the head)
    and of the teacher-forced continuation stay within a small KL of the token-by-token prefill and agree on the top-1
    almost everywhere, for chunks that split compressor groups (7), the whole prompt (64) and forced 16-row tiles."""
    path = ds4_files[variant]
    ref, rr, _ = _cpp(path, tmp_path, "--prefill", "token", "--logits-out", str(tmp_path / "t.npy"),
                      tokens=LONG_PROMPT, steps=6)
    lt = np.load(tmp_path / "t.npy")
    for chunk, bt in [("7", None), ("64", None), ("64", "16"), ("16", "8")]:
        if bt:
            monkeypatch.setenv("SUC_TILE_BT", bt)
        else:
            monkeypatch.delenv("SUC_TILE_BT", raising=False)
        gen, rec, _ = _cpp(path, tmp_path, "--prefill", "batch", "--chunk", chunk, "--logits-out",
                           str(tmp_path / "b.npy"), "--force", ",".join(map(str, ref)), tokens=LONG_PROMPT, steps=6)
        lb = np.load(tmp_path / "b.npy")
        assert rec["prefill"] == "batch" and lb.shape == lt.shape, (lb.shape, lt.shape)
        kl = _row_kl(lt, lb)
        top1 = (lt.argmax(-1) == lb.argmax(-1)).mean()
        # bounds of the tiny random models (bf16 roundings of GEMM vs gemv accumulation orders, amplified by near-tie
        # expert routing); the chunk attention alone is exact (SUC_FAST_OFF=ptr: KL 0). Real-model sizes:
        # scripts/prefill_batch_vs_token.py (report §1c).
        assert kl.mean() < 2e-2 and kl.max() < 0.3, (chunk, bt, kl.mean(), kl.max(), int(kl.argmax()))
        assert top1 >= 0.85, (chunk, bt, top1)
    monkeypatch.delenv("SUC_TILE_BT", raising=False)


@pytest.mark.parametrize("variant", ["base", "native"])
def test_fast_prefill_rows_and_indexer(ds4_files, variant, tmp_path, monkeypatch):
    """Fast batched prefill with the GEMM / tile / router parts row-exact (SUC_FAST_OFF=ptr): the row-batched
    hyper-connection / routing / MoE-output launches (bit-identical to their per-row launches: SUC_FAST_OFF=ptrh gives
    the same logits) and the chunk attention with the batched lightning-indexer selection (rows past index_topk
    entries; the tiny prompt reaches them from 20 tokens on) stay at the rounding level of the token-by-token prefill
    (the chunk attention's sequential online softmax vs the decode sdpa kernel: one-ulp differences, amplified at
    single positions by near-tie routing)."""
    path = ds4_files[variant]
    ref, _, _ = _cpp(path, tmp_path, "--prefill", "token", "--logits-out", str(tmp_path / "t.npy"),
                     tokens=LONG_PROMPT, steps=2)
    lt = np.load(tmp_path / "t.npy")
    for chunk in ("16", "64"):
        out = {}
        for off in ("ptr", "ptrh"):
            monkeypatch.setenv("SUC_FAST_OFF", off)
            _cpp(path, tmp_path, "--prefill", "batch", "--chunk", chunk, "--logits-out", str(tmp_path / "b.npy"),
                 "--force", ",".join(map(str, ref)), tokens=LONG_PROMPT, steps=2)
            out[off] = np.load(tmp_path / "b.npy")
        assert (out["ptr"] == out["ptrh"]).all(), chunk
        kl = _row_kl(lt, out["ptr"])
        # rounding level; a selected row's keys are summed in another grouping than the decode step's sdpa_vector
        # (device-dependent), and the tiny random model's near-tie routing can amplify that at a single position
        # (M1 Ultra: one row of 46 at KL 0.026)
        assert kl.mean() < 5e-3 and kl.max() < 0.1, (chunk, kl.mean(), kl.max(), int(kl.argmax()))
        assert (lt.argmax(-1) == out["ptr"].argmax(-1)).mean() >= 0.95, chunk


def test_chunk_attention_variants_bit_identical(tmp_path):
    """The chunk attention variants of head_dim 512 (decode roundings, RND = 1) give the same bits: one head per
    simdgroup (SUC_DS4_ATTN_HPS=1), two heads per simdgroup sharing each key load (default) with the scores kept in
    threadgroup memory or recomputed (SUC_DS4_ATTN_KEEP=0); 8 heads, rows with and without the indexer selection."""
    path = ds4_file(tmp_path / "h8.suc1", native=True, over=dict(WIDE, num_attention_heads=8))
    out = []
    for env in ({"SUC_DS4_ATTN_HPS": "1"}, {}, {"SUC_DS4_ATTN_KEEP": "0"}):
        r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, LONG_PROMPT)), "--steps", "1",
                            "--prefill", "batch", "--chunk", "16", "--logits-out", str(tmp_path / "x.npy")],
                           capture_output=True, text=True, env={**os.environ, **env})
        assert r.returncode == 0, r.stderr
        out.append(np.load(tmp_path / "x.npy"))
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[0], out[2])


def _long_run(tmp_path, heads: int, positions: int):
    """(Python logits, C++ logits) of a teacher-forced run over random tokens on a wide-shape file with `heads`
    attention heads and a sliding window of 8192 (so the key count grows with the position)."""
    over = dict(WIDE, sliding_window=8192, num_attention_heads=heads)
    path = ds4_file(tmp_path / f"long_h{heads}.suc1", native=True, over=over)
    toks = [int(x) for x in np.random.default_rng(1).integers(2, 300, size=positions)]
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    lp, _ = run_tokens(model, toks, hidden=False)
    r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, toks)), "--steps", "1", "--logits-out",
                        str(tmp_path / "c.npy"), "--max-context", str(len(toks) + 2), "--quiet"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return lp, np.load(tmp_path / "c.npy")


@pytest.mark.parametrize("heads", [8, 64])
def test_folded_attention_heads(tmp_path, heads):
    """MLX's unfused head_dim-512 attention folds the H query heads of the KV head into one matmul: 8 heads = q' Kᵀ
    through gemv_wide on Apple GPU generation >= 15 (two passes, 32 then 16 lanes past 64 keys) or a steel GEMM before
    it, 64 heads (the real model) = steel GEMM (split-k up to 512 keys); p V = steel GEMM (split-k from 512 keys on).
    600 positions (keys up to 600 + entries): logits bit-identical at every position."""
    lp, lc = _long_run(tmp_path, heads, 600)
    same = (lp == lc).all(-1)
    assert same.all(), f"first differing position {int(np.argmin(same))}"


@pytest.mark.skipif(os.environ.get("SUC_LONG_CONTEXT") != "1", reason="long-context check (~4 min per case on the "
                    "M3): set SUC_LONG_CONTEXT=1")
@pytest.mark.parametrize("heads", [2, 64])
def test_long_context_bit_exact(tmp_path, heads):
    """>= 8k keys: 8,400 positions with a sliding window of 8192 (the window layer attends 8,192 keys from position
    8,191 on, the ratio-4 layer ~10,300): MLX's looped softmax (> 4096 keys, 1024 threads), split-k p V over 8k+ keys,
    the indexer over 2k+ entries. Logits bit-identical at every position."""
    lp, lc = _long_run(tmp_path, heads, 8400)
    same = (lp == lc).all(-1)
    assert same.all(), f"first differing position {int(np.argmin(same))}"


def _dsv4_tokenizer() -> Path | None:
    snaps = sorted(DSV4.glob("*/tokenizer.json")) if DSV4.is_dir() else []
    return snaps[-1] if snaps else None


DSV4_REPO, DSV4_REV = "deepseek-ai/DeepSeek-V4-Flash-0731", "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def _dsv4_encoder():
    """encode_messages of the checkpoint's encoding/encoding_dsv4.py at the pinned revision (HF cache; fetched when
    online), or None."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    path = None
    for local in (True, False):
        try:
            path = hf_hub_download(DSV4_REPO, "encoding/encoding_dsv4.py", revision=DSV4_REV, local_files_only=local)
            break
        except Exception:   # not cached / offline
            continue
    if path is None:
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dsv4_encoding", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.encode_messages


@pytest.mark.skipif(_dsv4_tokenizer() is None, reason="DeepSeek-V4-Flash tokenizer not cached")
def test_dsv4_chat_prompt_ids(ds4_files):
    """--chat on a deepseek_v4 file = encoding_dsv4.encode_messages of the checkpoint (pinned revision; one user turn,
    optional system turn, thinking / chat mode, reasoning effort low / high / max) tokenized by HF tokenizers the way
    inference/generate.py does (tokenizer.encode; add_bos_token is false, the BOS comes from the text)."""
    tk = pytest.importorskip("tokenizers")
    enc = _dsv4_encoder()
    if enc is None:
        pytest.skip(f"{DSV4_REPO}@{DSV4_REV[:8]} encoding/encoding_dsv4.py not cached and not downloadable")
    path = _dsv4_tokenizer()
    tok = tk.Tokenizer.from_file(str(path))
    cases = [("Ciao, come stai?", None, False, None), ("  Hello world \n", "You are helpful.", True, None),
             ("12345 numbers 3.14159", None, True, "high"), ("日本語のテキスト", "Sys", False, "max"),
             ("Explain sparse attention.", "Be brief.", True, "max"), ("x", "", True, "low")]
    for prompt, system, think, effort in cases:
        msgs = ([{"role": "system", "content": system}] if system is not None else []) + \
            [{"role": "user", "content": prompt}]
        text = enc(msgs, thinking_mode="thinking" if think else "chat", reasoning_effort=effort)
        ref = tok.encode(text, add_special_tokens=False).ids
        assert ref == tok.encode(text).ids   # no post-processor tokens: generate.py's tokenizer.encode gives the same
        args = [str(GENERATE), str(ds4_files["base"]), "--prompt", prompt, "--chat", "--tokenizer", str(path),
                "--print-prompt", "--quiet"]
        args += (["--system", system] if system is not None else []) + ([] if think else ["--no-think"])
        args += ["--reasoning-effort", effort] if effort else []
        r = subprocess.run(args, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert [int(x) for x in r.stdout.split("prompt:")[1].split()[0].split(",")] == ref, (prompt, system, think, effort)
    r = subprocess.run([str(GENERATE), str(ds4_files["base"]), "--prompt", "x", "--chat", "--tokenizer", str(path),
                        "--print-prompt", "--reasoning-effort", "xhigh"], capture_output=True, text=True)
    assert r.returncode != 0 and "low|high|max" in r.stderr
