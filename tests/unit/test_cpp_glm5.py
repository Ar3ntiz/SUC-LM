"""Cross-check of the C++/Metal glm5_next graph (cpp/src/metal/glm5.cpp, `suc1-generate` on a glm5_next file) with the
Python/MLX SUC1 runtime of the same file (suc.runtime.suc1_model.load_suc1_model(group=False, affine_modules=False):
one SucLinear per projection, SucSwitchLinear / SucSwitchMixed expert stacks, decoded embedding / LM head = the C++
execution plan), resident and with SSD expert streaming.

Files: tiny random GLM-5.3-Flash-structured checkpoints in the original FP8 layout (tests/unit/tiny_models.py), encoded
tensor by tensor and written with write_suc1:
  KDA / MLA / MLP / shared-expert projections  vq / additive with rotations, int8 column and fp16 CSR residuals, one
                                              projection per kind as affine 8-bit (the bf16-decoded dense path);
  routed experts                              a different codec per expert in every stack: VQ K=16 G=2 (+ int8 column
                                              residual), VQ K=64 G=4, additive M=3 K=16 G=4 -> mixed-codec stacks;
  embedding / LM head                         affine 4-bit; kept tensors (norms, mHC, KDA gate factors / convs / A_log /
                                              dt_bias, kv_b, indexer, router, correction bias) raw as the source store
                                              returns them.
Shapes: "tiny" (hidden 128 = 512-wide streams, KDA head_dim 32 (row sums of <= 64 values: one thread), MLA rank 128
(MLX sdpa_vector)), "real" (the GLM-5.3-Flash widths where kernels branch: hidden 4096 = 16384-wide streams, KDA
head_dim 128, MLA kv_lora_rank 512 / nope 256 / v 256 (MLX's unfused attention), 32 indexer heads of 128, experts
2048 x 4096; 2 layers [KDA + MoE, MLA + DSA + MoE], 4 experts top-2) and "heads64" (the real 64 MLA heads, whose
fallback attention MLX folds into GEMMs, over 564 positions: split-k and fused GEMM orders, the indexer GEMM past 128
pools). Every shape also runs in float32 (`--dtype f32` against the Python model in float32), where any difference in
a float32 summation order shows up.

Per position (token-by-token decode, teacher-forced on the C++ token sequence, prompt + generated tokens well into the
DSA indexer's top-k regime: k-pool 4, index_topk 8, so the pool selection is active from position 11): logits and every
stream after the embedding and after every block are compared bit for bit, the free-running greedy tokens must be
identical. Expert streaming: budgets of top-k + 1 slots per MoE layer (evictions on every step), LRU / SLRU, prefetch,
a pinned hot set: same bits as resident.

Skipped unless suc1-generate is built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.io.safetensors_store import SafetensorsStore
from suc.io.suc1 import write_suc1
from suc.models.adapters import get_adapter
from suc.quant.codecs import AdditiveCodebook, AffineRTN, VectorCodebook
from suc.quant.residual import ResidualCodec
from suc.runtime.suc1_model import greedy, kl_rows, load_suc1_model, run_tokens
from tiny_models import tiny_glm5_next_config, write_tiny_glm5_next

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
GENERATE = BUILD / "suc1-generate"
HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
GLM = HUB / "models--zai-org--GLM-5.3-Flash" / "snapshots"

pytestmark = pytest.mark.skipif(not GENERATE.exists(), reason=f"suc1-generate not built in {BUILD} (run cpp/build.sh)")

TINY: dict = {}
# The GLM-5.3-Flash widths (2 layers: KDA + MoE, MLA + DSA + MoE; 2 KDA / MLA heads; 4 experts top-2).
REAL = {"hidden_size": 4096, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
        "q_lora_rank": 256, "kv_lora_rank": 512, "qk_nope_head_dim": 256, "qk_head_dim": 256, "v_head_dim": 256,
        "moe_intermediate_size": 2048, "intermediate_size": 2048, "n_routed_experts": 4, "num_experts_per_tok": 2,
        "index_n_heads": 32, "index_head_dim": 128, "first_k_dense_replace": 0,
        "mlp_layer_types": ["sparse", "sparse"], "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "indexer_types": ["full", "full"],
        "linear_attn_config": {"num_heads": 2, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                               "kda_layers": [0], "full_attn_layers": [1]}}
# 64 MLA heads on the one latent KV head (the real count: MLX folds the heads of the fallback attention into one GEMM,
# split-k up to 512 keys, the fused kernel with the K tail first above), run over 564 positions.
HEADS64 = dict(REAL, hidden_size=1024, num_attention_heads=64, num_key_value_heads=64, moe_intermediate_size=512,
               intermediate_size=512)
PROMPT = [5, 6, 7, 8]
STEPS = 36            # 40 positions: the indexer selects 2 of its pools from position 11 on


def _vq(k, g, rot_out=True, **kw):
    return VectorCodebook(k, block_size=g, rotate=True, rotate_out=rot_out, kmeans_iters=2, **kw)


# per-projection codecs (role -> codec); everything not listed and not an expert stays as stored
PROJ = {
    "self_attn.q_proj": lambda: ResidualCodec(_vq(16, 2), 0.02, "int8", structure="columns"),
    "self_attn.k_proj": lambda: AdditiveCodebook(16, block_size=4, num_codebooks=2, rotate=True, kmeans_iters=2),
    "self_attn.v_proj": lambda: AffineRTN(8, 64),
    "self_attn.o_proj": lambda: ResidualCodec(_vq(16, 4, rot_out=False), 0.01, "fp16"),
    "self_attn.q_a_proj": lambda: _vq(16, 2),
    "self_attn.q_b_proj": lambda: ResidualCodec(_vq(16, 2, rot_out=False), 0.02, "int8", structure="columns"),
    "self_attn.kv_a_proj_with_mqa": lambda: _vq(64, 4),
    "mlp.gate_proj": lambda: _vq(16, 2, rot_out=False),
    "mlp.up_proj": lambda: _vq(16, 2, rot_out=False),
    "mlp.down_proj": lambda: ResidualCodec(_vq(16, 2), 0.02, "int8", structure="columns"),
    "mlp.shared_experts.gate_proj": lambda: _vq(16, 2, rot_out=False),
    "mlp.shared_experts.up_proj": lambda: AffineRTN(8, 64),
    "mlp.shared_experts.down_proj": lambda: ResidualCodec(_vq(16, 2, rot_out=False), 0.02, "int8", structure="columns"),
}
EXPERT = [
    lambda: ResidualCodec(_vq(16, 2, rot_out=False), 0.02, "int8", structure="columns"),
    lambda: _vq(64, 4),
    lambda: AdditiveCodebook(16, block_size=4, num_codebooks=3, rotate=True, rotate_out=False, kmeans_iters=2),
]


def glm5_file(path: Path, *, over: dict, seed: int = 2) -> Path:
    """SUC1 file of a tiny glm5_next checkpoint (see the module docstring)."""
    model_dir = write_tiny_glm5_next(path.parent / (path.stem + "_ckpt"), seed=seed, extras=False, **over)
    cfg = json.loads((model_dir / "config.json").read_text())
    st = get_adapter(cfg).source_store(SafetensorsStore.from_dir(model_dir))
    t: dict = {}
    s = 1
    for n in st.names():
        s += 1
        role = n.split(".", 3)[3].removesuffix(".weight") if n.startswith("model.layers.") else n
        if n in ("model.embed_tokens.weight", "lm_head.weight"):
            t[n] = AffineRTN(4, 64).encode(st.load(n).astype(mx.float32), seed=s)
        elif ".mlp.experts." in n:
            e = int(n.split(".")[5])
            t[n] = EXPERT[e % 3]().encode(st.load(n).astype(mx.float32), seed=s)
        elif role in PROJ:
            t[n] = PROJ[role]().encode(st.load(n).astype(mx.float32), seed=s)
        else:
            t[n] = st.load(n)
    write_suc1(path, architecture=cfg, tokenizer_id="none", tensors=t)
    return path


@pytest.fixture(scope="module", params=["tiny", "real"])
def glm_file(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"glm5cpp_{request.param}")
    over = {"tiny": TINY, "real": REAL}[request.param]
    return {"path": glm5_file(tmp / "model.suc1", over=over), "kind": request.param}


def _cpp(path: Path, tmp: Path, *args: str, tokens=PROMPT, steps=STEPS, ok: bool = True):
    j = tmp / "out.json"
    r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, tokens)), "--steps", str(steps),
                        "--json", str(j), *args], capture_output=True, text=True)
    if not ok:
        assert r.returncode != 0
        return r.stderr
    assert r.returncode == 0, r.stderr
    gen = [int(x) for x in r.stdout.split("generated:")[1].split()[0].split(",")]
    return gen, json.loads(j.read_text()), r.stderr


def _rows(path: Path, tmp: Path, *args: str, tokens=PROMPT, steps=STEPS):
    """(generated, logits, streams, json) of a run with per-position outputs."""
    gen, rec, _ = _cpp(path, tmp, "--logits-out", str(tmp / "l.npy"), "--hidden-out", str(tmp / "h.npy"), *args,
                       tokens=tokens, steps=steps)
    return gen, np.load(tmp / "l.npy"), np.load(tmp / "h.npy"), rec


def test_graph_matches_python_runtime(glm_file, tmp_path):
    path = glm_file["path"]
    gen, lc, hc, rec = _rows(path, tmp_path)
    assert rec["model_type"] == "glm5_next"
    assert rec["expert_stacks"]["mixed"] > 0
    seq = PROMPT + gen[:-1]
    assert lc.shape[0] == hc.shape[0] == len(seq)
    model, _, rep = load_suc1_model(path, group=False, affine_modules=False)
    assert "SucSwitchMixed" in {c["kind"] for c in rep["converted"]}
    lp, hp = run_tokens(model, seq)
    kl = kl_rows(lp, lc)
    err_l, err_h = np.abs(lp - lc), np.abs(hp - hc)
    same_l, same_h = (lp == lc).mean(), (hp == hc).mean()
    print(f"\n[{glm_file['kind']}] positions {len(seq)}: logits max {err_l.max():.3g} mean {err_l.mean():.3g} "
          f"bit-identical {same_l:.4f}; KL max {kl.max():.3g} mean {kl.mean():.3g}; streams max {err_h.max():.3g} "
          f"mean {err_h.mean():.3g} bit-identical {same_h:.4f}")
    bad = [(p, b) for p in range(len(seq)) for b in range(hp.shape[1]) if not np.array_equal(hp[p, b], hc[p, b])]
    assert not bad, f"first differing (position, stream index): {bad[:6]}"
    assert same_l == 1.0
    assert greedy(model, PROMPT, STEPS) == gen


def test_float32_activations(glm_file, tmp_path):
    """float32 activations (every float32 summation order visible) on each fixture shape (tiny, real widths)."""
    _check_dtypes(glm_file["path"], tmp_path, (("f32", mx.float32),), STEPS)


def test_bf16_and_float32_64_heads(tmp_path_factory, tmp_path):
    """The real 64 MLA heads over 564 positions, bf16 and float32 (its own fixture, built once: not a cross-product
    with glm_file — audit skip follow-up)."""
    path = glm5_file(tmp_path_factory.mktemp("glm5cpp_h64") / "model.suc1", over=HEADS64)
    _check_dtypes(path, tmp_path, (("bf16", None), ("f32", mx.float32)), 560)


def _check_dtypes(path: Path, tmp_path: Path, runs, steps: int) -> None:
    for name, dt in runs:
        gen, lc, hc, _ = _rows(path, tmp_path, *(["--dtype", "f32"] if dt is not None else []), steps=steps)
        model, _, _ = load_suc1_model(path, group=False, affine_modules=False, dtype=dt)
        lp, hp = run_tokens(model, PROMPT + gen[:-1])
        bad = np.argwhere((hp != hc).any(-1))
        assert not len(bad), f"{name}: first differing (position, stream index): {bad[:6].tolist()}"
        assert np.array_equal(lp, lc), name


def test_row_rotation_layout(glm_file, tmp_path, monkeypatch):
    """Tuning cache with the rotation "row" layout (the M1 Ultra's choice for n = 1024; HEADS64 failed there while
    the port ran "blocks"): projections and per-expert stack rotations follow it, bit-identical to Python."""
    from tiny_models import row_rotation_cache
    row_rotation_cache(monkeypatch, tmp_path / "tuning.json")
    path = glm_file["path"]
    gen, lc, hc, _ = _rows(path, tmp_path, steps=16)
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    lp, hp = run_tokens(model, PROMPT + gen[:-1])
    assert np.array_equal(hp, hc) and np.array_equal(lp, lc)


def test_chunked_prefill_and_teacher_forcing(glm_file, tmp_path):
    path = glm_file["path"]
    prompt = [3, 9, 1, 4, 4, 17, 2, 1, 1, 30, 5, 6, 7, 8, 11, 12, 40, 41, 42, 43, 44, 45]
    a, rec, _ = _cpp(path, tmp_path, "--prefill", "exact", "--chunk", "5", tokens=prompt, steps=8)
    b, rb, _ = _cpp(path, tmp_path, "--prefill", "token", tokens=prompt, steps=8)
    assert rec["chunk"] == 5 and rec["prefill"] == "exact" and a == b
    assert rec["prefill_logits_sha256"] == rb["prefill_logits_sha256"]
    assert rec["prefill_hidden_sha256"] == rb["prefill_hidden_sha256"]
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    assert greedy(model, prompt, 8) == a
    gen, rec, _ = _cpp(path, tmp_path, "--force", "1,2,3,4,5", steps=5)
    assert rec["teacher_forced"] and len(gen) == 5
    err = _cpp(path, tmp_path, tokens=[5, 999], ok=False)
    assert "vocab" in err


def _slot_bytes(path: Path, tmp: Path) -> tuple[int, int]:
    rec = _cpp(path, tmp, "--expert-budget", str(1 << 34), steps=2)[1]["expert_streaming"]
    return rec["slot_bytes_max"], rec["layers"]


@pytest.mark.parametrize("opts", [("--expert-policy", "lru"), ("--expert-policy", "slru"),
                                  ("--expert-prefetch", "--expert-io-threads", "2")])
def test_streamed_equals_resident(glm_file, opts, tmp_path):
    path = glm_file["path"]
    resident = _rows(path, tmp_path, steps=16)
    slot, layers = _slot_bytes(path, tmp_path)
    budget = layers * 3 * slot                       # top-2 + 1 slot per MoE layer: evictions on every step
    got = _rows(path, tmp_path, "--expert-budget", str(budget), *opts, steps=16)
    assert got[0] == resident[0]
    assert np.array_equal(got[1], resident[1]) and np.array_equal(got[2], resident[2])
    st = got[3]["expert_streaming"]
    assert st["pool_bytes"] <= budget and st["misses"] > 0 and st["evictions"] > 0
    assert st["requests"] == st["hits"] + st["misses"]
    if "--expert-prefetch" in opts:
        assert st["prefetch_issued"] > 0
    # chunked prefill (row-exact) streamed = resident tokens
    prompt = [3, 9, 1, 4, 4, 17, 2, 1, 1, 30, 5, 6, 7, 8]
    a = _cpp(path, tmp_path, "--prefill", "exact", "--chunk", "4", tokens=prompt, steps=6)[0]
    b = _cpp(path, tmp_path, "--prefill", "exact", "--chunk", "4", "--expert-budget", str(budget), *opts,
             tokens=prompt, steps=6)[0]
    assert a == b


@pytest.mark.parametrize("slots", [3, 0])
def test_streamed_layer_major_prefill(glm_file, slots, tmp_path):
    """Streamed experts, --prefill exact: the prompt runs in row-exact layer-major chunks (every layer runs all chunk tokens up to the router,
    then one fetch of the union of their routed experts, in token groups that fit the slots). Same bits as decode
    steps: the last prompt position's logits and streams (sha256) and the greedy continuation equal the resident
    token-by-token run; far fewer expert fetches. slots = 3: top-2 + 1 slots per MoE layer (a chunk splits into
    groups); 0: a budget above the routed experts (one fetch per MoE layer per chunk)."""
    path = glm_file["path"]
    prompt = [int(x) for x in np.random.default_rng(3).integers(2, 200, size=37)]
    ref = _cpp(path, tmp_path, "--prefill", "token", tokens=prompt, steps=8)
    assert ref[1]["prefill_mode"] == "decode_steps"
    slot, layers = _slot_bytes(path, tmp_path)
    budget = str(layers * slots * slot if slots else 1 << 34)
    tok = _cpp(path, tmp_path, "--expert-budget", budget, "--prefill", "token", tokens=prompt, steps=1)[1]
    for chunk in (["--chunk", "8"], []):
        gen, rec, _ = _cpp(path, tmp_path, "--expert-budget", budget, "--prefill", "exact", *chunk, tokens=prompt,
                           steps=8)
        assert rec["prefill_mode"] == "layer_major" and rec["chunk"] == (8 if chunk else 64)
        assert gen == ref[0]
        assert rec["prefill_logits_sha256"] == ref[1]["prefill_logits_sha256"]
        assert rec["prefill_hidden_sha256"] == ref[1]["prefill_hidden_sha256"]
        n_chunks = -(-len(prompt) // rec["chunk"])
        if slots:
            assert rec["prefill_chunk_fetches"] > layers * n_chunks       # groups
        else:
            assert rec["prefill_chunk_fetches"] == layers * n_chunks
        # prefill fetch calls: one per MoE layer per token (decode steps) vs per chunk group (layer major)
        assert rec["prefill_chunk_fetches"] < tok["expert_streaming"]["fetch_calls"] == layers * len(prompt)
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    assert greedy(model, prompt, 8) == ref[0]


LONG_PROMPT = [int(x) for x in np.random.default_rng(5).integers(2, 250, size=45)]


def _row_kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """KL(softmax(a) || softmax(b)) per row (float64)."""
    def ls(x):
        x = x.astype(np.float64)
        m = x.max(-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
    la, lb = ls(a), ls(b)
    return (np.exp(la) * (la - lb)).sum(-1)


@pytest.fixture(scope="module")
def glm_topk64(tmp_path_factory):
    """Tiny shape with index_topk 64 (16 pools): no DSA selection over the 50 positions, so every chunk runs the batched
    MLA attention and its k-pools cross chunk boundaries."""
    return glm5_file(tmp_path_factory.mktemp("glm5cpp_topk64") / "model.suc1", over=dict(TINY, index_topk=64))


@pytest.mark.parametrize("which", ["tiny", "real", "topk64"])
def test_fast_batched_prefill_close_to_token(glm_file, glm_topk64, which, tmp_path):
    """Default batched prefill (GEMM projections, MMA expert tiles, float32 router GEMM, row-batched mHC / KDA conv /
    gated-delta recurrence / k-pools, chunk MLA attention; not bit-identical): the logits of every prompt position
    (--prefill batch --logits-out) and of the teacher-forced continuation stay within a small KL of the token-by-token
    prefill, for chunks of 7 (row gemvs below 8 rows), 16 and the whole prompt, resident and with streamed experts
    (3 slots per MoE layer: row groups per chunk; the tiny shapes' index_topk 8 puts most rows in the DSA selection,
    topk64 none). With SUC_FAST_OFF=ptra (projections, expert tiles, router and MLA per row) the batched path is
    bit-identical to decode steps (the row-batched mHC / KDA / k-pool kernels are exact); with the chunk attention and
    indexer scores batched (ptr) nearly so (measured: bit-identical on tiny, KL <= 2e-5 on the real widths)."""
    if which == "topk64":
        if glm_file["kind"] != "tiny":
            pytest.skip("topk64 runs once")
        path = glm_topk64
    elif which != glm_file["kind"]:
        pytest.skip("one shape per fixture")
    else:
        path = glm_file["path"]
    slot, layers = _slot_bytes(path, tmp_path)
    for extra in ([], ["--expert-budget", str(layers * 3 * slot)]):
        ref, _, _ = _cpp(path, tmp_path, "--prefill", "token", "--logits-out", str(tmp_path / "t.npy"), *extra,
                         tokens=LONG_PROMPT, steps=6)
        lt = np.load(tmp_path / "t.npy")
        for chunk in ("7", "16", "64"):
            gen, rec, _ = _cpp(path, tmp_path, "--prefill", "batch", "--chunk", chunk, "--logits-out",
                               str(tmp_path / "b.npy"), "--force", ",".join(map(str, ref)), *extra,
                               tokens=LONG_PROMPT, steps=6)
            lb = np.load(tmp_path / "b.npy")
            assert rec["prefill"] == "batch" and rec["prefill_mode"] == "batched" and lb.shape == lt.shape
            kl = _row_kl(lt, lb)
            top1 = (lt.argmax(-1) == lb.argmax(-1)).mean()
            # tiny random models: bf16 roundings of GEMM vs gemv accumulation orders (measured: KL mean <= 2e-3,
            # max <= 0.09); real-model sizes: scripts/prefill_batch_vs_token.py
            assert kl.mean() < 2e-2 and kl.max() < 0.3, (extra, chunk, kl.mean(), kl.max(), int(kl.argmax()))
            assert top1 >= 0.85, (extra, chunk, top1)
    for off in ("ptra", "ptr"):
        env = {**os.environ, "SUC_FAST_OFF": off}
        r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, LONG_PROMPT)), "--steps", "6",
                            "--prefill", "batch", "--chunk", "16", "--logits-out", str(tmp_path / "x.npy"), "--force",
                            ",".join(map(str, ref)), *extra, "--json", str(tmp_path / "x.json")], capture_output=True,
                           text=True, env=env)
        assert r.returncode == 0, r.stderr
        lx = np.load(tmp_path / "x.npy")
        if off == "ptra":   # only the row-batched kernels: decode-step bits
            assert np.array_equal(lx, lt)
        else:   # + the chunk attention and indexer scores (float32 dot-product orders)
            assert _row_kl(lt, lx).max() < 1e-3


def test_chunk_attention_variants_bit_identical(tmp_path):
    """The chunk MLA attention variants give the same bits: one head per simdgroup (SUC_GLM_ATTN_HPS=1) and two heads
    per simdgroup sharing each latent load (default), with the pass-1 scores recomputed or kept in threadgroup memory
    (SUC_GLM_ATTN_KEEP=1); the GLM-5.3 latent width (512: decode roundings), 8 heads, rows with and without the DSA
    selection (index_topk 8: from 12 tokens on)."""
    path = glm5_file(tmp_path / "h8.suc1", over=dict(REAL, num_attention_heads=8, num_key_value_heads=8))
    out = []
    for env in ({"SUC_GLM_ATTN_HPS": "1"}, {}, {"SUC_GLM_ATTN_KEEP": "1"}):
        r = subprocess.run([str(GENERATE), str(path), "--tokens", ",".join(map(str, LONG_PROMPT)), "--steps", "1",
                            "--prefill", "batch", "--chunk", "16", "--logits-out", str(tmp_path / "x.npy")],
                           capture_output=True, text=True, env={**os.environ, **env})
        assert r.returncode == 0, r.stderr
        out.append(np.load(tmp_path / "x.npy"))
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[0], out[2])


def _moe_layout(kind: str) -> tuple[list[int], int]:
    """(MoE layer indices, routed experts) of a glm_file fixture, from the config it was written with."""
    cfg = tiny_glm5_next_config(**{"tiny": TINY, "real": REAL}[kind])
    cfg = cfg.get("text_config", cfg)
    return [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "sparse"], cfg["n_routed_experts"]


def test_streamed_hot_set(glm_file, tmp_path):
    """--hot-experts pins experts in half of the pool's slots; streamed = resident bit for bit. Hot layers / expert
    ids come from the fixture (tiny: layers 1-3, 8 experts; real widths: layers 0-1, 4 experts); the list also has
    an out-of-range id and a duplicate, which are ignored. slots per layer = min(4, experts - 1) keeps >= top-2
    unpinned slots and fewer slots than experts, so the run really misses and evicts."""
    path = glm_file["path"]
    moe, n_exp = _moe_layout(glm_file["kind"])
    resident = _rows(path, tmp_path, steps=12)
    slot, layers = _slot_bytes(path, tmp_path)
    assert layers == len(moe)
    hot = tmp_path / "hot_experts.json"
    ids = [n_exp + 1, 1, 0, n_exp - 1, 0] + list(range(2, n_exp - 1))
    hot.write_text(json.dumps({"layers": {f"model.layers.{i}.mlp": ids for i in moe}}))
    slots = min(4, n_exp - 1)
    budget = layers * slots * slot
    got = _rows(path, tmp_path, "--expert-budget", str(budget), "--hot-experts", str(hot), steps=12)
    assert got[0] == resident[0] and np.array_equal(got[1], resident[1]) and np.array_equal(got[2], resident[2])
    st = got[3]["expert_streaming"]
    assert st["pinned_experts"] == (slots // 2) * layers
    assert 0 < st["pinned_bytes"] <= 0.5 * budget and st["pool_bytes"] <= budget
    assert st["misses"] > 0 and st["evictions"] > 0


def test_chat_prompt_format(glm_file, tmp_path):
    """--chat on a glm5_next file: the prompt ids equal transformers' apply_chat_template(add_generation_prompt=True)
    with the snapshot's chat_template.jinja ("[gMASK]<sop>", the default "<|system|>Reasoning Effort: Max" turn or
    reasoning_effort low / high, the system turn, "<|assistant|><think>"; enable_thinking is ignored by the template).
    Runs on both fixture widths. Needs the GLM-5.3-Flash tokenizer (tokenizer.json + chat template) in the HF cache:
    the only skip, and its message names the missing asset."""
    snaps = sorted(GLM.glob("*/tokenizer.json")) if GLM.is_dir() else []
    if not snaps:
        pytest.skip(f"GLM-5.3-Flash tokenizer.json not in the HF cache ({GLM}/*/tokenizer.json)")
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoTokenizer.from_pretrained(str(snaps[-1].parent), local_files_only=True)
    cases = [("Ciao, come stai?", None, False, None), ("Ciao, come stai?", None, True, None),
             ("Hello world", "You are helpful.", True, None), ("What is 17 * 23?", "You are helpful.", False, None),
             ("Hello world", None, True, "low"), ("Hello world", "Be brief.", True, "high"),
             ("Hello world", None, True, "max")]
    for prompt, system, think, effort in cases:
        msgs = ([{"role": "system", "content": system}] if system is not None else []) + \
            [{"role": "user", "content": prompt}]
        kw = {"reasoning_effort": effort} if effort else {}
        text = hf.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think, **kw)
        ref = hf.encode(text, add_special_tokens=False)
        args = [str(GENERATE), str(glm_file["path"]), "--prompt", prompt, "--chat", "--tokenizer", str(snaps[-1]),
                "--print-prompt", "--quiet"]
        args += (["--system", system] if system is not None else []) + ([] if think else ["--no-think"])
        args += ["--reasoning-effort", effort] if effort else []
        r = subprocess.run(args, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert [int(x) for x in r.stdout.split("prompt:")[1].split()[0].split(",")] == ref, (prompt, system, effort,
                                                                                              text)


# ------------------------------------------------------------------ a `suc compress` output (allocation, hot set)

from test_glm5_next import compressed  # noqa: E402,F401  (module fixture: tiny glm5_next through `suc compress`)


def test_compress_output_resident_and_streamed(compressed, tmp_path):
    """The container of `suc compress` (greedy allocation with expert-stack codebooks, role budgets, affine embedding /
    head, hot_experts.json next to the file): C++ resident = Python bit for bit; C++ streamed with the pinned hot set
    (`--hot-experts auto`) and a budget below the routed experts = resident."""
    path = Path(compressed[0]["container"]["path"])
    assert (path.parent / "hot_experts.json").exists()
    gen, lc, hc, rec = _rows(path, tmp_path, steps=24)
    seq = PROMPT + gen[:-1]
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    lp, hp = run_tokens(model, seq)
    assert np.array_equal(lp, lc) and np.array_equal(hp, hc)
    assert greedy(model, PROMPT, 24) == gen
    slot, layers = _slot_bytes(path, tmp_path)
    budget = layers * 4 * slot
    got = _rows(path, tmp_path, "--expert-budget", str(budget), "--hot-experts", "auto", "--expert-prefetch", steps=24)
    assert got[0] == gen and np.array_equal(got[1], lc) and np.array_equal(got[2], hc)
    st = got[3]["expert_streaming"]
    assert st["pinned_experts"] > 0 and st["misses"] > 0 and st["pool_bytes"] <= budget
    assert st["pool_bytes"] < st["expert_bytes_total"]
