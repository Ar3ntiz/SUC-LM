"""suc.runtime.expert_cache: routed experts streamed from a SUC1 file through a byte-budgeted cache.

Tiny random Qwen3-MoE SUC1 files (hidden 128, 8 experts, top-2, 2 layers) in two expert layouts — per-expert codecs
(vq / additive + int8 CSR residual / vq + fp16 residual on even experts) and a shared expert stack with int8 column
residuals (the final 30B-A3B codec family) — checked against the fully resident path:
  - records rebuild exactly the Encoded of the file;
  - SucSwitchLinear over the cached experts only (remapped ids) == the resident SucSwitchLinear, bit for bit;
  - a whole decode through the cache (every MoE block's experts fetched per token, small budget, evictions) gives
    logits bit-identical to load_suc1_model;
  - eviction, budgets, prefetch, pinning and the trace simulator agree with each other.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from suc.io.suc1 import SucModel, write_suc1
from suc.kernels.suc_switch import SucSwitchLinear
from suc.quant import stacks
from suc.quant.codecs import AdditiveCodebook, AffineRTN, VectorCodebook
from suc.quant.residual import ResidualCodec
from suc.runtime.expert_cache import (ExpertCache, ExpertLayout, budget_plan, compact, predict_experts, simulate,
                                      switch_from_records, zipf_trace)
from suc.runtime.suc1_model import load_suc1_model, make_cache, step_hidden

ARCH = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 128,
        "intermediate_size": 256, "moe_intermediate_size": 128, "num_experts": 8, "num_experts_per_tok": 2,
        "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": True, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 256, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000, "rope_scaling": None, "max_position_embeddings": 4096, "tie_word_embeddings": True,
        "torch_dtype": "bfloat16", "attention_bias": False, "hidden_act": "silu", "eos_token_id": 2}
PROJS = ("gate_proj", "up_proj", "down_proj")


def _attn():
    return VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2)


PER_EXPERT = {
    "gate_proj": lambda e: (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), {}),
    "up_proj": lambda e: (ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True,
                                                         kmeans_iters=2), 0.01, "int8"),
                          {"seed_in": 11, "seed_out": 12}),
    "down_proj": lambda e: ((ResidualCodec(VectorCodebook(64, block_size=4, rotate=True, rotate_out=False,
                                                          kmeans_iters=2), 0.02, "fp16") if e % 2 == 0 else
                             VectorCodebook(64, block_size=4, rotate=True, rotate_out=False, kmeans_iters=2)), {}),
}
SHARED_COLS = {p: ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False, kmeans_iters=2,
                                               fit_blocks=4096), 0.02, "int8", structure="columns") for p in PROJS}


def _randn(shape, rng, scale):
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32))


def _moe_file(path: Path, kind: str) -> Path:
    arch = ARCH
    rng = np.random.default_rng(7)
    h, d, E, fi = arch["hidden_size"], arch["head_dim"], arch["num_experts"], arch["moe_intermediate_size"]
    q, kv = arch["num_attention_heads"] * d, arch["num_key_value_heads"] * d
    t: dict = {}
    seed = 1
    for i in range(arch["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for role, shape in (("self_attn.q_proj", (q, h)), ("self_attn.k_proj", (kv, h)), ("self_attn.v_proj", (kv, h)),
                            ("self_attn.o_proj", (h, q))):
            t[p + role + ".weight"] = _attn().encode(_randn(shape, rng, 0.08 * (128 / shape[1]) ** 0.5), seed=seed)
            seed += 1
        for n, size in (("input_layernorm", h), ("post_attention_layernorm", h), ("self_attn.q_norm", d),
                        ("self_attn.k_norm", d)):
            t[p + n + ".weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(size).astype(np.float32)).astype(mx.bfloat16)
        t[p + "mlp.gate.weight"] = _randn((E, h), rng, 0.5).astype(mx.bfloat16)
        for proj, shape in (("gate_proj", (fi, h)), ("up_proj", (fi, h)), ("down_proj", (h, fi))):
            names = [f"{p}mlp.experts.{e}.{proj}.weight" for e in range(E)]
            ws = {n: _randn(shape, rng, 0.08 * (128 / shape[1]) ** 0.5) for n in names}
            if kind == "shared_cols":
                for n, _, enc in stacks.encode_stack(SHARED_COLS[proj], ws.__getitem__, names, seed=seed,
                                                     stack=f"{p}mlp.switch_mlp.{proj}.weight"):
                    t[n] = enc
                seed += 1
            else:
                for e, n in enumerate(names):
                    codec, kw = PER_EXPERT[proj](e)
                    t[n] = codec.encode(ws[n], seed=seed, **kw)
                    seed += 1
    t["model.norm.weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(h).astype(np.float32)).astype(mx.bfloat16)
    t["model.embed_tokens.weight"] = AffineRTN(4, 64).encode(_randn((arch["vocab_size"], h), rng, 1.0), seed=seed)
    write_suc1(path, architecture=arch, tokenizer_id="none", tensors=t)
    return path


@pytest.fixture(scope="module", params=["per_expert", "shared_cols"])
def moe(request, tmp_path_factory):
    path = _moe_file(tmp_path_factory.mktemp("xc") / f"{request.param}.suc1", request.param)
    return {"path": path, "kind": request.param, "sm": SucModel.load(path)}


def _expert_bytes(sm: SucModel, layer: int, e: int) -> int:
    return sum(sm.descriptor(f"model.layers.{layer}.mlp.experts.{e}.{p}.weight")["compressed_size"] for p in PROJS)


# ------------------------------------------------------------------------------------------------- layout / records

def test_layout_covers_every_expert_payload(moe):
    sm = moe["sm"]
    lay = ExpertLayout(sm)
    assert lay.layers == ["model.layers.0.mlp", "model.layers.1.mlp"]
    assert lay.num_experts == [8, 8]
    for li in range(2):
        for e in range(8):
            en = lay.entry(li, e)
            assert en.nbytes == _expert_bytes(sm, li, e)
            assert len(en.ranges) <= 4                      # one range per section touched (indices/scales/residual)
            assert en.read_bytes >= en.nbytes                # alignment padding between arrays only
            for _, r in en.arrays:
                assert r["nbytes"] == 0 or any(o <= r["offset"] and r["offset"] + r["nbytes"] <= o + n for o, n in en.ranges)
    s = lay.summary()
    assert s["experts"] == 16 and s["expert_bytes_total"] == sum(_expert_bytes(sm, l, e) for l in range(2)
                                                                  for e in range(8))
    assert lay.entry("model.layers.1.mlp", 3) is lay.entry(1, 3)


@pytest.mark.parametrize("io", ["pread", "mmap"])
def test_records_rebuild_the_file_encodings(moe, io):
    sm = moe["sm"]
    with ExpertCache(sm, 1 << 30, io=io) as c:
        recs = c.get(1, [5, 2])
        assert [r.expert for r in recs] == [5, 2]
        for r in recs:
            assert r.nbytes == _expert_bytes(sm, 1, r.expert) == sum(a.nbytes for a in r.arrays.values())
            for p in PROJS:
                a = r.encoded(p)
                b = sm.encoded(f"model.layers.1.mlp.experts.{r.expert}.{p}.weight")
                assert (a.codec, a.shape, a.params) == (b.codec, b.shape, b.params)
                assert sorted(a.arrays) == sorted(b.arrays)
                for k in a.arrays:
                    assert a.arrays[k].dtype == b.arrays[k].dtype
                    assert np.array_equal(np.array(a.arrays[k].astype(mx.float32)),
                                          np.array(b.arrays[k].astype(mx.float32))), k


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_compact_switch_matches_resident_switch(moe, dtype):
    sm = moe["sm"]
    rng = np.random.default_rng(3)
    with ExpertCache(sm, 1 << 30) as c:
        for layer in range(2):
            for proj in PROJS:
                full = SucSwitchLinear.from_encodings(
                    [sm.encoded(f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight") for e in range(8)])
                for n in (1, 2, 5, 12):
                    x = mx.array(rng.standard_normal((n, full.in_features)).astype(np.float32)).astype(dtype)
                    eidx = rng.integers(0, 8, n)
                    if n >= 3:
                        eidx[1] = eidx[0]
                    ref = full.rows(x, mx.array(eidx.astype(np.uint32)), path="gather")
                    ids, remap = compact(eidx)
                    assert [ids[i] for i in remap] == list(eidx)
                    sw = switch_from_records(c.get(layer, ids), proj)
                    y = sw.rows(x, mx.array(remap), path="gather")
                    assert np.array_equal(np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32)))


class _CachedSwitchGLU(nn.Module):
    """mlx-lm SwitchGLU (decode path, indices.size < 64) with the experts fetched from an ExpertCache per call."""

    def __init__(self, cache: ExpertCache, layer: int, activation):
        super().__init__()
        self._c, self._layer, self._act = cache, layer, activation

    def __call__(self, x, indices):
        ids, remap = compact(np.array(indices))
        recs = self._c.get(self._layer, ids)
        g, u, d = (switch_from_records(recs, p) for p in PROJS)
        idx = mx.array(remap)
        x = mx.expand_dims(x, (-2, -3))
        y = d(self._act(u(x, idx), g(x, idx)), idx)
        return y.squeeze(-2)


@pytest.mark.parametrize("policy", ["lru", "slru"])
def test_decode_through_cache_is_bit_identical(moe, policy):
    tokens = [3, 17, 99, 5, 42, 7, 200, 13, 64, 1]
    ref_model, _, _ = load_suc1_model(moe["path"], group=False)
    rc = make_cache(ref_model)
    ref = [np.array(step_hidden(ref_model, t, rc)[0].astype(mx.float32)) for t in tokens]

    model, sm, _ = load_suc1_model(moe["path"], group=False)
    lay = ExpertLayout(sm)
    budget = 3 * max(lay.entry(l, e).nbytes for l in range(2) for e in range(8))   # < one layer's 8 experts
    with ExpertCache(sm, budget, policy=policy) as c:
        for li, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            mlp.switch_mlp = _CachedSwitchGLU(c, li, mlp.switch_mlp.activation)
        kc = make_cache(model)
        for t, r in zip(tokens, ref):
            logits = np.array(step_hidden(model, t, kc)[0].astype(mx.float32))
            assert np.array_equal(logits, r)
            assert c.stats.resident_bytes <= budget
        st = c.stats
        assert st.requests == len(tokens) * 2 * 2
        assert st.hits + st.misses == st.requests and st.misses > 0 and st.evictions > 0


# ------------------------------------------------------------------------------------------------- cache mechanics

def test_lru_eviction_budget_and_stats(moe):
    sm = moe["sm"]
    lay = ExpertLayout(sm)
    sizes = [lay.entry(0, e).nbytes for e in range(8)]
    budget = sizes[0] + sizes[1] + sizes[2]
    with ExpertCache(sm, budget, policy="lru") as c:
        c.get(0, [0, 1])
        assert (c.stats.misses, c.stats.hits) == (2, 0)
        c.get(0, [1, 0, 1])                                  # repeats: 3 hits
        assert (c.stats.misses, c.stats.hits) == (2, 3)
        c.get(0, [2])
        assert c.resident(0) == [0, 1, 2]
        if sum(sizes[1:4]) <= budget:                         # expert 0 is least recent -> victim
            c.get(0, [3])
            assert 0 not in c.resident(0) and c.stats.evictions >= 1
        assert c.stats.resident_bytes <= budget
        assert c.stats.bytes_read_sync == c.stats.bytes_read >= sum(sizes[:3])
        with pytest.raises(MemoryError):
            c.get(0, list(range(8)))


def test_prefetch_hits_and_waste(moe):
    sm = moe["sm"]
    with ExpertCache(sm, 1 << 30) as c:
        assert c.prefetch(1, [4, 6, 6]) == 2
        recs = c.get(1, [4, 0])
        assert [r.expert for r in recs] == [4, 0]
        st = c.stats
        assert (st.prefetch_hits, st.misses, st.prefetch_wasted) == (1, 1, 1)   # 6 predicted, not routed
        assert 6 not in c.resident(1)
        assert c.prefetch(1, [4]) == 0                                          # already resident


def test_pinned_experts_survive_eviction(moe):
    sm = moe["sm"]
    lay = ExpertLayout(sm)
    big = max(lay.entry(0, e).nbytes for e in range(8))
    with ExpertCache(sm, 4 * big) as c:
        c.pin(0, [7])
        for e in range(6):
            c.get(0, [e])
        assert 7 in c.resident(0)
        c.reset_stats()
        c.get(0, [7])
        assert c.stats.hits == 1 and c.stats.bytes_read == 0
        assert c.stats.resident_bytes <= 4 * big


@pytest.mark.parametrize("policy", ["lru", "slru"])
def test_simulator_matches_cache(moe, policy):
    sm = moe["sm"]
    lay = ExpertLayout(sm)
    sizes = np.array([[lay.entry(l, e).nbytes for e in range(8)] for l in range(2)])
    trace = zipf_trace(40, 2, 8, 2, 1.0, seed=1)
    budget = int(sizes.max() * 7)
    with ExpertCache(sm, budget, policy=policy) as c:
        for tok in trace:
            for l in range(2):
                c.get(l, tok[l])
        sim = simulate(trace, sizes, budget, policy=policy)
        assert sim["hit_rate"] == pytest.approx(c.stats.hit_rate)
        miss = sim["miss_bytes_per_token"] * 40          # payload bytes; the reads add alignment padding only
        assert miss <= c.stats.bytes_read_sync <= 1.05 * miss + 64 * 4 * c.stats.misses


def test_predict_experts_topk():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    h = rng.standard_normal((3, 32)).astype(np.float32)
    got = predict_experts(w, h, 4)
    ref = np.argsort(-(h @ w.T), axis=1)[:, :4]
    assert np.array_equal(got, ref)


def test_budget_plan_qwen38_numbers():
    per = (1280 * 2560 + 2560 * 640) * 4.2 / 8          # one expert at 4.2 bpp
    kw = dict(resident_bytes=4e9, overhead_bytes=6e9, n_layers=48, n_experts=512, top_k=10, expert_bytes=per,
              ssd_bytes_per_s=5e9, compute_s=0.05)
    p111 = budget_plan(ram_bytes=111e9, **kw)
    assert p111["all_experts_resident"] and p111["io_ms_per_token"] == 0 and p111["tok_s_serial"] == pytest.approx(20)
    p32 = budget_plan(ram_bytes=32e9, **kw)
    assert p32["feasible"] and 0.3 < p32["capacity_fraction"] < 0.4
    assert p32["miss_mb_per_token"] == pytest.approx(48 * 10 * per * (1 - p32["hit_rate"]) / 1e6)
    assert not budget_plan(ram_bytes=10e9, **kw)["feasible"]
    sim = simulate(zipf_trace(400, 2, 64, 4, 0.0, seed=2), 1.0, 64)   # uniform: LRU hit ≈ capacity fraction 0.5
    assert abs(sim["hit_rate"] - 0.5) < 0.06
