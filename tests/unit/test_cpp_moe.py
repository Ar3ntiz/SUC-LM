"""Cross-check of the C++/Metal MoE code (cpp/src/metal/switch.cpp: SwitchLinear, MoeBlock; tool suc1-moe) with the
Python/MLX runtime of the same SUC1 file (suc.runtime.suc1_model.load_suc1_model, group=False).

Tiny random Qwen3-MoE files (hidden 128, head_dim 64, 8 / 16 experts, top-k 2 / 4 / 8, moe_intermediate 128 / 256),
experts encoded per tensor (vq / additive, rotations with per-expert or shared seeds, int8 / fp16 CSR residuals on some
experts) and one variant with shared expert-stack codebooks (suc.quant.stacks.encode_stack) and pairwise identical
router rows (tied gates), and two variants with column-structured residuals (per expert: int8 / fp16 columns with
and without CSR tails, ragged column counts, experts without residual; shared stack: the final 30B-A3B codec family):
  (a) one expert stack: C++ SwitchLinear == SucSwitchLinear.rows(x, eidx, "gather") bit for bit (f16 / bf16, several
      N, random expert ids with repeats), for every stack layout (auto, stacked, table);
  (b) one sparse block: C++ MoeBlock == model.model.layers[L].mlp(x) bit for bit (routing ids, scores and outputs).

Skipped unless suc1-moe is built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.io.suc1 import write_suc1
from suc.quant import stacks
from suc.quant.codecs import AdditiveCodebook, AffineRTN, VectorCodebook
from suc.quant.residual import ResidualCodec
from suc.runtime.suc1_model import load_suc1_model

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
MOE = BUILD / "suc1-moe"

pytestmark = pytest.mark.skipif(not MOE.exists(), reason=f"suc1-moe not built in {BUILD} (run cpp/build.sh)")

ARCH = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 128,
        "intermediate_size": 256, "moe_intermediate_size": 128, "num_experts": 8, "num_experts_per_tok": 2,
        "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": True, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 512, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000, "rope_scaling": None, "max_position_embeddings": 4096, "tie_word_embeddings": True,
        "torch_dtype": "bfloat16", "attention_bias": False, "hidden_act": "silu", "eos_token_id": 2}

ATTN = {
    "self_attn.q_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "self_attn.k_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "self_attn.v_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "self_attn.o_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "mlp.gate_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "mlp.up_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "mlp.down_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
}

# Per-tensor expert encodings. gate: vq, own seeds per expert (per-row sign tables, per-expert codebooks). up: additive
# M = 2 with an int8 CSR residual, shared seeds (plain rotation launches, per-expert codebooks). down: vq K = 64 G = 4,
# input rotation only, fp16 CSR residual on the even experts only.
PER_EXPERT = {
    "gate_proj": lambda e: (VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), {}),
    "up_proj": lambda e: (ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True,
                                                         kmeans_iters=2), 0.01, "int8"),
                          {"seed_in": 11, "seed_out": 12}),
    "down_proj": lambda e: ((ResidualCodec(VectorCodebook(64, block_size=4, rotate=True, rotate_out=False,
                                                          kmeans_iters=2), 0.02, "fp16") if e % 2 == 0 else
                             VectorCodebook(64, block_size=4, rotate=True, rotate_out=False, kmeans_iters=2)), {}),
}
# Shared expert-stack codebooks (suc.quant.stacks.encode_stack: one codebook set and one seed pair per stack).
SHARED = {
    "gate_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2, fit_blocks=4096),
    "up_proj": ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=2,
                                              fit_blocks=4096), 0.01, "int8"),
    "down_proj": VectorCodebook(64, block_size=8, rotate=True, kmeans_iters=2, fit_blocks=4096),
}

# Column-structured residuals (ResidualCodec structure="columns"), per expert: gate int8 columns on every expert; up
# fp16 columns + fp16 CSR tail (column_share 0.5) with fractions that differ per expert (different column counts);
# down int8 columns + int8 CSR tail on the even experts only (the odd ones have no residual).
PER_EXPERT_COLS = {
    "gate_proj": lambda e: (ResidualCodec(VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), 0.02, "int8",
                                          structure="columns"), {}),
    "up_proj": lambda e: (ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True,
                                                         kmeans_iters=2), 0.03 if e % 3 else 0.01, "fp16",
                                        structure="columns", column_share=0.5), {"seed_in": 11, "seed_out": 12}),
    "down_proj": lambda e: ((ResidualCodec(VectorCodebook(64, block_size=4, rotate=True, rotate_out=False,
                                                          kmeans_iters=2), 0.02, "int8", structure="columns",
                                           column_share=0.5) if e % 2 == 0 else
                             VectorCodebook(64, block_size=4, rotate=True, rotate_out=False, kmeans_iters=2)), {}),
}
# The codec family of the final Qwen3-30B-A3B files on a shared expert stack: VQ K = 256 G = 2, one-sided input
# rotation, int8 column residual (no tail).
SHARED_COLS = {proj: ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False, kmeans_iters=2,
                                                  fit_blocks=4096), 0.02, "int8", structure="columns")
               for proj in ("gate_proj", "up_proj", "down_proj")}

VARIANTS = {
    "e8k2": ({}, "per_expert"),
    "e8k4_mlp_only": ({"num_experts_per_tok": 4, "mlp_only_layers": [1], "moe_intermediate_size": 256}, "per_expert"),
    "e16k8_nonorm": ({"num_experts": 16, "num_experts_per_tok": 8, "norm_topk_prob": False}, "per_expert"),
    "e8k2_shared": ({}, "shared"),
    "e8k2_cols": ({}, "per_expert_cols"),
    "e8k4_shared_cols": ({"num_experts_per_tok": 4}, "shared_cols"),
}


def _randn(shape, rng, scale):
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32))


def _moe_file(path: Path, variant: str) -> tuple[Path, dict]:
    over, kind = VARIANTS[variant]
    arch = dict(ARCH, **over)
    rng = np.random.default_rng(7)
    h, d, vocab = arch["hidden_size"], arch["head_dim"], arch["vocab_size"]
    E, fi, f = arch["num_experts"], arch["moe_intermediate_size"], arch["intermediate_size"]
    q, kv = arch["num_attention_heads"] * d, arch["num_key_value_heads"] * d
    t: dict = {}
    seed = 1
    for i in range(arch["num_hidden_layers"]):
        p = f"model.layers.{i}."
        shapes = {"self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h), "self_attn.v_proj": (kv, h),
                  "self_attn.o_proj": (h, q)}
        if i in arch["mlp_only_layers"]:
            shapes.update({"mlp.gate_proj": (f, h), "mlp.up_proj": (f, h), "mlp.down_proj": (h, f)})
        for role, shape in shapes.items():
            t[p + role + ".weight"] = ATTN[role].encode(_randn(shape, rng, 0.08 * (128 / shape[1]) ** 0.5), seed=seed)
            seed += 1
        for n, size in (("input_layernorm", h), ("post_attention_layernorm", h), ("self_attn.q_norm", d),
                        ("self_attn.k_norm", d)):
            t[p + n + ".weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(size).astype(np.float32)).astype(mx.bfloat16)
        if i in arch["mlp_only_layers"]:
            continue
        router = _randn((E, h), rng, 0.5)
        if kind == "shared":   # pairwise identical router rows: tied gates, so top-k must follow MLX's tie order
            router = mx.repeat(router[::2], 2, axis=0)
        t[p + "mlp.gate.weight"] = router.astype(mx.bfloat16)
        for proj, shape in (("gate_proj", (fi, h)), ("up_proj", (fi, h)), ("down_proj", (h, fi))):
            names = [f"{p}mlp.experts.{e}.{proj}.weight" for e in range(E)]
            ws = {n: _randn(shape, rng, 0.08 * (128 / shape[1]) ** 0.5) for n in names}
            if kind in ("shared", "shared_cols"):
                stack = f"{p}mlp.switch_mlp.{proj}.weight"
                codec = (SHARED if kind == "shared" else SHARED_COLS)[proj]
                for n, _, enc in stacks.encode_stack(codec, ws.__getitem__, names, seed=seed, stack=stack):
                    t[n] = enc
                seed += 1
            else:
                for e, n in enumerate(names):
                    codec, kw = (PER_EXPERT if kind == "per_expert" else PER_EXPERT_COLS)[proj](e)
                    t[n] = codec.encode(ws[n], seed=seed, **kw)
                    seed += 1
    t["model.norm.weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(h).astype(np.float32)).astype(mx.bfloat16)
    t["model.embed_tokens.weight"] = AffineRTN(4, 64).encode(_randn((vocab, h), rng, 1.0), seed=seed)
    write_suc1(path, architecture=arch, tokenizer_id="none", tensors=t)
    return path, arch


@pytest.fixture(scope="module", params=list(VARIANTS))
def moe(request, tmp_path_factory):
    path, arch = _moe_file(tmp_path_factory.mktemp("moe") / f"{request.param}.suc1", request.param)
    model, _, _ = load_suc1_model(path, group=False)
    if VARIANTS[request.param][1].endswith("_cols"):   # the column residual path is really exercised
        for layer in _sparse_layers(arch):
            sw = model.model.layers[layer].mlp.switch_mlp
            assert all(getattr(sw, p).n_columns > 0 for p in ("gate_proj", "up_proj", "down_proj"))
    return {"path": path, "arch": arch, "model": model, "variant": request.param}


def _run(*args: str) -> str:
    r = subprocess.run([str(MOE), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stderr


def _sparse_layers(arch: dict) -> list[int]:
    return [i for i in range(arch["num_hidden_layers"]) if i not in arch["mlp_only_layers"]]


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_switch_linear_matches_python(moe, tmp_path, dtype) -> None:
    mdt = {"bf16": mx.bfloat16, "f16": mx.float16}[dtype]
    rng = np.random.default_rng(3)
    arch, E = moe["arch"], moe["arch"]["num_experts"]
    worst = 0
    for layer in _sparse_layers(arch):
        sw = moe["model"].model.layers[layer].mlp.switch_mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(sw, proj)
            for n in (1, 3, 8, 40):
                x = mx.array(rng.standard_normal((n, mod.in_features)).astype(np.float32)).astype(mdt)
                eidx = rng.integers(0, E, n).astype(np.uint32)
                if n >= 3:
                    eidx[1] = eidx[0]   # repeated experts
                ref = np.array(mod.rows(x, mx.array(eidx), path="gather").astype(mx.float32))
                np.save(tmp_path / "x.npy", np.array(x.astype(mx.float32)))
                np.save(tmp_path / "e.npy", eidx.astype(np.float32))
                for layout in ("auto", "stacked", "table"):
                    _run(str(moe["path"]), "--layer", str(layer), "--stack", proj, "--x", str(tmp_path / "x.npy"),
                         "--eidx", str(tmp_path / "e.npy"), "--y-out", str(tmp_path / "y.npy"), "--dtype", dtype,
                         "--layout", layout, "--no-verify")
                    y = np.load(tmp_path / "y.npy")
                    assert y.shape == ref.shape
                    bad = int((y != ref).sum())
                    worst = max(worst, float(np.abs(y - ref).max()))
                    assert bad == 0, (f"{moe['variant']} layer {layer} {proj} n={n} {layout}: {bad} of {y.size} "
                                      f"differ, max |d| {np.abs(y - ref).max()}")
    assert worst == 0


def _python_routing(mlp, x: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """inds / scores exactly as Qwen3MoeSparseMoeBlock.__call__ computes them."""
    gates = mx.softmax(mlp.gate(x), axis=-1, precise=True)
    k = mlp.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if mlp.norm_topk_prob:
        scores /= mx.sum(scores, axis=-1, keepdims=True)
    return np.array(inds)[0], np.array(scores.astype(mx.float32))[0]


def test_moe_block_matches_python(moe, tmp_path) -> None:
    rng = np.random.default_rng(5)
    arch = moe["arch"]
    k, h = arch["num_experts_per_tok"], arch["hidden_size"]
    for layer in _sparse_layers(arch):
        mlp = moe["model"].model.layers[layer].mlp
        for n in sorted({1, 3, 32 // k}):   # n · k <= 32 rows: SucSwitchLinear's "gather" path
            x = mx.array(rng.standard_normal((1, n, h)).astype(np.float32)).astype(mx.bfloat16)
            ref = np.array(mlp(x).astype(mx.float32))[0]
            inds, scores = _python_routing(mlp, x)
            np.save(tmp_path / "x.npy", np.array(x.astype(mx.float32))[0])
            _run(str(moe["path"]), "--layer", str(layer), "--x", str(tmp_path / "x.npy"), "--y-out",
                 str(tmp_path / "y.npy"), "--route-out", str(tmp_path / "r.npy"), "--no-verify")
            r = np.load(tmp_path / "r.npy")
            assert np.array_equal(r[:, :k].astype(np.int64), inds.astype(np.int64)), (r[:, :k], inds)
            assert np.array_equal(r[:, k:], scores)
            y = np.load(tmp_path / "y.npy")
            assert y.shape == ref.shape
            assert np.array_equal(y, ref), f"{moe['variant']} layer {layer} n={n}: max |d| {np.abs(y - ref).max()}"
            # experts + combine with Python's routing (the path model.cpp can use with a given routing)
            np.save(tmp_path / "rr.npy", np.concatenate([inds.astype(np.float32), scores], axis=1))
            _run(str(moe["path"]), "--layer", str(layer), "--x", str(tmp_path / "x.npy"), "--y-out",
                 str(tmp_path / "y2.npy"), "--routing", str(tmp_path / "rr.npy"), "--layout", "table", "--no-verify")
            assert np.array_equal(np.load(tmp_path / "y2.npy"), ref)


# Qwen3-30B-A3B MoE widths (hidden 2048, 128 experts, top-8, moe_intermediate 768 = 12 · 64: H_12 rotations; the router
# gemv takes MLX's 8-simdgroup split of K since 2048 >= 16 · 128), one sparse layer, shared expert-stack codebooks.
# Only the tensors of that block are written (MoeBlock needs nothing else; the Python model keeps random attention).
WIDE = dict(ARCH, hidden_size=2048, moe_intermediate_size=768, intermediate_size=6144, num_experts=128,
            num_experts_per_tok=8, num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4, head_dim=128)
WIDE_CODECS = {
    "gate_proj": VectorCodebook(256, block_size=8, rotate=True, kmeans_iters=1, fit_blocks=8192),
    "up_proj": ResidualCodec(VectorCodebook(256, block_size=8, rotate=True, kmeans_iters=1, fit_blocks=8192), 0.005,
                             "int8"),
    "down_proj": AdditiveCodebook(256, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=1, fit_blocks=8192),
}
# The final Qwen3-30B-A3B codec on every stack: VQ K = 256 G = 2 one-sided + 0.5% int8 column residual (36 columns per
# gate / up expert, 13 per down expert at these widths).
WIDE_FINAL = {proj: ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False, kmeans_iters=1,
                                                 fit_blocks=8192), 0.005, "int8", structure="columns")
              for proj in ("gate_proj", "up_proj", "down_proj")}


@pytest.mark.parametrize("codecs", ["mixed", "final_cols"])
def test_wide_moe_block_matches_python(tmp_path, codecs) -> None:
    codec_of = {"mixed": WIDE_CODECS, "final_cols": WIDE_FINAL}[codecs]
    arch, rng = WIDE, np.random.default_rng(11)
    h, fi, E = arch["hidden_size"], arch["moe_intermediate_size"], arch["num_experts"]
    t, p = {}, "model.layers.0."
    t[p + "mlp.gate.weight"] = _randn((E, h), rng, 0.05).astype(mx.bfloat16)
    for proj, shape in (("gate_proj", (fi, h)), ("up_proj", (fi, h)), ("down_proj", (h, fi))):
        names = [f"{p}mlp.experts.{e}.{proj}.weight" for e in range(E)]
        scale = 0.08 * (128 / shape[1]) ** 0.5

        def load(n, shape=shape, scale=scale):
            return _randn(shape, np.random.default_rng(abs(hash(n)) % (1 << 31)), scale)
        for n, _, enc in stacks.encode_stack(codec_of[proj], load, names, seed=3, stack=f"{p}mlp.switch_mlp.{proj}.weight"):
            t[n] = enc
    path = tmp_path / "wide.suc1"
    write_suc1(path, architecture=arch, tokenizer_id="none", tensors=t)
    mlp = load_suc1_model(path, group=False)[0].model.layers[0].mlp
    if codecs == "final_cols":
        assert [getattr(mlp.switch_mlp, p).n_columns for p in ("gate_proj", "up_proj", "down_proj")] == \
            [36 * E, 36 * E, 13 * E]
    k = arch["num_experts_per_tok"]
    for n in (1, 4):
        x = mx.array(rng.standard_normal((1, n, h)).astype(np.float32)).astype(mx.bfloat16)
        ref = np.array(mlp(x).astype(mx.float32))[0]
        inds, scores = _python_routing(mlp, x)
        np.save(tmp_path / "x.npy", np.array(x.astype(mx.float32))[0])
        _run(str(path), "--layer", "0", "--x", str(tmp_path / "x.npy"), "--y-out", str(tmp_path / "y.npy"),
             "--route-out", str(tmp_path / "r.npy"), "--no-verify")
        r = np.load(tmp_path / "r.npy")
        assert np.array_equal(r[:, :k].astype(np.int64), inds.astype(np.int64))
        assert np.array_equal(r[:, k:], scores)
        y = np.load(tmp_path / "y.npy")
        assert np.array_equal(y, ref), f"n={n}: max |d| {np.abs(y - ref).max()}"
