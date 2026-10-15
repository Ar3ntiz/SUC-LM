"""Trellis-coded (tcq) tensors in the C++/Metal runtime (cpp/: decode.cpp, metal::Linear, SwitchLinear, mixed expert
stacks, the model graphs) against the Python runtime of the same SUC1 files, bit for bit.

1. Reference decoder: suc1-decode == SucModel.decode_tensor (float32 bits) for 1mad / 3inst / pair3, k = 2 / 3,
   V = 1 / 2, L = 10 / 12, T = 128 / 256, rotations, column / CSR residuals.
2. One linear (suc1-linear): GPU vs the CPU reference decoder (tolerance), and bit-exact vs
   suc.runtime.compressed_model.module_from_encoded (SucLinear / SucResidualLinear, tcq mode) for f16 / bf16, batches
   on the "vec" (< 5 rows) and "mma" (5..16 rows) kernels and several batch tiles, forced vec / mma paths; the
   row-exact batch (encode_rows_exact: vec tiles of <= 4 rows) equals the one-row launches row for row.
3. Expert stacks (suc1-moe --stack): SwitchLinear == SucSwitchLinear.rows(x, eidx, "gather") for every stack layout
   (stacked / table), shared and per-expert input rotations, residuals; expert tiles of 2 / 4 rows (the batched
   prefill path) give the same bits; mixed vq + tcq stacks (ExpertStack, routed / all rows, tiled) == SucSwitchMixed;
   one Qwen3-MoE sparse block (router + tcq experts + combine) == the Python block.
4. Model graphs (suc1-generate, teacher-forced decode): qwen3 (separate and fused projection plans), qwen3_moe,
   deepseek_v4 (tcq attention projections and tcq experts inside mixed stacks, with native MXFP4 experts too; batched
   prefill and SSD expert streaming bit-identical), glm5_next, qwen4_exp and qwen3_5 (compressed files whose
   projections with in % 128 == 0 are re-encoded as tcq): logits and streams bit-identical to the Python runtime.

Skipped unless the C++ tools are built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.io.suc1 import SucModel, stored, write_suc1
from suc.quant.codecs import VectorCodebook
from suc.quant.residual import ResidualCodec
from suc.quant.trellis import TrellisCodec

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
DECODE, LINEAR, MOE, GENERATE = (BUILD / n for n in ("suc1-decode", "suc1-linear", "suc1-moe", "suc1-generate"))

pytestmark = pytest.mark.skipif(not (LINEAR.exists() and MOE.exists() and GENERATE.exists()),
                                reason=f"C++ Metal tools not built in {BUILD} (run cpp/build.sh)")

REL_L2 = {"f32": 2e-5, "f16": 2e-3, "bf16": 1.5e-2}


def tcq(code="3inst", k=2, V=2, L=10, T=128):
    return TrellisCodec(k=k, L=L, code=code, V=V, T=T, rotate=True, rotate_out=False)


def cols(base, fraction=0.02, dtype="int8", share=1.0):
    return ResidualCodec(base, fraction, dtype, "magnitude", structure="columns", column_share=share)


class SharedIn:
    """Codec wrapper: every encode without an explicit seed_in uses one input rotation seed (the stack-shared rotation
    `suc compress` gives tcq experts)."""

    def __init__(self, codec, seed_in: int):
        self.codec, self.seed_in = codec, seed_in

    def encode(self, w, *, seed, **kw):
        kw.setdefault("seed_in", self.seed_in)
        return self.codec.encode(w, seed=seed, **kw)


class TcqOr:
    """tcq where the input width allows it (in % T == 0), else another codec."""

    def __init__(self, t, other, T: int = 128):
        self.t, self.other, self.T = t, other, T

    def encode(self, w, *, seed, **kw):
        return (self.t if w.shape[1] % self.T == 0 else self.other).encode(w, seed=seed, **kw)


def _w(shape, seed, scale=None) -> mx.array:
    scale = scale if scale is not None else 1.0 / np.sqrt(shape[1])
    return mx.array((scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _run(tool: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run([str(tool), *args], capture_output=True, text=True, env={**os.environ, **(env or {})})
    if r.returncode != 0 and "no Metal device" in r.stderr:
        pytest.skip("no Metal device")
    assert r.returncode == 0, r.stderr
    return r


# ------------------------------------------------------------------------------------------------ dense tensors

DENSE = {
    "tcq_3inst_k2": (tcq(), (72, 256)),                                   # 72 rows: partial threadgroups
    "tcq_1mad_k3_v1": (tcq("1mad", k=3, V=1), (64, 256)),
    "tcq_1mad_k2_v1_L12": (tcq("1mad", k=2, V=1, L=12), (40, 512)),
    "tcq_pair3": (tcq("pair3"), (48, 512)),
    "tcq_3inst_k3_T256": (tcq(k=3, L=12, T=256), (40, 512)),
    "tcq_rot_m12": (tcq(), (32, 384)),                                    # 384 = 12 * 32: H_12 input rotation
    "tcq_norot": (TrellisCodec(k=2, L=10, code="3inst", V=2, T=128, rotate=False), (64, 256)),
    "tcq_cols_int8": (cols(tcq()), (64, 256)),
    "tcq_cols_fp16_tail": (cols(tcq("1mad", k=3, V=1), 0.03, "fp16", 0.5), (64, 256)),
    "tcq_csr_fp16": (ResidualCodec(tcq("pair3"), 0.02, "fp16"), (64, 256)),
}
PLAIN = [k for k, (c, _) in DENSE.items() if isinstance(c, TrellisCodec)]


@pytest.fixture(scope="module")
def dense(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("tcq_dense") / "dense.suc1"
    tensors = {f"t.{k}.weight": c.encode(_w(shape, i + 1), seed=i + 3) for i, (k, (c, shape)) in enumerate(DENSE.items())}
    write_suc1(p, architecture={"model_type": "qwen3"}, tokenizer_id="tok", tensors=tensors)
    sm = SucModel.load(p)
    assert all(sm.descriptor(n)["representation_type"] == "tcq" for n in sm.names())
    return p


def test_decode_bit_exact(dense: Path, tmp_path: Path) -> None:
    out = tmp_path / "cpp"
    _run(DECODE, str(dense), "--all", "--out", str(out))
    sm = SucModel.load(dense)
    bad = []
    for n in sm.names():
        ref, got = np.array(sm.decode_tensor(n)), np.load(out / f"{n}.npy")
        assert got.shape == ref.shape, n
        if not np.array_equal(_bits(got), _bits(ref)):
            bad.append((n, float(np.abs(got.astype(np.float64) - ref).max())))
    assert not bad, f"C++ tcq decode differs from SucModel.decode_tensor: {bad}"


def _rows(r: subprocess.CompletedProcess) -> list[dict]:
    import json
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_gpu_matches_cpu_reference(dense: Path, dtype: str) -> None:
    rows = _rows(_run(LINEAR, str(dense), "--all", "--json", "--dtype", dtype, "--batches", "1,3,4,5,16,18,21"))
    assert {r["tensor"] for r in rows} == {f"t.{k}.weight" for k in DENSE}
    assert all(r["codec"].startswith("tcq") for r in rows)
    bad = [(r["tensor"], r["batch"], r["rel_l2"]) for r in rows
           if not r["finite"] or r["rel_l2"] > REL_L2[dtype] or r["max_rel"] > 10 * REL_L2[dtype]]
    assert not bad, bad
    assert {r["path"] for r in rows} >= {"vec", "mma", "mma+vec", "mma+mma"}
    # the streams and row scales are read in place: zero-copy bytes >= streams (k bits per weight) + f16 row scales
    k = {f"t.{n}.weight": c.base.k if hasattr(c, "base") else c.k for n, (c, _) in DENSE.items()}
    assert all(r["zero_copy_bytes"] >= r["out"] * (r["in"] * k[r["tensor"]] / 8 + 2) for r in rows)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_bit_exact_vs_python(dense: Path, tmp_path: Path, dtype: str) -> None:
    from suc.runtime.compressed_model import module_from_encoded

    mdt = {"f16": mx.float16, "bf16": mx.bfloat16}[dtype]
    sm = SucModel.load(dense)
    xp, yp = tmp_path / "x.npy", tmp_path / "y.npy"
    bad = []
    for k in DENSE:
        name = f"t.{k}.weight"
        mod = module_from_encoded(sm.encoded_packed(name))
        assert mod.tcq is not None
        runs = [(b, "auto") for b in (1, 2, 3, 4, 5, 8, 16, 18, 21)]
        if k in PLAIN:
            runs += [(3, "mma"), (9, "vec"), (1, "mma"), (16, "vec")]
        for b, path in runs:
            x = np.random.default_rng(b).standard_normal((b, mod.in_features)).astype(np.float32)
            np.save(xp, x)
            _run(LINEAR, str(dense), "--tensor", name, "--x", str(xp), "--y-out", str(yp), "--dtype", dtype,
                 "--path", path, "--json")
            xm = mx.array(x).astype(mdt)
            if path == "auto":
                ref = mod(xm)
            else:   # no residual, no output rotation: from_rotated with a forced kernel path
                ref = mod.kernel(mod.rotate_in(xm), path=path)
            ref = np.array(ref.astype(mx.float32))
            got = np.load(yp)
            if not np.array_equal(_bits(got), _bits(ref)):
                bad.append((name, b, path, int((got != ref).sum()), float(np.abs(got - ref).max())))
    assert not bad, f"C++ GPU output differs from the MLX SucLinear (tcq): {bad}"


def test_rows_exact_equals_single_rows(dense: Path, tmp_path: Path) -> None:
    """encode_rows_exact (vec tiles of up to 4 rows) == the one-row decode launch, row for row, and == Python."""
    from suc.runtime.compressed_model import module_from_encoded

    sm = SucModel.load(dense)
    xp, yp = tmp_path / "x.npy", tmp_path / "y.npy"
    for k in ("tcq_3inst_k2", "tcq_1mad_k3_v1", "tcq_cols_fp16_tail", "tcq_csr_fp16", "tcq_3inst_k3_T256"):
        name = f"t.{k}.weight"
        mod = module_from_encoded(sm.encoded_packed(name))
        x = np.random.default_rng(7).standard_normal((11, mod.in_features)).astype(np.float32)
        np.save(xp, x)
        _run(LINEAR, str(dense), "--tensor", name, "--x", str(xp), "--y-out", str(yp), "--dtype", "bf16",
             "--rows-exact")
        got = np.load(yp)
        for r in range(x.shape[0]):
            np.save(xp, x[r:r + 1])
            _run(LINEAR, str(dense), "--tensor", name, "--x", str(xp), "--y-out", str(yp), "--dtype", "bf16")
            one = np.load(yp)
            ref = np.array(mod(mx.array(x[r:r + 1]).astype(mx.bfloat16)).astype(mx.float32))
            assert np.array_equal(_bits(got[r:r + 1]), _bits(one)), (name, r)
            assert np.array_equal(_bits(one), _bits(ref)), (name, r)


# ------------------------------------------------------------------------------------------------ expert stacks

MOE_ARCH = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 128,
            "intermediate_size": 256, "moe_intermediate_size": 128, "num_experts": 8, "num_experts_per_tok": 2,
            "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": True, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 512,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000, "rope_scaling": None, "max_position_embeddings": 4096,
            "tie_word_embeddings": True, "torch_dtype": "bfloat16", "attention_bias": False, "hidden_act": "silu",
            "eos_token_id": 2}
_VQ = VectorCodebook(16, block_size=8, rotate=True, rotate_out=False, kmeans_iters=2)


def _expert_codec(variant: str, proj: str, e: int):
    """(codec, encode kwargs) of expert e. shared: tcq with one seed_in per stack (gate: 3inst, up: 1mad k3 with int8
    columns on odd experts only, down: pair3 with a fp16 CSR residual); per_expert: own seeds (per-row sign tables);
    mixed: vq experts (int8 columns) and tcq experts (shared seed_in, int8 columns) alternating, + additive-free."""
    seed_in = {"gate_proj": 101, "up_proj": 102, "down_proj": 103}[proj]
    if variant == "mixed":
        if e % 2 == 0:
            return cols(_VQ), {}
        return cols(tcq()), {"seed_in": seed_in}
    base = {"gate_proj": tcq(), "up_proj": tcq("1mad", k=3, V=1), "down_proj": tcq("pair3")}[proj]
    codec = {"gate_proj": base, "up_proj": cols(base) if e % 2 else base,
             "down_proj": ResidualCodec(base, 0.02, "fp16")}[proj]
    return codec, ({"seed_in": seed_in} if variant == "shared" else {})


@pytest.fixture(scope="module", params=["shared", "per_expert", "mixed"])
def moe(request, tmp_path_factory):
    from suc.quant.codecs import AffineRTN
    from suc.runtime.suc1_model import load_suc1_model

    variant = request.param
    path = tmp_path_factory.mktemp("tcq_moe") / f"{variant}.suc1"
    arch, rng = MOE_ARCH, np.random.default_rng(7)
    h, fi, E = arch["hidden_size"], arch["moe_intermediate_size"], arch["num_experts"]
    t, p, seed = {}, "model.layers.0.", 1
    t[p + "mlp.gate.weight"] = mx.array((0.5 * rng.standard_normal((E, h))).astype(np.float32)).astype(mx.bfloat16)
    for proj, shape in (("gate_proj", (fi, h)), ("up_proj", (fi, h)), ("down_proj", (h, fi))):
        for e in range(E):
            codec, kw = _expert_codec(variant, proj, e)
            t[f"{p}mlp.experts.{e}.{proj}.weight"] = codec.encode(_w(shape, 1000 + seed, 0.08), seed=seed, **kw)
            seed += 1
    t["model.embed_tokens.weight"] = AffineRTN(4, 64).encode(_w((arch["vocab_size"], h), 5, 1.0), seed=seed)
    write_suc1(path, architecture=arch, tokenizer_id="none", tensors=t)
    model, _, _ = load_suc1_model(path, group=False)
    return {"path": path, "variant": variant, "mlp": model.model.layers[0].mlp, "E": E}


def _stack_ref(mod, x: mx.array, eidx: np.ndarray) -> np.ndarray:
    from suc.kernels.suc_switch import SucSwitchMixed

    if isinstance(mod, SucSwitchMixed):   # the sub-stacks on their "gather" path for any row count (as C++)
        import suc.kernels.suc_switch as sw
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sw, "GATHER_MAX_ROWS", 1 << 20)
            y = mod(x[:, None, :], mx.array(eidx))
            return np.array(y.reshape(x.shape[0], -1).astype(mx.float32))
    assert mod.tcq is not None
    return np.array(mod.rows(x, mx.array(eidx), path="gather").astype(mx.float32))


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_switch_matches_python(moe, tmp_path, dtype) -> None:
    from suc.kernels.suc_switch import SucSwitchMixed

    mdt = {"bf16": mx.bfloat16, "f16": mx.float16}[dtype]
    rng = np.random.default_rng(3)
    sw = moe["mlp"].switch_mlp
    mixed = moe["variant"] == "mixed"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(sw, proj)
        assert isinstance(mod, SucSwitchMixed) == mixed
        if moe["variant"] == "shared":
            assert mod.signs_in.shape[0] == 1
        for n in (1, 3, 8, 40):
            x = mx.array(rng.standard_normal((n, mod.in_features)).astype(np.float32)).astype(mdt)
            eidx = rng.integers(0, moe["E"], n).astype(np.uint32)
            if n >= 3:
                eidx[1] = eidx[0]
            ref = _stack_ref(mod, x, eidx)
            np.save(tmp_path / "x.npy", np.array(x.astype(mx.float32)))
            np.save(tmp_path / "e.npy", eidx.astype(np.float32))
            base = [str(moe["path"]), "--layer", "0", "--stack", proj, "--x", str(tmp_path / "x.npy"), "--eidx",
                    str(tmp_path / "e.npy"), "--y-out", str(tmp_path / "y.npy"), "--dtype", dtype, "--no-verify"]
            runs = ([["--mixed", "routed"], ["--mixed", "all"], ["--mixed", "routed", "--tiles", "4"],
                     ["--mixed", "all", "--tiles", "2"]] if mixed else
                    [["--layout", "auto"], ["--layout", "stacked"], ["--layout", "table"], ["--tiles", "4"],
                     ["--tiles", "2", "--layout", "table"]])
            for extra in runs:
                env = {"SUC_TILE_BT": extra[extra.index("--tiles") + 1]} if "--tiles" in extra else None
                r = _run(MOE, *base, *extra, env=env)
                y = np.load(tmp_path / "y.npy")
                assert y.shape == ref.shape
                bad = int((y != ref).sum())
                assert bad == 0, (f"{moe['variant']} {proj} n={n} {extra}: {bad} of {y.size} differ, "
                                  f"max |d| {np.abs(y - ref).max()}\n{r.stderr}")


def test_moe_block_matches_python(moe, tmp_path) -> None:
    """Uniform stacks: the qwen3_moe block equals Python bit for bit. Mixed stacks: qwen3_moe blocks (MoeBlock) take
    uniform stacks only (mixed stacks run in deepseek_v4 / glm5_next, tested there and in test_switch_matches_python),
    and the supported contract is an explicit rejection — a nonzero exit with the layout diagnostic, never a crash
    or a silently wrong result."""
    if moe["variant"] == "mixed":
        np.save(tmp_path / "x.npy", np.zeros((1, MOE_ARCH["hidden_size"]), dtype=np.float32))
        r = subprocess.run([str(MOE), str(moe["path"]), "--layer", "0", "--x", str(tmp_path / "x.npy"), "--y-out",
                            str(tmp_path / "y.npy"), "--no-verify"], capture_output=True, text=True, timeout=120)
        if r.returncode != 0 and "no Metal device" in r.stderr:
            pytest.skip("no Metal device")
        assert r.returncode > 0, (r.returncode, r.stderr)       # an error exit, not a signal
        assert "experts differ in shape or code layout" in r.stderr, r.stderr
        assert not (tmp_path / "y.npy").exists()
        return
    rng = np.random.default_rng(5)
    mlp, h = moe["mlp"], MOE_ARCH["hidden_size"]
    for n in (1, 3, 16):
        x = mx.array(rng.standard_normal((1, n, h)).astype(np.float32)).astype(mx.bfloat16)
        ref = np.array(mlp(x).astype(mx.float32))[0]
        np.save(tmp_path / "x.npy", np.array(x.astype(mx.float32))[0])
        for layout in ("auto", "table"):
            _run(MOE, str(moe["path"]), "--layer", "0", "--x", str(tmp_path / "x.npy"), "--y-out",
                 str(tmp_path / "y.npy"), "--layout", layout, "--no-verify")
            y = np.load(tmp_path / "y.npy")
            assert np.array_equal(y, ref), f"{moe['variant']} n={n} {layout}: max |d| {np.abs(y - ref).max()}"


# ------------------------------------------------------------------------------------------------ model graphs

def _tcq_count(path: Path) -> int:
    sm = SucModel.load(path, verify=False)
    return sum(sm.descriptor(n)["representation_type"] == "tcq" for n in sm.names())


# qwen3 projections: q tcq on the shared q|k|v rotation (k|v vq: fused in the fused plan), o tcq + int8 columns,
# gate / up tcq (gate|up rotation shared, never fused), down tcq T = 256
QWEN3 = {
    "self_attn.q_proj": tcq(),
    "self_attn.k_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "self_attn.v_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "self_attn.o_proj": cols(tcq("1mad", k=3, V=1)),
    "mlp.gate_proj": tcq(k=3, L=12),
    "mlp.up_proj": ResidualCodec(tcq("pair3"), 0.02, "fp16"),
    "mlp.down_proj": tcq(T=256),
}
QWEN3_EXPERTS = {"gate_proj": SharedIn(tcq(), 201), "up_proj": SharedIn(cols(tcq("1mad", k=3, V=1)), 202),
                 "down_proj": SharedIn(ResidualCodec(tcq("pair3"), 0.02, "fp16"), 203)}
PROMPT = [5, 17, 300, 42, 7, 99]


@pytest.mark.parametrize("fused", [False, True])
def test_qwen3_graph(tmp_path, fused) -> None:
    import test_cpp_generate as g

    f = g._tiny_file(tmp_path / "tiny.suc1", True, codecs=QWEN3, shared_seeds=True)
    assert _tcq_count(f) == 2 * 5
    r = g._compare(f, PROMPT, 12, tmp_path, 1e-4, fused=fused)
    assert r["agree"] == 1.0


def test_qwen3_batch_prefill(tmp_path) -> None:
    """Chunked prefill vs token by token (as tests/unit/test_cpp_generate.py::test_tiny_batch_prefill): chunks of 4
    rows keep the tcq projections on the vec kernel (rows independent: the first position is bit-identical), chunks of
    8 run the mma kernel (weights rounded to the activation dtype inside the MMA): rounding-level spread only; the
    greedy tokens agree."""
    import test_cpp_generate as g
    from suc.runtime.suc1_model import kl_rows

    f = g._tiny_file(tmp_path / "tiny.suc1", True, codecs=QWEN3, shared_seeds=True)
    prompt = [5, 17, 300, 42, 7, 99, 11, 250, 3, 64, 128]
    gt = g._cpp(f, prompt, 8, tmp_path, "--prefill", "token")
    lt, ht = np.load(tmp_path / "logits.npy"), np.load(tmp_path / "hidden.npy")
    for chunk in ("4", "8"):
        gb = g._cpp(f, prompt, 8, tmp_path, "--prefill", "batch", "--chunk", chunk)
        lb, hb = np.load(tmp_path / "logits.npy"), np.load(tmp_path / "hidden.npy")
        if chunk == "4":
            assert np.array_equal(hb[0], ht[0])
        assert np.linalg.norm(hb - ht) / np.linalg.norm(ht) < 2e-2
        assert kl_rows(lt, lb).max() < 1e-3 and gb == gt


@pytest.mark.parametrize("fused", [False, True])
def test_qwen3_moe_graph(tmp_path, fused) -> None:
    import test_cpp_generate as g
    from suc.runtime.suc1_model import greedy

    f = g._moe_file(tmp_path / "moe.suc1", g.ARCH_MOE, codecs=QWEN3, expert_codecs=QWEN3_EXPERTS)
    assert _tcq_count(f) > 3 * g.ARCH_MOE["num_experts"]
    r = g._compare(f, PROMPT, 12, tmp_path, 1e-4, fused=fused)
    assert r["agree"] == 1.0
    assert greedy(r["model"], PROMPT, 12) == r["gen"]


def _graph_vs_python(path: Path, tmp: Path, cpp, prompt: list[int], steps: int, *args: str) -> dict:
    """C++ teacher-forced run == Python runtime (load_suc1_model(group=False, affine_modules=False)): logits and
    every stream bit-identical, greedy tokens identical. cpp: the _cpp helper of the graph's test module."""
    from suc.runtime.suc1_model import greedy, load_suc1_model, run_tokens

    gen, rec, _ = cpp(path, tmp, "--logits-out", str(tmp / "l.npy"), "--hidden-out", str(tmp / "h.npy"), *args,
                      tokens=prompt, steps=steps)
    lc, hc = np.load(tmp / "l.npy"), np.load(tmp / "h.npy")
    seq = prompt + gen[:-1]
    model, _, rep = load_suc1_model(path, group=False, affine_modules=False)
    lp, hp = run_tokens(model, seq)
    hp = hp[:, :, : hc.shape[2]]
    bad = np.argwhere((hp != hc).reshape(hp.shape[0], hp.shape[1], -1).any(-1))
    assert not len(bad), f"first differing (position, stream): {bad[:6].tolist()}"
    assert np.array_equal(lp, lc)
    assert greedy(model, prompt, steps) == gen
    return {"rec": rec, "report": rep, "gen": gen}


# deepseek_v4: wq_b tcq, wkv tcq + fp16 CSR, the shared expert's down tcq + int8 columns; routed experts e % 3 == 1 tcq
# (+ int8 columns, one input rotation per stack) inside mixed vq / tcq / additive (/ native MXFP4) stacks
@pytest.fixture(scope="module")
def ds4(tmp_path_factory):
    import test_cpp_deepseek_v4 as d

    tmp = tmp_path_factory.mktemp("tcq_ds4")
    attn = dict(d.ATTN, **{"self_attn.wq_b": lambda: tcq(),
                           "self_attn.wkv": lambda: ResidualCodec(tcq("1mad", k=3, V=1), 0.01, "fp16"),
                           "mlp.shared_experts.down_proj": lambda: cols(tcq("pair3"))})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(d, "ATTN", attn)
        mp.setattr(d, "EXPERT", [d.EXPERT[0], lambda: SharedIn(cols(tcq()), 301), d.EXPERT[2]])
        files = {"base": d.ds4_file(tmp / "base.suc1", native=False, over=d.TINY),
                 "native": d.ds4_file(tmp / "native.suc1", native=True, over=d.TINY)}
    for p in files.values():
        assert _tcq_count(p) > 0
    return {**files, "d": d}


@pytest.mark.parametrize("variant", ["base", "native"])
def test_deepseek_v4_graph(ds4, variant, tmp_path) -> None:
    d = ds4["d"]
    r = _graph_vs_python(ds4[variant], tmp_path, d._cpp, d.PROMPT, d.STEPS)
    assert r["rec"]["expert_stacks"]["mixed"] > 0
    assert "SucSwitchMixed" in {c["kind"] for c in r["report"]["converted"]}


def test_deepseek_v4_batched_prefill_and_streaming(ds4, tmp_path, monkeypatch) -> None:
    """Row-exact batched prefill (--prefill exact; tcq vec tiles of <= 4 rows; expert tiles) and SSD expert streaming
    (tcq experts read into slot pools) give the token-by-token / resident bits."""
    d = ds4["d"]
    path, prompt = ds4["native"], d.LONG_PROMPT
    ref, rr, _ = d._cpp(path, tmp_path, "--prefill", "token", tokens=prompt, steps=6)
    for chunk, bt in (("7", None), ("64", "4"), ("5", "2")):
        if bt:
            monkeypatch.setenv("SUC_TILE_BT", bt)
        else:
            monkeypatch.delenv("SUC_TILE_BT", raising=False)
        gen, rec, _ = d._cpp(path, tmp_path, "--prefill", "exact", "--chunk", chunk, tokens=prompt, steps=6)
        assert rec["prefill"] == "batch"
        assert rec["prefill_logits_sha256"] == rr["prefill_logits_sha256"], (chunk, bt)
        assert rec["prefill_streams_sha256"] == rr["prefill_streams_sha256"], (chunk, bt)
        assert gen == ref
    monkeypatch.delenv("SUC_TILE_BT", raising=False)
    a, _, _ = d._cpp(path, tmp_path, "--logits-out", str(tmp_path / "a.npy"), steps=12)
    _, rec, _ = d._cpp(path, tmp_path, "--expert-budget", str(1 << 34), steps=2)
    st = rec["expert_streaming"]
    budget = st["layers"] * 3 * st["slot_bytes_max"]
    b, rec, _ = d._cpp(path, tmp_path, "--logits-out", str(tmp_path / "b.npy"), "--expert-budget", str(budget),
                       steps=12)
    assert a == b and np.array_equal(np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"))
    assert rec["expert_streaming"]["evictions"] > 0


# glm5_next: MLA q_b tcq + int8 columns, dense MLP down tcq, shared expert gate tcq; routed experts e % 3 == 1 tcq
@pytest.fixture(scope="module")
def glm5(tmp_path_factory):
    import test_cpp_glm5 as g5

    tmp = tmp_path_factory.mktemp("tcq_glm5")
    proj = dict(g5.PROJ, **{"self_attn.q_b_proj": lambda: cols(tcq()), "mlp.down_proj": lambda: tcq("1mad", k=3, V=1),
                            "mlp.shared_experts.gate_proj": lambda: tcq("pair3")})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(g5, "PROJ", proj)
        mp.setattr(g5, "EXPERT", [g5.EXPERT[0], lambda: SharedIn(tcq(k=3, L=12), 401), g5.EXPERT[2]])
        path = g5.glm5_file(tmp / "model.suc1", over=g5.TINY)
    assert _tcq_count(path) > 0
    return {"path": path, "g5": g5}


def test_glm5_graph(glm5, tmp_path) -> None:
    g5 = glm5["g5"]
    r = _graph_vs_python(glm5["path"], tmp_path, g5._cpp, g5.PROMPT, g5.STEPS)
    assert r["rec"]["expert_stacks"]["mixed"] > 0


def test_glm5_streamed_experts(glm5, tmp_path) -> None:
    g5 = glm5["g5"]
    a, _, _ = g5._cpp(glm5["path"], tmp_path, "--logits-out", str(tmp_path / "a.npy"), steps=12)
    _, rec, _ = g5._cpp(glm5["path"], tmp_path, "--expert-budget", str(1 << 34), steps=2)
    st = rec["expert_streaming"]
    budget = st["layers"] * 3 * st["slot_bytes_max"]
    b, rec, _ = g5._cpp(glm5["path"], tmp_path, "--logits-out", str(tmp_path / "b.npy"), "--expert-budget",
                        str(budget), steps=12)
    assert a == b and np.array_equal(np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"))


# qwen4_exp / qwen3_5: files of `suc compress` (vq, expert-stack codebooks, rotation groups) at widths that allow
# T = 128 sequences, whose projections with in % 128 == 0 are then re-encoded as tcq from their decoded weights (dense
# tensors keep their input rotation seed, so rotation groups stay shared; every expert stack becomes one tcq stack
# with one seed_in)
def tcq_rewrite(src: Path, dst: Path) -> int:
    sm = SucModel.load(src, verify=False)
    codecs = [tcq(), cols(tcq("1mad", k=3, V=1)), tcq("pair3")]
    t, stacks, n_tcq = {}, {}, 0
    for n in sm.names():
        d = sm.descriptor(n)
        ok = (d["representation_type"] in ("vq", "additive") and len(d["shape"]) == 2 and d["shape"][1] % 128 == 0
              and "embed" not in n and "lm_head" not in n)
        m = re.match(r"(.+\.experts)\.(\d+)\.(.+)$", n)
        if ok and m:
            stacks.setdefault((m.group(1), m.group(3)), []).append(n)
        elif ok:
            p = d["params"]["base"] if d["residual_type"] != "none" else d["params"]
            kw = {"seed_in": int(p["seed_in"])} if p.get("rotate") and p.get("seed_in") is not None else {}
            t[n] = codecs[n_tcq % 3].encode(sm.decode_tensor(n), seed=n_tcq + 1, **kw)
            n_tcq += 1
        else:
            t[n] = stored(sm, n)
    for i, (_, names) in enumerate(sorted(stacks.items())):
        for j, n in enumerate(names):
            t[n] = codecs[i % 3].encode(sm.decode_tensor(n), seed=1000 * (i + 1) + j, seed_in=500 + i)
            n_tcq += 1
    write_suc1(dst, architecture=dict(sm.architecture), tokenizer_id=sm.tokenizer_id, tensors=t,
               metadata=dict(sm.metadata))
    return n_tcq


@pytest.fixture(scope="module", params=["qwen4_exp", "qwen3_5"])
def qwen_next(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"tcq_{request.param}")
    if request.param == "qwen4_exp":
        import test_cpp_qwen4 as q
        src = q._compress(tmp, hidden_size=128, moe_intermediate_size=128, shared_expert_intermediate_size=128,
                          ple_embed_dim=128)
    else:
        import test_cpp_qwen3_5 as q
        src = q._compress(tmp, hidden_size=128, intermediate_size=256)
    dst = tmp / "tcq.suc1"
    n = tcq_rewrite(src, dst)
    assert n > 0
    return {"path": dst, "kind": request.param, "q": q, "n_tcq": n}


def test_qwen_next_graph(qwen_next, tmp_path) -> None:
    """Same acceptance as tests/unit/test_cpp_qwen4.py / test_cpp_qwen3_5.py (their float32 recurrences may drift by
    ulps on the M1 Ultra): per-position KL, argmax and greedy agreement; the bit-identical fraction is printed."""
    from suc.runtime.suc1_model import greedy, kl_rows, load_suc1_model, run_tokens

    q, path = qwen_next["q"], qwen_next["path"]
    gen, rec = q._cpp(path, tmp_path, "--logits-out", str(tmp_path / "l.npy"), "--hidden-out", str(tmp_path / "h.npy"))
    assert rec["model_type"] == qwen_next["kind"]
    lc, hc = np.load(tmp_path / "l.npy"), np.load(tmp_path / "h.npy")
    seq = q.PROMPT + gen[:-1]
    model, _, _ = load_suc1_model(path, group=False, affine_modules=False)
    lp, hp = run_tokens(model, seq)
    hp = hp[:, :, : hc.shape[2]]
    kl = kl_rows(lp, lc)
    print(f"\n[{qwen_next['kind']}: {qwen_next['n_tcq']} tcq tensors] positions {len(seq)}: logits bit-identical "
          f"{(lp == lc).mean():.4f}, streams {(hp == hc).mean():.4f}; KL max {kl.max():.3g}")
    assert kl.max() < 1e-3
    top2 = np.sort(lp, axis=-1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 0.25
    assert (lp.argmax(-1) == lc.argmax(-1))[clear].all()
    py = greedy(model, q.PROMPT, q.STEPS)
    first = next((i for i, (x, y) in enumerate(zip(py, gen)) if x != y), None)
    assert first is None or not clear[len(q.PROMPT) - 1 + first], (first, py, gen)


def test_qwen_next_batched_prefill(qwen_next, tmp_path, monkeypatch) -> None:
    """Row-exact batched (layer-major) prefill (--prefill exact) = token-by-token prefill (sha256 of the last prompt
    token's logits and streams)."""
    q, path = qwen_next["q"], qwen_next["path"]
    prompt = [int(x) for x in np.random.default_rng(7).integers(2, 127, size=45)]
    ref, rr = q._cpp(path, tmp_path, "--prefill", "token", tokens=prompt, steps=6)
    for chunk, bt in (("7", None), ("64", "4")):
        if bt:
            monkeypatch.setenv("SUC_TILE_BT", bt)
        else:
            monkeypatch.delenv("SUC_TILE_BT", raising=False)
        gen, rec = q._cpp(path, tmp_path, "--prefill", "exact", "--chunk", chunk, tokens=prompt, steps=6)
        assert rec["prefill"] == "batch"
        assert rec["prefill_logits_sha256"] == rr["prefill_logits_sha256"], chunk
        assert rec["prefill_streams_sha256"] == rr["prefill_streams_sha256"], chunk
        assert gen == ref
