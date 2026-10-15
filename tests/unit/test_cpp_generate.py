"""Cross-check of the C++/Metal model graph (cpp/, milestone 3; tool suc1-generate) with the Python/MLX runtime of the
same SUC1 file (suc.runtime.suc1_model.load_suc1_model, group=False: one SucLinear / SucResidualLinear per projection,
the C++ execution plan).

Per position (token-by-token decode, teacher-forced on the C++ token sequence): the hidden state after the embedding
and after every block and the logits must be bit-identical (the LM head runs MLX's gemv accumulation order on the
decoded weight since milestone 4); KL and greedy-token agreement are checked as well. Milestone 4 adds the fused
q|k|v / gate|up plan (C++ default) against the default Python runtime (group=True, fuse=True), pipelined decode, the
text prompt (C++ tokenizer) and qwen3_moe files. Synthetic tiny Qwen3 (head_dim 64, the smallest size MLX runs with its
sdpa_vector kernel), tied (affine 4-bit embedding = LM head) and untied (raw bf16 LM head); `-m model`: the dev
container, ≥ 64 generated tokens, free-running greedy agreement.

Skipped unless suc1-generate is built (cpp/build.sh) into $SUC_CPP_BUILD (default ~/suc-local/cpp-build).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from suc.evaluation.reconstruct import rotation_seeds
from suc.io.suc1 import write_suc1
from suc.quant.codecs import AdditiveCodebook, AffineRTN, VectorCodebook
from suc.quant.residual import ResidualCodec
from suc.runtime.suc1_model import greedy, kl_rows, load_suc1_model, run_tokens

BUILD = Path(os.environ.get("SUC_CPP_BUILD", Path.home() / "suc-local" / "cpp-build"))
GENERATE = BUILD / "suc1-generate"
DEV_FILE = Path(os.environ.get("SUC1_DEV_FILE", Path.home() / "suc-local/compress/compress_qwen3_0.6b/container/model.suc1"))

pytestmark = pytest.mark.skipif(not GENERATE.exists(), reason=f"suc1-generate not built in {BUILD} (run cpp/build.sh)")

ARCH = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "vocab_size": 512, "rms_norm_eps": 1e-6, "rope_theta": 1000000, "rope_scaling": None,
        "max_position_embeddings": 4096, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "attention_bias": False, "hidden_act": "silu", "eos_token_id": 2}

# one codec per projection role: rotations in_out / one-sided, additive and vq, CSR / column residuals
CODECS = {
    "self_attn.q_proj": ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=2),
                                      0.01, "int8"),
    "self_attn.k_proj": VectorCodebook(256, block_size=8, rotate=True, kmeans_iters=2),
    "self_attn.v_proj": VectorCodebook(16, block_size=4, rotate=True, rotate_out=False, kmeans_iters=2),
    "self_attn.o_proj": ResidualCodec(VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), 0.02, "int8",
                                      structure="columns"),
    "mlp.gate_proj": AdditiveCodebook(16, block_size=8, num_codebooks=3, rotate=True, kmeans_iters=2),
    "mlp.up_proj": ResidualCodec(VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), 0.01, "fp16"),
    "mlp.down_proj": VectorCodebook(64, block_size=8, rotate=True, kmeans_iters=2),
}
# Qwen3-8B widths (hidden 4096: the 1024-thread RMSNorm, q|k|v / gate|up prologue at 4096; intermediate 3072 =
# 12 · 256: an H_12 rotation like 8B's 12288), one block, 8 query / 2 KV heads of 128
WIDE = dict(ARCH, hidden_size=4096, intermediate_size=3072, num_hidden_layers=1, num_attention_heads=8,
            num_key_value_heads=2, head_dim=128, vocab_size=1024)
WIDE_CODECS = {r: ResidualCodec(VectorCodebook(256, block_size=8, rotate=True, kmeans_iters=1, fit_blocks=65536),
                                0.005, "int8") for r in CODECS}


def _randn(shape, rng, scale):
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32))


def _shapes(arch: dict) -> dict[str, tuple[int, int]]:
    h, f = arch["hidden_size"], arch["intermediate_size"]
    q, kv = arch["num_attention_heads"] * arch["head_dim"], arch["num_key_value_heads"] * arch["head_dim"]
    return {"self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h), "self_attn.v_proj": (kv, h),
            "self_attn.o_proj": (h, q), "mlp.gate_proj": (f, h), "mlp.up_proj": (f, h), "mlp.down_proj": (h, f)}


def _tiny_file(path: Path, tie: bool, arch: dict | None = None, codecs: dict | None = None,
               shared_seeds: bool = False) -> Path:
    """shared_seeds: q|k|v and gate|up share seed_in (suc.evaluation.reconstruct.rotation_seeds, as the compress
    pipeline encodes them), so the runtimes share / fuse their input rotation."""
    rng = np.random.default_rng(3)
    arch = dict(arch or ARCH, tie_word_embeddings=tie)
    codecs = codecs or CODECS
    h, d, vocab = arch["hidden_size"], arch["head_dim"], arch["vocab_size"]
    t: dict = {}
    seed = 1
    for i in range(arch["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for role, shape in _shapes(arch).items():
            name = p + role + ".weight"
            extra = rotation_seeds(name, 11) if shared_seeds else {}
            t[name] = codecs[role].encode(_randn(shape, rng, 0.08 * (128 / shape[1]) ** 0.5), seed=seed, **extra)
            seed += 1
        for n, size in (("input_layernorm", h), ("post_attention_layernorm", h), ("self_attn.q_norm", d),
                        ("self_attn.k_norm", d)):
            t[p + n + ".weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(size).astype(np.float32)).astype(mx.bfloat16)
    t["model.norm.weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(h).astype(np.float32)).astype(mx.bfloat16)
    t["model.embed_tokens.weight"] = AffineRTN(4, 64).encode(_randn((vocab, h), rng, 1.0), seed=seed)
    if not tie:
        t["lm_head.weight"] = _randn((vocab, h), rng, 0.1 * (128 / h) ** 0.5).astype(mx.bfloat16)
    write_suc1(path, architecture=arch, tokenizer_id="none", tensors=t)
    return path


def _cpp(file: Path, tokens: list[int], steps: int, out: Path, *extra: str) -> list[int]:
    r = subprocess.run([str(GENERATE), str(file), "--tokens", ",".join(map(str, tokens)), "--steps", str(steps),
                        "--logits-out", str(out / "logits.npy"), "--hidden-out", str(out / "hidden.npy"),
                        "--quiet", *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("generated:"))
    return [int(x) for x in line.split(":", 1)[1].split(",") if x.strip()]


def _compare(file: Path, prompt: list[int], steps: int, tmp: Path, kl_max: float, *extra: str,
             fused: bool = False) -> dict:
    """fused=False: C++ --projections separate vs the Python plan runtime (group=False); fused=True: the C++ default
    (fused q|k|v / gate|up launches) vs the default Python runtime (group=True, fuse=True)."""
    gen = _cpp(file, prompt, steps, tmp, "--prefill", "token", "--projections", "fused" if fused else "separate", *extra)
    assert len(gen) == steps
    seq = prompt + gen[:-1]
    cl, ch = np.load(tmp / "logits.npy"), np.load(tmp / "hidden.npy")
    model, _, _ = load_suc1_model(file, group=fused, fuse=fused)
    pl, ph = run_tokens(model, seq)
    assert ch.shape == ph.shape and cl.shape == pl.shape
    # hidden states after the embedding and every block: bit-identical at every position
    bad = [(p, l) for p in range(ph.shape[0]) for l in range(ph.shape[1]) if not np.array_equal(ph[p, l], ch[p, l])]
    assert not bad, f"hidden states differ at (position, layer) {bad[:5]} (of {len(bad)})"
    # logits: the LM head runs MLX's gemv accumulation order (milestone 4), so they are bit-identical too
    assert np.array_equal(pl, cl), f"logits differ at {np.argwhere(pl != cl)[:5].tolist()}"
    kl = kl_rows(pl, cl)
    assert kl.max() < kl_max, kl.max()
    # greedy agreement (teacher-forced): Python's argmax at every position that produced a C++ token
    py_next = pl.argmax(-1)[len(prompt) - 1:]
    agree = float(np.mean(py_next == np.array(gen)))
    return {"gen": gen, "kl_max": float(kl.max()), "agree": agree, "model": model}


@pytest.mark.parametrize("tie,opts", [(True, ()), (False, ()), (True, ("--fuse-norm", "on")),
                                      (True, ("--fuse-norm", "off", "--no-shared-rotations"))])
def test_tiny_model_matches_python_runtime(tmp_path, tie, opts) -> None:
    f = _tiny_file(tmp_path / "tiny.suc1", tie)
    r = _compare(f, [5, 17, 300, 42, 7, 99], 16, tmp_path, 1e-4, *opts)
    assert r["agree"] == 1.0


def test_wide_model_matches_python_runtime(tmp_path) -> None:
    f = _tiny_file(tmp_path / "wide.suc1", True, WIDE, WIDE_CODECS)
    r = _compare(f, [5, 17, 300, 42, 7, 99, 1000, 3], 12, tmp_path, 1e-4)
    assert r["agree"] == 1.0


# fusable members (same code layout and seed_in): q|k|v additive M = 2 (q with an int8 CSR residual, v with int8
# column residual), gate|up vq K = 256 (up with an fp16 CSR residual: one structured residual launch per group)
FUSE_CODECS = dict(CODECS, **{
    "self_attn.q_proj": ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=2),
                                      0.01, "int8"),
    "self_attn.k_proj": AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=2),
    "self_attn.v_proj": ResidualCodec(AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=2),
                                      0.02, "int8", structure="columns"),
    "mlp.gate_proj": VectorCodebook(256, block_size=8, rotate=True, kmeans_iters=2),
    "mlp.up_proj": ResidualCodec(VectorCodebook(256, block_size=8, rotate=True, kmeans_iters=2), 0.01, "fp16"),
})
# no residuals (split output rotation straight into the member buffers); k not fusable (other layout): q|v fused + k alone
SPLIT_CODECS = dict(CODECS, **{
    "self_attn.q_proj": VectorCodebook(64, block_size=4, rotate=True, kmeans_iters=2),
    "self_attn.k_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
    "self_attn.v_proj": VectorCodebook(64, block_size=4, rotate=True, kmeans_iters=2),
    "mlp.gate_proj": AdditiveCodebook(16, block_size=8, num_codebooks=3, rotate=True, kmeans_iters=2),
    "mlp.up_proj": AdditiveCodebook(16, block_size=8, num_codebooks=3, rotate=True, kmeans_iters=2),
})
# one-sided rotations (rotate_out False): the fused output is written unrotated, then distributed
ONESIDED_CODECS = dict(CODECS, **{r: VectorCodebook(16, block_size=8, rotate=True, rotate_out=False, kmeans_iters=2)
                                  for r in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                            "mlp.gate_proj", "mlp.up_proj")})


@pytest.mark.parametrize("codecs", ["fuse", "split", "onesided"])
@pytest.mark.parametrize("opts", [(), ("--fuse-norm", "on"), ("--fuse-norm", "off")])
def test_tiny_fused_projections_match_default_runtime(tmp_path, codecs, opts) -> None:
    """C++ default plan (FusedLinear q|k|v / gate|up) vs the default Python runtime: bit-identical hidden states."""
    import suc.runtime.compressed_model as cm
    cs = {"fuse": FUSE_CODECS, "split": SPLIT_CODECS, "onesided": ONESIDED_CODECS}[codecs]
    f = _tiny_file(tmp_path / "tiny.suc1", True, codecs=cs, shared_seeds=True)
    old = cm.FUSE_ELEMENTWISE
    cm.FUSE_ELEMENTWISE = {(): None, ("--fuse-norm", "on"): True, ("--fuse-norm", "off"): False}[opts]
    try:
        r = _compare(f, [5, 17, 300, 42, 7, 99], 12, tmp_path, 1e-4, *opts, "--fuse-swiglu",
                     "on" if cm.FUSE_ELEMENTWISE else ("off" if cm.FUSE_ELEMENTWISE is False else "auto"), fused=True)
    finally:
        cm.FUSE_ELEMENTWISE = old
    assert r["agree"] == 1.0
    groups = r["model"].model.layers[0].self_attn["suc_qkv"]
    assert groups.fused_parts == 1


@pytest.mark.parametrize("codecs,fused,opts", [("fuse", True, ("--fuse-norm", "on", "--fuse-swiglu", "on")),
                                                ("split", True, ()), ("fuse", False, ())])
def test_row_rotation_layout_matches_python(tmp_path, monkeypatch, codecs, fused, opts) -> None:
    """A tuning cache selecting the rotation "row" layout (as the M1 Ultra's does for n = 1024): single-segment,
    multi-segment (fused q|k|v / gate|up output rotation) and prologue (RMSNorm / SwiGLU + rotation) launches all
    follow it, bit-identical to the Python runtime."""
    import suc.runtime.compressed_model as cm
    from tiny_models import row_rotation_cache
    row_rotation_cache(monkeypatch, tmp_path / "tuning.json")
    cs = {"fuse": FUSE_CODECS, "split": SPLIT_CODECS}[codecs]
    f = _tiny_file(tmp_path / "tiny.suc1", True, codecs=cs, shared_seeds=fused)
    monkeypatch.setattr(cm, "FUSE_ELEMENTWISE", True if opts else False)
    r = _compare(f, [5, 17, 300, 42, 7, 99], 12, tmp_path, 1e-4, *opts, fused=fused)
    assert r["agree"] == 1.0


def test_tiny_shared_seeds_separate_plan(tmp_path) -> None:
    """Shared seeds with --projections separate (shared rotation, stage-by-stage members) = Python plan runtime."""
    f = _tiny_file(tmp_path / "tiny.suc1", True, codecs=FUSE_CODECS, shared_seeds=True)
    r = _compare(f, [5, 17, 300, 42, 7, 99], 12, tmp_path, 1e-4)
    assert r["agree"] == 1.0


@pytest.mark.parametrize("chunk", [3, 64])
def test_tiny_batch_prefill(tmp_path, chunk) -> None:
    """Prefill in chunks (projections at batch n, causal attention inside the chunk) vs token by token: the first
    position is computed alike (one row); later ones differ by kernel rounding only (vec vs mma lookup paths)."""
    f = _tiny_file(tmp_path / "tiny.suc1", True)
    prompt = [5, 17, 300, 42, 7, 99, 11, 250, 3, 64, 128]
    gt = _cpp(f, prompt, 8, tmp_path, "--prefill", "token")
    lt, ht = np.load(tmp_path / "logits.npy"), np.load(tmp_path / "hidden.npy")
    gb = _cpp(f, prompt, 8, tmp_path, "--prefill", "batch", "--chunk", str(chunk))
    lb, hb = np.load(tmp_path / "logits.npy"), np.load(tmp_path / "hidden.npy")
    assert lb.shape == lt.shape and hb.shape == ht.shape
    assert np.array_equal(hb[0], ht[0])
    rel = np.linalg.norm(hb - ht) / np.linalg.norm(ht)
    assert rel < 2e-2, rel
    assert kl_rows(lt, lb).max() < 1e-3
    assert gb == gt


def test_tiny_teacher_forcing_and_errors(tmp_path) -> None:
    f = _tiny_file(tmp_path / "tiny.suc1", True)
    forced = [3, 4, 5, 6]
    _cpp(f, [1, 2], 0, tmp_path, "--force", ",".join(map(str, forced)), "--prefill", "token")
    cl = np.load(tmp_path / "logits.npy")
    assert cl.shape == (2 + len(forced) - 1, 512)
    model, _, _ = load_suc1_model(f, group=False)
    pl, _ = run_tokens(model, [1, 2] + forced[:-1], hidden=False)
    assert kl_rows(pl, cl).max() < 1e-4
    r = subprocess.run([str(GENERATE), str(f), "--tokens", "600"], capture_output=True, text=True)
    assert r.returncode != 0 and "vocab" in r.stderr


@pytest.mark.model
@pytest.mark.skipif(not DEV_FILE.exists(), reason=f"dev container {DEV_FILE} not present")
def test_dev_container_greedy_64(tmp_path) -> None:
    # "The history of the printing press begins in the fifteenth century," (Qwen3 tokenizer)
    prompt = [785, 3840, 315, 279, 18484, 3493, 12033, 304, 279, 11128, 45387, 9294, 11]
    r = _compare(DEV_FILE, prompt, 64, tmp_path, kl_max=1e-4)
    assert r["agree"] == 1.0
    assert greedy(r["model"], prompt, 64) == r["gen"]   # free-running Python greedy: the same 64 tokens


@pytest.mark.model
@pytest.mark.skipif(not DEV_FILE.exists(), reason=f"dev container {DEV_FILE} not present")
def test_dev_container_fused_default_runtime(tmp_path) -> None:
    """The C++ default (fused) plan vs the default Python runtime (group=True, fuse=True)."""
    prompt = [785, 3840, 315, 279, 18484, 3493, 12033, 304, 279, 11128, 45387, 9294, 11]
    r = _compare(DEV_FILE, prompt, 64, tmp_path, kl_max=1e-4, fused=True)
    assert r["agree"] == 1.0
    assert greedy(r["model"], prompt, 64) == r["gen"]


@pytest.mark.parametrize("extra", [(), ("--depth", "3"), ("--projections", "separate")])
def test_tiny_pipelined_decode_matches_sync(tmp_path, extra) -> None:
    """Pipelined decode (argmax written to the GPU token buffer, next step encoded while the GPU runs) produces the
    same tokens as waiting for every step, and the same as the Python runtime's free-running greedy."""
    f = _tiny_file(tmp_path / "tiny.suc1", True, codecs=FUSE_CODECS, shared_seeds=True)
    prompt = [5, 17, 300, 42, 7, 99]

    def run(*a: str) -> list[int]:
        r = subprocess.run([str(GENERATE), str(f), "--tokens", ",".join(map(str, prompt)), "--steps", "24", "--quiet",
                            *extra, *a], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        line = next(ln for ln in r.stdout.splitlines() if ln.startswith("generated:"))
        return [int(x) for x in line.split(":", 1)[1].split(",") if x.strip()]
    pipelined, sync = run(), run("--sync")
    assert pipelined == sync and len(pipelined) == 24
    fused = "separate" not in extra
    model, _, _ = load_suc1_model(f, group=fused, fuse=fused)
    assert greedy(model, prompt, 24) == pipelined


@pytest.mark.model
@pytest.mark.skipif(not DEV_FILE.exists(), reason=f"dev container {DEV_FILE} not present")
def test_dev_container_text_prompt(tmp_path) -> None:
    """--prompt TEXT: the C++ tokenizer (pinned by the SUC1 tokenizer_id) gives transformers' ids, and the printed
    text is transformers' decode of the generated ids."""
    from transformers import AutoTokenizer
    text = "The history of the printing press begins in the fifteenth century,"
    r = subprocess.run([str(GENERATE), str(DEV_FILE), "--prompt", text, "--steps", "24", "--quiet"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    gen = [int(x) for x in next(ln for ln in lines if ln.startswith("generated:")).split(":", 1)[1].split(",")]
    out_text = r.stdout.split("\ntext: ", 1)[1][:-1]   # the last output: the decoded text (may span lines) + "\n"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids = tok.encode(text, add_special_tokens=False)
    assert gen == _cpp(DEV_FILE, ids, 24, tmp_path)
    assert out_text == tok.decode(gen)


# ------------------------------------------------------------------------------------------------ MoE (qwen3_moe)
ARCH_MOE = dict(ARCH, architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe", num_experts=8,
                num_experts_per_tok=2, moe_intermediate_size=128, decoder_sparse_step=1, mlp_only_layers=[],
                norm_topk_prob=True)
EXPERT_CODECS = {"gate_proj": VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2),
                 "up_proj": ResidualCodec(VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=2), 0.01, "int8"),
                 "down_proj": AdditiveCodebook(16, block_size=8, num_codebooks=2, rotate=True, kmeans_iters=2)}
# Column-structured expert residuals (the final Qwen3-30B-A3B family: VQ G=2 K=256 one-sided + int8 columns; up with
# fp16 columns and an fp16 CSR tail)
EXPERT_COLS = {"gate_proj": ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False,
                                                         kmeans_iters=2), 0.02, "int8", structure="columns"),
               "up_proj": ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False,
                                                       kmeans_iters=2), 0.02, "fp16", structure="columns",
                                        column_share=0.5),
               "down_proj": ResidualCodec(VectorCodebook(256, block_size=2, rotate=True, rotate_out=False,
                                                         kmeans_iters=2), 0.02, "int8", structure="columns")}


def _moe_file(path: Path, arch: dict, codecs: dict = FUSE_CODECS, expert_codecs: dict = EXPERT_CODECS) -> Path:
    """Tiny random Qwen3-MoE SUC1 file: SUC attention (shared seeds), a raw bf16 router, every expert of every
    projection its own SUC tensor (own codebooks / seeds); mlp_only_layers get a dense SUC MLP."""
    rng = np.random.default_rng(5)
    h, d, vocab, E = arch["hidden_size"], arch["head_dim"], arch["vocab_size"], arch["num_experts"]
    fi = arch["moe_intermediate_size"]
    t: dict = {}
    seed = 1
    for i in range(arch["num_hidden_layers"]):
        p = f"model.layers.{i}."
        sparse = i not in arch["mlp_only_layers"] and (i + 1) % arch["decoder_sparse_step"] == 0
        for role, shape in _shapes(arch).items():
            if sparse and role.startswith("mlp."):
                continue
            name = p + role + ".weight"
            t[name] = codecs[role].encode(_randn(shape, rng, 0.08 * (128 / shape[1]) ** 0.5), seed=seed,
                                          **rotation_seeds(name, 11))
            seed += 1
        if sparse:
            t[p + "mlp.gate.weight"] = _randn((E, h), rng, 0.3).astype(mx.bfloat16)
            for e in range(E):
                for proj, shape in (("gate_proj", (fi, h)), ("up_proj", (fi, h)), ("down_proj", (h, fi))):
                    w = _randn(shape, rng, 0.08 * (128 / shape[1]) ** 0.5)
                    t[p + f"mlp.experts.{e}.{proj}.weight"] = expert_codecs[proj].encode(w, seed=seed)
                    seed += 1
        for n, size in (("input_layernorm", h), ("post_attention_layernorm", h), ("self_attn.q_norm", d),
                        ("self_attn.k_norm", d)):
            t[p + n + ".weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(size).astype(np.float32)).astype(mx.bfloat16)
    t["model.norm.weight"] = mx.array(1.0 + 0.2 * rng.standard_normal(h).astype(np.float32)).astype(mx.bfloat16)
    t["model.embed_tokens.weight"] = AffineRTN(4, 64).encode(_randn((vocab, h), rng, 1.0), seed=seed)
    write_suc1(path, architecture=arch, tokenizer_id="none", tensors=t)
    return path


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("variant", ["top2", "top4_dense1", "no_norm", "top2_cols"])
def test_tiny_moe_matches_python_runtime(tmp_path, variant, fused) -> None:
    """qwen3_moe: router (raw bf16), softmax / top-k / norm_topk_prob, SucSwitchLinear expert stacks, combine; every
    hidden state bit-identical to the Python runtime (plan or default), teacher-forced and greedy. top2_cols: column-
    structured expert residuals."""
    arch = {"top2": ARCH_MOE, "top4_dense1": dict(ARCH_MOE, num_experts_per_tok=4, mlp_only_layers=[1]),
            "no_norm": dict(ARCH_MOE, norm_topk_prob=False), "top2_cols": ARCH_MOE}[variant]
    f = _moe_file(tmp_path / "moe.suc1", arch,
                  expert_codecs=EXPERT_COLS if variant.endswith("_cols") else EXPERT_CODECS)
    r = _compare(f, [5, 17, 300, 42, 7, 99], 12, tmp_path, 1e-4, fused=fused)
    assert r["agree"] == 1.0
    assert greedy(r["model"], [5, 17, 300, 42, 7, 99], 12) == r["gen"]


@pytest.mark.parametrize("cols", [False, True])
def test_tiny_moe_batch_prefill(tmp_path, cols) -> None:
    """MoE prefill in chunks (router / experts at n rows) vs token by token: same first row, rounding-level spread."""
    f = _moe_file(tmp_path / "moe.suc1", dict(ARCH_MOE, num_experts_per_tok=4),
                  expert_codecs=EXPERT_COLS if cols else EXPERT_CODECS)
    prompt = [5, 17, 300, 42, 7, 99, 11, 250, 3, 64, 128, 77, 401, 9, 23, 500, 1, 2]
    gt = _cpp(f, prompt, 8, tmp_path, "--prefill", "token")
    lt, ht = np.load(tmp_path / "logits.npy"), np.load(tmp_path / "hidden.npy")
    gb = _cpp(f, prompt, 8, tmp_path, "--prefill", "batch", "--chunk", "16")
    lb, hb = np.load(tmp_path / "logits.npy"), np.load(tmp_path / "hidden.npy")
    assert np.array_equal(hb[0], ht[0])
    assert np.linalg.norm(hb - ht) / np.linalg.norm(ht) < 2e-2
    assert kl_rows(lt, lb).max() < 1e-3
    assert gb == gt


@pytest.mark.parametrize("codecs", ["fuse", "split", "onesided"])
def test_tiny_fused_batch_prefill_equals_separate(tmp_path, codecs) -> None:
    """Batch prefill (n > 1 rows: fused outputs distributed by strided copies) with fused groups equals the separate
    plan bit for bit at these sizes (the tiny shapes give both plans the same rotation block shapes)."""
    cs = {"fuse": FUSE_CODECS, "split": SPLIT_CODECS, "onesided": ONESIDED_CODECS}[codecs]
    f = _tiny_file(tmp_path / "tiny.suc1", True, codecs=cs, shared_seeds=True)
    prompt = [5, 17, 300, 42, 7, 99, 11, 250, 3, 64, 128]
    out = {}
    for mode in ("fused", "separate"):
        g = _cpp(f, prompt, 6, tmp_path, "--prefill", "batch", "--chunk", "8", "--projections", mode)
        out[mode] = (g, np.load(tmp_path / "hidden.npy"))
    assert out["fused"][0] == out["separate"][0]
    assert np.array_equal(out["fused"][1], out["separate"][1])


@pytest.mark.model
@pytest.mark.skipif(not DEV_FILE.exists(), reason=f"dev container {DEV_FILE} not present")
@pytest.mark.parametrize("system,think", [(None, True), (None, False), ("Answer in one sentence.", True)])
def test_dev_container_chat_template(system, think) -> None:
    """--chat: the prompt ids equal transformers' apply_chat_template(add_generation_prompt=True, enable_thinking)."""
    from transformers import AutoTokenizer
    text = "What is the capital of France?\nAnswer briefly:  "
    args = [str(GENERATE), str(DEV_FILE), "--prompt", text, "--chat", "--steps", "4", "--quiet"]
    if system:
        args += ["--system", system]
    if not think:
        args += ["--no-think"]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ids = [int(x) for x in next(ln for ln in r.stdout.splitlines() if ln.startswith("prompt:")).split(":", 1)[1].split(",")]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=think)
    ref = ref["input_ids"] if isinstance(ref, dict) or hasattr(ref, "keys") else ref
    assert ids == list(ref)


def test_tiny_sampling_reproducible(tmp_path) -> None:
    """--temperature > 0: same seed -> same tokens; top-k 1 and temperature 0 -> the greedy tokens."""
    f = _tiny_file(tmp_path / "tiny.suc1", True)

    def run(*a: str) -> list[int]:
        r = subprocess.run([str(GENERATE), str(f), "--tokens", "5,17,300", "--steps", "20", "--quiet", *a],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return [int(x) for x in next(ln for ln in r.stdout.splitlines() if ln.startswith("generated:"))
                .split(":", 1)[1].split(",")]
    greedy_ids = run()
    assert run("--temperature", "0") == greedy_ids
    assert run("--temperature", "0.8", "--top-k", "1") == greedy_ids
    a = run("--temperature", "1.5", "--seed", "3")
    assert a == run("--temperature", "1.5", "--seed", "3")
    draws = {tuple(run("--temperature", "100", "--seed", str(s))) for s in range(4)}
    assert len(draws) > 1          # different seeds give different continuations (T = 100 flattens the tiny model)


@pytest.mark.model   # slow (thousands of positions in Python), grouped with the model tests
@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (8, 2), (8, 1)])
def test_tiny_long_context_two_pass_attention(tmp_path, heads, kv_heads) -> None:
    """Past MLX's switch to the two-pass decode attention (1024 keys on '...d' devices, 4096 with GQA elsewhere) the
    C++ dispatch rule keeps every hidden state bit-identical (milestone-3 deviation 2)."""
    from mlx_lm.models.cache import make_prompt_cache
    from suc.runtime.suc1_model import step_hidden
    start = 1024 if mx.device_info().get("architecture", "").endswith("d") else 4096
    n = start + 24
    arch = dict(ARCH, num_attention_heads=heads, num_key_value_heads=kv_heads, num_hidden_layers=1,
                max_position_embeddings=8192)
    f = _tiny_file(tmp_path / "long.suc1", True, arch=arch)
    toks = [int(x) for x in np.random.default_rng(1).integers(0, 512, n)]
    r = subprocess.run([str(GENERATE), str(f), "--tokens", ",".join(map(str, toks)), "--steps", "1", "--prefill", "token",
                        "--hidden-out", str(tmp_path / "h.npy"), "--quiet", "--max-context", str(n + 4)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ch = np.load(tmp_path / "h.npy")[:, -1]
    model, _, _ = load_suc1_model(f, group=False)
    cache = make_prompt_cache(model)
    bad = []
    for p, tok in enumerate(toks):
        _, hs = step_hidden(model, tok, cache)
        if p >= start - 8 and not np.array_equal(np.array(hs[-1].astype(mx.float32)), ch[p]):
            bad.append(p)
    assert not bad, f"positions {bad[:5]} differ (of {len(bad)})"
