"""Quality-track evaluator: codec reconstruction of a whole model, size accounting, PPL/KL."""
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_moe

from suc.analysis.sensitivity import get_weight, logit_divergence
from suc.evaluation.perplexity import perplexity
from suc.evaluation.reconstruct import (evaluate_quality, load_validation_tokens, model_size,
                                        reconstruct_model, reference_logits)
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from tiny_models import tiny_qwen3


# ---------------------------------------------------------------- stub codecs (interface of suc.quant.codecs)
@dataclass
class Encoded:
    codec: str
    shape: tuple[int, int]
    params: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)

    def bits(self) -> dict:
        n = self.shape[0] * self.shape[1]
        b = self.params["bits"]
        d = {"indices": n * b, "codebooks": 0 if b == 16 else 64, "scales": 0 if b == 16 else self.shape[0] * 16,
             "residual": 0, "metadata": 0 if b == 16 else 32}
        d["total"] = sum(d.values())
        return d

    def bpw(self) -> float:
        return self.bits()["total"] / (self.shape[0] * self.shape[1])


class IdentityCodec:
    name = "identity"

    def encode(self, w, *, seed, col_weights=None):
        return Encoded(self.name, tuple(w.shape), {"bits": 16}, {"w": w})

    def decode(self, e):
        return e.arrays["w"].astype(mx.float32)


class NoiseCodec:
    name = "noise"

    def __init__(self, hook=None):
        self.hook, self.calls = hook, []

    def encode(self, w, *, seed, col_weights=None):
        self.calls.append((seed, col_weights))
        if self.hook is not None:
            self.hook(w)
        x = w.astype(mx.float32)
        n = mx.random.normal(x.shape, key=mx.random.key(seed)) * 0.2 * mx.sqrt(mx.mean(x * x))
        return Encoded(self.name, tuple(w.shape), {"bits": 4}, {"w": x + n})

    def decode(self, e):
        return e.arrays["w"]


# ---------------------------------------------------------------- fixtures
def _save_store(tmp_path: Path, model, extra: dict | None = None, name: str = "ckpt") -> SafetensorsStore:
    d = tmp_path / name
    d.mkdir()
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if ".switch_mlp." in k:  # un-stack to the HF checkpoint namespace
            for e in range(v.shape[0]):
                weights[k.replace(".switch_mlp.", f".experts.{e}.")] = v[e]
        else:
            weights[k] = v
    weights.update(extra or {})
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    return SafetensorsStore.from_dir(d)


def _dense(tmp_path, tie=True):
    model = tiny_qwen3(seed=1, tie=tie)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    store = _save_store(tmp_path, model)
    cfg = {"model_type": "qwen3", "num_hidden_layers": 2, "tie_word_embeddings": tie}
    specs = [s for s in get_adapter(cfg).linears(store.names()) if s.eligible]
    return model, store, specs


def _moe(tmp_path):
    mx.random.seed(3)
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[], moe_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=128,
        num_key_value_heads=2, head_dim=16, rope_theta=10000.0, tie_word_embeddings=True,
        max_position_embeddings=512, norm_topk_prob=True)
    model = qwen3_moe.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    store = _save_store(tmp_path, model)
    cfg = {"model_type": "qwen3_moe", "num_hidden_layers": 2, "num_experts": 4, "tie_word_embeddings": True}
    specs = [s for s in get_adapter(cfg).linears(store.names()) if s.eligible]
    return model, store, specs


def _params(model) -> dict:
    return {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}


def _tokens(n=4, t=33, seed=0):
    return mx.random.randint(0, 128, (n, t), key=mx.random.key(seed)).astype(mx.int32)


# ---------------------------------------------------------------- tests
def test_identity_codec_is_bit_identical_and_16_bpp(tmp_path):
    model, store, specs = _dense(tmp_path)
    before = _params(model)
    kl_tok = _tokens()
    ref = reference_logits(model, kl_tok, chunk=2)
    recon = reconstruct_model(model, store, specs, lambda s: IdentityCodec(), seed=0)
    after = _params(model)
    assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
    assert len(recon["tensors"]) == len(specs) == 14
    assert all(t["rel_mse"] == 0 and t["bpw"] == 16 and t["codec"] == "identity" for t in recon["tensors"])
    size = model_size(store, recon)
    assert size["bpp"] == 16 and size["encoded_bpp"] == 16
    assert size["parameter_count"] == sum(v.size for v in before.values())
    q = evaluate_quality(model, ppl_tokens=kl_tok.reshape(-1), seq_len=32, max_tokens=None,
                         kl_tokens=kl_tok, ref_logits=ref, chunk=2)
    assert q["kl"]["kl_mean"] == 0 and q["kl"]["top1_agreement"] == 1 and q["kl"]["delta_ce"] == 0


def test_noise_codec_changes_only_targeted_tensors(tmp_path):
    model, store, specs = _dense(tmp_path)
    before = _params(model)
    kl_tok = _tokens()
    ref = reference_logits(model, kl_tok)
    codec = NoiseCodec()
    targeted = {"attn_q", "mlp_down"}
    recon = reconstruct_model(model, store, specs, lambda s: codec if s.role in targeted else None, seed=7)
    after = _params(model)
    changed = {k for k in before if not np.array_equal(before[k], after[k])}
    assert changed == {s.name for s in specs if s.role in targeted}
    assert {t["name"] for t in recon["tensors"]} == changed and all(t["rel_mse"] > 0 for t in recon["tensors"])
    assert len({seed for seed, _ in codec.calls}) == len(changed)  # distinct, stable per-tensor seeds
    q = evaluate_quality(model, ppl_tokens=kl_tok.reshape(-1), seq_len=32, max_tokens=None,
                         kl_tokens=kl_tok, ref_logits=ref)
    assert q["kl"]["kl_mean"] > 0
    size = model_size(store, recon)
    n_enc = sum(before[k].size for k in changed)
    assert size["encoded_params"] == recon["encoded_params"] == n_enc
    assert size["other_params"] == size["parameter_count"] - n_enc
    assert size["breakdown"]["unencoded"] == 16 * size["other_params"]
    assert sum(size["breakdown"].values()) == size["bits_total"] == recon["encoded_bits"] + 16 * size["other_params"]
    assert size["bpp"] == size["bits_total"] / size["parameter_count"] and size["bytes_total"] * 8 >= size["bits_total"]
    assert size["encoded_bpp"] == pytest.approx(recon["encoded_bits"] / n_enc)
    assert 4 < size["encoded_bpp"] < 4.5 and size["bpp"] < 16


def test_same_seed_is_deterministic_and_col_weights_are_forwarded(tmp_path):
    model, store, specs = _dense(tmp_path)
    cw = {s.name[:-7]: mx.ones((s_in,)) for s in specs for s_in in [store.info(s.name).shape[1]]}
    codec = NoiseCodec()
    r1 = reconstruct_model(model, store, specs, lambda s: codec, seed=5, col_weights=cw)
    assert all(c is cw[s.name[:-7]] for (_, c), s in zip(codec.calls, specs))
    w1 = _params(model)
    (tmp_path / "b").mkdir()
    model2, _, _ = _dense(tmp_path / "b")
    reconstruct_model(model2, store, specs, lambda s: NoiseCodec(), seed=5)
    w2 = _params(model2)
    assert all(np.array_equal(w1[k], w2[k]) for k in w1)
    assert [t["rel_mse"] for t in r1["tensors"]] and r1["seconds"] >= 0


def test_tied_embeddings_counted_once(tmp_path):
    model = tiny_qwen3(seed=1, tie=True)
    model.set_dtype(mx.bfloat16)
    embed = model.model.embed_tokens.weight
    store = _save_store(tmp_path, model, {"lm_head.weight": embed})  # HF tied checkpoints may carry both
    n = sum(v.size for _, v in tree_flatten(model.parameters()))
    size = model_size(store, {"tensors": [], "encoded_params": 0, "encoded_bits": 0, "seconds": 0.0})
    assert size["parameter_count"] == n and size["bpp"] == 16
    untied = tiny_qwen3(seed=1, tie=False)
    untied.set_dtype(mx.bfloat16)
    store2 = _save_store(tmp_path, untied, name="untied")
    s2 = model_size(store2, {"tensors": [], "encoded_params": 0, "encoded_bits": 0, "seconds": 0.0})
    assert s2["parameter_count"] == n + embed.size


def test_moe_experts_reconstructed_one_row_at_a_time(tmp_path):
    model, store, specs = _moe(tmp_path)
    before = _params(model)
    experts = [s for s in specs if s.expert is not None]
    assert len(experts) == 2 * 4 * 3
    calls = [0]

    def hook(_w):
        # at the i-th expert encode, expert tensors i..end must still hold their original weights
        for s in experts[calls[0]:]:
            orig = np.array(store.load(s.name).astype(mx.float32))
            assert np.array_equal(np.array(get_weight(model, s.name).astype(mx.float32)), orig), s.name
        calls[0] += 1

    codec = NoiseCodec(hook)
    codec_for = lambda s: codec if s.expert is not None else None
    recon = reconstruct_model(model, store, specs, codec_for, seed=0)
    after = _params(model)
    changed = {k for k in before if not np.array_equal(before[k], after[k])}
    assert changed == {f"model.layers.{i}.mlp.switch_mlp.{p}.weight" for i in range(2)
                       for p in ("gate_proj", "up_proj", "down_proj")}
    for s in experts:  # every expert row changed
        assert not np.array_equal(np.array(get_weight(model, s.name).astype(mx.float32)),
                                  np.array(store.load(s.name).astype(mx.float32)))
    assert [(t["layer"], t["expert"], t["role"]) for t in recon["tensors"]] == \
        [(s.layer, s.expert, s.role) for s in experts]
    size = model_size(store, recon)
    assert size["encoded_params"] == sum(store.load(s.name).size for s in experts)
    assert sum(size["breakdown"].values()) == size["bits_total"]
    assert size["parameter_count"] == sum(v.size for v in before.values())


def test_perplexity_matches_and_chunking_is_consistent(tmp_path):
    model, store, specs = _dense(tmp_path)
    kl_tok = _tokens(n=4, t=17, seed=2)
    ppl_tok = _tokens(n=1, t=200, seed=4).reshape(-1)
    ref1, ref4 = reference_logits(model, kl_tok, chunk=1), reference_logits(model, kl_tok, chunk=4)
    assert len(ref1) == 4 and len(ref4) == 1 and ref1[0].dtype == mx.bfloat16
    reconstruct_model(model, store, specs, lambda s: NoiseCodec(), seed=1)
    q1 = evaluate_quality(model, ppl_tokens=ppl_tok, seq_len=32, max_tokens=160, kl_tokens=kl_tok,
                          ref_logits=ref1, chunk=1)
    q4 = evaluate_quality(model, ppl_tokens=ppl_tok, seq_len=32, max_tokens=160, kl_tokens=kl_tok,
                          ref_logits=ref4, chunk=4)
    assert q1["perplexity"] == perplexity(model, ppl_tok, 32, 160)
    # batch-1 vs batch-4 bf16 forwards differ only by kernel round-off
    for k in ("kl_mean", "top1_agreement", "delta_ce", "ce_ref", "ce_test"):
        assert q1["kl"][k] == pytest.approx(q4["kl"][k], rel=1e-3, abs=1e-4)
    test = model(kl_tok).astype(mx.bfloat16)
    direct = logit_divergence(ref4[0][:, :-1], test[:, :-1], kl_tok[:, 1:])
    assert q4["kl"]["kl_mean"] == pytest.approx(direct["kl_mean"], rel=1e-6)
    assert q4["kl"]["tokens"] == 4 * 16
    q = evaluate_quality(model, ppl_tokens=ppl_tok, seq_len=32, max_tokens=None)
    assert q["kl"] is None


def test_load_validation_tokens(tmp_path):
    p = tmp_path / "x_data.safetensors"
    val = mx.arange(40, dtype=mx.int32).reshape(5, 8)
    mx.save_safetensors(str(p), {"calibration": mx.zeros((3, 8), dtype=mx.int32), "validation": val})
    got = load_validation_tokens(p, 3)
    assert got.shape == (3, 8) and np.array_equal(np.array(got), np.array(val[:3]))
    with pytest.raises(ValueError):
        load_validation_tokens(p, 6)


class ImportanceCodec(NoiseCodec):
    name = "noise_imp"

    def encode(self, w, *, seed, col_weights=None, importance=None):
        self.imp_calls = getattr(self, "imp_calls", []) + [(seed, importance)]
        return super().encode(w, seed=seed, col_weights=col_weights)


def test_importance_forwarded_only_to_codecs_that_accept_it(tmp_path):
    model, store, specs = _dense(tmp_path)
    imp = {s.name: mx.ones(store.info(s.name).shape) for s in specs[:3]}
    with_imp, plain = ImportanceCodec(), NoiseCodec()
    codec_for = lambda s: with_imp if s.role == "mlp_down" or s.name in imp else plain
    recon = reconstruct_model(model, store, specs, codec_for, seed=0, importance=imp)
    assert len(recon["tensors"]) == len(specs)
    got = {s.name: c for s, (_, c) in zip([s for s in specs if codec_for(s) is with_imp], with_imp.imp_calls)}
    assert got and all((c is imp[n]) if n in imp else c is None for n, c in got.items())
    assert len(plain.calls) == len(specs) - len(got)  # plain codec never saw an importance kwarg
    # without the mapping, accepting codecs get None
    (tmp_path / "b").mkdir()
    model2, store2, specs2 = _dense(tmp_path / "b")
    c2 = ImportanceCodec()
    reconstruct_model(model2, store2, specs2, lambda s: c2, seed=0)
    assert all(c is None for _, c in c2.imp_calls)


# ---------------------------------------------------------------- rotation groups (shared input rotation)
def test_rotation_group_keys_and_seeds():
    from suc.evaluation.reconstruct import rotation_group, rotation_seeds
    L = "model.layers.3"
    assert {rotation_group(f"{L}.self_attn.{p}_proj.weight") for p in "qkv"} == {f"{L}.self_attn.qkv"}
    assert {rotation_group(f"{L}.mlp.{p}_proj.weight") for p in ("gate", "up")} == {f"{L}.mlp.gate_up"}
    for n in (f"{L}.self_attn.o_proj.weight", f"{L}.mlp.down_proj.weight", f"{L}.mlp.experts.2.gate_proj.weight",
              f"{L}.mlp.shared_expert.up_proj.weight", "lm_head.weight", "model.embed_tokens.weight"):
        assert rotation_group(n) is None and rotation_seeds(n, 0) == {}
    q, k = rotation_seeds(f"{L}.self_attn.q_proj.weight", 5), rotation_seeds(f"{L}.self_attn.k_proj.weight", 5)
    g = rotation_seeds(f"{L}.mlp.gate_proj.weight", 5)
    assert q == k and q != g and q != rotation_seeds("model.layers.4.self_attn.q_proj.weight", 5)
    assert q != rotation_seeds(f"{L}.self_attn.q_proj.weight", 6)


class SeedCodec(NoiseCodec):
    name = "noise_seed"

    def encode(self, w, *, seed, col_weights=None, seed_in=None, seed_out=None):
        self.seed_calls = getattr(self, "seed_calls", []) + [(seed, seed_in, seed_out)]
        return super().encode(w, seed=seed, col_weights=col_weights)


def test_rotation_groups_forward_shared_seed_in(tmp_path):
    from suc.evaluation.reconstruct import rotation_seeds
    model, store, specs = _dense(tmp_path)
    c, plain = SeedCodec(), NoiseCodec()
    codec_for = lambda s: c if "self_attn" in s.name or "mlp" in s.name else plain
    reconstruct_model(model, store, specs, codec_for, seed=11, rotation_groups=True)
    by = dict(zip([s.name for s in specs if codec_for(s) is c], c.seed_calls))
    for n, (seed, s_in, s_out) in by.items():
        assert s_out is None
        assert s_in == rotation_seeds(n, 11).get("seed_in")
    q = [v[1] for n, v in by.items() if "layers.0.self_attn" in n and n.split(".")[-2] in ("q_proj", "k_proj", "v_proj")]
    assert len(q) == 3 and len(set(q)) == 1 and q[0] is not None
    # default: no seed_in passed (per-tensor seeds as before)
    (tmp_path / "b").mkdir()
    model2, store2, specs2 = _dense(tmp_path / "b")
    c2 = SeedCodec()
    reconstruct_model(model2, store2, specs2, lambda s: c2, seed=11)
    assert all(s_in is None for _, s_in, _ in c2.seed_calls)
