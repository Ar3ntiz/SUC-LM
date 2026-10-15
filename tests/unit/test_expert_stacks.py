"""Expert stacks (suc.quant.stacks): shared codebooks per MoE (layer, projection), stack allocation units, SUC1 dedup
and SucSwitchLinear's shared layout, on random tensors and a tiny random Qwen3-MoE."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml
from mlx.utils import tree_flatten

import suc.allocation.run as arun
from suc.allocation.greedy import build_candidates, greedy_allocate, uniform_plan
from suc.analysis.sensitivity import get_weight, set_weight
from suc.config import CodecConfig, allocate_config_from_dict, load_allocate_config, quantize_config_from_dict
from suc.evaluation.reconstruct import model_size, reconstruct_model
from suc.io.suc1 import SucModel, write_suc1
from suc.kernels.suc_switch import SucSwitchLinear
from suc.models.adapters import get_adapter
from suc.quant import stacks
from suc.quant.codecs import make_codec
from suc.quant.residual import ResidualCodec
from suc.runtime.compressed_model import convert_to_compressed
from test_reconstruct import _moe, _save_store
from tiny_models import tiny_qwen3_moe

VQ = {"kind": "vq", "k": 16, "block_size": 4, "rotate": True, "kmeans_iters": 5}
ADD2 = {"kind": "additive", "k": 16, "block_size": 4, "num_codebooks": 2, "rotate": True, "kmeans_iters": 5}


def _w(out=32, inn=64, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.normal(size=(out, inn)).astype(np.float32)) / np.sqrt(inn)


def _same_arrays(a, b):
    return sorted(a.arrays) == sorted(b.arrays) and all(
        np.array_equal(np.array(a.arrays[k]), np.array(b.arrays[k])) for k in a.arrays)


def _shared(cfg, residual=None):
    base = make_codec(cfg)
    codec = base if residual is None else ResidualCodec(base, residual, "int8")
    return stacks.set_codebook_scope(codec, "expert_stack")


def _stack_names(n_exp=4, layer=0, proj="up_proj"):
    return [f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight" for e in range(n_exp)]


# ------------------------------------------------------------------------------------------- codec API

@pytest.mark.parametrize("fit_blocks", [100000, 300])
def test_vq_fit_on_own_sample_reproduces_dense_encode(fit_blocks):
    codec = make_codec({**VQ, "fit_blocks": fit_blocks})
    w = _w()
    ref = codec.encode(w, seed=7)
    assert "codebook_shared" not in ref.params
    x, h = codec.sample_blocks(w, seed=7, n=codec.fit_blocks)
    e = codec.encode(w, seed=7, codebooks=codec.fit(x, h, seed=7))
    assert _same_arrays(ref, e)
    assert e.params["codebook_shared"] and e.params["codebook_owner"]
    assert e.bits() == ref.bits()


def test_additive_fit_on_all_blocks_reproduces_dense_encode():
    codec = make_codec(ADD2)
    w = _w(seed=1)
    ref = codec.encode(w, seed=3)
    x, h = codec.sample_blocks(w, seed=3, n=10 ** 9)
    assert x.shape == (32 * 64 // 4, 4) and h is None
    e = codec.encode(w, seed=3, codebooks=codec.fit(x, seed=3))
    assert _same_arrays(ref, e)


def test_fixed_codebook_refinement_never_increases_error():
    w = _w(seed=2)
    plain, refined = make_codec(ADD2), make_codec({**ADD2, "refine_iters": 2})
    cb = plain.fit(plain.sample_blocks(w, seed=1, n=10 ** 9)[0], seed=1)
    err = lambda c: float(mx.sum((c.decode(c.encode(w, seed=1, codebooks=cb)) - w) ** 2))
    assert err(refined) <= err(plain) * (1 + 1e-6)


def test_non_owner_counts_no_codebook_bits_and_shape_is_checked():
    codec = make_codec(VQ)
    w = _w()
    cb = codec.fit(codec.sample_blocks(w, seed=0, n=10 ** 9)[0], seed=0)
    own, ref = codec.encode(w, seed=0, codebooks=cb), codec.encode(w, seed=0, codebooks=cb, codebook_owner=False)
    assert own.bits()["codebooks"] == 16 * 4 * 16 and ref.bits()["codebooks"] == 0
    assert own.bits()["total"] - ref.bits()["total"] == 16 * 4 * 16
    with pytest.raises(ValueError, match="shape"):
        codec.encode(w, seed=0, codebooks=mx.zeros((2, 16, 4)))
    res = ResidualCodec(codec, 0.02, "int8")
    r = res.encode(w, seed=0, codebooks=cb, codebook_owner=False)
    assert r.bits()["codebooks"] == 0 and r.params["base"]["codebook_shared"]
    with pytest.raises(TypeError):
        ResidualCodec(make_codec({"kind": "affine", "bits": 4}), 0.02).encode(w, seed=0, codebooks=cb)


def test_scope_tagging():
    assert stacks.codebook_scope(make_codec(VQ)) == "tensor"
    assert stacks.shares_codebooks(_shared(ADD2, 0.01))
    with pytest.raises(ValueError, match="vq / additive"):
        stacks.set_codebook_scope(make_codec({"kind": "affine", "bits": 4}), "expert_stack")


# ------------------------------------------------------------------------------------------- stacks

def test_units_seeds_and_plan_expansion():
    names = ["model.layers.0.self_attn.q_proj.weight"] + [
        f"model.layers.0.mlp.experts.{e}.{p}.weight" for e in (1, 0) for p in ("gate_proj", "up_proj")]
    u = stacks.units(names)
    assert list(u) == ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                       "model.layers.0.mlp.switch_mlp.up_proj.weight"]
    assert u["model.layers.0.mlp.switch_mlp.up_proj.weight"] == [
        "model.layers.0.mlp.experts.0.up_proj.weight", "model.layers.0.mlp.experts.1.up_proj.weight"]
    plan = stacks.expand_plan({k: i for i, k in enumerate(u)}, u)
    assert plan == {n: (0 if "q_proj" in n else 1 if "gate" in n else 2) for n in names}
    g, up, dn = (stacks.stack_seeds(f"model.layers.3.mlp.switch_mlp.{p}.weight", 5)
                 for p in ("gate_proj", "up_proj", "down_proj"))
    assert g["seed_in"] == up["seed_in"] != dn["seed_in"]
    assert len({g["seed_out"], up["seed_out"], dn["seed_out"]}) == 3
    assert stacks.stack_seeds("model.layers.4.mlp.switch_mlp.up_proj.weight", 5)["seed_in"] != up["seed_in"]


@pytest.mark.parametrize("cfg,residual", [(VQ, None), (ADD2, 0.02)])
def test_encode_stack_shares_codebooks_and_rotation(cfg, residual):
    names = _stack_names()
    ws = {n: _w(seed=i) for i, n in enumerate(names)}
    codec = _shared(cfg, residual)
    imp = {n: mx.ones(w.shape) for n, w in ws.items()} if residual else None
    out = list(stacks.encode_stack(codec, ws.__getitem__, names, seed=0, stack="model.layers.0.mlp.switch_mlp.up_proj"
                                   ".weight", importance=imp))
    assert [n for n, _, _ in out] == names
    bases = [e.base() if residual else e for _, _, e in out]
    cb0 = np.array(bases[0].arrays["codebooks"])
    assert all(np.array_equal(np.array(b.arrays["codebooks"]), cb0) for b in bases)
    assert len({(b.params["seed_in"], b.params["seed_out"]) for b in bases}) == 1
    assert [b.params["codebook_owner"] for b in bases] == [True, False, False, False]
    assert sum(e.bits()["codebooks"] for _, _, e in out) == cb0.size * 16
    per_expert = make_codec(cfg)
    for n, w, e in out:   # one pooled codebook costs little against a per-expert fit on i.i.d. experts
        shared_err = float(mx.sum((codec.decode(e) - w) ** 2))
        own = per_expert.encode(w, seed=1)
        own_err = float(mx.sum((per_expert.decode(own) - w) ** 2))
        assert shared_err < 1.5 * own_err * (1.0 if residual is None else 1.5)


def test_reconstruct_expert_stacks_tensor_scope_is_unchanged(tmp_path):
    model, store, specs = _moe(tmp_path)
    codec = make_codec(VQ)
    a = reconstruct_model(model, store, specs, lambda s: codec, seed=0, keep_encodings=True)
    (tmp_path / "b").mkdir()
    model2, store2, specs2 = _moe(tmp_path / "b")
    b = reconstruct_model(model2, store2, specs2, lambda s: codec, seed=0, keep_encodings=True, expert_stacks=True)
    assert set(a["encodings"]) == set(b["encodings"])
    assert all(_same_arrays(a["encodings"][n], b["encodings"][n]) for n in a["encodings"])
    pa, pb = dict(tree_flatten(model.parameters())), dict(tree_flatten(model2.parameters()))
    assert all(np.array_equal(np.array(pa[k].astype(mx.float32)), np.array(pb[k].astype(mx.float32))) for k in pa)


def test_reconstruct_expert_stacks_shared_scope(tmp_path):
    model, store, specs = _moe(tmp_path)
    codec = _shared(VQ)
    recon = reconstruct_model(model, store, specs, lambda s: codec, seed=0, keep_encodings=True, expert_stacks=True)
    encs = recon["encodings"]
    experts = [s for s in specs if s.expert is not None]
    assert set(encs) == {s.name for s in specs}
    for i in range(2):
        for p in ("gate_proj", "up_proj", "down_proj"):
            names = _stack_names(4, i, p)
            cbs = [np.array(encs[n].arrays["codebooks"]) for n in names]
            assert all(np.array_equal(c, cbs[0]) for c in cbs)
            # the model holds each expert's decode in its stacked weight
            stacked = get_weight(model, f"model.layers.{i}.mlp.switch_mlp.{p}.weight")
            for e, n in enumerate(names):
                assert np.allclose(np.array(stacked[e].astype(mx.float32)),
                                   np.array(codec.decode(encs[n]).astype(mx.bfloat16).astype(mx.float32)))
    size = model_size(store, recon)
    n_attn = len(specs) - len(experts)
    assert size["breakdown"]["codebooks"] == (n_attn + 6) * 16 * 4 * 16
    # SUC1 stores each stack codebook once (content dedup) and the file decodes the same tensors
    info = write_suc1(tmp_path / "m.suc1", architecture={"model_type": "qwen3_moe"}, tokenizer_id="t", tensors=encs)
    assert info["codebooks"] == n_attn + 6
    sm = SucModel.load(tmp_path / "m.suc1")
    n = experts[5].name
    assert _same_arrays(sm.encoded(n), encs[n]) and sm.encoded(n).params == encs[n].params
    assert float(mx.max(mx.abs(sm.decode_tensor(n) - codec.decode(encs[n])))) < 1e-5


def test_reconstruct_rejects_mixed_codecs_in_a_stack(tmp_path):
    model, store, specs = _moe(tmp_path)
    a, b = make_codec(VQ), make_codec({"kind": "affine", "bits": 4, "group_size": 32})
    with pytest.raises(ValueError, match="share one codec"):
        reconstruct_model(model, store, specs, lambda s: b if s.expert == 1 else a, seed=0, expert_stacks=True)


def test_build_candidates_with_stack_units_matches_reconstruction(tmp_path):
    model, store, specs = _moe(tmp_path)
    names = [s.name for s in specs]
    umap = stacks.units(names)
    codecs = {"vq_xs": _shared(VQ), "affine": make_codec({"kind": "affine", "bits": 4, "group_size": 32})}
    loss = lambda n, w, w_hat: float(mx.sum((w_hat - w.astype(mx.float32)) ** 2))
    cands = build_candidates(store.load, list(umap), codecs, loss, seed=0, stacks=umap)
    assert list(cands) == list(umap) and len(umap) == len(specs) - 2 * 4 * 3 + 6
    recon = reconstruct_model(model, store, specs, lambda s: codecs["vq_xs"], seed=0, expert_stacks=True)
    bits = {t["name"]: t["bits"]["total"] for t in recon["tensors"]}
    for u, members in umap.items():
        assert cands[u][0].bits == sum(bits[n] for n in members)
    plan = stacks.expand_plan({u: c.codec for u, c in greedy_allocate(cands, 10 ** 12).items()}, umap)
    assert set(plan) == set(names)
    for u, members in umap.items():
        assert len({plan[n] for n in members}) == 1
    assert set(uniform_plan(cands, "affine")) == set(umap)


# ------------------------------------------------------------------------------------------- runtime

@pytest.mark.parametrize("cfg,residual", [(VQ, None), (ADD2, 0.03)])
@pytest.mark.parametrize("path", ["gather", "tiled", "grouped"])
def test_switch_linear_shared_layout_matches_reference(cfg, residual, path):
    names = _stack_names(6)
    ws = {n: _w(32, 64, seed=i) for i, n in enumerate(names)}
    codec = _shared(cfg, residual)
    encs = [e for _, _, e in stacks.encode_stack(codec, ws.__getitem__, names, seed=0,
                                                 stack="model.layers.0.mlp.switch_mlp.up_proj.weight",
                                                 importance={n: mx.ones((32, 64)) for n in names})]
    m = SucSwitchLinear.from_encodings(encs)
    assert m.shared_codebooks and m.codebooks.shape[0] == 1 and m.num_experts == 6
    assert m.signs_in.shape == (1, 64) and m.signs_out.shape == (1, 32)
    n = 40 if path == "tiled" else 8
    x = mx.random.normal((n, 64), key=mx.random.key(n))
    eidx = mx.array(np.random.default_rng(n).integers(0, 6, size=n).astype(np.uint32))
    ref = mx.stack([x[i] @ codec.decode(encs[int(k)]).T for i, k in enumerate(np.array(eidx).tolist())])
    y = m.rows(x, eidx, path=path)
    assert float(mx.linalg.norm(y - ref) / mx.linalg.norm(ref)) < 1e-5


def test_compressed_moe_with_shared_stacks_matches_reconstructed():
    ref, comp = tiny_qwen3_moe(seed=0), tiny_qwen3_moe(seed=0)
    codec = _shared(ADD2, 0.02)
    weights = dict(tree_flatten(comp.parameters()))
    enc = {}
    for i in range(2):
        for p in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{i}.mlp.switch_mlp.{p}.weight"
            names = _stack_names(8, i, p)
            load = lambda n, key=key: weights[key][int(n.split(".experts.")[1].split(".")[0])]
            imp = {n: mx.ones(weights[key].shape[1:]) for n in names}
            enc.update({n: e for n, _, e in stacks.encode_stack(codec, load, names, seed=0, stack=key,
                                                                importance=imp)})
    for n, e in enc.items():
        set_weight(ref, n, codec.decode(e))
    mx.eval(ref.parameters())
    report = convert_to_compressed(comp, enc)
    rows = [r for r in report["converted"] if r["kind"] == "SucSwitchLinear"]
    assert len(rows) == 6 and all(comp.model.layers[0].mlp.switch_mlp[p].shared_codebooks
                                  for p in ("gate_proj", "up_proj", "down_proj"))
    toks = mx.array(np.random.default_rng(0).integers(0, 128, size=(2, 12)).astype(np.int32))
    lr, lc = ref(toks), comp(toks)
    assert float(mx.linalg.norm(lc - lr) / mx.linalg.norm(lr)) < 1e-5


# ------------------------------------------------------------------------------------------- configs / runner

def test_config_scope_and_unit_validation():
    c = CodecConfig(kind="additive", num_codebooks=3, residual_fraction=0.005, codebook_scope="expert_stack")
    assert c.name == "additive_m3_k256_g8_rot_xs+res0.005_activation" and "codebook_scope" not in c.codec_dict()
    assert CodecConfig(kind="vq", block_size=4).name == "vq_k256_g4_rot"
    with pytest.raises(ValueError, match="vq / additive"):
        CodecConfig(kind="affine", bits=4, codebook_scope="expert_stack")
    base = {"name": "x", "model": {"id": "m"}, "eval": {}, "budgets_bpp": [3.0],
            "candidates": [{"kind": "vq", "codebook_scope": "expert_stack"}, {"kind": "bf16"}]}
    with pytest.raises(ValueError, match="allocation_unit"):
        allocate_config_from_dict(base)
    assert allocate_config_from_dict({**base, "allocation_unit": "expert_stack"}).allocation_unit == "expert_stack"
    q = {"name": "q", "model": {"id": "m"}, "eval": {}, "finetune": {},
         "codecs": [{"kind": "vq", "codebook_scope": "expert_stack"}]}
    with pytest.raises(ValueError, match="finetune"):
        quantize_config_from_dict(q)


def _moe_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "moe"
    d.mkdir()
    model = tiny_qwen3_moe(seed=0)
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if ".switch_mlp." in k:
            for e in range(v.shape[0]):
                weights[k.replace(".switch_mlp.", f".experts.{e}.")] = v[e]
        else:
            weights[k] = v
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    a = model.args
    cfg = {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": a.hidden_size,
           "num_hidden_layers": a.num_hidden_layers, "intermediate_size": a.intermediate_size,
           "num_attention_heads": a.num_attention_heads, "num_experts": a.num_experts,
           "num_experts_per_tok": a.num_experts_per_tok, "decoder_sparse_step": 1, "mlp_only_layers": [],
           "moe_intermediate_size": a.moe_intermediate_size, "rms_norm_eps": a.rms_norm_eps, "vocab_size": a.vocab_size,
           "num_key_value_heads": a.num_key_value_heads, "head_dim": a.head_dim, "rope_theta": a.rope_theta,
           "tie_word_embeddings": True, "max_position_embeddings": 512, "norm_topk_prob": True}
    (d / "config.json").write_text(json.dumps(cfg))
    get_adapter(cfg).validate(weights)
    return d


def test_run_allocate_with_stack_units(tmp_path, monkeypatch):
    mx.random.seed(0)
    tokens = mx.random.randint(0, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(arun, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(arun, "tokenize", lambda tok, text: tokens)
    monkeypatch.setattr(arun, "ARTIFACT_DIR", str(tmp_path / "artifacts"))
    art = tmp_path / "tiny_phase2_data.safetensors"
    mx.save_safetensors(str(art), {"validation": mx.random.randint(0, 128, (8, 64), dtype=mx.int32),
                                   "calibration": mx.random.randint(0, 128, (16, 64), dtype=mx.int32)})
    cfg = {"name": "a_moe", "output_dir": str(tmp_path / "results"), "allocation_unit": "expert_stack",
           "model": {"id": "tiny/qwen3_moe", "local_path": str(_moe_model_dir(tmp_path))},
           "eval": {"seq_len": 128, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2, "calib_artifact": str(art)},
           "candidates": [{"kind": "vq", "k": 16, "block_size": 4, "kmeans_iters": 5, "codebook_scope": "expert_stack"},
                          {"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 8}],
           "budgets_bpp": [6.0], "uniform": ["vq_k16_g4_rot_xs"], "fisher_sequences": 8, "fisher_micro_batch": 2}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_allocate_config(path).allocation_unit == "expert_stack"
    r = json.loads(arun.run_allocate(path).read_text())
    assert r["allocation_units"] == 2 * 4 + 2 * 3 and r["allocated_tensors"] == 2 * 4 + 2 * 8 * 3
    for p in r["plans"]:
        assert len(p["rows"]) == r["allocation_units"]
        assert p["allocated_bpw"] <= 6.0 + 1e-9 or p["plan"].startswith("uniform")
    assert "allocation units" in Path(r["report"]).read_text()


def test_compress_greedy_with_stack_units(tmp_path, monkeypatch):
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs
    from test_compress import _base_cfg, _write

    mx.random.seed(0)
    tokens = mx.random.randint(0, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(comp, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(comp, "tokenize", lambda tok, text: tokens)
    docs, ood = _docs()
    cfg = _base_cfg(tmp_path, _moe_model_dir(tmp_path), finetune=None, plan={
        "mode": "greedy", "budget_bpw": 2.5, "allocation_unit": "expert_stack",
        "candidates": [{"kind": "affine", "bits": 4},
                       {"kind": "vq", "k": 16, "block_size": 4, "kmeans_iters": 3, "residual_fraction": 0.01,
                        "codebook_scope": "expert_stack"}],
        "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_sequences": 8, "fisher_micro_batch": 2})
    r = json.loads(comp.run_compress(_write(tmp_path, cfg), tokenizer=FakeTokenizer(), docs=docs,
                                     ood_docs=ood).read_text())
    plan = json.loads((tmp_path / "run" / "plan" / "plan.json").read_text())
    assert plan["summary"]["allocation_units"] == 2 * 4 + 2 * 3 + 0
    by_stack: dict = {}
    for n, c in plan["tensors"].items():
        if (st := stacks.stack_of(n)) is not None:
            by_stack.setdefault(st[0], set()).add(c)
    assert len(by_stack) == 6 and all(len(v) == 1 for v in by_stack.values())
    sm = SucModel.load(r["container"]["path"], verify=True)
    xs = [k for k, v in by_stack.items() if v == {"vq_k16_g4_rot_xs+res0.01_activation"}]
    assert xs and r["plan"]["allocated_bpw"] <= 2.5 + 1e-9
    for key in xs:   # a shared stack is stored with one codebook for all experts
        prefix = key.replace(".switch_mlp.", ".experts.{}.")
        assert len({tuple(sm.descriptor(prefix.format(e))["codebook_ids"]) for e in range(8)}) == 1
