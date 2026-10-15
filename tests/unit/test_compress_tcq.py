"""`suc compress` with trellis (tcq) candidates on a tiny Qwen3-MoE (hidden 128): the hessians stage (BlockLDLQ second
moments, layer-streamed, resumable), LDLQ in scoring and encode, stack-shared input rotations, row-sampled scoring with
exact bits, reuse of an earlier run's candidate scores, and the SUC1 file run by the Python runtime."""
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_moe

import suc.pipeline.compress as comp
from suc.analysis.hessians import HessianStore
from suc.io.suc1 import SucModel
from suc.models.adapters import get_adapter
from suc.quant import stacks
from test_compress import _base_cfg, _status, env  # noqa: F401  (fixture)

AFF = [{"kind": "affine", "bits": 8}, {"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 2}]
VQ_XS = {"kind": "vq", "k": 16, "block_size": 2, "rotate_out": False, "kmeans_iters": 3, "residual_fraction": 0.02,
         "residual_structure": "columns", "codebook_scope": "expert_stack"}
TCQ2 = {"kind": "tcq", "tcq_bits": 2, "tcq_code": "3inst", "tcq_V": 2, "tcq_L": 10, "tcq_T": 128, "rotate_out": False,
        "ldlq": True, "residual_fraction": 0.02, "residual_structure": "columns", "codebook_scope": "expert_stack"}
TCQ3 = {**TCQ2, "tcq_bits": 3, "tcq_code": "1mad", "tcq_V": 1}
EXPERT_ROLES = ["expert_gate", "expert_up", "expert_down"]


def _moe_dir(tmp_path: Path) -> Path:
    d = tmp_path / "moe128"
    d.mkdir()
    mx.random.seed(3)
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=128, num_hidden_layers=2, intermediate_size=256, num_attention_heads=4,
        num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[], moe_intermediate_size=128,
        rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2, head_dim=32, rope_theta=10000.0,
        tie_word_embeddings=True, max_position_embeddings=512, norm_topk_prob=True)
    model = qwen3_moe.Model(args)
    mx.eval(model.parameters())
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if ".switch_mlp." in k:
            for e in range(v.shape[0]):
                weights[k.replace(".switch_mlp.", f".experts.{e}.")] = v[e]
        else:
            weights[k] = v
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    cfg = {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": 128,
           "num_hidden_layers": 2, "intermediate_size": 256, "num_attention_heads": 4, "num_experts": 8,
           "num_experts_per_tok": 2, "decoder_sparse_step": 1, "mlp_only_layers": [], "moe_intermediate_size": 128,
           "rms_norm_eps": 1e-6, "vocab_size": 128, "num_key_value_heads": 2, "head_dim": 32, "rope_theta": 10000.0,
           "tie_word_embeddings": True, "max_position_embeddings": 512, "norm_topk_prob": True}
    (d / "config.json").write_text(json.dumps(cfg))
    get_adapter(cfg).validate(weights)
    return d


def _plan(candidates, roles, **over) -> dict:
    return {"mode": "greedy", "budget_bpw": 3.2, "allocation_unit": "expert", "candidates": candidates,
            "candidate_roles": roles, "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_sequences": 8,
            "fisher_micro_batch": 2, **over}


def _names(cands) -> list[str]:
    from suc.config import CodecConfig
    return [CodecConfig(**c).name for c in cands]


def test_tcq_candidates_hessians_reuse_and_runtime(env, monkeypatch):  # noqa: F811
    tmp_path, _, run = env
    model_dir = _moe_dir(tmp_path)
    vq, t2, t3 = _names([VQ_XS, TCQ2, TCQ3])
    a8, a4, a2 = _names(AFF)
    base_roles = {vq: EXPERT_ROLES}
    cfg_a = _base_cfg(tmp_path, model_dir, finetune=None, run_dir=str(tmp_path / "run_a"),
                      plan=_plan(AFF + [VQ_XS], base_roles))
    run(cfg_a)

    # run B: two tcq candidates added, scores of the others reused from run A (calibrate / reference copied)
    rb = tmp_path / "run_b"
    rb.mkdir()
    for st in ("calibrate", "reference"):
        shutil.copytree(tmp_path / "run_a" / st, rb / st)
    scored = []
    real = comp.build_candidates

    def spy(get, names, codecs, *a, **kw):
        scored.extend(codecs)
        return real(get, names, codecs, *a, **kw)

    monkeypatch.setattr(comp, "build_candidates", spy)
    roles_b = {**base_roles, t2: EXPERT_ROLES, t3: EXPERT_ROLES}
    cfg_b = _base_cfg(tmp_path, model_dir, finetune=None, run_dir=str(rb),
                      plan=_plan(AFF + [VQ_XS, TCQ2, TCQ3], roles_b, candidates_reuse=str(tmp_path / "run_a"),
                                 tcq_score_rows=48, ldlq_min_tokens=1))
    r = json.loads(run(cfg_b).read_text())
    st = json.loads((rb / "status.json").read_text())["stages"]
    status = {s["stage"]: s["status"] for s in st}
    assert status["calibrate"] == status["reference"] == "skipped" and status["hessians"] == "ran"
    assert set(scored) <= {t2, t3, a8, a4, a2} and {t2, t3} <= set(scored) and vq not in scored

    # reused scores are identical; tcq bits are the exact full-tensor bits although scored on 48 of 128 rows
    ca = json.loads((tmp_path / "run_a" / "candidates" / "candidates.json").read_text())
    cb = json.loads((rb / "candidates" / "candidates.json").read_text())
    n0 = "model.layers.0.mlp.experts.3.down_proj.weight"
    assert [c for c in cb[n0] if c["codec"] == vq] == [c for c in ca[n0] if c["codec"] == vq]
    table = comp.CodecTable(type("C", (), {"cfg": comp.load_compress_config(tmp_path / "cfg.yaml")})(), {}, set())
    assert next(c for c in cb[n0] if c["codec"] == t2)["bits"] == table.obj(t2, False).bits_for((128, 128))
    assert next(c for c in cb[n0] if c["codec"] == t2)["loss"] < next(c for c in cb[n0] if c["codec"] == a2)["loss"]

    # hessians: shared gate/up matrix per MoE layer, per-expert down matrices
    hs = HessianStore(sorted((rb / "hessians").glob("block_*.safetensors")))
    g0 = hs["model.layers.0.mlp.experts.0.gate_proj"]
    assert g0 is hs["model.layers.1.mlp.experts.5.up_proj"] or g0.shape == (128, 128)
    assert hs["model.layers.0.mlp.experts.2.down_proj"].shape == (128, 128)

    plan = json.loads((rb / "plan" / "plan.json").read_text())["tensors"]
    tcq_names = [n for n, c in plan.items() if c in (t2, t3)]
    assert tcq_names, "the greedy chose no tcq expert"
    sm = SucModel.load(r["container"]["path"], verify=True)
    seeds = {}
    for n in tcq_names:
        d = sm.descriptor(n)
        p = d["params"]["base"]
        assert d["representation_type"] == "tcq" and p["T"] == 128 and "ldlq" in p
        seeds.setdefault(stacks.stack_seeds(stacks.stack_of(n)[0], 7)["seed_in"], set()).add(p["seed_in"])
    assert all(len(v) == 1 and k in v for k, v in seeds.items())       # every tcq expert: its stack's seed_in
    assert np.isfinite(r["quality"]["kl"]["kl_mean"])

    # Python runtime from the file == the model with every tensor replaced by the reference decode
    from suc.models.streaming import load_lazy
    from suc.runtime.suc1_model import load_suc1_model
    from suc.evaluation.reconstruct import _flush_experts, _EXPERT
    from suc.analysis.sensitivity import set_weight, get_weight

    m_run, _, _ = load_suc1_model(r["container"]["path"], dtype=mx.float32)
    m_ref = load_lazy(model_dir)[0]
    m_ref.set_dtype(mx.float32)
    pending = {}
    for n in sm.names():
        if sm.descriptor(n)["representation_type"] == "raw":
            continue
        w = sm.decode_tensor(n)
        if (m := _EXPERT.match(n)) is not None:
            pending.setdefault(f"{m.group(1)}.switch_mlp.{m.group(3)}.weight", {})[int(m.group(2))] = w
        elif n != "lm_head.weight":
            set_weight(m_ref, n, w)
    _flush_experts(m_ref, pending)
    toks = mx.array(np.random.default_rng(0).integers(0, 128, (2, 24)).astype(np.int32))
    a, b = m_run(toks), m_ref(toks)
    rel = float(mx.linalg.norm((a - b).reshape(-1)) / mx.linalg.norm(b.reshape(-1)))
    assert rel < 2e-3, rel
    assert get_weight(m_ref, "model.layers.0.mlp.switch_mlp.down_proj.weight").shape == (8, 128, 128)


def test_hessians_stage_resumes(env, monkeypatch):  # noqa: F811
    tmp_path, _, run = env
    model_dir = _moe_dir(tmp_path)
    t2 = _names([TCQ2])[0]
    cfg = _base_cfg(tmp_path, model_dir, finetune=None,
                    plan=_plan([TCQ2] + AFF[:1], {t2: EXPERT_ROLES}, budget_bpw=4.0, ldlq_min_tokens=1,
                               fisher_sequences=4))
    import suc.analysis.hessians as H
    real, calls = H._BlockHooks.arrays, []

    def flaky(self):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return real(self)

    monkeypatch.setattr(H._BlockHooks, "arrays", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg)
    prog = json.loads((tmp_path / "run" / "hessians" / "progress.json").read_text())
    assert prog["next_block"] == 1
    monkeypatch.setattr(H._BlockHooks, "arrays", real)
    r = json.loads(run(cfg).read_text())
    assert sorted(p.name for p in (tmp_path / "run" / "hessians").glob("block_*.safetensors")) == \
        ["block_00000.safetensors", "block_00001.safetensors"]
    assert np.isfinite(r["quality"]["perplexity"]["ppl"])


def test_batched_stack_scoring_equals_per_member():
    """One base encode over the sampled rows of every expert (shared rotation + hessian) scores like per-expert
    encodes (the trellis codec is row-independent; LDLQ only couples columns)."""
    from suc.allocation.greedy import _batchable, _score_trellis, _score_trellis_stack
    from suc.quant.residual import ResidualCodec
    from suc.quant.stacks import set_codebook_scope
    from suc.quant.trellis import TrellisCodec

    rng = np.random.default_rng(0)
    names = [f"model.layers.0.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
    ws = {n: mx.array(rng.normal(size=(64, 256)).astype(np.float32)) for n in names}
    x = rng.normal(size=(2048, 256)).astype(np.float32) * (1 + 5 * (rng.random(256) < 0.05))
    h = mx.array(x.T @ x / 2048)
    base = TrellisCodec(k=2, L=10, code="3inst", V=2, T=128)
    base.ldlq = True
    codec = set_codebook_scope(ResidualCodec(base, 0.02, "int8", "magnitude", structure="columns"), "expert_stack")
    codec.ldlq = True
    hess = {n[:-7]: h for n in names}
    fis = {n: mx.abs(mx.array(rng.normal(size=(64, 256)).astype(np.float32))) for n in names}

    def loss_fn(n, w, w_hat, rows=None):
        f = fis[n] if rows is None else fis[n][mx.array(rows.astype(np.int32))]
        return 0.5 * mx.sum(f * (w_hat - w) ** 2).item()

    assert _batchable(codec, names, 7, hess)
    got = _score_trellis_stack(codec, names, ws.__getitem__, seed=7, loss_fn=loss_fn, hessians=hess, row_sample=32,
                               importance=None)
    for n in names:
        ref = _score_trellis(codec, n, ws[n], {}, seed=7, loss_fn=loss_fn, hessians=hess, row_sample=32)
        assert got[n][0] == ref[0]
        assert got[n][1] == pytest.approx(ref[1], rel=2e-3) and got[n][2] == pytest.approx(ref[2], rel=2e-3)
