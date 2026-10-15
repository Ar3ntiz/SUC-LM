"""D8 allocation runner (suc allocate): layer-streamed Fisher + scoring, resume from a candidates artifact, and
layer-streamed plan validation, all equal to the earlier whole-model path (tiny dense Qwen3 and tiny MoE)."""
import json
from pathlib import Path

import mlx.core as mx
import pytest
import yaml
from mlx_lm.utils import load_model

import suc.allocation.run as arun
from suc.allocation.greedy import Candidate, build_candidates, greedy_allocate, uniform_plan
from suc.analysis.activations import Recorder
from suc.analysis.fisher import fisher_diag, fisher_proxy
from suc.config import load_allocate_config
from suc.evaluation.reconstruct import evaluate_quality, model_size, reconstruct_model, reference_logits
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import get_adapter
from suc.quant import stacks
from suc.quant.run import select_specs
from test_calibrate_run import _tiny_model_dir
from test_expert_stacks import _moe_model_dir

RES = {"kind": "affine", "bits": 2, "residual_fraction": 0.01}           # activation-selected residual
DENSE_CANDS = [{"kind": "affine", "bits": 2}, RES, {"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 8}]
MOE_CANDS = [{"kind": "vq", "k": 16, "block_size": 4, "kmeans_iters": 5, "codebook_scope": "expert_stack",
              "residual_fraction": 0.01},
             {"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 8}]


def _stats(model_dir: Path, out: Path) -> Path:
    model, _ = load_model(model_dir)
    rec = Recorder()
    rec.instrument(model, "")
    model(mx.random.randint(0, 128, (2, 32), dtype=mx.int32))
    rec.save(out)
    return out


def _cfg(tmp_path: Path, monkeypatch, kind: str, **over) -> Path:
    mx.random.seed(0)
    tokens = mx.random.randint(0, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(arun, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(arun, "tokenize", lambda tok, text: tokens)
    monkeypatch.setattr(arun, "ARTIFACT_DIR", str(tmp_path / "artifacts"))
    art = tmp_path / "tiny_phase2_data.safetensors"
    mx.save_safetensors(str(art), {"validation": mx.random.randint(0, 128, (8, 64), dtype=mx.int32),
                                   "calibration": mx.random.randint(0, 128, (16, 64), dtype=mx.int32)})
    if kind == "head":                      # untied embedding + LM head allocated in the budget
        from test_allocate_head import HEAD, _untied_model_dir
        model_dir = _untied_model_dir(tmp_path)
        cfg = {"candidates": DENSE_CANDS, "budgets_bpp": [3.0, 5.0], "uniform": ["affine_b4_g64"],
               "head_candidates": HEAD, "uniform_head": "affine_b8_g64"}
    elif kind == "moe":
        model_dir = _moe_model_dir(tmp_path)
        cfg = {"allocation_unit": "expert_stack", "candidates": MOE_CANDS, "budgets_bpp": [5.0, 6.0],
               "uniform": ["vq_k16_g4_rot_xs+res0.01_activation", "affine_b4_g64"]}
    else:
        model_dir = _tiny_model_dir(tmp_path)
        cfg = {"candidates": DENSE_CANDS, "budgets_bpp": [3.0, 5.0], "uniform": ["affine_b4_g64"],
               "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_group_size": 3}
    stats = _stats(model_dir, tmp_path / f"{kind}_activations.safetensors")
    monkeypatch.setattr(arun, "default_activation_stats", lambda model_id: stats)
    cfg.update({"name": f"a_{kind}", "output_dir": str(tmp_path / "results"),
                "model": {"id": f"tiny/{kind}", "local_path": str(model_dir)},
                "eval": {"seq_len": 128, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2,
                         "calib_artifact": str(art)},
                "fisher_sequences": 8, "fisher_micro_batch": 2}, **over)
    path = tmp_path / f"cfg_{kind}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class _Ctx:
    """What the pre-D8 whole-model runner derived from the config (reference implementation for the tests)."""

    def __init__(self, path: Path) -> None:
        self.cfg = cfg = load_allocate_config(path)
        self.model_dir = Path(cfg.model.local_path)
        self.store = SafetensorsStore.from_dir(self.model_dir)
        adapter = get_adapter(json.loads((self.model_dir / "config.json").read_text()))
        names = self.store.names()
        self.specs = select_specs(adapter, names, None, include_embeddings=False)
        self.hspecs = arun.head_specs(adapter, names) if cfg.head_candidates is not None else []
        self.emb_specs = arun.head_specs(adapter, names) if cfg.embedding_codec is not None else []
        self.head_names = [s.name for s in self.hspecs]
        self.stack_units = cfg.allocation_unit == "expert_stack"
        self.unit_map = stacks.units([s.name for s in self.specs]) if self.stack_units else \
            {s.name: [s.name] for s in self.specs}
        self.codecs = {c.name: arun._make_codec(c) for c in cfg.candidates}
        self.head_codecs = {c.name: arun._make_codec(c) for c in cfg.head_candidates or ()}
        rec = Recorder.load(arun.default_activation_stats(cfg.model.id))
        self.importance = {s.name: mx.broadcast_to(rec.moments[s.name[:-7]].second_moment_diag()[None],
                                                   self.store.info(s.name).shape) for s in self.specs}
        corpus = mx.load(str(cfg.eval.calib_artifact))
        self.calib = corpus["calibration"][: cfg.fisher_sequences]
        self.kl_tokens = corpus["validation"][: cfg.eval.kl_sequences]
        self.ppl_tokens = arun.tokenize(None, "unused")
        self.alloc_params = sum(arun._numel(self.store.info(s.name).shape) for s in self.specs + self.hspecs)


def _old_candidates(c: _Ctx) -> dict[str, list[Candidate]]:
    """The pre-D8 scoring: whole-model fisher_diag (with fisher_group_size), one build_candidates call."""
    cfg = c.cfg
    fisher = fisher_diag(load_model(c.model_dir)[0], c.calib, [s.name for s in c.specs + c.hspecs],
                         micro_batch=cfg.fisher_micro_batch, group_size=cfg.fisher_group_size, loss=cfg.fisher_loss,
                         seed=cfg.seed)
    loss_fn = lambda n, w, w_hat: fisher_proxy(w_hat.astype(mx.float32) - w.astype(mx.float32), fisher[n])  # noqa
    cands = build_candidates(c.store.load, list(c.unit_map), c.codecs, loss_fn, seed=cfg.seed,
                             importance=c.importance, stacks=c.unit_map)
    if c.hspecs:
        cands.update(build_candidates(c.store.load, c.head_names, c.head_codecs, loss_fn, seed=cfg.seed))
    return cands


def _old_validation(c: _Ctx, cands: dict[str, list[Candidate]]) -> dict:
    """The pre-D8 validation loop: full model, whole-model reconstruct_model + evaluate_quality per plan."""
    cfg, ev = c.cfg, c.cfg.eval
    plans = {f"greedy_{b:.1f}": greedy_allocate(cands, int(b * c.alloc_params)) for b in cfg.budgets_bpp}
    block = {n: cs for n, cs in cands.items() if n not in set(c.head_names)}
    for u in cfg.uniform:
        plans[f"uniform_{u}"] = uniform_plan(block, u)
        if c.hspecs:
            plans[f"uniform_{u}"].update(uniform_plan({n: cands[n] for n in c.head_names}, cfg.uniform_head))
    model = load_model(c.model_dir)[0]
    reference = evaluate_quality(model, ppl_tokens=c.ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens)
    ref_logits = reference_logits(model, c.kl_tokens, chunk=ev.chunk)
    emb_codec = arun.make_codec(cfg.embedding_codec.codec_dict()) if cfg.embedding_codec is not None else None
    out = {}
    for pname, plan in plans.items():
        model = load_model(c.model_dir)[0]
        chosen = stacks.expand_plan({n: x.codec for n, x in plan.items()}, c.unit_map)
        recon = reconstruct_model(model, c.store, c.specs + (c.hspecs or c.emb_specs),
                                  lambda s: (c.head_codecs if s.name in c.head_names else c.codecs)[chosen[s.name]]
                                  if s.name in chosen else emb_codec, seed=cfg.seed, importance=c.importance,
                                  expert_stacks=c.stack_units)
        quality = evaluate_quality(model, ppl_tokens=c.ppl_tokens, seq_len=ev.seq_len, max_tokens=ev.max_tokens,
                                   kl_tokens=c.kl_tokens, ref_logits=ref_logits, chunk=ev.chunk)
        summ = arun.plan_summary(plan)
        out[pname] = {"quality": quality, "size": model_size(c.store, recon), "rows": summ["rows"],
                      "predicted_loss": summ["loss_total"], "allocated_bpw": summ["bits_total"] / c.alloc_params,
                      "by_codec": summ["by_codec"]}
    return {"reference": reference, "plans": out}


def _assert_same_validation(r: dict, old: dict) -> None:
    assert r["reference"]["perplexity"] == old["reference"]["perplexity"]
    assert r["reference"]["kl"] is None
    got = {p["plan"]: p for p in r["plans"]}
    assert list(got) == list(old["plans"])
    for name, want in old["plans"].items():
        p = got[name]
        assert p["quality"]["perplexity"] == want["quality"]["perplexity"], name      # bit-identical floats
        assert p["quality"]["kl"] == want["quality"]["kl"], name
        assert p["size"] == want["size"], name
        for k in ("rows", "predicted_loss", "allocated_bpw", "by_codec"):
            assert p[k] == json.loads(json.dumps(want[k])), (name, k)


def _load_cands(path: str) -> dict[str, list[Candidate]]:
    return {n: [Candidate(**x) for x in cs] for n, cs in json.loads(Path(path).read_text()).items()}


# ------------------------------------------------------------------------------------------------ config
def test_candidates_artifact_config_field(tmp_path, monkeypatch):
    path = _cfg(tmp_path, monkeypatch, "dense")
    assert load_allocate_config(path).candidates_artifact is None
    d = yaml.safe_load(path.read_text())
    d["candidates_artifact"] = "experiments/artifacts/x_candidates.json"
    path.write_text(yaml.safe_dump(d))
    assert load_allocate_config(path).candidates_artifact == "experiments/artifacts/x_candidates.json"


# -------------------------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("kind", ["dense", "moe", "head"])
def test_streamed_run_equals_whole_model_path(tmp_path, monkeypatch, kind):
    path = _cfg(tmp_path, monkeypatch, kind)
    r = json.loads(arun.run_allocate(path).read_text())
    c = _Ctx(path)
    # scoring: the streamed Fisher + per-block scoring reproduce the whole-model candidates bit for bit
    got = _load_cands(r["candidates_artifact"])
    want = _old_candidates(c)
    assert list(json.loads(Path(r["candidates_artifact"]).read_text())) == sorted(want)   # artifact format kept
    ordered = {n: got[n] for n in want}
    assert list(ordered) == list(r["plans"][0]["rows"][i]["tensor"] for i in range(len(want)))
    heads = set(c.head_names)
    assert {n: v for n, v in ordered.items() if n not in heads} == {n: v for n, v in want.items() if n not in heads}
    for n in heads:     # embedding F: scatter-add summation noise only (tests/unit/test_fisher_streamed.py)
        assert [(x.codec, x.bits, x.rel_mse) for x in ordered[n]] == [(x.codec, x.bits, x.rel_mse) for x in want[n]]
        for x, y in zip(ordered[n], want[n]):
            assert x.loss == pytest.approx(y.loss, rel=1e-3)
    # validation: streamed reference / plans equal the whole-model loop on the same candidates (in the order the
    # old run built them: plan rows and the predicted-loss sum follow it)
    _assert_same_validation(r, _old_validation(c, ordered))
    assert r["streaming"]["reference"]["peak_memory_bytes"] > 0
    assert not any(Path(tmp_path / "artifacts").glob("*_work"))           # scratch state removed after success


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_resume_from_candidates_artifact(tmp_path, monkeypatch, kind):
    path = _cfg(tmp_path, monkeypatch, kind)
    first = json.loads(arun.run_allocate(path).read_text())
    d = yaml.safe_load(path.read_text())
    d["candidates_artifact"] = first["candidates_artifact"]
    path.write_text(yaml.safe_dump(d))

    def boom(*a, **kw):
        raise AssertionError("Fisher must not run when resuming from a candidates artifact")

    monkeypatch.setattr(arun, "fisher_diag_streamed", boom)
    monkeypatch.setattr(arun, "build_candidates", boom)
    r = json.loads(arun.run_allocate(path).read_text())
    assert r["candidates_artifact"] == first["candidates_artifact"]
    assert r["candidates_sha256"] == first["candidates_sha256"] and r["resumed_from_candidates"] is True
    assert r["reference"] == first["reference"]
    for a, b in zip(r["plans"], first["plans"]):
        for k in ("plan", "quality", "size", "rows", "allocated_bpw", "predicted_loss", "by_codec"):
            assert a[k] == b[k], (a["plan"], k)


def _tamper(tmp_path, monkeypatch, fn) -> Path:
    path = _cfg(tmp_path, monkeypatch, "moe")
    first = json.loads(arun.run_allocate(path).read_text())
    raw = json.loads(Path(first["candidates_artifact"]).read_text())
    fn(raw)
    bad = tmp_path / "bad_candidates.json"
    bad.write_text(json.dumps(raw))
    d = yaml.safe_load(path.read_text())
    d["candidates_artifact"] = str(bad)
    path.write_text(yaml.safe_dump(d))
    return path


@pytest.mark.parametrize("edit, match", [
    (lambda raw: raw.pop(sorted(raw)[0]), "missing"),
    (lambda raw: raw.update({"model.layers.9.self_attn.q_proj.weight": raw[sorted(raw)[0]]}), "extra"),
    (lambda raw: raw[sorted(raw)[-1]][0].update(codec="affine_b3_g64"), "codec"),
    (lambda raw: raw[sorted(raw)[-1]].pop(), "codec"),
    (lambda raw: raw[sorted(raw)[-1]][0].update(bogus=1), "field"),
])
def test_resume_rejects_mismatching_artifact(tmp_path, monkeypatch, edit, match):
    path = _tamper(tmp_path, monkeypatch, edit)
    with pytest.raises(ValueError, match=match):
        arun.run_allocate(path)


def test_resume_missing_artifact_fails_early(tmp_path, monkeypatch):
    path = _cfg(tmp_path, monkeypatch, "dense", candidates_artifact=str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="nope.json"):
        arun.run_allocate(path)
