"""Audit F08: candidate scoring and final encoding use the same rotation seeds (suc.quant.seeds), with rotation
groups on and off (identical stored arrays, hence identical Fisher losses); the candidates fingerprint carries the
seed contract; reuse of old-contract scores is limited to splits the contract does not affect."""
from __future__ import annotations

import functools
import json
from collections import defaultdict

import mlx.core as mx
import numpy as np
import pytest

import suc.pipeline.compress as comp
from suc.config import CodecConfig
from suc.quant import codecs as qc
from suc.quant.seeds import SEED_CONTRACT, encode_seed, rotation_group, rotation_kwargs, rotation_seeds
from suc.quant.stacks import stack_seeds
from test_compress import _base_cfg, env  # noqa: F401  (pytest fixture)

Q = "model.layers.0.self_attn.q_proj.weight"


class _Fake:
    """Minimal codec objects for rotation_kwargs (encode signature + codebook_scope)."""

    def __init__(self, seeds: bool, scope="tensor", trellis=False):
        self.codebook_scope = scope
        self._trellis = trellis
        if seeds:
            self.encode = lambda w, *, seed, seed_in=None: None
        else:
            self.encode = lambda w, *, seed: None


def test_audit_example_seed_values():
    """The audit's example: per-tensor default seed_in (1397435893) vs the group seed (515177825)."""
    assert qc._derive_seeds(encode_seed(Q, 12345), 2)[0] == 1397435893
    assert rotation_seeds(Q, 12345)["seed_in"] == 515177825
    vq = qc.VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=1)
    assert rotation_kwargs(vq, Q, 12345, True) == {"seed_in": 515177825}
    assert rotation_kwargs(vq, Q, 12345, False) == {}


def test_rotation_kwargs_cover_groups_and_stacks():
    vq = qc.VectorCodebook(16, block_size=8, rotate=True, kmeans_iters=1)
    aff = qc.AffineRTN(4, 64)
    names = {"q": Q, "k": Q.replace("q_proj", "k_proj"), "v": Q.replace("q_proj", "v_proj"),
             "gate": "model.layers.1.mlp.gate_proj.weight", "up": "model.layers.1.mlp.up_proj.weight",
             "o": "model.layers.0.self_attn.o_proj.weight", "down": "model.layers.1.mlp.down_proj.weight"}
    s = {k: rotation_kwargs(vq, n, 3, True).get("seed_in") for k, n in names.items()}
    assert s["q"] == s["k"] == s["v"] and s["gate"] == s["up"] and s["q"] != s["gate"]
    assert s["o"] is None and s["down"] is None                      # ungrouped: the per-tensor default
    assert rotation_kwargs(aff, Q, 3, True) == {}                    # codecs without seed_in are untouched
    ex = "model.layers.2.mlp.experts.5.gate_proj.weight"
    assert rotation_group(ex) is None and rotation_kwargs(vq, ex, 3, True) == {}   # experts are never grouped
    from suc.quant.trellis import TrellisCodec

    t = TrellisCodec(k=2.0, L=8, code="1mad", T=256)
    t.codebook_scope = "expert_stack"
    st = stack_seeds("model.layers.2.mlp.switch_mlp.gate_proj.weight", 3)["seed_in"]
    assert rotation_kwargs(t, ex, 3, True) == {"seed_in": st} == rotation_kwargs(t, ex, 3, False)


def _record(monkeypatch):
    """Record (encode seed -> [(seed_in, sha of the stored base arrays)]) of every VectorCodebook encode."""
    import hashlib

    calls = defaultdict(list)
    real = qc.VectorCodebook.encode

    @functools.wraps(real)                       # keep the signature: ResidualCodec inspects it for seed_in
    def enc(self, w, *, seed, **kw):
        e = real(self, w, seed=seed, **kw)
        h = hashlib.sha256(b"".join(np.array(e.arrays[k]).tobytes() for k in sorted(e.arrays))).hexdigest()
        calls[seed].append((e.params.get("seed_in"), h))
        return e

    monkeypatch.setattr(qc.VectorCodebook, "encode", enc)
    return calls


@pytest.mark.parametrize("groups", [True, False])
def test_scoring_and_encoding_identical(env, monkeypatch, groups):  # noqa: F811
    """Tiny greedy compress with a rotated VQ (+ residual wrapper) candidate: for every tensor the candidates stage
    and the encode stage call the VQ codec with the same seed and seed_in and get identical stored arrays; with
    rotation groups q/k/v and gate/up share one seed_in."""
    tmp_path, model_dir, run = env
    calls = _record(monkeypatch)
    vq = {"kind": "vq", "k": 16, "block_size": 8, "kmeans_iters": 2, "rotate_out": False}
    plan = {"mode": "greedy", "budget_bpw": 16.0,             # budget large enough that every tensor takes the VQ
            "candidates": [vq, {**vq, "residual_fraction": 0.01}],     # both VQ: the encode stage calls it too
            "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_sequences": 4, "fisher_micro_batch": 2}
    run(_base_cfg(tmp_path, model_dir, plan=plan, finetune=None, rotation_groups=groups))
    names = [n for n in json.loads((tmp_path / "run" / "plan" / "plan.json").read_text())["tensors"]
             if n.startswith("model.layers.")]
    by_seed = {encode_seed(n, 7): n for n in names}
    seen = 0
    for seed, got in calls.items():
        n = by_seed.get(seed)
        if n is None:
            continue
        # scoring: vq + residual base (2 calls), encode: at least the chosen codec's base (>= 1 call)
        assert len(got) >= 3, (n, got)
        assert len({g[0] for g in got}) == 1, (n, got)          # one seed_in for scoring and encoding
        assert len({g[1] for g in got}) == 1, (n, "encoded arrays differ between scoring and encoding")
        want = rotation_seeds(n, 7).get("seed_in") if groups else None
        if want is not None:
            assert got[0][0] == want
        seen += 1
    assert seen == len(names)
    man = json.loads((tmp_path / "run" / "candidates" / "manifest.json").read_text())
    assert ({"rotation_groups": True, "seed_contract": SEED_CONTRACT}.items() <= man["inputs"].items()) == groups
    assert ("seed_contract" in man["inputs"]) == groups              # no rotation groups: fingerprint unchanged


def test_reuse_of_old_contract_scores_skips_grouped_tensors(env, monkeypatch):  # noqa: F811
    """A candidates_reuse source scored under the old contract: VQ scores of every scoring split that contains a
    rotation-grouped tensor (q/k/v, gate/up) are re-scored (splits without one - e.g. MoE expert stacks - keep
    reusing them)."""
    tmp_path, model_dir, run = env
    vq = {"kind": "vq", "k": 16, "block_size": 8, "kmeans_iters": 2, "rotate_out": False}
    plan = {"mode": "greedy", "budget_bpw": 6.0, "candidates": [{"kind": "affine", "bits": 4}, vq],
            "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_sequences": 4, "fisher_micro_batch": 2}
    run(_base_cfg(tmp_path, model_dir, plan=plan, finetune=None))
    import shutil

    shutil.copytree(tmp_path / "run", tmp_path / "run2")             # same calibration (as the frontier chains)
    for st in ("candidates", "plan", "encode", "container", "evaluate", "report"):
        shutil.rmtree(tmp_path / "run2" / st, ignore_errors=True)
    man_p = tmp_path / "run" / "candidates" / "manifest.json"
    man = json.loads(man_p.read_text())
    for k in ("rotation_groups", "seed_contract"):                 # make the source look pre-F08
        man["inputs"].pop(k)
    man_p.write_text(json.dumps(man))
    scored = []
    real = comp.build_candidates

    def spy(get, names, codecs, *a, **kw):
        scored.append((tuple(names), tuple(codecs)))
        return real(get, names, codecs, *a, **kw)

    monkeypatch.setattr(comp, "build_candidates", spy)
    run(_base_cfg(tmp_path, model_dir, plan={**plan, "candidates_reuse": str(tmp_path / "run")}, finetune=None,
                  run_dir=str(tmp_path / "run2")))
    vq_name = CodecConfig(**vq).name
    rescored = {n for names, cs in scored if vq_name in cs for n in names}
    grouped = {n for names, _ in scored for n in names if rotation_group(n) is not None}
    assert grouped and grouped <= rescored                         # q/k/v, gate/up: re-scored under contract 2


def test_pre_contract_candidates_are_invalidated_explicitly_and_kept(env, capsys):  # noqa: F811
    """A finished candidates stage from before the seed contract (rotation groups on) is not silently reused: the
    rerun logs the changed inputs and keeps the old directory as candidates.stale-<fp>."""
    tmp_path, model_dir, run = env
    vq = {"kind": "vq", "k": 16, "block_size": 8, "kmeans_iters": 2, "rotate_out": False}
    plan = {"mode": "greedy", "budget_bpw": 6.0, "candidates": [{"kind": "affine", "bits": 4}, vq],
            "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_sequences": 4, "fisher_micro_batch": 2}
    cfg = _base_cfg(tmp_path, model_dir, plan=plan, finetune=None)
    run(cfg)
    man_p = tmp_path / "run" / "candidates" / "manifest.json"
    man = json.loads(man_p.read_text())
    for k in ("rotation_groups", "seed_contract"):
        man["inputs"].pop(k)
    man["fingerprint"] = "0123456789abcdef-old"
    man_p.write_text(json.dumps(man))
    capsys.readouterr()
    run(cfg)
    out = capsys.readouterr().out
    assert "stage candidates: earlier complete run is stale (changed: rotation_groups, seed_contract)" in out
    assert (tmp_path / "run" / "candidates.stale-0123456789ab" / "candidates.json").exists()
    assert json.loads(man_p.read_text())["inputs"]["seed_contract"] == SEED_CONTRACT


def test_calibration_identity_checked_on_reuse(env, capsys):  # noqa: F811
    """Audit observation (resume input identity): a reused calibrate stage must match the current tokenizer files and
    corpus; a mismatch fails before any later stage, a pre-identity manifest is reported as unverified."""
    from test_calibrate_run import FakeTokenizer, _docs
    from test_compress import _write

    tmp_path, model_dir, run = env
    cfg = _base_cfg(tmp_path, model_dir, finetune=None)
    run(cfg)
    man_p = tmp_path / "run" / "calibrate" / "manifest.json"
    assert set(json.loads(man_p.read_text())["info"]["input_identity"]) == {"tokenizer_sha256", "corpus_sha256"}
    (model_dir / "tokenizer_config.json").write_text('{"changed": true}')          # local tokenizer edited
    with pytest.raises(comp.PreflightError, match="tokenizer_sha256"):
        run(cfg)
    (model_dir / "tokenizer_config.json").unlink()
    docs, ood = _docs()
    docs = {k: [d + " edited" for d in v] for k, v in docs.items()}
    with pytest.raises(comp.PreflightError, match="corpus_sha256"):
        comp.run_compress(_write(tmp_path, cfg), tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood)
    man = json.loads(man_p.read_text())
    man["info"].pop("input_identity")                                               # a run from before the check
    man_p.write_text(json.dumps(man))
    capsys.readouterr()
    run(cfg)
    assert "reused unverified" in capsys.readouterr().out
