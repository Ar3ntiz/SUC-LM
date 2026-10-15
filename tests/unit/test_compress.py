"""End-to-end tests of the unattended pipeline `suc compress` on a tiny random Qwen3 (no network)."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml

import suc.pipeline.compress as comp
from suc.config import compress_config_from_dict, load_compress_config
from suc.io.safetensors_store import SafetensorsStore
from suc.io.suc1 import SucModel
from suc.provenance import REQUIRED_METADATA_KEYS
from test_calibrate_run import FakeTokenizer, _docs, _tiny_model_dir

STAGES = ("calibrate", "reference", "plan", "encode", "finetune", "container", "evaluate", "report")


def _base_cfg(tmp_path: Path, model_dir: Path, **over) -> dict:
    cfg = {
        "name": "c_tiny", "seed": 7, "output_dir": str(tmp_path / "results"), "run_dir": str(tmp_path / "run"),
        "model": {"id": "tiny/qwen3", "local_path": str(model_dir)},
        "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
        "plan": {"mode": "uniform", "codec": "additive_m2_k16_g8_rot+res0.01_activation",
                 "candidates": [{"kind": "additive", "k": 16, "block_size": 8, "num_codebooks": 2,
                                 "kmeans_iters": 3, "residual_fraction": 0.01}],
                 "embedding_codec": {"kind": "affine", "bits": 4}},
        "eval": {"seq_len": 128, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2},
        "finetune": {"block_steps": 3, "lr": 1e-3, "sequences": 8, "micro_batch": 4},
        "rotation_groups": True, "verify_tensors": 6,
    }
    cfg.update(over)
    return cfg


def _write(tmp_path: Path, cfg: dict, name="cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    mx.random.seed(0)
    tokens = mx.random.randint(0, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(comp, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(comp, "tokenize", lambda tok, text: tokens)
    docs, ood = _docs()
    model_dir = _tiny_model_dir(tmp_path)

    def run(cfg: dict, name="cfg.yaml"):
        return comp.run_compress(_write(tmp_path, cfg, name), tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood)

    return tmp_path, model_dir, run


def _status(tmp_path: Path) -> dict:
    return {s["stage"]: s["status"] for s in json.loads((tmp_path / "run" / "status.json").read_text())["stages"]}


def test_config_validation(tmp_path):
    d = _base_cfg(tmp_path, tmp_path)
    cfg = compress_config_from_dict(d)
    assert cfg.plan.mode == "uniform" and cfg.finetune.block_steps == 3 and cfg.preflight.require_pinned_revision
    with pytest.raises(ValueError, match="unknown key"):
        compress_config_from_dict({**d, "bogus": 1})
    with pytest.raises(ValueError, match="not among the candidates"):
        compress_config_from_dict({**d, "plan": {**d["plan"], "codec": "vq_k256_g8_rot"}})
    with pytest.raises(ValueError, match="budget_bpw"):
        compress_config_from_dict({**d, "plan": {**d["plan"], "mode": "greedy", "codec": None}})
    with pytest.raises(ValueError, match="calib_artifact"):
        compress_config_from_dict({**d, "eval": {**d["eval"], "calib_artifact": "x.safetensors"}})
    with pytest.raises(ValueError, match="kl_sequences"):
        compress_config_from_dict({**d, "eval": {**d["eval"], "kl_sequences": 9}})
    assert load_compress_config(_write(tmp_path, d)).name == "c_tiny"


def test_full_run_then_skip_then_downstream_rerun(env):
    tmp_path, model_dir, run = env
    cfg = _base_cfg(tmp_path, model_dir)
    json_path = run(cfg)

    assert set(_status(tmp_path).values()) == {"ran"} and set(_status(tmp_path)) == set(STAGES)
    r = json.loads(json_path.read_text())
    assert set(REQUIRED_METADATA_KEYS) <= set(r["metadata"])
    assert r["metadata"]["dataset_hashes"]["calibration_partitions"] == r["calibration"]["partitions_hash"]
    suc_path = Path(r["container"]["path"])
    m = SucModel.load(suc_path, verify=True)
    store = SafetensorsStore.from_dir(model_dir)
    assert set(m.names()) == set(store.names())                 # every checkpoint tensor is in the file
    assert r["container"]["verify"]["bit_exact"] == r["container"]["verify"]["checked"] >= 6
    assert r["container"]["sha256"] == r["stages"]["container"]["outputs"]["model.suc1"]["sha256"]
    enc = [t for t in r["encode"]["tensors"]]
    assert len(enc) == 2 * 7 + 1                                 # 14 projections + tied embedding
    assert {t["plan_codec"] for t in enc if t["role"] == "embed"} == {"affine_b4_g64"}
    s = r["size"]
    assert 0 < s["bpp"] < 16 and s["file_bits"] >= s["bits_total"]
    assert np.isfinite(r["quality"]["perplexity"]["ppl"]) and r["quality"]["kl"]["kl_mean"] >= 0
    assert r["finetune"]["blocks"] == 2
    assert r["preflight"]["memory"]["peak_estimate_bytes"] > 0 and r["preflight"]["disk"]["need_bytes"] > 0
    assert r["streaming"]["requires_full_model"] == [] and r["streaming"]["holds_all_encodings"] == []
    assert set(r["streaming"]["streaming"]) == set(STAGES) - {"plan", "report"}
    for st in STAGES[:-1]:
        assert r["stages"][st]["status"] == "ran" and r["stages"][st]["seconds"] >= 0
    assert json_path.with_suffix(".md").read_text().startswith("# Compression report: c_tiny")

    # second run: everything is up to date, the existing result is returned and nothing is rewritten
    mtime = suc_path.stat().st_mtime_ns
    assert run(cfg) == json_path
    assert set(_status(tmp_path).values()) == {"skipped"}
    assert suc_path.stat().st_mtime_ns == mtime

    # a downstream change (fine-tuning steps) reruns only fine-tuning and what depends on it
    cfg2 = {**cfg, "finetune": {**cfg["finetune"], "block_steps": 4}}
    json2 = run(cfg2)
    st = _status(tmp_path)
    assert [k for k in STAGES if st[k] == "skipped"] == ["calibrate", "reference", "plan", "encode"]
    assert [k for k in STAGES if st[k] == "ran"] == ["finetune", "container", "evaluate", "report"]
    assert json2 != json_path


def test_encode_resumes_completed_blocks(env, monkeypatch):
    """A crash inside the encode stage keeps the finished block shards; the rerun encodes only the rest."""
    tmp_path, model_dir, run = env
    cfg = _base_cfg(tmp_path, model_dir, finetune=None)
    real, seen, crashed = comp.reconstruct_model, [], []

    def flaky(model, store, specs, *a, **kw):
        seen.append(specs[0].layer)
        if specs[0].layer == 1 and not crashed:
            crashed.append(1)
            raise RuntimeError("simulated crash")
        return real(model, store, specs, *a, **kw)

    monkeypatch.setattr(comp, "reconstruct_model", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg)
    assert seen == [None, 0, 1]
    seen.clear()
    run(cfg)
    assert seen == [1]                                            # head and block 0 were resumed from disk
    assert _status(tmp_path)["encode"] == "ran"


def test_greedy_plan_with_mixed_codecs_and_finetune(env):
    tmp_path, model_dir, run = env
    cfg = _base_cfg(tmp_path, model_dir, plan={
        "mode": "greedy", "budget_bpw": 3.5,
        "candidates": [{"kind": "affine", "bits": 2}, {"kind": "affine", "bits": 4},
                       {"kind": "vq", "k": 16, "block_size": 8, "kmeans_iters": 3, "residual_fraction": 0.01}],
        "embedding_codec": {"kind": "affine", "bits": 4},
        "fisher_sequences": 8, "fisher_micro_batch": 2})
    r = json.loads(run(cfg).read_text())
    assert _status(tmp_path)["candidates"] == "ran"
    assert r["plan"]["allocated_bpw"] <= 3.5 + 1e-9
    assert sum(r["plan"]["by_codec"].values()) == 14
    assert r["streaming"]["requires_full_model"] == [] and "candidates" in r["streaming"]["streaming"]
    assert SucModel.load(r["container"]["path"], verify=True)


def test_preflight_fails_early_on_insufficient_disk(env, monkeypatch):
    tmp_path, model_dir, run = env
    monkeypatch.setattr(comp, "free_disk_bytes", lambda path: 1024)
    with pytest.raises(comp.PreflightError, match="disk"):
        run(_base_cfg(tmp_path, model_dir))
    assert not list((tmp_path / "run").glob("*/manifest.json"))  # no heavy stage started
    pre = json.loads((tmp_path / "run" / "preflight.json").read_text())
    assert pre["ok"] is False and pre["disk"]["free_bytes"] == 1024


def test_preflight_requires_pinned_revision(env):
    tmp_path, model_dir, run = env
    cfg = _base_cfg(tmp_path, model_dir, model={"id": "Qwen/Qwen3-0.6B", "revision": "main"})
    with pytest.raises(comp.PreflightError, match="revision"):
        run(cfg)


def test_preflight_fails_on_memory(env, monkeypatch):
    tmp_path, model_dir, run = env
    monkeypatch.setattr(comp, "physical_memory_bytes", lambda: 1 << 16)
    with pytest.raises(comp.PreflightError, match="memory"):
        run(_base_cfg(tmp_path, model_dir))


def test_config_delta_ignores_defaulted_fields():
    """Fingerprints use only non-default config values: a field added later with a neutral default does not
    invalidate finished stages, while any explicitly set value does."""
    from dataclasses import dataclass, make_dataclass

    @dataclass(frozen=True)
    class A:
        x: int
        y: int = 1

    B = make_dataclass("A", [("x", int), ("y", int, 1), ("new", bool, False)], frozen=True)
    assert comp.config_delta(A(3)) == comp.config_delta(B(3)) == {"x": 3}
    assert comp.config_delta(B(3, new=True)) == {"x": 3, "new": True}
    assert comp.config_delta(A(3, y=2)) == {"x": 3, "y": 2}
    cfg = compress_config_from_dict(_base_cfg(Path("/tmp"), Path("/tmp")))
    d = comp.config_delta(cfg.plan)
    assert d["candidates"][0] == {"kind": "additive", "k": 16, "num_codebooks": 2, "kmeans_iters": 3,
                                  "residual_fraction": 0.01, "name": "additive_m2_k16_g8_rot+res0.01_activation"}


def _greedy_plan(**over) -> dict:
    return {"mode": "greedy", "budget_bpw": 3.5,
            "candidates": [{"kind": "affine", "bits": 2}, {"kind": "affine", "bits": 4},
                           {"kind": "vq", "k": 16, "block_size": 8, "kmeans_iters": 3, "residual_fraction": 0.01}],
            "head_candidates": [{"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 8}],
            "fisher_sequences": 8, "fisher_micro_batch": 2, **over}


def test_streamed_container_equals_in_memory_writer(env):
    """The layer-streamed container (write_suc1_streamed over the block shards) is byte-identical to write_suc1 on
    the same in-memory content."""
    from suc.io.suc1 import write_suc1

    tmp_path, model_dir, run = env
    r = json.loads(run(_base_cfg(tmp_path, model_dir)).read_text())
    run_dir = tmp_path / "run"
    tensors = {}
    for f in sorted((run_dir / "encode").glob("*.suc1")) + sorted((run_dir / "finetune").glob("*.suc1")):
        sm = SucModel.load(f)
        tensors.update({n: sm.encoded(n) for n in sm.names()})
    store = SafetensorsStore.from_dir(model_dir)
    for n in store.names():
        if n not in tensors and n not in r["container"]["skipped_duplicates"]:
            tensors[n] = store.load(n)
    meta = SucModel.load(run_dir / "container" / "model.suc1").metadata
    info = write_suc1(tmp_path / "ref.suc1", architecture=json.loads((model_dir / "config.json").read_text()),
                      tokenizer_id=SucModel.load(run_dir / "container" / "model.suc1").tokenizer_id,
                      tensors=tensors, metadata=meta)
    assert info["sha256"] == r["container"]["sha256"]
    assert (tmp_path / "ref.suc1").read_bytes() == (run_dir / "container" / "model.suc1").read_bytes()


def test_candidates_resume_and_spilled_hidden_states(env, monkeypatch):
    """A crash while scoring keeps the finished Fisher blocks / scored parts: the rerun scores only the remaining
    groups. Hidden states spilled to disk (HIDDEN_IN_MEMORY_MAX = 0) give the same PPL / KL as in memory."""
    tmp_path, model_dir, run = env
    cfg = _base_cfg(tmp_path, model_dir, plan=_greedy_plan(), finetune=None)
    real, scored, crashed = comp.build_candidates, [], []

    def flaky(get, names, *a, **kw):
        scored.append(names[0])
        if names[0].startswith("model.layers.0.") and not crashed:
            crashed.append(1)
            raise RuntimeError("simulated crash")
        return real(get, names, *a, **kw)

    monkeypatch.setattr(comp, "build_candidates", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg)
    assert [n.split(".")[2] for n in scored] == ["1", "0"]           # reverse block order
    scored.clear()
    r1 = json.loads(run(cfg).read_text())
    assert scored and all(not n.startswith("model.layers.1.") for n in scored)   # block 1 was not rescored
    assert scored[-1] == "model.embed_tokens.weight"
    assert not (tmp_path / "run" / "candidates" / "fisher_work").exists()

    monkeypatch.setattr(comp, "HIDDEN_IN_MEMORY_MAX", 0)
    cfg2 = _base_cfg(tmp_path, model_dir, plan=_greedy_plan(), finetune=None, run_dir=str(tmp_path / "run2"))
    r2 = json.loads(run(cfg2).read_text())
    assert r2["reference"] == r1["reference"] and r2["quality"] == r1["quality"]
    c1 = json.loads((tmp_path / "run" / "candidates" / "candidates.json").read_text())
    c2 = json.loads((tmp_path / "run2" / "candidates" / "candidates.json").read_text())
    emb = "model.embed_tokens.weight"   # its Fisher has an atomic scatter-add (lookup gradient): run-to-run noise
    assert {k: v for k, v in c1.items() if k != emb} == {k: v for k, v in c2.items() if k != emb}
    for a, b in zip(c1[emb], c2[emb]):
        assert a["bits"] == b["bits"] and a["loss"] == pytest.approx(b["loss"], rel=1e-4)
    p1, p2 = (json.loads((tmp_path / d / "plan" / "plan.json").read_text())["tensors"] for d in ("run", "run2"))
    assert p1 == p2 and r1["container"]["sha256"] == r2["container"]["sha256"]
    assert not (tmp_path / "run2" / "reference" / "hidden").exists()


# ------------------------------------------------------------------ final recommended configurations (tiny models)
# One-sided rotation, low-dimensional single-codebook VQ (G = 2), activation-selected COLUMN residuals, affine-4
# embedding / head, shared q/k/v and gate/up input rotation: compress -> SUC1 -> reference decoder -> evaluate, and
# the C++ reference decoder (skipped when it is not built) bit-exact against SucModel.decode_tensor.

VQ_G2_COLS = {"kind": "vq", "k": 16, "block_size": 2, "rotate_out": False, "kmeans_iters": 3,
              "residual_fraction": 0.02, "residual_structure": "columns"}
VQ_G2_COLS_NAME = "vq_k16_g2_rot1s+res0.02_activation_cols"


def _final_plan(**codec_over) -> dict:
    c = {**VQ_G2_COLS, **codec_over}
    return {"mode": "uniform", "codec": codec_over.pop("name", None) or VQ_G2_COLS_NAME, "candidates": [c],
            "embedding_codec": {"kind": "affine", "bits": 4}}


def _cpp_bit_exact(tmp_path: Path, suc_path: str | Path) -> None:
    """C++ suc1-decode of every tensor == SucModel.decode_tensor (float32 bits); skipped if the tool is not built."""
    from test_cpp_decoder import DECODE, _bits, _cpp_decode

    if not DECODE.exists():
        return
    out = tmp_path / "cpp_out"
    _cpp_decode(Path(suc_path), out)
    m = SucModel.load(suc_path)
    bad = [n for n in m.names()
           if not np.array_equal(_bits(np.load(out / f"{n}.npy")), _bits(np.array(m.decode_tensor(n))))]
    assert not bad, f"C++ decode differs from SucModel.decode_tensor: {bad[:5]}"


def _assert_final_layout(sm: SucModel, name: str, k: int = 16, g: int = 2) -> dict:
    d = sm.descriptor(name)
    assert d["representation_type"] == "vq" and d["block_size"] == g and d["index_bits"] == max(1, (k - 1).bit_length())
    assert d["residual_type"] == "columns_int8" and d["rotation"] == "in", d
    assert d["params"]["base"]["rotate_out"] is False and d["params"]["columns"] > 0
    return d


def test_final_dense_config_columns_one_sided_with_finetune(env):
    tmp_path, model_dir, run = env
    cfg = _base_cfg(tmp_path, model_dir, plan=_final_plan())
    r = json.loads(run(cfg).read_text())
    assert r["plan"]["by_codec"] == {VQ_G2_COLS_NAME: 14}
    assert r["finetune"]["blocks"] == 2
    sm = SucModel.load(r["container"]["path"], verify=True)
    for i in range(2):
        p = f"model.layers.{i}."
        seeds = {n: _assert_final_layout(sm, p + n + ".weight")["params"]["base"]["seed_in"]
                 for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        assert seeds["self_attn.q_proj"] == seeds["self_attn.k_proj"] == seeds["self_attn.v_proj"]   # rotation groups
        assert seeds["mlp.gate_proj"] == seeds["mlp.up_proj"]
    emb = sm.descriptor("model.embed_tokens.weight")
    assert emb["representation_type"] == "affine" and emb["residual_type"] == "none"
    # fine-tuning changed the column residual values of at least one tensor (bits unchanged)
    enc = {n: SucModel.load(f).encoded(n) for f in sorted((tmp_path / "run" / "encode").glob("block_*.suc1"))
           for n in SucModel.load(f).names()}
    tuned = [n for n in enc if not np.array_equal(np.array(sm.encoded(n).arrays["res_colval"]),
                                                  np.array(enc[n].arrays["res_colval"]))]
    assert tuned and all(sm.encoded(n).bits() == enc[n].bits() for n in enc)
    assert r["container"]["verify"]["ref_decoder_max_rel_err"] < 1e-2
    assert np.isfinite(r["quality"]["perplexity"]["ppl"]) and r["quality"]["kl"]["kl_mean"] >= 0
    _cpp_bit_exact(tmp_path, r["container"]["path"])


def test_final_dense_config_9bit_indices_without_finetune(env):
    """K = 512 (9-bit indices, packed at exact width) with a column residual and a CSR tail."""
    tmp_path, model_dir, run = env
    plan = _final_plan(k=512, residual_column_share=0.5, kmeans_iters=1)
    plan["codec"] = "vq_k512_g2_rot1s+res0.02_activation_cols0.5"
    r = json.loads(run(_base_cfg(tmp_path, model_dir, plan=plan, finetune=None)).read_text())
    sm = SucModel.load(r["container"]["path"], verify=True)
    d = _assert_final_layout(sm, "model.layers.0.mlp.down_proj.weight", k=512)
    assert d["params"]["n"] > 0                                                   # CSR tail present
    _cpp_bit_exact(tmp_path, r["container"]["path"])


def _moe_env(env):
    from test_expert_stacks import _moe_model_dir

    tmp_path, _, run = env
    return tmp_path, _moe_model_dir(tmp_path), run


def _assert_shared_stacks(sm: SucModel, plan: dict, codec: str, experts: int = 8) -> list[str]:
    from suc.quant import stacks

    by_stack: dict = {}
    for n, c in plan.items():
        if (st := stacks.stack_of(n)) is not None:
            by_stack.setdefault(st[0], set()).add(c)
    assert all(len(v) == 1 for v in by_stack.values())
    xs = sorted(k for k, v in by_stack.items() if v == {codec})
    for key in xs:   # one codebook for every expert of a shared stack, column residual per expert
        prefix = key.replace(".switch_mlp.", ".experts.{}.")
        assert len({tuple(sm.descriptor(prefix.format(e))["codebook_ids"]) for e in range(experts)}) == 1
        for e in range(experts):
            _assert_final_layout(sm, prefix.format(e))
    return xs


def test_final_moe_config_uniform_expert_stacks(env):
    tmp_path, model_dir, run = _moe_env(env)
    xs = {**VQ_G2_COLS, "codebook_scope": "expert_stack"}
    name = "vq_k16_g2_rot1s_xs+res0.02_activation_cols"
    plan = {"mode": "uniform", "codec": name, "allocation_unit": "expert_stack", "candidates": [xs],
            "embedding_codec": {"kind": "affine", "bits": 4}}
    r = json.loads(run(_base_cfg(tmp_path, model_dir, plan=plan, finetune=None)).read_text())
    p = json.loads((tmp_path / "run" / "plan" / "plan.json").read_text())
    assert p["summary"]["allocation_unit"] == "expert_stack"
    sm = SucModel.load(r["container"]["path"], verify=True)
    assert len(_assert_shared_stacks(sm, p["tensors"], name)) == 2 * 3
    _assert_final_layout(sm, "model.layers.0.self_attn.q_proj.weight")
    assert np.isfinite(r["quality"]["perplexity"]["ppl"])
    _cpp_bit_exact(tmp_path, r["container"]["path"])


def test_final_moe_config_greedy_with_column_candidate(env):
    tmp_path, model_dir, run = _moe_env(env)
    name = "vq_k16_g2_rot1s_xs+res0.02_activation_cols"
    plan = {"mode": "greedy", "budget_bpw": 4.0, "allocation_unit": "expert_stack",
            "candidates": [{"kind": "affine", "bits": 4}, {"kind": "affine", "bits": 2},
                           {**VQ_G2_COLS, "codebook_scope": "expert_stack"}],
            "embedding_codec": {"kind": "affine", "bits": 4}, "fisher_sequences": 8, "fisher_micro_batch": 2}
    r = json.loads(run(_base_cfg(tmp_path, model_dir, plan=plan, finetune=None)).read_text())
    assert r["plan"]["allocated_bpw"] <= 4.0 + 1e-9
    p = json.loads((tmp_path / "run" / "plan" / "plan.json").read_text())
    assert name in p["tensors"].values()
    sm = SucModel.load(r["container"]["path"], verify=True)
    _assert_shared_stacks(sm, p["tensors"], name)
    _cpp_bit_exact(tmp_path, r["container"]["path"])


def test_preflight_rejects_block_size_not_dividing_in_features(env):
    """G = 3 does not divide the tiny model's in_features (64 / 128; Qwen3-8B: 4096): fail before any stage."""
    tmp_path, model_dir, run = env
    plan = _final_plan(block_size=3)
    plan["codec"] = "vq_k16_g3_rot1s+res0.02_activation_cols"
    with pytest.raises(comp.PreflightError, match="block_size 3"):
        run(_base_cfg(tmp_path, model_dir, plan=plan, finetune=None))
    assert not list((tmp_path / "run").glob("*/manifest.json"))
    greedy = _greedy_plan(candidates=[{"kind": "affine", "bits": 4}, {**VQ_G2_COLS, "block_size": 3}])
    with pytest.raises(comp.PreflightError, match="vq_k16_g3"):
        run(_base_cfg(tmp_path, model_dir, plan=greedy, finetune=None))


@pytest.mark.parametrize("name, in_features", [
    ("compress_qwen3_8b_final", (4096, 12288)),                       # hidden, intermediate (down_proj input)
    ("compress_qwen3_30b_a3b_final", (2048, 4096, 768)),             # hidden, o_proj input (32 x 128), expert down
    ("compress_qwen3_30b_a3b_final_greedy", (2048, 4096, 768))])
def test_final_configs_load_and_fit_the_target_shapes(name, in_features):
    repo = Path(__file__).resolve().parents[2]
    cfg = load_compress_config(repo / "experiments/configs" / f"{name}.yaml")
    assert cfg.rotation_groups and cfg.model.revision in ("b968826d9c46dd6066d109eabc6255188de91218",
                                                           "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    assert cfg.plan.embedding_codec.name == "affine_b4_g64"
    codecs = cfg.plan.codec_configs()
    final = [c for c in codecs.values() if c.kind == "vq" and c.block_size == 2]
    assert len(final) == 1 and not final[0].rotate_out and final[0].residual_structure == "columns"
    assert final[0].k * final[0].block_size * 2 <= 16 * 1024                  # D4 codebook ceiling
    if "30b" in name:
        assert cfg.plan.allocation_unit == "expert_stack" and final[0].codebook_scope == "expert_stack"
    if name.endswith("greedy"):
        assert cfg.plan.budget_bpw == 4.0
    for c in codecs.values():
        for inn in in_features:
            assert comp._codec_shape_error(c, (128, inn)) is None, (c.name, inn)
