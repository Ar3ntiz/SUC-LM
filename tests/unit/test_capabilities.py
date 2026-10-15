"""Audit F07: the central codec capability table and the `suc compress` export gate (CPU-only config checks plus a
tiny encode / SUC1 round trip that ties the table to the writer)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from suc.capabilities import CAPABILITIES, TCQ_SUC1_CODES, capability, export_errors
from suc.config import CodecConfig, CompressPlanConfig, load_compress_config

ROOT = Path(__file__).resolve().parents[2]


def _suc1_codec_kinds() -> tuple:
    tree = ast.parse((ROOT / "suc/io/suc1.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_CODEC_KINDS" for t in n.targets))


def test_table_matches_suc1_writer_kinds():
    exportable = {k for k, v in CAPABILITIES.items() if v["suc1_export"]}
    assert exportable == set(_suc1_codec_kinds())
    assert set(TCQ_SUC1_CODES) == {"1mad", "3inst", "pair3"}
    for row in CAPABILITIES.values():
        assert set(row) == {"encode", "quality_decode", "suc1_export", "cpp_reference_decode", "python_runtime",
                            "cpp_runtime"}
        if not row["suc1_export"]:        # nothing downstream of the container without a container
            assert not (row["cpp_reference_decode"] or row["python_runtime"] or row["cpp_runtime"])


def test_cpp_runtime_column_matches_graph_source():
    src = (ROOT / "cpp/src/metal/model.cpp").read_text()
    assert "scalar is not supported" in src                     # scalar: no C++ model-graph kernel
    assert CAPABILITIES["scalar"]["cpp_runtime"] is False
    dec = (ROOT / "cpp/src/decode.cpp").read_text()
    for k, row in CAPABILITIES.items():
        if row["cpp_reference_decode"] and k not in ("vq", "additive"):
            assert f'rep == "{k}"' in dec, k                    # explicit branch in the CPU reference decoder


@pytest.mark.parametrize("code,ok", [("1mad", True), ("3inst", True), ("hyb", False)])
def test_compress_plan_gate(code, ok):
    c = CodecConfig(kind="tcq", tcq_code=code, rotate_out=False)
    assert capability(c)["suc1_export"] is ok
    if ok:
        assert CompressPlanConfig(mode="uniform", candidates=(c,), codec=c.name).codec == c.name
    else:
        with pytest.raises(ValueError, match="research-only"):
            CompressPlanConfig(mode="uniform", candidates=(c,), codec=c.name)
        vq = CodecConfig(kind="vq", k=16, block_size=8)
        with pytest.raises(ValueError, match="research-only"):   # also as a greedy candidate / head candidate
            CompressPlanConfig(mode="greedy", candidates=(vq, c), budget_bpw=3.0)


def test_tcq_parameter_gate():
    assert export_errors(CodecConfig(kind="tcq", tcq_code="pair3", tcq_V=2, rotate_out=False)) == []
    assert export_errors(CodecConfig(kind="tcq", tcq_code="pair3", rotate_out=False))     # pair3 needs V = 2
    assert export_errors(CodecConfig(kind="tcq", tcq_code="1mad", tcq_V=4, tcq_bits=2.0, rotate_out=False))


def test_research_codec_still_available_to_quantize():
    from suc.config import QuantizeExperimentConfig  # noqa: F401  (the gate lives in CompressPlanConfig only)

    c = CodecConfig(kind="tcq", tcq_code="hyb", rotate_out=False)
    assert c.name.startswith("tcq_hyb") and capability(c)["encode"]


def test_existing_compress_recipes_pass_the_gate():
    paths = sorted((ROOT / "experiments/configs").glob("compress_*.yaml"))
    assert paths
    for p in paths:
        load_compress_config(p)


def test_hyb_rejected_by_suc1_writer(tmp_path):
    """The table's research-only row is what the writer actually refuses (tiny tensor, MLX)."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    from suc.io.suc1 import write_suc1
    from suc.quant.trellis import TrellisCodec

    w = mx.array(np.random.default_rng(0).standard_normal((4, 256)).astype(np.float32) * 0.02)
    e = TrellisCodec(k=2.0, L=8, code="hyb", T=256).encode(w, seed=1)
    with pytest.raises(ValueError, match="not a SUC1 representation"):
        write_suc1(tmp_path / "h.suc1", architecture={"model_type": "qwen3"}, tokenizer_id="t", tensors={"w": e})
