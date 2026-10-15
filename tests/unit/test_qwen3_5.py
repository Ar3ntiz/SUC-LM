"""Dense qwen3_5 (Qwen/Qwen3.8-27B): adapter (HF and SUC namespaces), the MLX model (suc.models.mlx_qwen3_5) against
mlx-lm's own qwen3_5 implementation on a tiny random checkpoint with the real config's structure, streamed / full /
cached forward consistency, and `suc compress` end to end -> SUC1 -> Python runtime."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml

from suc.analysis.sensitivity import set_weight
from suc.io.safetensors_store import SafetensorsStore
from suc.models.adapters import Qwen35Adapter, get_adapter
from suc.models.streaming import load_lazy, stream_blocks
from tiny_models import tiny_qwen3_5_config, write_tiny_qwen3_5


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return write_tiny_qwen3_5(tmp_path_factory.mktemp("q35") / "model")


def test_adapter_both_namespaces(tiny_dir):
    cfg = json.loads((tiny_dir / "config.json").read_text())
    ad = get_adapter(cfg)
    assert isinstance(ad, Qwen35Adapter) and ad.num_layers == 4 and not ad.is_moe and ad.residual_width == 64
    hf = SafetensorsStore.from_dir(tiny_dir)
    ad.validate(hf.names())
    st = ad.source_store(hf)
    ad.validate(st.names())
    assert not [n for n in st.names() if n.startswith(("model.visual.", "mtp.", "model.language_model."))]
    specs = ad.linears(st.names())
    elig = {s.role for s in specs if s.eligible}
    assert elig == {"attn_q", "attn_k", "attn_v", "attn_o", "linattn_qkv", "linattn_z", "linattn_out",
                    "mlp_gate", "mlp_up", "mlp_down"}
    assert sum(s.role == "linattn_qkv" for s in specs) == 3 and sum(s.role == "attn_q" for s in specs) == 1
    assert ad.kv_bytes_per_token() == 2 * 1 * 2 * 16 * 2          # only the one full-attention layer
    with pytest.raises(ValueError, match="missing"):
        ad.validate([n for n in st.names() if "mlp.up_proj" not in n])


def test_matches_mlx_lm_qwen3_5(tiny_dir):
    """Same function as mlx-lm's qwen3_5 (float32 checkpoint). Known differences: the (1 + w) norm weights (mlx-lm
    adds 1 at load) and the q / k L2-norm eps (mean vs sum): agreement to ~1e-4, not bit-exact."""
    from mlx_lm.models import qwen3_5 as ref_mod
    from mlx_lm.utils import load_model

    model, _ = load_lazy(tiny_dir)
    ref, _ = load_model(Path(tiny_dir), lazy=False,
                        get_model_classes=lambda config: (ref_mod.Model, ref_mod.ModelArgs))
    mx.random.seed(3)
    x = mx.random.randint(2, 128, (2, 24))
    a, b = model(x), ref(x)
    assert a.shape == b.shape == (2, 24, 128)
    err = mx.abs(a - b).max().item()
    assert err < 2e-3 * mx.abs(b).max().item(), err


def test_streamed_full_and_cached_forward_agree(tiny_dir):
    model, _ = load_lazy(tiny_dir)
    mx.random.seed(4)
    x = mx.random.randint(2, 128, (2, 30))
    full = model(x)
    r = stream_blocks(model, x, return_logits=True, release=False, micro_batch=1)
    assert mx.array_equal(r["logits"], full)
    cache = model.make_cache()
    steps = [model(x[:, :11], cache=cache)] + [model(x[:, t:t + 1], cache=cache) for t in range(11, 30)]
    assert mx.abs(mx.concatenate(steps, axis=1) - full).max().item() < 1e-4


def test_bf16_checkpoint_streamed_bit_exact(tmp_path):
    d = write_tiny_qwen3_5(tmp_path / "m", seed=1, dtype=mx.bfloat16)
    model, _ = load_lazy(d)
    x = mx.random.randint(2, 128, (3, 20))
    full = model(x)
    r = stream_blocks(model, x, return_logits=True, micro_batch=2)
    assert full.dtype == mx.bfloat16 and mx.array_equal(r["logits"], full)


VQ = {"kind": "vq", "k": 16, "block_size": 2, "rotate_out": False, "kmeans_iters": 3, "residual_fraction": 0.02,
      "residual_structure": "columns"}
VQ_NAME = "vq_k16_g2_rot1s+res0.02_activation_cols"


@pytest.fixture
def q35_env(tmp_path, monkeypatch):
    import suc.pipeline.compress as comp
    from test_calibrate_run import FakeTokenizer, _docs

    mx.random.seed(0)
    tokens = mx.random.randint(2, 128, (2048,), dtype=mx.int32)
    monkeypatch.setattr(comp, "load_wikitext2", lambda split: ("unused", "wikihash"))
    monkeypatch.setattr(comp, "tokenize", lambda tok, text: tokens)
    docs, ood = _docs()
    model_dir = write_tiny_qwen3_5(tmp_path / "model", seed=2, dtype=mx.bfloat16)

    def run(plan: dict):
        cfg = {"name": "c_q35", "seed": 7, "output_dir": str(tmp_path / "results"), "run_dir": str(tmp_path / "run"),
               "model": {"id": "tiny/qwen3_5", "local_path": str(model_dir)},
               "calibration": {"seq_len": 32, "n_calibration": 16, "n_validation": 8, "n_ood": 4, "micro_batch": 4},
               "plan": plan, "eval": {"seq_len": 64, "max_tokens": 1024, "kl_sequences": 4, "chunk": 2},
               "finetune": None, "rotation_groups": True, "verify_tensors": 8}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return json.loads(comp.run_compress(p, tokenizer=FakeTokenizer(), docs=docs, ood_docs=ood).read_text())

    return tmp_path, model_dir, run


def test_compress_then_python_runtime(q35_env):
    from suc.io.suc1 import SucModel
    from suc.runtime.suc1_model import greedy, load_suc1_model

    _, model_dir, run = q35_env
    a8 = {"kind": "affine", "bits": 8, "group_size": 32}
    r = run({"mode": "uniform", "codec": VQ_NAME, "candidates": [VQ, a8, {"kind": "bf16"}],
             "role_codecs": {"affine_b8_g32": ["attn_q", "attn_k", "attn_v", "attn_o"]},
             "embedding_codec": {"kind": "affine", "bits": 4}})
    assert r["architecture"]["family"] == "qwen3_5" and np.isfinite(r["quality"]["perplexity"]["ppl"])
    assert r["plan"]["by_codec"] == {VQ_NAME: 3 * 3 + 4 * 3, "affine_b8_g32": 4}
    sm = SucModel.load(r["container"]["path"], verify=True)
    rep = lambda n: sm.descriptor(n)["representation_type"]  # noqa: E731
    assert rep("model.layers.0.linear_attn.in_proj_qkv.weight") == "vq"
    assert rep("model.layers.0.linear_attn.in_proj_a.weight") == "raw"
    assert rep("model.layers.3.self_attn.q_proj.weight") == "affine"
    assert not [n for n in sm.names() if n.startswith(("model.visual", "mtp.", "model.language_model."))]
    model, _, _ = load_suc1_model(r["container"]["path"], group=False)
    ref, _ = load_lazy(model_dir)
    for n in sm.names():
        if sm.descriptor(n)["representation_type"] != "raw":
            set_weight(ref, n, sm.decode_tensor(n))
    x = mx.random.randint(2, 128, (2, 24))
    lr, lq = ref(x).astype(mx.float32), model(x).astype(mx.float32)
    p = mx.softmax(lr, axis=-1)
    kl = (p * (mx.log(p + 1e-9) - mx.log(mx.softmax(lq, axis=-1) + 1e-9))).sum(-1).mean().item()
    assert kl < 2e-2, kl
    out = greedy(model, [5, 6, 7, 8], 4)
    assert len(out) == 4 and all(0 <= t < 128 for t in out)


def test_real_config_args():
    cfg = tiny_qwen3_5_config()
    cfg["text_config"].update(hidden_size=5120, intermediate_size=17408, num_hidden_layers=64, head_dim=256,
                              num_attention_heads=24, num_key_value_heads=4, linear_num_value_heads=48,
                              linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128,
                              vocab_size=248320, layer_types=None)
    from suc.models import mlx_qwen3_5 as q35
    a = q35.ModelArgs.from_dict(cfg)
    assert a.text.output_gate_type == "silu" and a.text.rotary_dim == 64 and a.intermediate_size == 17408
    assert a.text.layer_types.count("full_attention") == 16
