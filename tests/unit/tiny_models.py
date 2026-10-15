"""Shared tiny randomly initialized models for unit tests (no downloads)."""

import mlx.core as mx
from mlx_lm.models import qwen3, qwen3_moe


def tiny_qwen3(seed: int = 0, tie: bool = True, vocab: int = 128) -> qwen3.Model:
    mx.random.seed(seed)
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=vocab, num_key_value_heads=2,
        max_position_embeddings=512, rope_theta=10000.0, head_dim=16, tie_word_embeddings=tie,
    )
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model


def tiny_qwen3_moe(seed: int = 0, tie: bool = True, vocab: int = 128, experts: int = 8, top_k: int = 2,
                   moe_intermediate: int = 64, mlp_only_layers: tuple[int, ...] = (),
                   layers: int = 2) -> qwen3_moe.Model:
    """`layers`-block (default 2) Qwen3-MoE (hidden 64, `experts` routed experts, top-`top_k`); mlp_only_layers
    get a dense MLP."""
    mx.random.seed(seed)
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=layers, intermediate_size=128,
        num_attention_heads=4, num_experts=experts, num_experts_per_tok=top_k, decoder_sparse_step=1,
        mlp_only_layers=list(mlp_only_layers), moe_intermediate_size=moe_intermediate, rms_norm_eps=1e-6,
        vocab_size=vocab, num_key_value_heads=2, head_dim=16, rope_theta=10000.0, tie_word_embeddings=tie,
        max_position_embeddings=512, norm_topk_prob=True)
    model = qwen3_moe.Model(args)
    mx.eval(model.parameters())
    return model


# ------------------------------------------------------------------ qwen4_exp (Qwen3.8-Flash-Next structure)

def tiny_qwen4_exp_config(layers: int = 4, **text_over) -> dict:
    """HF-format config.json of a tiny qwen4_exp built from the real config's structure (docs/arch/qwen4_exp.md §8
    test plan): hidden 64, 4 streams x 64, [lin, lin, lin, full] layers, PLE on layer index 1, 8 experts top-2 +
    shared expert, QSA indexer (budget 8, ratio 4: sparse from 12 tokens on), 2 n-gram shards."""
    text = {
        "model_type": "qwen4_exp_text", "hidden_size": 64, "num_hidden_layers": layers, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128, "rms_norm_eps": 1e-6,
        "layer_types": [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(layers)],
        "full_attention_interval": 4, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 32, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "hc_count": 4, "hc_lowrank": 8, "ple_layer_ids": [2],
        "ple_embed_dim": 64, "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 2,
        "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8, "split_ngram_parts": 2,
        "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16, "indexer_budget": 8,
        "indexer_compress_ratio": 4, "output_gate_type": "sigmoid", "hidden_act": "silu", "eos_token_id": 1,
        "bos_token_id": 1, "max_position_embeddings": 4096, "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                            "mrope_interleaved": True, "mrope_section": [1, 1, 0]},
    }
    text.update(text_over)
    return {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
            "tie_word_embeddings": False, "text_config": text,
            "vision_config": {"model_type": "qwen4_exp_vision", "hidden_size": 16, "depth": 1}}


def tiny_qwen4_exp_hf_weights(config: dict, seed: int = 0, dtype=None) -> dict:
    """Random weights of the tiny config in the ORIGINAL HF layout (model.language_model.*, fused expert stacks
    [E, 2I, H] / [E, H, I], n-gram shards, I64 hash buffers, plus vision / MTP tensors the adapter must drop)."""
    import numpy as np
    from mlx.utils import tree_flatten

    from suc.models import mlx_qwen4_exp as q4

    dtype = dtype or mx.float32
    mx.random.seed(seed)
    model = q4.Model(q4.ModelArgs.from_dict(config))
    a = model.text
    out = {}
    for k, v in tree_flatten(model.parameters()):
        last = k.rsplit(".", 1)[-1]
        if any(t in k for t in ("hc_norm", "q_norm", "k_norm", "_layernorm", "norm_key", "norm_query", "norm_conv")):
            v = 0.1 * mx.random.normal(v.shape)                       # (1 + w) norms
        elif "linear_attn.norm.weight" in k:
            v = 1.0 + 0.1 * mx.random.normal(v.shape)
        elif last == "A_log":
            v = mx.log(mx.random.uniform(1.0, 16.0, v.shape))
        elif last == "dt_bias":
            v = 0.5 * mx.random.normal(v.shape)
        elif "conv1d" in k:
            v = 0.4 * mx.random.normal(v.shape)
        else:
            v = mx.random.normal(v.shape) * (v.shape[-1] ** -0.5)
        out[k] = v.astype(dtype)
    hf = {}
    for k, v in out.items():
        n = "model.language_model." + k[len("model."):] if k.startswith("model.") else k
        if ".switch_mlp." in n:
            p = n.split(".mlp.switch_mlp.")[0] + ".mlp.experts."
            if n.endswith("gate_proj.weight"):
                hf[p + "gate_up_proj"] = mx.concatenate([v, out[k.replace("gate_proj", "up_proj")]], axis=1)
            elif n.endswith("down_proj.weight"):
                hf[p + "down_proj"] = v
            continue
        hf[n] = v
    for i in model.ple_layers():
        idx = a.ple_index(i)
        sizes, offsets, rows = q4.ngram_head_sizes(a, idx)
        dim = a.ple_embed_dim // a.ngram_heads
        per = rows // a.split_ngram_parts
        pre = f"model.language_model.layers.{i}.ple.ple_embedding."
        for j in range(a.split_ngram_parts):
            hf[pre + f"ngram_embedding.shard_{j}.weight"] = mx.random.normal((per, dim)).astype(dtype)
        hf[pre + "layer_multipliers"] = mx.array(np.array(q4.layer_multipliers(a.vocab_size, a.ngram_size, idx,
                                                                                 a.seed), dtype=np.int64))
        hf[pre + "ngram_heads_offsets"] = mx.array(np.array(offsets, dtype=np.int64))
        hf[pre + "ngram_heads_vocab_sizes"] = mx.array(np.array(sizes, dtype=np.int64))
    hf["model.visual.patch_embed.proj.weight"] = mx.random.normal((16, 3, 2, 4, 4)).astype(dtype)
    hf["mtp.fc_hidden.weight"] = mx.random.normal((64, 64)).astype(dtype)
    return hf


def write_tiny_qwen4_exp(d, seed: int = 0, dtype=None, **text_over):
    """Tiny qwen4_exp HF checkpoint directory (config.json + 2 safetensors shards + index); returns the path."""
    import json
    from pathlib import Path

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    cfg = tiny_qwen4_exp_config(**text_over)
    w = tiny_qwen4_exp_hf_weights(cfg, seed, dtype)
    names = sorted(w)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    wmap = {}
    for f, ns in shards.items():
        mx.save_safetensors(str(d / f), {n: w[n] for n in ns})
        wmap.update(dict.fromkeys(ns, f))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    (d / "config.json").write_text(json.dumps(cfg))
    return d


# ------------------------------------------------------------------ qwen3_5 dense (Qwen3.8-27B structure)

def tiny_qwen3_5_config(layers: int = 4, **text_over) -> dict:
    """HF-format config.json of a tiny dense qwen3_5 with the real Qwen3.8-27B config's structure: [lin, lin, lin,
    full] layers, gated attention (q doubled), partial RoPE, swish output gate of the Gated DeltaNet norm, dense MLP,
    untied head, vision config + MTP layer (dropped)."""
    text = {
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": layers,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128, "rms_norm_eps": 1e-6,
        "layer_types": [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(layers)],
        "full_attention_interval": 4, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 32, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "attn_output_gate": True, "output_gate_type": "swish", "hidden_act": "silu", "eos_token_id": 1,
        "bos_token_id": 1, "max_position_embeddings": 4096, "tie_word_embeddings": False, "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                            "mrope_interleaved": True, "mrope_section": [1, 1, 0]},
    }
    text.update(text_over)
    return {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "tie_word_embeddings": False, "text_config": text,
            "vision_config": {"model_type": "qwen3_5", "hidden_size": 16, "depth": 1}}


def tiny_qwen3_5_hf_weights(config: dict, seed: int = 0, dtype=None) -> dict:
    """Random weights of the tiny config in the ORIGINAL HF layout (model.language_model.*, raw (1 + w) norm weights,
    conv [C, 1, K]) plus vision / MTP tensors the adapter must drop."""
    from mlx.utils import tree_flatten

    from suc.models import mlx_qwen3_5 as q35

    dtype = dtype or mx.float32
    mx.random.seed(seed)
    model = q35.Model(q35.ModelArgs.from_dict(config))
    hf = {}
    for k, v in tree_flatten(model.parameters()):
        last = k.rsplit(".", 1)[-1]
        if "layernorm" in k or "q_norm" in k or "k_norm" in k or k == "model.norm.weight":
            v = 0.1 * mx.random.normal(v.shape)                       # (1 + w) norms
        elif "linear_attn.norm.weight" in k:
            v = 1.0 + 0.1 * mx.random.normal(v.shape)
        elif last == "A_log":
            v = mx.log(mx.random.uniform(1.0, 16.0, v.shape))
        elif last == "dt_bias":
            v = 0.5 * mx.random.normal(v.shape)
        elif "conv1d" in k:
            v = 0.4 * mx.random.normal(v.shape)
        else:
            v = mx.random.normal(v.shape) * (v.shape[-1] ** -0.5)
        hf["model.language_model." + k[len("model."):] if k.startswith("model.") else k] = v.astype(dtype)
    h = config["text_config"]["hidden_size"]
    hf["model.visual.patch_embed.proj.weight"] = mx.random.normal((16, 3, 2, 4, 4)).astype(dtype)
    hf["mtp.fc.weight"] = mx.random.normal((h, 2 * h)).astype(dtype)
    hf["mtp.layers.0.mlp.up_proj.weight"] = mx.random.normal((32, h)).astype(dtype)
    return hf


def write_tiny_qwen3_5(d, seed: int = 0, dtype=None, **text_over):
    """Tiny dense qwen3_5 HF checkpoint directory (config.json + 2 safetensors shards + index); returns the path."""
    import json
    from pathlib import Path

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    cfg = tiny_qwen3_5_config(**text_over)
    w = tiny_qwen3_5_hf_weights(cfg, seed, dtype)
    names = sorted(w)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    wmap = {}
    for f, ns in shards.items():
        mx.save_safetensors(str(d / f), {n: w[n] for n in ns})
        wmap.update(dict.fromkeys(ns, f))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    (d / "config.json").write_text(json.dumps(cfg))
    return d


# ------------------------------------------------------------------ deepseek_v4 (DeepSeek-V4-Flash structure)

def tiny_deepseek_v4_config(**over) -> dict:
    """config.json of a tiny deepseek_v4 with the real config's structure (docs/arch/deepseek_v4.md §8 test plan):
    hidden 256, 4 layers with compress ratios [0, 4, 16, 4] (window, CSA + indexer, HCA (ratio 16 instead of 128 to
    keep sequences short), CSA), 1 hash-routed layer, 8 experts top-2 + shared expert, 4 hyper-connection streams,
    o_groups 2, sliding window 8, indexer top-4 (active from 20 tokens on), + one MTP stage in compress_ratios."""
    cfg = {
        "architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "vocab_size": 300,
        "hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 1,
        "head_dim": 128, "qk_rope_head_dim": 64, "q_lora_rank": 128, "o_groups": 2, "o_lora_rank": 64,
        "sliding_window": 8, "compress_ratios": [0, 4, 16, 4, 0], "compress_rope_theta": 160000,
        "rope_theta": 10000, "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 64,
                                              "beta_fast": 32, "beta_slow": 1},
        "index_n_heads": 4, "index_head_dim": 64, "index_topk": 4, "n_routed_experts": 8, "num_experts_per_tok": 2,
        "moe_intermediate_size": 128, "n_shared_experts": 1, "num_hash_layers": 1, "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5, "norm_topk_prob": True, "swiglu_limit": 10.0, "hc_mult": 4,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
        "num_nextn_predict_layers": 1, "bos_token_id": 0, "eos_token_id": 1, "torch_dtype": "bfloat16",
        "expert_dtype": "fp4", "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                                                       "quant_method": "fp8", "scale_fmt": "ue8m0",
                                                       "weight_block_size": [128, 128]},
    }
    cfg.update(over)
    return cfg


def _write_safetensors(path, tensors: dict) -> None:
    """Minimal safetensors writer for {name: (dtype string, numpy array)} (F8_E4M3 / F8_E8M0 need raw dtype strings,
    which mx.save_safetensors cannot write)."""
    import json
    import struct

    import numpy as np

    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        dt, a = tensors[name]
        raw = np.ascontiguousarray(a).tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def tiny_deepseek_v4_tensors(cfg: dict, seed: int = 0, mtp: bool = True) -> dict:
    """{reference name: (safetensors dtype, numpy array)} of a random tiny checkpoint in the ORIGINAL layout: FP8 E4M3
    + 128x128 E8M0 scales for the attention / shared-expert / indexer linears, MXFP4 (packed I8 + per-32 E8M0) for the
    routed experts, BF16 / F32 / I64 for the rest, plus one mtp.0 stage the adapter must drop."""
    import numpy as np

    from suc.io.quantized_source import np_quantize_fp8_block, np_quantize_mxfp4

    rng = np.random.default_rng(seed)
    D, V, H, hd = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], cfg["head_dim"]
    qr, G, R = cfg["q_lora_rank"], cfg["o_groups"], cfg["o_lora_rank"]
    E, k, inter, hc = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"], cfg["hc_mult"]
    ih, idd = cfg["index_n_heads"], cfg["index_head_dim"]
    out: dict = {}

    def bf16(a):
        return ("BF16", np.array(mx.array(np.asarray(a, np.float32)).astype(mx.bfloat16).view(mx.uint16)))

    def f32(a):
        return ("F32", np.asarray(a, np.float32))

    def lin(o, i):
        return rng.normal(size=(o, i)).astype(np.float32) * i ** -0.5

    def fp8(name, o, i):
        q, s = np_quantize_fp8_block(lin(o, i))
        out[name + ".weight"], out[name + ".scale"] = ("F8_E4M3", q), ("F8_E8M0", s)

    def fp4(name, o, i):
        q, s = np_quantize_mxfp4(lin(o, i))
        out[name + ".weight"], out[name + ".scale"] = ("I8", q), ("F8_E8M0", s)

    def norm(name, n):
        out[name] = bf16(1.0 + 0.1 * rng.normal(size=n))

    def hc_mixer(prefix, mix):
        out[prefix + "_fn"] = f32(0.05 * rng.normal(size=(mix, hc * D)))
        out[prefix + "_base"] = f32(0.3 * rng.normal(size=mix))
        out[prefix + "_scale"] = f32(1.0 + 0.2 * rng.normal(size=(3 if mix > hc else 1)))

    def block(p, ratio, hash_layer, stage):
        norm(p + "attn_norm.weight", D)
        norm(p + "ffn_norm.weight", D)
        hc_mixer(p + "hc_attn", (2 + hc) * hc)
        hc_mixer(p + "hc_ffn", (2 + hc) * hc)
        a = p + "attn."
        out[a + "attn_sink"] = f32(rng.normal(size=H))
        fp8(a + "wq_a", qr, D)
        norm(a + "q_norm.weight", qr)
        fp8(a + "wq_b", H * hd, qr)
        fp8(a + "wkv", hd, D)
        norm(a + "kv_norm.weight", hd)
        fp8(a + "wo_a", G * R, H * hd // G)
        fp8(a + "wo_b", D, G * R)
        if ratio:
            coff = 2 if ratio == 4 else 1
            out[a + "compressor.ape"] = f32(0.1 * rng.normal(size=(ratio, coff * hd)))
            out[a + "compressor.wkv.weight"] = bf16(lin(coff * hd, D))
            out[a + "compressor.wgate.weight"] = bf16(lin(coff * hd, D))
            norm(a + "compressor.norm.weight", hd)
            if ratio == 4:
                fp8(a + "indexer.wq_b", ih * idd, qr)
                out[a + "indexer.weights_proj.weight"] = bf16(lin(ih, D))
                out[a + "indexer.compressor.ape"] = f32(0.1 * rng.normal(size=(4, 2 * idd)))
                out[a + "indexer.compressor.wkv.weight"] = bf16(lin(2 * idd, D))
                out[a + "indexer.compressor.wgate.weight"] = bf16(lin(2 * idd, D))
                norm(a + "indexer.compressor.norm.weight", idd)
        f = p + "ffn."
        out[f + "gate.weight"] = bf16(lin(E, D))
        if hash_layer:
            out[f + "gate.tid2eid"] = ("I64", np.stack([rng.choice(E, k, replace=False) for _ in range(V)])
                                       .astype(np.int64))
        else:
            out[f + "gate.bias"] = f32(0.1 * rng.normal(size=E))
        for e in range(E):
            fp4(f + f"experts.{e}.w1", inter, D)
            fp4(f + f"experts.{e}.w3", inter, D)
            fp4(f + f"experts.{e}.w2", D, inter)
        fp8(f + "shared_experts.w1", inter, D)
        fp8(f + "shared_experts.w3", inter, D)
        fp8(f + "shared_experts.w2", D, inter)

    out["embed.weight"] = bf16(rng.normal(size=(V, D)))
    out["head.weight"] = bf16(lin(V, D))
    norm("norm.weight", D)
    hc_mixer("hc_head", hc)
    L = cfg["num_hidden_layers"]
    for i in range(L):
        block(f"layers.{i}.", cfg["compress_ratios"][i], i < cfg["num_hash_layers"], False)
    if mtp:
        block("mtp.0.", 0, False, True)
        out["mtp.0.norm.weight"] = bf16(np.ones(D))
    return out


def write_tiny_deepseek_v4(d, seed: int = 0, mtp: bool = True, **over):
    """Tiny deepseek_v4 checkpoint directory in the original FP8 / MXFP4 layout (config.json + 2 shards + index)."""
    import json
    from pathlib import Path

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    cfg = tiny_deepseek_v4_config(**over)
    t = tiny_deepseek_v4_tensors(cfg, seed, mtp)
    names = sorted(t)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    wmap = {}
    for f, ns in shards.items():
        _write_safetensors(d / f, {n: t[n] for n in ns})
        wmap.update(dict.fromkeys(ns, f))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    (d / "config.json").write_text(json.dumps(cfg))
    return d


# ------------------------------------------------------------------ glm5_next (GLM-5.3-Flash structure)

def tiny_glm5_next_config(**text_over) -> dict:
    """config.json of a tiny glm5_next with the real config's structure (docs/arch/glm5_next.md §8 test plan): hidden
    128, 4 layers [KDA, KDA, KDA, DSA] (layer_types named as in the real config), layer 0 dense, layers 1-3 MoE with 8
    experts top-2 + shared expert, 4 hyper-connection streams, KDA 2 heads x 32, MLA 2 heads x 32 (NoPE), DSA indexer
    8 heads x 32 (enough heads that all-ReLU-zero score ties, which torch.topk and argpartition break differently, are
    rare) with k-pool 4 and index_topk 8 (2 pools: the selection is active from 12 tokens on), the MTP layer (index 4)
    and a vision tower stub in the checkpoint. FP8 tensors have dims of 64 / 128 / 256 (whole or single
    128 x 128 blocks, as the real checkpoint)."""
    text = {
        "model_type": "glm5_next_text", "vocab_size": 256, "hidden_size": 128, "intermediate_size": 256,
        "moe_intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
        "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 32, "qk_rope_head_dim": 0, "qk_head_dim": 32,
        "v_head_dim": 32, "head_dim": 0, "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "routed_scaling_factor": 2.5, "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "scoring_func": "sigmoid",
        "topk_method": "noaux_tc", "swiglu_limit": 10.0, "first_k_dense_replace": 1,
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        "indexer_types": ["full"] * 4,
        "linear_attn_config": {"num_heads": 2, "head_dim": 32, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                               "kda_layers": [0, 1, 2], "full_attn_layers": [3]},
        "index_n_heads": 8, "index_head_dim": 32, "index_topk": 8, "index_kpool": 4,
        "index_kpool_always_select_tail": True, "index_kpool_compress": True, "hc_mult": 4, "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20, "mhc": True, "mla_use_nope": True, "rms_norm_eps": 1e-5, "hidden_act": "silu",
        "attention_bias": False, "tie_word_embeddings": False, "num_nextn_predict_layers": 1, "dtype": "bfloat16",
        "eos_token_id": [1], "pad_token_id": 0, "max_position_embeddings": 4096,
    }
    text.update(text_over)
    return {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
            "text_config": text, "vision_config": {"depth": 1, "hidden_size": 16, "num_heads": 2},
            "tie_word_embeddings": False, "torch_dtype": "bfloat16",
            "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                    "weight_block_size": [128, 128]}}


def tiny_glm5_next_tensors(cfg: dict, seed: int = 0, extras: bool = True) -> dict:
    """{checkpoint name: (safetensors dtype, numpy array)} of a random tiny checkpoint in the ORIGINAL layout:
    "model.language_model." names, FP8 E4M3 + F32 128 x 128 block multipliers ("weight_scale_inv") for exactly the
    families the real checkpoint stores in FP8 (experts, shared experts, dense MLP, MLA q_a / q_b / kv_a / o), BF16 /
    F32 for the rest; plus (extras) the MTP layer and a vision tensor that the adapter must drop."""
    import numpy as np

    from suc.io.quantized_source import np_quantize_fp8_block_f32

    t = cfg["text_config"]
    rng = np.random.default_rng(seed)
    D, V, L = t["hidden_size"], t["vocab_size"], t["num_hidden_layers"]
    la = t["linear_attn_config"]
    lh, ld, K = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
    H, nope, vd, qr, kr = (t["num_attention_heads"], t["qk_nope_head_dim"], t["v_head_dim"], t["q_lora_rank"],
                           t["kv_lora_rank"])
    E, inter, dense_i, hc = t["n_routed_experts"], t["moe_intermediate_size"], t["intermediate_size"], t["hc_mult"]
    ih, idd, kp = t["index_n_heads"], t["index_head_dim"], t["index_kpool"]
    P = "model.language_model."
    out: dict = {}

    def bf16(a):
        return ("BF16", np.array(mx.array(np.asarray(a, np.float32)).astype(mx.bfloat16).view(mx.uint16)))

    def f32(a):
        return ("F32", np.asarray(a, np.float32))

    def lin(o, i):
        return rng.normal(size=(o, i)).astype(np.float32) * i ** -0.5

    def fp8(name, o, i):
        q, s = np_quantize_fp8_block_f32(lin(o, i))
        out[name + ".weight"], out[name + ".weight_scale_inv"] = ("F8_E4M3", q), ("F32", s)

    def norm(name, n):
        out[name] = bf16(1.0 + 0.1 * rng.normal(size=n))

    def layer(i: int, kda: bool, sparse: bool, hc_params: bool = True):
        p = f"{P}layers.{i}."
        norm(p + "input_layernorm.weight", D)
        norm(p + "post_attention_layernorm.weight", D)
        if hc_params:
            for k in ("attn", "ffn"):
                out[p + f"hc_{k}_fn"] = bf16(0.05 * rng.normal(size=((2 + hc) * hc, hc * D)))
                out[p + f"hc_{k}_base"] = f32(0.3 * rng.normal(size=(2 + hc) * hc))
                out[p + f"hc_{k}_scale"] = f32(1.0 + 0.2 * rng.normal(size=3))
        a = p + "self_attn."
        if kda:
            d = lh * ld
            for k in ("q", "k", "v"):
                out[a + f"{k}_proj.weight"] = bf16(lin(d, D))
                out[a + f"{k}_conv1d.weight"] = bf16(0.5 * rng.normal(size=(d, 1, K)))
            out[a + "f_a_proj.weight"] = bf16(lin(ld, D))
            out[a + "f_b_proj.weight"] = bf16(lin(d, ld))
            out[a + "g_a_proj.weight"] = bf16(lin(ld, D))
            out[a + "g_b_proj.weight"] = bf16(lin(d, ld))
            out[a + "b_proj.weight"] = bf16(lin(lh, D))
            out[a + "A_log"] = f32(0.5 * rng.normal(size=lh))
            out[a + "dt_bias"] = f32(0.5 * rng.normal(size=d))
            norm(a + "o_norm.weight", ld)
            out[a + "o_proj.weight"] = bf16(lin(D, d))
        else:
            fp8(a + "q_a_proj", qr, D)
            norm(a + "q_a_layernorm.weight", qr)
            fp8(a + "q_b_proj", H * nope, qr)
            fp8(a + "kv_a_proj_with_mqa", kr, D)
            norm(a + "kv_a_layernorm.weight", kr)
            out[a + "kv_b_proj.weight"] = bf16(lin(H * (nope + vd), kr))
            fp8(a + "o_proj", D, H * vd)
            x = a + "indexer."
            out[x + "wq_b.weight"] = bf16(lin(ih * idd, qr))
            out[x + "wk.weight"] = bf16(lin(idd, D))
            out[x + "k_norm.weight"] = bf16(1.0 + 0.1 * rng.normal(size=idd))
            out[x + "k_norm.bias"] = bf16(0.1 * rng.normal(size=idd))
            out[x + "weights_proj.weight"] = bf16(lin(ih, D))
            out[x + "index_kpool_compress_gate"] = bf16(lin(idd, D))
            out[x + "index_kpool_compress_ape"] = bf16(0.5 * rng.normal(size=(kp, idd)))
        m = p + "mlp."
        if sparse:
            out[m + "gate.weight"] = bf16(lin(E, D))
            out[m + "gate.e_score_correction_bias"] = f32(0.1 * rng.normal(size=E))
            for e in range(E):
                fp8(m + f"experts.{e}.gate_proj", inter, D)
                fp8(m + f"experts.{e}.up_proj", inter, D)
                fp8(m + f"experts.{e}.down_proj", D, inter)
            fp8(m + "shared_experts.gate_proj", inter, D)
            fp8(m + "shared_experts.up_proj", inter, D)
            fp8(m + "shared_experts.down_proj", D, inter)
        else:
            fp8(m + "gate_proj", dense_i, D)
            fp8(m + "up_proj", dense_i, D)
            fp8(m + "down_proj", D, dense_i)

    out[P + "embed_tokens.weight"] = bf16(rng.normal(size=(V, D)))
    out["lm_head.weight"] = bf16(lin(V, D))
    norm(P + "norm.weight", D)
    kinds = t["layer_types"]
    for i in range(L):
        layer(i, kinds[i] == "linear_attention", t["mlp_layer_types"][i] == "sparse")
    if extras:
        layer(L, False, True, hc_params=False)                       # MTP layer (DSA + MoE, no HC parameters)
        out[f"{P}layers.{L}.eh_proj.weight"] = bf16(lin(D, 2 * D))
        for k in ("enorm", "hnorm", "shared_head.norm"):
            norm(f"{P}layers.{L}.{k}.weight", D)
        out["model.visual.post_layernorm.weight"] = bf16(np.ones(16))
    return out


def write_tiny_glm5_next(d, seed: int = 0, extras: bool = True, **text_over):
    """Tiny glm5_next checkpoint directory in the original FP8 layout (config.json + 2 shards + index)."""
    import json
    from pathlib import Path

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    cfg = tiny_glm5_next_config(**text_over)
    t = tiny_glm5_next_tensors(cfg, seed, extras)
    names = sorted(t)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    wmap = {}
    for f, ns in shards.items():
        _write_safetensors(d / f, {n: t[n] for n in ns})
        wmap.update(dict.fromkeys(ns, f))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def row_rotation_cache(monkeypatch, path) -> None:
    """Point the Python runtime and (through the environment) the C++ tools at a tuning cache that selects the
    rotation "row" layout (suc.kernels.rotation._kernel) for every rotation size m·2^k (m ∈ 1, 12, 20, 28, 2^k ≥ 32):
    the layout the M1 Ultra's cache picks for n = 1024, which the C++ port ran as "blocks" before 2026-10 (not
    bit-identical there). The rest of this device's entry (codebook_in, mma_min_batch, ...) is kept."""
    import json
    import os
    from pathlib import Path

    from suc.kernels import autotune

    src = Path(os.environ.get("SUC_TUNING_CACHE", Path.home() / ".cache" / "suc" / "kernel_tuning.json"))
    key = autotune.device_key()
    try:
        entry = json.loads(src.read_text()).get(key, {})
    except (OSError, ValueError):
        entry = {}
    sizes = [m * (1 << k) for m in (1, 12, 20, 28) for k in range(5, 16)]
    entry = dict(entry, rotation={"choice": {str(n): {"layout": "row"} for n in sizes}})
    Path(path).write_text(json.dumps({key: entry}))
    monkeypatch.setenv("SUC_TUNING_CACHE", str(path))
    monkeypatch.setattr(autotune, "_memory", None)   # reloaded from `path`; the previous cache comes back at teardown
