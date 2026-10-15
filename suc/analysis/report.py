"""Aggregate Phase 1 per-tensor results and render the Markdown report (spec §8 deliverable)."""

from __future__ import annotations

from collections import defaultdict


def role_group(role: str) -> str:
    """attn_q -> attn, mlp_down -> mlp, expert_up -> expert, shared_expert_gate -> shared_expert."""
    return role.rsplit("_", 1)[0]


def summarize(tensors: list[dict]) -> list[dict]:
    """One row per (role group, rotation, block size, K) with mean/max held-out error across tensors."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in tensors:
        for cb in t["codebooks"]:
            groups[(role_group(t["role"]), cb["rotation"], cb["block_size"], cb["k"])].append(cb)
    rows = []
    for (group, rot, g, k), cbs in groups.items():
        mse = [c["rel_mse_heldout"] for c in cbs]
        rows.append({
            "group": group, "rotation": rot, "block_size": g, "k": k, "tensors": len(cbs),
            "nominal_index_bpw": cbs[0]["nominal_index_bpw"],
            "execution_viable": cbs[0]["execution_viable"],
            "gaussian_rd_bound": 2.0 ** (-2.0 * cbs[0]["nominal_index_bpw"]),
            "rel_mse_heldout_mean": sum(mse) / len(mse),
            "rel_mse_heldout_max": max(mse),
            "entropy_frac_mean": sum(c["occupancy"]["entropy_frac"] for c in cbs) / len(cbs),
            "occupied_frac_mean": sum(c["occupancy"]["occupied_frac"] for c in cbs) / len(cbs),
        })
    return sorted(rows, key=lambda r: (r["group"], r["rotation"], r["nominal_index_bpw"], r["block_size"]))


def markdown_report(result: dict) -> str:
    meta = result["metadata"]
    out = [f"# Phase 1 weight-structure report: {result['name']}", "",
           f"Model `{meta['model']}` · weights hash `{meta['model_hash'][:16]}` · git `{meta.get('git_commit')}`", "",
           "Relative MSE = sum ||b - C[i]||^2 / sum ||b||^2 on held-out blocks (row-RMS normalised).",
           "Nominal bpw counts index bits only; see spec §25 for actual bpp.",
           "Gaussian R(D) = 2^(-2·bpw): the Shannon bound for an iid Gaussian source at that rate (reference only).", "",
           "## Summary by role group", "",
           "| Group | Rotation | Block size | K | Index bpw | Viable (D4) | Tensors | Rel. MSE mean | Rel. MSE max | Gaussian R(D) | Entropy / log2 K |",
           "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in summarize(result["tensors"]):
        out.append(f"| {r['group']} | {'on' if r['rotation'] else 'off'} | {r['block_size']} | {r['k']} "
                   f"| {r['nominal_index_bpw']:.3f} | {'yes' if r['execution_viable'] else 'no'} | {r['tensors']} "
                   f"| {r['rel_mse_heldout_mean']:.4f} | {r['rel_mse_heldout_max']:.4f} | {r['gaussian_rd_bound']:.4f} "
                   f"| {r['entropy_frac_mean']:.3f} |")
    out += ["", "## Scalar structure (rotation effect)", "",
            "| Tensor | Kurtosis | Kurtosis (rotated) | Outliers >6σ | Outliers >6σ (rotated) |",
            "|---|---:|---:|---:|---:|"]
    for t in result["tensors"]:
        s, sr = t["scalar"], t.get("scalar_rotated") or {}
        out.append(f"| {t['name']} | {s['kurtosis']:.3f} | {sr.get('kurtosis', float('nan')):.3f} "
                   f"| {s['outlier_rate']['6.0']:.2e} | {sr.get('outlier_rate', {}).get('6.0', float('nan')):.2e} |")
    out += ["", "## Per tensor", "",
            "| Tensor | Rotation | Block size | Blocks | K | Index bpw | Entropy / log2 K | Rel. MSE (fit) | Rel. MSE (held-out) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in result["tensors"]:
        for c in sorted(t["codebooks"], key=lambda c: (c["rotation"], c["block_size"], c["k"])):
            out.append(f"| {t['name']} | {'on' if c['rotation'] else 'off'} | {c['block_size']} | {c['num_blocks']} "
                       f"| {c['k']} | {c['nominal_index_bpw']:.3f} | {c['occupancy']['entropy_frac']:.3f} "
                       f"| {c['rel_mse_fit']:.4f} | {c['rel_mse_heldout']:.4f} |")
    return "\n".join(out) + "\n"
