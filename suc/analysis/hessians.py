"""Input second moments for BlockLDLQ (suc.quant.trellis), recorded in one layer-streamed pass (spec D8).

`record_hessians` runs the model block by block over calibration sequences (the block-sequential pass of
suc.models.streaming.stream_blocks) with recording hooks on the tensors that may get an LDLQ codec, and writes ONE
safetensors file per block (block_{i:05d}.safetensors) as soon as the block has run:

  dense.<name>            (in, in) float32 E[x xᵀ] of the input of dense linear <name> (checkpoint name without
                          ".weight"); linears that read the same input object (q/k/v, gate/up) share one matrix,
                          stored under the first recorded name, the others listed in the metadata "aliases"
  moe.<prefix>.gate_up    (D, D) float32 E[x xᵀ] over every token entering the routed experts of the MoE layer
                          <prefix> ("model.layers.N.mlp"): ONE matrix shared by every gate / up expert (they all read
                          the same hidden state; per-expert matrices would cost E · D² · 4 bytes per layer)
  moe.<prefix>.down       (E, I, I) float32 per-expert E[h hᵀ] of the rows routed to expert e (the down projection
                          input, including whatever the model multiplies into it before w2, e.g. DeepSeek-V4's
                          routing weight)
  moe.<prefix>.down_count (E,) int64 routed rows per expert

Resumable: after each block the hidden states of all sequences and progress.json (keyed by the caller's
fingerprint) are written, so a crash resumes at the first unfinished block. Memory: one block of weights, the hidden
states, and the accumulators of one block (E · I² · 4 bytes for the per-expert down moments).

`HessianStore` serves the files to the encoders as a read-only mapping keyed like reconstruct_model's hessians
(tensor name without ".weight"): dense names, "<prefix>.experts.<e>.gate_proj" / "up_proj" (the shared matrix) and
"<prefix>.experts.<e>.down_proj" (expert e's matrix, absent when e saw fewer than min_tokens rows).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Iterator, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.switch_layers import SwitchGLU

from suc.analysis.activations import _recording_class
from suc.models.streaming import _release, block_prefix, embed, load_lazy, model_blocks

_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def block_file(i: int) -> str:
    return f"block_{i:05d}.safetensors"


def moe_prefixes(names: Sequence[str]) -> set[str]:
    """MoE layer prefixes ("model.layers.N.mlp") of expert tensor names (checkpoint names, with or without .weight)."""
    out = set()
    for n in names:
        m = _EXPERT.match(n.removesuffix(".weight"))
        if m is not None:
            out.add(m.group(1))
    return out


class _BlockHooks:
    """Recording hooks on one block: dense linears by name, SwitchGLU modules (routed experts) by layer prefix."""

    def __init__(self, dense: set[str], moe: set[str]):
        self.dense, self.moe = dense, moe
        self.sums: dict[str, mx.array] = {}
        self.counts: dict[str, int] = {}
        self.aliases: dict[str, str] = {}
        self.down: dict[str, tuple[mx.array, mx.array]] = {}
        self._swapped: list[tuple[nn.Module, type]] = []
        self._last: dict[int, tuple[mx.array, str]] = {}    # id of the parent module -> (last input, canonical name)

    def _swap(self, mod: nn.Module, hook) -> None:
        self._swapped.append((mod, type(mod)))
        mod._suc_hook = hook
        mod.__class__ = _recording_class(type(mod))

    def instrument(self, module: nn.Module, prefix: str) -> None:
        for path, mod in module.named_modules():
            name = f"{prefix}.{path}" if path else prefix
            if isinstance(mod, SwitchGLU):
                parent = name.rsplit(".", 1)[0]
                if parent in self.moe:
                    self._instrument_moe(mod, parent)
            elif isinstance(mod, nn.Linear) and name in self.dense:
                owner = name.rsplit(".", 1)[0]
                self._swap(mod, self._dense_hook(name, owner))

    def _acc(self, key: str, rows: mx.array) -> None:
        r = rows.reshape(-1, rows.shape[-1]).astype(mx.float32)
        s = r.T @ r
        self.sums[key] = s if key not in self.sums else self.sums[key] + s
        self.counts[key] = self.counts.get(key, 0) + int(r.shape[0])
        mx.eval(self.sums[key])

    def _dense_hook(self, name: str, owner: str):
        def hook(x, *_, **__):
            last = self._last.get(owner)
            if last is not None and last[0] is x:      # a sibling reading the same input object (q/k/v, gate/up)
                self.aliases[name] = last[1]
                return
            self._last[owner] = (x, name)
            self._acc(f"dense.{name}", x)
        return hook

    def _instrument_moe(self, glu: SwitchGLU, prefix: str) -> None:
        E, inner = glu.down_proj.num_experts, glu.down_proj.input_dims

        def glu_hook(x, *_, **__):
            self._acc(f"moe.{prefix}.gate_up", x)

        def down_hook(h, indices, *_, **__):
            rows = mx.broadcast_to(h, indices.shape + h.shape[-2:]).reshape(-1, inner).astype(mx.float32)
            e = indices.reshape(-1).astype(mx.uint32)
            order = mx.argsort(e)
            rows, e = rows[order], e[order]
            cnt = mx.zeros((E,), mx.int32).at[e].add(1)
            ends = mx.cumsum(cnt)
            seg = mx.stack([ends - cnt, ends], axis=1).astype(mx.uint32)
            s = mx.segmented_mm(rows.T, rows, seg)
            cur = self.down.get(prefix)
            self.down[prefix] = (s, cnt.astype(mx.int64)) if cur is None else (cur[0] + s, cur[1] + cnt)
            mx.eval(self.down[prefix])

        self._swap(glu, glu_hook)
        self._swap(glu.down_proj, down_hook)

    def remove(self) -> None:
        for mod, cls in reversed(self._swapped):
            mod.__class__ = cls
            del mod._suc_hook
        self._swapped.clear()
        self._last.clear()

    def arrays(self) -> tuple[dict[str, mx.array], dict[str, str]]:
        out: dict[str, mx.array] = {}
        for k, s in self.sums.items():
            out[k] = s / max(self.counts[k], 1)
        for prefix, (s, cnt) in self.down.items():
            out[f"moe.{prefix}.down"] = s / mx.maximum(cnt, 1).astype(mx.float32)[:, None, None]
            out[f"moe.{prefix}.down_count"] = cnt
        meta = {"aliases": json.dumps(self.aliases), "counts": json.dumps(self.counts)}
        return out, meta


def record_hessians(model_dir: str | Path, tokens: mx.array, out_dir: str | Path, *, dense: Sequence[str] = (),
                    moe: Sequence[str] = (), micro_batch: int = 8, fingerprint: str = "",
                    log: Callable[[str], None] | None = None) -> dict:
    """Run the block-sequential pass over `tokens` (B, T) and write one moments file per block that has any of the
    requested tensors (module docstring). dense: linear names without ".weight"; moe: MoE layer prefixes.
    Returns {"files": [...], "seconds", "blocks", "peak_memory_bytes"}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    prog_path = out_dir / "progress.json"
    dense_s, moe_s = set(dense), set(moe)
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else None
    if prog is None or prog.get("fingerprint") != fingerprint:
        shutil.rmtree(work, ignore_errors=True)
        for f in out_dir.glob("block_*.safetensors"):
            f.unlink()
        prog = {"fingerprint": fingerprint, "next_block": 0, "files": []}
    model = load_lazy(model_dir)[0]
    blocks = model_blocks(model)
    n = tokens.shape[0]
    spans = [(s, min(s + micro_batch, n)) for s in range(0, n, micro_batch)]
    start = int(prog["next_block"])
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    if start == 0:
        hs = [embed(model, tokens[s:e]) for s, e in spans]
        mx.eval(hs)
    else:
        saved = mx.load(str(work / "hidden.safetensors"))
        hs = [saved[f"h{j}"] for j in range(len(spans))]
        mx.eval(hs)
        if log:
            log(f"hessians: resumed at block {start}")
    for i in range(start):
        _release(blocks[i])
    for i in range(start, len(blocks)):
        block = blocks[i]
        prefix = block_prefix(i)
        want_dense = {d for d in dense_s if d.startswith(prefix + ".")}
        want_moe = {m for m in moe_s if m.startswith(prefix + ".")}
        hooks = _BlockHooks(want_dense, want_moe) if (want_dense or want_moe) else None
        if hooks is not None:
            hooks.instrument(block, prefix)
        tb = time.perf_counter()
        new = []
        for h in hs:
            o = block(h, create_attention_mask(h, None))
            mx.eval(o)
            new.append(o)
        hs = new
        if hooks is not None:
            hooks.remove()
            arrays, meta = hooks.arrays()
            missing = [d for d in want_dense if f"dense.{d}" not in arrays and d not in hooks.aliases] + \
                [m for m in want_moe if f"moe.{m}.down" not in arrays]
            if missing:
                raise RuntimeError(f"hessians: block {i}: no input recorded for {missing[:4]}")
            f = out_dir / block_file(i)
            tmp = f.with_name(".tmp." + f.name)
            mx.save_safetensors(str(tmp), arrays, metadata=meta)
            os.replace(tmp, f)
            prog["files"].append(f.name)
            del arrays
        _release(block)
        mx.clear_cache()
        work.mkdir(exist_ok=True)
        tmp = work / ".tmp.hidden.safetensors"
        mx.save_safetensors(str(tmp), {f"h{j}": h for j, h in enumerate(hs)})
        os.replace(tmp, work / "hidden.safetensors")
        prog["next_block"] = i + 1
        tmpp = prog_path.with_name(".tmp." + prog_path.name)
        tmpp.write_text(json.dumps(prog))
        os.replace(tmpp, prog_path)
        if log:
            log(f"hessians: block {i + 1}/{len(blocks)} {time.perf_counter() - tb:.1f} s"
                + (f" ({len(want_dense)} dense, {len(want_moe)} MoE)" if hooks is not None else ""))
    del model, hs
    shutil.rmtree(work, ignore_errors=True)
    return {"files": sorted(set(prog["files"])), "seconds": time.perf_counter() - t0, "blocks": len(blocks),
            "peak_memory_bytes": int(mx.get_peak_memory())}


class HessianStore(Mapping):
    """Read-only {tensor name without ".weight": E[x xᵀ] (float32 mx.array)} over the per-block files (module
    docstring). Files are opened lazily, one block at a time (the last one cached)."""

    def __init__(self, files: Sequence[str | Path], min_tokens: int = 0):
        self._files: dict[int, Path] = {}
        for f in files:
            f = Path(f)
            self._files[int(f.stem.split("_")[1])] = f
        self.min_tokens = min_tokens
        self._cur: tuple[int, dict, dict, dict] | None = None      # (block, arrays, aliases, counts)

    def _block(self, i: int):
        if self._cur is None or self._cur[0] != i:
            arrays, meta = mx.load(str(self._files[i]), return_metadata=True)
            self._cur = (i, arrays, json.loads(meta.get("aliases", "{}")), {})
        return self._cur

    def _lookup(self, key: str) -> mx.array | None:
        m = _LAYER.match(key)
        if m is None or int(m.group(1)) not in self._files:
            return None
        _, arrays, aliases, cache = self._block(int(m.group(1)))
        if (e := _EXPERT.match(key)) is not None:
            prefix, ex, proj = e.group(1), int(e.group(2)), e.group(3)
            if proj in ("gate_proj", "up_proj"):
                return arrays.get(f"moe.{prefix}.gate_up")
            down = arrays.get(f"moe.{prefix}.down")
            if down is None:
                return None
            cnt = arrays[f"moe.{prefix}.down_count"]
            if int(cnt[ex].item()) < max(self.min_tokens, 1):
                return None
            if ex not in cache:
                cache.clear()
                cache[ex] = down[ex]
            return cache[ex]
        return arrays.get(f"dense.{aliases.get(key, key)}")

    def __getitem__(self, key: str) -> mx.array:
        v = self._lookup(key)
        if v is None:
            raise KeyError(key)
        return v

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._lookup(key) is not None

    def __iter__(self) -> Iterator[str]:
        raise TypeError("HessianStore is not iterable (lookup by tensor name only)")

    def __len__(self) -> int:
        return len(self._files)
