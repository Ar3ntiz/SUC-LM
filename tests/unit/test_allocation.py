"""Tests for suc.allocation.greedy (Phase 7 first concrete allocator, spec §14 / §39)."""
import itertools

import mlx.core as mx
import numpy as np
import pytest

from suc.allocation.greedy import (Candidate, budget_for_bpp, build_candidates, greedy_allocate, lower_hull,
                                   plan_summary, uniform_plan)
from suc.quant.codecs import AffineRTN


def C(bits: int, loss: float, codec: str | None = None) -> Candidate:
    return Candidate(codec or f"c{bits}", bits, loss)


def brute_force(cands: dict[str, list[Candidate]], budget: int) -> float:
    names = sorted(cands)
    best = float("inf")
    for combo in itertools.product(*(cands[n] for n in names)):
        if sum(c.bits for c in combo) <= budget:
            best = min(best, sum(c.loss for c in combo))
    return best


def plan_loss(plan) -> float:
    return sum(c.loss for c in plan.values())


def plan_bits(plan) -> int:
    return sum(c.bits for c in plan.values())


# ------------------------------------------------------------------------------------- lower_hull
def test_lower_hull_removes_dominated_and_sorts():
    pts = [C(40, 1.5), C(10, 5.0), C(30, 2.0), C(35, 2.5), C(20, 3.0), C(25, 3.0)]
    h = lower_hull(pts)
    assert [c.bits for c in h] == [10, 20, 30, 40]  # 35 (more bits, more loss) and 25 (same loss) dominated


def test_lower_hull_removes_non_convex_middle_point():
    # (20, 4.5) lies above the chord (10,5)-(30,1): marginal value 0.05 then 0.175 -> not convex
    h = lower_hull([C(10, 5.0), C(20, 4.5), C(30, 1.0)])
    assert [c.bits for c in h] == [10, 30]


def test_lower_hull_equal_bits_keeps_lower_loss_and_single_point():
    h = lower_hull([C(10, 5.0, "a"), C(10, 4.0, "b")])
    assert [c.codec for c in h] == ["b"]
    assert lower_hull([C(7, 1.0)]) == [C(7, 1.0)]


def test_lower_hull_marginal_values_decrease():
    rng = np.random.default_rng(0)
    for _ in range(50):
        pts = [C(int(b), float(l)) for b, l in zip(rng.choice(1000, 8, replace=False), rng.random(8))]
        h = lower_hull(pts)
        slopes = [(a.loss - b.loss) / (b.bits - a.bits) for a, b in zip(h, h[1:])]
        assert all(s > 0 for s in slopes)
        assert all(s1 >= s2 for s1, s2 in zip(slopes, slopes[1:]))
        assert h[0].bits == min(p.bits for p in pts)
        assert h[-1].loss == min(p.loss for p in pts)


# -------------------------------------------------------------------------------- greedy_allocate
def _convex_equal_steps(rng, n_tensors: int, n_cands: int = 3, step: int = 100) -> dict[str, list[Candidate]]:
    """Equal bit steps for every tensor and convex losses: greedy is exactly optimal (exchange argument)."""
    out = {}
    for t in range(n_tensors):
        drops = np.sort(rng.random(n_cands - 1) * 10)[::-1]
        loss = [20.0 + rng.random()]
        for d in drops:
            loss.append(loss[-1] - d)
        out[f"t{t}"] = [C(step * (i + 1), float(l)) for i, l in enumerate(loss)]
    return out


@pytest.mark.parametrize("trial", range(20))
def test_greedy_optimal_on_convex_equal_step_instances(trial):
    rng = np.random.default_rng(trial)
    cands = _convex_equal_steps(rng, n_tensors=int(rng.integers(2, 5)))
    lo = sum(min(c.bits for c in v) for v in cands.values())
    hi = sum(max(c.bits for c in v) for v in cands.values())
    for budget in range(lo, hi + 1, 50):
        plan = greedy_allocate(cands, budget)
        assert plan_bits(plan) <= budget
        assert plan_loss(plan) == pytest.approx(brute_force(cands, budget), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("trial", range(20))
def test_greedy_within_hull_relaxation_bound_on_general_instances(trial):
    rng = np.random.default_rng(100 + trial)
    cands = {f"t{t}": [C(int(b), float(l)) for b, l in zip(rng.choice(np.arange(10, 400), 3, replace=False),
                                                         rng.random(3) * 10)]
             for t in range(int(rng.integers(2, 5)))}
    hulls = {n: lower_hull(v) for n, v in cands.items()}
    max_step = max((a.loss - b.loss for h in hulls.values() for a, b in zip(h, h[1:])), default=0.0)
    lo = sum(h[0].bits for h in hulls.values())
    hi = sum(h[-1].bits for h in hulls.values())
    for budget in np.linspace(lo, hi, 9).astype(int):
        plan = greedy_allocate(cands, int(budget))
        opt = brute_force(cands, int(budget))
        assert plan_bits(plan) <= budget
        assert opt - 1e-9 <= plan_loss(plan) <= opt + max_step + 1e-9


def test_greedy_tighter_budget_never_lowers_loss():
    rng = np.random.default_rng(7)
    cands = {f"t{t}": [C(int(b), float(l)) for b, l in zip(rng.choice(np.arange(10, 500), 4, replace=False),
                                                         rng.random(4))] for t in range(6)}
    lo = sum(min(c.bits for c in v) for v in cands.values())
    losses = [plan_loss(greedy_allocate(cands, b)) for b in range(lo, lo + 2500, 25)]
    assert all(a >= b - 1e-12 for a, b in zip(losses, losses[1:]))


def test_greedy_infeasible_budget_raises_and_min_budget_gives_cheapest():
    cands = {"a": [C(10, 3.0), C(20, 1.0)], "b": [C(5, 2.0), C(15, 0.5)]}
    with pytest.raises(ValueError):
        greedy_allocate(cands, 14)
    plan = greedy_allocate(cands, 15)
    assert plan == {"a": C(10, 3.0), "b": C(5, 2.0)}
    assert greedy_allocate(cands, 10 ** 9) == {"a": C(20, 1.0), "b": C(15, 0.5)}


def test_greedy_picks_largest_marginal_value_first():
    # a: 2.0 loss per 10 bits = 0.2/bit; b: 1.5 per 10 bits = 0.15/bit -> with 10 spare bits a wins
    cands = {"a": [C(10, 3.0), C(20, 1.0)], "b": [C(10, 2.0), C(20, 0.5)]}
    plan = greedy_allocate(cands, 30)
    assert plan["a"].bits == 20 and plan["b"].bits == 10


# ------------------------------------------------------------------------- summary / budget / uniform
def test_plan_summary():
    cands = {"a": [Candidate("x", 10, 3.0, 0.1), Candidate("y", 20, 1.0, 0.01)],
             "b": [Candidate("x", 5, 2.0, 0.2), Candidate("y", 15, 0.5, 0.02)]}
    s = plan_summary({"a": cands["a"][1], "b": cands["b"][0]}, cands)
    assert s["bits_total"] == 25 and s["loss_total"] == pytest.approx(3.0) and s["tensors"] == 2
    assert s["by_codec"] == {"x": 1, "y": 1}
    assert s["rows"][0] == {"tensor": "a", "codec": "y", "bits": 20, "loss": 1.0, "rel_mse": 0.01}
    assert s["bits_min"] == 15 and s["bits_max"] == 35


def test_budget_for_bpp():
    assert budget_for_bpp(2.0, 1000, 500) == 1500
    assert budget_for_bpp(2.5, 3, 0) == 7  # floor: never exceed the target
    with pytest.raises(ValueError):
        budget_for_bpp(1.0, 100, 101)


def test_uniform_plan():
    cands = {"a": [Candidate("x", 10, 3.0), Candidate("y", 20, 1.0)],
             "b": [Candidate("x", 5, 2.0), Candidate("y", 15, 0.5)]}
    assert uniform_plan(cands, "y") == {"a": cands["a"][1], "b": cands["b"][1]}
    with pytest.raises(KeyError):
        uniform_plan(cands, "z")


# ---------------------------------------------------------------------------------- build_candidates
def _sq_loss(name, w, w_hat):
    return mx.sum((w_hat - w) ** 2).item()


def _tensors():
    mx.random.seed(0)
    return {"w0": mx.random.normal((32, 128)), "w1": mx.random.normal((16, 64)) * 3}


def test_build_candidates_bits_up_loss_down():
    ts = _tensors()
    codecs = {"rtn2": AffineRTN(2), "rtn4": AffineRTN(4)}
    out = build_candidates(ts, ["w0", "w1"], codecs, _sq_loss, seed=1)
    assert list(out) == ["w0", "w1"]
    for name, cs in out.items():
        assert [c.codec for c in cs] == ["rtn2", "rtn4"]
        assert cs[0].bits < cs[1].bits and cs[0].loss > cs[1].loss > 0
        e = AffineRTN(4).encode(ts[name], seed=0)
        assert cs[1].bits == e.bits()["total"]
        rel = mx.sum((AffineRTN(4).decode(e) - ts[name]) ** 2).item() / mx.sum(ts[name] ** 2).item()
        assert cs[1].rel_mse == pytest.approx(rel, rel=1e-5)


def test_build_candidates_lazy_callable_matches_mapping():
    ts = _tensors()
    calls = []

    def load(name):
        calls.append(name)
        return ts[name]

    codecs = {"rtn2": AffineRTN(2), "rtn4": AffineRTN(4)}
    eager = build_candidates(ts, ["w0", "w1"], codecs, _sq_loss, seed=1)
    lazy = build_candidates(load, ["w0", "w1"], codecs, _sq_loss, seed=1)
    assert calls == ["w0", "w1"]  # each tensor loaded once
    assert lazy == eager


def test_build_candidates_forwards_importance_to_codecs_that_accept_it():
    from suc.quant.codecs import AffineRTN
    from suc.quant.residual import ResidualCodec

    mx.random.seed(0)
    w = mx.random.normal((32, 64))
    imp = mx.zeros_like(w).at[3, 5].add(1e6)  # one hugely important weight
    codecs = {"res": ResidualCodec(AffineRTN(2), 1 / (32 * 64), "fp16", "importance")}
    loss = lambda n, a, b: float(((b - a) ** 2 * imp).sum())  # noqa: E731
    with_imp = build_candidates({"t": w}, ["t"], codecs, loss, seed=0, importance={"t": imp})["t"][0]
    assert with_imp.loss < 1e-2  # the important entry is kept (only fp16 rounding of its residual remains)
