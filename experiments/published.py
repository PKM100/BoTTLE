"""Experiment 2: validate BoTTLE latency predictions against published measurements.

(a) spec-only roofline predictions for every published case;
(b) cross-device transfer: within each source/setting group, calibrate on one
    non-edge (datacenter/desktop) GPU measurement and predict every other device,
    with 'global' and 'overhead' calibration;
(c) software-stack spread: same model+device measured by different sources.

Usage: python experiments/published.py
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bottlebench.analytical import Efficiency, estimate  # noqa: E402
from bottlebench.calibrate import calibrated_estimate  # noqa: E402
from bottlebench.hardware import _REGISTRY, get_hardware  # noqa: E402
from bottlebench.models import get_model  # noqa: E402

OUT = ROOT / "experiments" / "results"


def ape(p: float, m: float) -> float:
    return abs(p - m) / m


def within_2x(p: float, m: float) -> bool:
    return 0.5 <= p / m <= 2.0


def summarize(errs: list[float], ratios: list[float] | None = None) -> dict:
    if not errs:
        return {"n": 0}
    out = {"n": len(errs), "mape": statistics.mean(errs), "median_ape": statistics.median(errs),
           "max_ape": max(errs), "within_25pct": sum(e <= 0.25 for e in errs) / len(errs)}
    if ratios is not None:
        out["within_2x"] = sum(0.5 <= r <= 2.0 for r in ratios) / len(ratios)
    return out


@contextmanager
def host_priors(fn):
    """Temporarily override registry host_speed with fn(hw) -> float."""
    old = {k: h.host_speed for k, h in _REGISTRY.items()}
    try:
        for h in _REGISTRY.values():
            h.host_speed = fn(h)
        yield
    finally:
        for k, v in old.items():
            _REGISTRY[k].host_speed = v


def transfer_rows(groups: dict) -> list[dict]:
    rows = []
    for g, cs in groups.items():
        if len(cs) < 2:
            continue
        for ref in cs:
            if get_hardware(ref["hw"]).kind == "edge":
                continue
            prof = {"model": ref["model"], "hardware": ref["hw"], "precision": ref["precision"],
                    "stages": {"_total": {"time_s": ref["total_ms"] / 1e3}}}
            for tgt in cs:
                if tgt is ref or tgt["model"] != ref["model"]:
                    continue
                model, hw = get_model(tgt["model"]), get_hardware(tgt["hw"])
                row = {"group": g, "model": tgt["model"], "ref": ref["hw"], "target": tgt["hw"],
                       "edge_target": hw.kind == "edge", "measured_ms": tgt["total_ms"],
                       "naive_same_ms": ref["total_ms"]}
                row["spec_ms"] = estimate(model, hw, tgt["precision"]).latency_s * 1e3
                for mode in ("global", "overhead"):
                    row[f"{mode}_ms"] = calibrated_estimate(model, hw, prof, mode=mode).latency_s * 1e3
                # constant additive residual (= overhead mode with all host priors equal)
                with host_priors(lambda h: 1.0):
                    row["additive_ms"] = calibrated_estimate(model, hw, prof, mode="overhead").latency_s * 1e3
                for k in PREDICTORS:
                    row[f"{k}_ape"] = ape(row[f"{k}_ms"], tgt["total_ms"])
                rows.append(row)
    return rows


PREDICTORS = ("naive_same", "spec", "global", "additive", "overhead")


def group_bootstrap(rows: list[dict], key: str, base: str | None = None, n: int = 5000,
                    seed: int = 0) -> dict:
    """95% CI of MAPE (or MAPE difference key-base) resampling source groups, not pairs."""
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    gs = list(by)
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [r for g in (rng.choice(gs) for _ in gs) for r in by[g]]
        v = statistics.mean(r[f"{key}_ape"] for r in sample)
        if base:
            v -= statistics.mean(r[f"{base}_ape"] for r in sample)
        stats.append(v)
    stats.sort()
    return {"lo": stats[int(0.025 * n)], "hi": stats[int(0.975 * n)],
            "p_le_0": sum(s <= 0 for s in stats) / n if base else None}


def main() -> None:
    cases = json.loads((ROOT / "experiments/data/published_latency.json").read_text())["cases"]

    spec_rows = []
    for c in cases:
        model, hw = get_model(c["model"]), get_hardware(c["hw"])
        e = estimate(model, hw, c["precision"])
        ideal = estimate(model, hw, c["precision"], Efficiency(1.0, 1.0))
        m = c["total_ms"] / 1e3
        row = {"id": c["id"], "model": c["model"], "hw": c["hw"], "edge": hw.kind == "edge",
               "measured_ms": c["total_ms"], "spec_ms": e.latency_s * 1e3, "ideal_ms": ideal.latency_s * 1e3,
               "spec_ape": ape(e.latency_s, m), "spec_ratio": e.latency_s / m}
        if c.get("stages_ms"):
            st = {s.name: s.time_s * 1e3 for s in e.stages}
            st["backbone"] = sum(v for k, v in st.items() if k in ("vision", "vision_dino", "vlm"))
            row["stages"] = {k: {"measured_ms": v, "spec_ms": st.get(k)} for k, v in c["stages_ms"].items()}
        spec_rows.append(row)

    groups = defaultdict(list)
    for c in cases:
        groups[c["group"]].append(c)

    xfer_rows = transfer_rows(groups)

    # host-prior sensitivity, recomputed from the registry
    edge = lambda h: h.kind == "edge"  # noqa: E731
    variants = {
        "registry": lambda h: h.host_speed,
        "all_equal": lambda h: 1.0,
        "edge_halved_gap": lambda h: (1 + h.host_speed) / 2 if edge(h) else h.host_speed,
        "edge_x0.75": lambda h: h.host_speed * 0.75 if edge(h) else h.host_speed,
        "edge_x1.25": lambda h: min(h.host_speed * 1.25, 1.0) if edge(h) else h.host_speed,
    }
    sens = {}
    for name, fn in variants.items():
        with host_priors(fn):
            rs = transfer_rows(groups)
        sens[name] = {"all": statistics.mean(r["overhead_ape"] for r in rs),
                      "edge": statistics.mean(r["overhead_ape"] for r in rs if r["edge_target"])}
    priors = {k: h.host_speed for k, h in _REGISTRY.items()}

    per_group = defaultdict(list)
    for r in xfer_rows:
        per_group[r["group"]].append(r)
    group_table = {g: {"n": len(rs), **{k: statistics.mean(r[f"{k}_ape"] for r in rs) for k in PREDICTORS}}
                   for g, rs in per_group.items()}

    spread = defaultdict(list)
    for c in cases:
        spread[(c["model"], c["hw"])].append((c["total_ms"], c["source"]))
    spread_rows = [{"model": k[0], "hw": k[1], "values_ms": [v for v, _ in vs],
                    "max_over_min": max(v for v, _ in vs) / min(v for v, _ in vs)}
                   for k, vs in spread.items() if len(vs) > 1]

    result = {
        "spec_only": {"all": summarize([r["spec_ape"] for r in spec_rows], [r["spec_ratio"] for r in spec_rows]),
                      "edge": summarize([r["spec_ape"] for r in spec_rows if r["edge"]],
                                        [r["spec_ratio"] for r in spec_rows if r["edge"]]),
                      "min_ratio": min(r["spec_ratio"] for r in spec_rows),
                      "max_ratio": max(r["spec_ratio"] for r in spec_rows),
                      "geo_mean_ratio": math.exp(statistics.mean(math.log(r["spec_ratio"]) for r in spec_rows)),
                      "rows": spec_rows},
        "transfer": {k: {"all": {**summarize([r[f"{k}_ape"] for r in xfer_rows],
                                              [r[f"{k}_ms"] / r["measured_ms"] for r in xfer_rows]),
                                 "ci95": group_bootstrap(xfer_rows, k)},
                         "edge": summarize([r[f"{k}_ape"] for r in xfer_rows if r["edge_target"]],
                                           [r[f"{k}_ms"] / r["measured_ms"] for r in xfer_rows
                                            if r["edge_target"]])}
                     for k in PREDICTORS},
        "n_groups": len(per_group),
        "diff_ci95": {f"overhead-{b}": group_bootstrap(xfer_rows, "overhead", b)
                      for b in ("global", "naive_same", "spec", "additive")},
        "per_group": group_table,
        "group_macro_mape": {k: statistics.mean(v[k] for v in group_table.values()) for k in PREDICTORS},
        "host_prior_sensitivity": sens,
        "host_priors": priors,
        "transfer_rows": xfer_rows,
        "stack_spread": spread_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "published.json").write_text(json.dumps(result, indent=2))

    print(f"Spec-only ({len(spec_rows)} cases): {result['spec_only']['all']}")
    print(f"  edge: {result['spec_only']['edge']}  geo-mean pred/meas={result['spec_only']['geo_mean_ratio']:.2f}")
    for r in spec_rows:
        print(f"  {r['id']:24s} meas {r['measured_ms']:7.1f}  spec {r['spec_ms']:7.1f}  ideal {r['ideal_ms']:6.1f}"
              + (f"  stages {r['stages']}" if 'stages' in r else ""))
    print(f"\nTransfer ({len(xfer_rows)} ref->target pairs):")
    for k, v in result["transfer"].items():
        a, ci = v["all"], v["all"]["ci95"]
        print(f"  {k:11s} all {a['mape']*100:6.1f}% [{ci['lo']*100:.1f},{ci['hi']*100:.1f}] "
              f"med {a['median_ape']*100:5.1f} max {a['max_ape']*100:6.1f} <=25% {a['within_25pct']:.2f} "
              f"2x {a['within_2x']:.2f}   edge {v['edge'].get('mape', 0)*100:6.1f}% (n={v['edge']['n']})")
    print(f"  groups={result['n_groups']}  diff CIs: {json.dumps(result['diff_ci95'])}")
    print("  group-macro MAPE:", {k: round(v * 100, 1) for k, v in result["group_macro_mape"].items()})
    print("  host-prior sensitivity:", {k: (round(v['all']*100, 1), round(v['edge']*100, 1)) for k, v in sens.items()})
    for g, v in group_table.items():
        print(f"  {g:18s} n={v['n']:2d} " + " ".join(f"{k}={v[k]*100:5.1f}" for k in PREDICTORS))
    for r in xfer_rows:
        print(f"  {r['model']:12s} {r['ref']:10s}->{r['target']:22s} meas {r['measured_ms']:6.1f} "
              f"spec {r['spec_ms']:6.1f} glob {r['global_ms']:6.1f} ovh {r['overhead_ms']:6.1f}")
    print("\nSoftware-stack spread (same model+device, different sources):")
    for r in spread_rows:
        print(f"  {r['model']:12s} {r['hw']:18s} {r['values_ms']}  max/min {r['max_over_min']:.2f}")


if __name__ == "__main__":
    main()
