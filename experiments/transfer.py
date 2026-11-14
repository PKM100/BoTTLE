"""Experiment 1: cross-configuration latency transfer on locally available devices.

Profiles every model on each (device, precision) configuration, then predicts each
configuration's latency from (a) roofline only and (b) roofline calibrated on every
*other* configuration, and reports absolute percentage error vs measurement.

Usage: python experiments/transfer.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bottlebench.analytical import Efficiency, estimate  # noqa: E402
from bottlebench.calibrate import calibrated_estimate  # noqa: E402
from bottlebench.hardware import get_hardware, load_hw_file  # noqa: E402
from bottlebench.models import get_model, list_models  # noqa: E402
from bottlebench.profiler import profile  # noqa: E402

OUT = ROOT / "experiments" / "results"
HW_DIR = ROOT / "experiments" / "hw"

# (config id, hardware name, torch device, precision)
CONFIGS = [
    ("cpu-fp32", "m5-cpu", "cpu", "fp32"),
    ("gpu-fp32", "m5-gpu", "mps", "fp32"),
    ("gpu-fp16", "m5-gpu", "mps", "fp16"),
]
# >1B-param models: profile 4 layers per stage on MPS (16 GB unified memory; avoids swapping)
BIG = {"pi0", "pi05", "openvla", "openvla-oft", "gr00t-n1.5", "gr00t-n1.6", "gr00t-n1.7"}


def stage_sum(e) -> float:
    return sum(s.time_s for s in e.stages)


def run_profiles(models, iters: int, refresh: bool) -> dict:
    profs = {}
    for cid, hw, dev, prec in CONFIGS:
        for m in models:
            path = OUT / "profiles" / f"{m}_{cid}.json"
            if path.exists() and not refresh:
                profs[(cid, m)] = json.loads(path.read_text())
                continue
            max_layers = 2 if dev == "cpu" else (4 if m in BIG else None)
            print(f"profiling {m} on {cid} (max_layers={max_layers})", flush=True)
            p = profile(get_model(m), hw, dev, prec, max_layers, warmup=3, iters=iters, energy_seconds=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(p, indent=2))
            profs[(cid, m)] = p
    return profs


def extrapolation_check(iters: int) -> dict:
    """Layer-extrapolation error: full smolvla vs 2-layer extrapolation on GPU fp16."""
    m = get_model("smolvla")
    full = profile(m, "m5-gpu", "mps", "fp16", None, 3, iters, 0)
    part = profile(m, "m5-gpu", "mps", "fp16", 2, 3, iters, 0)
    return {"full_s": full["latency_s"], "extrapolated_s": part["latency_s"],
            "ape": abs(part["latency_s"] - full["latency_s"]) / full["latency_s"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    for f in HW_DIR.glob("*.json"):
        load_hw_file(f)

    models = [m.name for m in list_models()]
    profs = run_profiles(models, a.iters, a.refresh)
    cfg = {c[0]: c for c in CONFIGS}

    rows = []
    for m in models:
        model = get_model(m)
        for tgt in cfg:
            _, thw, _, tprec = cfg[tgt]
            measured = profs[(tgt, m)]["latency_s"]
            hw = get_hardware(thw)
            preds = {
                "roofline_ideal": stage_sum(estimate(model, hw, tprec, Efficiency(1.0, 1.0))),
                "roofline_default": stage_sum(estimate(model, hw, tprec)),
            }
            for ref in cfg:
                if ref == tgt:
                    continue
                for mode in ("global", "stage", "overhead"):
                    e = calibrated_estimate(model, hw, profs[(ref, m)], precision=tprec, mode=mode)
                    preds[f"cal_{mode}<-{ref}"] = stage_sum(e)
            for k, v in preds.items():
                rows.append({"model": m, "target": tgt, "predictor": k, "pred_s": v,
                             "measured_s": measured, "ape": abs(v - measured) / measured})

    def summarize(rows, key):
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r["ape"])
        return {k: {"mape": statistics.mean(v), "median_ape": statistics.median(v), "n": len(v)}
                for k, v in sorted(groups.items())}

    family = lambda r: r["predictor"].split("<-")[0]  # noqa: E731
    pairs = {}
    for ref, tgt in permutations(cfg, 2):
        sub = [r for r in rows if r["target"] == tgt and r["predictor"].endswith(f"<-{ref}")]
        for mode in ("global", "stage", "overhead"):
            v = [r["ape"] for r in sub if r["predictor"].startswith(f"cal_{mode}")]
            pairs[f"{ref}->{tgt}:{mode}"] = statistics.mean(v)

    result = {
        "configs": CONFIGS,
        "rows": rows,
        "by_predictor_family": summarize(rows, family),
        "by_pair": pairs,
        "measured": {f"{m}|{c}": profs[(c, m)]["latency_s"] for c in cfg for m in models},
        "extrapolation_check": extrapolation_check(a.iters),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transfer.json").write_text(json.dumps(result, indent=2))

    print("\nMAPE by predictor family:")
    for k, v in result["by_predictor_family"].items():
        print(f"  {k:20s} {v['mape']*100:7.1f}%  (median {v['median_ape']*100:.1f}%, n={v['n']})")
    print("\nMAPE by transfer pair:")
    for k, v in pairs.items():
        print(f"  {k:32s} {v*100:7.1f}%")
    print("\nmeasured latency (s):")
    for k, v in result["measured"].items():
        print(f"  {k:28s} {v*1e3:9.1f} ms")
    print(f"\nlayer-extrapolation check: {result['extrapolation_check']}")


if __name__ == "__main__":
    main()
