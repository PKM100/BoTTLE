"""Experiment 3: cost per successful LIBERO task across VLAs x deployment devices.

Latency per (model, device): the published measurement from the anchor's source when one
exists, otherwise BoTTLE's overhead-mode calibration anchored on one published eager
datacenter/desktop measurement per model (Experiment 2 picks this predictor). Success rates are
the published LIBERO per-suite numbers (SUCCESS_SRC). A Monte Carlo propagates latency-prediction
error (E2 residuals), success-rate sampling error, assumed horizons, episode lengths, power and
prices into cost per success, and reports how often each configuration is cheapest. Also checks
predicted energy per inference against published Jetson Thor measurements (arXiv 2611.06710).

Usage: python experiments/libero_cost.py
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bottlebench.analytical import Efficiency, estimate  # noqa: E402
from bottlebench.calibrate import calibrated_estimate  # noqa: E402
from bottlebench.cost import TaskSpec, task_cost  # noqa: E402
from bottlebench.hardware import get_hardware  # noqa: E402
from bottlebench.models import get_model  # noqa: E402

OUT = ROOT / "experiments" / "results"

SUITES = {"spatial": 220, "object": 280, "goal": 300, "long": 520}  # max steps; +10 wait steps
WAIT = 10
HZ = 20.0

# Published LIBERO success (spatial, object, goal, long). See paper Table for sources.
SUCCESS = {
    "openvla": (0.847, 0.884, 0.792, 0.537),
    "openvla-oft": (0.976, 0.984, 0.979, 0.945),
    "pi0": (0.968, 0.988, 0.958, 0.852),
    "pi05": (0.988, 0.982, 0.980, 0.924),
    "smolvla": (0.90, 0.96, 0.92, 0.71),
    "gr00t-n1.6": (0.9765, 0.9845, 0.975, 0.9435),
}
SUCCESS_SRC = {  # source, trials per suite
    "openvla": ("OpenVLA-OFT arXiv 2502.19645 Table I (fine-tuned OpenVLA)", 500),
    "openvla-oft": ("OpenVLA-OFT arXiv 2502.19645 Table I", 500),
    "pi0": ("OpenVLA-OFT arXiv 2502.19645 Table I (pi0 fine-tuned)", 500),
    "pi05": ("openpi examples/libero/README.md (replan_steps=5)", 500),
    "smolvla": ("SmolVLA arXiv 2506.01844 LIBERO table (0.45B)", 500),
    "gr00t-n1.6": ("Isaac-GR00T n1d6 examples/LIBERO/README.md", 200),
}
# Actions executed per policy call in the published LIBERO evaluations. SmolVLA is assumed and
# swept over ASSUMED_H in the Monte Carlo; GR00T uses --n_action_steps 8 in its LIBERO eval.
EXEC_H = {"openvla": 1, "openvla-oft": 8, "pi0": 5, "pi05": 5, "smolvla": 10, "gr00t-n1.6": 8}
ASSUMED_H = {"smolvla": (5, 10, 25, 50)}
# One published *eager PyTorch* measurement per model on a non-edge GPU (calibration anchor):
# (device, ms, source group in published_latency.json). pi0 has no published eager measurement;
# it is anchored on the eager pi0.5 measurement scaled by the roofline ratio (same backbone and
# action expert). PI0_JAX is the compiled JAX on-board measurement, reported as a sensitivity.
ANCHOR = {
    "openvla": ("h100-nvl", 103.8, "where-openvla"), "openvla-oft": ("h100-nvl", 42.4, "where-oft"),
    "pi0": ("h100-nvl", None, "where-pi05"), "pi05": ("h100-nvl", 111.6, "where-pi05"),
    "smolvla": ("h100-nvl", 208.3, "where-smolvla"), "gr00t-n1.6": ("h100-sxm", 77.0, "gr00t-n1.6-repo"),
}
PI0_JAX = ("rtx-4090", 73.0)
DEVICES = ["h100-sxm", "a10g", "l4", "t4", "rtx-4090", "rtx-5090",
           "jetson-agx-thor", "jetson-agx-orin-64gb", "jetson-orin-nano-super"]
SCENARIOS = {  # robot-side costs
    "compute-only": dict(robot_usd_per_hr=0.0, robot_power_w=0.0),
    "robot": dict(robot_usd_per_hr=5.0, robot_power_w=150.0),
    "robot+operator": dict(robot_usd_per_hr=35.0, robot_power_w=150.0),
}
SUCC_FRAC = 0.6  # successful episodes end at this fraction of max steps (sensitivity below)
THOR_ENERGY_J = {"openvla": 20.93, "openvla-oft": 14.32, "pi05": 9.05, "gr00t-n1.7": 6.90, "smolvla": 7.35}
THOR_LAT_MS = {"openvla": 445, "openvla-oft": 181, "pi05": 208, "gr00t-n1.7": 149, "smolvla": 308}


_CASES = json.loads((ROOT / "experiments/data/published_latency.json").read_text())["cases"]
MEASURED = {(c["model"], c["hw"]): c["total_ms"] for c in _CASES
            for m, (_, _, g) in ANCHOR.items() if c["group"] == g and c["model"] == m}
CLOUD_CLASS = ("h100-sxm", "a10g", "l4", "t4")  # server parts: may sit behind a network hop


def anchor_ms(model: str) -> tuple[str, float]:
    hw, ms, _ = ANCHOR[model]
    if model == "pi0" and PI0_ANCHOR == "jax":
        return PI0_JAX
    if ms is None:  # pi0 eager proxy
        h = get_hardware(hw)
        ratio = estimate(get_model("pi0"), h, "bf16").latency_s / estimate(get_model("pi05"), h, "bf16").latency_s
        ms = ANCHOR["pi05"][1] * ratio
    return hw, ms


PI0_ANCHOR = "eager"


def latency_est(model: str, device: str, eff: Efficiency | None = None, use_measured: bool = True):
    """Returns (estimate, measured?)."""
    hw, ms = anchor_ms(model)
    prof = {"model": model, "hardware": hw, "precision": "bf16", "stages": {"_total": {"time_s": ms / 1e3}}}
    est = calibrated_estimate(get_model(model), get_hardware(device), prof, eff, mode="overhead")
    meas = MEASURED.get((model, device)) if use_measured and not (model == "pi0" and PI0_ANCHOR == "jax") else None
    if meas:
        est = replace(est, latency_s=meas / 1e3, energy_j=meas / 1e3 * est.avg_power_w)
    return est, bool(meas)


def expected_steps(p: float, max_steps: int, frac: float) -> int:
    return round(WAIT + p * frac * max_steps + (1 - p) * max_steps)


def kendall_tau(a: list, b: list) -> float:
    pairs = list(itertools.combinations(range(len(a)), 2))
    s = sum((1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1) for i, j in pairs if a[i] != a[j] and b[i] != b[j])
    return s / len(pairs)


def run(frac: float = SUCC_FRAC) -> list[dict]:
    rows = []
    for model, dev, (suite, max_steps), mode, (scen, sk) in itertools.product(
            SUCCESS, DEVICES, SUITES.items(), ("sync", "async"), SCENARIOS.items()):
        est, measured = latency_est(model, dev)
        p = SUCCESS[model][list(SUITES).index(suite)]
        spec = get_model(model)
        task = TaskSpec(episode_steps=expected_steps(p, max_steps, frac), control_hz=HZ,
                        exec_horizon=EXEC_H[model], success_rate=p, mode=mode, **sk)
        hw = get_hardware(dev)
        assert hw.price_usd > 0, f"{dev} needs a purchase price for 'own' pricing"
        c = task_cost(est, hw, task, spec.action_chunk)
        rows.append({"model": model, "device": dev, "suite": suite, "mode": mode, "scenario": scen,
                     "latency_ms": c.latency_ms, "latency_measured": measured, "success": p, "realtime_ok": c.realtime_ok, "fits": c.fits,
                     "time_per_success_s": c.time_per_success_s,
                     "energy_per_success_kj": c.energy_per_success_j / 1e3,
                     "usd_per_1k_success": c.usd_per_success * 1e3})
    return rows


def monte_carlo(n: int = 2000, seed: int = 0) -> dict:
    """Propagate uncertainty into $/1k successes (sync) and count how often each config wins."""
    rng = random.Random(seed)
    pub = json.loads((OUT / "published.json").read_text())
    resid = [math.log(r["measured_ms"] / r["overhead_ms"]) for r in pub["transfer_rows"]]
    base = {(m, d): latency_est(m, d) for m in SUCCESS for d in DEVICES}
    models, devs, scens = list(SUCCESS), DEVICES, list(SCENARIOS)
    wins_model = {s: {d: dict.fromkeys(models, 0) for d in devs} for s in scens}
    wins_dev = {s: {m: dict.fromkeys(devs, 0) for m in models} for s in scens}
    samples = {s: {(m, d): [] for m in models for d in devs} for s in scens}
    for _ in range(n):
        frac = rng.uniform(0.4, 0.8)
        kwh, life, duty = rng.uniform(0.08, 0.30), rng.uniform(2, 5), rng.uniform(0.25, 0.75)
        robot_hr, oper_hr, util = rng.uniform(2.5, 10), rng.uniform(20, 50), rng.uniform(0.2, 1.0)
        rtt = rng.uniform(0, 0.030)
        horizon = {m: rng.choice(ASSUMED_H[m]) if m in ASSUMED_H else EXEC_H[m] for m in models}
        succ = {m: [rng.betavariate(p * SUCCESS_SRC[m][1] + 1, (1 - p) * SUCCESS_SRC[m][1] + 1)
                    for p in SUCCESS[m]] for m in models}
        scen_kw = {"compute-only": dict(robot_usd_per_hr=0.0, robot_power_w=0.0),
                   "robot": dict(robot_usd_per_hr=robot_hr, robot_power_w=150.0),
                   "robot+operator": dict(robot_usd_per_hr=robot_hr + oper_hr, robot_power_w=150.0)}
        cost = {s: {} for s in scens}
        for (m, d), (est, measured) in base.items():
            if not est.fits:
                continue
            hw = get_hardware(d)
            mult = math.exp(rng.gauss(0, 0.15)) if measured else math.exp(rng.choice(resid))
            lat = est.latency_s * mult + (rtt if d in CLOUD_CLASS else 0.0)
            power = hw.idle_w + util * (hw.tdp_w - hw.idle_w)
            e = replace(est, latency_s=lat, avg_power_w=power, energy_j=lat * power)
            for s in scens:
                vals = []
                for (suite, max_steps), p in zip(SUITES.items(), succ[m]):
                    t = TaskSpec(episode_steps=expected_steps(p, max_steps, frac), control_hz=HZ,
                                 exec_horizon=horizon[m], success_rate=p, mode="sync", usd_per_kwh=kwh,
                                 lifetime_years=life, duty_cycle=duty, **scen_kw[s])
                    vals.append(task_cost(e, hw, t, get_model(m).action_chunk).usd_per_success * 1e3)
                cost[s][(m, d)] = statistics.mean(vals)
                samples[s][(m, d)].append(cost[s][(m, d)])
        for s in scens:
            for d in devs:
                cand = {m: cost[s][(m, d)] for m in models if (m, d) in cost[s]}
                if cand:
                    wins_model[s][d][min(cand, key=cand.get)] += 1
            for m in models:
                cand = {d: cost[s][(m, d)] for d in devs if (m, d) in cost[s]}
                wins_dev[s][m][min(cand, key=cand.get)] += 1

    def q(v, f):
        v = sorted(v)
        return v[min(int(f * len(v)), len(v) - 1)]
    return {
        "n": n,
        "p_cheapest_model": {s: {d: {m: c / n for m, c in w.items() if c} for d, w in v.items()}
                             for s, v in wins_model.items()},
        "p_cheapest_device": {s: {m: {d: c / n for d, c in w.items() if c} for m, w in v.items()}
                              for s, v in wins_dev.items()},
        "usd_per_1k": {s: {f"{m}|{d}": [q(v, 0.05), q(v, 0.5), q(v, 0.95)] for (m, d), v in sv.items() if v}
                       for s, sv in samples.items()},
    }


def mean_over_suites(rows, **f):
    sel = [r for r in rows if all(r[k] == v for k, v in f.items())]
    return {k: statistics.mean(r[k] for r in sel) for k in ("latency_ms", "time_per_success_s",
                                                             "energy_per_success_kj", "usd_per_1k_success", "success")}


def main() -> None:
    rows = run()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "libero_cost.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    # 1) Ranking: latency vs cost per success (per device, sync, robot scenario, averaged over suites)
    rank = {}
    for dev in DEVICES:
        agg = {m: mean_over_suites(rows, model=m, device=dev, mode="sync", scenario="robot") for m in SUCCESS}
        lat = [agg[m]["latency_ms"] for m in SUCCESS]
        usd = [agg[m]["usd_per_1k_success"] for m in SUCCESS]
        rank[dev] = {"tau_latency_vs_usd": kendall_tau(lat, usd),
                     "fastest": min(agg, key=lambda m: agg[m]["latency_ms"]),
                     "cheapest": min(agg, key=lambda m: agg[m]["usd_per_1k_success"]),
                     "per_model": agg}
    summary["ranking"] = rank

    # 2) Device choice per model & scenario (sync and async), averaged over suites
    best = {}
    for m, scen, mode in itertools.product(SUCCESS, SCENARIOS, ("sync", "async")):
        agg = {d: mean_over_suites(rows, model=m, device=d, mode=mode, scenario=scen) for d in DEVICES}
        ok = [d for d in DEVICES if all(r["fits"] for r in rows if r["model"] == m and r["device"] == d)]
        best[f"{m}|{scen}|{mode}"] = min(ok, key=lambda d: agg[d]["usd_per_1k_success"])
    summary["cheapest_device"] = best

    # 3) Real-time feasibility (async) per model x device
    summary["realtime"] = {m: {d: next(r["realtime_ok"] for r in rows if r["model"] == m and r["device"] == d
                                       and r["mode"] == "async") for d in DEVICES} for m in SUCCESS}

    # 4) Success-rate leverage: share of cost-per-success spread explained by success vs latency
    #    (LIBERO-Long vs Spatial for each model on Thor, sync, robot)
    lev = {}
    for m in SUCCESS:
        a = mean_over_suites(rows, model=m, device="jetson-agx-thor", mode="sync", scenario="robot", suite="long")
        b = mean_over_suites(rows, model=m, device="jetson-agx-thor", mode="sync", scenario="robot", suite="spatial")
        lev[m] = a["usd_per_1k_success"] / b["usd_per_1k_success"]
    summary["long_over_spatial_cost_ratio_thor"] = lev

    # 5) Sensitivity to successful-episode length fraction
    sens = {}
    for frac in (0.4, 0.6, 0.8, 1.0):
        rs = run(frac)
        a = {m: mean_over_suites(rs, model=m, device="jetson-agx-thor", mode="sync", scenario="robot")["usd_per_1k_success"]
             for m in SUCCESS}
        sens[frac] = {"cheapest": min(a, key=a.get), "order": sorted(a, key=a.get)}
    summary["succ_frac_sensitivity"] = sens

    # 6) Energy per inference on Thor vs published (default power model and leave-one-out util)
    thor = get_hardware("jetson-agx-thor")
    energy = []
    for m, j in THOR_ENERGY_J.items():
        lat = THOR_LAT_MS[m] / 1e3
        default_j = lat * (thor.idle_w + Efficiency().power_util * (thor.tdp_w - thor.idle_w))
        others = [(THOR_ENERGY_J[o] / (THOR_LAT_MS[o] / 1e3) - thor.idle_w) / (thor.tdp_w - thor.idle_w)
                  for o in THOR_ENERGY_J if o != m]
        loo_j = lat * (thor.idle_w + statistics.mean(others) * (thor.tdp_w - thor.idle_w))
        energy.append({"model": m, "measured_j": j, "implied_w": j / lat, "default_j": default_j, "loo_j": loo_j})
    summary["thor_energy"] = {
        "rows": energy,
        "default_mape": statistics.mean(abs(e["default_j"] - e["measured_j"]) / e["measured_j"] for e in energy),
        "loo_mape": statistics.mean(abs(e["loo_j"] - e["measured_j"]) / e["measured_j"] for e in energy),
    }
    # 7) pi0 anchored on the compiled JAX measurement instead (software-stack sensitivity)
    global PI0_ANCHOR
    PI0_ANCHOR = "jax"
    jax_rows = run()
    PI0_ANCHOR = "eager"
    summary["pi0_jax_sensitivity"] = {
        d: {"pi0_latency_ms": mean_over_suites(jax_rows, model="pi0", device=d, mode="sync", scenario="robot")["latency_ms"],
            "fastest": min(SUCCESS, key=lambda m: mean_over_suites(jax_rows, model=m, device=d, mode="sync",
                                                                  scenario="robot")["latency_ms"]),
            "cheapest": min(SUCCESS, key=lambda m: mean_over_suites(jax_rows, model=m, device=d, mode="sync",
                                                                   scenario="robot")["usd_per_1k_success"])}
        for d in DEVICES}
    summary["anchors"] = {m: anchor_ms(m) for m in SUCCESS}
    summary["measured_cells"] = [f"{m}|{d}" for (m, d) in MEASURED if d in DEVICES]
    summary["success_sources"] = SUCCESS_SRC
    summary["mc"] = monte_carlo()
    (OUT / "libero_cost.json").write_text(json.dumps(summary, indent=2, default=str))

    print("Anchors:", {m: (h, round(v, 1)) for m, (h, v) in summary["anchors"].items()})
    print("Measured cells:", summary["measured_cells"])
    print("Latency (ms; * = published measurement, else overhead-calibrated prediction):")
    print("  " + " ".join(f"{d[:12]:>12s}" for d in DEVICES))
    for m in SUCCESS:
        print(f"{m:12s}" + " ".join(
            f"{latency_est(m, d)[0].latency_s*1e3:11.0f}{'*' if latency_est(m, d)[1] else ' '}" for d in DEVICES))
    print("\n$ per 1k successes (sync, robot scenario, mean over suites):")
    for m in SUCCESS:
        print(f"{m:12s}" + " ".join(f"{rank[d]['per_model'][m]['usd_per_1k_success']:12.2f}" for d in DEVICES))
    print("\nKendall tau(latency, $/success) per device:",
          {d: round(v["tau_latency_vs_usd"], 2) for d, v in rank.items()})
    print("fastest vs cheapest:", {d: (v["fastest"], v["cheapest"]) for d, v in rank.items()})
    print("\nCheapest device:")
    for k, v in best.items():
        print(f"  {k:40s} {v}")
    print("\nAsync real-time feasible:")
    for m, d in summary["realtime"].items():
        print(f"  {m:12s}", [k for k, ok in d.items() if ok])
    print("\nLong/Spatial cost ratio on Thor:", {k: round(v, 2) for k, v in lev.items()})
    print("Success-length sensitivity:", sens)
    print("\npi0 JAX-anchor sensitivity:", summary["pi0_jax_sensitivity"])
    mc = summary["mc"]
    print("\nMonte Carlo P(cheapest model | device):")
    for sc, v in mc["p_cheapest_model"].items():
        print(f"  {sc}: " + "; ".join(f"{d}: " + ",".join(f"{m}={p:.2f}" for m, p in sorted(w.items(), key=lambda x: -x[1]))
                                         for d, w in v.items()))
    print("Monte Carlo P(cheapest device | model):")
    for sc, v in mc["p_cheapest_device"].items():
        print(f"  {sc}: " + "; ".join(f"{m}: " + ",".join(f"{d}={p:.2f}" for d, p in sorted(w.items(), key=lambda x: -x[1]))
                                         for m, w in v.items()))
    print("\nThor energy/inference: default MAPE %.0f%%, leave-one-out MAPE %.0f%%" %
          (summary["thor_energy"]["default_mape"] * 100, summary["thor_energy"]["loo_mape"] * 100))
    for e in energy:
        print(f"  {e['model']:12s} meas {e['measured_j']:5.1f} J ({e['implied_w']:.0f} W)  default {e['default_j']:5.1f}  loo {e['loo_j']:5.1f}")


if __name__ == "__main__":
    main()
