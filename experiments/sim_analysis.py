"""E4: closed-loop LIBERO with measured / emulated latency (SmolVLA, Colab T4).

Reads experiments/data/libero_sim_runs.json (from notebooks/bottle_libero_sim.ipynb) and writes
experiments/results/sim.json, paper/tables/sim.tex and paper/figs/sim.pdf.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

from bottlebench.calibrate import calibrated_estimate
from bottlebench.hardware import get_hardware
from bottlebench.models import get_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data" / "libero_sim_runs.json"
OUT = ROOT / "experiments" / "results"
FPS, H = 20, 10
# SmolVLA eager anchor used in E3 (arXiv 2611.06710 Table 1, H100 NVL, bf16)
ANCHOR = {"model": "smolvla", "hardware": "h100-nvl", "precision": "bf16", "stages": {"_total": {"time_s": 0.2083}}}
DEVICES = {"rtx-4090": "RTX 4090", "jetson-agx-thor": "Thor", "jetson-agx-orin-64gb": "AGX Orin",
           "jetson-orin-nano-super": "Orin Nano"}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def paired(a: list, b: list, n_boot: int = 10000, seed: int = 0) -> dict | None:
    """Per-task paired difference b - a over tasks present in both: mean, task bootstrap CI, exact sign-flip p."""
    d = [y - x for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 2:
        return None
    mean = statistics.mean(d)
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    flips = [abs(sum(s * x for s, x in zip(signs, d))) for signs in itertools.product((1, -1), repeat=len(d))]
    p = sum(f >= abs(sum(d)) - 1e-12 for f in flips) / len(flips)
    return {"n_tasks": len(d), "mean": mean, "ci": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]), "p": p}


def max_actions(hold: int, budget: int = 280) -> int:
    """Policy actions executed before the wrapper truncates: each chunk is `hold` steps then H actions."""
    full, rem = divmod(budget, hold + H)
    return full * H + min(max(rem - hold, 0), H)


def predict_ms(device: str, precision: str = "bf16") -> float:
    est = calibrated_estimate(get_model("smolvla"), get_hardware(device), ANCHOR, precision=precision, mode="overhead")
    return est.latency_s * 1e3


def main() -> dict:
    runs = json.loads(DATA.read_text())["settings"]
    stock = next(r for r in runs if r["name"] == "stock")
    rows = []
    for r in runs:
        tasks = [i for i, s in enumerate(r["success"]) if s is not None]
        if not tasks:
            continue
        n = r["n_per_task"] * len(tasks)
        k = round(sum(r["success"][i] for i in tasks) * r["n_per_task"])
        if "aggregate" in r:
            k, n = r["aggregate"]["k"], r["aggregate"]["n"]
        inf = [r["infer_ms"][i] for i in tasks]
        lat = statistics.median(inf) if r["mode"] == "measured" else r["latency_ms"]
        stock_tasks = range(10) if "aggregate" in r else tasks
        stock_same = round(sum(stock["success"][i] for i in stock_tasks) * stock["n_per_task"])
        rows.append({
            "name": r["name"], "mode": r["mode"], "latency_ms": lat, "hold_steps": math.ceil(lat * FPS / 1e3 - 1e-9),
            "tasks": tasks, "n": n, "k": k, "success": k / n, "ci": wilson(k, n),
            "stock_same_tasks": stock_same / (stock["n_per_task"] * len(stock_tasks)),
            "paired": None if r is stock else paired(stock["success"], r["success"]),
            "infer_ms_median": statistics.median(inf),
            "max_actions_in_budget": max_actions(math.ceil(lat * FPS / 1e3 - 1e-9)),
        })
    all_inf = [x for r in runs for x in r["infer_ms"] if x is not None]
    t4_meas = statistics.median(all_inf)
    t4 = {"measured_ms": t4_meas, "measured_range": (min(all_inf), max(all_inf)), "n_task_medians": len(all_inf),
          "pred_fp16_ms": predict_ms("t4", "fp16"), "pred_fp32_ms": predict_ms("t4", "fp32")}
    t4["ratio_fp16"] = t4_meas / t4["pred_fp16_ms"]
    t4["ratio_fp32"] = t4_meas / t4["pred_fp32_ms"]
    # E3's sync robot-time-per-success model assumes success is unaffected by stalls; check holds fraction
    devices = {}
    with (OUT / "libero_cost.csv").open() as f:
        for row in csv.DictReader(f):
            if row["model"] == "smolvla" and row["suite"] == "spatial" and row["mode"] == "sync" \
                    and row["scenario"] == "robot" and row["device"] in DEVICES:
                ms, meas = float(row["latency_ms"]), row["latency_measured"] == "True"
                devices[row["device"]] = {"label": DEVICES[row["device"]], "pred_ms": ms, "measured": meas,
                                          "pred_ms_x_t4ratio": ms if meas else ms * t4["ratio_fp16"]}
    res = {"rows": rows, "t4": t4, "devices": devices}
    OUT.mkdir(exist_ok=True)
    (OUT / "sim.json").write_text(json.dumps(res, indent=2))
    for r in rows:
        pr = r["paired"]
        print(f"{r['name']:22s} {r['latency_ms']:7.0f} ms hold {r['hold_steps']:2d}  {r['k']}/{r['n']} = {r['success']:.2f} "
              f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]  stock same tasks {r['stock_same_tasks']:.2f}"
              + (f"  Δ {pr['mean']:+.3f} [{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] p={pr['p']:.3f} ({pr['n_tasks']} tasks)" if pr else ""))
    print(f"T4 measured {t4_meas:.0f} ms (task medians {t4['measured_range'][0]:.0f}-{t4['measured_range'][1]:.0f}, "
          f"n={len(all_inf)}); BoTTLE fp16 {t4['pred_fp16_ms']:.0f} (x{t4['ratio_fp16']:.2f}), "
          f"fp32 {t4['pred_fp32_ms']:.0f} (x{t4['ratio_fp32']:.2f})")
    for d, v in devices.items():
        print(f"  {v['label']:10s} {'published' if v['measured'] else 'predicted'} {v['pred_ms']:.0f} ms; "
              f"if T4-like error {v['pred_ms_x_t4ratio']:.0f} ms")
    write_table(rows)
    write_figure(rows, devices, t4)
    return res


LABEL = {"stock": "Stock (physics paused)", "t4-measured": "T4, measured per call",
         "pred-rtx-4090": "RTX 4090 (predicted)", "pred-jetson-agx-thor": "Thor (published)",
         "sweep-1500ms": "Stress setting"}


def write_table(rows: list[dict]) -> None:
    lines = [r"\begin{tabular}{lrrrrcc}", r"\toprule",
             r"Setting & Latency & Holds & Max actions & $n$ & Success [95\% CI] & $\Delta$ vs.\ stock, same tasks [95\% CI] \\",
             r"\midrule"]
    for r in sorted(rows, key=lambda r: r["latency_ms"]):
        pr = r["paired"]
        if pr is None:
            delta = "--"
        else:
            delta = f"${100 * pr['mean']:+.0f}$ [${100 * pr['ci'][0]:+.0f}$, ${100 * pr['ci'][1]:+.0f}$]"
            if pr["n_tasks"] != 10:
                delta += f" ({pr['n_tasks']} tasks)"
        lines.append(f"{LABEL.get(r['name'], r['name'])} & {r['latency_ms']:.0f}\\,ms & {r['hold_steps']} & {r['max_actions_in_budget']} & {r['n']} & "
                     f"{100 * r['success']:.0f} [{100 * r['ci'][0]:.0f}, {100 * r['ci'][1]:.0f}] & {delta} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "sim.tex").write_text("\n".join(lines) + "\n")


def write_figure(rows: list[dict], devices: dict, t4: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    rows = sorted(rows, key=lambda r: r["latency_ms"])
    x = [r["latency_ms"] for r in rows]
    y = [100 * r["success"] for r in rows]
    err = [[100 * (r["success"] - r["ci"][0]) for r in rows], [100 * (r["ci"][1] - r["success"]) for r in rows]]
    ax.errorbar(x, y, yerr=err, fmt="o-", color="C0", capsize=3, lw=1.2, ms=4)
    for r in rows:
        ax.annotate(LABEL.get(r["name"], r["name"]).split(" (")[0].split(",")[0], (r["latency_ms"], 100 * r["success"]),
                    textcoords="offset points", xytext=(4, 5), fontsize=6.5)
    for d, v in devices.items():
        ax.axvline(v["pred_ms"], color="0.7", lw=0.6, ls=":")
        ax.text(v["pred_ms"], 2, v["label"], rotation=90, fontsize=6, color="0.4", ha="right", va="bottom")
    ax.set_xlabel("per-chunk inference latency (ms)")
    ax.set_ylabel("LIBERO-Spatial success (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-50, max(x + [v["pred_ms"] for v in devices.values()]) * 1.08)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "figs" / "sim.pdf")


if __name__ == "__main__":
    main()
