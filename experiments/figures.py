"""Make paper figures from experiments/results/*.json|csv -> paper/figs/*.pdf"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.bbox": "tight"})

MODELS = ["openvla", "openvla-oft", "pi0", "pi05", "smolvla", "gr00t-n1.6"]
LABEL = {"openvla": "OpenVLA", "openvla-oft": "OpenVLA-OFT", "pi0": r"$\pi_0$", "pi05": r"$\pi_{0.5}$",
         "smolvla": "SmolVLA", "gr00t-n1.6": "GR00T N1.6"}
DEVS = ["h100-sxm", "rtx-5090", "rtx-4090", "a10g", "l4", "t4",
        "jetson-agx-thor", "jetson-agx-orin-64gb", "jetson-orin-nano-super"]
DLAB = {"h100-sxm": "H100", "rtx-5090": "RTX5090", "rtx-4090": "RTX4090", "a10g": "A10G", "l4": "L4", "t4": "T4",
        "jetson-agx-thor": "Thor", "jetson-agx-orin-64gb": "AGX Orin", "jetson-orin-nano-super": "Orin Nano"}


def fig_transfer():
    d = json.loads((RES / "published.json").read_text())["transfer_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharex=True, sharey=True)
    for ax, (k, title) in zip(axes, [("spec", "Spec-only roofline"), ("global", "Global calibration"),
                                     ("overhead", "Overhead-aware (ours)")]):
        for r in d:
            ax.scatter(r["measured_ms"], r[f"{k}_ms"], s=10, marker="^" if r["edge_target"] else "o",
                       c="tab:red" if r["edge_target"] else "tab:blue", alpha=0.8)
        mape = statistics.mean(r[f"{k}_ape"] for r in d) * 100
        ax.plot([10, 1500], [10, 1500], "k-", lw=0.6)
        for f in (0.75, 1.25):
            ax.plot([10, 1500], [10 * f, 1500 * f], "k:", lw=0.5)
        ax.set(xscale="log", yscale="log", xlim=(12, 1200), ylim=(12, 1200), title=f"{title}\nMAPE {mape:.0f}%")
        ax.set_xlabel("measured (ms)")
    axes[0].set_ylabel("predicted (ms)")
    axes[0].scatter([], [], c="tab:blue", s=10, label="desktop/DC target")
    axes[0].scatter([], [], c="tab:red", marker="^", s=10, label="Jetson target")
    axes[0].legend(loc="upper left", fontsize=6)
    fig.savefig(FIGS / "transfer.pdf")


def load_cost():
    with (RES / "libero_cost.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("latency_ms", "usd_per_1k_success", "time_per_success_s", "success"):
            r[k] = float(r[k])
        r["realtime_ok"] = r["realtime_ok"] == "True"
        r["fits"] = r["fits"] == "True"
    return rows


def agg(rows, key, **f):
    sel = [r for r in rows if all(r[k] == v for k, v in f.items())]
    return statistics.mean(r[key] for r in sel)


def fig_scenarios(rows):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2), sharey=False)
    for ax, scen in zip(axes, ["compute-only", "robot", "robot+operator"]):
        for m in MODELS:
            xs, ys = [], []
            for d in DEVS:
                if not next(r["fits"] for r in rows if r["model"] == m and r["device"] == d):
                    continue
                xs.append(agg(rows, "latency_ms", model=m, device=d, mode="sync", scenario=scen))
                ys.append(agg(rows, "usd_per_1k_success", model=m, device=d, mode="sync", scenario=scen))
            ax.scatter(xs, ys, s=9, label=LABEL[m])
        ax.set(xscale="log", yscale="log", title=scen, xlabel="latency per call (ms)")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("\\$ per 1k successes")
    axes[2].legend(fontsize=6, loc="upper left")
    fig.savefig(FIGS / "scenarios.pdf")


def fig_local():
    p = RES / "transfer.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fam = d.get("by_predictor_family") or {}
    if not fam:
        return
    names = list(fam)
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.barh(range(len(names)), [fam[n]["mape"] * 100 for n in names], color="tab:gray")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("MAPE (%)")
    ax.set_xscale("log")
    fig.savefig(FIGS / "local.pdf")


if __name__ == "__main__":
    fig_transfer()
    rows = load_cost()
    fig_scenarios(rows)
    fig_local()
    print(sorted(p.name for p in FIGS.iterdir()))
