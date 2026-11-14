"""Emit LaTeX tables from experiments/results -> paper/tables/*.tex"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
TAB = ROOT / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

MODELS = ["openvla", "openvla-oft", "pi0", "pi05", "smolvla", "gr00t-n1.6"]
LABEL = {"openvla": "OpenVLA", "openvla-oft": "OpenVLA-OFT", "pi0": r"$\pi_0$", "pi05": r"$\pi_{0.5}$",
         "smolvla": "SmolVLA", "gr00t-n1.6": "GR00T N1.6"}
DEVS = ["h100-sxm", "rtx-5090", "rtx-4090", "a10g", "l4", "t4", "jetson-agx-thor", "jetson-agx-orin-64gb",
        "jetson-orin-nano-super"]
ROBUST = 0.8  # Monte Carlo probability of being cheapest required to mark a winner
DLAB = {"h100-sxm": "H100", "rtx-5090": "5090", "rtx-4090": "4090", "a10g": "A10G", "l4": "L4", "t4": "T4",
        "jetson-agx-thor": "Thor", "jetson-agx-orin-64gb": "AGX Orin", "jetson-orin-nano-super": "Orin Nano"}


def pub():
    d = json.loads((RES / "published.json").read_text())
    names = {"naive_same": "Same latency as reference", "spec": "Spec-only roofline",
             "global": "Global calibration", "additive": "Additive residual (no host prior)",
             "overhead": r"Overhead-aware (\textbf{ours})"}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Predictor & MAPE [95\% CI] & Group MAPE & Median & Jetson & $\le$25\% & $\le$2$\times$ \\", r"\midrule"]
    for k, n in names.items():
        a, ci = d["transfer"][k]["all"], d["transfer"][k]["all"]["ci95"]
        lines.append(f"{n} & {a['mape']*100:.1f} [{ci['lo']*100:.0f}, {ci['hi']*100:.0f}] & "
                     f"{d['group_macro_mape'][k]*100:.1f} & {a['median_ape']*100:.1f} & "
                     f"{d['transfer'][k]['edge']['mape']*100:.1f} & {a['within_25pct']*100:.0f}\\% & "
                     f"{a['within_2x']*100:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "transfer.tex").write_text("\n".join(lines) + "\n")

    names2 = {"naive_same": "Same", "spec": "Spec", "global": "Global", "additive": "Additive", "overhead": "Ours"}
    g = [r"\begin{tabular}{lr" + "r" * len(names2) + "}", r"\toprule",
         "Source group & pairs & " + " & ".join(names2.values()) + r" \\", r"\midrule"]
    for grp, v in d["per_group"].items():
        g.append(rf"\texttt{{{grp}}} & {v['n']} & " + " & ".join(f"{v[k]*100:.0f}" for k in names2) + r" \\")
    g += [r"\midrule", "Group mean & & " + " & ".join(f"{d['group_macro_mape'][k]*100:.0f}" for k in names2) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    (TAB / "groups.tex").write_text("\n".join(g) + "\n")
    return d


def cost():
    with (RES / "libero_cost.csv").open() as f:
        rows = list(csv.DictReader(f))

    fits = {(r["model"], r["device"]): r["fits"] == "True" for r in rows}
    measured = {(r["model"], r["device"]) for r in rows if r["latency_measured"] == "True"}
    summ = json.loads((RES / "libero_cost.json").read_text())
    mc = summ["mc"]

    def a(key, **flt):
        return statistics.mean(float(r[key]) for r in rows if all(r[k] == v for k, v in flt.items()))

    succ = {m: a("success", model=m, device="t4", mode="sync", scenario="robot") for m in MODELS}
    cols = "l r " + "r" * len(DEVS)
    out = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           "Model & SR & " + " & ".join(DLAB[d] for d in DEVS) + r" \\", r"\midrule"]
    for scen, title in [("compute-only", "compute only"), ("robot", r"+ robot (\$5/h, 150 W)"),
                        ("robot+operator", r"+ robot + operator (\$35/h)")]:
        out.append(rf"\multicolumn{{{len(DEVS)+2}}}{{l}}{{\emph{{{title}}}}} \\")
        mat = {m: [a("usd_per_1k_success", model=m, device=d, mode="sync", scenario=scen) for d in DEVS] for m in MODELS}
        for m in MODELS:
            mat[m] = [v if fits[(m, d)] else float("inf") for v, d in zip(mat[m], DEVS)]
        pm, pd = mc["p_cheapest_model"][scen], mc["p_cheapest_device"][scen]
        for m in MODELS:
            cells = []
            for j, v in enumerate(mat[m]):
                d = DEVS[j]
                if v == float("inf"):
                    cells.append("--")
                    continue
                s = f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"
                if pm[d].get(m, 0) >= ROBUST:
                    s = rf"\textbf{{{s}}}"
                if pd[m].get(d, 0) >= ROBUST:
                    s = rf"\underline{{{s}}}"
                cells.append(s)
            out.append(f"{LABEL[m]} & {succ[m]*100:.1f} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    (TAB / "cost.tex").write_text("\n".join(out) + "\n")

    lat = [r"\begin{tabular}{l" + "r" * len(DEVS) + "}", r"\toprule",
           "Model & " + " & ".join(DLAB[d] for d in DEVS) + r" \\", r"\midrule"]
    for m in MODELS:
        cells = []
        for d in DEVS:
            v = a("latency_ms", model=m, device=d, mode="async", scenario="robot")
            ok = next(r["realtime_ok"] == "True" for r in rows if r["model"] == m and r["device"] == d and r["mode"] == "async")
            mark = ("*" if (m, d) in measured else "") + ("" if ok else r"$^\dagger$")
            cells.append("--" if not fits[(m, d)] else f"{v:.0f}{mark}")
        lat.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    lat += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "latency.tex").write_text("\n".join(lat) + "\n")


def success():
    summ = json.loads((RES / "libero_cost.json").read_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location("lc", ROOT / "experiments" / "libero_cost.py")
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    out = [r"{\small\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrrrp{5.6cm}}", r"\toprule",
           r"Model & Spatial & Object & Goal & Long & Trials & $H$ & Source \\", r"\midrule"]
    for m in MODELS:
        src, n = summ["success_sources"][m]
        h = str(lc.EXEC_H[m]) + (r"$^\ddagger$" if m in lc.ASSUMED_H else "")
        out.append(f"{LABEL[m]} & " + " & ".join(f"{p*100:.1f}" for p in lc.SUCCESS[m]) +
                   f" & {n} & {h} & {src.replace('_', r'\_')} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    (TAB / "success.tex").write_text("\n".join(out) + "\n")


def local():
    p = RES / "transfer.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fam = d["by_predictor_family"]
    out = [r"\begin{tabular}{lrr}", r"\toprule", r"Predictor & MAPE & median APE \\", r"\midrule"]
    names = {"roofline_ideal": "Roofline, ideal efficiency", "roofline_default": "Roofline, default efficiency",
             "cal_global": "Global calibration", "cal_stage": "Stage calibration",
             "cal_overhead": r"Overhead-aware calibration (\textbf{ours})"}
    for k, n in names.items():
        v = fam[k]
        out.append(f"{n} & {v['mape']*100:.1f}\\% & {v['median_ape']*100:.1f}\\% \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "local.tex").write_text("\n".join(out) + "\n")


def dataset():
    cases = json.loads((ROOT / "experiments/data/published_latency.json").read_text())["cases"]
    out = [r"{\small\setlength{\tabcolsep}{4pt}\begin{tabular}{llrp{9.2cm}}", r"\toprule",
           r"Model & Device & ms & Source \\", r"\midrule"]
    for c in cases:
        src = c["source"].replace("_", r"\_").replace("~", r"$\sim$").replace("&", r"\&")
        out.append(f"{c['model']} & {c['hw']} & {c['total_ms']:g} & {src} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    (TAB / "dataset.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    dataset()
    pub()
    cost()
    success()
    local()
    print(sorted(p.name for p in TAB.iterdir()))
