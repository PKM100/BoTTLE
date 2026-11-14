"""`bottle` command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__
from .analytical import PRECISIONS, Efficiency, estimate
from .calibrate import calibrated_estimate, load_profile, validate
from .cost import TaskSpec, pareto_front, task_cost
from .hardware import get_hardware, list_hardware, load_hw_file
from .models import get_model, list_models, load_model_file


def _table(rows: list[list], headers: list[str]) -> str:
    cells = [[str(c) for c in r] for r in [headers] + rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    line = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([line(cells[0]), "  ".join("-" * x for x in w)] + [line(r) for r in cells[1:]])


def _fmt_s(s: float) -> str:
    return f"{s*1e3:.1f} ms" if s < 1 else f"{s:.2f} s"


def _eff(a) -> Efficiency:
    return Efficiency(a.compute_eff, a.memory_eff, a.power_util)


def _profiles(a) -> dict[str, dict]:
    """Map model name -> profile from --profile files and/or --profile-dir."""
    out = {}
    paths = list(getattr(a, "profile", None) or [])
    if getattr(a, "profile_dir", None):
        paths += sorted(Path(a.profile_dir).glob("*.json"))
    for p in paths:
        prof = load_profile(p)
        out[prof["model"]] = prof
    return out


def _est(model, hw, precision, a, profiles):
    prof = profiles.get(model.name)
    if prof:
        if prof["precision"] != precision:
            print(f"note: {model.name} profile is {prof['precision']}; transferring to {precision}", file=sys.stderr)
        return calibrated_estimate(model, hw, prof, _eff(a), precision=precision,
                                   mode=getattr(a, "calib_mode", "stage"))
    return estimate(model, hw, precision, _eff(a))


def _task(a) -> TaskSpec:
    return TaskSpec(
        episode_steps=a.steps, control_hz=a.control_hz, exec_horizon=a.exec_horizon,
        success_rate=getattr(a, "success", 1.0) or 1.0, mode=a.mode, reset_s=a.reset_s,
        robot_power_w=a.robot_power_w, robot_usd_per_hr=a.robot_usd_per_hr, usd_per_kwh=a.usd_per_kwh, pricing=a.pricing,
        lifetime_years=a.lifetime_years, duty_cycle=a.duty_cycle,
    )


# ---------------------------------------------------------------- commands

def cmd_hw(a):
    if a.action == "show":
        print(json.dumps(get_hardware(a.name).to_dict(), indent=2))
        return
    rows = [[h.name, h.kind, h.peak_tflops.get("fp16", "-"), h.mem_bw_gbs, h.mem_gb, h.tdp_w,
             f"${h.price_usd:,.0f}" if h.price_usd else "-"] for h in list_hardware()]
    print(_table(rows, ["name", "kind", "fp16 TFLOPS", "BW GB/s", "mem GB", "TDP W", "price"]))


def cmd_models(a):
    if a.action == "show":
        print(json.dumps(get_model(a.name).to_dict(), indent=2))
        return
    hw = get_hardware("t4")
    rows = []
    for m in list_models():
        e = estimate(m, hw, "fp16")
        rows.append([m.name, f"{e.params/1e9:.2f}B", m.action_chunk, len(m.stages), m.description])
    print(_table(rows, ["name", "params", "chunk", "stages", "description"]))


def cmd_estimate(a):
    profiles = _profiles(a)
    for hname in a.hw:
        hw = get_hardware(hname)
        e = _est(get_model(a.model), hw, a.precision, a, profiles)
        if a.json:
            print(json.dumps(e.to_dict(), indent=2))
            continue
        print(f"\n{e.model} on {e.hardware} [{e.precision}->{e.compute_precision}]"
              f"{' (calibrated)' if e.calibrated else ''}")
        rows = [[s.name, s.kind, f"{s.params/1e6:.0f}M", f"{s.flops/1e9:.1f}", f"{s.bytes/1e9:.2f}",
                 s.bound, _fmt_s(s.time_s)] for s in e.stages]
        rows.append(["host", "-", "-", "-", "-", "cpu", _fmt_s(e.host_s)])
        print(_table(rows, ["stage", "kind", "params", "GFLOP", "GB moved", "bound", "time"]))
        print(f"latency {_fmt_s(e.latency_s)} | mem {e.memory_gb:.1f} GB (fits={e.fits}) | "
              f"{e.avg_power_w:.0f} W | {e.energy_j:.2f} J/call")
        for n in e.notes:
            print(f"  note: {n}")


def cmd_profile(a):
    from .profiler import profile, save
    prof = profile(get_model(a.model), a.hw_name, a.device, a.precision, a.max_layers,
                   a.warmup, a.iters, a.energy_seconds)
    out = a.out or f"profiles/{a.model}_{a.hw_name}_{prof['precision']}.json"
    save(prof, out)
    print(json.dumps({k: prof[k] for k in ("model", "hardware", "precision", "latency_s", "avg_power_w")}, indent=2))
    print(f"saved {out}")


def cmd_probe(a):
    from .profiler import probe
    hw = probe(a.name, a.device, a.precision, a.iters)
    hw.update(mem_gb=a.mem_gb, tdp_w=a.tdp_w, idle_w=a.idle_w, price_usd=a.price_usd)
    text = json.dumps(hw, indent=2)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"saved {a.out} (use with --hw-file)")


def cmd_task(a):
    profiles = _profiles(a)
    model = get_model(a.model)
    rows, out = [], []
    for hname in a.hw:
        hw = get_hardware(hname)
        tc = task_cost(_est(model, hw, a.precision, a, profiles), hw, _task(a), model.action_chunk)
        out.append(tc.to_dict())
        rows.append(_cost_row(tc))
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(_table(rows, _COST_HEADERS))
        for r in out:
            for n in r["notes"]:
                print(f"  {r['hardware']}: {n}")


_COST_HEADERS = ["model", "hardware", "prec", "latency", "realtime", "success",
                 "s/success", "J/success", "$/1k successes", "calib"]


def _cost_row(tc) -> list:
    return [tc.model, tc.hardware, tc.precision, f"{tc.latency_ms:.1f} ms", "yes" if tc.realtime_ok else "NO",
            f"{tc.success_rate:.2f}", f"{tc.time_per_success_s:.1f}", f"{tc.energy_per_success_j:.0f}",
            f"{tc.usd_per_1k_successes:.3f}", "y" if tc.calibrated else "n"]


def _parse_success(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    if Path(spec).exists():
        return {k: float(v) for k, v in json.loads(Path(spec).read_text()).items()}
    return {k: float(v) for k, v in (p.split("=") for p in spec.split(","))}


def cmd_sweep(a):
    profiles = _profiles(a)
    success = _parse_success(a.success)
    models = a.models or [m.name for m in list_models()]
    hws = a.hw or [h.name for h in list_hardware()]
    results = []
    for mname in models:
        model = get_model(mname)
        for hname in hws:
            hw = get_hardware(hname)
            spec = _task(a)
            spec.success_rate = success.get(mname, 1.0)
            for prec in a.precisions:
                try:
                    tc = task_cost(_est(model, hw, prec, a, profiles), hw, spec, model.action_chunk)
                except ValueError as e:
                    print(f"skip {mname}/{hname}/{prec}: {e}", file=sys.stderr)
                    continue
                results.append(tc)
    results.sort(key=lambda r: r.usd_per_success)
    front = {id(r) for r in pareto_front([r for r in results if r.realtime_ok and r.fits])}
    print(_table([_cost_row(r) + ["*" if id(r) in front else ""] for r in results], _COST_HEADERS + ["pareto"]))
    if not success:
        print("\nnote: no --success given; success_rate=1.0 for all (cost-only ranking)")
    if a.out:
        _write(results, front, Path(a.out))
        print(f"saved {a.out}")


def _write(results, front, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**r.to_dict(), "notes": "; ".join(r.notes), "pareto": id(r) in front} for r in results]
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2))
    elif path.suffix == ".md":
        hdr = _COST_HEADERS + ["pareto"]
        body = [_cost_row(r) + ["*" if id(r) in front else ""] for r in results]
        lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
        path.write_text("\n".join(lines) + "\n")
    else:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def cmd_validate(a):
    pred = json.loads(Path(a.predicted).read_text())
    meas = json.loads(Path(a.measured).read_text())
    res = validate({k: float(v) for k, v in pred.items()}, {k: float(v) for k, v in meas.items()})
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- parser

def _add_eff(p):
    g = p.add_argument_group("roofline efficiency (uncalibrated defaults)")
    d = Efficiency()
    g.add_argument("--compute-eff", type=float, default=d.compute)
    g.add_argument("--memory-eff", type=float, default=d.memory)
    g.add_argument("--power-util", type=float, default=d.power_util)
    p.add_argument("--profile", action="append", help="calibration profile JSON (repeatable)")
    p.add_argument("--profile-dir", help="directory of calibration profiles")
    p.add_argument("--calib-mode", choices=["stage", "global", "overhead"], default="stage",
                   help="stage: per-stage ratio; global: one ratio; overhead: unexplained reference time "
                        "is host overhead scaled by host CPU speed (best for eager end-to-end measurements)")


def _add_task(p):
    d = TaskSpec()
    g = p.add_argument_group("task")
    g.add_argument("--steps", type=int, default=d.episode_steps, help="control steps per attempt")
    g.add_argument("--control-hz", type=float, default=d.control_hz)
    g.add_argument("--exec-horizon", type=int, default=None, help="actions executed per call (default: chunk)")
    g.add_argument("--mode", choices=["sync", "async"], default=d.mode)
    g.add_argument("--reset-s", type=float, default=d.reset_s)
    g.add_argument("--robot-power-w", type=float, default=d.robot_power_w)
    g.add_argument("--robot-usd-per-hr", type=float, default=d.robot_usd_per_hr,
                   help="amortized robot/operator cost per hour of robot time")
    g.add_argument("--usd-per-kwh", type=float, default=d.usd_per_kwh)
    g.add_argument("--pricing", choices=["own", "cloud"], default=d.pricing)
    g.add_argument("--lifetime-years", type=float, default=d.lifetime_years)
    g.add_argument("--duty-cycle", type=float, default=d.duty_cycle)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bottle", description="BoTTLE: Benchmark of Task-level Total Latency & Energy")
    p.add_argument("--version", action="version", version=f"bottle {__version__}")
    p.add_argument("--hw-file", action="append", help="JSON file adding/overriding hardware (repeatable)")
    p.add_argument("--model-file", help="JSON file adding/overriding model specs")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("hw", help="list/show hardware")
    s.add_argument("action", choices=["list", "show"], nargs="?", default="list")
    s.add_argument("name", nargs="?")
    s.set_defaults(fn=cmd_hw)

    s = sub.add_parser("models", help="list/show model specs")
    s.add_argument("action", choices=["list", "show"], nargs="?", default="list")
    s.add_argument("name", nargs="?")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("estimate", help="per-call latency/memory/energy estimate")
    s.add_argument("--model", required=True)
    s.add_argument("--hw", nargs="+", required=True)
    s.add_argument("--precision", default="fp16", choices=list(PRECISIONS))
    s.add_argument("--json", action="store_true")
    _add_eff(s)
    s.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("profile", help="measure per-stage latency/energy on this machine (needs torch)")
    s.add_argument("--model", required=True)
    s.add_argument("--hw-name", required=True, help="registry name of this device, e.g. t4")
    s.add_argument("--device", default="cuda")
    s.add_argument("--precision", default=None, choices=["fp16", "bf16", "fp32"])
    s.add_argument("--max-layers", type=int, default=None, help="profile N layers/stage and extrapolate")
    s.add_argument("--warmup", type=int, default=3)
    s.add_argument("--iters", type=int, default=10)
    s.add_argument("--energy-seconds", type=float, default=5.0)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("probe", help="microbenchmark this device into a hardware JSON (needs torch)")
    s.add_argument("--name", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--precision", default=None, choices=["fp16", "bf16", "fp32"])
    s.add_argument("--iters", type=int, default=20)
    s.add_argument("--mem-gb", type=float, default=0.0)
    s.add_argument("--tdp-w", type=float, default=0.0)
    s.add_argument("--idle-w", type=float, default=0.0)
    s.add_argument("--price-usd", type=float, default=0.0)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("task", help="cost per successful task for one model")
    s.add_argument("--model", required=True)
    s.add_argument("--hw", nargs="+", required=True)
    s.add_argument("--precision", default="fp16", choices=list(PRECISIONS))
    s.add_argument("--success", type=float, default=1.0, help="per-attempt success rate")
    s.add_argument("--json", action="store_true")
    _add_task(s)
    _add_eff(s)
    s.set_defaults(fn=cmd_task)

    s = sub.add_parser("sweep", help="models x hardware x precision cost table with Pareto front")
    s.add_argument("--models", nargs="+")
    s.add_argument("--hw", nargs="+")
    s.add_argument("--precisions", nargs="+", default=["fp16"], choices=list(PRECISIONS))
    s.add_argument("--success", help="'smolvla=0.87,pi0=0.94' or a JSON file {model: rate}")
    s.add_argument("--out", help=".csv, .json or .md")
    _add_task(s)
    _add_eff(s)
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("validate", help="MAPE of predicted vs measured latencies")
    s.add_argument("--predicted", required=True, help="JSON {case_id: seconds}")
    s.add_argument("--measured", required=True, help="JSON {case_id: seconds}")
    s.set_defaults(fn=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    for f in a.hw_file or []:
        load_hw_file(f)
    if a.model_file:
        load_model_file(a.model_file)
    if getattr(a, "action", None) == "show" and not a.name:
        raise SystemExit("show needs a name")
    try:
        a.fn(a)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"error: {e.args[0] if e.args else e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())

