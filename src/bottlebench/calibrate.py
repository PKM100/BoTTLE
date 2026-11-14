"""Calibrate the roofline model with measurements from a reference device and
transfer it to target hardware.

Assumption (to be tested with ``validate``): the ratio between measured and
roofline-estimated time for a stage is similar across devices of the same
software stack. This is the core hypothesis the benchmark evaluates.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from .analytical import Efficiency, Estimate, estimate
from .hardware import Hardware, get_hardware
from .models import ModelSpec


def load_profile(path: str | Path) -> dict:
    prof = json.loads(Path(path).read_text())
    for key in ("model", "hardware", "precision", "stages"):
        if key not in prof:
            raise ValueError(f"profile {path} missing '{key}'")
    return prof


def calibrated_estimate(model: ModelSpec, target: Hardware, profile: dict,
                        eff: Efficiency | None = None, precision: str | None = None,
                        mode: str = "stage") -> Estimate:
    """Scale the target estimate by measured/estimated on the reference device.

    mode="stage": one factor per stage; mode="global": a single factor for the whole call;
    mode="overhead": per-stage, time the roofline cannot explain on the reference is treated
    as host/framework overhead and scaled by relative host CPU speed.
    Profiles may give per-stage times or only ``stages["_total"]`` (end-to-end, incl. host).
    ``precision`` is the target precision (defaults to the profile's).
    """
    eff = eff or Efficiency()
    if profile["model"] != model.name:
        raise ValueError(f"profile is for {profile['model']}, not {model.name}")
    if mode not in ("stage", "global", "overhead"):
        raise ValueError("mode must be 'stage', 'global' or 'overhead'")
    ref_hw = get_hardware(profile["hardware"])
    ref = estimate(model, ref_hw, profile["precision"], eff)
    tgt = estimate(model, target, precision or profile["precision"], eff)

    measured = profile["stages"]
    hs = ref_hw.host_speed / max(target.host_speed, 1e-6)

    def transfer(m: float, r: float, t: float) -> float:
        if mode == "overhead":
            # unexplained reference time is treated as host/framework overhead
            return min(m / r, 1.0) * t + max(m - r, 0.0) * hs
        return t * m / r

    if "_total" in measured or mode == "global":
        if "_total" in measured:
            m_tot = measured["_total"]["time_s"]
            r_tot = sum(r.time_s for r in ref.stages) + ref.host_s
        else:
            known = [(r, measured[r.name]["time_s"]) for r in ref.stages
                     if measured.get(r.name, {}).get("time_s")]
            m_tot = sum(m for _, m in known)
            r_tot = sum(r.time_s for r, _ in known)
        t_tot = sum(t.time_s for t in tgt.stages) + (tgt.host_s if "_total" in measured else 0.0)
        pred_tot = transfer(m_tot, r_tot, t_tot) if r_tot > 0 else t_tot
        scale = pred_tot / t_tot if t_tot > 0 else 1.0
        new_stages = [replace(t, time_s=t.time_s * scale) for t in tgt.stages]
        host_scaled = tgt.host_s * scale  # used only when host is inside the measured total
    else:
        new_stages = []
        for r, t in zip(ref.stages, tgt.stages):
            m = measured.get(r.name, {}).get("time_s")
            new_stages.append(replace(t, time_s=transfer(m, r.time_s, t.time_s)) if m and r.time_s > 0 else t)

    host = tgt.host_s
    if "_total" in measured:
        host = host_scaled
    elif profile.get("host_s") is not None:
        host = profile["host_s"] * ref_hw.host_speed / target.host_speed

    power = tgt.avg_power_w
    if profile.get("avg_power_w"):
        span = ref_hw.tdp_w - ref_hw.idle_w
        util = min(max((profile["avg_power_w"] - ref_hw.idle_w) / span, 0.0), 1.0) if span > 0 else eff.power_util
        power = target.idle_w + util * (target.tdp_w - target.idle_w)

    latency = sum(s.time_s for s in new_stages) + host
    notes = list(tgt.notes) + [f"calibrated ({mode}) from {ref_hw.name}/{profile['precision']} profile"]
    return replace(tgt, stages=new_stages, host_s=host, latency_s=latency,
                   avg_power_w=power, energy_j=latency * power, calibrated=True, notes=notes)


def validate(predicted: dict[str, float], measured: dict[str, float]) -> dict:
    """Compare predicted vs measured latencies (keys = case ids). Returns MAPE etc."""
    common = sorted(set(predicted) & set(measured))
    if not common:
        raise ValueError("no overlapping cases between predicted and measured")
    errs = {k: (predicted[k] - measured[k]) / measured[k] for k in common}
    ape = [abs(e) for e in errs.values()]
    logs = [math.log(predicted[k] / measured[k]) for k in common]
    return {
        "n": len(common),
        "mape": sum(ape) / len(ape),
        "max_ape": max(ape),
        "geo_mean_ratio": math.exp(sum(logs) / len(logs)),
        "per_case": errs,
    }
