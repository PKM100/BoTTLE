import json
import math

import pytest

from bottlebench.analytical import estimate
from bottlebench.calibrate import calibrated_estimate, validate
from bottlebench.cli import main
from bottlebench.cost import TaskSpec, pareto_front, task_cost
from bottlebench.hardware import get_hardware
from bottlebench.models import Stage, ModelSpec, get_model


def _tiny(**kw):
    s = dict(name="s", kind="prefill", n_layers=2, d_model=64, d_ff=256, n_heads=4, tokens=10)
    s.update(kw)
    return ModelSpec("tiny", [Stage(**s)], action_chunk=10, host_overhead_ms=0.0)


def test_param_and_flop_count():
    e = estimate(_tiny(gated_mlp=False), get_hardware("t4"), "fp16")
    st = e.stages[0]
    per_layer = 4 * 64 * 64 + 2 * 64 * 256
    assert st.params == 2 * per_layer
    assert st.flops == pytest.approx(2 * st.params * 10 + 2 * 4 * 10 * 10 * 64)


def test_decode_is_memory_bound_and_scales_with_repeats():
    hw = get_hardware("t4")
    one = estimate(_tiny(kind="decode", tokens=1, d_model=4096, d_ff=11008, n_heads=32), hw)
    many = estimate(_tiny(kind="decode", tokens=1, d_model=4096, d_ff=11008, n_heads=32, repeats=7), hw)
    assert one.stages[0].bound == "memory"
    assert many.stages[0].time_s == pytest.approx(7 * one.stages[0].time_s)


def test_shared_weights_not_double_counted():
    m = get_model("openvla")
    e = estimate(m, get_hardware("t4"))
    assert e.params == pytest.approx(sum(s.params for s in e.stages if s.name != "action"))


def test_quantization_reduces_memory_bound_latency():
    hw = get_hardware("jetson-agx-orin-64gb")
    fp16 = estimate(get_model("openvla"), hw, "fp16")
    w4 = estimate(get_model("openvla"), hw, "w4a16")
    assert w4.latency_s < fp16.latency_s and w4.weights_gb == pytest.approx(fp16.weights_gb / 4)


def test_calibration_scales_stage_time_by_measured_ratio():
    m, ref = get_model("smolvla"), get_hardware("t4")
    base = estimate(m, ref)
    prof = {"model": "smolvla", "hardware": "t4", "precision": "fp16",
            "stages": {s.name: {"time_s": 2 * s.time_s} for s in base.stages}}
    same = calibrated_estimate(m, ref, prof)
    assert sum(s.time_s for s in same.stages) == pytest.approx(2 * sum(s.time_s for s in base.stages))
    thor = calibrated_estimate(m, get_hardware("jetson-agx-thor"), prof)
    raw_thor = estimate(m, get_hardware("jetson-agx-thor"))
    assert thor.stages[0].time_s == pytest.approx(2 * raw_thor.stages[0].time_s)
    assert thor.calibrated


def test_overhead_mode_carries_unexplained_time_as_host_overhead():
    m, ref, tgt = get_model("smolvla"), get_hardware("h100-sxm"), get_hardware("jetson-agx-thor")
    r, t = estimate(m, ref), estimate(m, tgt)
    extra = 0.1
    prof = {"model": "smolvla", "hardware": "h100-sxm", "precision": "fp16",
            "stages": {"_total": {"time_s": r.latency_s + extra}}}
    ovh = calibrated_estimate(m, tgt, prof, mode="overhead")
    assert ovh.latency_s == pytest.approx(t.latency_s + extra * ref.host_speed / tgt.host_speed)
    glob = calibrated_estimate(m, tgt, prof, mode="global")
    assert glob.latency_s == pytest.approx(t.latency_s * (r.latency_s + extra) / r.latency_s)


def test_calibrated_power_transfers_utilization():
    m, ref, tgt = get_model("smolvla"), get_hardware("t4"), get_hardware("jetson-agx-thor")
    prof = {"model": "smolvla", "hardware": "t4", "precision": "fp16", "stages": {},
            "avg_power_w": ref.idle_w + 0.5 * (ref.tdp_w - ref.idle_w)}
    e = calibrated_estimate(m, tgt, prof)
    assert e.avg_power_w == pytest.approx(tgt.idle_w + 0.5 * (tgt.tdp_w - tgt.idle_w))


def test_task_cost_retries_and_modes():
    hw = get_hardware("t4")
    e = estimate(_tiny(), hw)
    lat = e.latency_s
    sync = task_cost(e, hw, TaskSpec(episode_steps=100, control_hz=10, mode="sync", success_rate=0.5), 10)
    assert sync.calls_per_attempt == 10
    assert sync.attempt_s == pytest.approx(10 + 10 * lat)
    assert sync.time_per_success_s == pytest.approx(2 * sync.attempt_s)
    asy = task_cost(e, hw, TaskSpec(episode_steps=100, control_hz=10, mode="async"), 10)
    assert asy.attempt_s == pytest.approx(max(10, 10 * lat) + lat)
    assert asy.realtime_ok


def test_realtime_violation_flagged():
    hw = get_hardware("jetson-orin-nano-super")
    m = get_model("openvla")
    tc = task_cost(estimate(m, hw), hw, TaskSpec(control_hz=50), m.action_chunk)
    assert not tc.realtime_ok and not tc.fits


def test_cost_components():
    hw = get_hardware("t4")
    e = estimate(_tiny(), hw)
    spec = TaskSpec(episode_steps=10, control_hz=10, pricing="cloud", usd_per_kwh=0, robot_usd_per_hr=36)
    tc = task_cost(e, hw, spec, 10)
    assert tc.usd_per_success == pytest.approx(tc.attempt_s * (hw.cloud_usd_per_hr / 3600 + 0.01))


def test_pareto_front():
    hw = get_hardware("t4")
    e = estimate(_tiny(), hw)
    cheap_bad = task_cost(e, hw, TaskSpec(success_rate=0.5), 10)
    pricey_good = task_cost(e, get_hardware("h100-sxm"), TaskSpec(success_rate=0.9), 10)
    pricey_bad = task_cost(e, get_hardware("h100-sxm"), TaskSpec(success_rate=0.4), 10)
    front = pareto_front([cheap_bad, pricey_good, pricey_bad])
    assert cheap_bad in front and pricey_good in front and pricey_bad not in front


def test_validate_mape():
    r = validate({"a": 1.1, "b": 0.9}, {"a": 1.0, "b": 1.0, "c": 5})
    assert r["n"] == 2 and r["mape"] == pytest.approx(0.1)
    assert r["geo_mean_ratio"] == pytest.approx(math.sqrt(1.1 * 0.9))


def test_cli_sweep_and_profile_roundtrip(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["sweep", "--models", "smolvla", "--hw", "t4", "--success", "smolvla=0.8",
                 "--out", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert rows[0]["success_rate"] == 0.8
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps({"model": "smolvla", "hardware": "t4", "precision": "fp16",
                                "stages": {"vision": {"time_s": 0.05}}}))
    capsys.readouterr()
    assert main(["task", "--model", "smolvla", "--hw", "jetson-agx-thor", "--profile", str(prof), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result[0]["calibrated"] is True


def test_unknown_names_error():
    with pytest.raises(SystemExit):
        main(["estimate", "--model", "nope", "--hw", "t4"])
