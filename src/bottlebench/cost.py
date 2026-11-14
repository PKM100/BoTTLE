"""Task-level cost: turn per-call latency/energy into cost per *successful* task."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .analytical import Estimate
from .hardware import Hardware

SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass
class TaskSpec:
    episode_steps: int = 300  # control steps (actions) needed per attempt
    control_hz: float = 20.0  # rate at which the robot consumes actions
    exec_horizon: int | None = None  # actions executed per policy call (default: model chunk)
    success_rate: float = 1.0  # per-attempt success probability
    mode: str = "async"  # "sync" (robot waits for inference) or "async" (overlapped)
    reset_s: float = 0.0  # time to reset between attempts
    robot_power_w: float = 0.0  # actuators etc. (0 = compute-only accounting)
    robot_usd_per_hr: float = 0.0  # amortized robot/operator cost per hour of robot time
    usd_per_kwh: float = 0.15
    pricing: str = "own"  # "own" (amortized purchase) or "cloud" (hourly rate)
    lifetime_years: float = 3.0
    duty_cycle: float = 0.5  # fraction of lifetime the device is in use


@dataclass
class TaskCost:
    model: str
    hardware: str
    precision: str
    latency_ms: float
    calls_per_attempt: int
    attempt_s: float
    realtime_ok: bool
    achievable_hz: float
    success_rate: float
    expected_attempts: float
    time_per_success_s: float
    energy_per_success_j: float
    usd_per_success: float
    usd_per_1k_successes: float
    calibrated: bool
    fits: bool
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def task_cost(est: Estimate, hw: Hardware, task: TaskSpec, action_chunk: int) -> TaskCost:
    if not 0 < task.success_rate <= 1:
        raise ValueError("success_rate must be in (0, 1]")
    if task.mode not in ("sync", "async"):
        raise ValueError("mode must be 'sync' or 'async'")
    horizon = task.exec_horizon or action_chunk
    if horizon > action_chunk:
        raise ValueError(f"exec_horizon {horizon} > model action chunk {action_chunk}")

    lat = est.latency_s
    calls = math.ceil(task.episode_steps / horizon)
    act_s = task.episode_steps / task.control_hz
    infer_s = calls * lat
    # async: next chunk is computed while the current one executes; first call is exposed
    if task.mode == "sync":
        attempt_s = act_s + infer_s
    else:
        attempt_s = max(act_s, infer_s) + lat
    attempt_s += task.reset_s

    achievable_hz = horizon / lat if task.mode == "async" else 1.0 / (lat / horizon + 1.0 / task.control_hz)
    realtime_ok = lat <= horizon / task.control_hz if task.mode == "async" else True

    compute_j = infer_s * est.avg_power_w + max(attempt_s - infer_s, 0.0) * hw.idle_w
    energy_j = compute_j + attempt_s * task.robot_power_w

    if task.pricing == "cloud":
        rate = hw.cloud_usd_per_hr / 3600
    else:
        life_s = task.lifetime_years * SECONDS_PER_YEAR * task.duty_cycle
        rate = hw.price_usd / life_s if life_s > 0 else 0.0
    usd = energy_j / 3.6e6 * task.usd_per_kwh + attempt_s * (rate + task.robot_usd_per_hr / 3600)

    n = 1.0 / task.success_rate
    notes = list(est.notes)
    if not realtime_ok:
        notes.append(f"latency {lat*1e3:.0f} ms exceeds chunk budget {horizon/task.control_hz*1e3:.0f} ms (robot will stall)")
    if task.pricing == "cloud" and hw.cloud_usd_per_hr == 0:
        notes.append(f"{hw.name} has no cloud price; hardware cost = 0")
    if task.pricing == "own" and hw.price_usd == 0:
        notes.append(f"{hw.name} has no purchase price; hardware cost = 0")
    return TaskCost(
        est.model, est.hardware, est.precision, lat * 1e3, calls, attempt_s, realtime_ok,
        achievable_hz, task.success_rate, n, attempt_s * n, energy_j * n, usd * n,
        usd * n * 1000, est.calibrated, est.fits, notes,
    )


def pareto_front(rows: list[TaskCost], cost_key: str = "usd_per_success") -> list[TaskCost]:
    """Rows not dominated on (lower cost, higher success)."""
    front = []
    for r in rows:
        dominated = any(
            getattr(o, cost_key) <= getattr(r, cost_key) and o.success_rate >= r.success_rate
            and (getattr(o, cost_key) < getattr(r, cost_key) or o.success_rate > r.success_rate)
            for o in rows
        )
        if not dominated:
            front.append(r)
    return front
