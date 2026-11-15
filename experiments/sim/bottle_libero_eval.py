"""Closed-loop LIBERO evaluation with measured or emulated inference latency (LeRobot).

Stock LIBERO pauses physics while the policy computes, so success never depends on latency.
This wrapper around ``lerobot-eval`` makes it matter, emulating a *synchronous* robot: every time
the policy computes a new action chunk, the robot holds still (zero end-effector delta, last
gripper command) for ceil(latency * fps) physics steps before executing it. By default hold steps
consume the episode's step budget, like wall-clock time would on a real robot
(``--bottle.budget=steps``); ``--bottle.budget=actions`` gives every setting stock's budget of
policy actions instead, isolating the effect of stalls on control from running out of time. (In
LIBERO the underlying robosuite env still ends episodes at 1000 physics steps.)

    --bottle.mode=none        stock behaviour (physics paused); only records timing
    --bottle.mode=measured    hold for the latency measured on *this* GPU, call by call
    --bottle.mode=emulate     hold for --bottle.latency_ms (any device, e.g. a BoTTLE prediction)

All other arguments go to lerobot-eval, e.g.

    python bottle_libero_eval.py --bottle.mode=emulate --bottle.latency_ms=308 \\
        --policy.path=lerobot/smolvla_libero --policy.n_action_steps=10 \\
        --env.type=libero --env.task=libero_spatial --eval.batch_size=1 --eval.n_episodes=10 \\
        --output_dir=outputs/smolvla_thor

Writes ``bottle_timing.json`` (per-call inference times, per-episode steps/success/calls) next to
lerobot's ``eval_info.json``. Requires Linux (MuJoCo EGL): run on Colab/Kaggle with MUJOCO_GL=egl.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

STATE = {"mode": "none", "latency_ms": 0.0, "budget": "steps", "fps": 20, "pending_hold": 0, "calls_this_episode": 0,
         "infer_s": [], "episodes": [], "hold_steps": 0}


def parse_own_args(argv: list[str]) -> list[str]:
    """Consume --bottle.* options into STATE; return the remaining argv for lerobot-eval."""
    rest = []
    for a in argv:
        if a.startswith("--bottle."):
            k, _, v = a[len("--bottle."):].partition("=")
            if k == "mode":
                if v not in ("none", "measured", "emulate"):
                    raise SystemExit("--bottle.mode must be none|measured|emulate")
                STATE["mode"] = v
            elif k == "latency_ms":
                STATE["latency_ms"] = float(v)
            elif k == "budget":
                if v not in ("steps", "actions"):
                    raise SystemExit("--bottle.budget must be steps|actions")
                STATE["budget"] = v
            else:
                raise SystemExit(f"unknown option --bottle.{k}")
        else:
            if a.startswith("--env.fps="):
                STATE["fps"] = int(a.split("=", 1)[1])
            rest.append(a)
    if STATE["mode"] == "emulate" and STATE["latency_ms"] <= 0:
        raise SystemExit("--bottle.mode=emulate needs --bottle.latency_ms > 0")
    return rest


def hold_steps(latency_s: float, fps: int) -> int:
    return math.ceil(latency_s * fps - 1e-9)


def _queue_len(policy) -> int | None:
    q = getattr(policy, "_action_queue", None)
    if q is None:
        qs = getattr(policy, "_queues", None)
        if isinstance(qs, dict):
            q = qs.get("action")
    return None if q is None else len(q)


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wrap_policy(policy):
    """Time each select_action call that computes a new chunk and schedule the hold."""
    orig = policy.select_action

    def select_action(batch, *args, **kwargs):
        qlen = _queue_len(policy)
        _sync()
        t0 = time.perf_counter()
        out = orig(batch, *args, **kwargs)
        _sync()
        dt = time.perf_counter() - t0
        new_chunk = qlen == 0 if qlen is not None else dt > 2e-3
        if new_chunk:
            STATE["infer_s"].append(dt)
            lat = {"none": 0.0, "measured": dt, "emulate": STATE["latency_ms"] / 1e3}[STATE["mode"]]
            STATE["pending_hold"] = hold_steps(lat, STATE["fps"])
            STATE["calls_this_episode"] += 1
        return out

    policy.select_action = select_action
    return policy


def patch_env(env_cls) -> None:
    """Make env_cls.step insert the scheduled hold steps and enforce the step budget on them."""
    orig_step, orig_reset = env_cls.step, env_cls.reset

    def reset(self, *args, **kwargs):
        self._bottle_t, self._bottle_a, self._bottle_grip = 0, 0, -1.0
        STATE["pending_hold"], STATE["calls_this_episode"] = 0, 0
        return orig_reset(self, *args, **kwargs)

    def finish(self, info, truncated):
        STATE["episodes"].append({"task_id": getattr(self, "task_id", None), "steps": self._bottle_t, "actions": self._bottle_a,
                                  "success": bool(info.get("is_success", False)), "truncated": truncated,
                                  "calls": STATE["calls_this_episode"]})

    def step(self, action):
        limit = self._max_episode_steps
        holds_count = STATE["budget"] == "steps"
        n_hold, STATE["pending_hold"] = STATE["pending_hold"], 0
        hold = np.array([0, 0, 0, 0, 0, 0, self._bottle_grip], dtype=np.float64)
        for _ in range(n_hold):
            obs, rew, term, _, info = orig_step(self, hold)
            self._bottle_t += 1
            STATE["hold_steps"] += 1
            if term or (holds_count and self._bottle_t >= limit):
                trunc = not term
                finish(self, info, trunc)
                return obs, rew, term, trunc, info
        obs, rew, term, _, info = orig_step(self, action)
        self._bottle_t += 1
        self._bottle_a += 1
        self._bottle_grip = float(np.asarray(action).reshape(-1)[-1])
        trunc = (not term) and (self._bottle_t if holds_count else self._bottle_a) >= limit
        if term or trunc:
            finish(self, info, trunc)
        return obs, rew, term, trunc, info

    env_cls.step, env_cls.reset = step, reset


def summary() -> dict:
    t, eps = STATE["infer_s"], STATE["episodes"]
    steady = t[1:] or t  # first call includes warm-up
    succ = [e for e in eps if e["success"]]
    return {
        "mode": STATE["mode"], "latency_ms": STATE["latency_ms"], "budget": STATE["budget"], "fps": STATE["fps"],
        "n_calls": len(t),
        "infer_ms_median": statistics.median(steady) * 1e3 if t else None,
        "infer_ms_p90": sorted(steady)[int(0.9 * (len(steady) - 1))] * 1e3 if t else None,
        "infer_ms_all": [x * 1e3 for x in t],
        "hold_steps": STATE["hold_steps"],
        "n_episodes": len(eps),
        "success_rate": sum(e["success"] for e in eps) / len(eps) if eps else None,
        "mean_steps_success": statistics.mean(e["steps"] for e in succ) if succ else None,
        "mean_actions_success": statistics.mean(e["actions"] for e in succ) if succ else None,
        "mean_steps_all": statistics.mean(e["steps"] for e in eps) if eps else None,
        "episodes": eps,
    }


def _output_dir(argv: list[str]) -> Path:
    for a in argv:
        if a.startswith("--output_dir="):
            return Path(a.split("=", 1)[1])
    return Path(".")


def main() -> None:
    sys.argv = [sys.argv[0]] + parse_own_args(sys.argv[1:])
    os.environ.setdefault("MUJOCO_GL", "egl")
    # LIBERO's first import asks via input() whether to use a custom dataset path; answer "no".
    import builtins
    builtins.input = lambda *a, **k: "n"
    import lerobot.scripts.lerobot_eval as E
    from lerobot.sim.integrations import libero as L

    patch_env(L.LiberoEnv)
    orig_make_policy = E.make_policy
    E.make_policy = lambda *a, **k: wrap_policy(orig_make_policy(*a, **k))
    ok = False
    try:
        E.main()
        ok = True
    finally:
        res = summary()
        out = _output_dir(sys.argv[1:])
        out.mkdir(parents=True, exist_ok=True)
        # only a completed run gets bottle_timing.json, so resumable sweeps retry failed runs
        name = "bottle_timing.json" if ok else "bottle_timing.failed.json"
        (out / name).write_text(json.dumps(res, indent=2))
        print(f"[bottle] mode={res['mode']} latency={res['latency_ms']}ms success={res['success_rate']} "
              f"episodes={res['n_episodes']} infer_median={res['infer_ms_median']}ms -> {out / name}")


if __name__ == "__main__":
    main()
