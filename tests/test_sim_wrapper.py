import importlib.util
from collections import deque
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "ble", Path(__file__).parents[1] / "experiments/sim/bottle_libero_eval.py")
ble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ble)


class FakeEnv:
    def __init__(self, success_at=None, limit=20):
        self._max_episode_steps, self.success_at, self.task_id, self.actions = limit, success_at, 0, []

    def reset(self):
        self.n = 0
        return None, {}

    def step(self, a):
        self.actions.append(np.asarray(a).copy())
        self.n += 1
        ok = self.success_at is not None and self.n >= self.success_at
        return None, 0.0, ok, False, {"is_success": ok}


class FakePolicy:
    def __init__(self, h):
        self.h, self._action_queue = h, deque()

    def select_action(self, batch):
        if not self._action_queue:
            self._action_queue.extend(np.full(7, 0.5) for _ in range(self.h))
        return self._action_queue.popleft()


def setup(mode, lat_ms=0.0, budget="steps"):
    ble.STATE.update(mode=mode, latency_ms=lat_ms, budget=budget, fps=20, pending_hold=0, calls_this_episode=0,
                     infer_s=[], episodes=[], hold_steps=0)
    cls = type("E", (FakeEnv,), {})
    ble.patch_env(cls)
    return cls


def run(env, pol, max_calls=100):
    env.reset()
    for _ in range(max_calls):
        _, _, term, trunc, _ = env.step(pol.select_action(None))
        if term or trunc:
            return term


def test_parse_args_strips_bottle_options():
    ble.STATE.update(mode="none", latency_ms=0.0, fps=20)
    rest = ble.parse_own_args(["--bottle.mode=emulate", "--bottle.latency_ms=120", "--env.fps=10", "--x=1"])
    assert rest == ["--env.fps=10", "--x=1"]
    assert (ble.STATE["mode"], ble.STATE["latency_ms"], ble.STATE["fps"]) == ("emulate", 120.0, 10)


def test_emulated_latency_inserts_holds_per_chunk():
    cls = setup("emulate", 120.0)  # ceil(0.12*20) = 3 hold steps per chunk
    env, pol = cls(success_at=None, limit=1000), ble.wrap_policy(FakePolicy(h=5))
    env.reset()
    for _ in range(10):  # 2 chunks
        env.step(pol.select_action(None))
    assert env._bottle_t == 10 + 2 * 3
    holds = [a for a in env.actions if a[0] == 0]
    assert len(holds) == 6 and all(h[-1] == -1.0 for h in holds[:3]) and all(h[-1] == 0.5 for h in holds[3:])


def test_holds_consume_budget_and_truncate():
    cls = setup("emulate", 1000.0)  # 20 holds per chunk, limit 20 -> never acts
    env, pol = cls(success_at=None, limit=20), ble.wrap_policy(FakePolicy(h=5))
    assert run(env, pol) is False
    ep = ble.STATE["episodes"][-1]
    assert ep["steps"] == 20 and ep["truncated"] and not ep["success"]


def test_none_mode_matches_stock():
    cls = setup("none")
    env, pol = cls(success_at=7, limit=20), ble.wrap_policy(FakePolicy(h=5))
    assert run(env, pol) is True
    s = ble.summary()
    assert s["hold_steps"] == 0 and s["episodes"][0]["steps"] == 7 and s["n_calls"] == 2
    assert s["success_rate"] == 1.0


def test_action_budget_excludes_holds():
    cls = setup("emulate", 1000.0, budget="actions")  # same setting that never acts under a step budget
    env, pol = cls(success_at=None, limit=20), ble.wrap_policy(FakePolicy(h=5))
    assert run(env, pol) is False
    ep = ble.STATE["episodes"][-1]
    assert ep["actions"] == 20 and ep["steps"] == 20 + 4 * 20 and ep["truncated"]
