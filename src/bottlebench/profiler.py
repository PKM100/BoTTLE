"""Measure per-stage latency/energy on a real device (e.g. free Colab/Kaggle T4).

Builds *shape-faithful* synthetic transformer stacks from a ModelSpec (random
weights; latency does not depend on weight values) so no gated checkpoints are
needed. Requires ``torch``; energy needs ``nvidia-ml-py`` on NVIDIA GPUs.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .models import ModelSpec, Stage


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise SystemExit("profiling needs torch: pip install 'bottle-bench[profile]'") from e
    return torch


def _build_block(torch, s: Stage, dtype, device):
    nn, F = torch.nn, torch.nn.functional
    d, h, kvh = s.d_model, s.n_heads, s.n_kv_heads
    hd = d // h

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.q = nn.Linear(d, h * hd, bias=False)
            self.kv = nn.Linear(d, 2 * kvh * hd, bias=False)
            self.o = nn.Linear(h * hd, d, bias=False)
            self.up = nn.Linear(d, (2 if s.gated_mlp else 1) * s.d_ff, bias=False)
            self.down = nn.Linear(s.d_ff, d, bias=False)

        def forward(self, x, ck=None, cv=None):
            B, T, _ = x.shape
            y = self.n1(x)
            q = self.q(y).view(B, T, h, hd).transpose(1, 2)
            k, v = self.kv(y).view(B, T, 2, kvh, hd).unbind(2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            if ck is not None:  # cached prefix keys/values (no recompute)
                k, v = torch.cat([ck, k], 2), torch.cat([cv, v], 2)
            if kvh != h:
                k, v = k.repeat_interleave(h // kvh, 1), v.repeat_interleave(h // kvh, 1)
            a = F.scaled_dot_product_attention(q, k, v)
            x = x + self.o(a.transpose(1, 2).reshape(B, T, h * hd))
            u = self.up(self.n2(x))
            if s.gated_mlp:
                g, u = u.chunk(2, -1)
                u = F.silu(g) * u
            else:
                u = F.gelu(u)
            return x + self.down(u)

    return Block().to(device=device, dtype=dtype).eval()


def _sync_fn(torch, device: str):
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    if device.startswith("mps"):
        return torch.mps.synchronize
    return lambda: None


def _time(torch, fn, device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync = _sync_fn(torch, device)
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


class _Energy:
    """NVML total-energy counter (Volta+). Returns None when unavailable."""

    def __init__(self, device: str):
        self.h = None
        if not device.startswith("cuda"):
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = int(device.split(":")[1]) if ":" in device else 0
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        except Exception:
            self.h = None

    def read_j(self) -> float | None:
        return None if self.h is None else self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1e3


def _check_device(torch, device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available; use --device cpu or mps")
    if device.startswith("mps") and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available; use --device cpu")


def probe(name: str, device: str = "cuda", precision: str | None = None, iters: int = 20) -> dict:
    """Microbenchmark achieved matmul throughput, memory bandwidth and per-op launch overhead.

    Returns a Hardware-compatible dict. Achieved (not datasheet) peaks, so use with
    efficiency factors of 1.0 or as a sanity check on datasheet values.
    """
    torch = _torch()
    _check_device(torch, device)
    peaks = {}
    precs = [precision] if precision else (["fp32"] if device == "cpu" else ["fp16", "fp32"])
    for p in precs:
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[p]
        n = 2048 if device == "cpu" else 4096
        a = torch.randn(n, n, device=device, dtype=dt)
        b = torch.randn(n, n, device=device, dtype=dt)
        t = _time(torch, lambda: a @ b, device, 3, iters)
        peaks["fp16" if p == "bf16" else p] = 2 * n**3 / t / 1e12
    nbytes = 256 * 2**20
    x = torch.empty(nbytes // 4, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    t = _time(torch, lambda: y.copy_(x), device, 3, iters)
    bw = 2 * nbytes / t / 1e9
    s = torch.ones(16, device=device)
    sync = _sync_fn(torch, device)
    reps = 200

    def many():
        for _ in range(reps):
            s.add_(1)
    launch_us = _time(torch, many, device, 2, 5) / reps * 1e6
    sync()
    return {"name": name, "peak_tflops": {k: round(v, 3) for k, v in peaks.items()},
            "mem_bw_gbs": round(bw, 1), "mem_gb": 0, "tdp_w": 0, "idle_w": 0,
            "kernel_launch_us": round(launch_us, 2), "kind": "cpu" if device == "cpu" else "gpu",
            "source": f"bottle probe on {device} (achieved, torch {torch.__version__})"}


def profile(model: ModelSpec, hardware: str, device: str = "cuda", precision: str | None = None,
            max_layers: int | None = None, warmup: int = 3, iters: int = 10,
            energy_seconds: float = 5.0) -> dict:
    torch = _torch()
    _check_device(torch, device)
    if precision is None:
        precision = "fp32" if device == "cpu" else "fp16"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(precision)
    if dtype is None:
        raise SystemExit("synthetic profiler supports fp16/bf16/fp32 (quantized kernels: use a real-model profile)")
    if max_layers is None and device == "cpu":
        max_layers = 2

    runners, stages_out = [], {}
    with torch.inference_mode():
        for s in model.stages:
            n = min(s.n_layers, max_layers or s.n_layers)
            blocks = [_build_block(torch, s, dtype, device) for _ in range(n)]
            hd = s.d_model // s.n_heads
            x = torch.randn(s.instances, s.tokens, s.d_model, device=device, dtype=dtype)
            ck = cv = None
            if s.context:
                ck = torch.randn(s.instances, s.n_kv_heads, s.context, hd, device=device, dtype=dtype)
                cv = torch.randn_like(ck)
            head = None
            if s.extra_params:
                head = torch.nn.Linear(s.d_model, int(s.extra_params // s.d_model), bias=False).to(device, dtype)

            def one_pass(blocks=blocks, x=x, ck=ck, cv=cv, head=head):
                y = x
                for b in blocks:
                    y = b(y, ck, cv)
                if head is not None:
                    head(y[:, -1])
                return y

            t_pass = _time(torch, one_pass, device, warmup, iters)
            scale = s.n_layers / n
            t = t_pass * scale * s.repeats
            stages_out[s.name] = {"time_s": t, "pass_s": t_pass, "layers_profiled": n,
                                  "layer_scale": scale, "repeats": s.repeats}
            runners.append((one_pass, scale * s.repeats))

        energy = _Energy(device)
        avg_power = energy_per_call = None
        if energy.h is not None:
            call_s = sum(v["time_s"] for v in stages_out.values())
            n_calls = max(1, int(energy_seconds / max(call_s, 1e-4)))
            e0, t0 = energy.read_j(), time.perf_counter()
            for _ in range(n_calls):
                for fn, reps in runners:
                    for _ in range(max(1, round(reps))):
                        fn()
            torch.cuda.synchronize()
            e1, t1 = energy.read_j(), time.perf_counter()
            avg_power = (e1 - e0) / (t1 - t0)
            energy_per_call = avg_power * call_s

    info = {"torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return {
        "model": model.name, "hardware": hardware, "precision": precision,
        "stages": stages_out, "latency_s": sum(v["time_s"] for v in stages_out.values()),
        "host_s": None, "avg_power_w": avg_power, "energy_j_per_call": energy_per_call,
        "method": "synthetic shape-faithful stacks", "env": info,
        "notes": ["host preprocessing not measured (synthetic inputs)"]
                 + (["layers extrapolated linearly"] if any(v["layer_scale"] > 1 for v in stages_out.values()) else []),
    }


def save(profile_dict: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(profile_dict, indent=2))
