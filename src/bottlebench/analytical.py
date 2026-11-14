"""Hardware-free latency/memory/energy estimation via a per-stage roofline model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .hardware import Hardware
from .models import ModelSpec, Stage

# precision -> (weight bytes, activation bytes, compute precision key)
PRECISIONS: dict[str, tuple[float, float, str]] = {
    "fp32": (4, 4, "fp32"),
    "fp16": (2, 2, "fp16"),
    "bf16": (2, 2, "fp16"),
    "fp8": (1, 1, "fp8"),
    "int8": (1, 1, "int8"),
    "w8a16": (1, 2, "fp16"),
    "w4a16": (0.5, 2, "fp16"),
    "fp4": (0.5, 1, "fp4"),
}

KERNELS_PER_LAYER = 12  # rough count of launches per transformer layer (no CUDA graphs / fusion)


@dataclass
class Efficiency:
    """Fraction of datasheet peak actually achieved (uncalibrated defaults)."""

    compute: float = 0.5
    memory: float = 0.75
    power_util: float = 0.6  # avg power = idle + util * (tdp - idle) while inferring


@dataclass
class StageEstimate:
    name: str
    kind: str
    params: float
    flops: float
    bytes: float
    compute_s: float
    memory_s: float
    launch_s: float
    time_s: float
    bound: str


@dataclass
class Estimate:
    model: str
    hardware: str
    precision: str
    compute_precision: str
    stages: list[StageEstimate]
    host_s: float
    latency_s: float
    params: float
    weights_gb: float
    kv_gb: float
    memory_gb: float
    fits: bool
    avg_power_w: float
    energy_j: float
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def stage_params(s: Stage) -> float:
    hd = s.d_model / s.n_heads
    kv = s.n_kv_heads * hd
    attn = 2 * s.d_model * s.d_model + 2 * s.d_model * kv
    mlp = (3 if s.gated_mlp else 2) * s.d_model * s.d_ff
    return s.n_layers * (attn + mlp) + s.extra_params


def _stage_pass(s: Stage, wb: float, ab: float) -> tuple[float, float, float, float]:
    """Return (params, flops, bytes, kv_bytes_resident) for a single pass."""
    hd = s.d_model / s.n_heads
    kv = s.n_kv_heads * hd
    params = stage_params(s)
    tok = s.tokens * s.instances
    span = s.context + s.tokens
    flops = 2 * params * tok + s.n_layers * 4 * s.instances * s.tokens * span * s.d_model
    weight_bytes = params * wb
    act_bytes = s.n_layers * tok * (8 * s.d_model + 3 * s.d_ff) * ab
    kv_bytes = s.n_layers * s.instances * span * 2 * kv * ab
    return params, flops, weight_bytes + act_bytes + kv_bytes, kv_bytes


def estimate(model: ModelSpec, hw: Hardware, precision: str = "fp16",
             eff: Efficiency | None = None) -> Estimate:
    eff = eff or Efficiency()
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {list(PRECISIONS)}")
    wb, ab, ckey = PRECISIONS[precision]
    peak, used = hw.peak(ckey)
    notes = []
    if used != ckey:
        notes.append(f"{hw.name} has no {ckey} peak; using {used}")

    stages, total_params, kv_total = [], 0.0, 0.0
    for s in model.stages:
        params, flops, nbytes, kv_bytes = _stage_pass(s, wb, ab)
        c = flops / (peak * 1e12 * eff.compute)
        m = nbytes / (hw.mem_bw_gbs * 1e9 * eff.memory)
        launch = s.n_layers * KERNELS_PER_LAYER * hw.kernel_launch_us * 1e-6
        t = s.repeats * (max(c, m) + launch)
        stages.append(StageEstimate(
            s.name, s.kind, params, flops * s.repeats, nbytes * s.repeats,
            c * s.repeats, m * s.repeats, launch * s.repeats, t,
            "compute" if c >= m else "memory",
        ))
        if not s.shared_with:
            total_params += params
        kv_total = max(kv_total, kv_bytes)

    host = model.host_overhead_ms * 1e-3 / max(hw.host_speed, 1e-6)
    latency = sum(st.time_s for st in stages) + host
    weights_gb = total_params * wb / 1e9
    kv_gb = kv_total / 1e9
    mem_gb = weights_gb * 1.1 + kv_gb + 0.5  # 10% + 0.5GB runtime/activation slack
    power = hw.idle_w + eff.power_util * (hw.tdp_w - hw.idle_w)
    fits = mem_gb <= hw.mem_gb
    if not fits:
        notes.append(f"needs ~{mem_gb:.1f} GB > {hw.mem_gb} GB available")
    return Estimate(
        model.name, hw.name, precision, used, stages, host, latency, total_params,
        weights_gb, kv_gb, mem_gb, fits, power, latency * power, notes=notes,
    )
