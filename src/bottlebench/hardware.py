"""Hardware registry.

All numbers are approximate, dense (non-sparse) peaks collected from public
datasheets. Verify against vendor documentation before publishing results and
override with ``--hw-file`` (JSON) when you have better numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Hardware:
    name: str
    # Dense peak throughput in TFLOPS/TOPS keyed by compute precision
    # ("fp32", "fp16", "fp8", "int8", "fp4").
    peak_tflops: dict[str, float]
    mem_bw_gbs: float
    mem_gb: float
    tdp_w: float
    idle_w: float
    price_usd: float = 0.0
    cloud_usd_per_hr: float = 0.0
    kernel_launch_us: float = 5.0
    # Relative single-thread host CPU speed (1.0 ~ modern x86 server core). Prior from CPU
    # class/clock, not fitted: Thor Neoverse-V3AE 0.75, Orin Cortex-A78AE 0.45.
    host_speed: float = 1.0
    kind: str = "gpu"
    notes: str = ""
    source: str = "approximate public datasheet values; verify before citing"

    def peak(self, precision: str) -> tuple[float, str]:
        """Return (TFLOPS, precision actually used), falling back to fp16/fp32."""
        for p in (precision, "fp16", "fp32"):
            if p in self.peak_tflops:
                return self.peak_tflops[p], p
        raise KeyError(f"{self.name}: no peak for {precision}")

    def to_dict(self) -> dict:
        return asdict(self)


_BUILTIN: list[Hardware] = [
    Hardware(
        "t4", {"fp32": 8.1, "fp16": 65.0, "int8": 130.0}, 320, 16, 70, 10,
        price_usd=2000, cloud_usd_per_hr=0.35, kernel_launch_us=5, host_speed=0.8,
        notes="Turing. Free on Colab/Kaggle. Reference device for calibration.",
    ),
    Hardware(
        "l4", {"fp32": 30.3, "fp16": 121.0, "fp8": 242.0, "int8": 242.0}, 300, 24, 72, 15,
        price_usd=2500, cloud_usd_per_hr=0.70, kernel_launch_us=5, host_speed=1.0,
        notes="Ada. Colab paid tier.",
    ),
    Hardware(
        "rtx-4090", {"fp32": 82.6, "fp16": 165.2, "fp8": 330.3, "int8": 660.6}, 1008, 24, 450, 25,
        price_usd=1599, kernel_launch_us=4, host_speed=1.2,
        notes="Ada consumer GPU; fp16 tensor with fp32 accumulate. Common in VLA papers.",
    ),
    Hardware(
        "rtx-5090", {"fp32": 104.8, "fp16": 209.5, "fp8": 419.0, "fp4": 838.0}, 1792, 32, 575, 30,
        price_usd=1999, kernel_launch_us=4, host_speed=1.2,
    ),
    Hardware(
        "a10g", {"fp32": 31.2, "fp16": 125.0, "int8": 250.0}, 600, 24, 300, 20,
        price_usd=3500, cloud_usd_per_hr=1.0, kernel_launch_us=5, host_speed=1.0,
        notes="AWS g5 variant of A10; peaks and approx. price taken from A10.",
    ),
    Hardware(
        "l40", {"fp32": 90.5, "fp16": 181.0, "fp8": 362.0, "int8": 362.0}, 864, 48, 300, 35,
        price_usd=8000, cloud_usd_per_hr=1.0, kernel_launch_us=4, host_speed=1.0,
    ),
    Hardware(
        "a100-40gb", {"fp32": 19.5, "fp16": 312.0, "int8": 624.0}, 1555, 40, 400, 50,
        price_usd=10000, cloud_usd_per_hr=1.30, kernel_launch_us=4, host_speed=1.0,
    ),
    Hardware(
        "h100-sxm", {"fp32": 67.0, "fp16": 989.0, "fp8": 1979.0, "int8": 1979.0}, 3350, 80, 700, 70,
        price_usd=30000, cloud_usd_per_hr=2.50, kernel_launch_us=4, host_speed=1.0,
    ),
    Hardware(
        "h100-nvl", {"fp32": 60.0, "fp16": 835.0, "fp8": 1670.0, "int8": 1670.0}, 3900, 94, 400, 60,
        price_usd=30000, cloud_usd_per_hr=2.50, kernel_launch_us=4, host_speed=1.0,
    ),
    Hardware(
        "jetson-orin-nano-super", {"fp32": 1.3, "fp16": 17.0, "int8": 33.0}, 102, 8, 25, 5,
        price_usd=249, kernel_launch_us=10, host_speed=0.35, kind="edge",
        notes="67 INT8 sparse TOPS; dense values derived by halving.",
    ),
    Hardware(
        "jetson-agx-orin-64gb", {"fp32": 5.3, "fp16": 42.5, "int8": 85.0}, 204.8, 64, 60, 15,
        price_usd=1999, kernel_launch_us=8, host_speed=0.45, kind="edge",
        notes="GPU-only peaks (DLA excluded); 275 INT8 sparse TOPS headline incl. DLA.",
    ),
    Hardware(
        "jetson-agx-thor", {"fp32": 8.0, "fp16": 258.0, "fp8": 517.0, "int8": 517.0, "fp4": 1035.0},
        273, 128, 130, 20, price_usd=3499, kernel_launch_us=6, host_speed=0.75, kind="edge",
        notes="Headline 2070 FP4 sparse TFLOPS; other precisions derived by halving. "
              "Unified LPDDR5X shared with CPU.",
    ),
    Hardware(
        "cpu-generic", {"fp32": 1.0, "fp16": 0.5}, 50, 64, 65, 15,
        kernel_launch_us=1, host_speed=1.0, kind="cpu",
        notes="Placeholder for laptop/desktop CPU; calibrate with a profile.",
    ),
]

_REGISTRY: dict[str, Hardware] = {h.name: h for h in _BUILTIN}


def load_hw_file(path: str | Path) -> None:
    """Register/override hardware from a JSON file (object or list of objects)."""
    data = json.loads(Path(path).read_text())
    for item in data if isinstance(data, list) else [data]:
        hw = Hardware(**item)
        _REGISTRY[hw.name] = hw


def get_hardware(name: str) -> Hardware:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown hardware '{name}'. Known: {', '.join(_REGISTRY)}") from None


def list_hardware() -> list[Hardware]:
    return list(_REGISTRY.values())

