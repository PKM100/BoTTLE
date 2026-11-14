"""Architecture descriptions of robot foundation models (VLAs).

Presets are *approximate* shape-level descriptions reconstructed from public
papers/configs. They are meant for cost estimation, not reproduction. Override
or add models with ``--model-file`` (JSON with the same fields).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

STAGE_KINDS = ("encoder", "prefill", "denoise", "decode")


@dataclass
class Stage:
    """A stack of transformer layers executed ``repeats`` times per policy call.

    - encoder: vision tower, run once per camera image (``instances``)
    - prefill: VLM backbone over the multimodal prefix
    - denoise: diffusion/flow action head, ``repeats`` = number of steps,
      attending to ``context`` prefix tokens (KV cache)
    - decode: autoregressive action tokens, ``repeats`` = tokens generated,
      one token per pass, attending to ``context`` tokens
    """

    name: str
    kind: str
    n_layers: int
    d_model: int
    d_ff: int
    n_heads: int
    tokens: int
    n_kv_heads: int | None = None
    context: int = 0
    repeats: int = 1
    instances: int = 1
    gated_mlp: bool = True
    # weights read every pass outside the layer stack (e.g. LM head for decode)
    extra_params: float = 0.0
    # name of another stage whose weights this stage reuses (not double-counted in memory)
    shared_with: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in STAGE_KINDS:
            raise ValueError(f"stage {self.name}: kind must be one of {STAGE_KINDS}")
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads


@dataclass
class ModelSpec:
    name: str
    stages: list[Stage]
    action_chunk: int = 1  # actions produced per policy call
    action_dim: int = 7
    host_overhead_ms: float = 5.0  # preprocessing/tokenization on a 1.0-speed host
    description: str = ""
    source: str = "approximate; reconstructed from public papers/configs"

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        d = dict(d)
        d["stages"] = [Stage(**s) for s in d["stages"]]
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def _siglip_so400m(images: int, tokens: int = 256) -> Stage:
    return Stage("vision", "encoder", 27, 1152, 4304, 16, tokens, instances=images,
                 gated_mlp=False)


# Token counts follow the LIBERO configurations reported in arXiv 2611.06710 (Table 1)
# unless noted; layer shapes follow the public configs (openpi, LeRobot, HF GR00T configs).
_PRESETS: list[ModelSpec] = [
    ModelSpec(
        "smolvla", action_chunk=50, action_dim=6, host_overhead_ms=4.0,
        description="SmolVLA (450M): SmolVLM2 vision, first 16 LM layers, 0.75x-width flow expert, 10 steps.",
        stages=[
            Stage("vision", "encoder", 12, 768, 3072, 12, 1024, instances=2, gated_mlp=False),
            Stage("vlm", "prefill", 16, 960, 2560, 15, 149, n_kv_heads=5),
            Stage("action", "denoise", 16, 720, 2048, 15, 50, n_kv_heads=5, context=149, repeats=10),
        ],
    ),
    ModelSpec(
        "pi0", action_chunk=50, action_dim=7, host_overhead_ms=6.0,
        description="pi0 (3.3B): PaliGemma (SigLIP So400m + Gemma 2B) + 300M flow expert; 3 cameras, 10 steps.",
        stages=[
            _siglip_so400m(images=3),
            Stage("vlm", "prefill", 18, 2048, 16384, 8, 3 * 256 + 48, n_kv_heads=1),
            Stage("action", "denoise", 18, 1024, 4096, 8, 51, n_kv_heads=1, context=816, repeats=10),
        ],
    ),
    ModelSpec(
        "pi05", action_chunk=50, action_dim=7, host_overhead_ms=6.0,
        description="pi0.5: pi0 architecture with 200-token prompt; LIBERO 2 cameras (712 prefix tokens), 10 steps.",
        stages=[
            _siglip_so400m(images=2),
            Stage("vlm", "prefill", 18, 2048, 16384, 8, 712, n_kv_heads=1),
            Stage("action", "denoise", 18, 1024, 4096, 8, 50, n_kv_heads=1, context=712, repeats=10),
        ],
    ),
    ModelSpec(
        "openvla", action_chunk=1, action_dim=7, host_overhead_ms=5.0,
        description="OpenVLA (7.5B): SigLIP + DINOv2 -> Llama-2 7B (291 prefix tokens), 7 autoregressive action tokens.",
        stages=[
            _siglip_so400m(images=1),
            Stage("vision_dino", "encoder", 24, 1024, 4096, 16, 256, gated_mlp=False),
            Stage("vlm", "prefill", 32, 4096, 11008, 32, 291),
            Stage("action", "decode", 32, 4096, 11008, 32, 1, context=295, repeats=7,
                  extra_params=131e6, shared_with="vlm"),
        ],
    ),
    ModelSpec(
        "openvla-oft", action_chunk=8, action_dim=7, host_overhead_ms=5.0,
        description="OpenVLA-OFT (LIBERO): 3rd-person + wrist images, proprio, parallel decoding of an 8-step chunk (605 tokens).",
        stages=[
            _siglip_so400m(images=2),
            Stage("vision_dino", "encoder", 24, 1024, 4096, 16, 256, instances=2, gated_mlp=False),
            Stage("vlm", "prefill", 32, 4096, 11008, 32, 605),
        ],
    ),
    ModelSpec(
        "gr00t-n1.5", action_chunk=16, action_dim=32, host_overhead_ms=8.0,
        description="GR00T N1.5 (3B): Eagle (SigLIP2-400M + Qwen3-1.7B to layer 12) + 16-layer DiT, 4 steps, 1 camera.",
        stages=[
            Stage("vision", "encoder", 27, 1152, 4304, 16, 256, gated_mlp=False),
            Stage("vlm", "prefill", 12, 2048, 6144, 16, 94, n_kv_heads=8),
            Stage("action", "denoise", 16, 1536, 6144, 32, 49, context=94, repeats=4, gated_mlp=False),
        ],
    ),
    ModelSpec(
        "gr00t-n1.6", action_chunk=50, action_dim=32, host_overhead_ms=8.0,
        description="GR00T N1.6 (3B): Eagle-Block2A-2B to layer 16 + 32-layer DiT, horizon 50, 4 steps, 1 camera.",
        stages=[
            Stage("vision", "encoder", 27, 1152, 4304, 16, 256, gated_mlp=False),
            Stage("vlm", "prefill", 16, 2048, 6144, 16, 94, n_kv_heads=8),
            Stage("action", "denoise", 32, 1536, 6144, 32, 83, context=94, repeats=4, gated_mlp=False),
        ],
    ),
    ModelSpec(
        "gr00t-n1.7", action_chunk=40, action_dim=32, host_overhead_ms=8.0,
        description="GR00T N1.7 (3B): Cosmos-Reason2-2B (Qwen3-VL) to layer 12 + 16-layer DiT, horizon 40, 4 steps, 2 cameras.",
        stages=[
            Stage("vision", "encoder", 24, 1024, 4096, 16, 256, instances=2, gated_mlp=False),
            Stage("vlm", "prefill", 12, 2048, 6144, 16, 156, n_kv_heads=8),
            Stage("action", "denoise", 16, 1536, 6144, 32, 73, context=156, repeats=4, gated_mlp=False),
        ],
    ),
]

_REGISTRY: dict[str, ModelSpec] = {m.name: m for m in _PRESETS}


def load_model_file(path: str | Path) -> None:
    data = json.loads(Path(path).read_text())
    for item in data if isinstance(data, list) else [data]:
        m = ModelSpec.from_dict(item)
        _REGISTRY[m.name] = m


def get_model(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model '{name}'. Known: {', '.join(_REGISTRY)}") from None


def list_models() -> list[ModelSpec]:
    return list(_REGISTRY.values())

