"""One-shot Colab/Kaggle setup for the BoTTLE LIBERO simulation (Linux + NVIDIA GPU).

Lives in the repo (not the notebook) so fixes apply without re-opening the notebook:
    python /content/BoTTLE/experiments/sim/colab_setup.py
"""

import os
import subprocess
import sys

LEROBOT_REV = "fe1c54594f6ba275fccf210b9f7b11f0e2324b89"  # bottle_libero_eval.py patches lerobot internals
PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sh(cmd: str) -> str:
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-8000:])
        raise SystemExit(f"setup failed at: {cmd}")
    return r.stdout


def main() -> None:
    env = os.environ
    env["MUJOCO_GL"] = "egl"
    # egl_probe 1.0.2 declares cmake_minimum_required(2.8.12), which CMake 4 rejects without this
    env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
    sh("apt-get -qq install -y libegl1 libgl1 build-essential > /dev/null 2>&1 || true")
    # egl_probe / hf-egl-probe (LIBERO deps) compile with CMake. pip's `cmake` is a Python shim that breaks
    # inside pip's isolated build env, so build them without isolation. cmake<4 also satisfies lerobot.
    sh(f'{PY} -m pip install -q "cmake>=3.29.0.1,<4" setuptools wheel')
    sh(f"{PY} -m pip install -v --no-build-isolation hf-egl-probe==1.0.2 egl_probe==1.0.2")
    sh(f'{PY} -m pip install -q "lerobot[libero,smolvla] @ git+https://github.com/huggingface/lerobot.git@{LEROBOT_REV}"')
    sh(f"{PY} -m pip install -q -e {ROOT}")
    print(sh(f"{PY} -c \"import lerobot, libero, torch, bottlebench; "
             f"print('OK lerobot | torch', torch.__version__, '| cuda', torch.cuda.is_available())\""))
    print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv"))


if __name__ == "__main__":
    main()
