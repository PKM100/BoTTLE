# BoTTLE

**B**enchmark **o**f **T**ask-level **T**otal **L**atency & **E**nergy: cost per *successful* robot task for
robot foundation models (VLAs), estimated on hardware you don't own.

Existing VLA efficiency benchmarks report per-call latency on hardware they have. BoTTLE asks a different question:
*what does it cost (seconds, joules, dollars) to get one successful task done with model X on device Y?* It combines:

1. **Roofline estimation**: per-stage (vision encoder / VLM prefill / action denoise / autoregressive decode)
   FLOPs and bytes moved, from datasheet specs. No hardware needed.
2. **Calibration**: one measurement on any device you have (free Colab T4, your laptop's GPU, or a published
   number) is transferred to the target. `--calib-mode overhead` treats time the roofline can't explain as
   host/framework overhead that scales with host CPU speed; on 32 published cross-device pairs it cuts
   latency error from 49% (single-factor) to 28% MAPE.
3. **Task-level cost**: action chunking, sync vs async execution, real-time feasibility, retries
   (1 / success rate), energy, amortized hardware or cloud price, and robot time.

## Install

```bash
pip install -e .            # core: zero dependencies
pip install -e '.[profile]' # + torch for `bottle profile`
pip install -e '.[energy]'  # + NVML energy counters
```

## Quick start

```bash
bottle hw list
bottle models list
bottle estimate --model pi0 --hw t4 jetson-agx-thor
bottle task --model openvla-oft --hw jetson-agx-orin-64gb --success 0.97 --steps 300 --control-hz 20
bottle sweep --hw t4 jetson-agx-orin-64gb jetson-agx-thor --precisions fp16 w4a16 \
    --success smolvla=0.87,pi0=0.94 --robot-usd-per-hr 10 --out results/sweep.csv
```

Calibrate on a free T4 (see `notebooks/bottle_colab.ipynb`):

```bash
bottle profile --model pi0 --hw-name t4 --device cuda --out profiles/pi0_t4_fp16.json
bottle estimate --model pi0 --hw jetson-agx-thor --profile-dir profiles
```

Test whether latency changes success: `notebooks/bottle_libero_sim.ipynb` runs closed-loop LIBERO on a free
Colab T4, holding the robot still for the measured (or BoTTLE-predicted) inference latency of each chunk
(`experiments/sim/bottle_libero_eval.py`, a wrapper around LeRobot's `lerobot-eval`).

Calibrate from a published end-to-end number (e.g. SmolVLA eager on H100 NVL, 208 ms):

```bash
echo '{"model":"smolvla","hardware":"h100-nvl","precision":"bf16","stages":{"_total":{"time_s":0.2083}}}' > p.json
bottle estimate --model smolvla --hw jetson-agx-thor --profile p.json --calib-mode overhead   # ~294 ms (measured 308)
```

Microbenchmark your own device (peak FLOPs, bandwidth, launch overhead) into a hardware JSON:

```bash
bottle probe --device mps --name m5-gpu --out m5_gpu.json   # cuda | mps | cpu
bottle --hw-file m5_gpu.json estimate --model pi0 --hw m5-gpu
```

Check predictions against published or measured numbers:

```bash
bottle validate --predicted pred.json --measured measured.json   # {case_id: seconds}
```

## Cost model

For an episode of `S` control steps at `f` Hz, executing `h` actions per policy call with latency `L`:

- calls `= ceil(S / h)`; sync attempt time `= S/f + calls·L`; async `= max(S/f, calls·L) + L`
- real-time feasible (async) iff `L ≤ h / f`
- energy `= calls·L·P_infer + idle·(rest) + robot power·time`
- dollars `= energy·$/kWh + time·(hardware amortization or cloud $/hr + robot $/hr)`
- per success `= per attempt / success_rate`

## Experiments and paper

```bash
python experiments/published.py     # E2: validate vs 31 published latencies (experiments/data/published_latency.json)
python experiments/libero_cost.py   # E3: cost per successful LIBERO task, 6 VLAs x 9 devices
python experiments/transfer.py      # E1: local CPU/GPU cross-device transfer (Apple silicon or CUDA)
python experiments/figures.py && python experiments/tables.py
cd paper && tectonic main.tex
```

## Extending

- Hardware: `--hw-file my_hw.json` (fields of `bottlebench.hardware.Hardware`).
- Models: `--model-file my_model.json` (fields of `bottlebench.models.ModelSpec`; `bottle models show pi0` for an example).

## Caveats

- Hardware specs and model architectures are **approximate** (dense peaks; Jetson values partly derived from
  sparse headline numbers). Verify against datasheets and configs before publishing; override via JSON.
- The synthetic profiler measures fp16/bf16/fp32 transformer stacks only. It doesn't capture host preprocessing,
  quantized kernels or fused/compiled runtimes (TensorRT). Calibrate with a GPU, not a CPU, for GPU targets.
- Success rates are user inputs (from simulation such as LIBERO, or from papers). BoTTLE doesn't model how latency
  changes success.
- Published latencies for the same model and device differ by up to 1.95x across software stacks. Calibrate on
  the stack you will deploy; read predictions as roughly +/-30%.
