# BoTTLE — Idea-level assessment and roadmap (advisor: claude-opus-5.5, round 1)

Sources read: `paper/main.tex`, `paper/e1.tex`, `paper/tables/*.tex`, and the code that produces
them (`src/bottlebench/calibrate.py`, `analytical.py`, `hardware.py`, `experiments/published.py`,
`experiments/libero_cost.py`, `experiments/results/published.json`, `libero_cost.json`). I spot-checked
several claims against the code and outputs (noted inline). Budget constraint used throughout: no
hardware purchases and near-zero software spend. Available compute is free Colab/Kaggle T4s and an
Apple M5 (16 GB).

---

## 1. Idea quality now

**Today this is a solid workshop paper / tool paper. It is not yet a field-shaping framing, but it
is one experiment away from being one.**

The honest contribution, in order of real novelty:

1. **An empirical systems finding.** Published eager-mode VLA latencies are dominated by time the
   roofline cannot explain. That time behaves like an *additive, host-bound* term, not a
   *multiplicative* inefficiency. So the standard "measure once, scale by FLOPs/bandwidth ratio"
   calibration fails on the device class that matters most (Jetson), and an additive decomposition
   fixes most of the failure. This is the most defensible and most reusable result in the paper.
2. **An accounting framework.** Cost per *successful* task combines chunking, sync/async execution,
   retries, robot and operator time, and amortized device cost. It is useful and clearly argued.
   Technically, though, it is arithmetic: E3's headline findings follow almost directly from the
   formula plus the assumed parameters (execution horizon, prices, success rates).
3. **A curated dataset of 31 latencies and a dependency-free CLI.** Good community value with a
   low novelty bar.

**Has the ceiling moved?** Yes, upward, because of the overhead-aware result. "VLA inference on edge
is host-bound, so hardware-ratio predictions are wrong" is a statement people will cite. But the
headline the title promises, *cost per successful task*, still rests on a "per success" term that is
**imported, not measured or modeled**. Success rates are LIBERO numbers produced in a simulator that
waits for the policy. As a result, the paper's distinctive metric is today the product of one
validated quantity (latency, ±30%) and one quantity that is assumed independent of everything the
tool varies (device, latency, execution mode, precision). That ceiling has not moved.

---

## 2. Did the latest addition earn its keep?

I treat the overhead-aware calibration with its E2 validation, plus the local M5 experiment (E1), as
the latest additions.

- **Overhead-aware calibration plus E2: yes, strongly.** It turned a negative result (calibration
  makes Jetson predictions *worse*: 63% vs 54% MAPE) into the paper's best insight, and the
  σ-sensitivity analysis is honest. However, the paper's own sensitivity numbers expose a deeper
  limit. With σ = 1 everywhere, MAPE is already 31.2%; the host-speed priors buy only about 3 points
  (31.2 → 28.1). The win therefore comes from **additive vs multiplicative decomposition**, not from
  host-CPU modeling. The "scales with host-CPU speed" mechanism is asserted, not demonstrated. No
  experiment varies host speed while holding the accelerator fixed. Reframe the claim to what the
  data shows, then test the mechanism directly (roadmap item 3).
- **E1 (local M5): neutral.** It shows that the pipeline runs at zero cost, which is worth one
  paragraph. Scientifically, it cannot test the central hypothesis. All three "devices" share one
  host, so σ is constant. The synthetic profiler has little host overhead by construction. And the
  laptop's run-to-run spread (104 vs 167 ms) is comparable to the error being measured. Keep it, but
  shrink it.
- **E3 (LIBERO cost): it carries the title but adds the least new evidence.** The rankings depend on
  two *assumed* execution horizons (SmolVLA H = 10, GR00T H = 8), and GR00T N1.6 "wins" most
  columns. The successful-episode-length sensitivity check (0.4–1.0) only perturbs a parameter that
  scales every model nearly uniformly, so its invariance is unsurprising.

### Issues found while verifying claims (fix before any submission)

- **The "within 2×" claim is a metric artifact.** The paper says spec-only predictions are "within
  2× for all 31 cases". `published.py` defines `within_2x` as APE ≤ 100%. For *under*-prediction,
  that condition is always true. SmolVLA on H100 NVL is predicted at 14.9 ms against 208 ms measured,
  a 14× miss, and the paper itself cites it. Use symmetric log error (|log(pred/meas)| ≤ log 2), or
  report geometric-mean ratio and fold-error quantiles.
- **The 32 pairs are not 32 independent tests.** 18 come from two Isaac-GR00T README groups and 9
  from a single "oft-fused" group, so there are effectively about 8 groups. Report leave-one-group-out
  results and cluster-bootstrap CIs. Also state that the predictor was designed *after* seeing the
  Jetson failure on this same data. A held-out check is needed (roadmap item 6).
- **The "noise floor" argument conflates two different things.** The 1.95× spread is *cross-source*,
  but transfer is evaluated *within* source groups. Cross-source spread does not bound within-source
  transfer error, so "28% is close to what published data can support" does not follow. Say instead
  that the deployed stack can move latency by up to 2×, so calibrate on it.
- **Device count is inconsistent.** E3 sweeps nine devices (A10G included), and the text says "6 of
  9 devices", but Tables 2 and 3 show eight columns.

---

## 3. The weakest link now

**Success is treated as independent of latency, device, execution mode and precision.**

Everything that makes BoTTLE different from a latency calculator (synchronous stalls, async
staleness, real-time feasibility, the "cheap slow device vs fast expensive device" trade-off) only
matters if latency feeds back into whether the task succeeds. Currently it cannot. LIBERO's
simulator freezes time while the policy thinks, so an Orin Nano that takes 594 ms per π0 call is
credited with the same 94.2% success as an H100. The tool flags it as "not real-time", then prices it
as if it were. A reviewer will say that E3 is spreadsheet arithmetic on borrowed success rates, and
they will be right.

Fixing this is also the route to the field-shaping version of the paper. And, crucially, **it is
achievable with zero hardware**, because simulated time can be decoupled from wall-clock time
(Section 4.1).

---

## 4. How to improve the idea

### 4.1 Make success a function of latency: device-free closed-loop evaluation (the central move)

Key insight: **in simulation you do not need the target device to experience its latency.** You only
need to *emulate* it in simulator time. Wrap the LIBERO environment so that a policy call issued at
sim step t returns its action chunk at step t + k, where k = ⌈ℓ·f⌉. Take ℓ from BoTTLE's prediction
for the target device, or sweep it. The wall-clock speed of your T4 or M5 becomes irrelevant; it
only sets episode throughput. This gives a calibrated *latency→success* curve for each (model,
execution mode) pair, measured on free hardware, for any device you will never own.

Execution modes to emulate:

- **sync-hold:** the robot holds pose (zero delta-EEF action, gripper state held) for k steps, then
  executes the chunk. Mostly costs time; may cost success when objects settle or slip.
- **async-naive:** the previous chunk keeps executing. The new chunk is computed from the observation
  at step t and spliced in at t + k, skipping the first k actions (stale observation). This is where
  success should collapse once ℓ > H/f.
- **async with smoothing:** temporal ensembling (ACT-style) or a real-time-chunking-style inpainting
  of the overlap. This shows whether algorithmic mitigation moves the cost frontier, an axis no
  latency benchmark captures.

Output: p(ℓ, H, mode) per model. Feed it back into the cost model, so that cost per success uses
p(ℓ_device) rather than p₀. That is the paper's headline figure.

**Budget feasibility (this is how to run real closed-loop simulation for $0):**

- **Simulator.** LIBERO (robosuite on MuJoCo) runs on CPU. On Colab/Kaggle, use `MUJOCO_GL=egl` (or
  `osmesa` as a fallback). On the M5, MuJoCo runs natively; use `MUJOCO_GL=cgl` or glfw for
  offscreen rendering. Pin package versions; LIBERO's robosuite version is old, so use the fork that
  LeRobot or openpi pin.
- **Policies that fit the budget:**
  - *SmolVLA (0.45B):* the LeRobot LIBERO checkpoint runs comfortably on the M5 (MPS) and on a T4.
    **This is the workhorse.** The M5 has no quota, so run it overnight, repeatedly.
  - *π0 / π0.5 (≈3B):* LIBERO checkpoints exist via openpi/LeRobot. Weights fit a 16 GB T4 in fp16.
    The T4 has no bf16, so check for fp16 overflow first; fall back to fp32 on only the action expert
    if needed. Use Kaggle (about 30 free GPU-hours/week, 2×T4 sessions) for these.
  - *OpenVLA / OpenVLA-OFT (7B):* only with 4-bit quantization on a T4. That changes success itself,
    which is a useful data point (item 7) but not a clean baseline. Treat it as optional.
  - *GR00T N1.6:* skip at first. Its Turing/flash-attention compatibility on a T4 is a risk.
- **Episode budget.** At roughly 5–15 s per episode (sim and rendering dominate for SmolVLA), the M5
  does about 250–600 episodes per night. Use LIBERO-Spatial (short) and LIBERO-10/Long (where
  latency should hurt most). Run 10 tasks × 20–50 episodes × 6 latency levels {0, 50, 100, 200,
  400, 800 ms} × 2–3 modes. That is about 2.4k–9k episodes per model: one to three weeks of M5
  nights for SmolVLA, plus one or two weekly Kaggle quotas for π0.5. Use fixed seeds and shared
  initial states across latency levels (paired design), so that differences are detectable with
  fewer episodes than unpaired CIs would suggest. Report Wilson CIs.
- **Don't** attempt real-time (wall-clock) closed-loop runs on the T4 to "be the device". They are
  slower, noisier, quota-limited and emulate only one device. Emulated latency is strictly better
  for this question.

### 4.2 Reframe the latency claim to what the data supports

State the result as "VLA latency = accelerator term + additive overhead term; only the accelerator
term transfers with hardware ratios". Present host-speed scaling as a *second-order correction*
that is now tested (item 3) rather than asserted. Add a two-parameter baseline, "reference minus
roofline plus target roofline" (that is, σ = 1), as a named predictor in the table, so the reader
sees exactly where the gain comes from.

### 4.3 Evaluate the decisions, not only the errors

A deployment tool is judged by whether it picks the right configuration. Using groups that have at
least three measured devices (GR00T N1.6 has five, N1.7 has four, the OFT-fused group has four),
report:

- ranking fidelity (Kendall τ of predicted vs measured latency order);
- top-1 agreement of the cost-optimal device under each cost regime;
- **regret**: the % cost overpaid by choosing BoTTLE's pick instead of the measured-optimal one.

A 28% MAPE alongside, say, under 5% median regret is a much stronger claim than MAPE alone.

### 4.4 Propagate uncertainty into the recommendation

Replace point predictions with distributions: a lognormal latency error fitted from E2 residuals,
priors over the assumed execution horizons, success-rate CIs, and price ranges. Report
**P(configuration is cost-optimal)**. This turns the "±30%" caveat into a feature, honestly shows
which E3 conclusions are robust (likely: OpenVLA never wins; robot time flips device choice) and
which are fragile (GR00T vs OFT vs π0 at the margin), and costs one day.

---

## 5. What to build next (ordered by impact per effort)

Prior roadmap status, inferred from the paper state (no earlier roadmap document was provided).
**Done:** roofline + CLI, published-latency dataset, overhead-aware calibration with E2 validation,
local M5 E1, LIBERO cost study, energy leave-one-out check. **Newly most valuable:** closed-loop
latency→success (item 2) and the mechanism test for host overhead (item 3).

| # | Item | (a) Hypothesis | (b) Effort | (c) Payoff |
|---|------|----------------|------------|------------|
| 1 | **Evaluation hygiene + decision metrics** *(cheap, high value)* | Overhead-aware calibration remains best under leave-one-group-out and cluster bootstrap; median decision regret is small (under 10%) even at 28% MAPE. | 1–2 days, CPU only. Fix the `within_2x` bug (use symmetric fold error); add the σ = 1 additive baseline row; LOGO + cluster-bootstrap CIs; Kendall τ / top-1 / regret on the multi-device groups; fix the A10G column inconsistency; rewrite the noise-floor paragraph. | Removes the claims a reviewer would use to reject. Regret is the right metric for a decision tool and likely the strongest single number in the paper. |
| 2 | **Latency-emulated closed-loop LIBERO** (§4.1) *(core contribution)* | Success is flat in ℓ for sync-hold but drops sharply in async-naive once ℓ > H/f, and the drop is larger on LIBERO-Long. Smoothing (temporal ensembling or RTC-style) recovers part of it. Using p(ℓ) changes at least one E3 conclusion (e.g., Orin-class devices lose their compute-only win for some models, or async stops "hiding" latency). | 2–4 weeks. About 2 days for the env wrapper and harness (LeRobot LIBERO env + SmolVLA), then M5 overnight runs; π0.5 on Kaggle T4. $0. | Turns "cost per success" from arithmetic into a measured quantity and fixes the weakest link. Produces a new, reusable object: the **latency-sensitivity curve of a policy**. Highest-impact item by far. |
| 3 | **Mechanism test for the overhead term** | On a fixed accelerator (Colab T4), measured latency minus the roofline grows linearly with host slowdown. Induce slowdown with CPU contention (`stress`-style busy threads, `taskset` to fewer cores) or by profiling the CPU-side launch path. With CUDA Graphs / `torch.compile(mode="reduce-overhead")`, the unexplained term shrinks toward zero, and the roofline alone then predicts well. Run with *real* SmolVLA and π0 (LeRobot) checkpoints, including preprocessing, not only the synthetic profiler. | 3–5 days. Free T4 + M5 (eager vs compiled on MPS too). | Converts the paper's central assertion into a demonstrated mechanism. Adds about 10 own-measured, same-stack points to E2, with M5↔T4 transfer on real models. Shows the tool covers both the eager and the compiled regime. |
| 4 | **Uncertainty-propagated recommendations** (§4.4) | E3's coarse conclusions hold with P > 0.9; model-level winners on most devices are within the noise (P < 0.6). | 1 day. Monte Carlo over latency error, H, success CIs and prices. | Honest, robust E3, and a better CLI output ("P(optimal)") that differentiates the tool. |
| 5 | **Optimized-stack regime from published TensorRT / compiled numbers** | With overhead removed (TensorRT GR00T on Thor/Orin, JAX-compiled π0), spec-only roofline is within about 20%, and *global* calibration becomes adequate. The best predictor therefore depends on the stack regime, and BoTTLE can detect the regime from the reference measurement (m/ℓ̂ ratio). | 2–3 days of data curation (Isaac-GR00T TRT tables, openpi, systems papers). | Generalizes the finding from "a fix for eager stacks" to "a regime-aware predictor". Pre-empts the "everyone will compile anyway" objection. |
| 6 | **Prospective held-out validation** | The predictor, frozen today (tag a release, commit hashes of priors), achieves similar error on latencies published *after* the freeze. Candidates: new GR00T releases, new VLA systems papers, and community `bottle profile` submissions. | Low ongoing effort. Freeze now and collect over 1–3 months. | The cleanest answer to "you designed the method on the test set". Very persuasive at any venue. |
| 7 | **Precision → success coupling** | INT8/INT4 (bitsandbytes or torchao on T4; fp16 vs fp32 on MPS) cuts latency and memory enough to make a model fit or become real-time on a cheaper device, but costs measurable LIBERO success. The cost-optimal precision is not the fastest. | 3–5 days, reusing item 2's harness at ℓ = 0 plus emulated ℓ. | Completes the "success depends on configuration" story. Enables 7B OFT on a T4 as a data point. |
| 8 | **Moonshot: "LIBERO-RT", a device-aware closed-loop leaderboard** | Policies differ substantially in latency *robustness*, independent of nominal success. Ranking open checkpoints by predicted cost per success on named devices (Thor, Orin, 4090) reorders the standard LIBERO leaderboard. | 1–3 months. Standardize the emulation wrapper; add SimplerEnv/RoboCasa as a second benchmark if the CPU budget allows; accept community submissions; GitHub Actions + HF Spaces (free tier) host the board. | Field-shaping: it changes *how VLAs are evaluated*, making "success at deployable latency on device X" a standard axis. This is what gets cited and adopted. |

Explicitly deprioritized under the budget: renting Jetsons or H100s, real-robot validation, and
wall-clock real-time runs on the T4. Ask the authors of the two systems studies, or the Isaac-GR00T
maintainers, for raw per-stage traces instead; it costs only an email.

---

## 6. Venue and framing advice

**Where to submit:**

- **Now, as is (after item 1):** a CoRL/ICRA/RSS workshop on deployable or efficient robot learning,
  or a systems workshop at MLSys/NeurIPS. Short-paper framing: "Edge VLA latency is host-bound;
  calibrate additively."
- **With items 1–4 (plus 5 if possible):** **IISWC or ISPASS** (workload characterization +
  performance modeling is squarely in scope and values the dataset and tool), or the MLSys main
  track if the regime-aware predictor and regret results are strong.
- **With item 2 done well (plus 7):** **CoRL main track or RA-L**, or the **NeurIPS Datasets &
  Benchmarks** track if item 8 is shipped as a benchmark. This is the high-impact path; the robot
  learning community has no device-aware closed-loop evaluation, and they will use one.

**Framing (target version, with closed-loop results):**

- **Title:** something like *"What Does a Successful Robot Task Cost on Hardware You Don't Own?
  Device-Free Closed-Loop Evaluation of VLA Deployment"*. Alternative: *"Latency Is Not Free:
  Predicting Cost per Successful Task for Robot Foundation Models"*. Keep the BoTTLE name, but lead
  with the question, not the acronym.
- **Lead finding:** "Under emulated device latency, success of current VLAs drops X pp on
  [device/mode], so the cost-optimal device changes from A to B. Latency-only or success-only
  comparisons pick the wrong configuration Y% of the time." The host-bound latency result becomes
  the *enabling* finding (Section 3) that makes device latency predictable without the device.
- **Strengths to position:** (i) zero-hardware methodology end to end: emulated latency plus
  calibrated prediction; (ii) additive-overhead insight with a demonstrated mechanism; (iii) decision
  regret, not only MAPE; (iv) open dataset, CLI and harness reproducible on a free T4 or a laptop.
- **Honest negatives, stated up front as findings:** (a) published latencies for the same model and
  device vary up to 1.95× across software stacks, so *no* spec-based predictor can beat that
  without calibrating on the deployed stack; (b) emulated latency is not physical latency (no sensor
  jitter, network or actuator dynamics), so present it as a lower bound on degradation; (c) host
  priors are coarse and buy only about 3 MAPE points, and the additive decomposition does the
  heavy lifting.
- **Avoid:** leading with the dollar tables. The device-price-driven results (e.g., "a $249 Orin
  Nano is cheapest when only compute is billed") are true but obvious, and they invite the
  "spreadsheet" critique. Lead with what only BoTTLE can measure.

**Bottom line:** the latency half of BoTTLE is now credible. The *success* half is borrowed. Spend
the next month making success depend on latency via simulator-time emulation on the M5 and free
T4s; it costs nothing and is what turns a useful tool paper into a new evaluation axis for the
field.
