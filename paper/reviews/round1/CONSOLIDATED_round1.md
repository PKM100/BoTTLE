# Round-1 Review — CONSOLIDATED

Paper: *BoTTLE: Predicting Latency, Energy, and Cost per Successful Task for VLA Models on Hardware You Don't Own*
Reviewed after: initial draft (8 pages; E1 local transfer, E2 published validation, E3 LIBERO cost).

## VERDICTS

| Model | Family | Round 1 |
|-------|--------|---------|
| gpt-5.6-sol | GPT | **Major Revision 5/10** |
| claude-sonnet-5.5 | Claude | **Major Revision 5/10** |
| gemini-4-argon | Gemini | **Minor Revision 7/10** |
| grok-4.7 | Grok | **Major Revision 5/10** |
| claude-opus-5.5 (idea) | — | Idea is useful, but E3 is "arithmetic on borrowed numbers": success rates are not device- or latency-dependent |

**Headline:** 3× Major, 1× Minor. Everyone agrees overhead-aware calibration is a real, reproducible gain. The cost-ranking claims (E3) are not yet supported.

## Convergent required fixes (raised by ≥2 reviewers)
1. **"Within 2× for all 31" is false.** The `ape<=1.0` check misses underprediction; SmolVLA on H100 is predicted at 14.9 ms against 208 ms measured. Raised by all 4 referees and the advisor. *Blocking, writing-only, plus a script fix.*
2. **π0 is anchored on a JAX-compiled run while all other models are eager.** This confounds "fastest ≠ cheapest on 6 of 9". Raised by Claude, Gemini and Grok. *Blocking:* either re-anchor or label it and withdraw the claim.
3. **Counts don't match.** The text says 9 devices but tables show 8 (A10G missing), and the paper says "six" models but uses 8 presets. Raised by Claude, Gemini, Grok and GPT. *Writing-only.*
4. **E2 statistics.** The 32 pairs are correlated (about 8 source groups). Needed: group or bootstrap CIs, leave-one-group-out, and median/max APE. Raised by GPT, Claude, Grok and the advisor. *New analysis (CPU only).*
5. **Host-overhead mechanism is unproven.** The σ=1 ablation shows most of the gain is additive residual, not host scaling. Reframe it as an "additive residual", add a constant-residual baseline, and report a host-prior table. Grok says the 31.2%/35.2% sensitivity numbers don't match the code. Raised by GPT, Claude, Grok and the advisor. *Blocking: re-run plus writing.*
6. **E3 uncertainty.** Cost differences lie inside the predictor's error. Propagate E2 error, horizon assumptions and $/h rates into the costs; bold a winner only when it is robust. Raised by GPT, Claude, Grok and the advisor. *New analysis.*
7. **Sources for success rates and horizons.** Cite a per-suite table (GR00T N1.6 in particular) and sweep the assumed SmolVLA/GR00T horizons. Raised by GPT and Grok. *Writing plus a sweep.*
8. **No measured cost per success.** At least one real end-to-end validation is needed. Raised by GPT, Claude and the advisor. *NEW-EXPERIMENT (simulation).*

## Single-reviewer points worth considering
- Gemini: document the failure mode for A10G→H100 (~100% APE), where the residual comes from accelerator inefficiency rather than host overhead.
- Grok: use measured latency wherever the dataset has it and predictions only for missing cells. Also report per-model energy LOO and a u∈{0.2,0.6,1.0} sweep.
- Claude: add a network or tethering latency term for the datacenter (H100) scenario.
- GPT: validate memory-fit predictions, or label them as scenario estimates.
- Advisor: add decision metrics (ranking accuracy, cost regret of the recommended device).

## Advisor (idea/roadmap) synthesis
- **Weakest link:** LIBERO pauses physics during inference, so success never depends on latency. E3 just multiplies borrowed numbers.
- **Roadmap:**
  1. Closed-loop LIBERO with *emulated latency*: delay each chunk by ⌈ℓ·f⌉ sim steps to get latency→success curves for any device at $0. Run SmolVLA on the M5 and π0.5 on a Colab/Kaggle T4.
  2. Evaluation fixes and decision metrics (1–2 days, CPU).
  3. A T4 host-overhead mechanism test: add CPU contention, and compare eager with compiled/CUDA Graphs.

## STATUS
**Not converged.** Remaining blockers:
- Metric and claim fixes (#1–3, #5).
- Statistical treatment of E2 and E3 (#4, #6).
- One measured closed-loop cost-per-success validation (#8): the LIBERO simulation run.

## Revision status (after round 1)
- Applied: within-2x metric fix; group bootstrap + CIs; additive-residual baseline; host-prior sensitivity;
  eager π0 anchor (JAX as sensitivity) — "fastest ≠ cheapest" was a JAX-anchor artifact and is withdrawn;
  verified GR00T N1.6 / π0.5 success sources; measured-latency cells; 2,000-draw Monte Carlo with
  robust-winner marking (P ≥ 0.8); new appendices (per-group, success/horizons); expanded limitations.
- Built, not yet run: closed-loop latency emulation (`experiments/sim/bottle_libero_eval.py`,
  `notebooks/bottle_libero_sim.ipynb`). Running it on Colab is the remaining blocking experiment.
