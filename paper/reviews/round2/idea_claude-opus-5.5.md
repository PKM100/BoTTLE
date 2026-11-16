# BoTTLE — Round 2 idea assessment (claude-opus-5.5)

Basis: full read of `paper/main.tex` (incl. `e1.tex`, all tables, appendices), plus
`experiments/sim/bottle_libero_eval.py`, `experiments/data/libero_sim_runs.json` and
`experiments/results/sim.json` to check how E4 actually works.

---

## 1. Idea quality now

**Verdict: a solid, honest, useful conference/journal paper with a field-shaping *metric* inside it,
but the paper is still built around the part that is less likely to shape the field: the predictor.**

The contributions, roughly in order of how much they matter:

1. **The metric and its decomposition.** Cost per *successful* task, with
   `D_sync = N/f + C·ℓ`, retries as `1/p`, and billing regimes (compute / +robot / +operator). The
   most quotable result in the paper is not a MAPE. It is *"who pays for what decides the device"*:
   a $249 Orin Nano wins on compute-only cost (P>0.99), and tethered desktop GPUs win once robot
   time is billed. Robotics benchmarking needs this lens, and people will cite it.
2. **The empirical observation behind the predictor.** Published eager VLA latency is mostly time
   the roofline can't explain (geometric mean 2.4×). The same model on the same device varies by
   up to 1.95× across software stacks. So FLOP- or bandwidth-based device choice is badly wrong for
   today's VLAs. That is a real systems finding.
3. **Overhead-aware (additive-residual) calibration.** Of the 48.8→28.1% MAPE gain, most
   (48.8→32.8) comes from "add the residual instead of multiplying it". The host prior adds 4.7
   points. The result is correct and well controlled, but it is close to a one-line insight, and
   the paper itself says it is bounded by the stack noise floor. It is a means, not a headline.
4. **E4, the latency-emulating closed-loop wrapper.** It is new to the paper and the right *kind*
   of experiment, but in its current form it is weaker than it looks (Section 2).

**Has the ceiling moved?** Slightly up, mostly because E4 shows the authors will test the
"success ⟂ latency" assumption instead of just stating it. That was the biggest conceptual hole
from round 1. But the result can't yet tell a control effect apart from a budget-accounting
effect, so the impact ceiling has not truly been raised. A paper that showed
**"success is a function of latency, here is the curve, and here is how it changes which
configuration is cheapest"** would be field-shaping. The current paper shows "success drops when
you charge stalls against the clock", which is close to true by construction.

**Honest contribution in one sentence:** *a validated, uncertainty-aware calculator for VLA cost
per successful task on unowned hardware, plus evidence that host overhead, not accelerator
throughput, sets VLA latency and that billing assumptions, not model choice, decide the device.*

---

## 2. Did the latest addition earn its keep?

**Partly. It closed a credibility gap but exposed a deeper limit.**

What E4 does well:
- It answers round 1's "no closed-loop validation" complaint with a real experiment and paired
  statistics (task bootstrap, sign-flip). Seeds are identical across settings, and the
  1,500 ms collapse is attributed honestly.
- It produced an **unplanned prospective test** of the predictor, the most valuable thing it
  found. BoTTLE predicted 399 ms (fp32) and measured 624 ms (−36%). E3 used 281 ms, assuming fp16,
  so the E3 cell for SmolVLA/T4 is off by 2.2× for the checkpoint as shipped. This should be
  promoted, not buried. It is the only non-retrospective latency number in the paper, and it is a
  miss outside the typical error band.
- The `s0/sℓ` factor (1.04/1.05/1.19) is a clean bridge between E4 and E3.

What it exposes:
- **E4's hold model is the same mechanism E3's `D_sync` already charges, so its "success loss" is
  mostly budget truncation.** In `bottle_libero_eval.py` the robot holds with zero EE delta while
  a scene that is almost entirely *quasi-static* "keeps evolving". LIBERO-Spatial objects don't
  move unless touched. The observation the policy acts on is stale only by a hold during which
  nothing moved. So in synchronous mode a stall adds almost no information error. Its only
  first-order effect is spending the 280-step clock (630 ms → 130 actions, 1,500 ms → 70). The
  paper says this for 1,500 ms but still presents 630 ms (−13, p=0.19) as "success falls". Most
  likely tasks 4 and 9, the ones that degrade, are simply the ones whose successful stock episodes
  use more than ~130 actions. If so, E4 is mostly an arithmetic restatement of
  "time limit / (1 + holds/H)" against the stock episode-length distribution.
- **The test is underpowered at the one point that matters.** The Thor and 630 ms settings, where
  E3's conclusions could flip, have CIs of ±12–15 points, so "holds up to 308 ms" really means
  "no detectable effect at n=50".
- **Stock baseline mismatch.** Stock SmolVLA spatial is 80% here, against the 90% E3 uses, and one
  task (task 5) is 0/10 under stock. E3 also uses a 220-step spatial limit while E4 uses LeRobot's
  280. These are small but visible inconsistencies between the two halves of the
  "latency ↔ success" story.
- **Only synchronous execution.** The interesting failure mode on real robots is *asynchronous*
  execution: acting on stale observations while moving, plus discontinuities at chunk boundaries
  (the problem real-time chunking/RTC-style methods address). E4 doesn't touch it, and E3's async
  rows still assume latency-independent success.

Net: E4 strengthens the paper's *integrity* and turns a hidden assumption into a tested one. It
does not yet strengthen the *thesis*, because the effect it measures is the one the cost model
already prices. The in-progress action-budget rerun is exactly the right next step.

---

## 3. The weakest link now

**The success–latency coupling is still not identified, so cost-per-success is in practice
"cost-per-attempt ÷ a latency-independent published success rate", which is precisely what the
paper argues is insufficient.**

Every model-vs-model and device-vs-device conclusion in E3 depends on `p` being fixed per model.
E4 shows `p` *does* depend on latency once wall-clock limits apply. But it covers one model, one
suite, sync only, and mostly the budget effect, so it doesn't tell a reader whether the E3 rankings
survive. The model comparison is also confounded: success rates come from different codebases and
the predictor misses prospectively by 1.56×. The paper's most important claim, that *the cheapest
configuration per success is X*, therefore rests on an untested independence assumption for 5 of
6 models and on success numbers it did not produce. The predictor is good enough. The success side
of the ratio is now the bottleneck.

(Runner-up: there is still no prospective, held-out *latency* validation beyond one accidental T4
point, and that point missed.)

---

## 4. How to improve the idea further

1. **Make the success side a first-class, measured function: `p(ℓ, mode, budget)`.**
   Reframe E4 from "does latency hurt?" into a *latency–success curve* that BoTTLE consumes. Report
   three cleanly separated mechanisms:
   (a) **budget truncation**, predictable analytically from the stock episode-length survival
   curve;
   (b) **sync stall effects at fixed budget**, from the action-budget rerun;
   (c) **async staleness**, where the policy observes at t, the robot keeps executing the previous
   chunk, and the new chunk lands at t+ℓ.
   If (a) explains the sync data, say so and turn it into a *zero-simulation* correction BoTTLE
   applies automatically from published episode-length statistics. That is a stronger, more
   general result than one more simulation point.
2. **Add dynamics so latency can matter physically.** LIBERO is quasi-static. Add a perturbed
   variant (object nudged mid-episode, a moving target, or a conveyor/push task in
   robosuite/ManiSkill), or use a benchmark with genuinely dynamic tasks. Without this, any
   "latency doesn't hurt" result is partly a property of the benchmark.
3. **Close the loop for at least two more models with different horizons.** Use GR00T N1.6 (H=8,
   the E3 winner) and π0/π0.5 (H=5) via LeRobot/openpi LIBERO, so `s0/sℓ` is measured for the
   configurations whose ranking it could flip. Then fold `p(ℓ)` into the Monte Carlo and re-report
   robust winners. That is when E3 becomes a *closed-loop* cost study.
4. **Prospective latency validation by pre-registration.** Before measuring, publish BoTTLE's
   predictions (with intervals) for a handful of (model, device, precision) cells you can rent
   cheaply: T4/L4/A10G/A100 on cloud, plus any Jetson a collaborator owns. Then measure. A
   prospective hit rate inside the predicted interval is worth more than any further retrospective
   MAPE tuning.
5. **Precision as a first-class axis.** The T4 miss was an fp16-vs-fp32 assumption. Make
   deployment precision explicit in every cost table, and validate the cross-precision transfer
   (E1's strongest result, 57%→25%) on a CUDA device.
6. **Reframe the predictor as "calibrate on your stack" tooling, not a model contribution.** Lean
   into `bottle probe` + one measurement + additive residual as the recommended workflow, with the
   1.95× stack spread as the reason.

---

## 5. What extra to build (ordered by impact per effort)

**Status of prior roadmap items.**
Done: group-bootstrap CIs and LOGO; additive-residual ablation and host-prior sensitivity; eager
π0 anchor; sourced success rates and horizons; Monte Carlo robust winners; first closed-loop test
(E4).
Still open: closed-loop for more than one model; folding `s0/sℓ` into the Monte Carlo; any
prospective latency validation; a measured deployment bill.

| # | Item | (a) Hypothesis | (b) Effort | (c) Expected payoff |
|---|------|----------------|-----------|---------------------|
| 1 | **Truncation-only predictor from stock logs (cheap, high value)** | Under sync holds, success at latency ℓ ≈ P(stock successful episode needs ≤ ⌊280·H/(H+⌈ℓf⌉)⌋ actions). The wrapper already logs per-episode `actions`/`truncated`, so this reuses existing runs. | 0.5–1 day, no GPU beyond re-reading logs (or one stock rerun with logging kept) | Decides whether E4 is a control effect or arithmetic. Either answer is publishable. If it fits, BoTTLE gains an analytic `p(ℓ)` correction for every model with published episode lengths. |
| 2 | **Finish the action-budget rerun (630/1,500 ms) and add 308 ms at n=100** | With holds off the budget, sync success at 630 ms returns to ≈ stock (within ±5), and 1,500 ms recovers most of the way. | 1–2 Colab days | Separates mechanism (a) from (b), and tightens the Thor CI, which is the decision-relevant point for E3's embedded conclusions. |
| 3 | **Fold `p(ℓ)` into E3's Monte Carlo and update the SmolVLA/T4 cell to measured fp32** | Desktop and Thor winners are unchanged, while Orin Nano/AGX Orin rankings shift under billed robot time. | 1 day | Removes the stated limitation "rankings don't include s0/sℓ", makes the paper internally consistent, and gives a direct answer to "does closing the loop change the decision?". |
| 4 | **Second and third closed-loop models: GR00T N1.6 (H=8) and π0.5 (H=5)** on LIBERO-Spatial/Long, sync and budget-charged | Shorter horizons (more calls) and longer suites amplify latency sensitivity, so `s0/sℓ` grows with C = N/H. | 1–2 weeks (checkpoint and env plumbing, A100/L4 rental ≈ $50–150) | Makes "success depends on latency" a cross-model law rather than a SmolVLA anecdote. This is the biggest credibility upgrade available. |
| 5 | **Async staleness emulation** | In async mode the robot keeps executing the old chunk while the policy runs on an observation ℓ old. Success degrades smoothly with ℓ/(H/f), and naive chunk switching breaks earlier than sync. | 3–5 days (wrapper change: delayed observation buffer plus chunk-switch logic; optionally temporal ensembling or RTC-style blending as a mitigation arm) | Tests the case E3's async rows assume away, connects BoTTLE to the active real-time-chunking literature, and makes "real-time iff ℓ ≤ H/f" an empirically grounded threshold. |
| 6 | **Pre-registered prospective latency study** on 4–6 rentable cells (e.g., SmolVLA/π0.5/GR00T on T4, L4, A10G, A100, at fp16 and fp32) | Overhead-aware calibration from one reference cell lands within its E2-derived 80% interval for ≥ 70% of cells. | 3–5 days, ≈ $50 cloud | Converts the "retrospective validation" caveat into a prospective claim. Turns the T4 miss into a diagnosed data point (host vCPU prior and precision). |
| 7 | **Dynamic-task variant** (perturbed LIBERO: object displaced at a random time; or a ManiSkill moving-target or push task) | Once the scene moves, sync stalls cost success even at a fixed budget, and the 308 ms "safe" region shrinks. | 1–2 weeks | Prevents the main threat to the conclusion, "LIBERO is too static to care", and shows the latency–success curve is task-dependent. BoTTLE should then take a task "dynamism" input. |
| 8 | **Moonshot: a measured bill on a real low-cost robot** (SO-100/SO-101 arm + SmolVLA, policy on Orin Nano vs. tethered 4090 vs. cloud T4; wall-clock, wall-power meter, attempts to success) | The computed cost per success matches the measured one within the Monte Carlo interval, and the compute-only vs. robot-billed device ranking flip happens on real hardware. | 4–8 weeks, ≈ $600–1,000 hardware | Removes the "no measured bill" limitation and gives the first end-to-end validated cost-per-success number for a VLA. This is the result that would make BoTTLE *the* reference for deployment comparisons. |

Most valuable newly: items 1–3, which are cheap, already half-built, and settle what E4 means.
Highest-ceiling: items 4, 5 and 8.

---

## 6. Venue and framing advice

**Where:**
- **As it stands, plus items 1–3:** a strong workshop paper (CoRL/RSS workshops on robot learning
  systems, efficient or embodied foundation models, or deployment) or a **NeurIPS Datasets &
  Benchmarks** submission. The released dataset, CLI, Monte Carlo methodology and closed-loop
  wrapper fit the D&B "benchmark methodology + artifact" profile well.
- **With items 4–5 (multi-model, async closed loop):** **CoRL** or **RA-L (+ICRA/IROS)**. The
  robotics community will value the latency–success curve more than the MAPE.
- **With item 8:** a CoRL main-track paper with a claim nobody else has. MLSys is a weaker fit,
  because the predictor is not novel enough as a systems model on its own.

**Framing.** Lead with the *decision*, not the predictor.

- **Title:** *"What Does a Successful Robot Task Cost? Latency-Aware Cost per Success for VLAs on
  Hardware You Don't Own"*, or more punchily, *"The Cheapest Robot Brain Depends on Who Pays:
  Cost per Successful Task for Vision-Language-Action Models."*
- **Lead finding:** the cheapest device for a VLA flips with billing assumptions (Orin Nano when
  compute is billed, a tethered desktop GPU when robot time is billed). Close behind it:
  *success itself depends on latency once the robot can't wait.* Put the latency–success curve
  (Figure `fig:sim`, extended to more models and async) in the first two pages.
- **Position the predictor as an enabling result:** "eager VLA latency is host-bound, so calibrate
  additively on your own stack; this gets within ~28% and is bounded by a 1.95× stack noise floor."
  That framing is honest and robust to reviewers who would otherwise attack the 28% MAPE.
- **Own the negatives explicitly. They are assets.**
  (i) "Fastest ≠ cheapest" was withdrawn as a JAX artifact. Instead, say that *software stacks
  reorder latency but not cost-per-success rankings*.
  (ii) The prospective T4 check missed by 1.56× because of a precision assumption. Present it as
  the lesson that *deployment precision is part of the configuration*.
  (iii) E4's sync loss is (if item 1 confirms it) a budget effect. Present it as *a
  zero-simulation correction any benchmark can apply*.
- **Don't** call the 630 ms result "success falls by 13 points" in the abstract when p=0.19 and the
  mechanism is mostly budget. Say "falls by up to 13 points (n.s.) at ~630 ms and to zero at
  1.5 s, primarily by exhausting the episode time limit". Better still, replace it with the
  separated sync/async/budget result once items 1–2 land.
- Make **LIBERO's static nature** an explicit scope statement, and add a dynamic task before any
  top-venue submission.
