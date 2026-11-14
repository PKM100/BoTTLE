1. SUMMARY

BoTTLE combines an analytical per-stage VLA latency model, one-point cross-device calibration, and a task-level accounting model to estimate latency, energy, and dollars per successful robot episode on hardware the user does not own. The paper's central empirical claims are that an additive, host-scaled residual transfers published latency measurements better than multiplicative calibration, and that incorporating success rate and robot/operator time changes the preferred model and device on LIBERO.

2. VERDICT

VERDICT: Major Revision — 5/10

3. STRENGTHS

- The paper asks an important deployment question that conventional success-only or latency-only VLA evaluation misses: expected cost per successful task.
- The decomposition of task cost into action horizon, synchronous/asynchronous execution, retries, energy, and robot/operator time is useful and exposed through an apparently practical CLI (§3).
- The authors are commendably explicit about several limitations, including software-stack variation, device-independent simulator success, approximate hardware/model specifications, and omitted network latency (§5).
- E2 assembles a useful public latency dataset spanning six model families and heterogeneous devices, and compares against transparent baselines rather than presenting only favorable examples.
- The negative result that naive multiplicative calibration can worsen Jetson transfer is useful. The reported improvement from 48.8% to 28.1% MAPE is substantial enough to merit further investigation.
- The implementation and experiment scripts are readable and make the arithmetic behind the tables inspectable.

4. WEAKNESSES / THREATS TO VALIDITY

- **The principal E2 sample size is overstated by treating strongly dependent ordered transfers as 32 observations.** The 32 pairs come from only eight source/setting groups; 12 are from the single GR00T N1.6 group and nine from the single fused-OFT group. Multiple rows reuse the same measurements as references and targets. Consequently, the MAPE and “56% within 25%” in Table 1 do not represent 32 independent tests, and no group-level/model-level confidence interval is reported. The result may be driven disproportionately by two stacks.
- **The central “host/framework overhead” interpretation is not established by the experiments.** In §3.2, every residual above the roofline is assigned to host/framework overhead and scaled using coarse CPU-class constants. E2 contains end-to-end published timings, not measurements of Python dispatch, synchronization, preprocessing, or CPU utilization; E1 changes accelerator/precision on one host and likewise cannot identify CPU scaling. Better prediction by an additive residual is evidence for an empirical additive model, but not for the causal claim that the residual is host-CPU time.
- **Method development and evaluation are not cleanly separated.** The paper says the published measurements motivated the overhead-aware model and then evaluates that model on the same small corpus (§1, §3.2, §4.2). Although the CPU priors are stated not to be fitted, choices such as clipping, the additive decomposition, and CPU classes could still have been selected after observing these data. There is no held-out model/source analysis or ablation showing robustness to these design choices.
- **The headline E3 rankings are much finer-grained than the validated predictive accuracy supports.** The paper recommends reading latency uncertainty as roughly ±30% (§5), yet many claimed winners in Table 2 differ by only a few percent: under “+robot,” OFT versus GR00T is 24.6 versus 26.8 on H100, 20.0 versus 18.7 on 5090, and 20.8 versus 18.9 on 4090. Device winners are similarly close. No uncertainty is propagated through latency, power, success rate, assumed successful-episode length, hardware price, or execution horizon. Thus “cheapest model/device” and the abstract's device-flip claim are not supported at the reported resolution.
- **E3 combines success rates from different publications as though they were controlled measurements.** §4.3 does not document whether checkpoints, observations/cameras, demonstrations, task versions, action representations, or evaluation protocols are comparable. The supporting code even labels SmolVLA and GR00T execution horizons as assumed. Since success and horizon are precisely what drive cost rankings, these are not peripheral assumptions.
- **The task-cost analysis is synthetic rather than a validation of cost per success.** There are no measured end-to-end episodes, no observed completion-length distributions, and no hardware measurements linking stalls to success. Failed episodes are assumed to reach the limit and successful episodes to end at exactly 60%, with sensitivity only to the latter scalar and only for the model ranking on Thor (§4.3). This cannot validate the absolute dollar figures or broad device conclusions.
- **Energy results are too weak for the abstract/framework positioning.** The default energy model has 107% MAPE and still has 39% leave-one-out MAPE (§4.2). The leave-one-out utilization estimate also pools different models on one device rather than demonstrating cross-device energy transfer. Yet energy is presented as a core output of BoTTLE without corresponding uncertainty in Table 2.
- **The “noise floor” argument is not rigorous.** A maximum cross-source spread of 1.95× does not imply that 28% MAPE is close to an irreducible bound (§4.2); it mixes protocol/software differences and uses maxima rather than a variance decomposition or matched repeated measurements.
- **There are reproducibility/presentation inconsistencies.** The checked-in `experiments/results/published.json` reports spec-only MAPE of 53.9% and 9.7% within 25%, while Table 1 reports 51.8% and 6%. E3 states nine devices, but Tables 2 and 3 display eight (A10G is omitted). The abstract says the cost-optimal device flips “from ... Orin Nano to an H100,” whereas Table 2 shows H100 winning only three of six models with operator cost and Orin Nano cannot run two models.
- **Memory feasibility is based on a very coarse formula.** The fixed 10% weight overhead plus 0.5 GB slack (§3.1) is not validated against actual runtime peaks, yet it determines which configurations are excluded from the cost comparison.

5. (Round 1: write "N/A — first round".)

N/A — first round

6. REQUIRED FIXES for acceptance

1. **NEW-EXPERIMENT:** Re-evaluate E2 using source/setting groups—not ordered pairs—as the statistical unit. Report paired group- or model-clustered bootstrap confidence intervals, per-group results, and leave-one-model/source-group-out sensitivity. Reserve genuinely held-out sources/models if possible, or explicitly frame the current result as exploratory.
2. **NEW-EXPERIMENT:** Validate or substantially weaken the host-overhead claim. A convincing test would hold accelerator, model, precision, and software stack fixed while varying host CPU performance and measuring stage/dispatch/preprocessing time. At minimum, compare additive residual variants (constant, CPU-scaled, launch-scaled) and report sensitivity across plausible host priors.
3. **NEW-EXPERIMENT:** Propagate uncertainty through E3 and report robust conclusions rather than point-estimate winners. Include latency-error distributions from E2, energy error, protocol-compatible success uncertainty, successful/failed episode-length distributions, and the assumed SmolVLA/GR00T horizons. If rankings are not stable, present sets of statistically indistinguishable configurations or Pareto regions. Direct measurements of the decisive near-tied configurations would materially strengthen the paper.
4. **writing-only:** Fully source and tabulate every LIBERO success rate and execution horizon, including checkpoint, observation setup, number of demonstrations, benchmark version, evaluation trials, and whether values are directly comparable. Clearly label cross-paper comparisons as illustrative when protocols differ.
5. **writing-only:** Reconcile all generated results with the manuscript (including spec-only E2 metrics), explain the missing A10G column, and narrow the abstract/conclusion claims about device flips, energy prediction, and “close to” a noise floor.
6. **writing-only:** Distinguish the empirically supported claim (“an additive residual transfers better on this corpus”) from the currently unproven causal claim (“the residual is host-CPU/framework overhead”). Document how all priors and modeling choices were selected.
7. **NEW-EXPERIMENT:** Validate memory-fit predictions and absolute task-cost calculations on at least a small set of actual deployments or measured episode traces. Otherwise, present memory feasibility and dollar totals as scenario estimates rather than validated predictions.

7. QUESTIONS FOR THE AUTHORS

- Were the overhead decomposition, clipping rule, and host-speed priors specified before examining the 31-point corpus? If not, what data can serve as a genuine test set?
- How do E2 conclusions change when each source/setting group receives equal weight, and when GR00T N1.6 or fused OFT is removed?
- What direct measurement shows that the roofline residual scales with host CPU rather than being a device-specific kernel, framework, compilation, or preprocessing effect?
- Are all LIBERO success rates based on the same task definitions, observation modalities, demonstrations, and number of evaluation trials?
- How often do the model/device winners in Table 2 remain winners under the paper's own ±30% latency uncertainty?
- Why is A10G omitted from Tables 2 and 3 despite the nine-device claim?
- What generated artifact is authoritative for Table 1, given the mismatch with `experiments/results/published.json`?
