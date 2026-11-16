# Round-2 Review — CONSOLIDATED

Reviewed after: round-1 fixes plus new E4 (closed-loop LIBERO latency emulation, deadline budget) and the prospective T4 check.

## VERDICTS

| Model | Round 1 | Round 2 | Δ |
|---|---|---|---|
| gpt-5.6-sol | Major 5/10 | **Major 6/10** | +1 |
| claude-sonnet-5.5 | Major 5/10 | **Minor 6/10** | ↑ |
| gemini-4-argon | Minor 7/10 | **Minor 7/10** | = |
| grok-4.7 | Major 5/10 | **Major 6/10** | +1 |
| claude-opus-5.5 (advisor) | E3 is "arithmetic on borrowed numbers" | Weakest link is still the success side; E4's sync loss is probably mostly truncation | — |

All referees say the round-1 fixes landed. The prospective T4 miss was praised for being reported honestly.

## Convergent required fixes
1. **E4 action-budget confound.** Holds consume the 280-step limit, so control degradation and timeout can't be told apart. Report action-budget runs alongside the deadline runs as two separate estimands. Raised by all 4 referees and the advisor. **NEW-EXPERIMENT, blocking.**
   - *Status:* `--bottle.budget=actions` is implemented and tested (`b8593ca`). The Colab rerun at 1500/630 ms, N=5, is running.
2. **Overclaiming.** "holds", "unchanged", "fails on slow devices" and "pay twice" aren't supported, since CIs include 0 below 1500 ms. Raised by all 4. **Writing.**
   - *Status:* done (`237dbd5`). Abstract, contributions, E4, Limitations and Conclusion are rescoped to deadline success, with CIs and no claim of significance.
3. **Don't apply the s0/sℓ correction to E3 without qualification.** It double-counts stall time if there's no deadline, and "conclusions intact" was asserted without proof. Raised by all 4 and the advisor. **Writing now; re-analysis after #1.**
   - *Status:* E4 now states the correction applies only under a deadline and has CIs including 1. The unverified "rankings intact" claim is removed. The Monte Carlo fold-in is deferred.
4. **Stock 80% vs. the 90% published value used in E3; LeRobot's 280-step budget vs. E3's 220+10.** Raised by Claude, Gemini and Grok. **Writing.** *Status:* done.
5. **RTX 4090 tasks 7–9 per-task logs are missing.** The 10-task total is quoted next to a 7-task Δ. Raised by Claude, Gemini, GPT and Grok. **Data.** *Status:* the caption and text now say this. The export cell will recover the data.
6. **The max-actions arithmetic was wrong** (130/170 should be 120/161). Raised by Grok and GPT. **Writing.** *Status:* fixed via `max_actions()`.
7. **Statistics.** Wilson intervals treat episodes as independent; the sign-flip test has low power; the "holds" claim needs an equivalence margin. Raised by Claude and GPT. **Writing.** *Status:* labelled, power noted, "holds" claim dropped.
8. **Put the time-per-success numbers in reproducible code, using per-episode logs.** Raised by Gemini and Grok. **Data.** *Status:* the notebook export cell (section 6) was added; it waits on the user pasting its output.

## Single-reviewer points
- Claude: run Thor at n=100 and add 150–250 ms and 400–500 ms points. Resample the MC latency multiplier by source group. Scope the "6 of 9" abstract claim and note that the Orin Nano is not real-time.
- Gemini: the measured fp32 T4 latency (624 ms) exceeds the 500 ms real-time threshold at H=10.
- Grok: report how often robosuite's 1000-step cap binds in action-budget runs.
- GPT: treat the host explanation of the T4 residual as a hypothesis. *Done.*

## Advisor roadmap (top)
1. **Truncation-only check (cheap).** Predict deadline success from stock episode lengths: success iff steps + h·⌈steps/H⌉ ≤ 280. If it matches E4, the loss is arithmetic that BoTTLE can compute without simulation. *Needs the per-episode export.*
2. Finish the action-budget rerun (and Thor at n=100), fold s0/sℓ into the MC, and use the measured fp32 T4 latency.
3. Run more models (GR00T N1.6, π0.5) and emulate async stale observations. Moonshot: a measured bill on an SO-100 arm.

## STATUS
**Not converged. One blocking experiment remains:** the action-budget rerun, which is running. Writing fixes #2–4, #6 and #7 are applied. After the rerun and the per-episode export: update E4, run the truncation-only check, decide on the MC fold-in, then run round 3.
