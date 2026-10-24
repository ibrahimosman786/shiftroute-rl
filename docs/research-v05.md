# ShiftRoute-RL v0.5: adaptive trajectories invalidate fixed-action guarantees

**Status:** Synthetic, independently reproducible on GitHub Actions. This work corrects a prior overbroad conclusion; it is not a universal RL safety theorem.

Our main objective is to reduce expensive agent tool-action verification without increasing unauthorized action execution. In a prior experiment, sound independent certificates allowed a subset of individual proposals to bypass fallible expensive review while matching verify-all on the *same fixed proposals*. That statement does not account for the next agent action changing after an earlier decision is allowed.

In the v0.5 two-action trajectory, a sound certificate bypasses review of an authorized unlock request. A fallible reviewer would sometimes falsely deny that request. The unlocked agent then proposes an unauthorized follow-on action that the reviewer sometimes falsely accepts.

Let f be the false denial probability for authorized unlock and m the false acceptance probability for an unsafe follow-on proposal. Then P(unsafe|verify-all)=(1-f)m and P(unsafe|sound-proof-first)=m. The difference is fm. Under f=.15 and m=.08, the exact difference is .012. Across N=200,000 synthetic episodes with independent coins and seed 20261023, it was .011815 (Wilson 95% interval [.0113507,.0122980]). Verify-all unsafe rate: .067960; proof-first unsafe rate: .079775. The latter used only 1.00 expensive reviews per episode compared with 1.85008 for verify-all. Costs are *arbitrary modeled units*. 

An external stateful authorization shield blocks unauthorized follow-on actions and produced zero unsafe execution on these particular synthetic events at an extra checking cost of .02 modeled units per episode. That zero is built into the sound-shield assumption; it is **not** a learned security guarantee. The tradeoff between feasible machine-checkable authorization, action coverage, false refusals, and the expense of the shield has not yet been measured on real agents.

There is a simple sufficient condition for full adaptive trajectory equivalence: on common action histories, the reviewer must never falsely reject an authorized proposal, certificates must be sound and have no side effects, and common history leads to identical future proposals and environment transitions. Then gate decisions agree at every step by induction. Reviewer false denials invalidate this argument.

To reproduce: `python -m shiftroute_rl.trajectory_v05 --episodes 200000 --seed 20261023 --out results/v05/endogenous_counterexample.json`. Current paper-grade claims still need realistic multi-turn agents, independent authorization labels, multiple training seeds, direct comparison to CSA/GRAFT and outside review.
