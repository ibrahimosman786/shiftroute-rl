# ShiftRoute-RL: Policy-Induced Distribution Shift and the Limits of Verification-Efficient Tool Agents

**Working research manuscript — 23 October 2026. NOT peer reviewed.**

## Abstract

Reinforcement-learning agents may learn to exploit imperfect reward and safety monitors. Verification can mitigate this problem but costs inference, money and delay. We investigate what can be claimed when some tool actions are checked by a cheap learned risk scorer, some by an expensive reviewer and some by an independent capability issuer. Small PyTorch SFT/PPO policies in constructed sequential simulations show substantial monitor exploitation. An initial LoRA/PPO-style fine-tuning of a pretrained 135M language model ran successfully but collapsed to always approving actions. A fixed-stream proof establishes that sound authorization certificates allow fewer reviewer calls without increasing unauthorized executions **on identical proposed action streams**. A new two-stage counterexample shows that sound certificates alone cannot extend this equivalence to trajectories of adaptive policies when the reviewer falsely rejects legitimate actions. We derive a simple first-divergence coupling bound and construct a non-bypassable stateful authorization shield that eliminates unsafe executions in our synthetic example by assumption. This is a research artifact and qualified negative result; scientific novelty and out-of-distribution real-world performance remain unestablished.

## 1. Research question

Can `E[reviewer cost]` decrease while maintaining episode-level unauthorized execution risk and legitimate task completion, when the underlying policy is itself adapting to a learned, imperfect monitor? Unlike per-action risk measured on a fixed replay buffer, the treatment affects future agent states and proposals.

## 2. Experimental setup

The v0.2 policy experiment uses three seeds, a supervised initialization and policy-gradient PPO updates under weak, independently verified and capability-informed simulated rewards. Evaluation includes shifted task features. The v0.3 benchmark uses five reviewer-stream seeds for each stable/policy-shift/joint-shift condition, with a perfect or 94%-sensitive expensive reviewer. The reviewer-cost model is synthetic and does not represent vendor billing.

The v0.5 counterexample uses 200,000 paired episodes with independent reviewer false-denial probability f=0.15 on an authorized unlock and reviewer false-approval probability m=0.08 on an unauthorized later action.

## 3. Observations

### Fixed-proposal benchmark

Under the v0.3 joint shift and a perfect expensive reviewer, averaging five seeds, unauthorized execution fractions were 0% for verify-all, 46.96% threshold-only, 41.25% sentinel, 6.53% adaptive, and 0% proof-first. The corresponding simulated cost per proposal was 1.025 verify-all, 0.756 adaptive and 0.730 proof-first.

At 94% reviewer sensitivity, verify-all and sound proof-first both had 3.061% unauthorized executions; the cost difference persisted. **Those values depend on the constructed sound-proof oracle and fixed proposals.**

### Failure of episodic extension

A reviewer sometimes prevents an authorized unlock, stopping further interaction. Sound proof-first allows the unlock and the resulting policy may attempt an unauthorized second call. Exact probabilities: `Pr(harm|verify-all)=(1-f)m`, `Pr(harm|sound-proof)=m`, so `risk difference=fm`. With f=0.15,m=0.08, the difference is 1.2 percentage points. The paired simulation observed 6.796% vs 7.9775%, a difference of 1.1815 percentage points (Wilson 95% interval on added risk: [1.135%,1.230%]).

An *independent, correctly implemented, mandatory authorization check on each state transition* blocked this particular unauthorized second action. That is an assumption about external semantics and enforcement, not an emergent learned safety property.

### Pretrained language-model baseline

The first cloud experiment used SmolLM2-135M-Instruct with LoRA SFT and PPO-style updates on binary ALLOW/DENY prompts. Despite successfully saving three real adapters, all variants predicted ALLOW for every shifted holdout example (128 cases): accuracy 53.90625%, unauthorized-approval fraction 46.09375%. This is an empirical **failure**, not an improvement. Longer SFT, multi-epoch PPO and a separate sequential three-stage model trial are being evaluated; their results must be added only after their GitHub Actions jobs succeed.

## 4. Theory: an elementary coupled bound

On a fixed stream of proposed actions with coupled reviewer decisions, sound independent certificates never bypass reviews for unauthorized actions; hence proof-first does not add unsafe executions relative to verify-all on that fixed stream. Once the gate changes the future state, this claim ceases to apply.

For complete histories coupled until their first divergence, if a valid uniform conditional upper bound on first divergence at round t is δ_t, then

```
P(any history divergence by T) ≤ 1 − product_{t=1}^T (1 − δ_t).
```

Any binary episode-level harm risk differs by at most this bound. This is standard probabilistic coupling, not a novel theorem. Its applicability requires independently defensible δ_t bounds and covers side effects that affect future proposals.

## 5. Related work / novelty caveat

ToolFence already uses typed authorization and saves expensive judge calls (https://arxiv.org/abs/2609.37196). PACE already mediates provenance and tool effects (https://arxiv.org/abs/2610.01349). Local Allow, Global Deny studies failures of locally valid actions under global conditions (https://arxiv.org/abs/2610.27620). Verifier Errors in RLVR studies partial audits (https://arxiv.org/abs/2609.35677), CSA studies anytime selective acting (https://arxiv.org/abs/2605.20270) and GRAFT studies gradient routing to limit learned reward hacks (https://alignment.anthropic.com/2026/graft/). The current contribution is a reproducible research investigation, **not proven to be distinct from those prior contributions**.

## 6. Necessary independent validation

To make a publishable claim, benchmark on real independently labeled AgentDojo-like multi-turn tasks, fine-tune at least two pretrained agent policies across multiple seeds, reproduce ToolFence/PACE/CSA/GRAFT-style baselines fairly, compare safety/utility/cost on heldout task clusters, ablate reviewer false denials and certificate unsoundness, and obtain external replication. If this fails to improve the Pareto frontier, publish that negative result.

## Reproduction evidence

Main code: https://github.com/ibrahimosman786/shiftroute-rl

- Full three-seed cloud PPO reproduction: https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066573854
- v0.5 paired 200k-episode counterexample: https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066518102
- Initial 135M cloud fine-tune and adapters: https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066426969
- Executed numerical analysis notebook: `notebooks/reproducible-analysis-v05.ipynb`

No real-currency savings, general adversarial security guarantee, independently verified breakthrough, or peer-reviewed publication is claimed.
