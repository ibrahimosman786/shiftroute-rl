# Research status and evidence boundaries

**Question:** Can an adapting RL agent reduce expensive verification while not increasing unauthorized tool executions under policy/monitor shift?

## Verified
- PyTorch SFT/PPO policies were trained in a constructed six-step environment (three seeds; older full cloud run was successful in the former research branch).
- Cheap monitors and adaptive sentinel auditing did **not** preserve verify-all safety in deliberately shifted simulations.
- A *conditional fixed-proposal-stream* proof demonstrates exact equivalence of verify-all and proof-first when every bypassed action carries a sound independently issued authorization proof.
- Synthetic v0.3 comparisons and proof-corruption stress tests are reproducible and are **not** financial or production results.
- A separate v0.5 counterexample demonstrates that even sound proof-first bypass can increase episode-level harm when changed gate outcomes alter future proposals.

## Not established
- A novel RL learning algorithm surpassing GRAFT or verifier-error selective control
- Pretrained multi-turn tool-use agent training or independently replicated AgentDojo results
- General safety under unsound certificates, compromised trusted issuers, reviewers with false positives, and endogenous policy trajectories
- Accepted external upstream contribution or peer-reviewed paper

## Methodological warnings
The v0.4 SmolLM2-135M language experiment, when executed, is a **binary access-decision** task and does not constitute multi-turn LLM-agent reinforcement learning. Its current PPO implementation is a one-update clipped-policy-gradient demonstration rather than a fully tuned multi-epoch PPO baseline. Report numbers only after checking independent cloud logs and saved adapters.

**Never claim these simulated costs are USD or that a constructed counterexample proves an observed production exploit.**
