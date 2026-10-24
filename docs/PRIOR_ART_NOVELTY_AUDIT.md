# ShiftRoute-RL prior-art and novelty audit (October 2026)

The research team must not claim that deterministic capability enforcement, risk-budgeted auditing, or the dangers of noncomposable authorization were first discovered here.

Closest published or preprint work:
- ToolFence (https://arxiv.org/abs/2609.37196): typed tool-authorization blueprints and selective judge consultation. Reports fewer expensive reviews.
- PACE (https://arxiv.org/abs/2610.01349): provenance-aware capability and effect enforcement.
- Local Allow, Global Deny (https://arxiv.org/abs/2610.27620): per-action authorization may not satisfy global temporal or relational policies.
- Verifier Errors in RLVR (https://arxiv.org/abs/2609.35677): verifier failures and selective audit corrections.
- Conformal Selective Acting (https://arxiv.org/abs/2605.20270): anytime selective risk control.
- GRAFT (https://alignment.anthropic.com/2026/graft/): gradient routing in training to reduce monitor-targeted reward hacking.
- AgentDojo (https://github.com/ethz-spylab/agentdojo): publicly available adversarial tool-use tasks.

Unproven hypothesis: trajectory-aware causal allocation of independent reviews may reduce reviewer cost while matching **episode-level**, not just per-proposal, authorization risk and useful task completion for *changing pretrained RL policies*.

A defensible test requires held-out real external tool-action benchmarks, at least two pretrained policies and multiple seeds, honest comparator implementations including ToolFence/PACE/CSA/GRAFT families, independent authorization labeling, reviewer-cost accounting, and confidence intervals by task.

Status: This repository has not demonstrated a novel outperforming algorithm or external replication. The synthetic v0.5 counterexample is explanatory, not a new general theorem.