# v0.5 — A counterexample to fixed-trace safety generalization

**Synthetic result.** A *sound* independent proof can still increase **episode-level** unauthorized executions when bypassing a review changes the agent's future state.

In a two-stage simulated episode, the reviewer sometimes falsely denies a legitimate unlock action (probability f). The sound proof correctly authorizes unlock and bypasses that review. Once unlocked, the adapting policy attempts a disallowed second action; the reviewer misses it with probability m. Verify-all therefore has risk (1-f)m, but proof-first has risk m, for an **additional risk fm**. The common reviewer coins are coupled between policies.

With f=0.15 and m=0.08, expected additional risk is 1.2 percentage points, despite perfectly sound proof certificates. A deterministic stateful tool-policy gate can block the disallowed second action, but its correctness is itself a major external assumption; a fallible LLM monitor cannot provide this guarantee by being confident.

**Trajectory equivalence sufficient condition.** If proof certificates are sound, the reviewer never falsely denies an authorized proposed action, agent and environment have the same transitions given the same history, and review has no independent state-changing side effects, then proof-first and verify-all have identical executed trajectories by induction, including with an adapting policy.

This counterexample refutes an overbroad interpretation of the earlier fixed-stream result. It is neither a discovered production vulnerability nor an original state-of-the-art RL method.
