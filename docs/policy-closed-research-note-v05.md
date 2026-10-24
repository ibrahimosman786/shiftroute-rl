# ShiftRoute-RL Research Note v0.5 — Policy-Closed Verification

**Date:** October 2026. **Status:** working research note, not a peer-reviewed publication.
**Core question:** Can verification of RL agents become cheaper without adding undetected unauthorized tool executions when both policy and monitor adapt?

## Executive summary

The unrestricted claim is **false without extra information or control**. A learned monitor may fail to distinguish authorized from unauthorized actions, and feedback-based routing can change an agent's future action distribution. The current experiments establish two *conditional* engineering results and one critical counterexample:

1. **Fixed-stream result:** On an identical coupled stream of proposed actions, bypassing expensive review only for *sound independently issued authorization certificates* cannot increase unauthorized executions relative to reviewing all proposals.
2. **Trajectory counterexample:** If review-all can falsely reject a legitimate action, sound proof-first can advance an agent to a different state, expose a later unauthorized attempt, and generate **more episode-level harm**, even though the proof itself is never wrong.
3. **Stateful enforcement:** Independently checking a non-bypassable authorization invariant at every state transition blocks this particular second-stage misuse in the simulator. This requires correct tool/action semantics, cannot be supplied merely by a confident LLM monitor, and can harm utility if too restrictive.

No claim of a novel RL algorithm or production safety guarantee follows.

## Formal problem

Let history H_t contain policy proposals, tool outputs, gate decisions and executed state changes. An adaptive policy π_t(a|H_t) proposes action a_t. A monitor produces score m_t from potentially attacker-influenced features. The expensive reviewer has false approvals and false denials. A separate authorization service may supply a sound capability C_t bound to exact user, resource, arguments, policy version and expiry.

The relevant outcome is not only a per-action false acceptance rate on a pre-recorded dataset. It is **episode-level unauthorized executions under the changed trajectory induced by the gate**. The objective should be a constrained Pareto frontier of legitimate task completion, executed unauthorized actions, and independently measured reviewer cost.

### Basic identification limit

Construct two worlds whose observable model/monitor/reviewer histories are indistinguishable up to an approval decision, but the same proposed action is authorized in one world and unauthorized in the other. Without further trusted data, a gate that approves with probability p must suffer unsafe approval probability p in the second world. If the user demands a worst-case cap α for both worlds, the gate cannot approve with probability greater than α on that observation. This is an elementary indistinguishability argument, **not a new theorem**, and does not rule out useful guarantees under extra assumptions.

### Fixed-stream lemma

For identical proposed actions and coupled reviewer outputs, a sound independent proof cannot increase unsafe execution on that stream: any unauthorized proposal lacks proof and is sent to the same reviewer as verify-all. The reviewer may itself be wrong. This property does not guarantee equal *future histories*.

### When can the statement extend to complete adaptive trajectories?

If the reviewer never falsely rejects authorized proposals, certificates are sound, reviewer decisions are coupled for common proposals, and the agent/environment transition identically on matching histories, then proof-first and verify-all histories remain equal by induction. Each step gives the same execution decision, hence the next state and proposal agree. Introducing false denials breaks this induction.

## Exact trajectory counterexample (v0.5)

At time zero, a legitimate unlock is proposed; review-all falsely blocks it with probability f. A sound proof grants that action under proof-first. The now-unlocked agent proposes an unauthorized second action, which the reviewer misses with probability m.

| Method | Analytic unsafe episode probability | Expected expensive reviews per episode |
|---|---:|---:|
| Verify every proposal | (1−f)m | 2−f |
| Proof-first, sound certificate | m | 1 |
| Proof-first + *sound stateful action shield* | 0 in this specific simulator | 1 |

The **added unsafe probability is fm**. For f=.15 and m=.08 the analytic difference is 1.2 percentage points. In 200,000 paired episodes, the measured difference was 2,363/200,000 = **1.1815 percentage points**, two-sided Wilson 95% interval **[1.1351%, 1.2298%]**. Unsafe rates were 6.796% under verify-all versus 7.9775% under sound proof-first. The shield blocked all second-stage unsafe actions by assumption; zero observed harm does not establish real-world safety.

These numbers come from an intentionally constructed simulator with **independent random reviewer mistakes**, not from a benchmark sampled from production or an observed deployed LLM.

## Proposed engineering intervention: policy-closed certification

Use an external trusted policy service to issue narrow authorization capabilities for actions whose *actual arguments and proposed state transitions* have machine-checkable security semantics. Recheck every state transition, including those following previously permitted actions. For uncertifiable actions, fail closed or escalate to an independent reviewer. Randomly sample certified actions for **auditor diagnostics**, not as a substitute for formal proof. Keep audit propensities and policy epochs so a later evaluator can assess monitor shift without peeking at true labels during routing.

The security invariant and proof issuer must be outside the agent's control; compromised issuers, incomplete coverage, unsafe semantics in supposedly 'allowed' tools, and distribution changes remain serious failure modes. The approach may cut LLM-judge or human reviewer calls for easy-to-prove actions, but can increase latencies or false refusals. Exact business benefits remain unmeasured.

## Direct comparison to prior art

- *Verifier Errors in RLVR* (Moya et al., September 2026) already characterizes accepted reward errors and uses partial audits for selective control: https://arxiv.org/abs/2609.35677
- *Conformal Selective Acting* (Khosravi & Huo, May 2026) already studies anytime risk bounds for RLVR-trained specialist LLMs: https://arxiv.org/abs/2605.20270
- *GRAFT* (Hu, Ward et al., October 2026) routes training gradients to avoid retaining some reward hacks: https://alignment.anthropic.com/2026/graft/
- AgentDojo provides external multi-turn prompt-injection tasks: https://github.com/ethz-spylab/agentdojo

**Novelty is not established.** The potential added value is experimentally characterizing the *interaction between external semantic authorization, policy-induced trajectory shift, and reviewer cost* under realistic multi-turn workloads.

## Research release criteria still unmet

A strong paper would additionally need (a) realistic action schemas and held-out AgentDojo-like tasks with justified authorization labels, (b) genuinely adaptive pretrained *multi-turn* LLM fine-tuning, (c) comparison to CSA, GRAFT-like routing, verify-all and abstention, (d) paired seeded counterfactuals and calibrated confidence intervals, (e) compute, latency and reviewer-error sensitivity costs, and (f) independent replication. The 135M-language-model binary-choice trial in this repo is only an initial infrastructure experiment, **not** item (b).

## Reproduce this note's new result

```bash
pip install -e '.[research]'
python -m unittest discover -s tests -v
python -m shiftroute_rl.trajectory_v05 --episodes 200000 --seed 20261023 --out results/v05/endogenous_counterexample.json
```

The cloud workflow `.github/workflows/trajectory-v05.yml` reruns this and archives the JSON.
