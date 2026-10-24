# ShiftRoute-RL v0.4: mathematical scope, trusted certificates, and pretrained model trial

## Evidence hierarchy

1. **Verified mathematical claim (conditional, not novel):** On a fixed proposed-action trajectory and coupled reviewer decisions, verification of all proposals and a proof-first gate have identical unsafe-execution indicators **if** each skipped review carries a sound independently issued authorization certificate. The reduction in review calls equals the number certified. The reviewer may be imperfect; this does not heal its misses.
2. **Imperfect certificate bound:** An additional unsafe execution requires an unauthorized proposal, false certification, *and* a counterfactual reviewer rejection. Therefore additional unsafe executions are at most the number of falsely certified unsafe proposals. If conditional false-certificate probability is bounded by epsilon per unauthorized proposal, expected additional unsafe executions <= epsilon * expected unauthorized proposals; without a trustworthy epsilon no nontrivial bound is justified.
3. **On-policy historical audit bound:** Random audit coins independent of the current proposal and known inclusion probabilities permit a conservative time-uniform upper confidence bound on *past realized unauthorized proposals*, conditional on a true known minimum reviewer detection sensitivity. This is NOT a per-action safety certificate or a future-risk guarantee; the bound is often vacuous when audits are rare.
4. **Engineering model:** `capability_v04.py` binds HMAC token to actual request, session, user, resource, arguments, policy version, expiry and single-use nonce. It uses a trusted Policy object; untrusted agent text never signs capabilities. This is a pedagogical service within a single trust boundary, **not a deployed production authorization solution**.
5. **Pretrained LM trial (GitHub Actions):** `llm_trial_v04.py` uses HuggingFaceTB/SmolLM2-135M-Instruct to produce two possible action decisions, LoRA SFT, and PPO under weak vs verified feedback, with distinct held-out prompt formats. It is a pretrained causal language model but *not a multi-turn interactive tool agent*. Cloud results must not be reported as complete until the actual workflow succeeds.

## Why novelty has not been established

- [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677) already establishes insufficiency of unverified feedback and partial-audit selective control (2026).
- [Conformal Selective Acting](https://arxiv.org/abs/2605.20270) already claims time-uniform selective risk control under its own assumptions (2026).
- [GRAFT](https://alignment.anthropic.com/2026/graft/) already uses gradient routing to avoid certain learned reward hacks (2026).
- [AgentDojo](https://agentdojo.spylab.ai/) is a public external tool-using agent benchmark; **v0.4 does not claim a full AgentDojo evaluation or an upstream accepted contribution**.

## Protocol and limitations

- Synthetic sandbox actors in v0.2/v0.3 remain the only interactive six-step agents trained so far; v0.4 neural LM trial is a binary decision policy.
- The proof-first result holds for unchanged action sequences and coupled reviewer counterfactuals. In deployment, changing the gate may change agent behavior and future proposal distribution. Coupling then fails at the trajectory level. No broad RL policy-safety theorem follows.
- HMAC secret security, issuer compromise, offline revocation, concurrency safe nonce storage and scalable key rotation are out of scope. Nothing here should authorize access to real user accounts.
- There are no known precise cost models for the trusted policy issuer, human reviewers, legal damages or production operations. Cost claims are relative, simulator-defined units only.
- LLM prompts are synthetic and the pretrained model is small. Holdout prompt format is not evidence of robust generalization to the wild.
- For future publication, pre-register a concrete improvement target, train policy weights while interacting with imperfect reviewers, run real AgentDojo benchmark tasks with paired random seeds, reproduce paper baselines and report statistically calibrated CIs and all negative results.

## Reproduce

```bash
cd research/shiftroute-rl
python -m unittest discover -s tests -v
# Public pretrained weights and internet are needed for this experiment:
python -m pip install 'transformers==4.46.3' 'peft==0.14.0' 'accelerate==1.1.1' torch
python -m shiftroute_rl.llm_trial_v04 --seed 7 --sft-steps 24 --rl-steps 24 --batch 6 --eval 128
```

**Status:** A credible negative result and a restricted solution, not a general solution or novel state-of-the-art safety algorithm. Independent peer review requires researchers outside this project to evaluate the work; a GitHub pull request to one's own repository is not independent peer review.


## Actual pretrained experiment outcome (verified cloud run, October 2026)

The [GitHub Actions run](https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066426969) completed successfully on CPU and published adapter checkpoints ([artifact](https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066426969/artifacts/12550075024)). This replaces the earlier pre-execution status statement above.

- Base: `HuggingFaceTB/SmolLM2-135M-Instruct`, small causal LM; LoRA SFT 24 updates, PPO-style 24 updates each under weak and verified synthetic rewards, single seed 7.
- Evaluation: 128 held-out shifted synthetic authorization prompts. **All three checkpoints predicted ALLOW for all 128 prompts.**
- After SFT: accuracy 0.5390625; unauthorized approvals as fraction of all proposals 0.4609375; false refusals 0.
- After weak PPO-style RL: identical values.
- After independently verified PPO-style RL: identical values.
- **This is a degenerate negative result.** It is not evidence of improved safety, model generalization or useful policy learning. A trivial always-ALLOW baseline matches all three evaluated variants.
- With just one PPO gradient update per sampled minibatch, the clipped ratio starts at one and the method is best described as *PPO-style single-update policy gradient*, not an extensively optimized PPO implementation.

A separate `llm_trial_v05.py` and [extended-training workflow](../.github/workflows/pretrained-v05.yml) now use more SFT updates, multiple PPO epochs and threshold calibration on independent train-format examples. Results remain **unverified** until that workflow completes. Neither experiment is a multi-turn tool-using pretrained language agent.
