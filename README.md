# ShiftRoute-RL
**Verification-efficient reinforcement learning under adaptive policy and monitor shift.**

This repository studies whether an adapting RL agent can reduce expensive safety reviews without increasing unauthorized tool execution. It is a **research prototype**, not a general safety solution.

## What has actually been established

1. An SFT-initialized, PPO-trained **small PyTorch policy** can learn unsafe shortcuts in constructed multi-step environments when supervised with an imperfect monitor.
2. Fixed thresholds and randomized auditing did not, in the studied shifted simulations, preserve the safety of reviewing every proposed action.
3. On an identical stream of proposed actions, proof-first gating with **sound, independently issued authorization certificates** preserves unsafe-execution outcomes relative to reviewing every action, while reducing reviewer calls. This conditional observation is **not** a general guarantee for adaptive trajectories, imperfect certificates or compromised issuers.
4. Costs reported in this repository are **simulated units**. No validated monetary savings or real-world safety improvement is claimed.

**Important:** The pretrained 135M language-model SFT/PPO experiment is proposed, not yet a verified result. The pretrained policy only makes binary ALLOW/DENY decisions; it is not an interactive pretrained multi-turn agent.

## Reproduce

```bash
python -m pip install -e '.[research]'
python -m unittest discover -s tests -v
python -m shiftroute_rl.deep_experiment --out results/reproduced --seeds 7 13 23 --updates 200 --batch 192 --episodes 1000
python -m shiftroute_rl.risk_budget_v03 --n 3600 --out results/v03/reproduced.json
```

## Project

- `src/shiftroute_rl/`: environments, neural SFT/PPO training, imperfect monitors, routing and capability studies
- `tests/`: regressions and methodology tests
- `docs/`: experiment methodology, conditional results and limitations
- `results/`: labeled **synthetic** evaluation records and training configurations
- `.github/workflows/`: reproducible GitHub Actions CPU experiments

## Prior art and outstanding work

Research on auditing and verifier-error feedback already exists. Compare [GRAFT](https://alignment.anthropic.com/2026/graft/), [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677), [Conformal Selective Acting](https://arxiv.org/abs/2605.20270) and [AgentDojo](https://github.com/ethz-spylab/agentdojo). We have **not** established scientific novelty, pretrained multi-turn agent generalization, or an accepted upstream open-source contribution.

See `docs/research-v03.md`, `docs/research-v04.md`, and `docs/RESEARCH_STATUS.md` for the evidence hierarchy.
