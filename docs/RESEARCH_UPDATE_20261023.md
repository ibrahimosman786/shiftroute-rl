# Verified experiments and current blockers

The dedicated public repository is https://github.com/ibrahimosman786/shiftroute-rl.

- Local test suite: 68 tests pass including the new synthetic three-turn agent training *utility* tests. This does not establish the correctness of later Hugging Face training.
- Full original SFT/PPO neural policy training in dedicated repository: https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066573854 (successful; downloadable cloud model checkpoints and synthetic results).
- v0.5 paired adaptive-trajectory counterexample: https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066518102 (successful; reproducible).
- Pretrained SmolLM2-135M v0.4 SFT/PPO-style action experiment: https://github.com/ibrahimosman786/shiftroute-rl/actions/runs/40066426969 (successful *execution* but negative performance: all-ALLOW policy after every phase).
- Extended multi-epoch pretrained model experiment: `.github/workflows/pretrained-v05.yml` is a separate follow-up and has no measured result until completed successfully.
- v0.6 in this bundle is a *new experimental scaffold* for a simulated three-turn language-model tool policy. It uses no external tools or real data; its LoRA model training is not yet verified. The model code has passed only import-independent unit tests and Python compilation.

Research impact and novelty are not proven. Do not claim any multi-turn LLM generalization, guaranteed safety of imperfect reviewers, real cost savings, or accepted upstream contributions. The next high-value experimental work is independent AgentDojo-style task labels and a prospective evaluation across changing policy distributions.
