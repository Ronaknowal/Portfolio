# Authoring notes: GRPO, RLOO, KTO & Advanced Preference Methods

Canonical topic ID: grpo-rloo-kto-advanced-preference-methods

## 2026-09-22 — Distinguish group-baseline estimators from probability-scoring objectives

- Status: open
- Origin: [Typed decision-model curriculum plan](../../curriculum/TYPED-DECISION-MODELS-PLAN.md).
- Destination and ownership rationale: this topic owns how policy-gradient estimators, baselines and clipping/normalization differ. `typed-decision-models-calibrated-neural-decision-systems` owns the full proper-score probability-reporting application; avoid a second competing implementation.
- Idea and learning benefit: learners should identify the policy's sampled object, objective, estimator and baseline from code rather than assuming every method described as GRPO-style is PPO or the same algorithm.
- Existing coverage: an older published GRPO/RLOO/KTO page exists; no completed current rewrite or explicit Laya objective bridge was established by this scoped review. Earlier decision-theory teaching already covers log/Brier proper scoring.
- Proposed treatment: add an evidence-based connection to probability-reporting policies. Distinguish baseline variance reduction, leave-one-out/group means, group normalization, clipping, reward shaping and auxiliary supervised losses. Compare a direct differentiable loss baseline before asserting a need for reinforcement learning.
- Explanation/example: sample several perturbed logit vectors for one typed question, evaluate a stated scoring reward, and inspect what a baseline changes in the estimator. Use an independently derived small case and explain finite-sample and differentiability assumptions.
- Prerequisites and boundaries: do not claim proper scoring guarantees finite-sample calibration. Spherical and ranked probability score derivations and ordinal semantics are owned by the integrated topic; add a precise bridge rather than duplicating them.
- Evidence: [Laya fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), [Laya score code](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py), and [proper scoring rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), reviewed 22 September 2026. Recheck and pin the actual training implementation when authoring; Jev's RLCD name does not disclose its full objective.
- Resolution: not yet reviewed.
- Implementation/verification links: none yet.
