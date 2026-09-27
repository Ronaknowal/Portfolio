# Normalization: concept-level intuition review

26 September 2026. Read all ten sections, displayed code, practice/solutions and the seven original lab/figure implementations. Full-mode manuscript is the topic JSX. Independent and browser review pending.

| Transition / location | Disposition |
| --- | --- |
| Opening, changing scale and saturation | Removed premature control catalogue. Retained motivating tanh example, collection question, first-pass route and local tensor definitions. |
| One group's mean, variance, epsilon, affine | Existing 1,3,5,7 calculation and ruler figure explain the arithmetic. Retained corrected output-variance formula, affine-sharing boundary and lost-offset counterexample. |
| Tensor axes / BN, LN, GN, IN | Existing cell-edit membership lab makes exactly which values participate visible; retained. Group limits, shape and parameter-contract differences remain. |
| RMS versus centered normalization | Added mean-square = variance + mean² with [1,3] and [11,13] values. This connects the existing vector geometry and results to the precise removed information. |
| Causal sequence grouping | Added fixed-present / changed-future numerical diagram, showing how the statistics route changes an earlier output despite unchanged early input. Contrasts per-token normalization. |
| BatchNorm mode, buffers and estimators | Existing one-pass table and immediate state lab distinguish current population variance, corrected stored update and evaluation. Retained. |
| Running momentum / immature state | Added three-update weight strip and expansion: .729 initial weight, .081/.09/.1 batch weights, final mean 1.244. Shows why this is not an equal average. |
| Spatial batch size / switch semantics / accumulation | Existing spatial counterexample, three-switch table and microbatch explanation already connect assumptions to behavior; retained. |
| Input backward correction | Existing branching graph and contribution table already show all paths. Added common-shift and positive-scale invariances to explain why the corrections must cancel those directions; finite epsilon distinction explicit. |
| Affine update | Existing full values, direct update and gradient investigation retained. |
| Real-data experiment | All twelve fits, code, fixed protocol, data provenance, results and exact evidence preserved. No new fit claim. |
| Residual placement | Existing zero-branch counterexample, branch diagram and live placement lab already ground pre/post difference; retained. |
| Optimization explanations | Retained distinction between operation, mechanism, historical motivation and empirical claim. |
| Epsilon and precision | Corrected ambiguous 'preserve more scale sensitivity' prose. Added v=ε and v=.01ε output variances; separate amplification of perturbations from common-rescaling invariance. Existing dtype promotion code and range/precision distinction retained. |
| Cost / memory / synchronization | Existing explicit tensor size and byte counts with no benchmark claim suffice; retained. |
| InstanceNorm application / weight normalization | Added concrete (3,4), g=2 direction/magnitude example to distinguish weight reparameterization from activation statistics. |
| Scratch VJP / library / parameter sharing | Existing complete implementation retained. Added two-token grid: row statistics versus column parameter-gradient collection, with batch extension. |
| Train versus eval backward | Existing fixed-stat versus input-stat derivative distinction retained. |
| Practice and references | All eight applications, nulls, changed-state and grouping exercises retained, plus complete scratch modification exercise. |

## Research

Consulted PyTorch 2.14 LayerNorm and BatchNorm2d API contracts: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html and https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html . Checked last-dimension statistics, affine shape/sharing, biased forward variance, corrected running variance and fixed-momentum update. These ground the original diagram fixtures and state expansion. The lesson already annotates D2L's alternate chapter and its mode shortcut; no new empirical explanation or speed claim is adopted.

## Visual/implementation contract

Three native HTML/CSS figures: a dependency calculation over present/future token cells, a probability-sized strip of history weights with an ordered numeric key, and a tensor grid distinguishing horizontal statistics from vertical parameter reuse. No fixed diagram quota and no new control-shaped decoration. Exact values accompany all encoding; amber/neutral palette. Phone verification must inspect the tensor-grid labels and narrow history segments, whose values are deliberately outside the strip.

Production JSX remains the full written source. Existing models, program sources and measured data remain untouched. Author checks in scripts/verify-normalization-intuition.mjs verify the new calculations, parse the source and bind unchanged numerical evidence. Independent reading and rendered desktop/narrow review remain pending.
