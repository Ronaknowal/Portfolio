# Capsules: concept-level author review

Read the entire generated lesson (all ten numbered sections, scratch/library route and eight practices). Updated the canonical manuscript and authoring generator together. Original trained weights, recorded results and interactive models are conserved.

| Location / transition | Disposition and support |
| --- | --- |
| Opening / §1 scalar feature → grouped representation | Retained: wheel/whole motivation, grouped arrows, vector/matrix distinction, geometry and probability caveats. |
| §2 child representation → transformed vote | Retained: explicit dimensions, three-child numerical votes, reinforcement/cancellation and weighted-sum axis. |
| §2 squash length/direction/zero | Retained: rewritten zero-safe formula, worked parent vectors, live norm/direction investigation. |
| §3 routing updates, stable softmax and state lifetime | Retained: three-step numeric trace, editable vote diagram, symmetry/zero cases, trained vs temporary state and concentration limits. |
| §4 tensor grouping, margin objective, reconstruction | Retained: full shape path, coordinate-preserving reshape, margin arithmetic, sum/mean coefficient, selected-class mask and instance capacity. |
| §5 matched training vs post-fit interventions | Retained: complete six fits, measured tables, uncertainty of interpretation and actual native code. |
| §6 invariance/equivariance, transformation and latent edits | Retained: task-specific transformation meaning, measured shift failure, frozen-model intervention, masked-out coordinate null case and composite-split reasoning. |
| §7 radial/tangent derivatives | Fixed: derive each eigen-direction from length and perpendicular motion; added native equal-input-perturbation figure with computed sensitivities, not a performance graphic. |
| §7 detach choice | Fixed: explicit direct-vote and coupling-feedback paths, independently constructed sigmoid product example exposes the missing chain-rule term. |
| §7 coordinate frames / linear intertwining constraint | Retained: actual homogeneous matrix multiplication, noncommuting learned vector map and geometric assumptions. |
| §7 EM mass, means, variances and activation | Retained existing live EM and weighted arithmetic; fixed coding-cost leap with a narrow/wide spread calculation, density-vs-probability explanation and floor motivation. |
| Scratch vs Torch / temperature variation | Retained complete two implementation routes, matched-state checks, memory bound and worked positive-temperature change. |
| §8 full architecture / local sharing | Retained original parameter/storage counts; added explanation of why right-multiplication is restrictive and why parameter sharing alone does not remove per-location votes. |
| §8 variational / straight-through alternatives | Fixed table-only transitions: generic prior/posterior precision example and hard-forward/surrogate-backward example. Clearly distinguished from reproduction of either paper. |
| §8 spread loss | Fixed relative-margin meaning with (.6,.3) vs (.9,.8), zero/positive penalty and common-offset invariance. |
| §9–10 practice/readiness/resources | Retained all eight worked tasks and annotated primary/author resource routes. |

## Actual research

- Read [Matrix Capsules](https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf), §§2–3, coding-cost and activation derivation (PDF pages 2–3). Used to target the missing interpretation of the existing cost equation.
- Read the [Variational Bayes routing abstract](https://ojs.aaai.org/index.php/AAAI/article/view/5785), not its full derivation, for the uncertainty/variance-collapse motivation. The added scalar conjugate-normal example is independently derived and explicitly separate from that algorithm.
- Read [STAR-Caps](https://karim-ahmed.github.io/publications/starcaps.pdf), abstract, introduction and routing background, to check discrete-routing/surrogate-gradient purpose. The new sigmoid backward illustration is explicitly generic and not attributed as the paper's exact estimator. No video watched.

## Checks / visual contract

`CapsuleSensitivityFigure` draws equal-size radial/tangent input segments at s=(3,4); output multipliers are textual derivatives with first-order approximation declared. Semantic panels stack below 600px. All existing routing, squash, EM, geometry and frozen-image labs remain.

`node scripts/verify-structure-learning-intuition.mjs capsule-networks` passed JSX parsing, five arithmetic groups and four original engine/weight/result hashes. Generator now preserves identical JSON bytes during prose-only rebuilds. It repacks the same exact float32 weights; no fit is run. Source hashes are in `author-checks.json`. Independent review, rendering at actual viewport sizes and shared build remain pending with root.
