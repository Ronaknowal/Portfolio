# NMF: whole-lesson concept-transition review

26 September 2026. Read all ten sections, three programs' complete surrounding explanations, eight practices and references. Production JSX is the content checkpoint. Independent teaching and browser review remain pending.

| Concept / exact source location | Finding and action | Representation / transfer evidence |
| --- | --- | --- |
| §1 nonnegative weighted patterns and matrix orientation | Sufficient concrete additive row and changing amounts. Retained. | BuildRowFigure and MixtureLab; practice 1. |
| §1 interpretation contract | Sufficient factor versus probability/physical source distinction. Retained. | Concrete loss of cancellation explanation. |
| §2 residuals and loss choice | Sufficient squared, generalized KL and Itakura–Saito comparison with assumptions. Retained. | ResidualFigure; practice 7. |
| §3 multiplicative correction | Expanded one-coordinate changed-input calculation and data-evidence/current-reconstruction ratio; removed prediction-task meta wording. | Exact changed H11 result 2.307692; UpdatePhaseFigure/UpdateLab; practice 2. |
| §3 complete scratch update, zero locking and stopping | Sufficient full implementation and nonstationary-zero counterexample. Retained. | multiplicativeStep, ZeroLockFigure; practice 8. |
| §4 scale/permutation/nonuniqueness and normalization | Sufficient product-preserving transformations. Retained. | AmbiguityFigure; practice 1. |
| §5 real digits, held-out fit/transform, dictionary and rank | Sufficient measured figures and explicit selection boundary. Retained. | FitTransformFigure, CandidateFigure, DictionaryFigure, ContributionLab; practices 3–5. |
| §6 new-row transform → constrained solve | Gap: solver vocabulary did not show why multiplication by a transpose fails. Added two nonorthogonal dictionary patterns: exact amounts [0,1], dot products [1,2] reconstruct [3,2]. | Minimal numerical counterexample before solver alternatives. |
| §6 initialization/sparsity/diagnosis | Sufficient solver-specific tradeoffs and diagnosis table. Retained. | Library settings and objective distinctions preserved. |
| §7 separate convexity and KKT boundary | Added one-sided feasible-motion explanation of nonnegative boundary slope. | Existing stationarity conditions and practice 8. |
| §7 majorization | Added touching upper-bowl intuition before the proof and an exact scalar slice after it. | New nmf-touching-upper-bound curve/table; current h=1, update 4/3, F=5/18 and G=1/3. |
| §7 nonnegative rank | Sufficient ordinary-rank versus nonnegative-support example. Retained. | SupportFigure; practice 6. |
| §7 separability | Added why normalized nonnegative sums become convex averages and pure anchors become corners. | Existing assumptions retained; no universal uniqueness claim. |
| §8 text/spectral applications | Sufficient model-specific interpretation and full wordPatterns program. Retained. | Spectral caveats retain physical boundary. |
| §8 streaming sufficient statistics | Added scalar sufficient-statistics update: sums 5/10 then 14/28 yield pattern 2; changing old codes invalidates old summaries. | Connects saved aggregates to actual streamed update. |
| §9–10 transfer and resources | Sufficient eight explained practices and annotated alternatives. Retained. | Complete scratch/library pathways remain intact. |

## Research actually inspected

- [Gillis, The Why and How of Nonnegative Matrix Factorization](https://arxiv.org/html/1401.5226v2), section 3.1, especially NNLS decomposition, KKT, multiplicative majorization, zero locking, and solver distinctions. The explanation motivates exposing constrained transform and touching-bound geometry locally.

The scalar slice, nonorthogonal dictionary and streamed sums are original, checkable examples. Existing learner references already contain the relevant paper.

## Representation and checks

The scalar plot uses one coordinate with the other fixed, not a claimed simultaneous two-coordinate update. Amber is the true loss; dashed neutral is its touching quadratic upper bound. Exact formulas, values and limitations also appear in HTML. Curves are evaluated from the displayed formulas on a bounded domain, not invented solver measurements.

The scoped verifier checks bound nonnegativity/touching, update values, the transform counterexample, streaming sufficient statistics and JSX parsing. Numerical engines, executed programs and prior fit evidence are unchanged. Independent review and rendered mobile/desktop inspection remain pending.
