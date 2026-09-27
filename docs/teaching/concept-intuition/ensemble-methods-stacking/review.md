# Ensemble Methods & Stacking — concept intuition author review

26 September 2026. Read the entire production lesson, every worked example's explanation, all twelve changed practice tasks and the advanced branches. Direct JSX remains the content checkpoint; no duplicate manuscript or topic-body generator was introduced. Existing example programs, recorded fits and live lab engines are unchanged.

## Entire-lesson concept map and decisions

| Location / new concept | Support assessed and author action |
| --- | --- |
| §1 multiple forecasts → one target; bagging/boosting/stacking roles | Existing shipment illustration and method comparison distinguish output combination from training procedure. Sufficient; retained. |
| §2 convex weights, hard versus soft votes, class alignment, labels/probabilities/margins | Existing direct arithmetic and movable voting comparison explain how the same models can yield different decisions. Retained; heterogeneous meta features receive a later concrete diagram in §8. |
| §3 residual cancellation and fitted weight | Existing per-row cancellation lab, finite square identity and constrained two-model optimum form a complete visual-to-formula sequence. Retained. |
| §3 repeated-data variance, shared covariance and valid correlation; independent majority vote | Existing interpretation distinguishes variance over fits from a fixed residual list, states admissible common correlation, and contrasts independent versus copied errors. Retained; no unconditional diversity guarantee added. |
| §4 bootstrap multiplicity, distinct observations, OOB eligibility, instability and learned preprocessing | Existing row-ownership lab and bag example show duplicated counts and missing eligible votes directly. Retained. |
| §5 weighted stump fit → vote → row update | Existing live lab plus six-case derivation expose the first update, normalization and next round. Retained. |
| §6 why the logarithmic coefficient is chosen | Added exact two-term loss-balance plot before the derivative. It shows the competing decrease/increase and why equality occurs at the optimum, linking the preceding weights lab to optimization. |
| §6 normalization product → training-error bound → weak-edge conditions | Added plain-language multiplication bridge (.8 then .9 leaves .72) before the product formula. Existing uniform-edge versus vanishing-edge discussion, shrinkage qualifier and practice remain. |
| §6 signed AdaBoost versus SAMME; full NumPy comparison | Existing coefficient conversion, boundary policy, complete programs and honest capacity caveat sufficient. Retained. |
| §7 predictions as features, memorization trap, OOF destinations | Existing fold manipulation and A/D arithmetic show exactly which labels are excluded. Retained. |
| §7 OOF meta fit → full-data bases → serving | Added lifecycle diagram with fixed column identities, actual A OOF row, learned weight and new-query arithmetic. This addresses a different question from the existing fold lab: which trained objects survive, and why the meta fit must not be repeated on full-data predictions. |
| §7 fold-size mismatch, correlated columns, matrix dimensions | Existing concrete shape checkpoint and variance/regularization qualifications retained. |
| §8 library output contracts and mixed scores | Added constructed two-branch probability/margin diagram, coefficient contributions and logistic link, so “treat as features” is an explicit computation rather than a slogan. Not presented as a measured API fit. |
| §8 preprocessing and outer selection boundaries | Existing scaler example and nesting explanation distinguish excluding a target from excluding all learned transformations. Retained. |
| §9 group/time ownership, uncovered prefixes, blending | Existing time illustration, executable explicit loops, data-boundary explanation and missing-row rule sufficient. Retained. |
| §10 calibration versus proper loss, calibrated-mean counterexample, fitted prediction field | Existing exact joint-law figure, arithmetic and measured live boundary lab already isolate the concepts. Retained without decorative duplicate. |
| §11 controlled comparison, validation choice, costs, threshold and loss disagreement | Existing full executed report and expected-cost derivation retain all ownership and finite-sample qualifiers. No result changes. |
| §12 context versus interactions, recommender/sensor/labeling applications | Existing slope diagram, exact eight-row construction and availability caveats already supply local intuition. Retained. |
| §12 fit counts, inference counts, parallelism and distillation | Existing derived counts and stated linear-cost assumption avoid fake timing claims. New serving lifecycle reinforces which models are evaluated. Retained. |
| §13 changed practice A–L | Read all arithmetic, ownership and reporting solutions; preserved the full set and capstone reference output. |

## Added visual contracts

- `AdaBoostVoteBalanceFigure`: ε=1/6, α in [0,1.6], correct mass (5/6)e⁻ᵅ, wrong mass (1/6)eᵅ and sum Z. Shared explicit axes, solid/dashed/amber legend and optimum dot α=½log5. At the optimum each term is √5/6, giving total √5/3. SVG paths come from those formulas; not an empirical curve. Numeric labels and explanation remain HTML. Layout stacks on narrow displays.
- `StackTrainingServingFigure`: semantic ordered flow, not a second fold simulator. Features consistently ordered [neighbor, line]. Existing OOF A row [2,−.35], original target 1; learned neighbor weight approximately .228273; new query uses full-data outputs [2,13/3]. Explains what gets refitted and what stays fixed. No fake sliders or prediction gates.
- `MixedStackScoresFigure`: independent constructed combiner with intercept −1, weights [2,.5], inputs [.8,2]. Contributions 1.6 and 1 yield score 1.6, logistic output .832018. The input margin is not a probability; a raw mean 1.4 is invalid. Two column identities and their own coefficients make the representational choice visible. No calibration claim.

## Research actually read

Read [John Duchi's Stanford boosting notes](https://cs229.stanford.edu/extra-notes/boosting.pdf), the signed weighted exponential objective, coefficient derivation and weak-edge bound (pages 2–5), and [scikit-learn's stacking section](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization), specifically whole-training base fits, out-of-sample meta fits and output-method selection. These are already linked as useful alternatives/primary documentation in the lesson. Used their mechanism/contract distinction, with original local figures and existing lesson data; no full video viewing claim. Earlier source annotations and the recorded 1.9.1 native environment remain.

## Author checks and handoff

Scoped JSX parse; exact coefficient, balanced contributions, loss multiplier and complete plot domain; OOF line fit and served blend arithmetic; mixed-score/logistic calculation and twelve practice blocks. Source hashes in `author-checks.json`. Existing native fits and code have not changed and were not rerun. Browser/mobile visual checks and independent review are pending root; no such check is claimed as passed here.
