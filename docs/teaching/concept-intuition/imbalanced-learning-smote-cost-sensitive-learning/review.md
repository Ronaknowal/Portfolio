# Imbalanced learning: whole-lesson concept-transition review

26 September 2026. Read every section of the ten-section lesson, ten practices, all native-program explanations and references, including the source-specific Yeast preprocessing details. Production JSX is the content checkpoint. Independent/browser review is pending.

| Concept / exact location | Finding and action | Representation and transfer evidence |
| --- | --- | --- |
| Intro / §1 rarity and operational decisions | Sufficient explicit counted cases instead of accuracy alone; retained. | Existing count and threshold investigations. |
| §2 threshold, prevalence and precision | Sufficient empirical precision reversal and base-rate arithmetic; retained. | Existing operating-point/ROC–PR representations and original rows. |
| §3 costs versus capacity | Sufficient conditional risk derivation and review-budget distinction; retained. | Cost threshold and capacity examples. |
| §4 weighted logistic loss | Sufficient stable loss, gradient, normalizer and population optimum; retained. | Complete from-scratch weighted implementation and weighting lab. |
| §4 duplication equivalence | Sufficient fixed objective/normalization assumptions; retained. | Existing comparison explicitly limits equivalence under stochastic fitting. |
| §5 SMOTE | Sufficient neighbor selection, interpolation coefficient, conflicting geometry and categorical constraints; retained. | Direct-manipulation geometry lab and complete scratch implementation. |
| §6 training-only resampling | Sufficient train/assessment identity boundary, imbalanced-learn pipeline and temporal/group caveats; retained. | Pipeline figure and concrete validation restrictions. |
| §7 real Yeast study | Sufficient source identity, duplicate protein rule, continuous-score scope, splits, actual outcomes and unknown biological validity; retained. | Recorded data, exact programs and five live labs are not replaced. |
| §8 ADASYN | Added normalization of local difficulty: fractions 0.2/0.8 allocate a budget of ten as two/eight, showing why a difficult mislabeled anchor may be amplified. | Original arithmetic connects the name to its sampling mechanism. |
| §8 NearMiss | Variant table was correct but abstract. Added a common-coordinate diagram and distance table where nearest and farthest rules select different rows. | New nearmiss-opposite-neighborhoods: minority 0,1,10; majority 0.5,5,20,30,40; one retained row, one neighbor. Remote losing candidates are explicitly outside the view. |
| §8 cleaning and ensembles | Sufficient mutual-nearest definition, vote rule, eligible classes and distributed majority retention; retained. | Existing Tomek/ENN/ensemble distinctions, no claim that disagreement proves a wrong label. |
| §8 focal loss | Sufficient derivative of modulating factor and gradient comparison; retained. | Existing live gradient lab and scratch expression. |
| §8 prior shift | Added prior frequency versus class-conditional measurement evidence explanation before the odds correction. | Existing live correction lab and shift assumptions retained. |
| §8 information and cost | Sufficient independent-evidence limits, neighbor costs and scope of resampling; retained. | Existing procedural cautions and operational accounting. |
| §9–10 transfer | Ten explained practices and readiness retained. | New NearMiss and ADASYN constructions deepen the variants section without forcing a generic lab. |

## Research actually inspected

- [Imbalanced-learn under-sampling guide](https://imbalanced-learn.org/stable/under_sampling.html), section 3.2.1.2: all three NearMiss selection rules; also Tomek/edited-neighbor distinctions.
- [Imbalanced-learn over-sampling guide](https://imbalanced-learn.org/stable/over_sampling.html), sections 2.1.3–2.1.4 and 2.2.1: interpolation, anchor choice, ADASYN neighborhood allocation and failure cases. API parameter details were not inferred from an old textbook.

## Checks and representation

The static NearMiss illustration uses a shared coordinate scale, circle/square shape distinctions, textual coordinates and an exact distance table. It is not a slider or a measured dataset. Three remote majority candidates are named so the class labels and global retention decision are coherent.

The author verifier recomputes nearest/farthest distances across all five candidates, ADASYN proportions and cancellation of unchanged likelihood ratios in prior correction. JSX parse included. All existing data/fit campaigns, complete code and interactive models remain unchanged. Independent teaching and browser inspection remain pending.
