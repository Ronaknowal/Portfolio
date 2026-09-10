# Mutual Information: independent mathematical and teaching review

Reviewed 10 September 2026 by a second author, separately from the lesson author's native and browser checks. No actionable mathematical defect was found. One program-prompt presentation defect was identified and repaired before integration. This is a bounded review of the implemented lesson, not a claim that tests establish all information-theoretic theorems or learner mastery.

## Reviewed source and reasoning

Read the full lesson, its nine complete native programs, pure models, design and author verification record; inspected the investigation formulas, state boundaries and relevant rendering. The following claims were assessed with their stated assumptions:

- Finite joint-law MI, conditional entropy and signed pointwise information; zero-mass rows, symmetry and invertible relabeling; nonlinear dependence despite zero Pearson correlation.
- Conditional MI and XOR; the Markov factorization behind data processing, its exact information-loss identity and the condition for sufficient representations.
- The IB objective R − βT: the trivial optimum argument for β ≤ 1, the equality qualification at β = 1, and the distinction between β > 1 and a universal promise of informative solutions.
- The auxiliary IB free-energy decomposition, independent marginal/decoder minimization and normalized encoder update. The lesson correctly claims exact block descent rather than joint convexity, global optimality or uniqueness; its symmetric stationary counterexample matters.
- Variational rate upper bounds and prediction lower bounds, their KL gaps and fixed-target constants; alternate β conventions and the Gaussian encoder's variance/standard-deviation distinction.
- Continuous MI as relative entropy, singular deterministic relations and the finite discrete-sign exception; Gaussian noise scaling and invariance under invertible rescaling.
- Empirical plug-in values versus known-law information, shuffled comparisons versus calibrated uncertainty, and the independent marginal-negative sampling contract behind the population InfoNCE lower bound.

The complete practice route contains changed inputs, diagnosis, explicit assumptions and optional hints. No extra diagram or lab was requested simply to meet a count. The existing joint-cell, conditioning, encoder, iteration, variational-gap and sample-table representations each expose a different mechanism.

## Complementary calculations actually run

Command from the repository root:

```powershell
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-mutual-information-independent.py
```

Passed at `2026-09-10T17:30:30.976945+00:00`. The [durable numerical record](evidence/mutual-information-independent-review.json) retains the exact reviewed source identities and results; the script is retained in `scripts/verify-mutual-information-independent.py`.

| Independent check | Why it complements the author's checks |
| --- | --- |
| Six rectangular/relabelled/refined joint laws, including a zero row and three independent category-splitting fractions | Fraction-defined probabilities and 70-digit entropy calculations check MI preservation under refinement unrelated to the other variable, rather than only the square UI fixtures. |
| Six perturbations on both sides of the local binary IB curvature threshold | Independently expands the objective near the constant encoder and compares the calculated quadratic coefficient with the JavaScript model. This is a local fixture analysis, not a global theorem for arbitrary IB problems. |
| Nine nonuniform three-state/two-codeword free-energy identities | Direct Fraction/Decimal evaluation checks the displayed decomposition with arbitrary auxiliary marginals/decoders, deterministic encoder rows and β = 0. |
| Twelve exactly enumerated InfoNCE expectations for K = 2, 3, 4 and four positive score functions | Enumerates conditional positives and independent marginal negatives instead of relying on a Monte Carlo average. A separate wrong-negative counterexample produces a positive claimed bound while true MI is zero, demonstrating why that assumption is essential. |

This run does not repeat or claim ownership of the author's broad native grids, SciPy row optimizations, nine stdout executions or desktop/mobile review. Those remain documented in [the author verification](MUTUAL-INFORMATION-VERIFICATION.md).

## Finding and final disposition

The original body passed stored examples directly to `RunnableExample`, but that shared component does not display `example.question`. Consequently, all nine intended prediction questions were absent from the reading route. The author added a topic-owned `PromptedExample` that renders the question directly before the relevant program and corrected question-string spacing.

The reviewer inspected the final wrapper and all nine call sites and independently read the final file hashes. The author reran the nine generated programs, compared every code/output/title field with the preceding snapshot, and ran `scripts/review-mutual-information-prompts.cjs` with actual public fonts at 1440, 390 and 320 pixels. The author's saved prompt-adjacency, fit and error checks passed, and the author opened the 390px reading screenshot. These browser actions are attributed to the author, not represented as an independent second browser run.

| Source | Initially reviewed SHA256 | Final amendment SHA256 |
| --- | --- | --- |
| Lesson body | `07ae308c3464f2b65e26c07bfd28face7903b0d8d42dc5a053b88cf2fcd86e35` | `5e0e545b07dafdae348914e655c2b2ffe403a3d908125f73ac1910aa67663786` |
| Native examples | `e961c99921b332e99108e1bae07e7bda43f61f8100e8b240cf22664663c38098` | `29574e13ad103256ca0ebd3df5970cc72c825c2b832e92b438ad7119922aca35` |

The pure model remains `f3c40318bf0235e545f6882f40c92c6bafdbfff622f041b493d444181cb3540a`. The lab, CSS and blueprint hashes are unchanged in the author's combined evidence. The initial numerical-review record is deliberately preserved rather than relabeled as a new numerical run against a prose-only amendment.

## Research scope and limits

The reviewer inspected the original [Information Bottleneck paper](https://arxiv.org/html/physics/0004057), sections 3.1–3.3 through the alternating update equations, to check the objective, auxiliary distributions and exponent conventions; and [Contrastive Predictive Coding](https://arxiv.org/html/1807.03748v2), section 2.3, to check its sampling contract, categorical objective and MI lower bound. The review used independent original derivations and calculations, not copied source wording. No video viewing or benchmark reproduction is claimed. The lesson author's broader source and alternate-resource inspections are recorded in its design.

The scoped mathematical/source review and the single teaching-flow finding are complete. Production loading, shared curriculum integration and user acceptance are separate root-owned checks; this record does not substitute for them or an observed first-time learner study.
