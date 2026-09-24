# Loss Functions and Normalization — independent implementation review

Reviewer: `implement_dl_perceptrons`, separate from the root author of both lessons. Review date: 21 September 2026. Scope: the complete prepared manuscripts, designs and every A–G visual specification; the generated learner bodies; authoring renderer/generators; `NeuralLessonElements.jsx`, `LossFunctionsLabs.jsx`, `NormalizationLabs.jsx` and their three stylesheets; both pure model modules and retained measurement contracts. This is an independent source/correctness and learning-experience review, not user acceptance or final production integration.

## Correctness and conservation

Both full prepared manuscripts were read, including deeper branches, practice and resource caveats. The authoring renderer preserves the scientific explanations, equations, tables, API snippets and practice solutions and replaces named representation placeholders at guarded single-use anchors. Seven Loss and eight Normalization changed practices remain separate from live exploration. The on-demand complete CPU programs and their actual output records are present. No new broad architecture or performance claim was introduced.

The author's `evidence/loss-normalization-models.json` records 114 passing assertions and exact parsed replay of the complete nine-fit and twelve-fit programs. Its pure-model hashes match the files inspected. The reviewer reused that evidence and ran complementary checks with `scripts/verify-loss-normalization-independent.mjs`; results and reviewed source hashes are in `evidence/loss-normalization-independent-probes.json`:

1. A changed signed seven-measurement Huber example has a locally minimal optimum at zero.
2. Triplet boundaries remain strict and equal eligible distances retain the first-row rule.
3. InfoNCE is invariant to a common similarity shift, and the duplicate-score case approaches log 2.
4. Unweighted BCE cancellation at ninety negatives differs from focusing or balanced-alpha weighting.
5. Changed three-feature normalization gradients agree with a finite difference and cancel in the common-offset direction.
6. A GroupNorm edit outside the selected channel group leaves that group's outputs unchanged.
7. Large epsilon breaks exact positive-scale invariance while common-offset invariance remains.
8. Evaluation is a BatchNorm-buffer null, and changing affine parameters does not change the running-statistic update.

The complete generated JSX was independently parsed and all actual mathematical strings rendered with strict KaTeX: 151 expressions in ten Loss sections and 107 in nine Normalization sections. No mathematical parse failure occurred. The declared finite UI domain avoids singular pair directions and zero temperature; the corresponding scientific corner conventions remain explicit. No additional numerical correctness finding arose from these probes. They do not certify arbitrary tensor shapes, external deployment data or fresh package installation.

## Actionable findings and disposition

The author repaired R1–R7. The reviewer confirmed source and browser behavior, then applied three explicitly coordinated presentation corrections: observation-figure bottom padding, graph labels moved off their connecting lines, and ordinary table-cell wrapping so numerical tokens remain intact in local scrollers. The scientific helpers and native experiments were unchanged.

| ID | Priority / category | Finding and consequence | Required closure | Current disposition |
| --- | --- | --- | --- | --- |
| R1 | P2, representation | RegressionInfluenceLab originally showed observations as equally spaced value buttons, with only a candidate-loss curve. Spec B requires a location plot; without it, moving the outlying observation does not show its changing position relative to the other measurements and fitted constant. | Add a quantitative common-scale observation representation, retaining identity and the editable selection. Inspect original/outlier/null states at desktop and phone widths. | Closed. Fixed quantitative tracks preserve row identity; independent browser changes move the selected dot from 55% to 85%, and the equal-value null aligns every dot. Desktop/320 captures show the complete last mark. |
| R2 | P2, representation | NormalizationGradientLab originally arranged prose in a sequence of boxes. The direct numerator path and mean/variance dependency branches could not be followed as the computation graph requested by spec E. | Draw explicit forward dependencies and the shared-statistic gradient paths, with the existing numeric table; preserve readable phone composition. | Closed. Explicit numerator, mean and variance dependencies feed the normalized output; the exact three-term gradient table shows their sum. Labels fit the SVG and avoid connecting lines. Desktop and both sides of the phone scroller were inspected. |
| R3 | P2, representation | All negative points in TripletGeometryLab originally used the same square irrespective of hard/semi-hard/easy category. Spec E requests category identity beyond hue, because changing margin or position changes eligibility while the entity stays the same. | Use distinct category markers or persistent adjacent category labels and a matching legend; keep mined selection distinct from category. | Closed. Hard triangles, semi-hard squares and easy diamonds have a matching legend; a separate white border identifies the mined candidate. Desktop/320 captures were inspected. |
| R4 | P2, coverage | LossScalingFigure implemented matrix area/storage but omitted spec G's concrete two-sequence reduction comparison. Abstract denominator prose did not show how unequal lengths change influence. | Add one unequal-length, unequal-loss example contrasting token sum, token mean and mean of per-sequence means. | Closed. The [4] and [1,1,1] comparison gives sum 7, token mean 1.75 and sequence mean 2.5 with correct weighting interpretation. Desktop/320 captures show intact values and a local table scroller. |
| R5 | P2, input integrity | Typing 1.4 into the negative-count editor silently became 1 through Math.round. The field appeared to accept one number while deriving results from another. | Reject nonintegral counts with a local explanation and last-valid state; retain real integer slider endpoints and reset. | Closed. Typing 1.4 triggers the whole-number error without changing the derived result; blur restores 1000. Typing 90 with gamma zero reaches exact BCE cancellation. |
| R6 | P2, linked comparison | LossDecisionLab originally omitted the interval between adjacent recorded probability values for a fixed confusion result, and selection did not identify which confusion cell contained the inspected specimen. These are the specific linked views requested by spec D. | Display the threshold's valid decision interval with the ≥ equality policy, and mark the selected specimen's TP/FP/FN/TN cell; keep probability metrics fixed. | Closed. Independent browser probes confirm the lower observed boundary is excluded, the upper is included, domain endpoints 0/1 work, and selecting one specimen from each of TP/FP/FN/TN highlights its correct cell. |
| R7 | P2, representation | NormalizationPlacementLab described a separate identity path in prose boxes without showing its fork and merge. Spec G calls for explicit pre-/post-norm graphs, since the placement of normalization relative to the sum is the mechanism. | Draw both dependency paths and their addition node, preserving the F = 0 contrast and branch multiplier; inspect the mobile order. | Closed. Both fork/merge diagrams show the identity bypass and addition explicitly, with LayerNorm before the learned branch or after the sum. The F = 0 contrast remains visible; desktop/320 captures were inspected. |

The repairs close missing teaching relationships and input semantics within the prepared scope. Existing model evidence remains applicable to unchanged helpers; focused rendered/control checks close the corrected representations without rerunning unchanged CPU experiments.

## Separate learning-experience checklist

| Required question | Independent assessment |
| --- | --- |
| 1. Route | Both openings state a useful first-pass route and name deeper branches before they arise. Loss connects regression, probabilities and candidate learning; Normalization starts with statistic membership before architecture terminology. |
| 2. Cautions | Conditions have distinct scientific jobs: objective versus decision, validation versus test, probability versus ranking; statistic versus affine sharing, mode versus autograd, raw traffic versus performance. The displayed code prints outputs rather than disclaimers. No new repeated qualification block is requested. |
| 3. Real question | Real digit images and retained measurements answer declared tasks. Loss uses a rare-nine classification question; Normalization matches initial weights, split and data order while measuring evaluation-mode CE. |
| 4. Live investigations | Current results are derived on render; there is no learner-answer, commitment, grading or answer-unlock state. Meaningful entity edits, null cases and resets exist. R5 is closed by actual invalid-input and last-valid-result checks. The author's 16-group browser run also passes actual pointer/endpoints and desktop/390/320 layout checks. |
| 5. Figures | The chosen forms now realize all prepared representation contracts. R1–R4 and R6–R7 were real gaps; source corrections and separate desktop/phone screenshot inspection closed them. Arithmetic checks alone were not treated as visual evidence. |
| 6. Connections | Scalar error→parameter update, focal weight→full derivative, pair→triplet→candidate CE, and mean/variance→group dependence→state/gradient connections are explicitly taught. Canonical and deeper sources remain annotated with actual review extent. |
| 7. Code | Complete CPU programs, targeted API examples and formula helpers retain mechanisms. Verification is outside learner code. API conventions distinguish squared/unsquared distances and population/corrected variance. |
| 8. Practice | Changed units, weighting, margins, candidates, axes, running state, affine equivalence and evaluation protocols require transfer. Hints/solutions remain independently openable; no lab answer-entry requirement remains. |
| 9. Screenshots | The author's development record now passes 16 groups at 1366/390/320 px. The reviewer inspected all distinct represented mechanisms at desktop and 320 px, then separately inspected the final corrected graph/table/observation captures. The 380 px dependency graph deliberately uses a labelled local phone scroller, checked at both ends. Final production integration remains the root owner's separate step. |

## Final closure and evidence

Independent review is complete with no remaining material correctness, content-conservation or learning-experience finding. The author's full browser run passed 16 groups after the observation-padding repair. Subsequent changes only repositioned graph labels and prevented table-cell word/number splitting; these have their own final-source capture and containment evidence. No claim of final production integration or user acceptance is made here.

- `evidence/loss-normalization-independent-probes.json`: eight complementary numerical probes plus complete JSX/258-expression strict KaTeX checks.
- `evidence/loss-independent-browser-closure.json`: three complementary browser groups and eight desktop/320 captures, generated by `scripts/verify-loss-review-closure.cjs`.
- `evidence/normalization-independent-visual-closure.json`: two final-source geometry/containment groups and five captures, generated by `scripts/capture-normalization-review.cjs`.
- `evidence/loss-normalization-independent-final.json`: exact reviewed source hashes and evidence hashes.

Root may now run the shared final production build/browser integration and bind those results to the delivery ledger.

### Deferred complete-program source review

The final performance correction uses `NeuralProgram.jsx` to import each full program only when its disclosure opens and to mount formatted code only while open. The reviewer parsed the component, assessed open/close/error callback handling, and confirmed the enclosing topic content is keyed by topic ID so navigation remounts it. Both deferred strings exactly match their complete downloadable Python files after CRLF normalization. The generated learner bodies still contain ten/nine sections and all 258 mathematical expressions pass strict KaTeX. The final source receipt includes this component, both program modules and refreshed generator/body hashes. Root retains the final production deferred-fetch/open/close/download check.
