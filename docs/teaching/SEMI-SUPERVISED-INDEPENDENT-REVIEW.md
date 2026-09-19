# Semi-supervised learning — independent implementation review

19 September 2026. Reviewer: root agent, separate from the semi-supervised implementation author. Scope: this topic's complete prepared manuscript/specifications and final implementation; no earlier completed lesson is reopened. Final shared build/loading checks belong to the increment record.

## Disposition

No material findings remain. Three issues were corrected and checked against the actual rendered behavior:

| Finding | Correction and evidence |
| --- | --- |
| P2: A valid, well-conditioned anchored chain with both weights `1e-14` was rejected as singular by an absolute pivot test. Global conductance scaling must not alter the harmonic solution. | Row-normalize the hard equations before solving and compute normalized averaging terms before multiplication. Keep the reported residual in original conductance units. Compute symmetric normalization without multiplying tiny degrees first. Independent resistance and closed-form star references pass, including uniform scales down to `1e-300`; the browser accepts the equal-weight tiny chain and correctly grades its half-score. |
| P2: Two donors could send opposite labels into the same previously unseen recipient category. The refit correctly withheld that category rule, but the UI counted one newly learned rule. | Count only categories actually supported after refitting. The independent four-row counterexample, reversed row order and swapped views pass; the browser displays zero new rules and correctly grades the unsupported recipient as unknown. |
| P2: Prototype axis ticks used approximate flex spacing, so printed coordinates did not exactly align with the SVG coordinate map. | Derive each HTML tick's percentage from the same quantitative map and center its text there. All five rendered tick centers agree with their SVG projections within 0.1 px at desktop and phone widths. |

The graph legend's imprecise “double-weight” outline wording was also changed to “thick.” Screenshot-only navigation hiding was corrected so the fixed navigation cannot conceal rows inside tall isolated captures. These changes do not alter the retained phase-one packet.

## Correctness checklist

- Read the complete ten-section manuscript, all practice answers, the visual/interaction contracts, implementation models/components, author record and exact runnable-source handling. The manuscript remains fully represented; programs are complete and alternate sources remain annotated.
- Reused the author's actual native execution of all three programs and twelve grouped model checks for unchanged experiment/source bytes. No claim is made that the reviewer independently repeated the native ML experiment.
- Added [a reproducible complementary verifier](../../scripts/review-semi-supervised-independent.mjs), recorded in [source-bound evidence](evidence/semi-supervised-independent.json): **234 analytical numerical comparisons and six groups**. Series resistance supplies the hard-network reference; direct algebraic substitution supplies the unequal-weight soft star reference. Neither reference calls the implementation's solver or its trace.
- The hard score `b/(a+b)` and symmetric-soft readout `sqrt(b)/(sqrt(a)+sqrt(b))` remain distinct. Raw evidence is not presented as a normalized probability. Unanchored components and alpha-zero support remain unavailable rather than secretly assigned a label.
- Prototype translation symmetry, empty/rejected pools and final refits hold. Co-training retains observed labels, separate recipient arrays, synchronous offers, conflicts and provenance.
- Author model checks and the thirteen-group browser suite were rerun after the fixes. Earlier unchanged native evidence remains valid. The final browser record binds the modified source rather than only the earlier author's revision.

## Learning-experience checklist

- The lesson starts from the scarcity of labels, then distinguishes assumptions behind graph, self-training and multi-view mechanisms. Concrete arithmetic and the identical-input/different-target counterexample precede abstractions.
- Seven inline figures are useful without interaction. Three investigations expose different editable entities and intermediate states. The graph can be repaired, a pseudo-label can move a later decision boundary, and a recipient learns through its own representation.
- Predictions start unset; committed and final states are distinct; changing an input retires stale answers. The complementary browser checks exercised new learner-authored tiny-weight and conflicting-category cases, plus the coordinate-change prototype example, at 1366 and 390 px.
- Inspected the author's numerical evidence-strip, promotion-audit and changed graph captures, plus the reviewer's own final mobile prototype capture. Counts, direct numerical tables, corrected tick positions, origin lanes and feedback are readable and agree with the code. The author independently inspected the remaining figures and phone states; that full visual review is retained in its record rather than attributed to this reviewer.
- The baseline is allowed to win the real-data comparison. Guessed labels never become observed evidence merely because the model is confident. Deeper methods are optional; practice includes changed cases and explanations rather than only preset completion.

Independent review is complete. User approval is not asserted. Production integration and the central phase-two checkpoint are owned by [the five-topic increment](CLASSICAL-QUERY-EVALUATION-IMPLEMENTATION.md).
