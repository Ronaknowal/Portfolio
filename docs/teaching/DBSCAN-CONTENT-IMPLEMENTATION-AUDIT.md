# DBSCAN prepared-content implementation audit

Reviewed 14 September 2026 within the user's PCA-through-GMM audit. Scope: `dbscan-density-based-clustering` only. This reviewer did not author the original manuscript or implementation. Root subsequently assigned the concrete corrections below to this reviewer; browser execution and final source-bound integration remain root's responsibility.

## Assessment and coverage

The implementation preserves the complete manuscript's fourteen-section progression, all nine executable teaching programs, all twelve changed practice tasks with separate closed hints/solutions, real Iris analysis and annotated alternative resources. It is a substantive implementation rather than a shortened summary. Four concrete defects were found in interaction commitment, the traceability of a comparison population, proportional figure geometry and accessible numeric descriptions. They have been corrected in source; this report does not claim that the new browser checks have run.

| Prepared teaching contract | Implemented location and assessment |
| --- | --- |
| §§1–3: closed, self-counting neighborhoods; core/border/noise; graph components; order-dependent shared borders; asymmetric reachability and failure of transitivity through borders | Production §§1–3 preserve definitions, exact A–J counts, the proof and DBSCAN* distinction. F1/F2 make the intervals and forbidden bridge visible. L1 supports changed coordinates, counts and visiting order. Practice A–D transfer the reasoning. |
| §§4–5: complete small expansion algorithm; monotonic core membership versus nonmonotonic component count; inverse count/core-radius definition | Program 1 retains the full finite algorithm and explanation; Program 2 and F3 preserve all ten ordered distances and the self/index convention. Radius/count limiting cases remain explicit. F3's accessible-number formatting required correction below. |
| §6: units versus metric, unequal-axis counterexample, geographic distance | L2 retains true equal-axis geometry, per-axis multipliers, both radii, a cited pair and the four/five-row contrast/null. Geographic program and practice E remain complete. Previously documented omission of arbitrary row edits is reasonable here: transformation inputs and changed radius/pair provide the investigation, while L1 owns arbitrary coordinates. |
| §7: complete measured-data analysis, species withheld from fitting, representation, coverage and conditional populations | Program 3, seven supported reference settings, all-four-feature fit and selectable plotting projection are retained. L3 shows actual counts, row details, conditional scores and common-row species ARI. The missing exact saved/common population IDs were a genuine handoff gap and are now exposed in closed text disclosures. Practices G/K retain the changed setting and report criteria. |
| §8: incompatible radius intervals, equal-density null, repair, curved-shape comparison | F4/L4 preserve exact dyadic inputs, open/closed boundaries and an actual graph at a learner-selected radius. The shared adjacent table is a reasonable documented replacement for a duplicate F4 table. The optional ring branch retains the generated coordinates, fit memberships, actual bisector, complete program and specific-reference limitation. Practice F gives an independently changed interval. |
| §§9–10: OPTICS candidate direction and high-reachability core starts; HDBSCAN mutual maximum, MST argument, condensation and EOM versus leaf selection | F5 plus its complete ordering table, the OPTICS program, mutual-distance derivation, MST program, abstract stability tree and HDBSCAN program remain. The implementation keeps the sklearn/contrib self-count distinction and avoids treating membership strength as a calibrated posterior. Practices H/I exercise changed values. F5 accessible numbers and F6 proportional areas required corrections below. |
| §§11–12: output-sensitive cost, duplicate multiplicity, sparse graphs, partition boundaries, frozen-reference versus refit | All substantive prose and the complete duplicate program remain. Practices J/L assess memory arithmetic and the new-row contract. The successor remains Anomaly Detection. No invented performance plot or sklearn `predict` was introduced. |
| §§13–14: practice, readiness and resources | A–L are all retained with full solutions; the readiness table maps outcomes to lessons/labs. UBC video and companion, JSS, original paper, implementation-author HDBSCAN article, official APIs and data attribution are retained and annotated. The video is honestly marked not watched. DBCV remains an explicitly reasoned specialist deferral in the destination note, not an undisclosed omitted outcome. |

The manuscript's writing-phase status text and visual authoring directions correctly do not appear as learner-facing website instructions. Execution targets were replaced with the recorded real code/output. Previously recorded mobile scrolling, numeric coordinate entry, closed-choice predictions with reasoning in prose/practice and fixed native OPTICS/HDBSCAN outputs are appropriate adaptations. F7 relies on its computed plot, declared coordinate generator and accompanying complete program rather than adding a separate selected-point table; the relevant shape/center comparison remains self-contained without another interaction.

## Correctness findings and source corrections

1. **F6 area encoded the wrong row ratio.** In `DbscanFigures.jsx`, the parent rectangle had width 120 for six rows while each child had width 55 for three rows. Since the text claims width means row count and area means stability, the drawn children had only 11/12 of their intended width. Changed both child widths to 60, retaining the visual gap independently. Browser assertions inspect actual SVG widths and child/parent area ratios of 18/12 and 6/12. No density or stability formula changed.
2. **Accessible descriptions rounded exact values using array position.** F3 and F5 called `.map(number)` where the formatter's second argument is decimal precision; Array.map supplies the index there. F3 therefore described its first 0.5 as 1, and F5 rounded 0.75 to 0.8. Both now call the formatter through a one-argument wrapper. Browser assertions check the complete exact accessible sequences separately from visible labels.

The official [DBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) parameter/implementation notes and [HDBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) membership/Notes sections were re-opened on 14 September 2026; they identify version 1.9.1 and support the stated self-count, multiplicity, neighborhood-memory and cross-package convention. This is a focused source check, not a new full literature review or a claim that the linked video was watched.

## Learning-experience findings and source corrections

1. **Predictions were not actually frozen.** `Prediction` originally graded live dropdown values; changing a choice after reveal could change a miss to a match. All four labs now store a copy of the answers with the applied state and disable those fields while the comparison is shown. Input edits clear the next prediction. L1 now includes the selected row in its question key, preventing a prediction for border I from being silently regraded for noise J. L2 similarly includes the cited pair. Reset and configuration changes retain their existing hidden-result workflow. Exact correct/missed, selected-row, cited-pair and changed-input cases were added to the existing browser verifier.
2. **The common population could not be inspected.** L3 showed intersection sizes but omitted the actual saved/current/common ID sets promised by the specifications. Closed disclosures now list current assigned/noise rows, snapshot A assigned/noise rows, common IDs and IDs retained by only one setting. These retain the original CSV IDs and do not require a species reveal. Browser checks verify the two 150-row partitions, 116/61 retained counts, the 61-row intersection and exact differences. The added text wraps naturally; root must verify it at the existing narrow widths with the disclosures open.
3. **F6's comparison disappeared beyond the phone-width scroll.** The coordinator's subsequent visual inspection found that the 320 px view showed only one scenario and the scrolling wrapper offered no keyboard region. F6 now uses two separately named SVG panels inside the existing responsive `db-figure-pair`, which stacks them on phones. Both retain the same 220 × 194 coordinate frame, λ-level grid, parent width 120 and child widths 60. Scenario-specific arithmetic replaces “Left/Right” instructions, and labels use the existing readable paired-figure sizing. No new CSS or horizontally scrolling wrapper is needed. The existing verifier additionally checks both scenarios, equal coordinate/rendered scales, viewport containment and phone stacking. The coordinator will rerun the affected production/browser check; this source edit is not itself a claim of visual closure.

## Separate learning-experience checklist

1. First-pass route and deeper branches are explicit; advanced OPTICS/HDBSCAN mechanics follow a complete DBSCAN analysis.
2. Cautions are collected near their actual issue; no new repetitive warning or code guards were added.
3. The task is concrete: group measured flowers while accounting for observations left out. Iris and the constructed trail have distinct stated roles.
4. Each lab offers useful unsolved parameter/entity changes and specific contrast/null cases. Frozen prediction semantics needed correction rather than merely counting four labs as sufficient.
5. Inline figures are placed at the conceptual hurdle and preserve actual geometry. The area/accessible-value defects show why numeric tests alone were insufficient. The coordinator's 320 px inspection additionally exposed F6's hidden second scenario; its two named panels now stack with matching scales. Final rendered confirmation of that targeted repair remains with root.
6. Count, core-radius, OPTICS and mutual-reachability views are explicitly connected; border geometry and expansion have a local correctness argument.
7. All nine displayed programs are complete and focus on the mechanism. Their unchanged code/output is bound to existing native evidence.
8. A–L change values, data or assumptions. Practice K supplies checkable results for a new representation and then requires an independently justified radius.
9. Previous selected screenshots remain historical evidence for the original rendering. This reviewer did not open or capture new screenshots; root owns inspection of the corrected informative states and narrow layouts. This is a heuristic review, not a learner study.

## Verification and handoff

- Read the complete prepared manuscript and specifications, complete production lesson, all four labs and seven figures, pure models, all nine teaching programs/outputs, stylesheet, provenance, design/deviation record and relevant independent-review/evidence records.
- Checked the live topic command and its destination notes. Read current repository/handoff, teaching-stage and learning-experience requirements. No curriculum reorder or new topic is part of this audit.
- Exact numerical model and generated Iris-data hashes match `evidence/dbscan-models.json`; its 23 grouped numerical checks remain applicable. All nine displayed code/output hashes match `evidence/dbscan-native.json`. These unchanged runs were reused, not relabeled as new executions.
- Executed `node --check scripts/verify-dbscan-browser.cjs`; parsed both changed JSX files with the installed esbuild; checked code/output hash equality. No mathematical model changed, so the numerical suite was not rerun.
- Added regression assertions to `scripts/verify-dbscan-browser.cjs`. **Root must run them on the final production build, inspect F6 and the corrected L1/L3 states at desktop/phone widths, then record final hashes in the shared ledger.** No new browser pass or overall readiness is claimed by this file alone.
- No scratch artifact or unrelated source was created/removed. No deployment, stage or commit was performed.

After the coordinator's initial corrected-browser pass, the focused F6 phone-layout change received another JSX parse, browser-verifier syntax check and scoped whitespace check. Area formulas, pure models, data and displayed programs did not change; their numerical/native evidence remains reusable. The new F6 layout assertions require the coordinator's fresh run.

## Source identity

The table below identifies the final topic source handed to root; later root browser corrections, if any, must update the shared final record. All prepared manuscript/specification bytes, numerical model/data and teaching programs are unchanged. See the design appendix for the original corrected-source baseline.

<!-- Source hashes appended by the audit's bounded hash check. -->

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/dbscan-density-based-clustering.jsx` | `5a1cc13ec51cb4c6f8d3287e268cec5f02b4bab09c2f6cfe45b92e69dea3855e` |
| `src/learn/components/lesson-labs/DbscanLabs.jsx` | `db1a1b86de5d6508fa708036569864998131b794881a6b4637b0e45ea6cd261f` |
| `src/learn/components/lesson-labs/DbscanFigures.jsx` | `3b047ec31e69f4124b1aca9b534ff5e26720288b343a8efda55a43348b5310f1` |
| `src/learn/data/dbscan-models.js` | `cd32c82e8ee50030d7d2670f05b0a28d4b29629a122bea391b1f9bb88eb921b6` |
| `src/learn/data/dbscan-iris-data.js` | `4361a654de47ef3f8b97643f7339accf15af8489dfdb238633c8a6b214c1cfae` |
| `src/learn/data/dbscan-examples.js` | `b27124dbfa1154b62879418e3d178e64201771b5b0e3b4597b8c764f0aa0ac2e` |
| `src/learn/components/lesson-labs/dbscan-labs.css` | `27dc2cb98a8f662353737c69f497c636c466e5ad5c8c78499cb224255ea3eb9f` |
| `scripts/verify-dbscan-browser.cjs` | `4733cf4713c5a7813b5854aa33c7b8b0543ac1040de275fb372417a628460422` |
| `docs/teaching/drafts/dbscan-density-based-clustering/lesson.md` | `8e884b5d3177f098f44c7ab8732e227da3fa5fce048e71475d5ee4698d46e7f0` |
| `docs/teaching/drafts/dbscan-density-based-clustering/visual-specifications.md` | `b089b3176ed3b74e9b9b14087e0097623cc697460a3d93e4472ae2fcaa27813a` |
| `public/learn-assets/dbscan/iris.csv` | `387c9585511b1d4ab2513075377d2f07c5cbf05e7a04fd934fcb254290a7a651` |

## Coordinator browser closure, 14 September 2026

The reviewer-only browser-pending statements above describe the state and responsibility at the time of that review. The coordinator has now completed the affected production browser checks and inspected selected informative captures; the [five-topic closure](PCA-THROUGH-GMM-IMPLEMENTATION-AUDIT.md) names the cases, actual visual observations, evidence reuse and limitations. Its source-comparison evidence binds the final repaired files. This closure does not attribute the coordinator's browser work to the independent reviewer.
