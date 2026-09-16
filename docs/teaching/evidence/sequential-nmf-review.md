# Sequential NMF implementation review — 16 September 2026

Scope: Classical ML position 19 only, `non-negative-matrix-factorization-nmf`. The independent reviewer compared the complete prepared manuscript and visual specifications, the packet design and prior independent-review disposition, the current reader, nine figures, three investigations, shared primitives/state, models and styles. The integration owner owns production repairs, model/build/browser runs and checkpoint closure. This review does not advance the queue.

## Prepared coverage and learning experience

No missing prepared section, program, figure, practice problem or substantive investigation contract was found. All ten sections, nine figures (F1–F8 plus the separate zero-lock contrast), three investigations, three executed programs and eight optional explained solutions remain present. The first five exercises retain separate hints; the deeper three use the prepared solution-only structure. The first-pass/deeper distinction is preserved.

The full comparison retains observations-in-rows shapes and cell multiplication; additive amounts versus probabilities or physical identity; signed residuals and the separate Frobenius/KL/IS scales; the two multiplicative phases, positive-denominator assumptions and zero locking; joint versus separate convexity, majorization and boundary stationarity; ambiguity beyond permutation and scale; product-preserving normalization and its penalty implications; actual train/validation/test digit fits and the unfavorable PCA comparison; coordinate descent versus complete NNLS, initialization and penalty semantics; nonnegative rank/support rectangles and separability; complete word transfer, spectral assumptions, sparse costs, streaming statistics and the tensor bridge. Current execution of the previously unexecuted word program is explicitly reconciled with the prepared packet. Optional coefficient scaling remains intentionally omitted; the prior review documents that decision.

I1 uses a declared quarter-step lattice, so its supported mixture changes do not encounter the arbitrarily small decimal-input grading issue found in the preceding lesson. Its results bind to the committed inputs and the preceding applied state; edits invalidate them, history is bounded, and post-grade exploration cannot replace the result. I2 keeps separate H/W phases, immutable setup, bounded history and a working Back operation. Its pre-answer operand worksheet is part of the prepared task: the manuscript explicitly asks the learner to inspect the numerator and denominator before predicting the ratio's effect, so that scaffolding is not treated as an accidental answer leak. The phase-label defect below is distinct. I3 holds actual fitted H/W fixed, compares old versus new masks, preserves source-row IDs and row-major pixels, and separately grades total and pixel error. Image changes reset masks and predictions.

F1's selected feature remains aligned across addends; F2 separates signed residual from squared loss; F4's two clipped cones use the numerical rays and common observations; F5 draws the shared dictionary and read/write directions; F6 preserves all candidate runs, its inset excludes rather than clamps out-of-range points, and only the true validation winner is ringed; F7 keeps actual and display-normalized dictionaries separate with explicit scales/divisors; F8 retains the six forbidden pairs and exact nonsingular minor. The reviewer inspected the retained/current-baseline 320px fit-transform capture: the shared node and lanes paint legibly, and it corroborates the arrow-shaft discrepancy below. Current screenshot and computed-style checks belong to the integration owner's final browser run.

## Findings and bounded repairs

### N19-1 — optional pixel choice was graded during ungraded exploration (P2, repaired)

In I3, select a pixel, switch a component, select a pixel direction and click **Apply without recording a prediction**. The old result was marked ungraded, making `pixelCorrect` false, but a nonempty pixel choice still triggered “that second prediction missed.” It therefore rejected an answer the learner explicitly chose not to grade. The repaired message reports the computed pixel change and says the exploration was not graded; only a graded result can append matched/missed. The regression uses source row 242, pixel index 35 and component 2, and rejects either grading word in the exploratory result.

### N19-2 — every unchanged total was attributed to zero activation (P2, repaired)

At I3's initial full mask, choose “stays the same” and check without toggling a component. The old unchanged branch claimed a zero-activation component was removed, although nothing changed. The repair records which component switches occurred and distinguishes an unchanged mask, switches of zero-activation components, and other total differences within tolerance. The browser regression requires the unchanged-mask explanation; the existing component-8 removal regression still requires its genuine zero-activation explanation. The fallback does not infer zero contribution from an approximately unchanged scalar loss.

### N19-3 — training update arrow's shaft contradicted its legend (P2, repaired)

F5's training line set a gold SVG `stroke` presentation attribute, but the `.nm-flow` CSS rule set green `stroke` and a thinner width. CSS overrode the attributes: the shaft painted green while only its head remained gold, contrary to the “one gold arrow into H” legend. The repaired line has `.is-update`, with the more specific `.nm-flow.is-update` rule supplying gold and 2px. A browser assertion reads computed stroke and width (`rgb(231, 185, 74)`, `2px`), so this check concerns actual painted properties rather than the presence of an attribute.

### N19-4 — post-H worksheet described the wrong future operands (P2, repaired)

After H but before W, the operand tables use the current W and newly updated H. The caption said these operands described the H phase that would follow W, but W must change first. Direct evaluation on the taught fixture gives a hypothetical extra-H ratio of `1.0025267590805405` with old W, versus `1.0049930331960597` for the scheduled H after W. The new copy names the hypothetical extra H step with W still fixed and says the operands will be recomputed after scheduled W. The regression requires that distinction and rejects the old following-H wording. No update arithmetic changed.

## Evidence reuse and verification limits

The source-bound prior independent review supplies its exact-arithmetic derivations and independent real-data refit; these were read and reused rather than repeated. Its initial source snapshot predates its own repair disposition, so it is not misrepresented as the current reader hash. The integration owner's fresh baseline passed 21 model groups and 12 browser cases with 31 captures. The final browser run after all four repairs passed 13 cases with 33 captures, including computed gold stroke and corrected phase caption. The owner inspected the newly painted flow and ungraded-pixel captures and confirmed both are clear.

Independent identity checks confirmed the current example module, all three displayed program code hashes and all three expected outputs exactly match `nmf-native.json` (25 oracles). All six `nmf-data.json` source hashes match current files, including the data module, calculated inputs, packet and served CSV, served provenance and verifier (224 checks). Thus no native refit or program rerun was needed. CSV SHA-256 remains `d93f963c4b2610eb07122a312eec3ddceac18a031477370d71e33835eced728e`; data and native provenance have not been substituted.

The reviewer independently evaluated the two post-H/after-W ratios, inspected all four repair branches, and added the authorized bounded browser regressions. `node --check scripts/verify-nmf-browser.cjs` passes. Production files were changed only by the integration owner; the reviewer changed this record and the browser script. No new external factual claim was introduced.

## Reviewed source identity

Final SHA-256 snapshot after all four repairs:

| Source | SHA-256 |
| --- | --- |
| `src/learn/data/topics/non-negative-matrix-factorization-nmf.jsx` | `734dcb902eb95db45f406f6565eb8e87427fc954646799a03bc6ffcef76da9b0` |
| `src/learn/data/nmf-models.js` | `c4e5a55b6915f8028da3bf7bf2bbeea954f0faf7bed290e6b3e7338cf2e0bc42` |
| `src/learn/data/nmf-data.js` | `28dacb781b62c7479b23f16f774b94cb58de3f239d10759d7a8052c01cb8ee85` |
| `src/learn/data/nmf-examples.js` | `7d6f704030abd4524b46127ff83defca977148ba8602ba02d709e1bf8ae66c86` |
| `src/learn/components/lesson-labs/NmfShared.jsx` | `45ccdc1ccdcd2ea9962fbe583f38cb3ed5a0271558085f027a3bbbba2606a762` |
| `src/learn/components/lesson-labs/NmfLabs.jsx` | `bcbe95241025a553f7313d7ee74ae5b2b2e6590884719d6588aef2adab7b78b0` |
| `src/learn/components/lesson-labs/NmfFigures.jsx` | `4e45c2507d61edf0afd7181ddc048b9754f2496b5064d8aca1f56e68dd4f899e` |
| `src/learn/components/lesson-labs/nmf-labs.css` | `693210fb14e74fb55ac462597280e340eba3adc00a29206449fc905aef52e26e` |
| `scripts/verify-nmf-browser.cjs` | `a6c2b7ab401d43d12d89981920bab38d14063fdf37d45d086e5783fbd241a092` |

Pre-repair identities for reproductions: labs `d287c94617829ed4041262b2c293d581f7a9cb2bbddb662b9bad2ed30fb87da2`; figures `60edb99b9539e91c8c40511516de982e11819cf81e72e5273db335383d510b8e`; styles `6b09ca679a10e11993038b413df9ab5375e1675b084c80c18f07363e511d96d4`. The reader, shared helper, models, data and programs were unchanged by this sequential review.

Independent source review accepts the four repairs. No remaining content omission or concrete source-review defect was identified in this bounded pass. All current checks are closed; position 19 is ready for the integration owner's checkpoint closure. No later topic was audited.
