# Published lesson usability review — 21 September 2026

The user's follow-up asks whether the misleading controls and theme defects occur elsewhere, and authorizes fixing confirmed issues. This review covers the published learning interface. It does not claim that every older article has received the current end-to-end teaching/content review, nor does it implement the 42 prepared Deep Learning revisions.

## Scope and outcomes

The three completed-module scopes cover all 135 improved topics: 17 Programming, 22 DSA, 57 mathematics and 39 Classical ML. See the [Programming/DSA record](PROGRAMMING-DSA-CONTROL-REVIEW.md), [mathematics record](MATH-CONTROL-AFFORDANCE-REVIEW.md) and [Classical ML record](CLASSICAL-CONTROL-FOLLOWUP-REVIEW.md) for exact interactions, dispositions and source bindings. The earlier [control-theme repair](LESSON-CONTROL-AFFORDANCE-REVIEW.md) remains applicable to unchanged files.

The other 96 published topics received initial-state control/layout triage at 1366, 390 and 320px, including labels, theme candidates, escaping controls and escaping prose/token labels. This is 288 rendered cases, not 288 full lesson reviews. Local horizontal tables and preformatted code blocks remain intentional scrollers. Initial-state scans omit hidden states; selected interactions and source review supplement them. A zero-candidate result is not a claim that all imaginable inputs or all numerical content are flawless.

The same initial-state triage additionally rendered the 135 improved lessons at those three widths: [405 cases](evidence/reviewed-lesson-ui-triage.json), complementing their deeper module checks. Its raw candidates include small checkbox glyphs inside larger clickable labels and controls inside intentional local scrollers; these are signals for inspection, not 130 defective lessons. The recorded Complexity selector overflow predates the final repair; the final Programming/DSA checker covers that exact 320px case. Module follow-ups document actual label activation, target spacing and keyboard access to locally scrolled controls.

## Confirmed repairs

| Area | Repair and retained teaching meaning |
| --- | --- |
| K-Means | Continuous feature-weight slider values now satisfy the model validator; moving from 1 to 1.01 no longer crashes the lesson. All 800 weight/unit settings match an independent rectangle-cost formula. |
| Gaussian Processes | Numeric twins have their own labels, and range values retain exact typed/default values instead of snapping to an artificial grid. |
| Vectors and Matrix Calculus | Explicit input labels prevent an adjacent output element stealing the label association. Narrow selectors fit their containers. |
| Momentum and learning-rate schedules | Parameter changes preserve valid inspection cursors. Clearly labelled complete trajectory/policy previews expose the effect of edits while step-by-step tables retain their role. A policy replay is not a retrained model or a forecast of future losses. Singleton scrubbing is disabled rather than presenting a dead control. |
| Numerical PDEs | Changing transport parameters retains the inspection step. The unchanged initial condition is identified explicitly. |
| Topological Data Analysis | A normalized native range coordinate maps the endpoint to the exact computed threshold, so End reaches the same full complex as the explicit button. Mathematical comparisons are not weakened by a tolerance workaround. |
| Programming/DSA | Eleven lingering prediction-first lab instructions now direct live inspection. Independent practice remains separate. Algorithm Correctness and Complexity controls and a Backtracking recurrence fit at 320px; the Ordered Patterns interval diagram retains legible labels inside an accessible local scroller. |
| BPE trainer | Corpus edits immediately restart segmentation; its label is connected to the textarea. Step performs a real merge, restart retains edited text, and restore returns the example. Vocabulary retains the original alphabet and all learned merges. Empty/oversized/reserved-marker inputs are explained and withheld; object-like words and Unicode work. |
| Older shared visuals | Long inline code and token labels wrap without altering copyable text/token identity. BPE labels and trace/token captions use the existing readable secondary text color. Actual code blocks retain whitespace and scrolling. |

The BPE model is intentionally bounded: 1,200 Unicode code points, 40 distinct words, 40 code points per word, and 64 merge steps. It splits on whitespace and uses a reserved `</w>` marker; this is a character-level teaching trainer, not a production tokenizer. Pair-frequency ties follow displayed word order, and merging proceeds left to right without overlap. Weighted pair frequencies and replacement counts are different when adjacent pairs overlap. Preserve the measured/constructed distinction elsewhere in lessons.

The BPE lesson's pre-existing [destination note](topic-notes/byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram.md) about unsupported performance comparisons remains open for its substantive content revision. Fixing its trainer and layout does not certify the older article's benchmark claims or complete either authoring phase. No historical published article is promoted to implementation-complete merely because this UI survey passed.

## Verification and continuation

Retained evidence includes the module records above, [older-published UI triage](evidence/published-lesson-control-triage.json), [BPE browser checks](evidence/published-lesson-ui-repairs.json), [K-Means full slider-domain checks](evidence/k-means-live-geometry.json), and [production source identity](evidence/lesson-usability-production-source.json). The BPE model verifier checks weighted merges, cumulative vocabulary, exact reconstruction, overlapping pairs, Unicode, reserved object keys, bounds, empty input and terminal state. Its model received independent scoped source review. Browser checks exercise actual pointer, keyboard and touch interactions where relevant, not only synthetic state assignment.

The final ledger reconciliation preserves older revisions and untouched topic rows, and binds changed completed lessons to the reviewed source. Shared cosmetic repairs and the older BPE trainer have application-level evidence; prepared content and publication mappings stay unchanged. Future authors follow the updated teaching/engineering instructions for interior slider values, exact endpoints, labelled numeric twins, cursor-preserving replay, null explanations and settled pointer coordinates during testing.

This is a technical usability review with explicit coverage limits, not a whole-site WCAG certification or a global claim of perfect content. No unverified total quality score is assigned. Confirmed defects are tracked by their user impact: the K-Means crash was blocking; inaccessible or misleading controls and phone clipping were functional defects; no decorative redesign was needed. Retain the topic-specific representations, lazy loading, bounded calculations and existing dark/amber visual system.

## Final integration

The final production build contains 29 changed/new runtime files relative to the preceding control-repair snapshot, with all 1,701 application-source hashes bound in its production receipt. Four BPE browser groups, eight Classical ML repair groups and five mathematics repair groups passed on that build; Programming/DSA retains its per-route results and captures. Root visual inspection additionally covered the final BPE phone trainer, GP phone controls, momentum desktop preview, Complexity phone layout and locally scrolling interval diagram.

`verify-bpe-trainer-model.mjs`, `verify-k-means-live-geometry.mjs`, `verify-content-import-boundary.mjs`, `verify-runtime-source-organization.mjs`, `verify-curriculum.mjs` and `verify-learning-artifacts.mjs` passed. The build retains the existing large-chunk advisory; lazy lesson loading and the lightweight content import boundary are intact.

The [final reconciliation](evidence/lesson-usability-reconciliation.json), produced by `scripts/verify-lesson-usability-reconciliation.mjs`, checks the 24 affected completed revisions, their preserved flat histories, the 153 untouched ledger rows, current source/checkpoint identities and retained publication membership. Counts remain 177 current content checkpoints, 135 completed implementations and 42 prepared/unimplemented packets. The two rows affected only by shared inline-code styling (Evaluation Metrics and Calibration) receive a scoped cosmetic checkpoint, not a claim of new mathematical review. Earlier exact-source evidence remains historical where these repairs changed bytes. No commit or deployment is part of this task.
