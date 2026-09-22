# Authoring notes: Rademacher Complexity & Generalization Bounds

Canonical topic ID: `rademacher-complexity-generalization-bounds`.

## 12 September 2026 — Prepared scope and implementation boundaries

Status: **closed 20 September 2026.** Addressed in prepared content, then implemented, independently reviewed and integrated; see the resolution at the end of this note. No earlier destination note existed when this scoped authoring began.

The [full prepared lesson](../drafts/rademacher-complexity-generalization-bounds/lesson.md) replaces material inaccuracies in the old theorem, loss constants, simulation, framework comparisons and noise-fitting interpretation. The [design record](../drafts/rademacher-complexity-generalization-bounds/design.md) preserves the original source hash, conservation decisions, actual primary-source reading and author checks. The [visual handoff](../drafts/rademacher-complexity-generalization-bounds/visual-specifications.md) specifies topic-specific mechanisms and editable investigations, including complete contrast/null fixtures.

Preserve the no-absolute-value, 1/n convention; state the loss class in the generalization theorem; distinguish empirical from expected complexity and sign simulation from sampling new datasets. A fitted random-label model is lower evidence about a supremum unless an upper optimization certificate exists. Norm, loss range, margins, class selection and representation information boundaries must remain explicit.

The real experiment intentionally keeps improving measured validation performance alongside vacuous bound expressions. It uses disjoint representation80/fit240/validation80/assessment80 roles on the retained480-row Banknote subset. Do not replace these measurements with an invented U-shaped error curve or present the fixed-corpus diagnostic as an iid deployment certificate.

Immediate sequence: Calibration & Conformal Prediction → this lesson → ML Problem Formulation, Baselines & Data Leakage. Full PAC/VC lower bounds stay with the preceding theory owner; advanced local-complexity, PAC-Bayes, stability and neural proofs are optional research routes here. No runtime reordering or new title is required.

## 2026-09-20 — Resolution

Closed. The prepared content's treatment was checked claim by claim against the frozen manuscript — convention, loss class, the empirical/expected separation, the lower-evidence direction of a random-label fit, the disjoint role allocation, improving validation beside vacuous expressions, and the topic's position in sequence — and all were accurate. The duplicate-feature disclosure now travels in the served attribution as well as the manuscript.

The independent review reproduced the mathematical layer exactly: exhaustive sign-pattern enumeration in exact rationals for all six classes with the positional arrays checked element-by-element rather than by aggregate, and a full replay of the bounded-norm experiment by an independent FISTA solver reproducing the energy factor and confidence term **bit-for-bit**. One packet-level disagreement is recorded rather than edited: `author-checks.json` states `manuscript_words: 7678` where the frozen `lesson.md` counts **7,769**, the manuscript having gained the duplicate-feature disclosure after that author pass. Nothing computes from it and the packet is untouched.

One defect found during implementation is worth carrying beyond this topic. A lesson stylesheet's root-scoped `svg { height: auto }` also matches KaTeX's own inline SVGs, which take `height: inherit`: it collapsed **all 24 radicals on this page to under a pixel**, so √(ln(1/δ)/2n) rendered as its own radicand, and collapsed nine stretchy accents with them — turning an empirical complexity into the population quantity this topic exists to distinguish it from. That is wrong mathematics, not a blemish, and no offline verifier could see it. The rule is now applied by the shared `Drawing` component so a new figure cannot omit it. See the class record in the PAC design record.

Implementation, independent review ([RADEMACHER-INDEPENDENT-REVIEW.md](../RADEMACHER-INDEPENDENT-REVIEW.md)) and integration are complete, with source hashes and evidence bound in the delivery ledger. This note is closed.
