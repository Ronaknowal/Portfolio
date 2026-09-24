# Review of the final six Classical ML lessons

Requested 21 September 2026: independently verify the six lessons completed since the CRF–Evaluation Metrics increment, correct substantive gaps, and reconcile module completion. This is a review of Classical ML positions 34–39, not authorization to implement the next module or rewrite earlier completed topics.

The [starting snapshot](evidence/classical-final-audit-baseline.json) preserves all phase-ledger rows, publication mappings, exact scoped source hashes and module order. Existing uncommitted work is retained. Each reviewer reads the complete prepared packet and implementation, reuses applicable prior native results, and checks complementary numerical, teaching and interactive risks. New evidence is saved beside the original reports; source-mutating falsification campaigns are not part of this audit.

| Position | Topic | Audit checkpoint |
| --- | --- | --- |
| 34 | PAC Learning & VC Dimension | Complete, including production integration — [paired review](CLASSICAL-FINAL-PAC-CALIBRATION-AUDIT.md) |
| 35 | Calibration & Conformal Prediction | Complete, including production integration — [paired review](CLASSICAL-FINAL-PAC-CALIBRATION-AUDIT.md) |
| 36 | Rademacher Complexity & Generalization Bounds | Complete, including production integration — [paired review](CLASSICAL-FINAL-RADEMACHER-FORMULATION-AUDIT.md) |
| 37 | ML Problem Formulation, Baselines & Data Leakage | Complete, including production integration — [paired review](CLASSICAL-FINAL-RADEMACHER-FORMULATION-AUDIT.md) |
| 38 | Time-Series Validation & Forecasting Baselines | Complete, including production integration — [paired review](CLASSICAL-FINAL-TIMESERIES-ENDTOEND-AUDIT.md) |
| 39 | End-to-End Supervised Learning & Error Analysis | Complete, including production integration — [paired review](CLASSICAL-FINAL-TIMESERIES-ENDTOEND-AUDIT.md) |

## Starting status and scope distinction

All **39 Classical ML topics are published**, including the final three new publication entries. A fresh inventory reports **38 current completed implementations** and the pre-existing **K-Means & Hierarchical Clustering working revision as stale/pending review**. Its earlier completed revision remains historical. This audit must not erase that proposal or claim that all 39 current revisions are reviewed. Detailed new content review covers the six rows above; earlier current source-bound reviews are reused, not represented as newly repeated reviews.

The old “PAC is next” and “48 prepared implementations remain” statements described 19 September and have been removed from the current entry-point summary. The live ledger now has 42 prepared, not-yet-implemented revisions, all in Deep Learning Fundamentals & Architectures. Its first topic is Perceptrons, Neurons & Activation Functions; starting it requires a new user request. User acceptance is distinct from implementation/reviewer completion.

## Confirmed repairs

No confirmed blocker remains in the six audited lessons. The priority summary below groups related defects by lesson rather than presenting individual assertions as separate findings. Each linked paired review records the precise reproductions, repairs, source files and limitations.

| Priority | Lesson and impact | Completed repair |
| --- | --- | --- |
| P1 | PAC: supported repeated observations could crash; manual/generated requests compared an undefined earlier risk; boundary roundoff rejected an exact target. | Resolve agreeing/conflicting duplicate labels correctly, establish a new baseline between question modes, and use a narrowly justified numerical comparison tolerance. Enforce bounds and repair reset/strip captions. |
| P1 | Calibration: the smoothed one-class explanation was mathematically wrong; tied knots and unbounded intervals produced incorrect feedback or exceptions. | Explain the lab's evidence policy separately from identifiability; handle one-block isotonic maps and empty/finite/whole-line sets explicitly. Correct ECE, preset and branch-specific feedback. |
| P1 | Rademacher: the absolute convention disagreed with its enumeration; the zero-radius class rejected its unique feasible answer; comparisons used the wrong baseline. | Keep tables, plots and grading on the selected convention, handle the singleton origin, preserve edited rows and compare against the actual commitment. Reject infeasible commitments and conceal assessment results until selection. |
| P1 | Problem Formulation: stale capacity exchanges and a supposed other-entity null could change the selected entity; translations exceeded declared bounds. | Clear invalid exchange selections, edit only other entities and enforce the uniform-shift domain. Correct taxonomy and practice captions. |
| P1 | Time-Series: future observations were revealed by input values before issuing a forecast. | Give counterfactual controls independent defaults, stop copying hidden observations, and use an explicit replacement preset. Preserve honest uniform shifts and reset the optional result panel. |
| P1 | End-to-End: prose disclosed held-out errors; specified comparison marks were absent; edited score tiles overlapped; empty slices printed invalid numeric output. | Remove the premature answer, show distinct reference/candidate error rings and exact IDs, stack colliding tiles at their true score coordinate, and explain ungradable empty slices. Correct a reversed subtraction footnote. |

These are six repaired P1 finding groups, with related P2 copy/control polish documented within each group. No P0 was confirmed. The teaching standard now explicitly requires inspecting input values, option labels and accessible descriptions at reveal boundaries, and keeping conventions, units and baseline changes consistent across all representations.

## Checks and findings during work

The integration owner's initial 390 px sweep across all six found no duplicate DOM IDs, missing in-page anchor targets, unlabeled visible form controls, visible KaTeX SVGs collapsed below one pixel, or page overflow. This checks opening states only; it does not validate lab transitions or establish complete accessibility conformance. Topic reviewers assess informative figures and changed lab states separately.

[Shared checks](evidence/classical-final-audit-shared-checks.json) pass for the 1,460-topic catalogue, all 29 modules and 10 paths, generated metadata and routes, 231 publication mappings and 485 separate outlines, and the lightweight content-import boundary. There are 620 individual briefs. These checks establish catalogue/loading structure, not the substantive correctness of each lesson.

The production build passed after the final source correction. [Final production integration](evidence/classical-final-audit-integration.json), via `scripts/verify-classical-final-audit.cjs`, passed **22 groups**: the hub, six lessons at each of 1366/390/320 px, consecutive navigation with earlier styles still loaded, and controlled import/render failures with successful reload recovery. It checks publication/order conservation, actual requested lesson bundles, local lesson links, served download byte identity, persistent completion, formula geometry, theme and labeled controls. Each isolated page requested only its own lesson body and shared dependencies; the hub requested no lesson bodies or outlines. The independent reports own technical and teaching correctness; integration alone is not a content-quality certificate.

The first integration attempt timed out waiting for reload's DOMContentLoaded event. An isolated reproduction recovered both failure modes in two import attempts; the final full production run also passed both cases. No reproducible reload defect was established and no reader recovery code was changed. The verifier now records its active stage and completed groups on failure rather than discarding them.

The final [reconciliation](evidence/classical-final-audit-reconciliation.json) binds the current independent reviews and integration to all 80 scoped implementation files, including 23 repaired files. It conserves all 55 prepared-content file bindings, all 231 publication entries, the 39-topic module sequence and the other 171 ledger rows. The phase-ledger validator and fresh inventory pass. The six ledger records now link their new paired reviews, which retain the original full packet/design and evidence trail; inaccurate copied completion-date prose was replaced without altering actual historical completion timestamps.

## Scoped technical quality assessment

Anti-pattern verdict: pass for the reviewed lesson surfaces. Interval constructions, sign enumerations, availability timelines, capacity exchanges and specimen/error plots represent the relevant mechanisms; they are not decorative variants of one generic lab. Existing dark/gold styling is preserved.

| Dimension | Heuristic score / 4 | Evidence and limits |
| --- | --- | --- |
| Accessibility | 3 | Visible controls have labels; selected keyboard paths, explanatory text and non-color error marks work. No screen-reader or exhaustive contrast audit was conducted. |
| Performance | 3 | Production request graphs isolate lesson bodies; hub loads no body/outline. Cold lesson requests total roughly 1.71–1.85 MB decoded JS (370–416 kB gzip estimate), including shared code/catalogue. No new Core Web Vitals or low-end-device measurements. |
| Responsive design | 3 | All six pass three widths; changed visual states were actually inspected. Some dense SVG labels are small at 320 px, with readable HTML tables/inspectors carrying the details. |
| Theming | 3 | No browser-default blue/purple article links; consecutive-route styles and formula geometry pass. Not an exhaustive assessment of every color/state. |
| Anti-patterns | 4 | Topic-specific representations, intentional hierarchy, relevant practice and explicit evidence types remain intact. |
| Total | **16 / 20 — Good** | A bounded expert assessment, not certification of accessibility, measured learner mastery or every possible input. |

The narrow-screen figure-text limitation is a P2 consideration for future focused `/typeset` work if user feedback warrants it, followed by `/polish`; it is not an instruction to start another redesign. No broad extra verification campaign is required to close this request. Retain the exact-value alternatives, source-bound evidence and complementary boundary checks in future changes.

## Environment note

The first shared development server stalled on a plain root request. Only that task-owned session was stopped. A replacement at port 4184, with a task-local Vite dependency cache, explicit dependency entries and scratch/docs/dist watch exclusions, returned HTTP 200 in a measured 0.475 seconds. No application configuration or unrelated server was changed. Retained review evidence belongs under this document's linked evidence paths.

After verification, the task-owned development and production-preview sessions were stopped. Only the integration owner's generated Vite cache and disposable reload probe were removed, after checking the resolved workspace paths. User servers and retained evidence were preserved.

Automatic approval review rejected removal of the Time-Series/End-to-End temporary audit folder because it contains potentially useful untracked baseline/evidence files. That folder was retained and no alternate deletion was attempted. This does not block source verification or ledger completion.

No commit or deployment is requested. Prepared-content checkpoints and every unrelated ledger row remain conserved at final reconciliation.
