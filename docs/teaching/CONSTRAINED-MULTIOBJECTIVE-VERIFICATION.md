# Constrained & Multi-Objective Optimization — author verification

Stable ID `constrained-multi-objective-optimization`, mathematics position14. Author source frozen 2026-09-10T13:42:10.535678+00:00. Root owns independent review, shared registration and production integration; this record does not claim those checks have already completed.

## Delivered learning contract

The complete previous body was read and saved at `scratch/constrained-multiobjective-authoring/original-lesson.jsx`; its original Small/Medium/Large example, hard latency filter, Pareto/weighted/epsilon/lexicographic concepts, units, infeasibility, memory exercise and metric/fairness/tail-latency cautions were retained and developed.

Nine sections now explain allowed versus preferred choices; exact coupled projection and both failed one-pass orders; projected gradient mapping and its assumptions; finite squared penalties, a calculable hinge threshold and strict log-barrier domains; an exact two-block ADMM derivation with actual residuals, tolerances and convergence limits; Pareto equivalence/dominance and unsupported points; unit-aware preferences and continuous scalarization; gradient-mixture limitations; and original-metric diagnosis. Five interactive investigations and two numerical comparison figures are chosen by distinct teaching hurdles. Eleven complete executed programs, three short checkpoints and seven changed-input practice groups supply hints, explained solutions and acceptance criteria. Next teaching bridge remains Probability Distributions & Bayes' Theorem.

The [design](CONSTRAINED-MULTIOBJECTIVE-DESIGN.md) records the title/scope decision, original coverage, visual model contracts and source reading. The [incoming coupled-constraint note](topic-notes/constrained-multi-objective-optimization.md) is implemented and resolved. No shared manifest, index, inventory, ledger or handoff was changed by this author.

## Executed mathematical and native evidence

Command: `node scripts/verify-constrained-multiobjective.mjs` using `scratch/lesson-tools/Scripts/python.exe`. Passed 2026-09-10T13:29:49.811Z.

- All11 displayed programs executed and matched their complete displayed stdout.
- 288 two-order projection states checked against an independently optimized bounded scalar objective, plus projection variational inequalities over41 feasible comparison points each.120 further native targets/budgets extend beyond the visual fixture bounds.
- 21 penalty/barrier states checked against independent scalar minimization, including hinge threshold neighbors, zero penalty and the strict barrier domain.80 changed native center/bound/strength cases checked by derivative/subgradient conditions.
- 120 ADMM configurations cover six target patterns, four budgets and five fixed penalties;9,600 updates satisfy each subproblem's optimality conditions, multiplier update, residual signs/norms, repair feasibility and cost bounds. The actual Python helper also converges to an independently computed solution with tighter tolerances. These are bounded numerical checks, not a proof for arbitrary ADMM variants.
- 1,458 budget/preference states cover empty sets, exact budget boundaries, all three selection rules, zero price and score ties. An independent latency-sorted frontier scan checks the2-D candidate frontier;200 changed native lists in1,2,3,5 objective dimensions are compared against a vectorized domination relation, retaining duplicates.
- 202 continuous trade-off states are checked with independent scalar/SLSQP solves and original constraints.25 unit-price fixtures confirm millisecond/second invariance when coefficients are converted.
-160 changed gradient-pair cases in dimensions1,2,3,8 are compared with a scalar convex-mixture optimizer and the common-descent inequalities. Seven changed practice calculations are checked independently, including the ADMM case that reaches the correct primal point at update1 but moves away at update2 while the multiplier adjusts.
-25 invalid input/domain cases pass. Small nonzero numbers retain scientific notation; values outside the barrier domain return an explicit null for drawing rather than a fabricated finite cost.

Evidence: `scratch/constrained-multiobjective-verification/native-results.json`, `cases.json`, `scripts/verify-constrained-multiobjective.mjs`, `scripts/verify-constrained-multiobjective-native.py`. NumPy2.3.5; independent SciPy optimizers ran in the existing isolated runtime. No CVXPY learner dependency or cloud environment is required.

## Actual browser and reading evidence

Command: `node scripts/review-constrained-multiobjective.cjs`. Passed 2026-09-10T13:38:13.170Z in headless Edge at1440×1050 and390×1050.

At each viewport:96 projection frames,13 penalty modes/strengths,180 ADMM frames,75 Pareto filter/rule states and11 continuous choices. That is375 checked interaction states per viewport. Actual state/readout values are compared with already independently checked pure models. Presets/reset behavior, all step buttons, disabled controls, empty-feasible feedback, passing/failing residual feedback, keyboard button activation,9 real anchors, all11 complete rendered code/output pairs,7 practice groups and5 actual source links pass. No page errors, KaTeX errors, clipped SVG text or document-wide overflow were found.

Command: `node scripts/review-constrained-multiobjective-reading.cjs`. Final reading pass 2026-09-10T13:39:54.729Z at1440,390 and320 pixels. All9 ordinary section openings captured, both inline comparisons and sources captured,20 initially enabled controls traversed in actual keyboard Tab order with visible outlines, slider/select keyboard actions and a practice disclosure activated. All11 displayed mathematics blocks fit, including opened deeper branches. Narrow tables were successfully scrolled with arrow keys (2 at390,3 at320). No page overflow, SVG text clipping, math errors or browser errors remained.

Actual screenshots opened and inspected by the author: all9 ordinary390px section openings; projection, barrier, first/final ADMM, Compact/empty Pareto and continuous investigations at390; desktop ADMM and continuous plots; both390px inline comparisons and sources; unit comparison and ADMM reading at320. The corrected projection and ADMM screenshots were reopened after the fixed coordinate-window change. Plots are original computed teaching figures, not inferred empirical benchmarks. Tall element screenshots can be scaled down by the evidence viewer; ordinary viewport captures establish reading size.

Main screenshots: `scratch/constrained-multiobjective-browser/projection-390.png`, `admm-first-390.png`, `admm-first-1440.png`, `penalty-barrier-390.png`, `pareto-compact-390.png`, `continuous-390.png`, `reading-1-390.png` through `reading-9-390.png`, `inline-1-320.png`, `sources-390.png`. Machine records: `results.json`, `reading-results.json` in that folder.

## Corrections and review bounds

The first ordinary-reading pass found wide displayed equations, not document overflow. Long projection, descent, penalty, split-objective, dominance and residual formulas were broken into smaller mathematical steps; all pass down to320px. Initial wide fixed coordinate windows made the first ADMM disagreement hard to distinguish: equal axes now fit the entire80-step trace for the selected input and remain fixed while stepping. Separate projection axes fit all three stages of the current input. A narrow-table hint exposes the third unit-conversion column.

Before final review, a changed ADMM exercise was corrected to acknowledge that a first correct primal point can be left at the next iteration before the multiplier converges. The x=3 direction example explicitly extends the earlier functions to an unconstrained real-line problem; it is not an allowed x for the earlier[0,2] problem. Equivalent objective vectors are distinguished from incomparable trade-offs. Positive projected-gradient step size is explicit.

`formatting-results.json` records normalized AST equality (including JSX and template values) for conventional formatting of the body, model and lab files. After the last full behavioral run, final edits only clarified the equivalence/step-size prose, added the narrow-table hint and applied AST-equivalent formatting; the final reading suite covers those rendered changes. Models and example programs are unchanged in behavior since native verification. No extra broad rerun is claimed for wording-only changes.

Resources are annotated with honest limits: exact Stanford lecture video URL and bookmarks were verified, relevant transcript passages were read, and full video playback was not performed. Boyd book and ADMM theorem passages and selected consensus/exchange slides were inspected; historical software performance claims were not reused. Specific locators are in the design record.

Frozen runtime source hashes: `scratch/constrained-multiobjective-verification/final-source-hashes.json` (six semantic files). The original body SHA-256 was `3f3470678e7b5d5a063e69c27354035c7455dd62dad0011916b3e3532ffcf654`; the amendment below records its reviewed replacement.

## Paragraph semantics and epsilon wording amendment — 10 September 2026

The parent clarified that epsilon bounds equal to a Pareto target's other objective values force every tied primary optimum to match all target objectives; dominated primary ties are possible for general chosen bounds. This preserves the proof and removes an ambiguity. The author retained that exact change while repairing invalid `<Prose><p>…` nesting:51 groups now contain80 individual `Prose` paragraphs inside fragments. `Prose` itself renders a paragraph. No other text, inline semantics, equations, models or examples changed during the markup repair; normalized-AST equality verifies that precise transformation against the already-clarified body. [Source evidence](../../scratch/optimization-paragraph-repair/source-results.json). Revised body SHA-256: `2dbf3150a4e5fdbef457e0670de54ccc034016f02c6f675f78b39415763a824f`; the six-file source freeze was refreshed.

`node scripts/review-optimization-paragraph-repair.cjs` passed at1440/390/320:139 paragraphs,97 with intended Prose styling, no nested blocks, text/page overflow, math errors, page exceptions or console errors/warnings. All nine ordinary section openings were recaptured and all disclosures opened for DOM checks. The author actually opened the opening and corrected epsilon paragraph screenshots at all three widths, confirming clear paragraph flow and readable text. [Browser results](../../scratch/optimization-paragraph-repair/browser-results.json); [mobile opening](../../scratch/optimization-paragraph-repair/constrained-reading-1-390.png); [corrected epsilon paragraph at320px](../../scratch/optimization-paragraph-repair/constrained-epsilon-320.png). Initial harness attempts hit an intentionally blocked development socket and a transient network-resource failure; the final run passed with the socket allowed and no console errors. No model or broad interaction rerun is claimed for these markup/wording-only amendments.
