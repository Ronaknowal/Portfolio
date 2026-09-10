# Optimal Transport — implementation and verification

10 September 2026. Mathematics position 24, stable ID `optimal-transport-wasserstein-distance-sinkhorn`. Author implementation, independent mathematical/model review and actual desktop/mobile checks are complete. Root production integration and user acceptance remain separate.

The [design](OPTIMAL-TRANSPORT-LESSON-DESIGN.md) records the original scope and representation choices. The [independent review](OPTIMAL-TRANSPORT-INDEPENDENT-REVIEW.md) reports its bounded mathematical checks and one repaired numerical defect. Final runtime hashes, actual native/browser payloads and opened-image fingerprints are saved in [the author evidence](evidence/optimal-transport-author-review.json).

## Teaching and preservation

The original body was archived at `scratch/optimal-transport-review/original-lesson.jsx`, SHA256 `a1ad58038dd29146e4f64d1ec73749c3e1d539f90480306e669a8184bfb8b076`. Its complete 100-iteration Python program and displayed output are byte-preserved in `optimal-transport-examples.js` and were executed again. The original half-mass locations and entropy interpretation remain the running example. Useful geometric applications, splitting and cost-choice caveats were retained and made explicit. The title, identity, module order and existing publication mapping are unchanged.

The new explanation builds a coupling from row/column conservation, solves the entire two-by-two feasible family and derives a dual lower bound. It separates arbitrary cost from rooted Wasserstein distance, derives weighted quantile/CDF transport, entropy stationarity and log-domain scaling, and distinguishes optimization error, regularization bias and sampling error. The full objective, its linear term and the three-solve Sinkhorn divergence use declared consistent constants. Finite-moment, positive-support, cost/kernel and continuous-map conditions accompany their claims.

The extension section supplies actual calculations for barycentric loss of variance, Gaussian transport, mass relaxation and missed dependence under finite projections. Gromov transport has its explicit quadratic objective and translated-shape counterexample. Timing, feature alignment, generative comparison and return distributions illustrate different uses of geometry. Links point to their existing specialist owners; those owners were not rewritten here.

Ten conceptual sections contain eight complete programs, nine independent tasks with separately revealed hints/solutions, two checkpoints, six investigations and three inline figures. These counts follow this topic's hurdles, not a fixed template:

- A same-axis atom comparison makes near/far movement visible before notation.
- A selected matrix entry highlights its actual mass-width flow. Changing weights, feasible position or costs reveals splitting, ties, reversed preferences and forced plans.
- A price/slack ledger distinguishes a loose valid certificate from a proof of optimality.
- CDF steps and exact shaded gap areas connect cumulative imbalance to required crossing mass and distance units.
- A real alternating matrix trace uses unequal source/target weights so one marginal correction visibly disturbs the other.
- A common-cost-offset experiment produces actual exponential underflow while the stable plan retains the invariant answer.
- The objective comparison displays closed-form and iterative values separately, including a genuinely capped, unconverged sharp case.
- Split destinations versus their conditional mean, and movement versus endpoint mixing, show distinct resulting laws.

## Numerical and native evidence

`node scripts/verify-optimal-transport-models.mjs` invokes the Python companion with the existing scratch interpreter. Final numerical run before semantics-preserving formatting: **16:17:24 UTC**. Python 3.12.14, NumPy 2.3.5 and SciPy 1.18.1. Formatting subsequently preserved parsed JavaScript, all strings and CSS meaning; it did not alter the algorithm or Python source.

| Contract | Actual evidence |
| --- | --- |
| Feasible plans and duals | 1,452 changed two-location states across four costs; 484 independent HiGHS linear programs. Exact marginal constraints, attained lower/upper bound and weighted slack identities. |
| Weighted one-dimensional optimum | 180 independently solved LPs from 60 random weighted datasets at orders 1, 2 and 3, including repeated locations, zero weights and unequal counts. SciPy W1 independently checks the CDF result. |
| Alternating scaling | 36 regularized problems, including the exact lab's unequal weights and 40 half-steps. Independent scalar convex minimization and direct matrix normalization check the log-factor recurrence; finite linear-bias bounds are checked against LP optima. |
| Closed forms and debiasing | 24 equal-weight references, 15 three-solve comparisons, 11 offset cases and 12 CDF scenarios. Unconverged iterates remain explicitly flagged; analytic references are separate. |
| Independent review extensions | 24 additional LPs at orders 1.5 and 4; 21 entropy/KL constant checks; three row/column-cost and split-atom invariance configurations. Raw entropy changes under duplicate representation, while the consistent debiased result is preserved. Maximum discrepancy 7.47e−12. |
| Boundary contracts | 13 malformed-input rejections; frozen output matrices; a separate tiny-positive-mass regression retains both 5e−16 moves and total cost 1e−7. |
| Complete programs | All eight exact stdout comparisons pass: original scaling, Fraction certificate, weighted sweep/CDF, general LP, stable Sinkhorn with stopping, conditional projection, relaxed mass and selected projections. |
| Independent practice | Exact fractions and independent optimization check changed cost 13/10, powered costs/units, CDF area, self-entropy −2.0064088681, discarded variance and the unbalanced stationary point. The integration task's duplication and metric-scaling expectations are checked independently. |

The principal Python suite made 8,570 numeric comparisons. These finite checks do not prove infinite-dimensional theorems. The written arguments and qualified primary-source review are separate.

The independent reviewer found that advancing the weighted sweep at residual `<=1e-14` could drop tiny but expensive positive mass. The repaired minimum-and-subtract sweep advances only at exhausted mass `<=0`; at least one side is exhausted at each step. A marginal tolerance alone would not have exposed the relative cost error. This correction and its concrete counterexample are preserved in the independent record.

`scripts/prepare-optimal-transport-examples.mjs` and its Python companion generate the complete source/output data. New learner programs were formatted with Black 26.5.1 after AST-equivalence checks; the original source was excluded from formatting to preserve it exactly. The standalone standard-library stable solver and the separate browser model agree with independent references; the browser does not execute Python.

## Browser and visual review

`node scripts/review-optimal-transport-lesson.cjs` passed on actual Edge through Playwright at **1440×1000, 390×1000 and 320×1000** after the final copy and format pass. Saved final timestamp and payload: `scratch/optimal-transport-review/browser/results.json`, embedded in the durable author evidence.

Each width exercised 18 meaningful changed states across all six investigations; 28 enabled controls were checked for keyboard focus and at least 44px height. Actual Home/ArrowRight slider updates, route selection, next/previous/end, all resets, tied costs, zero source mass, loose/tight duals, identical CDFs, kernel underflow and capped Sinkhorn states passed. Overflowing tables scroll through keyboard input within their own region.

All ten section anchors match actual heading IDs. Every complete code/output block and its preceding question match the example data. Both checkpoint prompts/solutions and all nine independent questions, hints and solutions are present and keyboard-operable. All 21 equations, including deeper disclosures, fit every tested width with no KaTeX error. No page overflow, invalid paragraph nesting, SVG text outside the drawing region, application exception or failed network request remained.

HMR sockets were deliberately closed to prevent concurrent author changes from replacing the loaded test page. The resulting Vite diagnostic is recorded and is the only excluded console error. Initial sandbox font denial is not described as an application success; final approved-network browser runs loaded the site's existing public fonts without request errors. The final test assertions reject other network/console failures.

Review repaired thirteen initially wide 320px equations using meaningful line breaks, named objectives and shorter equivalent notation; notation was not shrunk. The CDF viewbox gained room for its axis caption. The displayed Lagrangian's missing addition signs were corrected before the final review. Effective SVG labels remain at least 15.36px in the 320px lab drawings, with larger inline labels.

Ordinary reading was captured with the actual site navigation. Isolated component screenshots temporarily hid only the sticky navigation before restoring it. Actual opened images include the changed mass-flow ledger, second alternating correction, scaled CDF area, ordinary opening, capped objective comparison, barycentric figure, corrected 320px Lagrangian, entropy introduction, extension introduction and changed capstone. Captured images are not all claimed as personally inspected; the durable evidence lists those actually opened.

The hardest displayed 5,000-sweep comparison was bounded and memoized. Fifteen warm local Node runs measured about 9.16–12.03ms after moving full objective calculation outside the inner sweep. Actual Playwright fill-plus-render checks took about 59–74ms across the tested widths, including automation overhead; these are local checks, not general user-latency guarantees or a benchmark. There are no timers, background simulations or added browser solver dependencies.

## Research and boundaries

The design's initial primary-source ledger is supplemented by the actual writing-stage review:

- Computational Optimal Transport: one-dimensional section 2.6, finite/entropic derivations, selected numerical sections, unbalanced section 10.2 equation 10.8 with its equal-total limiting qualification, sliced section 10.4's defining projected integral, and Gromov section 10.6.3's pairwise-distance objective. The worked fixtures were derived independently; apparent rendered transcription errors and broad empirical claims were not copied.
- Cuturi 2013 section 4 supplies matrix-scaling structure, with reciprocal parameter convention translated to epsilon.
- Feydy et al. 2019 equations 1–3 and Theorem 1 supply the consistent debiasing and explicitly qualified positivity setting.
- Official SciPy `linprog`/`wasserstein_distance` documentation was checked for the independently executed interfaces; POT 0.9.5 quick start was checked for returned plan/value conventions and stable methods.
- The PIMS institutional lecture page and its course links were checked as an alternate spoken route. Full audiovisual playback was not claimed. The author's OT4ML finite dual page was read; other linked notebooks were not all executed.

No existing incoming note was pending for this topic. Scoped ownership search found the existing flow-matching, GAN and distributional-RL destinations; only appropriate bridges were added. There was no new substantive unresolved destination discovery requiring an invented topic. Dynamic transport PDEs, general unbalanced solvers, statistical convergence rates and full generative training remain further study, rather than falsely claiming exhaustive coverage from finite examples.

Final source organization follows semantic topic ownership: one lesson body, pure models, complete-example data, named lab components, scoped CSS and an individual brief. `scripts/format-optimal-transport.cjs` records AST/string and CSS-rule conservation. Root owns registration, compact metadata generation and production loading/recovery checks. This review is neither observed beginner testing nor user approval.
