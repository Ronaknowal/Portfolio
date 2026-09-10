# Convex Duality & Lagrangian Methods — verification

Root integration amendment: one raw comparison character in JSX prose was entity-escaped after the original freeze. Complete normalized-AST equality verifies unchanged text and behavior. Final integrated body SHA-256 is `3051164629e9ebcab2de42ad6de094b5d62b378c0a52aaa6c07c7ae883ab236f`; other owned runtime files are unchanged. [Exact before/after evidence](evidence/lesson-comparison-entities.json) and [production integration](DSA-MATH-FOUNDATIONS-INTEGRATION.md) supplement the dated evidence below.

10 September 2026. Stable ID `convex-duality-lagrangian-methods-kkt-conditions`, mathematics position11. Author implementation and local checks complete; root owns the subsequent production build/loading integration and full-goal ledger. Existing publication mapping, title, progress identity and module order were preserved.

## Source and teaching review

Read the full original lesson and retained its useful budget example, inequality practice, KKT coverage and ML/resource connections; see the original snapshot at `scratch/duality-kkt-authoring/original-lesson.jsx` and [individual design](CONVEX-DUALITY-KKT-DESIGN.md). The new lesson develops nine connected sections, four investigations with seven plots, two inline figures, ten complete programs and six independent practice groups. It derives the global lower-bound mechanism before names, then separates signs/domains, convex sufficiency, qualification/attainment, sensitivity and algorithms. Labs are placed at their learning hurdles rather than collected into a single generic lab.

Root independently read the full body, all ten Python programs, pure models and six practice solutions. No mathematical blocker remained after the reachable numerical gap defect below was corrected. This was a finite source/math review, not a learner study or a proof of arbitrary solver robustness.

Final source fingerprints: `scratch/duality-kkt-verification/final-source-hashes.json`, recorded **12:02:55 UTC**. It contains the lesson, models, examples, labs, CSS and individual blueprint. Conventional formatting at12:00:39 UTC preserved the normalized AST, including JSX text and template values: `scratch/duality-kkt-verification/formatting-results.json`, script `scratch/format-duality-kkt.cjs`. The completed behavioral/numerical runs immediately precede that formatting-only change; the final keyboard run uses the frozen source. No behavior-changing source edits followed the successful runs.

## Native and model verification

Run `node scripts/verify-duality-kkt.mjs`; it invokes the installed isolated Python via `LESSON_PYTHON` or `scratch/lesson-tools/Scripts/python.exe`. Independent oracle implementation: `scripts/verify-duality-kkt-native.py`. Final result **11:57:16 UTC**, saved in `scratch/duality-kkt-verification/results.json`; full stdout and serialized states are in the same directory.

| Check | Actual result and independent method |
| --- | --- |
| Complete programs | All10 complete displayed programs executed successfully and matched their expected output after newline normalization. Includes actual CVXPY/Clarabel budget and separately solved hard-margin SVM primal/dual programs. |
| Projection |343 visual states; global L minimizers checked through a NumPy linear system and original objective substitution, gap decomposition/sign/feasibility checks. The actual generalized Python projection helper was compared with64 independently modeled CVXPY problems across dimensions1,2,3,7, including violated, tight and slack targets and changed normals; additional random feasible witnesses checked objective ordering. |
| Scalar KKT |1,989 states; each center's constrained optimum independently obtained by SciPy scalar minimization plus the endpoint, then all four condition contracts and any claimed certificate checked. Negative prices, infeasible stationary points, active zero prices and unused positive-priced slack are included. |
| Sensitivity |525 states;57 distinct changed QP/LP value problems actually solved in CVXPY. Supporting-line inequality, corner one-sided derivatives and nonunique supporting prices checked separately from differentiability. |
| Resource decomposition |5,880 iteration frames;10 primal budget programs independently solved in CVXPY and5,598 scalar local minimizations independently obtained through SciPy plus boundary comparisons. Actual Python price iterations agree with UI states; each dual lower bound and repaired feasible upper witness bracket the independent optimum within stated numerical tolerances. |
| Validation |23 invalid pure-model groups and11 invalid native-input cases rejected. Includes nonfinite values, shape/normal errors, out-of-range controls, negative rates/prices/budgets and boolean/fractional update counts. Nested returned state is immutable. Scientific formatting preserves nonzero1e−12. |
| Practice |Six changed-contract groups independently checked, including exact rational projection onto a new normal, signed equality price, invalid gap, qualification/finite-change distinction, scaled sensitivity, and a resource-step repair. |

Runtime observed: Python3.12.14, NumPy2.3.5, SciPy1.18.1, CVXPY1.9.2; installed solver Clarabel0.11.1. Native checks compare stated moderate-sized educational inputs and meaningful cases, not arbitrary-magnitude floating-point safety. The model is deterministic; there are no invented measured timing/performance claims.

### Fixed numerical counterexample

The cross-review script `scratch/inspect-duality-numeric-boundaries.mjs` found that direct subtraction `repairedObjective−dualValue` could round to −1.7763568394002505e−15 for reachable controls, including b=2.75, α=1.75, initial price0, state18; b=3.25/state17 and b=4/state16 are also covered. A negative number labeled a certificate gap was misleading.

The model and complete Python now evaluate the equivalent sum of weighted local displacement squares, the local boundary derivative contribution and λ times unused repaired budget. The priority remainder computes slack without subtracting nearly equal total costs. The mathematical derivation is in section7. This is not a clamp to an exact zero: the first regression now shows a positive8.620e−18. Raw subtraction remains available only as a test diagnostic. Four negative-subtraction states in the selected regression trajectories receive additional70-digit Decimal sign checks; all stable gaps are nonnegative. The browser has exact regressions for the three control combinations above, at1440 and390. Original and repaired computations remain explicitly distinguished from an exact proof based merely on rounded displays.

## Browser, keyboard and ordinary reading

Used actual headless Microsoft Edge through the installed Playwright package, local dev server5173, at the existing full-curriculum route. HMR WebSockets were closed for stable long checks. No local-storage progress edits, deployment or shared publication changes were needed.

- `node scripts/review-duality-kkt.cjs`: final **12:00:10 UTC**, `scratch/duality-kkt-browser/results.json`. At each1440/390 viewport,274 model-linked states passed:52 projection,83 scalar,40 sensitivity and99 resource states. Checks include all control kinds, presets, parameter-reset behavior, first/last disabled buttons, all9 real anchor jumps,10 complete code/expected-output blocks,6 independent practice groups and7 rendered reference links. Invalid pair readouts decline a certificate; bounded sliders have no free-text invalid-entry path. Zero page exceptions, zero KaTeX errors and zero page overflow.
- `node scripts/review-duality-kkt-reading.cjs`: final **11:59:13 UTC**, `scratch/duality-kkt-browser/reading-results.json`. Normal reading screenshots for all9 section openings at1440/390/320, both inline figures and references. With deeper branches open, all15 displayed equations fit; SVG text stays within view bounds and the document does not overflow. The one overflowing KKT table at320 is a locally scrollable, keyboard-operable region; ArrowRight scrolling was verified. Several equations were broken at meaningful mathematical boundaries instead of shrinking their text.
- `node scripts/review-duality-kkt-keyboard.cjs`: `scratch/duality-kkt-browser/keyboard-results.json`. At1440/390/320, actual Tab traversal reached all26 initially enabled lab controls in DOM order with visible nonzero focus outlines. The conditional kink slider was additionally reached and changed with Home/ArrowRight; select changes and practice solutions were operated from the keyboard. The two initially disabled step controls are separately verified by the main interaction run. There is no focus trap.

Actually opened and visually inspected the saved images, including all four lab mechanisms in390px screenshots, their important changed states and the exact repaired numerical case. Read the desktop projection and smooth-sensitivity figures, both inline figures at390, every390 section opening (including7 additionally at320), and the actual rendered sources. Representative files:

- `scratch/duality-kkt-browser/projection-default-1440.png` and `projection-default-390.png`: half-plane and equal-unit distance geometry, separate target/candidate/optimum/L-minimizer markings with textual/numerical alternatives.
- `scratch/duality-kkt-browser/scalar-positive-390.png`, `scalar-zero-390.png`, `scalar-complementarity-fails-390.png`: shared signed derivative scale and independently labeled four-condition readouts.
- `scratch/duality-kkt-browser/sensitivity-smooth-1440.png`, `sensitivity-kink-390.png`, `sensitivity-active-zero-390.png`: value curve, supporting line and actual changed optimum; no derivative claimed at the corner.
- `scratch/duality-kkt-browser/resource-cycle-390.png`, `resource-cancellation-2.75-390.png`, `resource-converging-390.png`: common resource scale, explicit feasible repair, price trace and tiny positive stable gap.
- `scratch/duality-kkt-browser/inline-1-390.png`, `inline-2-390.png`, `reading-1-390.png` through `reading-9-390.png`, `reading-7-320.png`, `sources-390.png`: ordinary reading, implication assumptions, complete visible references and practical next-topic bridge.

For isolated element captures only, the fixed `.learn-nav` overlay was temporarily hidden and then restored; ordinary reading screenshots retain the real navigation. Screenshot existence alone was not treated as visual review. Interaction correctness, numerical oracles and proof review are separate evidence from appearance.

## Sources, handoff and limits

The [design's completed source ledger](CONVEX-DUALITY-KKT-DESIGN.md) records exactly which primary slides, Stanford official video pages and substantive transcript passages, decomposition notes, CVXPY documentation and SVM mathematical formulation were inspected. Direct Stanford recordings and a transcript are annotated alternate routes in the actual rendered References. No full-video playback, full-book review, arbitrary network-convergence or learner-mastery claim is made. Solver comparisons are numerical acceptance checks on these explicit problems; the lesson separately derives mathematical certificates and explains tolerance limits.

The [incoming KKT note](topic-notes/convex-duality-lagrangian-methods-kkt-conditions.md) is implemented and resolved. The [new SVM note](topic-notes/support-vector-machines-svm.md) preserves a scoped changed-data caveat for its future author; the SVM body was not edited. The next module topic remains Second-Order Methods. Root owns the current blueprint registration, integrated build/recovery checks and final rollout status; this record does not equate a preexisting publication mapping with completed production integration.
