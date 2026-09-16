# Sequential review — Regularization (22)

Reviewed 16 September 2026, limited to `regularization-l1-l2-elastic-net-dropout`. Production repairs belong to the integration owner; this reviewer changed only the two bounded verification scripts and this record. All ten finding groups are closed on the source identified below.

## Coverage reconciliation

The prepared manuscript, visual specifications and design reconcile with the current reader: all eleven sections, eight figures, four investigations, three complete programs, ten independent practices and readiness/source material are present. The core route retains objective normalization, units and the unpenalized intercept; shrinkage versus exact thresholding; residual coordinate updates and convergence; duplicate-feature ambiguity; the actual 18-candidate airfoil comparison and fold-owned preprocessing; and exact dropout expectations versus nonlinear evaluation. The deeper branches retain SVD filters, paths/solver costs, early stopping/weight decay, normalized MAP priors, factor parameterization, smoothness/group penalties, and the distinct AIC/BIC/MDL assumptions and coding example. No substantive scope omission found. The unchanged empirical campaign is still a development comparison with the 303-row reserve unpredicted, not a new benchmark claim.

The previous independent review's broad derivations and full airfoil refit are reused for unchanged data and examples. This pass adds source/control/paint review and adversarial legal inputs rather than repeating those derivations. In particular, the old review's statement that a constant feature under positive lasso penalty has a flat objective was incorrect and is superseded below.

## Concrete findings and repairs

| ID | Reproduction and consequence | Repair and bounded verification |
| --- | --- | --- |
| R22-1 | I1, z=1.4, λ=ρ=1: at the nonzero optimum 0.4, `scalarSlopes` reported −2 and 0 by applying the zero-coordinate subgradient interval everywhere. Corner copy also ignored λ=0; endpoint copy confused general KKT equality with this strictly convex scalar problem. | Nonzero branches use the same ordinary derivative on both sides; only zero uses an interval. Corner requires λρ>0; endpoint uniquely minimizes at zero. Model derivative checks on both signs and browser feedback assertion. |
| R22-2 | I1, z=±0.07, λ=.7, ρ=.1: decimal multiplication dust left an approximately 8.5e−18 coefficient, graded nonzero while the endpoint text said zero. | Relative four-ULP boundary handling, without an absolute floor. Both endpoint signs tested; nearby ±.0701 survives and unpenalized 1e−20 is preserved. |
| R22-3 | I1, legal λ=0,z=±4: the map's fixed output range [−3,3] put the answer outside its plotted axes. | Range [−4,4]; browser asserts both actual marker centers lie inside the axes and captures the positive endpoint. |
| R22-4 | I2, edit a target/λ/ρ/order then choose the output or row: handlers reset the whole draft, erasing the experiment. Reset itself retained auxiliary output/row/sweep state. | Selector edits retire feedback while preserving the draft; Reset restores all auxiliary state. Browser checks edited values, order, feedback retirement and restored selectors. |
| R22-5 | I2 constant-column preset, λ=ρ=1: zero denominator was called a flat objective although the remaining λ|w| term uniquely minimizes at zero. | Flat requires both zero curvature/divisor and zero threshold. Independent trial weights show the objective increase is exactly |w|; λ=0 remains a genuinely flat case. Both visible cases are covered. |
| R22-6 | I2 at ρ=.5: the table called the pure-lasso value3 the λ making every coefficient zero; the actual mixed model needs6 (and ridge generally has no finite threshold). | Table explicitly labels the value as the pure-lasso threshold; browser asserts the label under mixed penalty. |
| R22-7 | I4 x=(.0001,0), w=(.0001,0), q=.5: the positive excess5e−17 was graded “Exactly equal” under a hidden1e−12 rule on subtraction of nearly equal losses. | Direction uses the analytic nonnegative excess with genuine equality only at zero; feedback exposes that excess and warns that subtraction can round it away. Analytic model assertion and browser commitment/capture. |
| R22-8 | Shared numeric predictions used an inclusive tolerance without rounding slack, so a decimal answer at the advertised boundary could fail; the tolerance was revealed only after commitment. | Four-ULP inclusive comparison and tolerance stated before grading. Browser accepts1±.0001 and rejects1±.000101 in I1; the shared component serves all four labs. |
| R22-9 | F4 followed-path width and F7 connector width attributes lost to `.rg-curve`; F3's “thick” minimizer segment similarly painted1.8 instead of4; F7 difference stems had no stroke and were invisible. | Inline intended widths and a gold base stem stroke. Computed-style assertions require F4 one2.4/nineteen1.1, F3 width4, F7 connectors1.1 and all three gold difference stems. F4 width discovery was the integration owner's; the complementary F3/F7 stem defects came from this review. |
| R22-10 | The integration owner's inspection of the new tiny-dropout capture found unreadable black tree labels (its `.rg-stage` missed the text rule), long scientific outputs colliding with probability labels, a clipped distribution tick, and an “agree exactly” claim despite5.551e−17 versus5e−17. | Separate tree label rows, visible text styles, short distribution labels with the numeric values in a table, exact output grouping, and an honest floating-point explanation/analytic excess readout. New browser assertions inspect both SVGs before reset: nonempty labels, correct computed fill, no intersecting text boxes or clipped labels at1366/390; both widths captured. Final browser passes and the integration owner inspected the repaired390 capture. |

## Evidence and limits

- Reviewer executed `node scripts/verify-regularization-models.mjs`: **PASS,115 grouped checks**, including the independent derivative, endpoint, constant-coordinate objective and tiny-dropout regressions. Browser script syntax check passed.
- Integration owner executed the final production browser run: **PASS,14 cases and36 captures**, after a passing build. The owner inspected the repaired tiny-dropout390 image, threshold boundary and F7 captures: the tree text is visible/separated, the numeric table preserves values, and the corrected geometry is readable. Added one bounded browser group and three captures; existing cases remain. This reviewer did not independently operate or visually inspect that final browser session.
- Native evidence is reused: all three programs and59 recorded oracles, with the examples source unchanged at the digest below. The unchanged data module and59,984-byte dataset still match the previous independent source record and data campaign. Dataset SHA256 `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3`.
- No external citations were re-researched; no full native/data campaign rerun; no later topic reviewed. Historical source hashes are identity evidence, not proof that every historical claim was correct.

## Reviewed source identity (SHA256)

| Source | SHA256 |
| --- | --- |
| packet `lesson.md` | `931a38af31e9a4dd199ac664633c28a9a5eb150aa433dc9d49bb169e66d7a095` |
| packet `visual-specifications.md` | `46c194a9d0ba96b75330349dc1888c54b9043cfec9e32abd77238d6fc9e7251b` |
| packet `design.md` | `13274e7f876b925a832bcfc47034d2c603eaaf5a817ce96acc8632f39ce7e8ca` |
| packet `calculated-inputs.json` | `efbfd554886fc36a8832b05e59510133babaa0be8106b8de9a82b983e363e465` |
| `src/learn/data/topics/regularization-l1-l2-elastic-net-dropout.jsx` | `4beba2cc9794d7a2a43ace68ce3b0c05bb3cfdf4052ba07aee61247513758367` |
| `src/learn/data/regularization-models.js` | `ebaa342983bdda47941647f3bb2eae997d6e8e343fe232aa7982bb3111aa5aee` |
| `src/learn/data/regularization-data.js` | `1c0f6b6f63a69c97a51c0d68b46c32ea1ae92da977690240ce4cf95ab9664609` |
| `src/learn/data/regularization-examples.js` | `1ba29d70893361a5048139d44556fc056e2ff2a53ac0be497e030dd985b9f65e` |
| `src/learn/components/lesson-labs/RegularizationLabs.jsx` | `6900b15be09014ac6798e8a2c3d4182bc2ad6f2c952cb85e3fa9963355a359be` |
| `src/learn/components/lesson-labs/RegularizationShared.jsx` | `e2a51efd7f7ca4f0a3ef44a77486e1be02396ea6f52312a4c4c2a8430b4bc030` |
| `src/learn/components/lesson-labs/RegularizationFigures.jsx` | `4b05c391cb1ae435adfe9de768627f5315054a56f521303a1eb48857eec4d14a` |
| `src/learn/components/lesson-labs/regularization-labs.css` | `3318647c5eb636cac0d489fc26699b724633ee00547a2ea999631f07435b3f9a` |
| `scripts/verify-regularization-models.mjs` | `7b2e61669b93f040ae7629e81bef4df6373a39026fed406d034e3feec8be1599` |
| `scripts/verify-regularization-browser.cjs` | `13e495b4749aeaf8070e040d82ee77b046cff727f620e374eadd8b12e49426d3` |
