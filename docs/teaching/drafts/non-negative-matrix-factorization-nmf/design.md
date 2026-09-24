# NMF content design and continuation

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Edit activation amounts and pattern cells and follow every multiplicative term and reconstructed feature live, with an explicitly saved comparison reference. Edit X and initial H to restart a bounded update trace, then step H and W phases separately and step back. Select a reserved image, toggle fitted contributions and inspect image error and pixel changes immediately against the all-components reference.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Stable ID `non-negative-matrix-factorization-nmf`; Classical ML position19, authorized batch position1. Author `/root/classical_feature_content`, 12 September 2026. **Research/write only.** The central phase ledger owns current status; this packet supplies the written checkpoint and continuation instructions. No production code, publication, browser or formal independent review is completed here.

## Scope, learner and conservation

Retain the title: the revised scope still teaches NMF from additive matrix entries through practical use and its important theoretical distinctions. The predecessor is the completed ICA manuscript; next is Feature Scaling, Encoding & Imputation. NMF's local matrix multiplication, squared-error and gradient refreshers avoid assuming the learner can already derive a solver. GMM responsibilities, manifold coordinates, ICA sources and additive activations are explicitly different at the boundary.

Original full JSX read in sections at commit `8c5da59f18516be77c29d5aeeafca3decca4f738`, SHA-256 `732ee926807d266a33230ebdf6d9678000502b8db44afd97f8b8670f7dcd2fbe`. Preflight `node scripts/build-curriculum-inventory.mjs --topic non-negative-matrix-factorization-nmf --work content` read its full notes and individual-design-required state. Published source remains unchanged.

| Original coverage | Disposition in this manuscript |
| --- | --- |
| Purpose, image/text/spectral applications and PCA/ICA/LDA comparison | Retained with mechanism examples; removed universal physical-parts, calibrated-LDA and runtime-ranking assertions |
| Matrix shapes, multiplicative Frobenius/KL, beta losses | Shapes/Frobenius fully taught; generalized-KL objective and library program retained; KL-specific update derivation is an optional primary-paper follow-on, because the core update mechanism is already exposed and duplicating its matrix algebra would burden the first route |
| Auxiliary-function convergence and joint nonconvexity | Corrected: actual nonconvex loss counterexample, explicit positive-denominator assumption, upper-bound derivation, boundary KKT vs monotone objective vs local/global optimum |
| ANLS and coordinate descent | Corrected independent problem counts (n row fits and d column fits), coordinate sweep vs complete NNLS solve and zero activation behavior |
| 20×50 hand-built word-block recovery and 20 Newsgroups fetch | Replaced claimed numerical outputs and hand-authored heatmaps with smaller inspectable exact fixtures, genuine offline digits and a complete constructed word program; word-pattern, mixed-document and new-document transfer outcomes preserved |
| Initialization, sparse factors and regularization | Current NNDSVD variants/defaults and native alpha/l1_ratio support repaired; factor scale/penalty interaction added |
| Rank selection, nonuniqueness, separability | Replaced universal rank/uniqueness statements with exact broader ambiguity, count selection policy, nonnegative-rank counterexample and anchor geometry |
| Sparse arithmetic, online NMF, tensors and alternative libraries | Cost terms corrected, sparse MU retained, MiniBatchNMF current route; tensor CP/Tucker bridge retained. NIMFA catalog and named fastest-library claims omitted because not required to learn this outcome and not verified as a current ranking |
| Exercises and sources | Replaced copied answers with changed numerical/diagnostic/application tasks and optional solutions; canonical chapter and current primary APIs remain learner-facing |

The existing NMF destination note was investigated and incorporated: latent-meaning bridge, exact two-factorizations fixture, normalization and convergence distinctions. Its status remains open until actual implementation; content disposition is linked there. No title change or new catalogue entry is justified.

## Learning hurdle and evidence map

| Hurdle / outcome | Core explanation and example | Representation | Practice / depth |
| --- | --- | --- | --- |
| What an activation contributes | Shapes and [2,1] mixture of two three-feature patterns | F1/I1 aligned contribution strips | New mixture and normalization exercise1; core |
| How a loss values an error | Signed residual before square, exact KL/IS contrasts and scaling | F2 aligned residuals/table | Loss scaling exercise7; core with deeper variation |
| Why entries change | Fully substituted H11 ratio, distinct H/W phases | F3/I2 matrix operands/trace | Changed one-column update exercise2; core |
| Why a factor is not a physical identity | Two exact dictionaries, cone containment, scale invariance | F4 dual cones | Scientific-claim diagnosis3; core |
| How a fitted dictionary handles new data | Actual 180/60/60 image split, held-fixed H | F5 data lanes, F6 real errors, F7/I3 image contribution inspection | Held-out image removal4, pipeline repair5; core |
| Why optimization claims need conditions | Joint nonconvex example, auxiliary upper bound, KKT | Formula derivation plus zero-lock static contrast | Deeper exercises6/8 |
| Why nonnegative rank differs and anchors help | Rank3/nonnegative-rank4 support argument, normalized endpoints | F8 positive support rectangles, simple anchor geometry described in prose | Deeper theoretical branch |
| How to transfer to words/spectra/streaming | Complete constructed text program, normalized mixture and 3-band spectrum, sufficient-statistic shapes | Existing additive and shape vocabulary reused | Deeper independent input variation; no unsupported production demo |

The first-pass route is immediately after the introduction and ends with exercises1–5; deeper KKT/rank theory is not a hidden core readiness requirement. Cautions have dedicated homes: interpretation in§1, objective/zero behavior in§3, split scope in§5. Displayed code contains no printed disclaimers or reviewer-heavy guards.

## Research and reference coverage

All retrievals12 September 2026. Primary sources inform accuracy; original fixtures, calculations, explanations and applications are independently developed rather than copied wording.

| Claim / coverage | Source and substantive material inspected | Result / limitation |
| --- | --- | --- |
| Canonical coverage map | Gillis, [The Why and How of NMF](https://arxiv.org/pdf/1401.5226), section headings and §§1–4, especially 3.1.1–3.1.8,3.2,4 | §2 applications→core images/deeper text/spectra; §3.1 algorithms→MU/NNLS/CD and stopping/initialization; ALS clipping failure is not taught as a recommended algorithm; historical timing comparison stays a reference, not reused measurements; §3.2 self-dictionary/sparse regression and geometric algorithms→anchor geometry locally, full noisy bounds/LP formulations deeper reading; §4 nonnegative rank→explicit rank gap and support argument, full extension-complexity/communication-complexity proofs deferred as mathematical specialist detail. No canonical headline fact silently omitted. |
| Exact updates and auxiliary bound | Lee–Seung [original paper](https://papers.nips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf), problem definitions, update equations and auxiliary-function proof | Transposed to observations-in-rows convention; independent tiny arithmetic and PSD identity expose the proof assumptions |
| Stationarity vs monotonicity | Lin [2007 paper](https://www.csie.ntu.edu.tw/~cjlin/papers/multconv.pdf), abstract, §§II–IV conditions/modified update argument | Ordinary MU monotonicity does not establish all limit points stationary; zero-boundary directions and denominator conditions taught; no claim that our simple update implements Lin's modified solver |
| Physical interpretation / nonuniqueness | Donoho–Stodden [2003 paper](https://papers.nips.cc/paper_files/paper/2003/file/1843e35d41ccf6e63273495ba42df3c1-Paper.pdf), initial geometric and decomposition discussion; Gillis§3 | Incorporated saved note and exact different nonnegative products; additivity separated from recovery assumptions |
| General exact-NMF hardness | Vavasis [original abstract](https://arxiv.org/abs/0708.4149) and Gillis complexity discussion | Claim confined to general exact-NMF hardness; full reduction not inspected or reproduced; nonconvexity not offered as proof |
| Current API/solver/default/penalty semantics | [NMF guide](https://scikit-learn.org/stable/modules/decomposition.html#nmf), its Frobenius/beta/minibatch sections; [NMF API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html), objective, parameters and transform; [MiniBatchNMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html), forget_factor/partial_fit semantics | Inspected docs1.9.1, local1.9.1. Current native regularization/defaults and zero-input beta restrictions explicit. No runtime benchmark or n_jobs claim. |
| Data origin and permission | [UCI optical digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), acquisition/feature construction, data/license | Offline byte-identical prior subset with attribution and exact split; writer-generalization limit explicit |
| Alternate visual learning resource | [Faces decompositions example](https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html), substantive NMF setup and pattern comparison | Useful same-data constraint comparison, separate dataset retrieval and no inspected local execution. No video watch claimed; full canonical chapter and this worked visual article are meaningful alternatives without a format quota. |

API details that are not exercised locally (the supplementary KL text program and incremental fitting) retain clear execution boundaries. Future phase-two execution may reveal necessary local corrections; that is a continuation, not permission to invent their output now.

## Author calculations and data

`author-calculations.py` executed once using shared read-only `scratch/lesson-tools/Scripts/python.exe`; Python 3.12.14, NumPy 2.3.5, scikit-learn 1.9.1. It generated `calculated-inputs.json` from the supplied `digits-300.csv`, preserving raw data. Bounded arithmetic/new-data calculations are content evidence, not a formal native/example campaign.

Actual checked quantities: both ambiguous products equal X; all 41 MU states including initial loss 17.06 and final 4.9313891e−8; changed X[0,1] step; KL/IS loss table; all eight k/seed fits and withheld reconstruction metrics; PCA8 and mean baseline; actual H, W and reconstructed held-out image rows. No recorded fit reached its 2,000 iteration limit (largest count 575). No warnings appeared in this bounded run. Fitted outputs are retained, not inferred from paper figures.

Additional small direct probes calculated all eight removals from first test image source row 242: full row MSE .021503642603; removals one-based 1…8 yield .024686350768, .068409087403, .037522476594, .022283346218, .030620852640, .067307402976, .023776212335, .021503642603. Its component 8 activation is exactly zero. Root noted that the k8 panel must not be described as validation-selected; the manuscript already states its predeclared inspection purpose. Author self-review corrected removal-exercise reasoning to distinguish improved individual pixels from non-improvement of total error at an exact NNLS row optimum.

Exact hand checks: modified H11 initially 3 yields 5.5/7.15×3 = 30/13, declining rather than growing; exact-fit factors give unit ratios and unchanged zero loss; a zero-activation additive edit is unchanged; the nonnegative-rank support matrix has six forbidden cross pairs and rank 3. Independent exercises reconstruct [2,3,7], solve H = [2,1.4] with half-squared loss .1, and give convexity midpoint loss .6328125. These fixture claims can be checked directly without new fitting.

## Author learning-experience check

Complete manuscript read in ordinary order during authoring; complete specifications checked against it. Concrete revisions: replaced universal physical-parts story with factor observations; introduced shapes before products; separated initial/post-H/post-W state; retained an unfavorable PCA comparison and no artificial elbow; distinguished selected count from demonstration count; added unused-zero-column assumption to deeper inverse proof after root's focused finding; clarified row NNLS removal. No formal independent review is claimed.

1. Route: explicit core/deeper split immediately after intro, core readiness only its taught skills.
2. Cautions: single interpretation contract; dedicated optimization and split homes. No warning strings printed in teaching code.
3. Real question: additive image reconstruction returns to measured held-out residuals and factors, with supplied licensed data.
4. Investigations: three different mechanisms, committed input-bound predictions, editable entities and checked contrast/null fixtures. I2 includes an actual shrinking-entry case, avoiding a permanently-growing target.
5. Figures: exact values, scales, raw-vs-normalized factor distinction, one-component baseline and inset plan; actual desktop/mobile perceptibility remains deferred.
6. Connections: repeated component product, normalization and fit/transform correspondences named; majorization vs EM inequality direction explained; canonical rank/geometry facts present.
7. Code: compact full updates and real-data programs, complete setup; supplementary text program explicitly unexecuted. Renderer/export contract belongs to phase two.
8. Practice: different numbers, source-row image choice, diagnostic repair and exact independent variations; optional answers and reasoning.
9. Screenshots: not taken; correct phase boundary. Inline figures and informative lab states have specific later capture contracts.

## Files and precise next action

Required packet: `lesson.md`, `visual-specifications.md`, `design.md`, `digits-300.csv`, `data-provenance.md`, `author-calculations.py`, `calculated-inputs.json`. No temporary downloads or app assets created. Root binds frozen files in the central ledger; destination-note disposition is outside those content hashes.

**State at content-phase close, 12 September 2026 — superseded; see the phase-two section below.** The continuation instruction recorded here was: preflight this stable ID with `--work finish`, read the full packet, implement semantic topic-owned visual/lab/model/example files, export only the required small dataset/fixture portions and attribution, fully execute the displayed programs, compare independent arithmetic and actual rendered states, complete independent correctness/learning-experience review and fixes, then browser/accessibility/loading/integration checks. At that date implementation, formal review, browser/visual review, production integration and user acceptance were all pending, and content-phase research and the stated bounded calculations were complete with no unresolved material writing gap handed off.

**Current state, 14 September 2026.** Implementation is complete and is recorded in "Phase two: implementation" below. An independent review has been performed and its disposition is the final section of this file. What remains outside this record: the central phase ledger and the generated inventory, which the integration owner closes; publication; and user acceptance.

## Phase two: implementation — 14 September 2026

Implemented from this packet under a later-continuation request. The manuscript, specifications,
`digits-300.csv`, `data-provenance.md`, `author-calculations.py` and `calculated-inputs.json` are
unchanged; this section is the only addition. No commit, deploy, publication or independent review
is claimed here.

### What was built

The published lesson is the rewritten `src/learn/data/topics/non-negative-matrix-factorization-nmf.jsx`:
ten sections, nine inline figures, three investigations, three executed Python programs and
practices 1 to 8, following the manuscript's declared first-pass route (§§1–6 then exercises 1–5,
with §§7–8 and exercises 6–8 as the separate deeper sitting).

Topic-owned files:

| File | What it holds |
| --- | --- |
| `src/learn/data/nmf-models.js` | Every number the page states: matrix arithmetic, the half squared Frobenius objective, the signed residual, the three entrywise losses and their scaling factors, both multiplicative phases with their per-cell numerators and denominators, the bounded sweep trace, the gradient and the nonnegative first-order conditions, product-preserving normalization, cone coordinates, exact integer rank by Bareiss elimination, crossed-zero support pairs, simplex/anchor positions, contribution masks with per-pixel squared-error change, image grey levels and diverging scales, the bilinear nonconvexity counterexample and sweep cost counts. Every entry point validates and throws `RangeError` rather than substituting. |
| `src/learn/data/nmf-data.js` | Generated, and this is its whole content: the dataset hash, licence, attribution and asset paths; the scale, side and split sizes; the fit settings; the 60 reserved images; the fitted 8 × 64 dictionary and its 60 activation rows; all eight candidate runs; the three reserved baselines; and the library versions. 28 KB; the full 300-row CSV is served for download, not imported. The loss comparison, the support pairs, the altered step and the eight row-242 removals are **not** in this module — they are computed at render time from `nmf-models.js`, which is why `verify-nmf-data.py` checks them against the packet while the page derives them. |
| `src/learn/data/nmf-examples.js` | Generated. The three displayed programs with their executed output. |
| `src/learn/components/lesson-labs/NmfShared.jsx` | Draft/commit investigation state, the prediction contract, number fields on declared lattices, feature strips, 8 × 8 image panels at a true square aspect ratio, scale keys, and a plot frame whose viewBox sits close to its rendered width. |
| `src/learn/components/lesson-labs/NmfFigures.jsx` | F1 to F8 plus the zero-lock static contrast. |
| `src/learn/components/lesson-labs/NmfLabs.jsx` | I1 mixture editor, I2 two-phase update worksheet, I3 held-out image contributions. |
| `src/learn/components/lesson-labs/nmf-labs.css` | Every class prefixed `nm-`. |
| `public/learn-assets/nmf/` | The byte-identical `digits-300.csv` (SHA-256 `d93f963c…728e`) and a served `data-provenance.md`. This topic serves its own copy; the t-SNE lesson's copy is untouched. |
| `src/learn/data/curriculum/blueprints/non-negative-matrix-factorization-nmf.js` | Blueprint. Its entry in `blueprints/index.js` was added by the integration owner, not by this work; as of 14 September 2026 that registration is present in the working tree. |

### Departures from the manuscript, and why

1. **The §8 word program was executed.** The manuscript said it "has not been executed in the
   content phase, so no fitted numbers are presented as observed outputs." The continuation contract
   in this record requires executing the displayed programs, so it was run under the same Python
   3.12.14 / NumPy 2.3.5 / scikit-learn 1.9.1 and its actual output is recorded. The sentence now
   states that it was run in phase two rather than asserting it was not. Its result is pedagogically
   useful and is reported as it came out: the fitted patterns divide the vocabulary exactly, but the
   component order is the reverse of the constructed example in the paragraph above it, which makes
   §4's permutation ambiguity visible immediately. The page says so.
2. **Nine display formulas were re-set** into `\begin{gathered}` lines, and the two six-decimal
   factor matrices additionally use `\small`, so that no `.katex-display` overflows a 280 px column.
   Measured, not guessed: a one-off harness reported each formula's rendered width at 320 px until
   all were inside. No symbol or value changed.
3. **Two displayed programs were re-indented** at their continuation lines so arguments align under
   their opening parenthesis. No token, value or behaviour changed; both still run exactly as shown,
   and the verifier executes the displayed text itself.
4. **§3's three alternation steps** are one paragraph rather than a numbered list, because the
   content components render a numbered list as body prose anyway.
5. **F7's normalized toggle became a three-way mode** — actual pattern values, contribution to this
   image, patterns normalized to sum to 1 — each with its own stated maximum. The specification
   asked for actual patterns, a normalized inspection with labelled divisors, and contribution
   images `W[i,r] × H[r,·]`; three modes over eight panels delivers all three without 24 panels.
   The screenshots forced this (see below).
6. **I3's optional coefficient-scaling slider was not built.** The specification marks it "not
   required if it burdens mobile comprehension"; the lab already carries two recorded predictions
   (total and per-pixel) and 60 image choices, and a third control would have crowded the phone
   layout. Nothing else in the specification depends on it.
7. **Two derived values the manuscript does not state are displayed**: the post-H intermediate loss
   1.602432465 in F3, and the per-cell ratio in I2. The specification requires the post-H state to
   precede the post-W state as a separate state, which needs its own number. Both are computed by
   the model layer and checked by `verify-nmf-models.mjs`.
8. **The Sources block adds a licence claim and link the manuscript omits.** `lesson.md` names the
   UCI dataset without stating its licence; the page says "licensed CC BY 4.0" and links the deed.
   The claim is correct — UCI's record states it verbatim — and the deed asks for the link, so this
   is a deliberate addition rather than an oversight.
9. **The §5 validation table is delivered twice by construction** — once as the executed program's
   own stdout and once as F6's exact table — rather than as a third static copy in prose.

### Checks actually run, with results

| Check | Result |
| --- | --- |
| `node scripts/verify-nmf-models.mjs` | **PASS**, 21 grouped checks over both exact factorizations, all 41 multiplicative states, the altered-input step, the loss comparison and its scaling factors, every candidate fit, the reserved factors, all eight removals from source row 242, the cone coordinates, exact integer rank, all six crossed-zero pairs, the anchor hull, and eleven refusal cases (ragged, negative, non-finite, empty, mismatched, off-lattice, out-of-range, zero-denominator, zero display maximum, Itakura–Saito at zero). Evidence `docs/teaching/evidence/nmf-models.json`. |
| `verify-nmf-examples.py --write`, then again without `--write` | **PASS** both times: 3 programs executed, 25 oracle assertions. The recorded output is therefore proven to match a fresh run. Evidence `docs/teaching/evidence/nmf-native.json`. |
| `verify-nmf-data.py` | **PASS**, 224 checks. The CSV hash, the subset selection against the bundled original collection, the seed-19 split, all eight fits, the dictionary, the activations, the reserved reconstruction, PCA and mean baselines, the 41-state trace, the altered step, the loss table, both exact factorizations, the practice values, the nonnegative-rank matrix and the zero-lock case are all recomputed and compared before the module is written. Evidence `docs/teaching/evidence/nmf-data.json`. |
| `npx vite build --outDir dist-nmf` | **PASS**. The lesson chunk is 157.34 kB raw, 53.06 kB gzipped. |
| `DIST_DIR=dist-nmf … node scripts/verify-nmf-browser.cjs` | **PASS**, 12 cases, 29 screenshots at 1366, 1024, 768, 390 and 320 px plus a 200 % root-text pass. Evidence `docs/teaching/evidence/nmf-browser.json`. |

Reserved-image outcome, published as it came out: training-mean 0.069995, **PCA with 8 components
0.019678**, NMF with 8 components 0.025381. PCA wins, the page says so, and nothing was retuned.

### What the screenshots made me change

Every screenshot was opened and inspected. Fourteen defects survived a fully green verifier run:

1. **F6's magnified panel clamped out-of-range points onto its axis**, drawing a flat segment at
   0.030 for k = 1 and k = 4 — a line at a value no run produced. The polyline now breaks around
   out-of-range points, and the panel says the two runs are neither drawn nor connected.
2. **F6's in-SVG "best seen" label sat on top of the curve** in both panels at both widths. The text
   is gone; the ring remains and the panel note names the point.
3. **F6's caption said "filled markers are training error, open markers are validation error"**,
   which is the opposite of what is drawn. Corrected to solid with faded markers for training and
   dashed with filled markers for validation.
4. **F7 drew the eight dictionary patterns on the image intensity scale** (maximum 1.0177) although
   raw H entries reach 1.3893, so several panels carried over-scale markers and saturated. A raw
   pattern weight is not an image intensity; the gallery now has three modes with three stated
   maxima, and the browser check asserts no panel exceeds its own scale in any mode.
5. **The selectable 8 × 8 image was 0.66 : 1, not square**, because its pixel buttons inherited the
   investigation's 42 px minimum button height. A pixel is a cell of an image, not a control.
6. **The zero-lock figure's matrix was titled `∇_H F`** and rendered as a literal "V_H F". Renamed
   in words.
7. **F4's "feature 2" axis label was crossed by the first dictionary's vertical ray**, and the
   "(3, 3)" point label ran to the edge at 320 px. Axis names moved to HTML beneath the drawing,
   point labels flip to the left of their marker past the midpoint, and the rays are thicker so the
   first dictionary's cone boundary is visible along the axes.
8. **I2's 41-sweep logarithmic trace printed eleven decade labels that overlapped.** Capped at six.
9. **The exact-fit preset drew an empty plot frame**, because every loss is exactly zero. The plot
   is now hidden when no positive loss exists; the note that already named those sweeps remains.
10. **The zero-activation removal announced "a scale to ±1.000 × 10⁻¹⁵"** for a panel in which
    nothing moved. It now says every pixel is unchanged.
11. **`signed(0)` printed "+0"** in the residual table. Zero now prints as 0.
12. **The document scrolled 41 px horizontally at 320 px**, because F1's feature selector sat in a
    two-column grid row. Single column below 640 px.
13. **I1's feedback read "4 against 4 before"**: the baseline was read from the active inputs, which
    the commit had already replaced. The state applied when the prediction was recorded is now kept
    separately, and the browser check pins the exact sentence.
14. **At 390 px I2's fifteen number fields were one per row.** The control grid's minimum column is
    now 150 px, giving two columns on a phone.

### What is still not claimed

- No independent second-agent correctness or learning-experience review has been performed.
- Accessibility was checked programmatically — accessible names, keyboard operation, radio and
  button state, and `role="img"` labels carrying the numbers a sighted reader gets from each
  picture — but no screen reader or assistive-technology session was run.
- Numerical reproducibility is tied to Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1. The page
  says so.
- The digit experiment is one fixed split of one curated balanced subset with no writer identities.
  It is not a recognition benchmark, and no component is claimed to be a physical part.
- Nothing is committed, deployed or published, and the phase ledger is closed by the integration
  owner, not here.

## Disposition of the independent review — 14 September 2026

The [independent review](../../NMF-INDEPENDENT-REVIEW.md) found no numerical disagreement: every
value it recomputed from first principles agrees, much of it in exact rational arithmetic; all three
displayed programs reproduce their published output byte-for-byte; and the served digit CSV is
byte-identical to the packet copy and to the t-SNE lesson's, which this work copied rather than
edited. Its findings are about presentation and record-keeping. Each is listed below with what was
done. Findings assigned to the integration owner — the phase ledger, the generated inventory and the
authoring handoff — are named but not acted on here.

### Blocking

| Finding | Disposition |
| --- | --- |
| **B1** — three records instruct the next agent to implement an already-implemented lesson, and this file's own continuation paragraph still said implementation was pending. | **Fixed, my half.** The paragraph above is now headed "State at content-phase close, 12 September 2026 — superseded", and a "Current state, 14 September 2026" paragraph follows it naming what is done and what is still outside this record. The ledger entry, its stale `design.md` hash, the generated inventory and the handoff entry belong to the integration owner and are theirs to correct; this record does not claim they have been. |
| **B2** — F7's caption said three panels share one intensity scale and then that one of them does not, and said "below" of panels that are above. | **Fixed.** `NmfFigures.jsx`: the caption now reads "The first two panels above, observed and reconstructed, share one intensity scale … The third panel, the signed residual, has its own symmetric scale … The eight component panels carry a third scale, stated with them." `verify-nmf-browser.cjs` now asserts the corrected sentence and asserts the superseded one is absent, since a green pass had not caught it. |

### Should-fix

| Finding | Disposition |
| --- | --- |
| **S2** — F5 draws nothing, though information flow is its whole subject. | **Fixed.** F5 now opens with an inline SVG: three lanes of `X → ÷16 → fit|transform → W`, one shared `H` node drawn once, a single gold arrow from the training lane *into* H, and two green arrows *out of* H into the two transform lanes. The three step lists and the column table are kept beneath it as the text equivalent, and the SVG carries a full accessible description. New assertions count twelve lane boxes, one shared node and twelve arrowheads. Re-captured at desktop and, for the first time, at 320 px. |
| **S3** — I1's explore button stayed live after a graded check, so one click destroyed the verdict and appended a duplicate history row. | **Fixed.** `Prediction`'s explore button now carries `disabled={Boolean(shown)}`, matching its sibling `ContributionLab`. The browser pass asserts both commit actions are inert after grading and captures `nmf-mixture-graded-desktop.png` showing it. |
| **S4** — I2 carried a graded verdict into the next sweep without naming the sweep it described. | **Fixed by naming the sweep**, which the review preferred and which keeps the trace visible: the verdict now reads "At sweep 0, which you have since stepped past, the H phase moved H[1,1] from … to …". The browser pass asserts both the sweep-labelled verdict and the "stepped past" clause once a full sweep has been applied. |
| **S5** — an I3 result computed without a prediction was marked only by an `aria-hidden` glyph. | **Fixed.** The ungraded branch is now prefixed "Calculated without a recorded prediction.", the same wording `Prediction` uses. Asserted, and visible in `nmf-contribution-empty-desktop.png`. |
| **S6** — this record credited `nmf-data.js` with four things it does not contain. | **Fixed.** The row now lists the module's actual keys and says explicitly that the loss comparison, support pairs, altered step and row-242 removals are *not* in it — they are derived at render time from `nmf-models.js`, which is why `verify-nmf-data.py` checks them against the packet while the page computes them. |
| **S7** — F3's caption promised a cross-highlight of the numerator *and denominator* operands, but H₂₁ carried no marking. | **Fixed by completing the marking** rather than narrowing the caption: `H⁽⁰⁾` now marks H₂₁ as a related cell and its note names it as "the denominator's other summand". |
| **S8** — the §8 reconciliation sentence called the *pattern rows* alphabetical when the *vocabulary columns* are, and the most instructive number in the output went unremarked. | **Fixed, both halves.** The sentence now reads "the four *columns* are alphabetical, while the two component *rows* come out in whichever order the fit produced", and names which fitted component is the sports pattern. A new sentence points at the last printed line: the unseen pair `rocket team` reconstructs as `[0.5, 0.5, 0.5, 0.5]`, putting as much mass on `goal` and `orbit` — two words that document never contained — as on the two it did, because an additive dictionary rebuilds a document out of whole components. **`lesson.md` was deliberately not edited**; see "declined" below. |
| **S9** — strip and matrix numbers were fixed at 12 px and did not respond to an enlarged root text size. | **Fixed.** `.nm-cell` and `.nm-matrix-cell` and their two media-query overrides are now `rem`-based (.78rem, and .72rem/.66rem at the narrow breakpoints). The browser pass measures a strip cell and a matrix cell at 100 % and 200 % root text and requires the larger to exceed 1.8× the smaller; the recorded case names both measured pairs. `nmf-build-row-enlarged-text.png` now shows the cell numbers at the same size as the prose beside them. |

### Observations acted on

| Finding | Disposition |
| --- | --- |
| **O1** — a green browser pass missed five findings. | Six assertions added, one per finding it missed: the corrected F7 caption and the absence of the old one; F5's drawn boxes, shared node and arrowheads; both I1 commit buttons disabled after grading; the sweep-labelled I2 verdict; I3's "without a recorded prediction" wording; and the strip/matrix type growing with root text. |
| **O2** — the contribution-versus-coefficient contrast is invisible on the image the lab opens with. | Fixed by naming a case: the transfer task now points at image 2 of 60, source row 256, whose largest raw coefficient is component 7 while its largest contribution total is component 8, and says that 16 of the 60 reserved images disagree that way. All three numbers are now asserted in `verify-nmf-models.mjs`. (The review's own illustrative figures, "5 against 2", describe image **3** of 60, source row 40; image 2's pair is 7 against 8. The page uses the verified numbers.) |
| **O3** — the page adds a licence claim and link the manuscript omits, undeclared. | Added as departure 8 in the departures list above. |
| **O4** — the Gillis pointer sends "geometric algorithms" to §4, which is about connections in mathematics and computer science. | Fixed on the page: "section 3.2 for near-separable geometry, and section 4 for the connections that place nonnegative rank beside problems in mathematics and computer science". Recorded as an implementation correction of a manuscript imprecision. |
| **O5** — F6's comparison bars never say which direction is better. | Fixed: "Bar length is mean squared error relative to the training-mean baseline, so a **shorter bar is the better reconstruction**." |
| **O6** — Lin's title is abbreviated. | Spelled out on the page: "Non-negative Matrix Factorization". |
| **O7** — `hCellDirection` exists for I2's grading job and I2 hand-rolled a different rule. | Fixed: `doUpdateH` now calls `hCellDirection`, so the grade uses the ratio rule the verdict text explains, and the duplicate absolute-change rule is gone. |
| **O8** — dead exports and one unused import. | The unused `componentColours` import and the unused `percent` export are removed, and `limits.coefficientScale` — which backed the slider deliberately not built — is deleted. The remaining model exports are kept: `verify-nmf-models.mjs` genuinely exercises them, and two of them (`integerRank`, `integerDeterminant`) are now called by a component as well. |
| **O9** — §7 asserts the ordinary rank rather than showing it. | Fixed. `integerDeterminant` and `nonsingularMinor` were added to the model, verified against an independent cofactor expansion along the other axis, and F8 now shows the nonsingular 3 × 3 minor with its exact determinant beside the two row sums that cap the rank at 3. |
| **O10** — I1's strips were the only intensity display with no stated scale. | Fixed: a `ScaleKey` beneath them states the current maximum and says it is the larger of 3 and the largest value the mixture draws. Asserted. |
| **O11** — `readTime` invented a complete-read figure. | Fixed: "~50 min core reading · ~60 min code and practice · deeper branches a separate sitting", which is what the manuscript says. |
| **O13** — a provenance claim rests on scikit-learn's docs while sitting under a UCI citation. | Fixed in the served `data-provenance.md`: the sentence now attributes the test-set-portion claim to scikit-learn's documentation and notes that UCI's record reports the combined 5,620 instances. The packet copy is hash-bound and was not edited. |
| **O14** — the signed panels' accessible text reported "the largest value", which is the maximum rather than the largest magnitude. | Fixed: a signed panel now reports the largest value *by magnitude*, so an all-negative residual panel names its most extreme cell rather than its least. |
| **O17** — this record disclaimed a blueprint registration that exists in the tree. | Fixed: the row now says the entry was added by the integration owner and that, as of 14 September, it is present. |
| **O18** — F5 laid out as two lanes plus an orphan. | Resolved by the S2 fix; the lanes are now drawn, and the three text boxes beneath are explicitly a text equivalent rather than the figure. |
| **O19** — F6's ring was positioned by a hard-coded index. | Fixed: it is drawn at `ks.indexOf(best.k)`. |
| **O20** — I1's prompt restates the post-commit baseline above a just-resolved verdict. | Softened: the prompt now says "stay at its currently applied 4", which distinguishes it from the verdict's "against 3 before the edit". |

### Declined, with reasons

- **S8's `lesson.md` half.** The review asks that the manuscript take the same correction. It was not
  edited, for two reasons. The defective clause ("the fitted patterns come out alphabetically") is
  **implementation-added**; the manuscript's own sentence — "The exact vocabulary order is
  `['goal', 'orbit', 'rocket', 'team']`; each normalized nonzero pattern sums to 1" — is correct as
  it stands and survives to the page. And the `rocket team` observation is a new finding from a run
  the content phase explicitly did not perform, so adding it to the frozen manuscript would break
  the ledger's content checkpoint for a file whose other six bindings the review confirms still
  match. It is recorded here and stated on the page instead.
- **O12** — I3 shows the recorded digit unconditionally where the specification made it optional.
  Left as is. The risk the word "optional" guards against is a learner reading the digit as what the
  components encode, and the paragraph directly beneath the selector says the opposite in words:
  "It never entered the factorization and it is not what the components encode." Adding a toggle for
  a label that is already disclaimed would add a control without adding teaching.
- **O15** — pixel buttons fall below the usual touch-target size. Left as is, deliberately. The
  alternative is a non-square image, and an image-shaped display drawn at the wrong aspect ratio was
  itself a defect fixed in phase two. Every pixel is reachable by keyboard with an accessible name,
  and the exact values are available as text in the `ImageValues` table. Recorded against the
  accessibility checklist rather than changed.
- **O16** — neither manuscript nor page said the monotonicity argument does not transfer to the
  other beta divergences. Not declined: **added**, as a clause in §7 — "it is derived here for the
  squared-loss column problem only: the other beta divergences have their own multiplicative updates
  and their own auxiliary functions, so this derivation does not by itself carry over to them."
  The manuscript is unchanged for the same checkpoint reason as S8.
- **B1's ledger, inventory and handoff.** Not mine to close; the integration owner is fixing them.

### Re-verification after these fixes

| Check | Result |
| --- | --- |
| `node scripts/verify-nmf-models.mjs` | **PASS**, 21 grouped checks, now including the exact determinant and minor routines (cross-checked against an independent cofactor expansion along the other axis) and the three transfer-image numbers the I3 task names. |
| `scratch/lesson-tools/Scripts/python.exe scripts/verify-nmf-examples.py` | **PASS**, 3 programs, 25 oracles. Untouched by these fixes; run to confirm no drift. |
| `scratch/lesson-tools/Scripts/python.exe scripts/verify-nmf-data.py` | **PASS**, 224 checks. Untouched; run to confirm no drift. `nmf-data.js` is unchanged. |
| `npx vite build --outDir dist-nmf` | **PASS**. |
| `DIST_DIR=dist-nmf … node scripts/verify-nmf-browser.cjs` | **PASS**, 12 cases, 31 screenshots, including the two new captures the review asked for and a first 320 px capture of F5. |

Screenshots re-opened and confirmed after the fixes: `nmf-figure-fit-transform-desktop.png` and
`-320.png` (the flow is drawn: one gold arrow into H, two green arrows out), the F7 caption read back
from the rendered page (self-consistent and correctly located), `nmf-mixture-graded-desktop.png`
(both commit buttons inert, scale stated), `nmf-update-desktop.png` (the verdict names sweep 0
beneath a sweep-1 question), `nmf-contribution-empty-desktop.png` ("Calculated without a recorded
prediction"), `nmf-figure-support-desktop.png` (the nonsingular minor with determinant 1) and
`nmf-build-row-enlarged-text.png` (strip numbers now the size of the prose beside them).
