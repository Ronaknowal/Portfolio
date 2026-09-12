# PCA & Dimensionality Reduction — independent review

Reviewed 12 September 2026. Topic `pca-dimensionality-reduction`, implemented lesson (phase two). This review is recorded in two separate parts as the teaching standard requires: Part A correctness, Part B learning experience. A final ranked list of actionable findings closes the document.

## Reviewer statement

I did not author the content packet, the implementation, or the author's verifiers. I read the teaching standard sections named in the brief, the design record, the manuscript, the visual specifications, the eight implemented source files, the stylesheet, the CSV asset, the three evidence JSON records and all nine screenshots.

What I **executed** myself, in a temporary directory outside the repository (`%TEMP%\pca-review`, not retained as evidence):

- A Node script that copied `pca-wine-data.js` and `pca-models.js` to `.mjs`, imported them, and dumped the embedded data plus the browser model's outputs (four-point projections at 0°/45°/90°/135°, `principalDirections` on the original, translated, collinear, identical and an asymmetric cloud, `rectangleMetric` at m = 1/2/10 raw and standardized, `labelCollisions` for all five specified configurations plus a = 12 and (2, 1.5), `wineValidationCurve`, `smallestComponentCount` at 0.10/0.06/0.11/0.60/0.01, `wineFullScores(2)`, `wineReconstruction(0, k)` for k = 0/8/13, `storageScalars`).
- A Python script (Python 3.12.14, NumPy 2.3.5, scikit-learn 1.9.1) that recomputed the four-point fixture with exact rationals; recomputed the (10, 1) label collision and rectangle variances with NumPy; regenerated `default_rng(23).normal(size=(40, 20))`; refit Wine from `load_wine` (full standardized, raw, and the 133/45 split) and compared every embedded quantity to it; compared the published CSV to `load_wine` byte-for-value; and executed seven of the eight displayed programs directly against their `expected` strings.
- A textual diff of the eight programs and output blocks in `pca-examples.js` against the eight fenced blocks in the manuscript.
- Pixel sampling and 2× crops of two screenshots (budget curve axis labels; Gaussian figure x-axis label) with Pillow.

What I **reused** without rerunning: the native execution of the `pipeline` program (its recorded stdout in `pca-native.json`; the file hash recorded there, `605e3a70…`, equals the current `pca-examples.js`), the browser structural records in `pca-browser.json`, and the nine screenshots. I did **not** build the site or drive a browser; two interaction findings below are marked as reasoned from the source, not observed.

## Reviewed source versions (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/pca-dimensionality-reduction.jsx` | `48cf5450e39ca103e5722941f662572665648a481ab6f086facdfdc3a23c7380` |
| `src/learn/components/lesson-labs/PcaLabs.jsx` | `50f28997cbb4aa202c9e8d7f6c860ffa35be59e3eed6b91d7d8f3851d1f1f5c3` |
| `src/learn/components/lesson-labs/PcaFigures.jsx` | `bc6f851429dfe2280d8de0080580ae796cd7745ef1e3d8b141da6c5d71a51f59` |
| `src/learn/data/pca-models.js` | `cd541c523da2d9e119122779f27b2f22cb1b9bf16c8f77d41afec03ab5bc39e5` |
| `src/learn/data/pca-wine-data.js` | `d5964935fb9bd4af7e4164fb1275fcdee339b5720267ad6e7cdf7ba85e141e84` |
| `src/learn/data/pca-examples.js` | `605e3a70c8b7f790d4ddf29dd5cf2a392b6af2c10a528eb3ea1e418abf8bd41d` |
| `src/learn/data/curriculum/blueprints/pca-dimensionality-reduction.js` | `229de2e79201b983ab78feffd8c8667d180fea8902107c9d2297cf938ee5ff2d` |
| `src/learn/components/lesson-labs/pca-labs.css` | `9939d51f8e32fbc1e36a6f61399876d9ee1c8a909dfd1f697bf86763eaa93c33` |
| `public/learn-assets/pca/wine.csv` | `cad77e1c82e5f78b6537152a5f17eae62db32dc8b2c862a3bee1ff1d4e74c16f` |

These match the hashes recorded in `pca-models.json`, `pca-native.json` and `pca-browser.json`, so the author's evidence applies to the version reviewed here. A hash proves identity, not correctness.

## Part A — Correctness

### A1. Independent recomputation

All values below were recomputed by me, not read from the author's verifiers.

| Check | Method | Result |
| --- | --- | --- |
| Four-point fixture: mean, scores, reconstructions, residuals, SSE, retained, total, λ₁, λ₂, ratio | Exact rational arithmetic (`fractions.Fraction`) | mean (3, 2); scores ∓3/√2 in pairs; reconstructions (3/2, 1/2)×2, (9/2, 7/2)×2; residuals ±(1/2, −1/2); SSE 2; retained 18; total 20; λ₁ = 6; λ₂ = 2/3; ratio 9/10. Browser `projectAtAngle(fourPoints, 45)` agrees to floating precision. |
| Alternative rulers 0°/90°/135° | NumPy | retained/SSE 10/10, 10/10, 2/18 — the L1 fixture table and F2 bars are correct, and 0° vs 90° is a genuine equal-error case. |
| Covariance matrix, eigenvectors, regression slope | Exact | C = (1/3)[[10, 8],[8, 10]]; C(1,1) = 6(1,1); C(1,−1) = (2/3)(1,−1); slope 8/10 = 0.8 as §10.7 states. |
| Feature-1/PC1 correlation √0.9 (§7, F5 table) | `np.corrcoef` and the displayed formula | 0.948683 both ways; matches √0.9. |
| F5 dot product z_A · a₁ = −2, +3 → 1 | NumPy | −2.0000 (floating). |
| Whitening variances 1 (ddof 1) and 0.75 (ddof 0) | `PCA(n_components=2, whiten=True, svd_solver="full")` | [1, 1] and [0.75, 0.75]. |
| New observation (6, 4) → (5.5, 4.5) | Hand | (3, 2) + 2.5·(1, 1). |
| Wine value chain (brief item ii): standardize row 0 with embedded full-fit mean/scale, project on embedded PC1 | Compared with `StandardScaler` + `PCA` on `load_wine` | Standardized row 0 begins (1.5186125, −0.5622498, 0.2320525) both ways; PC1 score 3.3167508 both ways; browser `wineFullScores(2)` equals `transform(Z)[:, :2]` for all 178 rows. |
| Embedded full-fit mean, scale, 13 components, standardized ratios, raw ratios, raw PC1 | `np.allclose` vs scikit-learn 1.9.1 | All agree. Sign convention of the embedded PC1 matches what scikit-learn prints, so a learner running the program sees the same signs as the §7 table (total phenols 0.3947, flavanoids 0.4229, nonflavanoid phenols −0.2985, proline 0.2868). |
| Split indices, training mean/scale/components/ratios, baseline MSE, 14 loss ratios | `train_test_split(..., test_size=0.25, random_state=42, stratify=target)` then my own reconstruction loop | Indices identical; all quantities agree; baseline 1.0962540507; ratios k = 7…10: 0.1265121, 0.0959546, 0.0723539, 0.0520663; monotone nonincreasing; smallest k for 0.10/0.06/0.11 = 8/10/8. Browser `wineValidationCurve` agrees to 1e-12. |
| "95% training variance selects ten" (§6) | Cumulative training ratios | k = 9 gives 0.9446, k = 10 gives 0.9637 → 10. |
| Full-collection cumulative 55.41% / 92.02% / 96.17% (§6) | Cumulative | 55.4063%, 92.0175%, 96.1697%. |
| Label collision at (10, 1) (brief item iii) | `np.linalg.eigh` on the covariance | Eigenvalues 400/3 and 4/3, fractions 0.990099 = 100/101; PC1 coordinates ±10 collide pairs (A, B) and (C, D) under y-labels; PC2 coordinates ±1 separate them; under x-labels PC1 separates and PC2 collides (A, C), (B, D). Browser `labelCollisions` agrees in all five configurations; eigenvalues unchanged by the label rule. |
| Rectangle metric m = 1/2/10 | NumPy variances | 16/3 vs 4/3 (80%), tie, 16/3 vs 400/3 (96.1538%); standardized always tie. |
| Gaussian spectrum (brief item iv) | Regenerated with seed 23 | Embedded 20 ratios equal regeneration to 1e-12; sum 1.0000; first two 0.2459146. |
| Storage arithmetic (practice 6, §10.5) | Hand | 1220/2000 = 61%; tie at n = 22, first saving at n = 23; 11,100 vs 100,000; 5×10⁹ × 8 B = 40.0 GB. |
| CSV vs `load_wine` (brief item iv) | Parsed all 178 rows | Header is `cultivar` + the loader's 13 feature names; all 178×13 values equal `wine.data` exactly; cultivar column equals `target + 1`; counts 59/71/48. Embedded `wineRows` and `wineCultivar` also equal the loader. |
| Practice 1, 2 solutions | NumPy | Mean (3, 2); scores −2√2, 0, 2√2; variance 8; SSE 0; new point → (6.5, 5.5), error 4.5. Practice 2: A score −1/√2, reconstruction (2.5, 2.5), SSE 18, MSE 2.25, A's error 4.5. |
| Browser 2×2 eigen-solver on an asymmetric cloud | vs `np.linalg.eigh` on (0,0),(1,3),(4,1),(6,5),(2,2) | Eigenvalues 8.0230/1.4770 and angle 35.644° agree. |
| Per-wine readout in L3 ("validation average at this k") | Direct mean over wines of the 13-feature SSE at k = 8 | 1.3674783 = ratio × baseline × 13, as the lab computes; row 35 SSE 0.2672852 and flavanoids 2.98 → 2.9135 agree. |

### A2. Displayed programs versus the manuscript

Seven of the eight code strings are byte-identical to the manuscript's fenced blocks; `library` adds one leading comment (`# Continue the NumPy program above: X, mean and directions are already defined.`), which is a justified clarification given the "continuations" convention the JSX explains in §4. All eight `expected` strings are identical to the manuscript's output blocks. I executed `svd`, `library`, `eigen`, `wineScaling`, `budget`, `originalUnits` and `gaussian` myself with the stated continuation namespaces; every output matched exactly. `pipeline` is reused from the native record (same file hash). No program prints a cautionary sentence; the only comments are shape annotations.

### A3. Manuscript preserved? Prose transcription

I read the JSX against the manuscript section by section. Formulas (`MathBlock` strings), tables, numbers and qualifiers transfer correctly; I found no mathematical or factual error introduced in conversion. Notable differences, each with my assessment:

- §1 links to `data-provenance.md`/`wine.csv` in `docs/` were replaced by the served CSV download and a provenance paragraph in Sources. Justified (the spec forbids links into `docs/`). The Sources paragraph states version and construction but **not a retrieval date**; the standard asks for provenance, version and retrieval date in the lesson. Minor.
- §2 investigation: the manuscript's transfer question "Explain why the mean moves but the centered geometry stays the same" became a statement in the JSX ("The mean moves; the centered geometry, the variances and every loss stay exactly the same") and is stated again in the lab caption. The explanation is correct but the learner's task was converted into an answer. Minor; see B4/B2.
- §4 PCA API link, §10.2 StandardScaler link and §10.6 "consult the API…" sentence moved to Sources or were dropped. Justified consolidation.
- §10.7 "It is not guaranteed to unroll every such dataset correctly" dropped. Justified hedging removal; the preceding sentence already carries the condition.
- The route paragraph was moved ahead of §1 and the time estimate changed to "~50 min first pass"; §10 gained a one-line branch preface. Fine.

### A4. Visual contracts: specified requirements dropped or changed

| Contract | Status | Assessment |
| --- | --- | --- |
| F1 stack coincident labels, right-angle marker, mean at (3, 2), equal aspect | Implemented (screenshot `pca-shadow-desktop.png`) | Correct. |
| L1 editable 4–12 points, ±10 bounds, unset prediction, hidden proposed SSE until compare, stale invalidation, Back/Reset, Fit best direction from active data, translation | Implemented | `Fit best direction` rounds to the integer slider step; the caption gives the exact angle (e.g. 35.6°). Because SSE(θ) is a sinusoid in 2θ, the nearest integer is the minimum over the slider's reachable angles, so nothing misleading results. No action. |
| L1 "exact ties show a family of equally good directions" | Caption text only | Acceptable; L2 draws the family. |
| L2 sliders **plus numeric fields**; display previous and proposed variance fractions | Sliders only (step 0.25); only applied fractions shown | The dropped numeric field matters: the tie multiplier m = a/b is reachable only when the ratio is a multiple of 0.25, yet §5 prose tells the learner to "edit the rectangle's width or height and predict the multiplier at which the two axes tie". For a = 2, b = 0.75 the tie is at 2.667 and cannot be set. Minor. |
| L2 tie shown as a tie; standardized mode is its own geometry | Implemented (four dashed directions; standardized plot in fitted SDs) | Correct; tie detection is exact at grid ties (verified m = 2 for a = 2, b = 1). |
| L3 exact split, unset integer prediction, k = 0 and 13 on the curve, budget line, preceding failing k, per-record residual table, cumulative training variance kept separate, "more digits near the boundary" | Implemented except the extra digits | On the 0.01 slider grid no budget lies within 0.0005 of a ratio, so no false crossing can occur; typed budgets compare at full precision. No action. |
| F4 raw/standardized bars on 0–100%, score scatter with hidden-by-default cultivar shapes, 13 signed coefficients **and a text table** | Bars carry their numeric values and an aria-label lists all 13 | Acceptable substitute; the spec's "text table" intent (exact inspection) is met by the visible values. |
| F5 single observation + arrow, stated ×2 multiplier, dot-product readout in original coefficients, no angle-as-correlation claim | Implemented | Correct. |
| L4 a ∈ [2, 12], b ∈ [0.25, 1.5], label rule, kept components, stacked labels, retained fraction, inset with stated magnification | Implemented | Correct; inset label "×3" is computed from the actual scale ratio. |
| F6 0-based axis, 0.05 reference, no elbow annotation | Implemented | Correct; x-axis label clipped on phones (B5). |
| F7 fixed direction, P and Q, zero baseline, no rejection zone | Implemented | Correct (score 3√2 = 4.2426, residual² 18). |

### A5. Correctness of the interaction code (read, not driven)

- `Prediction` disables the check button once revealed and until inputs change; the L1/L3 `stale` flags key on a serialized state, so a changed angle or budget cannot be graded against an old answer. Correct.
- L1 `editPoint` and the translation fields: `Number(event.target.value)` with a controlled `<input type="number">`. In Chromium a lone typed `-` is reported as `""`; `Number("")` is `0`, which passes the finiteness/bounds test and **sets the coordinate to 0**; React then writes `"0"` back into the DOM, discarding the minus. Typing a negative coordinate (needed for the specified collinear fixture and for any negative translation other than the default) therefore appears impossible from the keyboard except by arrow keys or paste. The author's browser run only filled positive values (`fill('3')`). I did not drive a browser; this is reasoned from the source. Minor, but it is an unintended data mutation.
- `wineScaling.question` begins "Before running: will the raw…", and the `Program` wrapper prefixes "**Before running:**", so the page reads "Before running: Before running: will the raw first component…". Minor.

## Part B — Learning experience (nine-item checklist)

Author's heuristic assessment only; no beginner walkthrough was available to me either.

| # | Item | Finding (with evidence) |
| --- | --- | --- |
| 1 | **Route** | Yes. `Prose.pca-route` before §1: "Read sections 1 through 6, run the four labs and the first four programs…, then try practice 1 to 4… Section 7 develops interpretation, section 8 connects…, section 10 holds the deeper mathematics". §10 opens with "Each branch below answers a question that becomes relevant after the core route." Branch starts are labeled. |
| 2 | **Cautions** | Largely yes. The variance-versus-meaning caution has one home, the §3 `Callout` ("Every percentage below should be read with this distinction; we will not repeat it after each one"); leakage has its home in §6 ("This is the lesson's home for data leakage"); scaling choice in §5. Hedging density is low (about ten hedging phrases in 152 `Prose` paragraphs). No program prints a caution. **But** several figure/lab captions restate the adjacent prose almost verbatim: F7's last sentence and the following paragraph both say an alarm needs data on normal variation, faults and false-alarm costs; the F4 figure text and the next paragraph both say the labels were not given to PCA and ask whether the summary aligns with a grouping; the L3 caption and the following "Why not just minimize…" paragraph both make the monotonicity argument; the L1 translation result is stated in the prose after the lab and in the lab caption. Minor consolidation. |
| 3 | **Real question** | Yes. §1 opens with 178 wines × 13 measurements and "Which two numbers should represent each wine?", promises two answers, and the lesson returns to them: F4 ("This picture retains 55.41% of the standardized variance") and §6/L3 ("eight is the smallest count satisfying this particular budget"). Real data present and verified equal to `load_wine`; served as a CSV. The return is implicit; §11's summary does not name the wines. Optional: one sentence in §6 or §11 tying "8 coordinates instead of 13" back to the opening question. |
| 4 | **Labs as investigations** | **L1**: prediction recorded (`select` less/same/more + "Compare prediction"), compared against the model's verdict with both SSEs; entities editable (coordinates, add/remove, translation, arbitrary angles); fixture shows the promised contrast (10 vs 2) and the null (0° vs 90° equal). The default comparison is partly pre-resolved by §2 prose, but the learner is then asked to edit a point. **L2**: prediction recorded (first/second/tie) and compared after "Apply and compare"; a, b, m and mode are real inputs; contrasts m = 1 → first (80%), m = 2 → tie, m = 10 → second (96.15%), standardized → tie at every m all verified; null case (standardized, no contrast) present. Gap: tie multiplier unreachable for many a/b (A4). **L3**: prediction recorded (0–13 select) against the exact crossing with bracketing ratios; budget is user-set; observation and feature selectable; null (0.10 and 0.11 both → 8) stated and verified. **L4**: prediction recorded (distinct/collide) against grouping of actual retained coordinates; a, b, label rule and kept components are real controls; all four contrasts and the null (eigenvalues unchanged under label switch) verified. All four labs satisfy the three review requirements. |
| 5 | **Figures** | F1, F2, F3, F4, F5, F7: the claimed features are visible at the screenshot sizes (F1 right angle and shared score; F2 equal-length bars; F4 the 99.81% sliver versus 36.2%; L4 ±1 beside ±10 is visible at 325 px and the inset states ×3). Baselines present (F2 0°; L3 k = 0 at ratio 1; F6 0.05 line). No caption apologizes. **Two defects**: (a) the L3 validation-error curve's SVG text — k labels 0–13, the y ticks 0/0.5/1 and the axis caption — renders with the SVG default black fill; pixel sampling of `pca-budget-desktop.png` gives text luminance ≈ 8 on background ≈ 14 (the gold "budget 0.1" label, which has an explicit fill, is at ≈ 100). The k axis is the figure's message; this is material. (b) `pca-gaussian-mobile.png`: the x-axis label is clipped to "…fraction of sample varian" at 350 px. Minor. |
| 6 | **Connections** | Yes. Variance/loss identity (§3 + F2), covariance/eigen/SVD (§10.1 with the `eigen` program comparing reconstructions), Z = AVₖ = UₖSₖ, right-triangle accounting reused for pair distances (§8), F3 relating symbolic Vₖ columns to code rows, K-Means link, canonical survey facts developed or routed per the design record. |
| 7 | **Code** | Yes. Each program is mechanism-dominant: `svd` has no validation lines at all; `budget` is a 14-iteration reconstruction loop; the browser model's bounds checks live in `pca-models.js` helpers, not on the page. No printed disclaimers. |
| 8 | **Practice** | Yes. Numbers and context change (new three-point cloud plus an out-of-span new observation; second direction retained; 6% budget; storage audit; leak diagnosis; U/W dependence; changed seed). Exact check values are supplied and verified (0.072354/0.052066 → 10; 4.5; 2.25; 1220/61%/n = 23). The L3 lab reproduces the 6% values, as practice 4 says. |
| 9 | **Screenshots** | Partly. Informative states captured: L1 after comparison, L3 after reveal with the crossing, F4 with cultivars overlaid, F1/F2/F3/F6. **Not captured**: L2 in an applied state (the tie with four dashed directions, or the m = 10 flip) and L4 after a checked prediction (collision feedback) — both phone captures are default states; F5 and F7 have no capture at all. |

## Findings, ranked

### Material

1. **L3 budget-curve axis labels are effectively invisible.** `src/learn/components/lesson-labs/pca-labs.css`, rule `.pca-curve svg text { font-size: 9px; }` sets no `fill`, and unlike `.pca-plot text` and `.pca-figure svg text` nothing else supplies one, so the k labels, y ticks and axis caption render black on `#0b0f10` (measured luminance 8 vs 14 in `pca-budget-desktop.png`). A learner cannot read which k the crossing point is without the feedback text. **Fix:** add `fill: #bac6bf;` to that rule (one line), then recapture `pca-budget-desktop.png`.

### Minor

2. **Duplicated "Before running:" prefix.** `src/learn/data/pca-examples.js`, `wineScaling.question` starts with "Before running:", and `Program` in the lesson JSX prefixes the same words. **Fix:** drop the prefix from the question string ("Will the raw first component…"); rerecord the examples file hash in `pca-native.json` (code strings and outputs are unchanged).
3. **Typed negative coordinates in L1 probably cannot be entered and silently zero the point.** `PcaLabs.jsx`, `editPoint` (line 138) and the translation `onChange` handlers (line 184): `Number("")` is `0`. **Fix:** `if (raw === '') return;` at the top of `editPoint` and the same guard on the two shift inputs; then confirm in the browser by typing `-2` into a coordinate field and check the collinear fixture (−2, −2), (0, 0), (2, 2) plus one point.
4. **L2 tie multiplier unreachable for many rectangles; numeric fields specified but absent.** `PcaLabs.jsx` `PcaMetricLab` (sliders step 0.25) and lesson JSX §5 line 129 ("predict the multiplier at which the two axes tie: it is the ratio of width to height"). **Least-intrusive fix:** change the prose to "choose a width and height whose ratio is a quarter-step, then predict…", or add the specified typed field for the multiplier.
5. **Adjacent restatements in captions.** `PcaFigures.jsx` F7 closing sentence vs lesson JSX line 286; F4 figure text vs JSX line 136; `PcaLabs.jsx` L3 caption ("That is why a budget is needed…") vs JSX line 155; L1 caption's translation sentence vs JSX line 81. **Fix:** keep the explanation in prose and shorten each caption to what the picture shows; for L1 restore the manuscript's question form ("Explain why the mean moves but the centered geometry stays the same").
6. **Screenshots miss the L2 and L4 informative states and F5/F7.** `scripts/verify-pca-browser.cjs` lines 183–208. **Fix (verifier only):** capture `metric` after applying m = 2 (tie) or m = 10, `labels` after "Check", and `.pca-figure` indices 4 and 6; look at them.
7. **F6 x-axis label clipped on phones.** `PcaFigures.jsx` `GaussianSpectrumFigure`, the `<text x="170" y="220">` label. **Fix:** split into two shorter texts ("ordered sample component" under the axis; "fraction of sample variance" rotated or as the y-axis caption) or reduce to "component rank · variance fraction".
8. **Retrieval date absent from the lesson's data provenance.** Lesson JSX Sources, line 334. **Fix:** append "retrieved 12 September 2026 from the scikit-learn 1.9.1 bundle" (the date recorded in the design record).

### None (checked and found correct)

Four-point fixture and all derived numbers; 0°/90° equal-error null; Wine full-fit and split quantities, sign convention, indices and baseline; browser reconstruction of the curve and per-record residuals; label-collision contrasts and null; rectangle contrasts and standardized null; Gaussian spectrum regeneration and normalization; CSV equal to `load_wine`; all eight displayed outputs; practice solution values; storage and 40 GB arithmetic; whitening ddof claims; √0.9 correlation; F5 dot product; F7 score/residual; cumulative-variance claims; formula transcription in every `MathBlock`; no cautionary output from code; F4 scatter range fits its viewBox (PC1 ±4.31, PC2 −3.87…3.52 within the drawn ±4.6 / −3.9…4.3 extents).

## Closing note

The implementation preserves the manuscript's explanations and the specified visual contracts with only the deviations listed; the numbers are right everywhere I recomputed them, and each of the four labs is a genuine investigation. The one material defect is a rendering problem, not a content problem, and is a one-line CSS fix followed by a recapture. Items 2–8 are small; none degrades the teaching surface to fix.
