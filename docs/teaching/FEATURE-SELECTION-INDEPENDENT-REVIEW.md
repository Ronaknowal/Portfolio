# Feature Selection & Importance (SHAP, Permutation, Mutual Info) — independent phase-two review

Reviewed 15 September 2026 against the working tree plus the uncommitted feature-selection implementation. The reviewer
authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or
`docs/teaching/drafts/` was edited — every one of those retains the mtime and hash it had before this review began.
Throwaway scripts live under `scratch/fsel-review/`. No state-changing git command was run.

Two writes beyond this document, both disclosed rather than assumed harmless: the production build was regenerated into
`dist-fsel/`, and **re-running the four verifiers rewrote their own evidence records** —
`docs/teaching/evidence/selection-{models,native,data,browser}.json` and the 35 screenshots under
`evidence/screenshots/`. That is what those scripts do on every run; all four passed with counts identical to the
builder's, against source hashes identical to the table below, so the regenerated records assert the same results about
the same bytes. If the builder wants the originals back, re-running the four verifiers reproduces them.

## Reviewer statement: executed, read, looked at, delegated

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/fsel-review/`, importing **nothing** from `selection-models.js`, `selection-data.js`, `selection-examples.js`, any `verify-selection-*` script or `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.1, shap 0.52.0, mpmath 1.3.0). **3,194 independent constructed checks, all passing.** The whole Wine study was refitted from the served `wine.data` under the manuscript's declared protocol and diffed value by value against the published module (214 checks). The model layer was driven through a Node harness and its outputs checked against reviewer oracles written in exact rational arithmetic and 50-digit `mpmath` (2,852 checks), including 300 pseudo-random contingency tables checked for both MI routes, nonnegativity, the H(Y) bound and transpose symmetry; Shapley values re-derived by explicit enumeration over all d! arrival orders in `fractions.Fraction`; every permutation increase re-derived in exact `Fraction` arithmetic; and the saved tree's JS inference compared against a fresh native `DecisionTreeClassifier` on all 38 inspection rows, all 100 background rows and 12 hand-built threshold-boundary probes. The tree was then checked **node for node** (9 nodes × children, feature, threshold, samples, impurity and the 3-class value array) against both the native fit and a reviewer recomputation of Gini from the raw fitting rows, together with the weighted impurity decreases, the confusion matrix and the observed column ranges (128 checks). Both float32 boundary claims were verified in `Decimal` and against the JS. `shap.TreeExplainer` was re-run independently against a reviewer-written exhaustive 16-coalition oracle for **all twelve** explained rows. The four verifiers were re-run. The production build and preview were rebuilt and the page driven with Playwright/Edge at 1366, 1024, 768, 700, 683, 640, 620, 600, 390 and 320 px, including a pass with every investigation revealed and a pass at 200 % text size. |
| **Read in full** | `lesson.md` (651 lines), `visual-specifications.md`, `data-provenance.md`, `data-source.json`, `design.md` including both the Phase A and Phase C appends, the published `.jsx` body (391 lines), `SelectionShared.jsx`, `SelectionLabs.jsx`, `SelectionFigures.jsx`, `selection-labs.css`, `selection-models.js`, `selection-data.js`, `selection-examples.js`, the blueprint and its registration, `ATTRIBUTION.txt`, the pointer-contract and example-module blocks of `verify-selection-models.mjs`, the extraction mechanism of `verify-selection-examples.py`, sections 8 and 10 of `verify-selection-browser.cjs`, `scripts/lib/lesson-visual-layout.cjs`, and the investigation contract and learning-experience checklist of `LESSON-TEACHING-STANDARD.md`. |
| **Looked at** | Reviewer-captured screenshots at 1366, 390 and 320 px: all five `.fs-figure` blocks at desktop, Figure 1 at 320 magnified, the k = 6 fold mask matrix, the permutation/impurity two-axis panel, both I5 waterfall states, the saved-tree diagram at 320 and 390, and each investigation in its post-Apply state. Of the builder's own evidence, `selection-investigation-5-390.png` was opened and is the basis of **S3**. Findings **B1**, **S1**, **S2**, **O2** come from this pass. |
| **Delegated** | Every external URL was fetched and checked against the sentence citing it, with all six PDFs downloaded and text-extracted locally (`pdftotext -layout`) rather than trusted to a summariser, and every claimed section number grepped in the extracted text. The UCI record was checked through its raw API payload and `wine.zip` was re-downloaded and hashed. Separately, a full coverage analysis of the four verifiers was commissioned and its findings independently spot-checked against the source before use (**S3**, **S4**, **S5**). |
| **Not done** | No beginner walkthrough. No screen-reader pass; accessible names were read in the DOM but not heard. No colour-contrast or axe audit. No test of library versions other than the pinned ones. Fourteen of the builder's 35 evidence screenshots were not opened (the reviewer captured its own instead). The blueprint's curriculum-inventory integration was not exercised beyond confirming registration. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/feature-selection-importance-shap-permutation-mutual-info.jsx` | `1a47c2b44cbc0a88559fab29e01dcde83bd4c48ae432ab3249fd718dad1be26d` |
| `src/learn/data/selection-models.js` | `c8ed7bca0e19756f3ccd54dc238daba42570b4537df37527c8090e5784bc2526` |
| `src/learn/data/selection-data.js` | `6bae5793a66a6531ba5470607b1bf554837430ddae86dcce6b2cb4dd68e45937` |
| `src/learn/data/selection-examples.js` | `9dd716db1d8972318d41dd1f1acfe6e501f9765dd62b63d31d884f076612a76e` |
| `src/learn/components/lesson-labs/SelectionShared.jsx` | `1513337b9f12ff37a64114d8f856ba437b1e54f86cdbc79f90ae96cafc13fa03` |
| `src/learn/components/lesson-labs/SelectionLabs.jsx` | `79d09f32dac6fb129b315bffe4a7899758025535be31f9b315c2d1ef372f4ffd` |
| `src/learn/components/lesson-labs/SelectionFigures.jsx` | `006db17b234460dc7a14b03989c5d2cb02978550fb48ee309dae0686ae9db0db` |
| `src/learn/components/lesson-labs/selection-labs.css` | `2bb72e2db43fbd86f0f78c367d4c09b56ba14819b628b9885ff9839553f415b9` |
| `src/learn/data/curriculum/blueprints/feature-selection-…js` | `7a2af3ecb83aca4cb4a3d266bc6dbb2d77f95307c162a4ef2fe122b51aaded1c` |
| `public/learn-assets/feature-selection/wine.data` | `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659` |
| `public/learn-assets/feature-selection/ATTRIBUTION.txt` | `bd795e57af39e89adbbf90542330fa8efff5ccbe062d5928e0fee881ddfd635d` |
| `docs/teaching/drafts/…/lesson.md` | `2e4125594caf47ff5f108fa69beff804b67f55195503958d4ba0041fb017b035` |
| `docs/teaching/drafts/…/visual-specifications.md` | `499da63eaf6c2b1d9e3f3ef668a3bd679938caf19528ae1fb0a71d8c49bdf19e` |
| `docs/teaching/drafts/…/calculated-inputs.json` | `f13f6d7a7897872e371f417154465e053a49e903047d161abf68ff205fe39109` |
| `docs/teaching/drafts/…/data-provenance.md` | `18f3d3698e5929b1a0d39e9f52b591a85fe162233d8ca3b5e5967dab5d5193c9` |
| `docs/teaching/drafts/…/design.md` | `830f48fb5d60db3a1165c909112ae801bf873dc77595dbf4920cfacadc5e942c` |
| `docs/teaching/drafts/…/wine.data` | `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659` |
| `scripts/verify-selection-models.mjs` | `57f85e68d49e0e1219faa259cc688b3dc78014898ed337bdd5c5818c4e5e23a3` |
| `scripts/verify-selection-data.py` | `dd3e6bbdb92e9612935beb1d62a7a6460d00860d01a55c480e523b34c3fa7668` |
| `scripts/verify-selection-examples.py` | `9d26819e938f4f640176fd5acc65534e76d2e86dee83342b0342262714b2c592` |
| `scripts/verify-selection-browser.cjs` | `4a2eff4f97bdc439e17f3b22bccce70fe09e9121c3bb9b5efb377868453c1c5f` |

The served `public/learn-assets/feature-selection/wine.data` is **byte-identical** to the packet's copy (both 10,782
bytes, same SHA-256, LF line endings, 178 rows, no trailing blank). A hash proves identity, not correctness.

---

# Part A — what I recomputed, and where I found no disagreement

This is the substantive half of the review. Recording where an independent method **agreed** is a finding, because it
bounds what the defects below can be.

## A1. The Wine study, refitted from the served bytes — no disagreement

I reimplemented the declared protocol from the manuscript alone (`scratch/fsel-review/recompute_wine.py`): the 138/40
stratified split at seed 51, the 100/38 split at seed 52, three stratified folds at seed 53, `SelectKBest` over
`mutual_info_classif(discrete_features=False, n_neighbors=3, random_state=54)`, a depth-3 / min-leaf-5 / seed-55 tree,
twenty donor permutations at seed 56 + r. I then dumped the published module with `node` and diffed it value by value
(`diff_data.py`, `diff_data2.py`). **214 checks, zero mismatches.** Specifically reproduced exactly:

- All four row-ID sets, in order, and their disjointness and union (`developmentIds ∪ reservedIds = 0…177`,
  `fittingIds ∪ inspectionIds = developmentIds`).
- All three fold memberships, fit and validation IDs, in order.
- All nine candidate fits: every retained mask, all thirteen MI estimates in nats per fold, every correct count and
  fold total — **30/34, 25/33, 26/33; 30/34, 29/33, 26/33; 30/34, 26/33, 28/33** — and the three mean accuracies at
  full double precision (0.8092691622103386, 0.8496732026143791, 0.8395721925133689). The rule selects **six**.
- The selected refit's columns `[0, 5, 6, 9, 11, 12]` = alcohol, total phenols, flavanoids, color intensity,
  OD280/OD315, proline; its 36/38; the four-field model's 36/38; the majority baseline class 2 at 15/38; the full
  38-row prediction vector and class vector.
- MDI `[0.402897330915…, 0, 0.460170935778…, 0.136931733307…]`, and the raw weighted decreases
  `0.255517, 0, 0.291840, 0.086842` that the figure prints.
- All eighty permutation drops, and the four mean/SD pairs — **0.221053/0.048809, 0/0, 0.359211/0.082076,
  0.189474/0.029539** — matching the manuscript's table.
- Source row 104 = (12.51, 1.73, 1.92, 672); baseline 0.33; all sixteen coalition values; attributions
  **(−0.45, 0, +0.07, +0.05)**. The manuscript's individual claims hold: the alcohol-only coalition is 0, flavanoids-only
  0.42, proline-only 0.38, flavanoids+proline 0.62, and **every** coalition retaining this instance's alcohol is 0.
- All twelve explained cases; the alcohol 12.51 → 13.5 contrast with attributions (+0.216667, 0, +0.206667, +0.246667)
  and output 1; the malic acid 1.73 → 4.1 null, where I confirmed all sixteen coalition values are **bit-identical** to
  the unedited case; and the class-3 cohort contrast (baseline 0, attributions −0.416667, 0, +0.375, +0.041667) whose
  twelve reference rows I confirmed are all class 3.

The manuscript's "floating-point reconstruction error about 1.1×10⁻¹⁶" is exactly right: `baseline + phi.sum() - v(F)`
with NumPy's pairwise summation is `−1.1102230246251565e−16`, which is also the `efficiencyError` the module records.

## A2. Shapley values by exhaustive enumeration — no disagreement

For every game in the lesson I enumerated all d! arrival orders explicitly and summed the marginal increments in exact
`fractions.Fraction`, never using the |S|!(d−|S|−1)!/d! weight formula the implementation uses. The two routes agree at
double precision everywhere. Verified against the manuscript's own literals: the zero-reference game [0, 2, 3, 11] →
(5, 6); the two-row background game [1.5, 3.5, 5, 11] → (4, 5.5) summing to 9.5 above baseline 1.5; f(0.5, 0.5) = 1.25 ≠
the mean output 1.5; practice 4's [0, 1, 2, 7] → (3, 4); the conditional/replacement duplicate pair → (1/4, 1/4) versus
(1/2, 0); three-player unanimity 1/3 each with A+B = 2/3 against the grouped 1/2; four-player 1/4 each with A+B+C = 3/4
against the grouped 1/2. Efficiency (v(∅) + Σφ = v(F)) holds in every game to 1e-12.

## A3. Mutual information in 50-digit precision — no disagreement

`informationFromCounts` was run on 300 pseudo-random contingency tables (2–4 rows × 2–4 columns, counts 0–11) and
checked against an `mpmath` oracle at 50 decimal digits built from exact `Fraction` probabilities. For every table, both
published routes (entropy difference and the direct Σ p log p/pq form) matched the oracle to 1e-12, MI was nonnegative,
MI ≤ H(Y), and MI was invariant under transposing the table. The named tables match the manuscript exactly:
3/1/1/3 → H(Y) = 1, H(Y|X) = 0.811278…, I = 0.188722…; 4/0/0/4 → 1 bit; 2/2/2/2 → exactly 0; practice 1's 2/0/0/6 →
0.811278…; and tripling every count leaves MI bit-identical. XOR: I(A;Y) = I(B;Y) = 0 and I((A,B);Y) = 1.

## A4. The subset lattice, permutation table and search costs — no disagreement

I rebuilt the lookup predictor independently (group states by retained coordinates, majority with ties to 0). The XOR
world gives 1/2, 1/2, 1/2, 1 exactly as the manuscript says; strict forward improvement never leaves the empty set;
forcing two additions reaches the pair at accuracy 1; relabelling to Y = A lets the strict rule find A. The
duplicate-sensor table reproduces exactly in `Fraction` arithmetic — first-sensor model (4, 0, 4), second-sensor model
(0, 4, 4), average model (1, 1, 4) — as does practice 3 (0.5 and 2). Fit counts: forward 5 → 2 gives 9 subsets and 27
fits with three folds; simple RFE 5 → 2 gives 4; practice 8's forward 6 → 3 over four folds gives 15 subsets, 60
candidate fits and 61 total, and RFE gives 4. sigmoid(0.5) = 0.622459… and 1 − 0.95¹⁰⁰ = 0.9941… both check out.

## A5. The float32 boundary claims — both verified, both correct as stated

Verified in `Decimal` and against `Math.fround` (`float32_probe.py`):

- **Root, node 0.** `12.78000020980835` is exactly `12.780000209808349609375`, and it is **not** a float32. Casting it
  to float32 gives `12.780000686645508`, which is strictly greater, so `cast <= threshold` is false and the row takes
  the **right** branch. I confirmed the stronger form of the claim too: this double is *exactly* the arithmetic midpoint
  between the float32 below it and `np.float32(12.78000020980835)`, so the upward cast is round-half-to-even, not
  accident. A comparison of unconverted doubles would take the left branch. The builder's statement is precisely right.
- **Node 1.** `1.0049999952316284` **is** a float32 (`1.00499999523162841796875`). The next double above it,
  `1.0049999952316286`, casts back down onto it, so the float32 route takes the **left** child while a raw double
  comparison takes the right. The band of doubles that round back onto it is `(t, 1.005000054836273193359375]`, width
  5.96×10⁻⁸. So `Math.fround` is load-bearing, exactly as claimed.

The implementation honours this: `treeDecision` (`selection-models.js` L691–711) computes `const cast = toFloat32(raw)`
and compares `cast <= tree.threshold[node]` against the stored **double**, which is what scikit-learn's Cython predictor
does. I confirmed behavioural agreement on all 38 inspection rows, all 100 background rows and 12 boundary probes
against `model.predict_proba` and `model.apply` — leaf index and class-1 probability identical in every case, at
tolerance 0.

Incidentally, node 4's `1.5899999737739563` is *also* a non-float32 midpoint. The record names only two facts, which is
fine — but see **B1**, because node 4 is where the drawing goes wrong.

## A6. The executed SHAP comparison — independently reproduced, and the mode is genuinely what it claims

I built the model and my own exhaustive oracle from scratch (`check_shap.py`), importing nothing from
`coalition_attribution.py` or the lesson modules. Results:

- The explainer really is configured as claimed: `feature_perturbation == 'interventional'`,
  `model_output == 'probability'` on both the explainer and its underlying model object, and `data` is the 100 × 4
  background.
- **The oracle is a true exhaustive enumeration, not a restatement of TreeSHAP.** Mine evaluates all 2⁴ coalitions by
  materialising the 100 hybrid rows per coalition and averaging `predict_proba`, then averages marginal increments over
  all 4! explicit orderings. It shares no code path with the tree algorithm.
- Agreement over **all twelve** explained rows (the displayed program checks only row 104 against the oracle): worst
  |Δφ| = **3.28×10⁻⁹**, worst |Δbaseline| = **0.0**. Reconstruction of each row's own predicted probability: worst
  **9.83×10⁻⁹**. Both comfortably inside the claimed 1e-6.
- I tested that "interventional" is not a silent no-op by also running `tree_path_dependent`, which gives
  (−0.469737, 0, +0.109474, +0.030263) for row 104 against the interventional (−0.45, 0, +0.07, +0.05) — a 0.0395
  difference. If those had matched, the mode claim would have been unproven; they do not.
- The baseline equals the mean `predict_proba` over the background exactly (0.33), which is the interventional identity.

The lesson's §6 sentence about this program is honest in a way worth naming: it says explicitly that "the printed lines
above come from the imported study, which this program re-runs; the program's own result is that its three assertions
pass silently and a waterfall is drawn." No library stdout is fabricated or passed off as the program's own.

## A7. The verbatim-program mechanism — it holds, end to end

This one deserved scrutiny because drift would be invisible. I re-extracted the ` ```python ` fences from the frozen
`lesson.md` myself, hashed them, and compared three ways (`check_verbatim.py`):

| Fence | Lines | SHA-256 | Matches the verifier's pin | Byte-identical to the displayed `code` |
| --- | ---: | --- | --- | --- |
| `information_from_counts.py` | 20 | `d1d399c9…6627b` | yes | yes |
| `coalition_attribution.py` | 43 | `2cab1585…5919e` | yes | yes |
| `wine_feature_study.py` | 87 | `3cf2ae8b…386cb` | yes | yes |
| `check_tree_explanation.py` | 24 | `aaaa8fcc…fe25f` | yes | yes |

`extract_programs()` asserts the fence count equals four and each SHA equals a hardcoded pin before executing; the
module the lesson renders is generated from the same extracted text. Manuscript → pin → executed file → displayed code
is one chain of identical bytes. **Verified; no transcription layer exists.**

## A8. Provenance and served data — accurate in every particular I could check

- Served file byte-identical to the packet's (SHA `6be6b120…a659`, 10,782 bytes). 178 rows, 14 comma-separated fields
  each, no missing values, class counts 59/71/48, LF endings, no header, source order preserved — all matching
  `provenance` and `data-provenance.md`.
- I re-downloaded `https://archive.ics.uci.edu/static/public/109/wine.zip` and its SHA-256 is
  `2bae62c4481220623579d4c4fb36b55652b6b75e06e49fa1981b8198362dfdab`, **exactly** the `archiveSha256` the packet
  records, and the `wine.data` member inside it hashes to the served bytes. The "unchanged archive member" claim is
  literally true.
- The UCI API (`/api/dataset?id=109`) confirms 178 instances, 13 features, three cultivars from the same region of
  Italy, creators Aeberhard and Forina, 1992, DOI `10.24432/C5PC7J`, no missing values, and the CC BY 4.0 licence. The
  DOI resolves. `ATTRIBUTION.txt` states source, DOI, archive URL, licence, byte count, hash, layout and the thirteen
  field names, and correctly says the split and analysis are the lesson's own work.
- The pipeline matches the provenance description: comma delimiter, 178 rows, `raw[:, 1:]` features and `raw[:, 0]`
  target, source order retained, the split rule as documented, exactly eleven fits.
- **This lesson never reaches into another lesson's asset directory.** Every reference in the lesson, the components,
  the data module and the verifiers resolves to `learn-assets/feature-selection/`, and the browser verifier asserts the
  rendered page issues no `wine.data` request outside that path — which my own run confirmed (zero such requests at all
  three widths).

## A9. External links — all fifteen live, all claims supported

Every URL returns 200 and supports the sentence citing it. Section numbers were checked by grepping locally extracted
PDF text, not by trusting a summariser. Janzing et al. **section 3** really is "How should we sample the dropped
features?" and really does contain the `f(x₁,x₂) = x₁` duplicate-input example with φ₂ = 1/4 conditional and φ₂ = 0
marginal, plus the verbatim program-versus-world caution. arXiv 1802.03888 **section 3.2** really states the O(TLD²)
bound and the shared-path recursion, with section 4 on interaction values. Fisher et al. sections 2–4 match. Lundberg
and Lee sections 2–4 match, including the "for a given simplified input mapping h_x" qualifier the lesson leans on. The
scikit-learn `mutual_info_classif` page confirms all four sub-claims verbatim, including **nat units** and the
"replaced by zero" clipping. Two loose citations are noted as **O4**.

## A10. The investigation contract — honoured, verified by operating the page

Driven in Edge against the production build (`drive-labs.cjs`). For **each** of I1–I5, before any interaction:
0 verdicts, 0 revealed stages, 0 tables, 0 SVGs inside the investigation, 0 pre-checked radios, and the Apply button
**disabled**. Selecting a radio reveals nothing. After Apply: the verdict appears, the stage content mounts, and the
radios lock. Editing any input retires the verdict **and** the stage content **and** the recorded choice, and raises the
"Inputs changed; record a new prediction" notice. Loading any preset leaves the outcome radios unselected. Zero console
errors and zero page errors throughout.

The grading is against the committed draft, not live state: `check` calls `answerFor(draft)` and stores
`key: describeKey(draft)` at commit time (`SelectionShared.jsx` L146–150), and every mutating path routes through
`edit`, which retires. Fixtures and null cases are present and genuinely null — I1 offers 2/2/2/2 (no association) and
a ×3 count scaling (no change in MI, explicitly contrasted with evidence); I2 offers an all-zero labelling and a
display-order change that must leave every score identical; I3 offers the identity donor map and a zero-coefficient
column; I4 offers γ = 0, a target-only edit, and an instance equal to its own reference; I5 offers the malic-acid null.
I confirmed by computation that each of these produces the outcome the adjacent prose promises.

## A11. Verifier counts — all four reproduce exactly

| Claim in the record | Reviewer's re-run |
| --- | --- |
| 181 grouped model checks across 62 groups | `PASS: 181 grouped feature-selection model checks across 62 groups.` |
| 51 program oracles over 4 verbatim programs | `PASS: 4 displayed programs executed verbatim from the manuscript, 51 oracle assertions.` |
| 11 fits matched, data module byte-identical | `PASS: 11 fits re-run and matched against the packet, module 25 KB (byte-identical).` |
| 14 browser cases, 35 screenshots | `{"status":"passed","cases":14,…,"screenshots":35}` |
| `selection-data.js` 25.7 KB | 25,738 bytes |
| shap 0.52.0 + slicer 0.0.8, numba 0.67.0, llvmlite 0.49.0 | all four present at those exact versions |
| Registration complete | blueprint imported (`blueprints/index.js` L21) and keyed (L146); manifest maps the stable ID (L147) |

Also confirmed: every entry point refuses malformed input with a `RangeError` (10/10 guard probes), and the lesson's
prose contains exactly **one** hardcoded decimal literal in the whole file (`0.05`, the significance level) — every other
number on the page is interpolated from the modules I verified above. That is an unusually strong design and it means
the 3,194 checks above cover essentially every number a learner sees.

---

# Part B — Blocking

## B1. I5's tree diagram states a split rule that the tree does not follow, at a value the control accepts

**Where.** `SelectionShared.jsx` `TreeDiagram` threshold rendering; surfaced in `SelectionLabs.jsx` L820–826 (I5 stage 1).

**What is wrong.** The diagram prints node 4 as **`flavan. ≤ 1.59`**. The tree's actual threshold is
`1.5899999737739563`. The I5 flavanoids control has range 0–7, `step 0.01` and 2 decimals, so **1.59 is exactly
enterable**. For that value the drawn rule and the tree disagree, and the disagreement is maximal:

```
flavanoids = 1.59 → float32 = 1.590000033378601 > 1.5899999737739563 → RIGHT
diagram says "≤ 1.59"                                                → LEFT
```

Reproduced on the live page with alcohol 13.5, malic acid 1.73, flavanoids 1.59, proline 672. A learner who reasons from
the diagram predicts the left branch (leaf 5, class-1 probability **0**); the tree goes right to leaf 8, class-1
probability **1**. The page's verdict reads:

> ≠ You recorded Exactly zero; the calculation gives Exactly one. The saved tree sends this sample to leaf 8, whose
> class-1 probability is 1.

I enumerated every value each of the four controls can hold and checked the drawn rule against the tree's rule at each
one. Node 0, node 1 and node 6 have **no** disagreeing enterable value; node 4 has exactly one, and it is 1.59 — the
round number a learner is most likely to type when probing that split.

**Why it matters.** This is the one place in the lesson where it matters most. I5's stated purpose is "follow the actual
split comparisons"; §6 builds an entire float32 argument on thresholds being read exactly; and the investigation
*grades* the learner against the answer. A learner who correctly applies the rule the lesson drew for them is told they
are wrong, with no indication that the drawing, not their reasoning, was the problem. Phase C fix #6 addressed exactly
this defect class at node 1 ("a split was shown as a different rule from the one the tree uses") and moved to two
decimals; two decimals is still not the tree's rule, and at node 4 the residual gap is reachable where at node 1 it is
not. The exact threshold *is* disclosed in the split table underneath — but only after the prediction has been committed
and graded, and I5 (unlike F5, `SelectionFigures.jsx` L577–580) carries **no** "the thresholds are rounded for the
drawing" caveat at all.

**How I verified it.** `scratch/fsel-review/drawn_vs_actual.py` enumerates the full reachable control domain against
both rules and confirms against a fresh native `DecisionTreeClassifier`; `scratch/fsel-review/b1.cjs` reproduces it in
Edge against the production build and captures the verdict text and the page's own split table, which prints
`1.59 | 1.590000033 | 1.5899999737739563 | right (>)` directly beneath a diagram that says `≤ 1.59`.

**Note for the fix.** Rounding to more decimals only moves the boundary; the robust options are to print the threshold
at enough significant digits to be unambiguous against the control's step, to draw the rule as `≤ 1.5899999…`
(truncated with an explicit marker), or to carry F5's rounding caveat into I5 *and* guarantee no enterable value
straddles a drawn threshold. The last is the only one that removes the contradiction rather than labelling it.

---

# Part C — Should-fix

## S1. Six tables scroll horizontally at desktop, not two, and there is no visible cue that they do

**Where.** `selection-labs.css` L75 (`.fs-table-scroll { overflow-x: auto; scrollbar-gutter: stable; }`) and L199–222
(stacking only below 620 px); `design.md` Phase C, "Not fixed, and why".

**What is wrong.** The record names two: "the thirteen-field route table in F1, the full node table in F5". Measured on
the production build, the count and severity are larger:

| Viewport | Regions overflowing | F1 route table hidden | F4 "three predictors" table hidden |
| ---: | ---: | ---: | ---: |
| 1366 px | **6** | 678 px (**48 %**) | 246 px (25 %) |
| 1024 px | 2 | 796 px (56 %) | 365 px (38 %) |
| 768 px | **10** | 1027 px (**72 %**) | 595 px (61 %) |
| 640 px | 4 | 862 px (61 %) | 430 px (44 %) |
| ≤ 620 px | 0 | — (stacks, fully visible) | — (stacks) |

The four undeclared ones are the fold-accuracy table, the "Three separately declared predictors on the same 38
inspection rows" table, the permutation mean/SD table, and the explained-case alcohol table.

Worse than the count: **`scrollbarPx` is 0 on every one of these regions.** `scrollbar-gutter: stable` reserves no space
because Edge is using overlay scrollbars, so nothing is drawn until the reader scrolls or hovers. At 1366 px the F1
route table is simply cut mid-word ("the model is ref…", "the performance metr…") with its third and fourth columns
entirely invisible and no affordance of any kind. In F4, the hidden column of the "three predictors" table is *what each
model got right* — the table's entire purpose.

**Why it matters, and what I do not claim.** No content is lost: the columns are reachable by scroll and by keyboard
(every region has `tabIndex=0`, `role="region"` and an accessible name — I verified this, and the claim that it matches
the neighbouring Regularization lesson's convention is accurate, `RegularizationShared.jsx` L92), the §1 `LessonTable`
carries the same five questions with "what stays fixed" and "suitable starting point" in full, and the F4 prose states
36 of 38 immediately below. So this is not blocking. But F1's caption explicitly routes the reader to that table — "The
labels in the diagram are short names; each route's full question is the first column of the table below" — so the
figure was designed to lean on a table that is majority-hidden at both common reading widths, and the mobile reader at
320 px sees **more** of it than the desktop reader does. The record should also name all six, not two.

**How I verified it.** `scratch/fsel-review/mid.cjs` and `bar.cjs` measure `scrollWidth − clientWidth` and
`offsetHeight − clientHeight` per region across ten widths; the desktop screenshot `shots2/F-00.png` shows the cut.

## S2. Investigation diagrams render label text at 6.8–7.4 effective CSS px at 320, and the code's stated invariant is false

**Where.** `selection-labs.css` L103–107 (`.fs-lesson svg text { font-size: 12px }`, `.is-small { font-size: 10px }`);
`SelectionShared.jsx` L226–227.

**What is wrong.** The SVGs are authored at `viewBox="0 0 340 …"` and sized by CSS, so the declared px size is scaled by
the render width / 340. Measured with every investigation revealed:

| Viewport | Scale | Smallest effective label in an investigation SVG |
| ---: | ---: | --- |
| 1366 px | 1.65 | ~16.5 px — comfortable |
| 390 px | 0.89–1.20 | **8.9 px** (the saved-tree diagram); 4 of 7 SVGs fall below 9.5 px |
| 320 px | 0.68–0.92 | **6.8 px** (saved tree); **all 7** SVGs below 9.5 px, most at 7.4 px |

Figure SVGs at 320 px sit at 8.2 px (130 of 143 text nodes below 9.5 px). The saved-tree labels are the ones carrying
the split fields and thresholds — the content a learner must read to do the exercise.

Relatedly, the comment at `SelectionShared.jsx` L226–227 — "The viewBox matches the rendered width, so a 12px label
renders at 12px" — is false at both ends: ≈1.65× at desktop and ≈0.68× at 320 px. It is the sort of comment a later
maintainer would trust.

**Why it matters.** This is the same defect class Phase C found by looking (fix #1, "SVG labels rendered at 16px inside
a 340-unit viewBox … illegible"). That fix corrected the *inheritance* problem but pinned absolute px sizes that do not
survive downscaling. On a high-DPI phone the glyphs are physically resolvable, so this is a readability complaint rather
than an illegibility failure — which is why it is should-fix and not blocking — but 6.8 px beside 16 px body text in a
teaching diagram is below any reasonable floor.

**How I verified it.** `scratch/fsel-review/narrow-labs.cjs` and `findtree.cjs` compute
`computed font-size × (svg.getBoundingClientRect().width / viewBox.width)` for every `<text>` node with all five
investigations revealed; `shots4/i5-svg0-320.png` is the magnified capture.

## S3. The five-width geometry sweep never inspects a single investigation diagram, and no narrow-width screenshot shows a revealed lab

**Where.** `verify-selection-browser.cjs` L352–359 (section 8) and L379–400 (section 10); resets at L179, 206, 242, 277,
316; `scripts/lib/lesson-visual-layout.cjs` L33 and L39.

**What is wrong.** Each investigation block ends with a `Reset` click. Every investigation's drawn content is inside
`{state.result && …}`, so by the time section 8 runs, all **seven** investigation SVGs are unmounted. The sweep is:

```js
for (const width of [1366, 1024, 768, 390, 320]) {
  const issues = await page.evaluate(inspectLessonVisualLayout, '.fs-lesson');
  assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Figure layout collides at ${width}px`);
}
```

and `inspectLessonVisualLayout` returns `[]` for any SVG that is not visible or carries no `<text>`. There is **no
assertion that anything was inspected** — no minimum SVG count, no label count. Section 10's narrow-width checks
(document overflow, KaTeX overflow, table stacking) run after the same resets. Section 10 does get this right for
tables (`assert.ok(stacking.length > 0, …)`, L397); section 8 has no equivalent. I confirmed the consequence in the
builder's own evidence: `selection-investigation-{1..5}-390.png`, `selection-information-320.png` and
`selection-wine-320.png` all show the **unrevealed** state — controls, presets and a disabled Apply button. **No
screenshot anywhere shows a revealed investigation below 1366 px.**

This matters because Phase C's found-by-looking defects lived precisely there: fix #1 names "the subset lattice, the
donor-arrow map and the saved tree", all three inside `.fs-stage`, and fix #4 is the waterfall overlap. The class of
defect the builder had to find by eye is the class its automated guard does not cover, and the record's Phase C summary
("No label leaves its own SVG, overlaps another label or is crossed by a foreground line, at five widths") reads as
covering the whole lesson.

**In fairness — I tested the unguarded region and it is currently clean.** Revealing all five investigations at 390 and
320 px, I measured across all 20 SVGs: **0** text nodes outside any viewBox, **0** overlapping label pairs, **0**
overflowing stackable tables, **0** KaTeX overflow, and no sideways document scroll. So there is no hidden rendering
defect today; the finding is that nothing would catch one tomorrow, and that the record implies otherwise. The fix is
small: run the sweep again after revealing each lab, and add `assert.ok(inspected >= N)`.

**How I verified it.** Read the verifier and the helper; `scratch/fsel-review/narrow-labs.cjs` reveals every lab and
re-runs equivalent geometry checks; opened `selection-investigation-5-390.png`.

## S4. The pointer contract is real but narrower than the record claims, and the verdict's failure branch is never exercised

**Where.** `verify-selection-models.mjs` L842–893; `design.md` Phase C, "Pointer contract".

**What is wrong.** The record says the verifier "asserts that each of the five places where the prose says an
investigation loads a case from a preset resolves to a preset that exists **and carries those exact values**". The check
is genuinely cross-file and not a tautology — it reads the lesson source and the labs source as separate texts, and the
control ranges it tests against come from `limits` in the models module and `fourFieldModel.limits` in the data module,
which are different sources again. Credit where it is due. But of the five pointers:

| Pointer | What the assertion actually matches |
| --- | --- |
| Practice 1 table | label **and** `counts: [[2, 0], [0, 6]]` — full |
| Practice 4 coalition | label **and** `instance: [1, 2], gamma: 2` — full (reference row not matched) |
| Practice 3 donor | label only; the ordering `donor: [0, 2, 1, 3]` is asserted as a free-floating substring of the file, not as part of that preset |
| Class-3 cohort | **label string only** |
| Alcohol 12.51 → 13.5 contrast | **label string only** |

So "carries those exact values" holds for two of five. It is also a source-text check: a preset object defined in the
file but never rendered, unreachable, or wired to the wrong lab would still pass. (In practice the browser verifier does
click all five by accessible name, so the contract *is* behaviourally covered — but by a different script than the one
whose claim it is.)

Separately, and more substantively: **the browser verifier never records a wrong prediction.** All eight verdict
assertions match `/Your prediction matches: …/`. The `is-miss` branch (`SelectionShared.jsx` L216–221), the `≠` mark and
the "You recorded X; the calculation gives Y" string are untested at every viewport, as is the optional numeric-guess
branch and its within/outside-tolerance message. A regression making `correct` unconditionally true would pass all 14
cases. Likewise, "nothing revealed before a prediction" and "the verdict is retired when an input changes" are asserted
for **I1 only**; I2–I5 get a single "no radio is checked" assertion each. I exercised all of these by hand and they
behave correctly today (see **A10**) — the gap is in the guard, not the behaviour.

## S5. Some of the 181 model checks cannot fail

**Where.** `verify-selection-models.mjs` L418–420, L470, L542–543, L701–703, L711.

Five assertions compare an expression with itself or with a hardcoded constant:

- L418 re-derives the expected hybrid rows by calling `hybridRows(…)` with the same arguments `coalitionGame` used to
  build them (`selection-models.js` L504).
- L419–420 restates `selection-models.js` L513 character for character (`value === outputs.reduce(…)/outputs.length`).
- L542–543 compares `treeProbability(tree, row, i)` with `tree.value[decision.leaf][i]` — `treeProbability` *is* that
  expression.
- L701–703 asserts `(v / extent) * extent === v`.
- L470 and L711 assert `fraction.defined` and `points.evaluatedOnly`, both of which the module returns as unconditional
  `true` literals (`selection-models.js` L667, L877).

None of these is wrong; they inflate a headline count that readers treat as coverage. Note also that the "181 checks"
number counts `record()` calls, many inside loops, whereas the examples verifier's "51 oracles" is an honest assertion
counter. Worth either deleting these or restating what the number counts. In the same spirit: `checkRange` — the guard
behind every control bound — is never imported by the verifier, and no test asserts that an out-of-bounds entry is
refused; `selectedModel.miNats`, `featureLabels`, and `limits[].step`/`.decimals` are rendered literals with no packet
oracle; and the "11 fits" figure is an asserted label (9 + 1 + 1 = 11 over literals), not a counted fact.

---

# Part D — Observations

**O1. The page scrolls sideways at 200 % text size at 390 and 320 px — but so does its neighbour.** With
`html { font-size: 200% }`, `documentElement.scrollWidth` is 482 against a 390 px viewport and 464 against 320 px; at
100 % text both are exact. The widest boxes are the `<th>` cells of the visually hidden stacked `<thead>` (up to
1164 px), though suppressing them did not by itself remove the overflow, so I am reporting the symptom rather than
asserting a mechanism I could not confirm. **In fairness:** the Regularization lesson does the same (425 against
390 px) while `/learn` does not, so this is lesson-shell behaviour affecting at least two lessons, not a defect
introduced here. Worth recording because Phase C fix #8 closed "the document scrolled sideways at 320 px" and the class
is not fully closed at increased text size. At 200 % *zoom* (683 px effective) the page is clean: no document overflow,
no text outside any viewBox, no sub-9.5 px label.

**O2. The waterfall's zero-length baseline bar is acceptable; its caption is looser than the drawing.** With the alcohol
contrast loaded, all four contributions are positive so the baseline (0.33) is the axis floor and its bar collapses to a
1.5-unit stub while "reconstructed 1" spans the panel. I judge this **acceptable as declared**: the value is printed at
the right of the row, a dotted vertical rule marks the baseline, and the zero-valued malic acid contribution is handled
well — rendered as an explicit `0` glyph plus "exactly 0" rather than an invisible rectangle. Two smaller notes. First,
in the *default* state (row 104) the baseline bar (0.33) and the alcohol bar (−0.45) are drawn at **identical length**,
because both happen to span [−0.12, 0.33]; colour separates sign but nothing separates magnitude. Second, the caption
says "Each bar starts where the one above it finished", which is true of the four contribution bars but not of the two
summary bars — the closing bar is drawn from the axis floor, not from where proline finished. Anchoring the axis at
min(0, minimum) for probability-unit waterfalls would resolve both.

**O3. Mandatory prediction before Apply: the right call, with a real cost worth naming.** The specification permitted an
inspect-without-predicting path; the builder made prediction mandatory. I judge this an **improvement** — it satisfies
the teaching standard's "the prediction is recorded, not merely suggested" unconditionally, and it removes the failure
mode where a learner drifts into reading answers. The cost is that the *manipulate → observe* half of the loop is heavy:
after committing, the radios lock, so exploring (sweeping γ in I4, or walking flavanoids across the node-4 threshold in
I5) requires a fresh full commitment per value, and the drawn content disappears between commitments. A "keep my
prediction, recompute" affordance after a first commitment — which cannot leak an answer, since the answer is already
shown — would restore exploration without weakening the contract.

**O4. Two citations are looser than the rest.** The Fisher/Rudin/Dominici link is titled "Model Class Reliance"; the
paper is "All Models are Wrong, but Many are Useful: …". "Model Class Reliance" is its central concept and the title of
its section 4, so the citation is not misleading, but it is not the title. And "Guyon and Elisseeff … sections 2–4 give
small geometrical examples" is loose: the geometrical examples and Figures 1–3 are specifically section 3 ("Small but
Revealing Examples", 3.3 being the XOR case), with section 2 on univariate ranking and section 4 on subset search. The
companion claim about sections 5–7 is exact. Everything else checked out precisely.

**O5. `model_output="probability"` is a no-op for this model, which the lesson could say.** For a scikit-learn
`DecisionTreeClassifier` the raw output *is* the class probability: I ran `model_output="raw"` alongside and got
bit-identical baselines and attributions. Naming the mode explicitly is exactly the right pedagogy — the lesson's point
is "specify the intended mode explicitly rather than relying on defaults" — but a half-sentence saying the two coincide
for this estimator and diverge for margin/log-odds models would make the distinction concrete rather than assumed.

**O6. The cross-lesson asset guard is narrower than its message.** `verify-selection-models.mjs` L947 asserts the lesson
source "never points at another lesson's assets" by testing for the single literal `learn-assets/feature-scaling`. The
property is true — I checked every reference in the lesson, components, data module and verifiers, and the live page
issues no `wine.data` request outside `learn-assets/feature-selection` — but the assertion would not detect a pointer at
any *other* lesson's directory. A regex over `learn-assets/(?!feature-selection)` would match the message.

**O7. Minor record drift.** The Phase C record gives the topic chunk as "193 KB (62 KB gzip)"; the current build emits
195,105 bytes. Immaterial, but the record reads as measured.

**O8. A prose claim with no implementation or oracle behind it.** §3 states that "backward removal from the perfect pair
would reject either one-column reduction under the same strict-improvement requirement". This is true — I verified it
from the lattice (removing A or B drops accuracy from 1 to 1/2, so neither is a strict improvement) — but only forward
selection is implemented (`forwardSelection`, policies `strict` and `forceTwo`), so the backward claim is asserted in
prose and left for the reader to confirm from the displayed table. The table does contain everything needed, so this is
a note rather than a gap.

---

# Summary

One blocking finding, five should-fix, eight observations.

The numerical substance of this lesson is, as far as I can establish, **correct without exception**. 3,194 independent
checks — a from-scratch refit of the entire Wine campaign, exhaustive-ordering Shapley in exact rational arithmetic,
50-digit mutual information over 300 random tables, node-for-node tree comparison against both a native fit and a
hand-rolled Gini recomputation, and an independent `shap` comparison over all twelve explained rows — produced **zero
disagreements** with any published value. Both float32 boundary claims are exactly right, including the subtle one about
the root threshold being a midpoint. The verbatim-program mechanism genuinely holds from frozen manuscript to rendered
page. The provenance is accurate down to the archive's own SHA-256. All fifteen external links support their claims. The
investigation contract — recorded prediction, graded against the committed draft, retired on any edit, nothing on first
paint, a real null case in every lab — is honoured in all five investigations, confirmed by operating the real page. The
four verifiers reproduce their stated counts exactly.

What fails is narrower and almost entirely presentational, but **B1** is not cosmetic: the one diagram a learner is
asked to reason from states a rule the model does not follow at a value the control accepts, and the lesson then marks
their correct reasoning wrong. Fix that, widen the record's account of the scrolling tables and the pointer contract,
give the narrow-width diagrams a font floor, and point the geometry sweep at the investigations it currently skips.
