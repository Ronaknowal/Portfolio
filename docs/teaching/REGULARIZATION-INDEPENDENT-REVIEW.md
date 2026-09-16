# Regularization (L1, L2, Elastic Net, Dropout) — independent phase-two review

Reviewed 14 September 2026 against the working tree plus the uncommitted regularization implementation. The reviewer
authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or
`docs/teaching/drafts/` was edited; this review document is the only write. Throwaway scripts live under
`scratch/regularization-review/`. No state-changing git command was run.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/regularization-review/`, written from the manuscript's own definitions and importing neither `regularization-models.js` nor any `verify-regularization-*` script nor `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1). **368 independent constructed checks, all passing**: the soft-threshold and scalar solution recomputed by a ternary scan of the objective over 160 (z, λ, ρ) combinations rather than by the formula; the one-sided slopes recovered by finite differences; the two-coordinate solutions with their attained budgets, and the constrained twin re-derived by a 6,001-point scan of the diamond at budget 2; the mixed-penalty level set re-derived as the positive root of its own quadratic and checked at 37 directions; a from-scratch coordinate-descent solver (centring, residual maintenance, KKT stopping) run on the four-row, changed-target, target-shift, duplicate and practice fixtures and compared against `calculated-inputs.json` value by value, including the 18-sweep ridge and 29-sweep elastic-net duplicate histories and both coordinate orders; the exact duplicate-segment tie proved by scanning eleven allocations; the dropout enumeration rebuilt from `itertools.product` and matched against the analytic `(1−q)/(2q)Σ(wⱼxⱼ)²` at three fixtures plus the q = 1 and target-shift nulls; the ridge and early-stopping filters and their separately matching λ = 9 and 6; AIC/BIC for both records and both practice records; the factor optimum checked against a 900,000-point brute-force scan of the (a, b) surface at five strengths; the three-position denoising system solved with NumPy; the sixteen-bit code; and **every one of practices 1 to 10**. Separately, the **entire airfoil campaign was refitted from the served `.dat`**: the 1,200/303 split, the three folds, all 18 candidates with their 54 fold fits, every fold coefficient vector, every nonzero count, both baselines, the three selected refits with their scalers and intercepts, and the inference fixture's twenty polynomial terms, standardized values and signed contributions. All three displayed programs were extracted with `node`, written to the filenames the lesson names, and executed beside the served data file. `sha256sum` was run on every reviewed file. |
| **Read in full** | `lesson.md` (673 lines), `visual-specifications.md`, `data-provenance.md`, `design.md` including its phase-two append, the structure and both top-level blocks of `calculated-inputs.json`, the published `.jsx` body (393 lines), `RegularizationShared.jsx`, `RegularizationLabs.jsx`, `RegularizationFigures.jsx`, `regularization-labs.css`, `regularization-models.js`, `regularization-data.js`, `regularization-examples.js`, the blueprint and its registration in `blueprints/index.js`, the four evidence JSONs, the regularization entry of `lesson-delivery-progress.json`, the topic note and its 14 September append, `LESSON-AUTHORING-HANDOFF.md`, and the label-collision and loading-closure sections of `verify-regularization-browser.cjs`. |
| **Looked at** | Nineteen of the twenty-five evidence screenshots were opened and inspected: all eight desktop figures, `figure-2-320`, `figure-4-320`, `threshold-desktop`, `coordinate-constant-desktop`, `coordinate-duplicate-desktop`, `coordinate-history-desktop`, `airfoil-desktop`, `airfoil-changed-desktop`, `airfoil-390`, `dropout-q1-desktop`, `dropout-practice-desktop`. Findings B2, S3, S4, S5, S7 and observations O4, O5 come from that pass. |
| **Reused, not rerun** | `evidence/regularization-models.json` (113 grouped checks), `regularization-native.json`, `regularization-data.json` and `regularization-browser.json` (13 Playwright cases). Every source hash they record equals the hash below for the same file, so their recorded results bind to the bytes reviewed here. The production build, the Playwright run and `verify-curriculum.mjs` were not repeated. |
| **Delegated** | Every external URL in the Sources block and in the body was fetched and checked against the sentence that cites it, including all five PDFs (extracted locally with `pdftotext` rather than trusted to a summariser, because the fetch model returned hallucinated section headings for one of them), the three scikit-learn pages with their three anchors, the UCI record's raw API payload, the PyTorch page and the CC BY deed. Section A5 and observation O9 come from that pass. |
| **Not done** | No production build, no browser session, no keyboard pass, no 200 % zoom check, no reduced-motion check: every interaction finding below was established by reading the components and the screenshots, not by operating the page. Six screenshots were not opened (`threshold-family-desktop`, `coordinate-desktop`, `dropout-desktop`, `threshold-390`, `dropout-390`, `figure-1-320`). The version sensitivity of the airfoil fits was not tested on any library version other than the pinned one. No beginner walkthrough. |

## Source versions reviewed (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/regularization-l1-l2-elastic-net-dropout.jsx` | `8be2a7d7459278fbb6172e93d7dda827111904d191f9c5a3f4003750aec9ee29` |
| `src/learn/data/regularization-models.js` | `ae7eab80bb8686fd8c5956691782c618d6fb39733abd46429832fade868e6281` |
| `src/learn/data/regularization-data.js` | `1c0f6b6f63a69c97a51c0d68b46c32ea1ae92da977690240ce4cf95ab9664609` |
| `src/learn/data/regularization-examples.js` | `1ba29d70893361a5048139d44556fc056e2ff2a53ac0be497e030dd985b9f65e` |
| `src/learn/components/lesson-labs/RegularizationShared.jsx` | `1848ee5191ad55b5190cacdb8e751b7a3fb9cf7b7cfdbd1771ce882d21538574` |
| `src/learn/components/lesson-labs/RegularizationLabs.jsx` | `a25349ae6e4b99c51394eedf3c6caf4b9e6405311729f7bc839f1003b53e76d4` |
| `src/learn/components/lesson-labs/RegularizationFigures.jsx` | `fd72e969721f111becfe38fc8beef902a8692d8448f9449e00215038f008e6ec` |
| `src/learn/components/lesson-labs/regularization-labs.css` | `0927c735f23d9aa66cab6a67b4b52515a8062537cf79173b0cee60df0197f0e8` |
| `src/learn/data/curriculum/blueprints/regularization-l1-l2-elastic-net-dropout.js` | `8bf4eade2a0c67fc2e62819df89ad98258e971ff195e6f05ac278d0b13071f3b` |
| `public/learn-assets/regularization/airfoil-self-noise.dat` | `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3` |
| `public/learn-assets/regularization/ATTRIBUTION.txt` | `889189c0d759ee29b4d336679a6aec1ea073de82bb3bc10c187cc3713ec330b1` |
| `docs/teaching/drafts/.../lesson.md` | `931a38af31e9a4dd199ac664633c28a9a5eb150aa433dc9d49bb169e66d7a095` |
| `docs/teaching/drafts/.../visual-specifications.md` | `46c194a9d0ba96b75330349dc1888c54b9043cfec9e32abd77238d6fc9e7251b` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `efbfd554886fc36a8832b05e59510133babaa0be8106b8de9a82b983e363e465` |
| `docs/teaching/drafts/.../airfoil-self-noise.dat` | `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3` |
| `docs/teaching/drafts/.../design.md` | `19a9f6ba1d7bdefffa58940faf49c7328055620846a4a6ee01f1da93e92784df` |
| `docs/teaching/topic-notes/regularization-l1-l2-elastic-net-dropout.md` | `98dd9acca19cdc96e1fd4f8c4c0f5351433e7e6e97a4c902f984e4ead92e25ea` |
| `docs/teaching/lesson-delivery-progress.json` | `76ca3009ea4a9c611b8cfd7bf16b7976548e6e73ebcb87effb66fcf330e5e2d5` |
| `scripts/verify-regularization-models.mjs` (read only) | `053dce5c657f8b6a3c29b624204c74027a229ff087791d4a0facb6ac0724b4a3` |
| `scripts/verify-regularization-examples.py` (hash only) | `34b084fe280b16ce90317d7313ccf7a92fab3a6b8a4b636722b56ff9771edee0` |
| `scripts/verify-regularization-data.py` (hash only) | `b92db5179c7ea4a438e8f95bf2e330cd2b0baf85459dd7154c1b4af2bc8d7ca3` |
| `scripts/verify-regularization-browser.cjs` (hash only) | `79c9cf5e1b95bd8a3e5ccfa5432a1081420ee37ed8af7c33f4d51bec23b7af61` |

A hash proves identity, not correctness. All four evidence files record source hashes equal to those above, so their
recorded results apply to the version reviewed here. The served `public/learn-assets/regularization/airfoil-self-noise.dat`
is **byte-identical** to the packet's copy (both 59,984 bytes, same SHA-256), and both are the file the evidence names.
The four packet files the content checkpoint binds still match their recorded hashes; `design.md` does not, because
phase two appended to it (see **B1**).

## Part A — correctness

### A1. Manuscript sections, claims, cautions and tasks preserved?

All eleven manuscript sections are present in the published body in order, with the declared first-pass route
(`.jsx` L56), the three-family penalty table, all three deeper branches labelled "Deeper branch" in their headings,
all three programs in the manuscript's order, practices 1–10 with hints and solutions in separate closed disclosures,
and a readiness-check table. An automated shingle comparison of every manuscript sentence against the implementation
found no substantive prose dropped: the 79 sentences it flagged are all either figure/investigation *specification*
blurbs (correctly realised as components rather than prose), sentences whose content is carried by an interpolated
computed value, or sentences whose only content is a display equation.

The five characteristic caveats named in the review brief were checked sentence by sentence and are **all present**:

- **A penalty is a modelling choice, not a truth** — the intro ("The preference changes the problem being solved.
  Whether it improves future predictions is something to assess using the data boundaries from the previous lesson.",
  `.jsx` L61), §1's "That does **not** by itself establish that w=1.5 predicts future cases better" (L72), F1's closing
  paragraph (`RegularizationFigures.jsx` L57–61), §3's "These costs come from **different penalty functions**, so the
  smallest of those numbers is not a valid way to choose which family predicts best" (L136), and I1's own family table
  caption ("These are minima of three different objectives, and the smallest of them is not a way to choose a family",
  `RegularizationLabs.jsx` L128).
- **The coefficient path depends on the feature scaling** — §1's whole "Units change the meaning of a penalty"
  subsection (L75–79) with F1's metres/centimetres inset showing the correct direction (L62–77 of the figures file),
  and F4's coefficient panel: "coefficient value in this fold's own standardized coordinates … One fold's scaler and
  coefficients belong to that fold alone: these are not averaged across folds and not combined with the final
  all-development refit" (`RegularizationFigures.jsx` L358–363).
- **L1 sparsity is not guaranteed feature selection** — §3's KKT paragraph ("not a one-time test of the
  ordinary-least-squares coefficient or proof that a feature is irrelevant to the world", `.jsx` L122), §4's closing
  ("Its nonzero list is still a property of this fitted model, its feature representation and its penalty. Prediction,
  stable selection and causal explanation require different evidence", L152), §5's "L1 can create sparsity; the
  selection objective and sample do not guarantee that the chosen fit will be sparse" (L181), and practice 6.
- **Dropout is stochastic and its expectation is a different object from one realisation** — this is handled in the
  strongest possible way: **nothing is sampled anywhere in the lesson.** I4's note reads "Every branch below is
  enumerated with its actual probability. There is no seed, no sampling and no displayed realisation: this is the
  exact expectation" (`RegularizationLabs.jsx` L493), `dropoutEnumeration`'s own source comment says the same
  (`regularization-models.js` L363–366), and the section derives the expectation, enumerates all four masks with their
  real probabilities, and shows the discrete output distribution with its mean marked separately from the branches.
  The specification's demand that the figure "name its seed and say it is one realisation, **or** show the expectation
  exactly" is met by the second branch, exactly.
- **Any airfoil result is one dataset under one declared protocol** — §5's opening three paragraphs (L163–166),
  `provenance.limits` (three sentences in `regularization-data.js` L44–48), F4's source paragraph
  (`RegularizationFigures.jsx` L383–390), practice 6, and the lesson's closing paragraph (L389).

Also preserved and checked: elastic net "is a family of preferences, not a guarantee that its answer or prediction error
lies between those of separately tuned ridge and lasso"; "ridge simply has no threshold region that systematically
creates sparsity" (the manuscript's correction of "L2 can never produce a zero"); "Diamond corners and faces make exact
zero coordinates possible … They do not force every optimum to a corner"; "Full-column-rank lasso is unique even when
its columns are correlated"; "Strict inequality therefore forces zero; equality alone can occur with a zero or nonzero
coefficient"; "a `weight_decay` argument is not a universal mathematical identity"; "BIC values are not exact posterior
probabilities"; "**MDL and BIC are not identical in general**"; "no_grad() … does not itself switch dropout into
evaluation behavior"; and the §6 callout's "There is no universal best rate, fixed extra-epoch multiplier or rule that
adding more regularizers must improve a model."

Deviations found, each judged:

| Location | Deviation from manuscript / specification | Judgement |
| --- | --- | --- |
| `regularization-examples.js` `coordinateFit.expected` | `elastic_net [1.666667 0.      ]` replaces the manuscript's `[1.666667 0.]` | Correct and honestly handled — declared departure 1, and the page says so (`.jsx` L140). See **A3**. |
| `airfoil_regularization.py` L23–24 | Two comment lines added at `np.loadtxt` naming the tab-separated 1,503 × 6 layout | Declared departure 2; verified to be the *only* code change (byte diff against the manuscript block). |
| `.jsx` L137–139 | A `Checkpoint` posing the manuscript's two I2 contrasts as a question | Declared departure 4; adds no claim. |
| `.jsx` L207–209 | §6's closing paragraph rendered as a `Callout` | Declared departure 5; same words. |
| `.jsx` L178 | "The equivalent full calculation ran serially in the author environment with no warnings" becomes "it ran with no warnings", now asserting it of the *displayed* program | Not declared, but **true**: my own run of the extracted program emitted nothing on stderr. Belongs in the departures list. |
| `RegularizationShared.jsx` L187 | Every investigation gains a "Calculate without recording a prediction" button | Not declared; contradicts a binding contract line. Finding **S1**. |
| `RegularizationFigures.jsx` L334–364 | F4's coefficient panel omits the specification's per-term selector | Not declared. Finding **S2**. |
| `regularization-models.js` L35–41 | I3's five raw controls use a wider physical band (e.g. frequency 100–25,000) than the observed columns the specification named as the control ranges | Reasonable, and the observed ranges are stated in a caption and a table column — but undeclared. Observation **O3**. |
| `RegularizationFigures.jsx` L428–432 | A paragraph describing F5 that contradicts the figure, its own legend and its own `aria-label` | Finding **B2**. |
| `.jsx` L294–301 | A readiness-check table not in the manuscript | Good addition; maps each item to where it was taught. |
| `.jsx` L389 | A closing "constructed calculations" paragraph not in the manuscript | Good addition; it is the clearest single statement of what is and is not measured in the whole lesson. |
| `design.md` "Departures" | Lists six; at least ten exist | Finding **S6**. |
| Ledger / topic note | Assert contradictory states | Finding **B1**. |

### A2. Independent recomputation (reviewer's own scripts, no lesson code)

Every numeric claim in the manuscript and in the published body was recomputed from first principles. **Nothing
disagreed.** 368 constructed checks and the full airfoil refit passed with zero failures.

| Claim | Independent result | Agrees? |
| --- | --- | --- |
| **Soft-threshold and scalar solution** `S(z, λρ)/[1+λ(1−ρ)]` | a ternary scan of `½(w−z)² + λ[ρ\|w\| + (1−ρ)w²/2]` over 160 (z, λ, ρ) combinations reproduces the closed form to 1e−6 everywhere, including both sides of every threshold | Yes |
| §2 table: z = 3 → 1.5 / 2 / 5/3; z = 0.4 → 0.2 / 0 / 0; z = −2 → −1 / −1 / −1 | identical, with the two zeros exact | Yes |
| **One-sided slopes at the minimum** `−z ∓ λρ` | finite differences at h = 1e−6 on three fixtures match the analytic left/right slopes to 1e−4 | Yes |
| I1's contrasts and nulls: z = 0.4 → 1.4 gives 0.4; z = −0.6 stays exactly 0; λ = 0 returns z for every ρ | identical | Yes |
| **Two-coordinate solutions** at z = (3, 0.4): lasso (2.9, 0.3) at λ = 0.1 and (2, 0) at λ = 1; ridge z/(1+λ); EN (5/3, 0) | identical | Yes |
| **Attained budgets** 3.2, 2, 3.785124, 1.145, 3.572562, 1.527778 | recomputed from the penalty measure `ρ‖w‖₁ + (1−ρ)‖w‖²/2`; all six match the published table to six decimals | Yes |
| **The constrained twin really touches there** | a 6,001-point scan of `\|w₁\|+\|w₂\| = 2` minimising `½‖w−z‖²` returns (2.000, 0.000) to 2e−3 — the contact point is the arithmetic's answer | Yes |
| **Mixed-penalty level set**, closed form | re-derived as the positive root of `((1−ρ)/2)r² + ρ(\|u₁\|+\|u₂\|)r − B = 0` and evaluated at 37 directions: the penalty measure on the returned boundary equals the budget to 1e−9 at every one | Yes |
| **Four-row fit**: ridge (1.5, 0.2), lasso (2, 0), EN (5/3, 0), intercept 0, one sweep each | identical from my own solver; objectives **2.29, 2.58, 2.4966666667** reproduced to 1e−12 and matched against `calculated-inputs.json` | Yes |
| λ_max = 3 for the four-row design | `max_j \|Z_jᵀt\|/n = 3` exactly | Yes |
| I2 contrast: row 0 target 3.4 → 7.4 gives (3, 0.4) with intercept 1; null: +7 on every target gives (2, 0) with intercept 7 | identical, both in one sweep | Yes |
| **Duplicate design**: ridge (2/3, 2/3) in **18 sweeps**, EN (0.6, 0.6) in **29 sweeps**, lasso (1, 0) or (0, 1) by coordinate order | every sweep count, every intermediate weight pair and every KKT residual in the packet's histories reproduced; final ridge KKT `2.9103830456733704e−11` matches the page's `2.910 × 10⁻¹¹` | Yes |
| Duplicate lasso segment: (1,0), (0.5,0.5), (0,1) all objective **1.5**; ridge **2/3**; EN **1.1**; sums 1, 4/3, 1.2 | eleven allocations of the sum all give exactly 1.5; the other two objectives reproduce | Yes |
| **Constant-column case**: coefficient 2 declared 0 on a flat objective, objective 2.5, MSE 1, λ_max 3 | recomputed by hand: `a₂ = 0`, `c₂ = 0`, data 0.5, penalty 2, total 2.5 | Yes |
| **Dropout enumeration** at x = (2,1), w = (1,−1), y = 1, q = 0.5: outputs 0, −2, 4, 2; half-losses 0.5, 4.5, 4.5, 0.5; probabilities all 1/4; mean 1; expected loss 2.5 | identical; and `(1−q)/(2q)·(4+1) = 2.5` equals `expectedLoss − cleanLoss` to 1e−12 | Yes, exactly |
| **Practice 5** at x = (1,2), w = (2,0), y = 1, q = 0.75: mean 2, clean ½-loss 1/2, expected 7/6, difference 2/3 | identical to 1e−12, and the analytic term `(1−q)/(2q)·4 = 2/3` matches | Yes |
| I4 nulls: q = 1 gives extra loss 0 and expected = clean; y = 1 → 3 leaves the difference at 2.5 | identical | Yes |
| Nonlinear inset: mean of `max(0, u−1)` over {0, 2} is 0.5; `f(mean) = 0` | identical | Yes |
| **Ridge filters** 16/17 ≈ 0.9412, 0.2, 0.8, 0.0588235 at nλ = 1 and 4 | identical | Yes |
| **Early stopping** factors 0.1 and 0.4 at η = 0.1, t = 1; matching ridge λ = 9 and 6 | `a(1−f)/f` gives exactly 9 and 6; one common λ cannot do both | Yes |
| **AIC 306 / 302; BIC 313.8155105580 / 315.0258509299**; BIC penalty gap 9.2103403720 | identical to 1e−9 | Yes |
| **Factor optimum** p\* = max(1−2λ, 0): at λ = 0.25 product 0.5, data 0.125, penalty 0.25, total 0.375 against the balanced pair's 0.5; (2, 0.5) penalty 1.0625 | identical; and a 900,000-point brute-force scan of the (a, b) surface at five strengths confirms **no pair beats the reduced answer** (agreement to 3e−4, the scan's own resolution) | Yes |
| **Denoising**: (I + LᵀL)w = y gives (0.5, 1, 0.5) against identity ridge (0, 1, 0); edge differences (0.5, −0.5) | identical; the published `I + LᵀL` matrix `[[2,−1,0],[−1,3,−1],[0,−1,2]]` is correct | Yes |
| **Sixteen-bit code**: `0101…` → 5 bits, `0101010001010101` → 17 bits, `1110…` → 5 bits | identical | Yes |
| **Practices 1–10**, every stated value: −1.5 / −1.8 / −21/13 ≈ −1.615385; (2.5, 0) with intercepts 0 and 3; alpha 16 and 12 against Lasso alpha 0.2; sum 2.5 with (2.5,0) and (1.25,1.25); 2, 1/2, 7/6, 2/3; the λ = 10 baseline identity; 0.8, √0.8 ≈ 0.894427, 0.02 / 0.16 / 0.18 against 0.2; mode 1, payload 1110, 5 bits; AIC 164/162 and BIC 167.8240/169.6481; (3.5, 4, 3.5) against (1.5, 2.5, 1.5) | every one reproduced | Yes |
| **Airfoil split** 1,200 / 303 from `train_test_split(..., train_size=1200, random_state=41)` | index-for-index identical to `development_indices` and `reserved_indices` | Yes |
| **Three folds** from `KFold(3, shuffle=True, random_state=202)` | identical after mapping positional indices through `development`; 800 fitting and 400 validating in each | Yes |
| **All 18 candidates** — 18 mean MSEs, 54 fold MSEs, 54 nonzero counts, 54 twenty-element coefficient vectors | every value reproduced to 1e−9 (coefficients to 1e−8); `np.loadtxt` → `PolynomialFeatures(2)` → `StandardScaler` → estimator, refit from the raw `.dat` | Yes |
| **Baselines** 45.0727679595 and 17.3502676828, and the three fold training means 124.978983 / 125.025092 / 125.037473 | identical | Yes |
| Lasso keeps **9, 9, 9** at λ = 0.1, **0, 0, 0** at λ = 10 and 100 with the fold MSEs equal to the mean baseline's, **19, 20, 19** at λ = 0.001, and **20** at the all-development refit | every count reproduced; the λ = 10 lasso fold MSEs equal the baseline fold MSEs to the last bit | Yes |
| **Three selected refits** at λ = 0.001, with intercepts, twenty coefficients, twenty scaler means and twenty scales | identical to 1e−8 | Yes |
| **Inference fixture**: row 1203 at (1250, 17.4, 0.0254, 31.7, 0.0176631) predicts **124.4653054006435 dB**; at 1750 Hz, **123.62864627978571 dB**; decrease **0.836659121 dB** | identical to the last digit, both by the pipeline and by hand from the saved scaler and coefficients | Yes |
| **All twenty signed contributions** on the base row, and the intercept 125.013849167 | every one matches the page to six decimals; contributions sum to −0.548543766 and the total is 124.465305401 | Yes |
| Changing frequency moves exactly **six** of the twenty terms | indices {0, 5, 6, 7, 8, 9} change and no others | Yes |
| The campaign raises **no warnings** | `warnings.catch_warnings(record=True)` around the whole 60-fit refit caught nothing; the extracted standalone program wrote nothing to stderr | Yes |
| `reservedPredictionsComputed: false` | the data module ships **no reserved-row data at all** — not an index, not a prediction — and the browser verifier asserts the page never fetches the `.dat` | Yes |

**Could not verify numerically:** the asymptotic cost statements `O(nd²)`, `O(d³)`, `O(d²)` and `O(nd)` per sweep (they
are not computed quantities); the reading-time estimates; and the version-independence of the airfoil fits — every
airfoil number is bit-exact only in the pinned environment, which is also the environment the author used. I confirmed
run-to-run determinism but tested no other library version. See **O8**.

### A3. Displayed programs versus their published output

All three programs were extracted from `regularization-examples.js` with `node`, written to
`coordinate_regularization.py`, `airfoil_regularization.py` and `dropout_masks.py` beside a copy of the served data
file, and executed. **Every one reproduces its stored `expected` block exactly** — including the 24-line airfoil table,
the three `selected …` lines, and NumPy's padded `elastic_net [1.666667 0.      ]`. The only differences are Windows
`\r\n` line endings on my run.

The displayed code was also diffed against the manuscript's own code fences: `coordinate_regularization.py` and
`dropout_masks.py` are **byte-identical** to the manuscript, and `airfoil_regularization.py` differs by exactly the two
declared comment lines. No program prints an explanatory or cautionary sentence; the only guards are three real
`ValueError`s on the fit's own preconditions and one `RuntimeError` on non-convergence. Displayed code is 44, 55 and 14
lines.

**The padding departure is handled honestly.** The page publishes the output an actual run produced and adds one
sentence beside it: "The third line is padded by NumPy's own array formatting, which aligns the two entries; the values
are 1.666667 and exactly 0" (`.jsx` L140). That is the right disposition — the executed bytes are shown, the cosmetic
difference is named, and the manuscript's claim about the *values* is restated. Nothing is fudged and nothing is hidden.

### A4. Model and lab logic read for defects

- `softThreshold` (`regularization-models.js` L68–74) returns a literal `0`, not a small number, and every consumer
  tests `=== 0`. `coefficientText` (`RegularizationShared.jsx` L33) then prints the word "exactly 0". The
  exact-zero-versus-rounded-tiny distinction the specification demands is real throughout.
- `twoCoordinateSolution` (L137–159) returns `budget` as the **attained penalty measure** and `contactRadius` as the
  Euclidean distance from z to the solution, and `GeometryPanel` draws the disk/diamond/mixed boundary at that budget
  and the data contour at that radius, both on one scale (`212/(high−low)` for the x axis, the y axis and the circle
  radius alike). The contact point is therefore a consequence of the arithmetic, not a placed dot. I confirmed the
  constrained optimum independently by scanning the diamond.
- `penaltyBoundaryRadius` (L164–178) solves the level-set quadratic in closed form and handles both degenerate cases
  (`ρ = 1` linear, `ρ = 0` quadratic) separately. Sampling it at 241 angles preserves the axis corners; my own
  evaluation confirms the sampled polygon lies on the exact level set.
- `coordinateFit` (L261–326) stops on the **optimality condition**, not on a small step, keeps a per-visit trace with
  the partial residual after every coordinate, and returns `converged: false` with the iterate intact when the sweep
  cap is reached — and `CoordinateLab` L303–306 prints the unconverged sentence verbatim. The constant-column branch
  (`denominator > 0 ? … : 0`) carries a source comment naming it a declared representative solution, and the table cell
  says "declared 0 (flat objective)".
- `dropoutEnumeration` (L367–397) computes both routes and exposes `agrees`, which the lab renders as a sentence rather
  than asserting silently. `possible: probability > 0` drives the dashed outline, so a q = 1 branch is listed as
  impossible rather than deleted.
- `useInvestigation` (`RegularizationShared.jsx` L105–131) is the strongest part of the interaction design: `edit`
  clears the result, the radio choice and the numeric guess together; `check` computes the answer **from the draft at
  the moment of commitment** (`answerFor(draft)`, not `answerFor(active)`); and `describeKey` stamps the committed
  inputs onto the result. **No investigation grades a prediction against state committed after the prediction was
  recorded.** I traced all four: `ThresholdLab`, `AirfoilTraceLab` and `DropoutLab` derive their answer only from
  `inputs`; `CoordinateLab` additionally reads `output` and `row`, and both of those setters call `state.reset()`, so
  they cannot change under a live prediction. The `sweep` selector affects only which sweep is displayed.
- Defects found by reading: the explore button that bypasses the contract's "require a prediction before Apply"
  (**S1**); `explore()` leaving `state.choice` set so an ungraded result can appear beside a checked radio (**O13**);
  the F5 paragraph (**B2**); the F4 term selector (**S2**); `changeText` producing "a change of no change" (**S5**);
  three practice values and one figure column printing an ASCII hyphen for a negative number (**S4**).

### A5. Links, references and provenance

**All twelve external URLs resolve to live documents and support the sentences that cite them.** The five PDFs were
downloaded and their text extracted locally rather than trusted to a fetch summariser — which mattered, because the
summariser invented section headings for Zou & Hastie that local extraction contradicted.

- **scikit-learn `linear_model.html`** — all three cited anchors exist as literal `id` attributes
  (`#lasso`, `#ridge-regression-and-classification`, `#aic-and-bic-criteria`). The page documents `Ridge` as
  `min_w ||Xw − y||²₂ + α||w||²₂` (summed, no `1/n`, no `1/2`) and `Lasso`/`ElasticNet` with an explicit
  `1/(2·n_samples)` factor — **exactly the asymmetry the lesson's §5 conversion table rests on**. The Lasso section
  really does document coordinate descent with a duality-gap convergence criterion, and the AIC/BIC section really does
  state its Gaussian log-likelihood convention. The lesson's `Ridge(alpha = n_fit * strength)` conversion follows from
  multiplying our objective by 2n, which is what practice 3 says.
- **Zou & Hastie (2005)**, *J. R. Statist. Soc. B* 67(2), 301–320: §2.3 is "The grouping effect", §3.1–3.2 are the
  naive-estimate deficiency and the corrected estimate, and §2.4 is "Bayesian connections and the Lq-penalty" with the
  elastic-net prior. Every element the lesson attributes to sections 2–3 is there. One wording quibble at **O9**.
- **arXiv 1206.0313** is **Ryan J.** Tibshirani, "The Lasso Problem and Uniqueness"; §2.1 is literally "Basic facts and
  the KKT conditions", Lemma 1(ii) is the equal-fitted-values property verbatim, and Lemma 3 is the general-position
  uniqueness condition. All three cited elements present, and the author initial is right.
- **JMLR srivastava14a**, **Wager/Wang/Liang**, **Loshchilov & Hutter** — titles, authors, venues and the specific
  claims attached to each all verified from the documents' own text.
- **Grünwald's tutorial** — §2.5.3 is "NML as an Optimal Universal Model", §2.6.3 is "Bayesian Interpretation" with the
  Laplace approximation of the evidence, and §2.9.2 is "MDL and Bayesian Inference" containing the explicit statement
  that "MDL = BIC" is wrong. All three cited section numbers are correct and on-topic, which is the most likely place
  for a mis-citation in a lesson like this and is not one.
- **UCI dataset 291** — 1,503 instances, 5 features + 1 target, creators Thomas Brooks / D. Pope / Michael Marcolini,
  1989, DOI `10.24432/C5VW2C`, licence **Creative Commons Attribution 4.0 International (CC BY 4.0)**, and the five
  variable names and units exactly as the lesson states. The DOI in `provenance.doi` and in `ATTRIBUTION.txt` matches.
  The CC BY deed link resolves. Attribution is complete: creator, title, source link, DOI, licence with a link, and an
  explicit statement that the file is unchanged.
- **PyTorch `nn.Dropout`** at `docs.pytorch.org/docs/main/...` returns 200 directly and documents `p` as the drop
  probability, per-element independent masking, `1/(1−p)` scaling during training, and identity in eval — all three
  claims the lesson makes.
- **Both scikit-learn example pages** resolve; the lesson's titles are one-word paraphrases of the real ones
  ("with L1 prior (Lasso)" → "with an L1 prior"; "AIC-BIC / cross-validation" → "AIC, BIC and cross-validation").

**Provenance of the served bytes.** The served `.dat` is byte-identical to the packet copy (59,984 bytes, SHA-256
`74c75fd7…`), which is the hash the ledger's content checkpoint, `provenance.sha256`, `ATTRIBUTION.txt`, the Sources
block and `regularization-data.json` all record. **The separator the program relies on is stated on the page**: "The
file is tab separated with CRLF line endings, which is the whitespace layout `np.loadtxt` reads by default, so the six
columns arrive in source order with no parsing options" (`.jsx` L177). I verified this against the bytes: 1,503 CRLF
lines, no bare LF, exactly five tab characters per line.

## Part B — learning-experience checklist (reviewer's heuristic run)

| # | Item | Finding |
| --- | --- | --- |
| 1 | **Route** | Present and declared: `.rg-route` (`.jsx` L58) names sections 1–6 and practice 1–6 as the first pass and sends 7–9 to a later sitting. All three deeper sections carry "Deeper branch" in their headings. Both time bands are in `readTime`. |
| 2 | **Cautions** | Each caveat has a single home and is used as mechanism thereafter rather than restated as a warning. The unit caveat lives in §1 and returns in F4 as a *fact about the fold's coordinates*; the "development selection, not independent evidence" caveat lives in §5 and returns as practice 6's task rather than as a repeated hedge. No displayed program prints a disclaimer. |
| 3 | **Real question** | Opens on a modelling choice and returns to it in §5 with 1,503 real aeroacoustic rows, a downloadable unchanged file with its hash, a CC BY 4.0 attribution with a licence link, a split declared before the rule, a predeclared λ grid, and a result the reader can judge — including the fact that every family selected the grid's boundary and that the final lasso refit is not sparse. |
| 4 | **Labs as investigations** | **I1**: prediction unset, threshold band drawn on the z axis, the z→w map plotted as a separate dependent output, both nulls reachable as presets, and the family table explicitly refusing to be a family-selection criterion. **I2**: the rows, both features, the targets, λ, ρ *and the coordinate order* are all editable; the per-visit association/curvature/threshold/divisor table and the residual-after-every-coordinate table are exactly what the contract asked for; the duplicate preset returns (1,0) or (0,1) with the order recorded alongside the prediction; the constant column is declared rather than hidden. **I3**: the editable entities really are the five physical measurements, the twenty terms are derived, six chips light up when frequency moves, and the reference target is labelled a comparison value in three places. **I4**: exact enumeration, non-uniform probability strips, an impossible-branch convention, and the analytic twin shown independently. All four grade against the committed draft. The shared defect is the explore button (**S1**). |
| 5 | **Figures** | Eight figures against eight contracts. F1, F2, F3, F6, F7 and F8 are legible and correct at desktop and (for F2) at 320 px. F2 is the best figure here: the diamond and the circle are drawn on one genuinely equal scale from the *attained penalty values*, the touching point is where the arithmetic puts it, and the caption says so. F4 is correct but drops its term selector and loses its y scale (**S2**). F5's picture is repaired but its paragraph is not (**B2**). |
| 6 | **Connections** | Strong. `(3, 0.4)` is introduced in §2, solved geometrically in F2, re-derived as `ZᵀT/n` in §3, and its `λ_max = 3` returns in §7. The duplicate design appears in §4, in I2 as a preset, and again in §7's uniqueness discussion. The dropout penalty `(1−q)/(2q)Σ(wⱼxⱼ)²` is derived in §6, enumerated in I4, and re-tested in practice 5. The scaling caveat of §1 is what makes F4's per-fold coefficient warning necessary. §9's criteria are tied back to §5's "development-selected score is not a test result". |
| 7 | **Code** | Compact and mechanism-dominated: 44, 55 and 14 displayed lines. The guards are real preconditions, not reviewer-driven noise. The two `pip install` pins are inline `Code`, not fenced blocks. |
| 8 | **Practice** | Ten tasks, each changing both numbers and context, hints and solutions in separate closed disclosures, every stated value reproduced. Only one task points into a lab (practice 5 → the dropout investigation's practice preset) and that pointer is **true**: the preset exists at `RegularizationLabs.jsx` L458 with exactly x = (1,2), w = (2,0), y = 1, q = 0.75. |
| 9 | **Screenshots** | Twenty-five captures at five widths covering most specification-required informative states. Gaps at **S7**. |

## Ranked actionable findings

Severity: **blocking** = wrong on the published page, or a record that asserts something untrue; **should-fix** = an
inaccuracy, a defeated teaching mechanism, or a dropped specification item with a bounded fix; **observation** =
recorded for the ledger.

### Blocking

**B1. Two project records assert contradictory implementation states, and the content checkpoint's design-record hash is stale.**

Locations: `docs/teaching/lesson-delivery-progress.json`, topic entry `regularization-l1-l2-elastic-net-dropout`;
`docs/teaching/topic-notes/regularization-l1-l2-elastic-net-dropout.md` L23–32.

Evidence:
- The ledger entry reads `"implementation": {"status": "not-started"}` and its `nextAction` ends
  "Content is complete; implementation has not started."
- The topic note's append, dated **14 September 2026** and titled "**Implemented and closed**", states "The
  factor-symmetry idea is now implemented in the published lesson's section 8 … `factorOptimum` in
  `src/learn/data/regularization-models.js` computes all of it, and `scripts/verify-regularization-models.mjs` checks
  it … Status: **closed**; the destination lesson carries the content". Both files cannot be right.
- The ledger's content checkpoint binds `docs/teaching/drafts/.../design.md` at
  `4061830bd85c2cba327140e728613dd63d5faa127f2c8ee962a24d7696c6e214`; the file is now
  `19a9f6ba1d7bdefffa58940faf49c7328055620846a4a6ee01f1da93e92784df` after the phase-two append. The other six packet
  files still match.
- `LESSON-AUTHORING-HANDOFF.md` does not mention this topic at all; its "Latest completed scope" still names t-SNE/UMAP
  (position 17) and its "Next eligible prepared topic" is ICA (position 18). It asserts nothing false about
  regularization, but it does not record this work either, and this review document is linked from nowhere.

Mitigating, and worth crediting: `design.md` closes with "the phase ledger is the parent's to close", so the implementer
disclosed the gap rather than papering over it. The defect is that the **topic note did not wait**: it declares the work
closed while the authoritative ledger declares it not started.

Fix: (a) update the ledger entry to `implementation: complete` with the final source hashes, a real next action and a
link to this review, and refresh the design-record hash in the content checkpoint; (b) once (a) is done, the topic note's
"closed" becomes true and needs no edit — until then it is the record that is wrong; (c) add a regularization section to
`LESSON-AUTHORING-HANDOFF.md` linking the design record, the evidence and this review.

**B2. Figure 5's explanatory paragraph says the arrows are drawn to their singular values. They are not — and the same figure's legend and `aria-label` say so.**

Location: `src/learn/components/lesson-labs/RegularizationFigures.jsx` L428–432:

```jsx
<p>
  Coefficient space, equal scale on both axes. Moving one unit along v₁ changes the fitted observations by {first}; moving one
  unit along v₂ changes them by {second}. The arrows are drawn to each direction's own singular value; the directions themselves
  are orthogonal and of equal length.
</p>
```

Evidence: the two direction stems are drawn at L413–414 as

```jsx
<line className="rg-stem is-data" x1="74" y1="112" x2="120" y2="112" />
<line className="rg-stem is-penalty" x1="74" y1="112" x2="74" y2="66" />
```

— **both exactly 46 viewBox units**, not 4 and 0.5. The quantities drawn to the singular values are the two *bars* on
the right (L421 and L423: `width={(28 * first)}` = 112 and `width={(28 * second)}` = 14). The same `<svg>`'s in-figure
legend reads "two unit directions, equal length" (L418–419), and its `aria-label` (L410) reads "drawn at the same
length because both are unit vectors". So the paragraph contradicts the picture, the legend printed inside the picture,
and the accessible description of the picture; a sighted reader and a screen-reader user are told opposite things. I
confirmed the rendered result in `regularization-figure-5-desktop.png`: the two stems are visibly equal.

This matters more than a stray sentence because it re-creates precisely the misreading the design record says was
repaired. Repair note 5 reads: "the 'v₂' arrow was a 12-unit stub beside a much longer grid line, so the grid read as
the direction … Redrawn as two equal-length unit directions paired with two data-sensitivity bars on one shared scale."
**The drawing part of that repair is real and verified.** The sentence is a survivor of the version being repaired, and
it instructs the reader to look for a length difference that the fixed figure deliberately no longer has.

Fix, one clause: "The two bars beside them are drawn to each direction's own singular value; the directions themselves
are orthogonal and of equal length." No verifier rerun is needed.

### Should-fix

**S1. Every investigation offers a one-click way to skip the prediction, which the binding contract does not permit and the intro does not mention.**

Locations: `RegularizationShared.jsx` L124–127 (`explore`), L150 and L187 (the button);
`visual-specifications.md`, "Shared implementation contract".

Evidence: the contract's second sentence is "**Require a prediction before Apply**; bind it to the complete applied
input and the named output", and the only relief it grants is "Explanatory **figures** can show their already worked
answer." The implementation renders a second button, "Calculate without recording a prediction", in all four
investigations; it calls `state.explore`, which sets `active` and displays the **complete** result — every stage table,
every plot, the full verdict — with `graded: false`. The primary button is correctly gated (`disabled={!ready}`), so the
gate exists; the second button walks around it. The browser verifier itself uses this path to render all four
investigations for the layout pass (`verify-regularization-browser.cjs` L333–337), which is a fair signal of how easy
the bypass is.

Unlike the GMM specification, this one does not authorise an explore action anywhere, and the departure is not in the
design record's list. The intro also promises "Every investigation asks for a prediction before it shows an answer"
(`.jsx` L57), which the button makes into a statement about the interface's *request* rather than its behaviour.

Fix, cheapest version that keeps the affordance honest: keep the button but make the ungraded path show the readout and
the numeric answer **without** the staged explanation, or label it "Show the worked answer without recording a
prediction" and say in the intro that it exists. If the contract is to be followed literally, remove the button from
the three investigations and keep it nowhere.

**S2. Figure 4's coefficient panel drops the specification's per-term selector, so twenty identical lines carry no readable individual value.**

Location: `RegularizationFigures.jsx` L334–364; evidence `regularization-figure-4-desktop.png`,
`regularization-figure-4-320.png`.

Evidence: the F4 contract says "All 20 named terms have signed values; zero has an explicit mark. **A selector can
inspect one term or a small selected group**, while full data remain accessible." Three selectors exist — family, fold
and λ — but none for a term. Twenty `<polyline>`s are drawn in one colour (`stroke="#6f8b7e"`, L342) with no legend, so
no individual coefficient path can be followed. Compounding it, the vertical axis carries exactly one label, `0`
(L339): there is no scale, so a reader cannot even estimate a magnitude from the picture. The full table below is
correct and complete, which is why this is should-fix rather than blocking, but the panel's stated job — showing *paths*
— is not delivered for any single term.

What is right and worth keeping: the exact zeros are marked with an open red cross on the zero line (L346–352) and the
caption says the cross is an exact zero rather than a rounded small value; the panel uses one fold rather than an
average, and says why.

Fix: highlight the term selected in the table on hover/focus, or add a `<select>` of the twenty `featureNames` that
thickens one path and dims the rest; and label two more y ticks from `coefficientExtent`, which the component already
computes.

**S3. Two table cells that carry the teaching point are clipped at desktop width.**

Locations: `RegularizationLabs.jsx` L273 inside the coordinate-step table; `RegularizationFigures.jsx` L604–610 (F7's
"Neighbouring differences" panel). CSS: `regularization-labs.css` L61–64 (`overflow-x: auto` with
`white-space: nowrap`).

Evidence, from the screenshots rather than from the assertions:
- `regularization-coordinate-constant-desktop.png`: the flat-column row's coefficient cell renders as
  **"declared 0 (flat"** — the words "objective)" are past the region's right edge. That cell is the whole point of the
  constant-column preset, and the specification required exactly this state to be legible ("a constant centered feature
  remains valid and selects the declared zero representative").
- `regularization-figure-7-desktop.png`: the right panel's third column header renders as **"edge differ"** and the
  panel caption as "only one sp | it".

Both live in `role="region" tabIndex={0}` scroll containers, so the content is reachable by keyboard and by scrolling —
this is friction, not inaccessibility, and the browser verifier's collision check is SVG-only so it could not have
caught it. But at 1366 px, on a wide screen, a reader sees a truncated sentence with no visible scrollbar cue.

Fix: allow `white-space: normal` on the last column of these two tables, or shorten the cell to "declared 0 — flat" and
the header to "edges".

**S4. Negative numbers print with an ASCII hyphen in two places, and the design record claims they do not.**

Locations: `src/learn/data/topics/regularization-l1-l2-elastic-net-dropout.jsx` L314;
`src/learn/components/lesson-labs/RegularizationFigures.jsx` L669.

Evidence:
- Practice 1's solution is `Ridge is −2.4/1.6={scalarSolution(-2.4, 0.6, 0).coefficient.toFixed(1)}. Lasso is
  {scalarSolution(-2.4, 0.6, 1).coefficient.toFixed(1)}. Elastic net is (−2.4+0.3)/1.3=−21/13≈{scalarSolution(-2.4,
  0.6, 0.5).coefficient.toFixed(6)}.` The literal text uses U+2212 and the three interpolated values use
  `Number.prototype.toFixed`, which emits U+002D. The rendered sentence is "Ridge is −2.4/1.6=-1.5. Lasso is -1.8.
  Elastic net is (−2.4+0.3)/1.3=−21/13≈-1.615385." The file's own `num()` helper (L15) exists for exactly this and is
  not used here.
- F8's table row is `[record.model, record.logLikelihood, record.parameters, record.fit, fixed(record.aic, 4),
  fixed(record.bic, 4)]`, so the log-likelihood column prints `-150` and `-146` while §9's own table two elements above
  hard-codes `'−150'` and `'−146'` (`.jsx` L287–288). The same two numbers appear with two different minus signs on one
  screen. Visible in `regularization-figure-8-desktop.png`.

The design record's repair note 11 states: "**Everywhere** — printed numbers used a hyphen where the prose uses a
typographic minus. Every number the labs and figures print now uses U+2212." F8 is a figure, so that claim is false as
written.

Fix: wrap the three practice values in the file's existing `num()`, and the F8 column in `round()` from
`RegularizationShared.jsx`; then correct repair note 11 or restate it as "every number the labs and figures *format*".

**S5. Investigation 3's verdict reads "a change of no change from the recorded row" in its default state.**

Locations: `RegularizationLabs.jsx` L355 (`changeText`) and L393 (`describe`).

Evidence: `changeText = digits => (change === 0 ? 'no change' : `${signed(change, digits)} dB`)` is substituted into
`…dB, a change of ${changeText(6)} from the recorded row.` On the unedited baseline row — the first state any learner
sees — this renders "The prediction is 124.465305401 dB, **a change of no change** from the recorded row." Confirmed in
`regularization-airfoil-desktop.png`. The same helper reads correctly at L428 ("so this is no change").

This is worth fixing beyond the grammar: the sentence is the page's statement about its own floating-point agreement
tolerance, which is a real and well-handled honesty point (the repair note explains that an unchanged row was printing
`−2.842 × 10⁻¹⁴ dB` before the 10⁻⁹ tolerance was introduced), and it deserves a clean sentence.

Fix: `a change of ${changeText(6)}` → `${change === 0 ? 'unchanged from' : `a change of ${signed(change, 6)} dB from`} the recorded row`.

**S6. The design record's "Departures from the manuscript" list is materially incomplete.**

Location: `docs/teaching/drafts/regularization-l1-l2-elastic-net-dropout/design.md` L115–131 (six numbered items).

Not recorded, at least: (a) the "Calculate without recording a prediction" action in all four investigations, which
contradicts a binding contract line (**S1**); (b) F4's missing per-term selector (**S2**); (c) the rewording of "The
equivalent full calculation ran serially in the author environment with no warnings" into a claim about the displayed
program itself; (d) I3's raw-measurement control ranges being a wider physical band than the observed columns the
specification named as the control ranges (**O3**); (e) the readiness-check table added in §11; (f) the closing
"constructed calculations" paragraph added after the Sources block; (g) the `rawFeatureNames` export that reaches
neither the page nor any verifier (**O7**). Most are improvements or harmless; the record is what lets the next author
tell a deliberate choice from a slip, and it is the only place several of these can be found.

**S7. The one dropout state the contract insists on most is asserted but never photographed, and its screenshot filename points at a different state.**

Locations: `scripts/verify-regularization-browser.cjs` L281–295;
`docs/teaching/evidence/screenshots/regularization-dropout-practice-desktop.png`.

Evidence: the I4 contract says "Its branch probabilities change with q; **there is no need for a misleading four-state
uniform average when q is not one-half**" and "Do not average branch values uniformly when q ≠ .5". The verifier loads
the q = 0.75 practice fixture, asserts `expected noisy half-squared loss 1.166666667`, asserts the zero-coefficient
sentence, and asserts that the four branch probabilities are not all equal — and then clicks **"Null: change the target
to 3"** before taking the screenshot. The file named `regularization-dropout-practice-desktop.png` therefore shows the
q = 0.5 target-shift null, not the practice fixture. I verified this by opening the image: it shows q = 0.5, y = 3, four
0.25 probabilities and expected loss 4.5. **No image in the evidence set shows non-uniform branch probabilities.**

Other gaps in the same pass: no narrow capture of the coordinate lab (the densest tables on the page, and the one that
would have caught **S3**), and no narrow capture of figures 3, 5, 6, 7 or 8.

Fix: move the screenshot call to before the null click and add a second capture after it; add a 390 px capture of the
coordinate lab.

### Observation

**O1. The keep-probability floor is wider than the contract's.** `regularization-models.js` L28 sets
`keep: { minimum: 0.05, maximum: 1 }`; the I4 contract says `q ∈ [.1, 1]`. Wider, still strictly positive, and the
guard at L372 refuses `q ≤ 0` regardless. Harmless, but undeclared.

**O2. Control ranges are not visible until you break them.** `NumberField` (`RegularizationShared.jsx` L62–89) sets
`min`/`max` attributes and surfaces "Keep it between −4 and 4" only as an error after an invalid entry. I3 is the
exception and does it well: its observed ranges are in a caption and in a table column. I1, I2 and I4 carry no
persistent range or unit beside their controls.

**O3. I3's control bounds are a wider physical band than the observed columns.** `limits.raw`
(`regularization-models.js` L35–41) allows frequency 100–25,000 Hz against an observed 200–20,000, chord 0.01–0.4 m
against 0.0254–0.3048, and so on. The I3 contract says "use observed column ranges as default control ranges". The page
discloses both the observed ranges and the hypothetical-scenario caveat, so nothing false is shown, but the departure
is undeclared. Related: `decimals[4] = 7` (`RegularizationLabs.jsx` L330) means the observed displacement minimum
`0.000400682` (nine decimals) cannot be typed into the field that reports it as the range floor.

**O4. F4's "unpenalized OLS" annotation lands on the row of the "20" y tick.** `RegularizationFigures.jsx` L288 places
the label at `liftFull(17.35) − 7 ≈ y 113`, and the "20" tick label sits at `y ≈ 112`. They do not overlap — the tick is
right-anchored at x = 46 and the label starts at x = 56 — but in both `regularization-figure-4-desktop.png` and
`-320.png` they read as one run of text, "20 unpenalized OLS". The design record's repair 4 says this label was moved
"off the data line"; it landed on the tick row instead.

**O5. F3's three diagonal markers are crowded.** The lasso midpoint (0.5, 0.5), elastic net (0.6, 0.6) and ridge
(2/3, 2/3) span 0.167 units on a 1.8-unit axis, about 19 px apart at rendered size
(`RegularizationFigures.jsx` L184–190). In `regularization-figure-3-desktop.png` the elastic-net dot is hard to separate
from the lasso midpoint. The five-row table below carries every value, and the labels are placed on opposite sides, so
nothing is unreadable — but the visual distinction between "a point on the segment" and "a point off it" is the
figure's whole argument and it is at the limit of resolution.

**O6. Trailing-zero stripping makes decimal places vary row to row.** `round()`
(`RegularizationShared.jsx` L25) does `value.toFixed(digits).replace(/0+$/, '')`, so a column can print `0.2` beside
`0.058824`. The implementer correctly used `fixed()` for the columns that must align (F2's budgets, F4's fold MSEs,
F5's multipliers), so this is confined to the readouts and the threshold table.

**O7. Surplus exports.** `rawFeatureNames` (`regularization-data.js` L50) reaches neither the page nor any verifier —
only the generator emits it. `selected` (L150) and `elasticNetRatio` (L81) reach only
`verify-regularization-models.mjs`; `selected` carries all three final refits (three × 20 coefficients + 20 means + 20
scales), of which only the ridge one is used on the page, and that through the separate `ridgeModel` export. In
`regularization-models.js`, `checkFinite`, `checkRange`, `softThreshold`, `penaltyMeasure`, `penaltyBoundaryRadius`,
`curvatures`, `penalisedObjective`, `solveLinear` and `polynomialTerms` are exported although the page reaches them only
through other functions in the same module; they are exercised by the verifier, so this is naming rather than payload,
except for `selected`, which is real bytes.

**O8. No page-visible statement that the airfoil fits are version-sensitive.** `regularization-data.json` records
"Numerical fitting can differ on other library versions" as an explicit limitation and the generated module's header
comment names the three pinned versions, but the only version text a reader sees is the `pip install
numpy==2.3.5 scikit-learn==1.9.1` line inside §5's program setup, which attaches naturally to the program rather than to
the eighteen-row table and the 20-term inference fixture five elements later. The manuscript has no such sentence
either, so this is not an implementation infidelity — but the GMM lesson lost the same sentence and it was a finding
there. One clause in §5 would close it.

**O9. The Zou–Hastie annotation inverts the paper's own emphasis.** `.jsx` L379 (and `lesson.md` L661, identically):
"sections 2–3 explain grouping and the original paper's distinction between a mixed-penalty estimate and its historical
rescaled variant." In the paper, the *naive* elastic net is the deficient one (§3.1) and the `(1+λ₂)`-rescaled estimate
is the corrected, recommended one (§3.2). Calling the rescaled version "historical" is defensible given the very next
clause ("Current `ElasticNet` follows its documented mixed objective; do not add the paper's extra rescaling to a
library prediction automatically"), and the practical advice is right, but a reader who goes to the paper will find the
adjectives the other way round. Manuscript-level; no implementation edit needed.

**O10. `explore()` does not clear the recorded choice.** `RegularizationShared.jsx` L124–127 sets `choice: ''` on the
*result* but leaves `state.choice`, so a learner who selects a radio and then presses "Calculate without recording a
prediction" sees their selection still filled and disabled beside a verdict that says "Calculated without a recorded
prediction". Nothing false is claimed, but the screen shows a prediction that was not graded next to an answer.

**O11. F6's stated penalty for the same-prediction alternative is not visible text.** The contract asks F6 to "show
same-prediction alternative (2, .5), penalty 1.0625". The value appears in the panel's `aria-label`
(`RegularizationFigures.jsx` L500) and is derivable from §8's "(2,0.5) costs 4.25λ" at λ = 0.25, but no sighted reader
sees `1.0625` beside the marked point. The `(1, 1)` total *is* in the four-row table.

**O12. One table cell is a hard-coded string rather than a computed value.** `.jsx` L20:
`z === 3 ? '5/3' : num(scalarSolution(z, strength, 0.5).coefficient)`. It matches the manuscript exactly and the
computed value (1.666666667) appears elsewhere, so nothing is wrong; it is simply the one number in the body's tables
that the model does not produce.

**O13. The blueprint's `sequence` lists twelve items for eleven sections.** It is an idea sequence rather than a section
list (§7 contributes two entries), so nothing resolves incorrectly — noted only because a reader comparing it with
`headings` will find an off-by-one.

**O14. Reading-time estimates are unverifiable.** `readTime: '~65 min first pass · ~120 min complete read + 70–110 min
code and practice'` was not measured by me or, as far as the evidence shows, by anyone.

## Closure

The mathematics and the data handling of this lesson are, as far as I can establish, **flawless**. Every one of 368
constructed values I recomputed from first principles agrees to the full stated precision — the soft-threshold recovered
by scanning the objective rather than by the formula, the one-sided slopes by finite differences, the attained budgets
and the constrained contact point by an independent scan of the diamond, the mixed-penalty level set by its own
quadratic at 37 directions, the whole 18-sweep and 29-sweep duplicate histories from my own coordinate solver, the
duplicate-segment tie, the exact dropout enumeration against its analytic twin at three fixtures and two nulls, the
ridge and early-stopping filters with their separately matching λ = 9 and 6, AIC and BIC, the two-factor optimum against
a 900,000-point brute-force scan of the surface, the denoising system, the bit code, and all ten practice answers. The
airfoil campaign reproduces bit-exactly from the raw `.dat`: the split, the folds, all 18 candidates with 54 fold fits
and 1,080 coefficients, both baselines, the three selected refits, and an inference fixture whose twenty signed
contributions match the page to six decimals, term for term. All three displayed programs reproduce their published
output exactly, and two of them are byte-identical to the manuscript's code. Every external URL resolves to the
document it claims, with the right author initial on the Tibshirani paper and the right section numbers in Grünwald —
the two places a lesson like this most often gets it wrong. The data boundary is kept absolutely: no reserved row is
predicted, scored, shipped or indexed anywhere in the page, the module or the packet.

The interaction design is the second thing worth crediting. A prediction is graded against the draft committed with it,
never against what happens to be on screen; any edit retires both the prediction and its feedback; presets fill inputs
and never the outcome; an exact zero is said in words rather than shaded; and the dropout investigation answers the
stochasticity problem by refusing to sample at all and enumerating the expectation instead.

What needs work is one sentence, one record, and a handful of presentation items. A figure paragraph describes a picture
that was deliberately redrawn to be different, contradicting its own legend and its own accessible description (B2); the
topic note declares the work closed while the authoritative ledger declares it not started (B1); every investigation
offers a one-click way around the contract's "require a prediction before Apply" (S1); F4's coefficient panel has no way
to follow a single term and no vertical scale (S2); two cells that carry a teaching point are clipped at desktop width
(S3); three practice values and one figure column print the wrong minus sign, which also falsifies a recorded repair
claim (S4); the airfoil verdict reads "a change of no change" in its default state (S5); the departures list is missing
at least four real departures (S6); and the one dropout state the contract insists on is asserted but never
photographed (S7).

Recheck plan after the fixes: **no numerical verifier needs to be repeated for any finding in this review.** B1, S6 and
O8/O9 are documentation and prose. B2, S4, S5 and O11 are string edits in `RegularizationFigures.jsx`,
`RegularizationLabs.jsx` and the topic body; after them re-run `node scripts/verify-regularization-models.mjs` only
because it hashes the source files, and re-capture `regularization-figure-5-desktop.png` and
`regularization-figure-8-desktop.png`. S1, S2 and O10 are component changes; after them re-run
`node scripts/verify-regularization-browser.cjs` (its layout pass currently *uses* the explore button, so that block
will need adjusting) and capture two images that would have caught the defects: an investigation immediately after an
ungraded calculation, and F4 with one term selected. S3 is a CSS or copy change needing fresh captures of
`regularization-coordinate-constant-desktop.png` and `regularization-figure-7-desktop.png`. S7 needs only a reordered
screenshot call plus one new narrow capture.
