# Calibration & Conformal Prediction — independent phase-two review

Reviewed 19–20 September 2026 against the working tree plus the uncommitted Calibration implementation. The
reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/`,
`docs/teaching/drafts/` or `docs/teaching/topic-notes/` was edited; this review document is the only write outside
`scratch/`. Throwaway scripts live under `scratch/cal-independent/`. No state-changing git command was run. The
build directory `dist-cal/` was deleted and the preview on 4193 stopped at the end.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/cal-independent/`, importing neither `calibration-models.js` nor `calibration-data.js` nor `calibration-examples.js` nor any `verify-calibration-*` script nor the packet's `calibration_calculations.py` / `uncertainty_experiments.py`. Run with `scratch/lesson-tools/Scripts/python.exe` (NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.1, scikit-learn 1.9.1, mpmath) and with Node for the JavaScript-semantics work. **54 exact-rational checks** in `fractions.Fraction` of every finite constructed number in the manuscript, the practice solutions and the lesson body — zero failures. **260,040 (n, α) rank cases** decided by BigInt rational arithmetic against both the float recipe and a textual replica of the shipped integer routine. A **60-digit mpmath damped-Newton** re-solve of the Platt objective with an exact gradient test. A full independent refit of **both** pipelines from the **served** CSVs. A reviewer-written Playwright drive (`drive.cjs`, `geom.cjs`) sharing no code with `verify-calibration-browser.cjs`, at 1366, 390 and 320 px on Edge. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `checked-results.json`, both programs); the destination note; the lesson body, model layer, both generated modules, all three lab/figure/shared components and the CSS; the blueprint; the preset guard in `verify-calibration-models.mjs`; the served assets. |
| **Reused (declared)** | scikit-learn's `SVC`, `CalibratedClassifierCV(FrozenEstimator(...))`, `Ridge` and `GradientBoostingRegressor` at the declared hyperparameters and seeds, because the protocol *is* defined as those calls. NumPy/SciPy/mpmath as general libraries. Conformal ranks, thresholds, Brier, AUC, reliability bins, ECE, PAV and every set/interval construction were written from the definitions rather than reused. |
| **Not re-derived by me** | The falsification harness. The coordinator ran `scripts/falsify-calibration.mjs` during this review — 23 of 23 caught, 21 citing the guard aimed at, tree restored, four verifiers green afterwards — and I took that as given rather than re-running it. The four verifier counts in §9 below were likewise confirmed by the coordinator, not instrumented by me; my own verifier work was confined to reading the preset guard and proving its domain (B2). A subagent tasked with a full inert-guard audit of all six verifiers died to a spend limit before reporting, so **the systematic "assertions that cannot fail" sweep the brief asks for was not completed**. See "What remains unchecked". |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/calibration-conformal-prediction.jsx` | `5aebb72e02618ccd2247e494c94639e0595c45de845dc587cdefece507164eb1` |
| `src/learn/data/calibration-models.js` | `fd730020fea3ea2bb66ffd720a4d850b83421a4ad063d28382e8de4cc2979a89` |
| `src/learn/data/calibration-data.js` | `74b58179686799f4614602c4d19656f31cc792cd0df9f17f1b06a1c259f94a51` |
| `src/learn/components/lesson-labs/CalibrationLabs.jsx` | `9da73c1f1397c565b6ec1af1d4e01e7b80c5457ca2ad4a24c6c57787c3ec2468` |
| `src/learn/components/lesson-labs/CalibrationFigures.jsx` | `8b92ab7dc965c6e49ce0f5404b981bac3cb913f139ae43e834c850a3a4b457fc` |
| `src/learn/components/lesson-labs/calibration-labs.css` | `f58f3a06a7eaf0d28e8215e231dd73d3f0e28e72821e13577ca462d9569ebee5` |
| `scripts/verify-calibration-models.mjs` | `1fcd4aac680b953ebddc52fca729edb3b8915cad988c3393ba80470b39088a8c` |
| `docs/teaching/drafts/.../lesson.md` | `bdb8848ff1e47c3f4c7e1292bce29154c4fe684a37a197acc856935dbcf4fd81` |
| `docs/teaching/drafts/.../design.md` | `973f43634de0844b9aee4f6bd5fa6b83678d7fa7823995b7b65a8e0ecc392263` |
| `docs/teaching/topic-notes/calibration-conformal-prediction.md` | `25d5dfa907516d389c1fb56c1160db32cf895db30889ac8ee7b48ebc6c6eb7aa` |

---

# Part A — correctness

## A1. Every finite construction, in exact rational arithmetic — **no disagreement**

`scratch/cal-independent/exact.py` and `conformal.py`. Exact `Fraction` throughout; nothing compared at floating
tolerance where a rational answer exists. **54 checks, zero failures.**

- **§1 conditioning fork.** Mean top confidence and mean correctness are both exactly 4/5 for the (.2, rate .3) /
  (.8, rate .9) pair, so the page's "a confidence-only diagram looks perfect" is exact, not approximate.
- **§1 decision and resolution.** Threshold exactly 1/4; coarse-score expected cost exactly 16; informed policy
  exactly 14. Brier of the constant .2 score exactly 4/25, of the exact group rates exactly 3/20 — the .16-versus-.15
  contrast is a rational identity.
- **§2 ten-card sample.** Bin fractions 2/5 and 3/5; two-bin ECE exactly 1/5; merged-bin ECE exactly 0 with both
  means exactly 1/2; Brier 7/25 falling to 6/25; AUC exactly 3/5 **before and after** the replacement, so the
  "ordering is unchanged" claim is exact rather than a coincidence of rounding.
- **§3 isotonic.** The eight-score PAV fit is exactly `[0, 1/3, 1/3, 1/3, 1/2, 1/2, 1, 1]`. The tie fixture
  `[-2,-2,0,1]` / `[1,0,0,1]` pools to `[1/3, 1/3, 1/3, 1]`, i.e. `[1/3, 1/3, 1]` at the three distinct knots.
  Practice 2's `[0, .5, .5, .5, .5, 1]` reproduces.
- **§3 temperature.** Softmax of `[3,1,0]` at T=1 is .843794734…, .114195199…, .042010066…; at T=2,
  .628531719…, .231223898…, .140244383… — the six-decimal values the manuscript prints.
- **§5–§6 conformal.** The nine-score example (k=8, q=.6; k=10, q=∞), all three rows of the class-set table
  including the empty one, the normalised-scale fixture (q_abs=6, q_norm=2, intervals [8,12] / [14,26] / [4,16]),
  the scale-doubling null, the 12-and-16 edit raising q from 2 to 4, and the ten-card rotation (**exactly 8 of 10**
  covered at α=.2, **10 of 10** under full ties).
- **Practices 1, 2, 3, 4, 5.** All exact, including P3's k=12/q=12 and k=15/q=∞, and P5's k=4, q=−2, [12,18].
- **§8 Beta.** k=73, n+1−k=8, mean exactly 73/81 = .9012345679…, matching the Beta(73, 8) statement.
- **§7 quantile conventions.** `np.quantile(linear, 8/9)` = .6333333333333332 and `method="higher"` = .9, against
  the eighth order statistic .6 — all three distinct, as the manuscript claims.
- **§7 tie boundary.** With p = 103/240, `(1 - p) <= q` is **true** and the rearranged `p >= 1 - q` is **false** in
  IEEE double: `1 - q` is .4291666666666667 while `p` is .42916666666666664. The manuscript's warning is exactly
  right and is not a hypothetical.

I found **no disagreement with any constructed number** in the manuscript, the practice solutions or the rendered
lesson body.

## A2. The conformal rank, derived and enumerated — **implementation correct; the record about it is not**

`scratch/cal-independent/rank.mjs` … `rank5.mjs`. I derived the finite-sample rank from the exchangeability
argument independently (among n+1 exchangeable scores the new one is equally likely to take any rank, so
Pr{rank ≤ k} = k/(n+1) ≥ 1−α forces k = ⌈(n+1)(1−α)⌉), implemented it in BigInt rational arithmetic, and compared
three ways: exact, the naive float recipe, and a **textual replica of the shipped integer routine** (`decimalParts`
plus `Math.floor((scaled + denominator − 1) / denominator)`), written out rather than imported.

**The shipped computation is right at every point I could reach.** 199,800 cases (n = 1…200 × every three-decimal
α in (0,1)) and a further 60,240 six-decimal cases: **zero mismatches** against exact rational arithmetic. The
intermediate `(n+1)(10^d − numerator)` peaks at 200,999,799 for the extreme admissible input, three orders of
magnitude inside `Number.MAX_SAFE_INTEGER`, so the integer path cannot silently lose precision. `String(0.440)`
normalising to `"0.44"` does not perturb it. This is a genuine strength and the mechanism is sound.

**The float recipe is wrong far less often than recorded.** See S1: on the grid the controls actually admit there
is exactly **one** disagreement, not 26.

## A3. The Platt sigmoid, adjudicated at 60 digits — **the recorded tie is not a tie**

`scratch/cal-independent/sigmoid.py`, `sigmoid2.py`. I re-solved the smoothed objective with an mpmath damped
Newton at `mp.dps = 60`, from a different start, and tested stationarity with an exact 60-digit gradient.

The minimiser is

```
a* = 0.2677017730355834493948659937111045816716
b* = -0.1338508865177917246974329968555522908358
```

and `b* = -a*/2` holds to 62 digits — structurally, because the eight scores are symmetric about ½, so with
`b = -a/2` the logits are antisymmetric and the offset gradient vanishes identically. That is a closed-form check
on the answer, not just a numerical one.

Gradient norms at the three candidate points:

| point | a | ‖∇‖ (60-digit) |
| --- | --- | --- |
| exact minimiser | 0.26770177303558344939… | 3.1 × 10⁻⁴¹ |
| builder's damped Newton | 0.26770177303558 | **3.9 × 10⁻¹⁵** |
| frozen SciPy BFGS value (`calibration-data.js:3034`) | 0.2677017730874301 | **6.7 × 10⁻¹¹** |

The builder's own solver is correct to 3.4 × 10⁻¹⁵. The frozen SciPy value is wrong by 5.2 × 10⁻¹¹. This is not two
optimisers stopping at different last digits of a flat minimum — it is one converged answer and one under-converged
one, and the gradient separates them by four orders of magnitude. The objective gap is only 2.0 × 10⁻²¹, which is
why the builder's 6,561-point grid could not tell them apart; a grid search on a quadratic minimum is the one test
that cannot. See **B1**.

**The rendered page is not affected.** It prints `a ≈ 0.267701773` and `b ≈ −0.133850886` at nine decimals, and
both optimisers agree to nine decimals. The page also names the solver in situ ("fitted by SciPy BFGS on the
smoothed log-loss objective, recorded by the content packet"), which is honest. The defect is in the manuscript's
tenth decimal and in the record's adjudication, not in what a learner sees.

## A4. Both pipelines, refit from the served CSVs — **no disagreement**

`scratch/cal-independent/pipelines.py`. Reads `public/learn-assets/calibration/*.csv` (not the packet copies — I
confirmed the two are byte-identical by SHA-256 first). My own rank, threshold, Brier, tie-aware AUC, set
construction and coverage counting; scikit-learn only for the estimators the protocol names.

**Classification (banknote).** 480 rows, 480 distinct `source_row`, class counts 275/205; splits 240/80/80/80;
training prior exactly 103/240; my exact rank on n=80 at α=1/10 is **k = 73**. All five rows reproduce:

| procedure | correct/80 | Brier | covered/80 | mean set size | empty/singleton/both |
| --- | ---: | ---: | ---: | ---: | --- |
| naive sigmoid of SVC score | 79 | .060271 | 69 | .8625 | 11/69/0 |
| held-out sigmoid | 79 | .007839 | 71 | .8875 | 9/71/0 |
| held-out isotonic | 79 | .004766 | 76 | .9500 | 4/76/0 |
| held-out temperature | 79 | .008521 | 69 | .8625 | 11/69/0 |
| training-prior constant | 51 | .235538 | 80 | 2.0000 | 0/0/80 |

Every cell matches the manuscript. AUC = 1 for all four SVC-derived scores and exactly 0.5 for the constant, as the
page states. Isotonic's threshold is exactly 0.0, confirming the tie explanation. The constant baseline's q is
0.5708333333333333 — the float image of 137/240 — which is the tie-boundary fixture.

**Regression (airfoil).** 480 rows, splits 240/120/120, all ids distinct. My exact rank on n=120 is **k = 109**.
q_cqr = **0.7663505895** to ten decimals. All four rows reproduce: constant 100/120 at 18.991917 dB, ridge 108/120
at 16.370437, raw quantiles 93/120 at 14.353792, CQR 103/120 at 15.886494. The declared slices reproduce exactly:
ridge covers **54/66 below 2,000 Hz and 54/54 at or above**. Ridge point MAE 3.939045, constant 5.764865. Quantile
crossings before rearrangement: 0 on calibration and 0 on test — so the rearrangement step is a correctly declared
safeguard that happens not to bite on this corpus, which the page does not overclaim.

**The drawn reliability diagram matches an independent recount.** `scratch/cal-independent/bins.py`: the held-out
sigmoid bins on `[0, .2, .4, .6, .8, 1]` are **50, 0, 1, 1, 28** (sum 80) with mean forecasts .017087, —, .595559,
.718261, .943145 and observed fractions 0, —, 0, 1, 1. Every dot in figure 10 sits where my recount puts it, the
empty bin carries no dot, and the count rail prints 50 / 0 / 1 / 1 / 28.

## A5. What I checked on the rendered page and found clean

Production build previewed on 127.0.0.1:4193, driven at 1366, 390 and 320 px on Edge. **I opened the images and
looked at them**, not only the DOM.

- **Nothing revealed on first paint.** `.cal-graded` count is 0 at all three widths.
- **Zero console errors, zero page errors, 169 KaTeX nodes, zero `.katex-error`.**
- **CSS collapsing geometry — clean.** No `<svg>` renders below 1 px in either dimension; no KaTeX radical has zero
  height. This is the mechanism that made a sibling lesson publish a radius squared, and it is not present here.
- **Paint order — clean.** Zero `<text>` elements covered more than 40% by a *later*, opaque sibling `<rect>`,
  `<path>` or `<circle>`. The Phase C fix for "count-rail bars drawn over the tick labels" is genuinely present: I
  can see the labels in figure 10 and the bars sit below them.
- **CSS beating a presentation attribute — clean.** Zero non-colour presentation attributes (`width`, `height`,
  `font-size`, `opacity`) disagree with their computed value anywhere in the page's SVG.
- **Prose outside its viewBox — clean.** Nothing is painted outside its own `<svg>` box in screen coordinates. My
  first pass flagged seven `<text>` nodes with negative raw bbox x; all seven carry `rotate(-90 …)` and are
  correctly placed once the transform is applied (see O7). The Phase C fix for "prose running 100 px outside its
  viewBox" holds.
- **The 80 conformal calibration scores are on their own rail**, tagged as a calibration-sample quantity, with rank
  73 marked at q ≈ .0963. Verified in the image, not only in the DOM.
- **Coverage language.** 25 percent-shaped figures on the page, each read in context. Every *measured* coverage
  figure carries its denominator ("Coverage here is 71 of 80 for that procedure, below the 90% the rank was chosen
  for"; "9 empty sets and 71 singletons"; the size table's 0/9, 1/71, 2/0). Every *population* figure is tagged
  "EXACT CONSTRUCTED POPULATION" and correctly carries no denominator, because it is not an estimate. Figure 10's
  three-level table is present with `Beta(73, 8)` and its conditions. I found no place where a coverage number is
  offered as a conditional or future guarantee; the page repeatedly says the opposite, in the right places.
- **Responsive.** `document.scrollWidth === clientWidth` at 390 and 320 px; the page does not scroll horizontally.

---

# Part B — findings

## Blocking

### B1. The recorded sigmoid adjudication is false, and the manuscript's tenth decimal is wrong

**Where.** `docs/teaching/drafts/calibration-conformal-prediction/design.md`, departure 3 (lines 137–150);
`docs/teaching/drafts/calibration-conformal-prediction/lesson.md` §3 (the sentence "it obtains a≈.2677017731 and
b≈−.1338508865"); the frozen value at `src/learn/data/calibration-data.js:3034`.

**What is wrong.** The record states that the SciPy value and the builder's damped-Newton value are "**both
stationary to within 10⁻¹⁴**" and that "**neither is wrong** — they are two optimisers stopping at different last
digits of the same flat minimum." Neither half holds. At 60-digit precision the gradient norm at the SciPy point is
**6.7 × 10⁻¹¹**, not below 10⁻¹⁴; at the builder's own Newton point it is 3.9 × 10⁻¹⁵. The minimiser is
a* = 0.26770177303558344939…, so the builder's solver is right to 3.4 × 10⁻¹⁵ and the frozen SciPy value is wrong by
5.2 × 10⁻¹¹.

Consequently the manuscript's ten-decimal figure is wrong in its last digit: a* rounds to **.2677017730**, and
.2677017731 is the rounding of the under-converged value. (b ≈ −.1338508865 is correct, and `b* = −a*/2` exactly.)

**Why it matters.** Three reasons, in increasing order. First, a frozen manuscript states a wrong digit at the
precision it chose to quote. Second, the record's own summary line — "Every number the packet records was confirmed.
One disagreement is recorded below; it is a difference between two optimisers, not an error in the packet" —
certifies as confirmed the one number that is not. Third, and most instructive: the builder had the correct answer
in hand from its own solver and discarded it in favour of the packet's, on the strength of a grid search that
*cannot* discriminate between two points 5 × 10⁻¹¹ apart on a quadratic minimum (the objective gap is 2 × 10⁻²¹,
far below double precision). The right test — the gradient — was available and would have settled it immediately.
The stated reason for freezing, "displaying the live value would contradict the manuscript's tenth decimal", is
also moot: the page displays nine decimals, at which the two agree.

**How I verified it.** `scratch/cal-independent/sigmoid.py` and `sigmoid2.py`, mpmath at `mp.dps = 60`, damped
Newton from a different start than the builder's, converged to `|Δ| < 10⁻⁵⁰`; gradient evaluated symbolically in
60-digit arithmetic at all three candidate points; the exact relation `b* = −a*/2` derived from the score set's
symmetry about ½ and confirmed to 62 digits.

**In fairness.** The rendered page is not wrong — a learner sees `0.267701773`, which is correct. The disclosure
that a frozen value is displayed for the frozen fixture is honest and prominently made. What fails is the
mathematical adjudication, not the honesty of the disclosure.

### B2. The preset-built-from-draft defect survives in investigation 3, and the guard written for it cannot reach it

**Where.** `src/learn/components/lesson-labs/CalibrationLabs.jsx` lines **765–770** (the two rotation-mode preset
buttons). The guard is `scripts/verify-calibration-models.mjs` lines **1632–1641**. The record claim is
`design.md` Phase C item 8.

**What is wrong.** The record states: "All four investigations had the same shape. Presets now build from the
applied state, so the change a label names is the change that gets graded, **and a static guard refuses a preset
block that reads the draft**." The tree contradicts both clauses.

Investigation 3's rotation branch still builds its suggested states from the **draft**:

```jsx
<button type="button" onClick={() => state.suggest({
  ...draft, rotationScores: fixtures.tiedRotationScores.slice(),
})}>Make every card .2 — all ties</button>
<button type="button" onClick={() => state.suggest({
  ...draft, rotationScores: fixtures.rotationScores.slice(),
})}>Back to ten distinct cards</button>
```

Every other preset in the lesson correctly reads `state.active` (investigation 1 at line 122, investigation 2 at
404, investigation 3's *set* branch at 670 via `const from = state.active`, investigation 4 at 988 via
`const from = active`).

The guard cannot see these two. Its domain is `matchAll(/const presets = \[([\s\S]*?)\n {2,4}\];/g)` — it inspects
only the bodies of `const presets = [...]` arrays. The two offending calls are inline JSX buttons, in no such array.
I instrumented this directly: the regex finds exactly 4 preset blocks (lines 122, 404, 670, 988), **all four clean**;
there are 6 `state.suggest` call sites in the file; the only two that spread `...draft` are at lines 765 and 768 and
**neither lies inside any guarded block**. The guard passes, reports `record('preset baseline')` four times, and the
defect stands. This is the brief's "guard whose domain excludes the defect it was written for", in its purest form:
the regex is alive, the subject set is non-empty, and it still cannot fire on the one instance that exists.

**Why it matters.** `state.suggest` sets the draft and retires the verdict; the verdict then grades draft against
*applied*. If a learner edits α or a rotation card, then clicks "Back to ten distinct cards", the committed change
includes the α edit the button did not name, and the verdict attributes the whole difference to the preset. The
learner impact here is **moderate rather than severe** — neither rotation button is labelled "an exact null", and
the caption that promises "A suggested setup makes one change to the state that is currently *applied*" is rendered
only inside the `set` branch, so the explicit promise is not made in rotation mode. What is blocking is not the
learner harm but the pair of false assurances: a record asserting a property the tree does not have, and a guard
whose green result is being read as proof of that property.

**How I verified it.** `node` instrumentation over the source, reported above; read of the guard at
`verify-calibration-models.mjs:1632–1641`; read of all six `state.suggest` call sites.

## Should-fix

### S1. "26 points of the grid these controls admit" — only one of the 26 is reachable, and the stated failure mode never occurs

**Where.** `design.md` lines 205–212.

**What is wrong.** Two claims in one paragraph.

*The count.* The record says the float recipe "returns the wrong integer at **26 points of the grid these controls
admit**, including n=24 at alpha=.44". The 26 is real but it is the count over the **verifier's scan range**,
n = 1…200 × 491 α values = 98,200 pairs — which is also exactly where the record's own "98,200" comes from. The grid
*the controls admit* is much smaller: `controlSteps.alpha` is min .01, max .5, step .001, and the largest n any
control can produce is bounded by `limits.cards` (30), `limits.pavRows` (24), `limits.calibrationPairs` (24) and
`limits.calibrationScores` (20). Over n ≤ 30 there is exactly **one** disagreement — n=24 at α=.44. The other 25 sit
at n = 49, 74, 99, 124, 149, 174, 179 and 199, none of which any control on this page can reach. My sweep:

| n ≤ | disagreements |
| ---: | ---: |
| 24 | 1 |
| 30 | 1 |
| 50 | 3 |
| 100 | 9 |
| 200 | **26** |

The closing sentence — "recorded in the evidence file so the guard is shown to bite **inside the reachable grid**
rather than in principle" — is therefore twenty-six times stronger than the evidence supports. The guard does bite
inside the reachable grid, once.

*The failure mode.* The record says the n=24, α=.44 case "asks for rank 15 of 14 available scores", and the module
comment at `calibration-models.js:754–755` generalises this to "a rank one too large asks for an order statistic the
calibration set does not have". Neither is true. At n=24 there are 24 scores and rank 15 exists; the exact rank is
14 and the float rank is 15, so the consequence is a *wrong, too-large threshold* — silent over-coverage — not a
missing order statistic. I scanned all 98,200 pairs: the float rank **never exceeds n** at any of the 26
disagreements. The real hazard is worth stating correctly, because "asks for an index that does not exist" is a
failure that would throw or return `undefined`, whereas "returns a quietly wrong threshold" is exactly the kind that
no plot would show — which is the point the record is trying to make.

**Why it matters.** This paragraph is the record's headline justification for the integer rank, and it is the item
the brief singles out. The underlying engineering decision is correct and worth keeping (A2). Overstating its
reach by a factor of 26 and misdescribing its failure mode weakens a record that is otherwise unusually candid.

**How I verified it.** `scratch/cal-independent/rank.mjs`, `rank2.mjs`, `rank3.mjs`, `rank4.mjs` — BigInt exact ranks
against `Math.ceil((n+1)*(1-alpha))` over every sub-grid, plus an explicit `float > n` test across all 98,200 pairs.

### S2. Both doc comments justifying the integer rank cite numerical examples that are false

**Where.** `src/learn/data/calibration-models.js` lines **32–35** (module header) and lines **136–140**
(`decimalParts`).

**What is wrong.** The header says: "`(n+1)*(1-alpha)` in ordinary floating point returns **8.000000000000002** for
n=9, alpha=.25 and a ceiling of **9**, silently asking for a rank that does not exist." In IEEE double, 0.25 is
exactly representable, `1 - 0.25` is exactly 0.75, and `10 * 0.75` is exactly **7.5**. Its ceiling is 8 — which is
also the exact answer. There is no error at this input, the quoted value 8.000000000000002 does not arise, and rank
8 of 9 exists.

The `decimalParts` comment repeats the same fabricated example with a different fabricated value — "`(9+1)*(1-0.25)`
is **7.500000000000001**" — and then states the consequence as "**whose ceiling is 8 rather than the correct 8**",
which is self-contradictory on its face. It goes on: "and for alpha=.35 the same slip produces a rank one too
large". I checked α=.35 at n = 9, 10, 19 and 24: float and exact agree at every one (7, 8, 13, 17).

**Why it matters.** These two comments are the in-code rationale a future maintainer will read before touching the
one function the module calls "the one number on this page that must not be approximated". A maintainer who checks
the cited example will find it does not reproduce, and may reasonably conclude the integer path is cargo cult and
remove it. The correct example was available and is a single line away: n=24, α=.44 gives 15 against the exact 14.

**How I verified it.** `scratch/cal-independent/rank4.mjs`, evaluating the cited expressions in Node at full
precision.

### S3. Low-contrast segment labels in the temperature figure — the Phase C contrast fix covered only the leading bar

**Where.** The stacked temperature figure (SVG index 6 on the rendered page, `viewBox="0 0 300 160"`), drawn by
`CalibrationFigures.jsx`; palette in `calibration-labs.css`.

**What is wrong.** Phase C item 3 records fixing "gold text on the gold winning bar" — the label inside the
*leading* segment. That fix is present and effective: the "A 0.98" / "A 0.844" / "A 0.629" / "A 0.481" labels are
dark on gold and read cleanly. The **non-leading** segments were not covered by the same pass. Three labels —
"B 0.231", "B 0.292", "C 0.227" — are drawn in `rgb(163, 174, 170)` on a `rgb(74, 95, 84)` fill at **9.5 px**, a
contrast ratio of **3.01:1**, below the 4.5:1 that small text requires.

I opened the image and confirmed it visually: the B and C figures are noticeably harder to read than the A figures
in the same rows, and at T=4 "C 0.227" is the least legible element in the figure. The same gold threshold rule also
passes through the "B 0.231" label at T=2.

**Why it matters.** This is the brief's fourth "DOM right, picture wrong" mechanism — right size, correct content,
inside its own box, and no geometry or attribute check can see it. The figure's whole job is to let a learner read
how the non-leading classes gain mass as T rises; those are precisely the labels that are hard to read.

**How I verified it.** `scratch/cal-independent/geom.cjs` computes WCAG relative luminance for every SVG `<text>`
against the topmost *earlier-painted* shape whose box contains the text centre; three failures, all in this figure.
Then `scratch/cal-independent/shots/svg6.png` at deviceScaleFactor 3, opened and read.

### S4. Departure 7's production closure is real but thinner than the note asked for, and the record does not say so

**Where.** `design.md` departure 7; `docs/teaching/topic-notes/calibration-conformal-prediction.md`, Naive Bayes
section, fourth bullet.

**The builder's reading is correct.** I confirmed it independently: the note asks the destination to explain "how
class absence/small calibration samples alter interpretation", and the manuscript does not treat class absence at
all — the only occurrence of "absent" in `lesson.md` is an unrelated sentence about dataset metadata. The
manuscript does treat small calibration samples, repeatedly and well. So the sub-point genuinely is thinner than
the note implies, and disclosing that rather than papering over it is the right call.

**The three production closures exist.** `fitSigmoid` returns `a: null, b: null` with a stated reason when the
sample carries one outcome class (`calibration-models.js:637–647`); `cardQuantity` propagates `rocAuc`'s `because`
so investigation 1 reports the AUC as having no value on a one-class sample (`CalibrationLabs.jsx:54–62`);
investigation 2 offers "Make every label 1 — a legitimate constant fit" (`CalibrationLabs.jsx:427–429`).

**What is thin.** The note asked for an *explanation*. What shipped is *behaviour*. Nothing in the rendered prose
tells a learner that class absence alters interpretation — I searched the full 115k-character rendered page text and
the only occurrences of "one class" are "one class passes" and "one class — not a label with a 90% chance of being
right", neither about absence. A learner meets the closure only by constructing a one-class sample and reading the
`because` string. That is a good way to *encounter* the idea and a poor way to be *told* it, and the record does not
draw that distinction — it presents the production closure as equivalent to what the note asked for.

**Judgement.** Closing it in production rather than in frozen content is **acceptable**: the manuscript was frozen
before this sub-point was reconciled, the alternative is editing frozen content, and a no-value answer with a stated
reason teaches the operational consequence more durably than a sentence would. But the record should say that the
closure is behavioural and discoverable rather than explained, so the next author knows the prose gap is still open.
I did not edit the note.

## Observations

### O1. The rotation verdict states a rank equivalence that is false under ties

`CalibrationLabs.jsx:818–821`. The explain string asserts unconditionally that "a hidden card is covered exactly
when its combined rank among all N is at most `targetRank`". I verified this is exactly right for distinct scores
(covered ⟺ combined rank ≤ 8 for ten cards at α=.2, giving 8 of 10). Under full ties all ten are covered while only
eight can have rank ≤ 8, so the sentence is false in precisely the case the button "Make every card .2 — all ties"
produces. The very next clause does explain the tie behaviour correctly, so a careful reader is not left wrong — but
the two sentences contradict each other with no reconciliation of the first.

### O2. Investigation 4's null explanation attributes the null to the wrong cause in two of three branches

`CalibrationLabs.jsx:1197–1203`. "Double every calibration AND query scale — an exact null" is offered in all three
score branches. In the **normalised** branch the explain text is right and illuminating. In the **absolute** and
**cqr** branches the local scales are not used at all, so nothing changes for a trivial reason; the text instead
says "The threshold is an order statistic, so many numerical changes leave it exactly where it was — only a change
that crosses the selected rank moves it." No calibration score changed, so that is not why. The outcome is correct
and the explanation is plausible, which is the shape of the defect class B2 belongs to, at much lower stakes.

### O3. One percentage appears without its denominator restated

Rendered §7: "Conversely isotonic's 95% does not prove a population guarantee". The denominator (76 of 80) is in the
table immediately above and the sentence is about the *logic* rather than the measurement, so this is defensible —
but it is the one place on the page where a coverage percentage stands alone in a sentence, and the parallel
sigmoid sentence three lines earlier does restate "71 of 80". Worth aligning.

### O4. Three-pixel clip at 320 px

Two visible KaTeX spans (`.base`, `.mord`) extend to x = 323 in a 320 px viewport. `document.scrollWidth` equals
`clientWidth`, so the page does not scroll and the overflow is clipped rather than reachable. 137 of the 139
overflowing nodes at that width are the hidden `.katex-mathml` accessibility layer and are not a rendering concern.

### O5. A one-row count bar is a hairline

Phase C item 2 reports making a one-row bin distinguishable from an empty one. It **is** distinguishable — the empty
bin draws a dashed outline and a one-row bin draws a solid sliver — and the printed counts (50, 0, 1, 1, 28) carry
the information unambiguously. But the visual result is that the dashed *empty* marker is more conspicuous than the
*occupied* one-row bar, which inverts the intended emphasis on a figure whose stated purpose is that "a dot without
its denominator hides that difference". The printed count rescues it; the encoding does not.

### O6. The record's two verifier tables are correctly scoped, and its "Not claimed" section is accurate for the
builder's own work

The Phase A table (538 checks / 69 groups, 76 hygiene checks over 18 files, 1,750 escape sites) and the Phase C
table (543 / 71, 78 over 19 files, 1,836 escape sites) differ, but each is labelled with its phase and the Phase C
numbers match the current tree as confirmed during this review. That is a record keeping its history rather than a
contradiction, and it is the right pattern. Likewise, the "Not claimed" note that the full falsification harness was
not re-run after the Phase C fixes remains **accurate as a statement about what the builder did** — the harness was
subsequently run during this review (23 of 23 caught, 21 citing the guard aimed at), which supersedes the disclosure
without making it dishonest. A one-line update to the record would now be appropriate.

### O7. Two of my own checks produced false positives, recorded so they are not re-raised

Both are traps a future reviewer of these figures will hit. (a) Seven `<text>` nodes report a raw `getBBox()` origin
outside their `viewBox`; all seven carry `rotate(-90 …)`, and `getBBox()` is pre-transform, so the check must be run
in screen coordinates — where nothing is outside. (b) 137 nodes overflow the viewport at 320 px; all are inside
`.katex-mathml`, KaTeX's visually-hidden MathML layer, which has layout but is not painted.

---

# Part C — what remains genuinely unchecked

The brief asked what the trust root does **not** assert, and what a green verifier still permits. Stating this
plainly matters more than the finding count.

1. **I did not complete the systematic inert-guard audit.** The subagent tasked with reading all six verifiers line
   by line for assertions that cannot fail died to a spend limit before reporting. B2 is an inert-guard finding I
   reached by hand, from the one direction the brief pointed me at; it is not evidence that the other guards are
   sound. **The six verifiers have not been swept for dead regexes, empty subject sets or inverted comparisons by
   this review.** That work remains outstanding and is where I would spend the next reviewer's budget first — B2
   demonstrates the class is live in this lesson.

2. **"5,376 of 5,376 leaf paths asserted, none uncovered" is a real mechanism, not a tautology — but it is a
   coverage claim, not a correctness claim.** The comparison helper collects the paths it visits, so the denominator
   is produced by the same traversal that produces the numerator; that is what makes it non-fabricable by a hand
   counter, and it is the right design. What it establishes is that no leaf of the two frozen JSON files escaped
   comparison. It does **not** establish that the values compared against are right — that depends on
   `verify-calibration-data.py` re-deriving them independently, which it does, and which I have now corroborated for
   the two pipelines end to end (A4). The residual risk is a leaf whose independent re-derivation shares a wrong
   assumption with the packet; I found none, but 5,376/5,376 does not exclude it.

3. **Keyboard traversal and screen-reader output were not exercised** — by the builder, who discloses it, or by me.

4. **The screenshots in the evidence directory were read by their author.** I read the live page instead of the
   stored captures, which is a genuinely independent look at the same artefacts but does not audit the stored files.
   I confirmed 51 `calibration-*.png` are on disk, matching the recorded count.

5. **Interaction was not driven.** I verified the investigations by reading their state machine and by confirming
   that nothing is revealed on first paint. I did **not** click through a commit-and-reveal cycle in the browser, so
   B2's consequence is established by code reading rather than observed on screen, and the browser assertion that
   pins a verdict's quoted baseline to the quantity it names (Phase C item 6) is unconfirmed by me.

6. **Only the default fixture states were rendered.** The figures I inspected are those the page paints on load; a
   defect that appears only after a particular edit — the class Phase C item 10 belongs to — would not have been
   seen.

---

# Summary

| ID | Severity | Finding |
| --- | --- | --- |
| B1 | Blocking | The recorded sigmoid adjudication is false: the frozen SciPy value is not stationary to 10⁻¹⁴ (‖∇‖ = 6.7 × 10⁻¹¹) and is wrong by 5.2 × 10⁻¹¹; the builder's own Newton value is correct. The manuscript's `a ≈ .2677017731` should be `.2677017730`. |
| B2 | Blocking | Investigation 3's two rotation presets still build from the draft (`CalibrationLabs.jsx:765–770`), and the static guard written to forbid exactly this scans only `const presets = [...]` blocks, so it cannot reach them. The record claims both are fixed. |
| S1 | Should-fix | "26 points of the grid these controls admit" — one of the 26 is reachable; and the float rank never exceeds n, so "asks for a rank that does not exist" is wrong at all 26. |
| S2 | Should-fix | Both doc comments justifying the integer rank cite float values that do not arise, and one reads "whose ceiling is 8 rather than the correct 8". |
| S3 | Should-fix | Three non-leading segment labels in the temperature figure at 3.01:1 contrast, 9.5 px; the Phase C contrast fix covered only the leading bar. |
| S4 | Should-fix | Departure 7's closure is behavioural, not explanatory; no prose tells a learner class absence alters interpretation, and the record does not distinguish the two. |
| O1–O7 | Observation | Tie-case rank sentence; null explained by the wrong cause in two branches; one bare percentage; 3 px clip at 320 px; hairline one-row bars; record tables correctly scoped; two false positives of my own recorded. |

**Overall.** The mathematics of this lesson is in unusually good shape. Fifty-four exact-rational checks, both
pipelines refit from the served CSVs, every conformal fixture, every practice solution, the reliability bins behind
the drawn figure and 260,040 rank cases all reproduce with **no disagreement**. The integer rank is correct
everywhere I could reach it and the reasoning behind it is sound. The coverage caution — the thing this lesson
exists to teach — holds on the rendered page: every measured figure carries its denominator and its kind, the
calibration rail is separated from the assessment plot, and I could not find a place where a marginal guarantee is
dressed as a conditional one.

Both blocking findings are failures of the **record** rather than of the page. In B1 the builder computed the right
answer and then argued itself out of it with the one test that could not discriminate. In B2 the builder found a
real defect class, fixed four of five instances, and wrote a guard that structurally cannot see the fifth — then
recorded the class as closed. The pattern in both is the same and worth naming: a verification that returns green
was treated as proof of the proposition it was written to test, without checking that its domain contained the case.
