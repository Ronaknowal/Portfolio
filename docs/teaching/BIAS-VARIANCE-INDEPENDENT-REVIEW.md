# Bias–Variance Tradeoff & Learning Curves — independent phase-two review

Reviewed 15 September 2026 against the working tree plus the uncommitted bias-variance implementation. The reviewer
authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or
`docs/teaching/drafts/` was edited; this review document is the only write outside `scratch/`. Throwaway scripts live
under `scratch/bv-independent/` and `scratch/bv-review/`. No state-changing git command was run.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/bv-independent/`, importing neither `bias-variance-models.js` nor `bias-variance-data.js` nor any `verify-bias-variance-*` script nor `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1, mpmath 1.3.0). **22,098 independent constructed checks, all passing**: 105 exact-rational checks of every number the manuscript, the visual specification and the practice solutions state; 21,993 exact-rational checks of the packet's own `calculated-inputs.json` trust root; 78 real-data checks refitting the airfoil pipeline from the served `.dat`, twice — once by running the manuscript's displayed programs verbatim, once through a hand-rolled reimplementation of the protocol that never calls `learning_curve`; and a finite-sample Monte-Carlo of the minimum-norm ridgeless estimator to settle the double-descent bias/variance split, which the manuscript never states. The generated module was parsed as **text** (regex + `json`) rather than imported, and its 500 fold values compared against my own refit. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `data-source.json`); the lesson body, model layer, generated data module, examples module, both labs, the shared scaffolding, the figures module and the CSS; all four verifiers; `LESSON-TEACHING-STANDARD.md`; the served asset and its attribution. |
| **Reused (declared)** | scikit-learn/NumPy as a *general* library, which is what the manuscript's own programs are written against — not lesson code. Three subagents for fan-out work: external-link fetching, verifier re-runs and source auditing, and Playwright capture. Every conclusion below that rests on their output, I re-derived or re-inspected myself: I opened the images, I read the capture loop, and I re-checked the one framing they got wrong (see O7). |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/bias-variance-tradeoff-learning-curves.jsx` | `66e6a25ced4bee4b8c399d666a09d383f8c658e64478c24b376f706577aeb602` |
| `src/learn/data/bias-variance-models.js` | `fbe6ce41f9a260925f54a784542fdc400e7864965935f38eaee60001c3baee38` |
| `src/learn/data/bias-variance-data.js` | `f6e4a912ab961c03c633aaef9a5ba645774c8cabf5fe22169e5be109ef9a3a82` |
| `src/learn/data/bias-variance-examples.js` | `fb2acb3099d5898f07939043b7dcccaa51146aa9c310011a579708436ff815a7` |
| `src/learn/components/lesson-labs/BiasVarianceShared.jsx` | `514fc93d928ab157626b90ad6d00b9941c6c402606394ea7e3a0ee06108f3afa` |
| `src/learn/components/lesson-labs/BiasVarianceLabs.jsx` | `2b70fdbbf1be941189ed35fe0a3631332f1702ab5e6f84f49ac4cceb8f700cb2` |
| `src/learn/components/lesson-labs/BiasVarianceFigures.jsx` | `cc08d5d7d0c2d39392fe8579a47a5757f1a03b562e92cab647305b514187e403` |
| `src/learn/components/lesson-labs/bias-variance-labs.css` | `6c7ad43d001785f22ef0189d07371727ec4b882f14a8eda0f60ab2c753e3bf93` |
| `docs/teaching/drafts/.../lesson.md` | `70f2c70817e26a1e52a7805b72befc3efa8b76e9428b1fd687adcf988f40fa8e` |
| `docs/teaching/drafts/.../design.md` | `78cf1992bdf0163b4284e99f1bc5fab0a9530a2740dbb0b5f83b5622c74c48e3` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `eac92529128d58691b4c771d47fdb8ac0be2097ca817359092f23e5718f3888e` |

---

# Part A — correctness

## A1. Exact recomputation of the finite content — **no disagreement**

`scratch/bv-independent/exact.py`. Exact rational arithmetic throughout (`fractions.Fraction`), with my own Gaussian
elimination and my own normal-equations least squares. Nothing is compared at floating tolerance where a rational
answer exists. **105 checks, zero failures.**

- **§1 sensor table.** A = [8,10,12]: mean 10, squared bias 0, variance 8/3, noise 1, total 11/3. B = [9,9,9]: 1, 0, 1,
  total 2. Both totals confirmed a second way, by averaging all six equally weighted (prediction, outcome) pairs — the
  decomposition and the direct enumeration agree exactly, not to tolerance.
- **All eight of the visual specification's I1 fixtures**: default 11/3; → [9,10,11] = 5/3; → [6,10,14] = 35/3; the
  permutation null exactly unchanged; [10,10,10] = 1; [9,11,13] = 14/3; the translate-by-2 null exactly unchanged; and
  the stated "variance stays 8/3 while squared bias becomes 1" on shifting up 1.
- **§3 table at x = .5, c = 1, σ = .5.** Constant 5/3, 1/144, 1/12, 49/144. Line 13/6, 25/144, 11/96, 155/288.
  Quadratic 7/4, 0, 23/128, 55/128. The interpolation weights are exactly [−1/8, 3/4, 3/8], they sum to exactly 1, and
  σ²·Σw² reproduces 23/128 exactly. Squared bias + variance = excess over noise, exactly, for all three fits.
- **§3 at x = 0.** The constant's squared bias is exactly 4/9; constant and line give *identical* predictions in every
  one of the eight worlds (verified world by world, not just in aggregate), total 7/9 each; the quadratic's total is
  exactly 1/2, which is lower — as the prose claims.
- **§3 at c = 0.** Line 35/96 with zero bias, quadratic 55/128, constant 7/12 (the visual specification's value).
- **Five-input design.** 29/80 = .3625, 31/80 = .3875, 12/35 — matching the visual specification's table to the exact
  rational. The stated contrast is real and my numbers reproduce both halves of it: the constant's variance falls
  (1/12 → 1/20) while its squared bias rises (1/144 → 1/16), so its total rises. The σ = 0, c = 0 tie at exactly 0 holds
  for both line and quadratic.
- **§7.** n = 6, p = 2, σ² = 4 → 8/3, 16/3, gap 8/3, prediction variance 4/3, and the gap is exactly twice the variance.
  I rebuilt the hat matrix of the constructed design [−2.5,…,2.5] with an intercept in exact arithmetic: it is exactly
  idempotent, its trace is exactly 2, mean leverage is exactly p/n, and the general smoother formula
  σ²(n − 2tr S + tr SᵀS)/n reproduces 8/3 and 16/3 exactly. **`design.md`'s claim that a curvature term the design
  cannot represent adds 6.222222 to both errors is exactly 56/9** — correct.
- **§8.** Brier A = .22, B = .26 with squared bias .01 and variance .04, each confirmed against the direct
  η(1−p)² + (1−η)p² expectation; zero-one errors .3 and .5; the η = .8 line .8 − .6q at q = 0, .25, .75, 1; the
  correlated-average variance 2.5 against v/B = 1.
- **§9 and practices.** All nine published ratios to six decimals; practice 8's .41 and .50; practices 1, 2, 6 and 7 in
  full.

I found **no disagreement with any number in the manuscript, the visual specification, the practice solutions or the
lesson body.**

## A2. The packet's trust root — **no disagreement**, but see O1

`scratch/bv-independent/trust_root.py`. The `finiteExperiments` block (21 records × 61-point grids) and the
`doubleDescentApproximation` block (12 records) of `calculated-inputs.json` are consumed as ground truth by
`verify-bias-variance-models.mjs` and are re-derived by **nothing in the repository**. I re-derived every value in both
blocks in exact rational arithmetic — targets, per-world curves, truth curve, mean curve, squared-bias curve, variance
curve and expected-error curve at each of 61 grid points, for all 21 experiments, plus both double-descent branches.

**21,993 checks, zero failures.** The trust root is correct. The fact that nothing in the repo establishes that is a
separate matter (O1).

## A3. The real airfoil pipeline — **no disagreement**

`scratch/bv-independent/real.py` and `real2.py`, refitting from the **served** file
`public/learn-assets/bias-variance/airfoil-self-noise.dat`.

- Running the manuscript's three displayed programs verbatim reproduces **every published value**: all 20 mean
  validation MSEs, the validation curve's six values (8.2079 … 21.8071), the five printed boosting rounds, and
  `best inspected round 120`. 56 checks, zero failures. The leaf-1 tree's training column really does print `-0.0`
  (departure 8 is accurate), and the returned arrays really are shape (5, 5).
- A **hand-rolled reimplementation that never calls `learning_curve`** — my own KFold iteration, one permutation stream
  consumed in fold order, explicit prefix subsets, explicit MSE — reproduces all 20 validation means and both quoted
  train/validation pairs at size 900 (Ridge 22.7528/23.3720, leaf-20 12.5438/16.6298) to nine decimals, and confirms
  960/240 rows per fold. 22 checks, zero failures. My first attempt reset the RNG per fold and disagreed with
  everything; the packet's provenance sentence *"prefixes of one permutation stream with seed 44"* is what makes it
  agree, so **the documented protocol is demonstrably the protocol that produced the numbers.**
- The generated module, parsed as text and never imported, matches my refit on **all 500 fold values** (4 procedures ×
  5 sizes × 5 folds × train/valid), all 60 validation-curve fold values and all 240 trajectory values. The two
  "mismatches" my comparator flagged are 9-decimal serialization rounding at exactly 5e-10, not errors.
- **Derived claims re-derived, not taken on trust.** The crossover really is first at 240 fitted rows (at 120 Ridge is
  still ahead, 23.8044 against 24.4041); the leaf-20 restriction really helps only at 60 and hurts at 120/240/480/900;
  the boosting minimum really is the last round; the monitoring trace really rises at exactly **nine** rounds
  (64, 70, 71, 77, 80, 99, 109, 110, 118) with a largest single rise of **0.038342**; and rounds 108→109 really are
  13.0848 → 13.1231. Every one of Phase C's numeric self-corrections checks out.

## A4. Double descent — the split the manuscript never states

The manuscript gives only the *total*. Figure 7's table publishes a `squared bias` and a `variance term` column that
come from the model layer's own split, which no source in the packet states. I derived it from first principles
(E[β̂] = γβ for the minimum-norm solution, so squared bias = (1−γ)²‖β‖²) and then **checked it by direct simulation**
(`scratch/bv-independent/dd_sim.py`, 240 replications at p = 600). At γ = 0.1 the simulation gives bias² 0.8143 and
variance 0.0923 against the lesson's 0.8100 and 0.0944; the plausible alternative split would have required 0.9044.
Across γ ∈ {0.1, 0.5, 0.8, 0.9, 1.5, 2.0} the simulated fresh-target MSE tracks the published values throughout.

**The model layer's split is correct and is a genuine improvement on the manuscript**, which states only the sum.
`doubleDescentRisk` also correctly returns `defined: false` at γ = 1 rather than a number.

## A5. Served data and provenance — **no disagreement**

- `public/learn-assets/bias-variance/airfoil-self-noise.dat` is **byte-identical** to the packet copy (`cmp` clean),
  59,984 bytes, SHA-256 `74c75fd7…b097b3` — matching `data-source.json`, `data-provenance.md`, `ATTRIBUTION.txt` and
  the generated module's `provenance`.
- The description is accurate on every detail I checked: tab separated (7,515 tabs = 1,503 × 5), **CRLF** line endings
  (1,503 CR, 1,503 LF, 1,503 CRLF, file ends with CRLF), 1,503 rows, six columns, no header. 1,200 development and 303
  reserved rows with seed 41, disjoint and covering all 1,503 — verified by set arithmetic on my own split.
- **This lesson serves its own bytes.** No source file under this lesson references `learn-assets/regularization`; the
  browser verifier positively asserts that no request touches that path (`verify-bias-variance-browser.cjs:319`). The
  two lessons happen to serve identical bytes, which is correct — they are the same unchanged upstream member — but
  through independent copies, which is what was asked.
- Licence CC BY 4.0 confirmed verbatim on the UCI page. The provenance note's aside that UCI's variable table labels
  attack-angle "Binary" while the column is numeric is **true**, and the manuscript correctly treats it as numeric.

## A6. Links and cross-references — **no disagreement**

All nine external links resolve and support the claim made of them, including the two that most invited doubt:
Nakkiran's Claims 1–2 do state the piecewise formulas the lesson uses (with ‖β‖ = 1 as a specialization of the paper's
‖β‖₂ ≤ 1), and Belkin et al. really do cover tree ensembles — Appendix D is "Additional results with Random Forests".
The Caltech slides really do use "bias" for the squared-bias contribution, as the lesson warns. scikit-learn **1.9.1 is
a real and current release**, so the "reviewed at version 1.9.1" note is accurate.

All five internal stable IDs resolve in `lesson-manifest.json` and `navigation.js` with the right module query. The
pointer *"Section 9 of the earlier Regularization lesson"* is correct: that lesson's heading 9 is literally
"9. Deeper branch: AIC, BIC and description length", and it contains the MDL coding example the sentence promises.

Inside the lesson, the two self-referential figure claims resolve: §2's *"drawn as the second panel of the figure
above"* matches `DeviationFigure`'s second panel (the hidden-setting lanes), and §7's *"as the last table of the figure
above shows"* matches `SameInputsFigure`'s leverage table. The readiness table's nine figure/section pointers all map
to the right figure index.

## A7. Practice instructions against the real controls — **no disagreement**

Every practice instruction that names a value is reachable in the lab that is supposed to accept it. Practice 2's
σ = 1 lies on the 0.05 step inside [0, 1.5]; the "Louder noise" preset really is curvature 1, σ = 1, probe 0.5 with the
**quadratic as the candidate**, so *"apply it and read the candidate row"* is correct. All eight I2 presets (including
c = −1 and the x = −1.5 edge probe) sit inside their ranges and on their steps, as do all eight I1 setups on the 0.25
step. The §3 checkpoint's four quoted numbers (0.340278 → 0.3625, variance 0.083333 → 0.05, mean 5/3 → 1.5, squared
bias 0.006944 → 0.0625) are all exactly right.

## A8. Investigation contract — met

Read in the code and driven on the real page. `useInvestigation` (`BiasVarianceShared.jsx:127–163`) holds `draft`,
`active` and `previous` separately; `check` sets `previous = active` before advancing and grades
`answerFor(draft, active)` — i.e. **against the state that was committed, not live state**. `edit` retires the verdict
and clears both the radio and the numeric guess. Driving the page confirmed all of it: first paint has zero radios
checked, Apply disabled, no verdict and no waterfall; after committing a deliberately wrong direction the mismatch
branch renders correctly (*"≠ You recorded It falls; the calculation gives Exactly unchanged"*); changing one
spinbutton drops the radio and the verdict and raises the pending notice. Both labs carry genuine null cases
(permutation, translation, probe-0 coincidence, noiseless) and I confirmed each is an exact null in exact arithmetic.

---

# Part B — findings

## Blocking

### B1. The record's only "not fixed" item describes the wrong file, and the trade-off it justifies is not the one that exists

`docs/teaching/drafts/bias-variance-tradeoff-learning-curves/design.md`, final section, states:

> One screenshot path is written twice: the figure-4 capture in the per-figure loop runs after the round-30 selection,
> so `bias-variance-figure-4-desktop.png` shows the inspected round rather than the default. The state it captures is
> the more informative one, so the order was left alone; 30 captures produce 29 files.

Two of those three assertions are false, and the third is right for the wrong reason.

- **The path written twice is `bias-variance-figure-3-desktop.png`, not figure 4.** It is written explicitly at
  `scripts/verify-bias-variance-browser.cjs:271` and again by the per-figure loop at `:290` (index 2). Verified
  directly: `docs/teaching/evidence/bias-variance-browser.json` lists 30 screenshot paths of which 29 are unique, and
  the single duplicate is `…figure-3-desktop.png`.
- **`figure-4-desktop.png` is not a duplicated path at all.** It is a distinct path whose *bytes* are identical to
  `bias-variance-figure-4-round30-desktop.png` (`cmp` reports identical; both 165,256 bytes), because `:277` captures
  the round-30 state under its own name and the loop at `:290` then captures the same on-screen state again under the
  default name. So the round-30 state is stored **twice under two names**, and figure 4's default state is stored
  **nowhere**.
- Consequently the justification — *"the state it captures is the more informative one"* — does not apply. The
  informative state already had its own file. What was actually traded away was the only desktop capture of figure 4's
  default state, for nothing.

The same evidence file's `visualInterpretation` string claims the screenshots include "every inline figure at desktop
and at 320 or 390 px", which is false for figure 4's default.

**Why it matters.** This is the record-versus-tree class the brief calls most recurrent, and here the builder reasoned
to an explicit "leave it alone" decision from facts that do not hold. I captured figure 4's default state myself
(`scratch/bv-review/shots/plot-fig4-1.png`) and it contains no defect — the default `inspected` is already
`trace.bestRound`, so the stem sits at round 120 — so nothing is being hidden. The defect is the record, not the page.

**Judgement on the question I was asked to rule on:** the 30-captures-29-files outcome is acceptable in itself. The
*explanation* of it is not, and should be corrected rather than merely re-worded; and since the round-30 state is
already captured under its own name, the loop should simply restore the default selection before it runs, which costs
one line and yields 30 files.

### B2. Four data curves cross value labels, in exactly the blind spot the record identified but scoped too narrowly

Phase C records the layout inspector's limitation and its mitigation:

> The visual-layout inspector does not examine curved paths, so defect 4 above could recur elsewhere in this lesson
> without being caught. It was checked by eye in every drawing that uses a curved flow: figures 2 and 5.

The mitigation covers `<path>` *flow arrows*. It does not cover `<polyline>` **data curves**, which is where the
recurrence actually is. I ran a precise geometric test — sample each `<polyline>`/`<path>` by `getTotalLength()` +
`getPointAtLength()`, transform through `getScreenCTM()`, and test containment in each `<text>`'s client rect — at
first paint and again with both investigations committed and every `<details>` open. Identical results both times, so
none is state-dependent. **Four real collisions** (script and output: `scratch/bv-review/review.cjs`,
`scratch/bv-review/probe.json`):

| # | Figure / plot | Label | Curve | Samples inside |
| --- | --- | --- | --- | ---: |
| 1 | Fig 7, main log plot | `noise 0.04` | `polyline.bv-curve.is-validation` (the risk curve) | 110 / 482 = **22.8%** |
| 2 | Fig 7, linear inset | `0.50` | `polyline.bv-curve.is-validation` | 29 / 584 = 5.0% |
| 3 | Fig 6, thresholded-error plot | `0.20` | `polyline.bv-curve.is-validation` | 18 / 513 = 3.5% |
| 4 | Fig 3, combined plot | `tree better from 240` | `polyline.bv-curve` (baseline) | 2 / 532 = 0.4% |

I opened each. Collision 1 is the worst and is plainly visible in `scratch/bv-review/shots/plot-fig7-0.png`: the
descending risk curve runs the full width of the `noise 0.04` label. Collision 3 is the most damaging per-pixel,
because **`0.20` is the only value label in that plot without the `bv-halo` backplate** — `0.80`, `0.65` and `0.35`
all have it and all clear the line — so the crossing is unmitigated; see `scratch/bv-review/shots/fig6-1366.png`.
Collision 4 is a graze and cosmetic.

`scripts/lib/lesson-visual-layout.cjs:46` iterates `svg.querySelectorAll('line')` only, so `<polyline>`, `<path>`,
`<polygon>`, `<circle>` and `<rect>` are never tested. The browser verifier therefore reports zero label issues at all
five widths while these four stand. The fix is two-part: place the four labels clear of their curves, and extend the
inspector to sample `<polyline>`/`<path>` geometry so the class cannot recur silently again.

### B3. §9 tells the learner the approximation is undefined on an interval, and contradicts itself two sentences later

`BiasVarianceFigures.jsx:739–745`, rendered beneath figure 7's main plot:

> The shaded band from γ = 0.97 to 1.03 is where this approximation has no value: the two branches stop at its edges
> and are never joined across it, because γ = 1 is a singular boundary rather than a point on the curve. … The values
> the table gives at γ = 0.99 and 1.01 are finite only because those two ratios were chosen, so they are listed rather
> than plotted.

The first clause is false. The approximation has a value everywhere except **exactly γ = 1**; it is defined and finite
at γ = 0.97, 0.99, 1.01 and 1.03, all of which lie inside the band. The lesson's own table, eight lines below, prints
**4.010000 at γ = 0.99 and 4.040000 at γ = 1.01** — and I confirmed both exactly. The paragraph therefore asserts the
function has no value on an interval and then prints two of its values on that interval.

The second sentence compounds it: those values are not "finite only because those two ratios were chosen" — they are
finite because the function is finite there. The choice determines which values are *listed*, not whether they exist.

**Why it matters more than a wording slip.** The single conceptual point §9 is built to deliver is that the
singularity is a *point*, not a region — the manuscript says "mark the undefined/asymptotic singular boundary at
γ = 1", and the visual specification (F6) says "Mark 1 as singular/asymptotic boundary". Widening the undefined set
from a point to an interval, in prose, teaches the opposite of the section's thesis. The shaded band is a legitimate
*drawing* device introduced in Phase C; the prose needs to describe it as one ("the branches are cut short either side
of the boundary so the two are not read as one curve") rather than as a property of the mathematics.

## Should-fix

### S1. `design.md` says four defects were invisible to the verifiers; it then lists eight

`design.md:209`: *"That second pass is what found most of the defects below; four of them were invisible to a green
verifier."* The section immediately below, "Defects found only by looking at the images", contains **eight** numbered
items — all of which are by construction invisible to a green verifier, since they were found only by looking. The
preceding table lists four *separate* defects that the inspector did catch. The sentence appears to have transposed
the two counts. Verified by counting both sections.

### S2. The model layer still carries the retracted Phase A claim

`src/learn/data/bias-variance-models.js:432–433`:

```js
/** The trace still wiggles: these are the rounds where it rose from the one
 * before, and they are drawn rather than smoothed away. */
```

Phase C retracted exactly this ("This replaces the claim recorded as departure 10 above: the wiggles are *recorded and
checkable*, not visible"), and the figure prose was correctly rewritten to say the rises are "far too small to see at
this scale rather than being smoothed away". This comment is the last surviving statement of the retracted claim, and
it sits in the file a future maintainer would read first. The builder's self-correction was right; this is the one
place it did not reach.

### S3. Figure 3's headline plot separates the crossover pair by colour alone

`BiasVarianceFigures.jsx:22`: `const dash = { mean: '2 4', ridge: '', tree_leaf1: '', tree_leaf20: '6 3' }`. In the
combined "All four validation means on one axis" plot, the baseline is dotted and the leaf-20 tree is dashed, but
**Ridge and the leaf-1 tree are both solid**, distinguished only by stroke colour (`#91aecf` against `#e7b94a`), with
same-colour circle markers. Those two series are precisely the pair whose crossover is the figure's headline claim and
the subject of its annotation. Confirmed visually in `scratch/bv-review/shots/plot-fig3-0.png`.

The packet's own shared contract says "Use text/shape as well as color, an exact-value table, and plain captions". The
information is recoverable elsewhere — the four small multiples are individually headed and the exact tables are
present — and blue-versus-gold survives the common red–green deficiencies, so this is not severe. But it is a
departure from the lesson's own rule in the one plot where the distinction carries the argument, and giving one of the
two a dash pattern costs nothing.

### S4. Figure 7's branches still read as one spike, and the drawn peak is a fifth of the tabulated one

Phase C defect 3 reports this as fixed:

> The two double-descent branches read as a single spike. They were never joined in the path data, but a 0.06-wide gap
> in γ looks like a notch. The boundary is now a shaded band from γ = 0.97 to 1.03 that the two branches stop at.

Looking at the rendered plot (`scratch/bv-review/shots/plot-fig7-0.png`), the fix is a genuine improvement but has not
achieved its goal. The two branch ends are at **1.3633** (γ = 0.97) and **1.3733** (γ = 1.03) — within 1% of each other
in height — separated by a narrow band, so the eye completes the curve across it and the band reads as a bar drawn
*over* a continuous peak rather than as a discontinuity. Two things reinforce that reading:

- The drawn maximum is ≈1.37 on a log axis labelled to 4, while the table beside it gives 4.01 and 4.04. A learner who
  reads the picture takes away a peak that is roughly a fifth of the tabulated one.
- The divergence arrow spans `scaleY(1.7)` to `scaleY(4.4)`, so it starts visibly *above* where the branches stop,
  with a gap between them. It reads as a floating annotation rather than as the continuation of the curve.

Ending the branches at the two tabulated ratios γ = 0.99 and 1.01 — which is what the visual specification asks for
("The plotted finite endpoints at .99/1.01 do not define a finite maximum") — would put the branch ends at 4.01 and
4.04, near the top of the axis, make the asymptote legible, connect the arrow to the curve, and let the band shrink to
a genuinely narrow marker at γ = 1. It would also remove the premise of B3.

## Observations

### O1. The 19,764-value comparison rests on a trust root nothing in the repo re-derives

`verify-bias-variance-models.mjs` compares the model layer against `calculated-inputs.json` — a genuinely independent
NumPy/scikit-learn oracle, which is the right design. But `verify-bias-variance-data.py` re-derives only that file's
`real` section. Its `finiteExperiments` (21 records, the source of all 19,764 compared grid values) and
`doubleDescentApproximation` blocks are re-derived by nothing, `author-calculations.py` is never re-run, and the file
is never hashed by any verifier. A regenerated-from-buggy-code or hand-edited block would produce agreement and the
same green PASS line.

**I closed this gap for the current bytes** (A2: 21,993 exact checks, zero failures), so no value is wrong today. The
recommendation is structural: hash `calculated-inputs.json` in the data verifier, or re-derive the two finite blocks
there as it already does for `real`.

### O2. What the verifiers do not assert

Re-run confirms every count in the record: models `179 grouped checks across 45 groups, including 19,764 recorded grid
values`; examples `3 displayed programs executed, 55 oracle assertions`; data `100 learning fits, 30 validation fits
and one 120-round trajectory`; browser `11 cases and 30 screenshots`. All four exit 0. The counts are honest — the
grid counter is a deliberate undercount, and every evidence file carries a `limitations` block. Beyond the curved-path
gap of B2, the substantive blind spots are:

- **No assertion ever reads a plotted coordinate.** All seven figure checks are regexes over rendered text. Swapping
  `trainingMeans` for `validationMeans` in a plot, or drawing the crossover stem at the wrong x while its label stays
  correct, would pass.
- **The grading contract is tested only along its happy path.** All five verdict assertions match on
  `/Your prediction matches: …/`; the `is-miss` branch is never rendered during the run, the numeric guess is never
  filled so the tolerance comparison is never exercised, and the "Graded against the committed state" caption is never
  asserted. I exercised the mismatch and retirement paths by hand and both behave correctly (A8), but the verifier
  would not catch an inversion.
- **Zero keyboard or focus coverage.** Every interaction is a programmatic Playwright click. Note that radios are
  `disabled={Boolean(shown)}` after a commit, which drops focus to `<body>`; nothing checks that.
- **The code-on-page check whitespace-normalizes** (`verify-bias-variance-browser.cjs:26`), and the displayed programs
  are Python. A regression that destroys indentation would leave the shown program syntactically invalid and still pass.

### O3. Investigation 1 states one decomposition term before any prediction is recorded

At first paint the prediction ruler draws the `m − f` bracket from the draft, and the accompanying sentence reads
"…the bracket has zero length, and the squared bias is exactly zero" (`BiasVarianceLabs.jsx:144–150`). This is
defensible — it is a function of inputs the learner can see and edit, the visual specification explicitly asks for
"a distance bracket [that] identifies the signed mean offset", and the *graded* quantity is the direction of change of
the total against the previous applied state, which cannot be read off it. The waterfall and the moved-terms table are
both correctly gated behind the commit. Recording it because the contract phrase is "nothing revealed on first paint"
and a decomposition term is, strictly, stated there.

### O4. Phase C's "the two lines visually meet near 200" is off by about sixty rows

On the drawn segments, Ridge and the leaf-1 tree cross where the joined segment between 120 and 240 changes sign:
at 120 the gap is +0.5997, at 240 it is −3.4724, so the visual crossing is at ≈**138** fitted rows, not near 200. The
fix the number was used to justify — retitling the annotation `tree better from 240` and adding the sentence that the
segments are joins with the 120-row values — is correct and adequate regardless. Only the number in the record is wrong.

### O5. Two citation refinements, neither an error

- The Further Reading anchor "scikit-learn — Learning and validation curves" is not that page's title, which is
  "3.5. Validation curves: plotting scores to evaluate models". The page does contain both subsections, so the anchor
  describes the content correctly.
- The lesson's claim that incremental mode "requires the estimator's partial-fit interface; a warm-start flag alone
  does not make it valid" is **true of scikit-learn** (`_validation.py` raises exactly that ValueError), but the cited
  API page does not state it. The claim is sourced from the implementation, not the citation.

### O6. Nakkiran's Claim 2 is itself attributed to Hastie et al.

§9 cites "Nakkiran, Claims 1–2" for the piecewise formula. Claim 1 is Nakkiran's; Claim 2 is credited in that paper to
Hastie et al. 2019, Theorem 1. The citation is defensible — the reader is sent to the right two claims — but the
secondary attribution is invisible, and §9 is the one section built entirely on a cited asymptotic result.

### O7. The wide tables scroll; they are not clipped

I record this because it is the one place my own fan-out tooling was wrong and I want the correction on the record.
Several tables are wider than their containers — four `.lesson-table-wrap` tables at 390 and five at 320, and four
`.bv-table-scroll` tables at 1366, including figure 6's, whose last column is visibly cut at the card edge in
`scratch/bv-review/shots/fig6-1366.png`. These are **not** overflow defects. `.lesson-table-wrap` is `overflow: auto`
and `.bv-table-scroll` is `overflow-x: auto` with `scrollbar-gutter: stable`, and both are rendered as
`role="region" aria-label={caption} tabIndex={0}` — proper named, keyboard-focusable scroll regions, which is exactly
what the packet's shared contract permits ("A text table may scroll in its own named region; the page must not
overflow horizontally"). Document `scrollWidth` equals `innerWidth` at both 390 and 320, so the page itself never
scrolls sideways. The only residue is cosmetic: figure 6's table needs horizontal scrolling at 1366 px, where the page
has width to spare and the figure card does not.

### O8. Things the record gets right that were worth checking

Ten of the eleven "departures from the specification" hold up against the tree: the seventh figure exists and its
content is the manuscript's own arithmetic; the §7 fixed design really does reproduce 8/3, 16/3, 4/3 and 56/9 through
the general smoother formula with trace exactly 2; the five-input checkpoint is present with correct numbers; I2's
panels do collapse to one column; the §5 figure does tabulate training rather than validation means; the crossover and
restriction sentences are printed from the model layer; `-0.0` is real; section 11 exists and the intro route reaches
it. Departure 10 is the one superseded, and it is explicitly cross-referenced from Phase C — the retraction itself is
handled honestly, and the builder was right to make it: I measured the largest rise at 0.038342 on a 0–50 axis, which
is indeed imperceptible. The blueprint is now registered in `blueprints/index.js`, as Phase C claims.

---

## Closure

The mathematics of this lesson is the strongest I have reviewed in this series. Across 22,098 independent checks —
exact rational arithmetic wherever a rational answer exists, a from-scratch reimplementation of the real pipeline, a
text-parse rather than an import of the generated module, and a Monte-Carlo settling a split the manuscript never
states — **I found no disagreement with any published number**, in the manuscript, the visual specification, the
practice solutions, the lesson body, the generated data module, or the packet's own trust root. The served data is
byte-identical and honestly described, this lesson serves its own copy, every link resolves and supports its claim,
every practice instruction is reachable in the lab it names, and the investigation contract is met in code and on the
page. Where the model layer departs from the manuscript — the double-descent bias/variance split — it is right and the
manuscript is merely silent.

The defects are in the record and in the drawings. **B1** is the familiar failure: a record asserting an
implementation state that contradicts the tree, with a decision reasoned from it. **B2** is the failure the record
predicted and then scoped too narrowly, leaving four real collisions standing behind a green verifier. **B3** is a
paragraph that contradicts itself on the one idea its section exists to teach, and **S4** is the drawing decision that
produced it — fixing S4 dissolves B3.

Not complete. Fix B1–B3, then S1–S4.
