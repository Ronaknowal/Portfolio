# Rademacher Complexity & Generalization Bounds — independent phase-two review

Reviewed 19–20 September 2026 against the working tree plus the uncommitted Rademacher implementation. The reviewer
authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/`,
`docs/teaching/drafts/` or `docs/teaching/topic-notes/` was edited; this review document is the only write outside
`scratch/`. Throwaway scripts live under `scratch/rad-review/`. No state-changing git command was run. The build
directory `dist-rad/` was deleted and the preview on 4194 stopped at the end.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/rad-review/` (`exact1.py`, `exact2.py`, `replay.py`, `gradegap.py`, `look.cjs`, `probe8.cjs`, `probe2.cjs`), importing nothing from `rademacher-models.js`, `rademacher-data.js`, `rademacher-examples.js` or any `verify-rademacher-*` script, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, mpmath 1.3.0). Exhaustive sign-pattern enumeration in `fractions.Fraction`; geometry, kernels and Massart in 50-digit `mpmath`. A complete independent replay of the bounded-norm experiment from the **served** CSV using my own FISTA projected-gradient solver — no `scipy.optimize`, no scikit-learn. A reviewer-written Playwright drive sharing no code with `verify-rademacher-browser.cjs`, at 1366, 390 and 320 px on Edge 153.0.4234.48. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `checked-results.json`, `author-checks.json`, `experiment-results.json`, both programs); the destination note; the lesson body, model layer, both generated modules, all three lab/figure/shared components and the CSS; all five verifiers and the falsification harness; the served assets and `ATTRIBUTION.txt`. |
| **Reused (declared)** | NumPy's PCG64 at the recorded seeds, because the Monte-Carlo records *are* defined as that generator at that seed — everything deterministic around them (per-draw ceiling, Hoeffding correction, endpoint, standard error) I recomputed from the definitions. Two subagents for fan-out: a falsification-harness audit and a design.md-against-tree audit. **Every finding below that a subagent surfaced, I re-derived myself before reporting it** — I read each cited line and, where the claim was empirical, measured it in the live page or ran the verifier. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/rademacher-complexity-generalization-bounds.jsx` | `d7dad6233f2ac6cdae18358774e9dd9941363db43f3a679cae52edadae8af5dc` |
| `src/learn/data/rademacher-models.js` | `2efda3c2ec09227c672589fbf184e321c6cb71649b1a8e89cc76927b2f984e2a` |
| `src/learn/data/rademacher-data.js` | `001572787c88eb965e334e50ac45e23e8f883f978642894019846fcaa0701701` |
| `src/learn/data/rademacher-examples.js` | `f664f16c0bdb73aeddb2163d6f42505eaa18935f84a90de7a6716124f5c9398f` |
| `src/learn/components/lesson-labs/RademacherShared.jsx` | `5e7d6fb8140e8af60c66c9d36f7ca8ba560d59f379a0e2c0cd22d65d2b85c501` |
| `src/learn/components/lesson-labs/RademacherLabs.jsx` | `dfe59f0b43441078fbcb0c76c2fb508a1b569d2b6db3bdc2e1a8305aaf04eb1f` |
| `src/learn/components/lesson-labs/RademacherFigures.jsx` | `8c46e5684f4cd5df67377a1983885a733d387999292e6e8947e79e18b9b217d7` |
| `src/learn/components/lesson-labs/rademacher-labs.css` | `97b4629e5da8150865e96603984b00d3696c0554b2c8fcd48171de3c4e330a1e` |
| `scripts/falsify-rademacher.mjs` | `4eafd8068af87e514c7363a83e8ddbd67900eb83867233a2f872c1251ffce76e` |
| `docs/teaching/drafts/.../lesson.md` | `a3a938040b9992076d0cd8bba500ec67149a0eda166bd7af7ae893496f77aef4` |
| `docs/teaching/drafts/.../design.md` | `cf14357e2d67fcc7d3fdeecb725c6e7fe752c433a6acf6a71b684c3a11ac6ac3` |
| `docs/teaching/drafts/.../checked-results.json` | `aa990e932d68b371ca1e680b9ee52e97fec81bfbf17cb5dcf52ded190f8fcc6b` |
| `docs/teaching/drafts/.../experiment-results.json` | `26caac6ec64c9223a1ee225c01d7fe8adaf74e6e023fb1e6440ed88be1648c20` |

---

# Part A — correctness

## A1. Every finite Rademacher complexity, by exhaustive enumeration in exact rational arithmetic — **no disagreement**

I reimplemented the definition from scratch: for a value matrix `f[h][i]`, enumerate all `2^n` sign vectors, form
`sum_i sigma_i f[h][i] / n` as a `Fraction`, take the per-pattern maximum with **no** flooring at zero and **no**
absolute value, and average. I rebuilt the threshold-class restrictions myself from the definition
`h_t(x) = +1 iff x >= t`, including both extreme cutoffs.

All six classes on the three inputs `(-1, 0, 1)` reproduce the packet exactly:

| Class | Mine (exact) | Packet | Value matrix |
| --- | --- | --- | --- |
| singleton | `0` | 0.0 | matches positionally |
| constants | `1/2` | 0.5 | matches positionally |
| thresholds | `2/3` | 0.666666666666666**6** | matches positionally |
| both orientations | `5/6` | 0.83333333333333**33** | matches positionally |
| all labels | `1` | 1.0 | matches positionally |
| classification loss | `1/3` | 0.3333333333333333 | matches positionally |
| absolute-convention singleton | `1/2` | 0.5 | — |
| practice, thresholds on `n = 2` | `3/4` | 0.75 | matches positionally |

Also reproduced exactly: the Euclidean-ball complexities for all six geometry fixtures and the practice `(3,0),(0,4)`
case, each as `(B/n) * mean over 2^n of ||sum_i sigma_i x_i||`, together with their feature-energy upper bounds and
the Jensen inequality between them; the three kernel complexities as `(B/n) * mean sqrt(sigma^T G sigma)`; all three
Massart values as `sqrt(2 ln M) * A / n` with `A = sqrt(n)`; the ramp losses and the contraction addend
`2 B * energy / rho` at both thresholds; and the finite-population symmetrization example by exhaustive enumeration
of all four size-two samples with replacement (`E[largest gap] = 1/4`, `E[ghost sup] = E[Rademacher] = 3/8`,
`2 E[Rademacher] = 3/4`, and `1/4 <= 3/4` as symmetrization requires).

Growth functions, checked by building the restrictions rather than asserting the formula: the positive-threshold
class gives exactly `n + 1` distinct labelings for `n = 1..8`, the both-orientation class exactly `2n`. Sauer–Shelah
sums `sum_{i<=d} C(n,i)` computed in Python `int`.

**One ulp of disagreement, in the implementation's favour.** The packet records
`two_orientations/complexity = 0.8333333333333333`; the shipped `recordedFixtures.bothOrientationsComplexity`
(`rademacher-data.js:848`) is `0.8333333333333334`. Exactly `5/6` rounds to the latter, so the module carries the
*more* accurate double and the packet is one ulp low from a different summation order. Not a defect; noted because a
naive byte comparison of those two files would flag it.

## A2. Sign-pattern enumeration order — **correct on every positionally-indexed array**

This was the named trap. The packet enumerated with `itertools.product([-1, 1], repeat=n)`, i.e. the **last**
observation varying fastest. `signPatterns` (`rademacher-models.js:127-139`) builds
`pattern[index] = (code >> (n - 1 - index)) & 1`, which is that order, and `signPatternIndex`
(`:142-145`) inverts it consistently.

I did not take the aggregate agreement as evidence. For every one of the six classes plus the practice case I
compared, element by element and in order, my own `correlations` matrix (`2^n` rows × `|H|` columns), my `maxima`
vector, and my first-winner index vector against the packet's stored parallel arrays. **All aligned.** The same
positional check passed for the six geometry fixtures' `signed_sums` and `maxima` arrays.

The order is also visible and correct on the rendered page: figure 2 prints the eight columns as
`σ1: −−− , σ2: −−+ , σ3: −+− , σ4: −++ , σ5: +−− , σ6: +−+ , σ7: ++− , σ8: +++`, and its first column reads
`1.0000, 0.3333, −0.3333, −1.0000`, which is the packet's `correlations[0]`.

## A3. The bounded-norm experiment, replayed from the served CSV with a different optimiser — **no disagreement**

I loaded `public/learn-assets/rademacher/banknote-subset.csv` directly, recomputed the representation-set mean and
scale myself (population standard deviation, not `StandardScaler`), applied the declared clip-then-bias map, and
refit all five budgets with my own FISTA accelerated projected-gradient solver onto the L2 ball — no
`scipy.optimize`, no scikit-learn.

Reproduced to the last recorded digit: `energy_factor = 0.079949643120159841` and
`confidence_addend = 0.3351715385564154` (both exact bit-for-bit matches, and I confirmed the confidence term is
`3 sqrt(ln(2K/delta) / 2n)` with `K = 10`, `delta = .05`, `n = 240` — the correct union-bound allocation `delta/K`
for ten predeclared comparisons). All fifteen error counts (fit/validation/assessment at each of the five budgets);
all ten `empirical_ramp` values to ~1e-11, which is my solver's gap and not a disagreement — my objective values sit
within `3.3e-14` to `2.2e-13` of the packet's, always slightly *lower*, i.e. my solver reached the same minimum from
a different direction; all ten `complexity_addend`, `raw_upper` and `clipped_upper` values exactly; the selection
rule independently choosing `B = 4` from validation errors `16, 14, 10, 6, 4`; the majority baseline (`class_sign
= −1`, 39 validation and 29 assessment errors); the zero-predictor counts under the `score >= 0` tie rule (134/41/51
on fit/validation/assessment, matching `author-checks.json`); 476 distinct feature vectors among 480 rows with the
four duplicate groups in the same roles; and the 2,048-draw unit-ball Monte Carlo including its correction and
standard error.

**Every one of the ten bound expressions exceeds 1.** My independently computed raw sums are
`1.251894, 1.293533, 1.181744, 1.258458, 1.121654, 1.198668, 1.244384, 1.129848, 1.724800, 1.198469`. The lesson is
correct that none of them says anything, and it says so — see Part B.

I also confirmed the feasibility claim myself: the maximum mapped row norm over all 480 rows is `1.8489724660539508`,
comfortably under the declared global ceiling `sqrt(5)`.

## A4. The Monte-Carlo records — **no disagreement, and the Hoeffding range is honest**

The four-row table estimates the parallel-geometry fixture `[[1,0],[1,0]]` at `B = 1`, whose exact value is `1/2`. I
replayed all four with my own code and reproduced every estimate, correction, endpoint and standard error exactly.
Importantly, I checked the *range* rather than accepting it: the per-draw values are observed to lie in `{0, 1}`, so
the recorded `per_draw_upper = 1.0` is exact and the one-sided Hoeffding correction `Q sqrt(ln(1/eta) / 2m)` is
correctly applied. The standard error matches a `ddof=1` Bernoulli, which is what the recorded numbers imply
(`0.03125` at 256 draws is exactly `0.5/16`).

For the unit-ball record the declared `per_draw_upper = 1.2252815721318897` is the mean row norm — the triangle
inequality ceiling, a valid deterministic per-draw upper bound (observed maximum per-draw value: `0.230`). It is
conservative, not wrong.

## A5. The recorded packet disagreement — **the builder's count is right and recording rather than fixing was correct**

`author-checks.json:72` records `manuscript_words: 7678`. My own whitespace-token count of the frozen `lesson.md`
gives **7,769**, identical to `wc -w`; the file hashes to
`a3a938040b9992076d0cd8bba500ec67149a0eda166bd7af7ae893496f77aef4`, which is the hash the evidence records, so we
counted the same bytes. The builder's 7,769 is correct and the packet's 7,678 is stale.

Nothing computes from it. The only two references in the tree are in the verifier itself:
`verify-rademacher-data.py:638` re-derives the count, and `:115-124` declares the disagreement. No lesson body, model
layer, generated module, figure or investigation reads it.

**Recording rather than editing was right,** for three reasons the verifier itself states at `:110-114`: the packet
is a frozen trust root and silently rewriting it would destroy the thing the trust-root count is measured against;
the discrepancy has an explanation that is checkable (the manuscript gained the duplicate-feature disclosure after
the author pass); and the `DECLARED_DISAGREEMENTS` gate is narrow — it requires the disagreement to be explained, the
packet value left untouched, and nothing the lesson teaches to depend on the packet's figure, and it is enforced by
being a literal dict the verifier must find a key in rather than a tolerance. A numerical or teaching-relevant
disagreement would still be a failure. I would have made the same call.

## A6. The trust root — the mechanism is real, not a tautology

The claim is 5,689 of 5,701 scalar leaves re-derived across the three packet files, with twelve exclusions. It holds
up and the mechanism is genuine:

- The count is **per-file and enumerated**, not asserted: `checked-results` 803/804, `experiment-results` 4838/4847,
  `author-checks` 48/50. These sum to 5,689/5,701 and the difference is exactly 12.
- Each file carries an `uncovered` list, and **all three are empty** — so every leaf is either re-derived or named in
  `declaredExclusions`. That is what makes the number non-tautological: a leaf cannot be quietly dropped, it has to
  appear in one list or the other.
- The twelve exclusions are declared **individually with a reason string each**, not as a count: `/convention`,
  `/versions/numpy`, `/versions/scipy`, `/versions/sklearn`, `/selection/rule`, five `/models/i/solver/iterations`,
  `/type`, `/date`.
- The fitted coefficient vectors are **not** excluded. They are re-derived by projected gradient descent — a
  different optimiser from the packet's SLSQP — with `largestObjectiveGapFromRecorded = 2.2e-13`. I confirmed this
  independently with my own FISTA solver (A3), which is a third route.

**Each exclusion is honest.** The three version strings and the two dates/type strings are not derivable from
anything. `/convention` and `/selection/rule` are prose, and the selection rule is checked by *applying* it rather
than by string-matching. The five SLSQP iteration counts are solver-internal and genuinely not re-derivable from the
definitions by any independent route — and the builder did not simply wave them through: they are re-derived by
executing the program (`verify-rademacher-examples.py`, which regenerates both frozen result files byte for byte).
That is the right treatment.

One accounting nit is carried in S6.

## A7. Served assets — **no disagreement**

`banknote-subset.csv`, `complexity_calculations.py` and `bounded_norm_experiment.py` under
`public/learn-assets/rademacher/` are byte-identical to the packet copies (SHA-256 verified for each). The CSV hashes
to the value recorded in `provenance` and in `data-provenance.md`. `ATTRIBUTION.txt` correctly states the licence,
the DOI, the archive and member hashes, exactly which two columns were added, and that no measurement was altered.

The duplicate-feature disclosure **is** in the served attribution, and its content matches my independent
recomputation line for line: four pairs with identical measurements in all four coordinates, 476 distinct vectors
among 480, two pairs crossing fitting and validation, one crossing representation design and fitting, one inside
fitting, none involving assessment.

---

# Part B — the rendered page

Built with `npx vite build --outDir dist-rad` against a verified-clean source tree and previewed on 4194. Driven at
1366, 390 and 320 px on Edge 153 with a script sharing no code with the lesson's own browser verifier. Zero console
errors, zero `.katex-error` nodes. Seventeen figures render, and their captions read `Figure 1.` through
`Figure 17.` in document order — the numbering fix is real in the page, not just in the source.

**The five "DOM right, picture wrong" mechanisms, checked directly:**

1. **CSS beating a presentation attribute.** I compared every `fill`/`stroke` presentation attribute inside
   `.rad-lesson svg` against its computed value and flagged any case where `none` became painted or the reverse.
   **None found.**
2. **CSS collapsing geometry.** This was the priority, since the lesson uses hatted empirical quantities heavily. I
   measured the rendered height of all 636 KaTeX structural elements. `hide-tail` (the radical surd and wide-hat
   tails): 24 nodes, **zero** at zero height, minimum 16.98 px. `accent-body`: 2 nodes, minimum 18.88 px. `vlist`:
   513 nodes, one sub-pixel case which is a fraction rule, not a radical. The 27 sub-pixel `<svg>` nodes are all
   inside `.katex` and all wide-and-flat — they are KaTeX's horizontal rules, correctly a fraction of a pixel tall.
   **The sibling lesson's collapsed-radical failure is not present here.** `rademacher-labs.css` scopes its SVG rule
   at the lesson root, which `verify-rademacher-sources.py` also asserts.
3. **Paint order.** Covered by check 4.
4. **A shape drawn over a label.** I hit-tested the centre point of every `<text>` inside `.rad-lesson svg` with
   `document.elementFromPoint` and reported any case where the topmost element was not the text or an ancestor.
   **None found**, at any of the three widths.
5. **Contrast.** No gold-on-gold case found. Figure 15's legend uses filled swatches beside text rather than coloured
   words, and figure 10's two optima are named in a swatch legend below the plot rather than by colour alone — both
   claimed fixes are visible in the images.

**Layout.** `document.documentElement.scrollWidth` equals the viewport at both 390 and 320 px — no horizontal page
scroll. The only elements extending past the viewport are KaTeX's visually-hidden `<math>` MathML accessibility
layer, which is not painted. At 320 px the figure-15 table restacks into per-budget cards and the bars still extend
past the dashed ceiling.

**The central risk — a bound read as a promise — is handled well.** Figure 4 is the load-bearing figure and it is
excellent: it names the three terms separately, then gives a three-row table headed "What is random, and what each
randomness is averaged over" distinguishing the sample draw ("the 1 − δ failure allowance covers this"), the
auxiliary signs ("an exact average, or an estimate of one") and the ghost sample ("no — it is a device, not data to
collect"), with a "Present in a real implementation?" column. Its prose states plainly: *"The probability 1 − δ
belongs to the draw of the sample, and covers every member of the class at once. It is not a probability attached to
any individual prediction, and the statement is not a deterministic promise about this particular dataset."* I
searched the full 87,655-character rendered text for per-prediction-confidence and deterministic-guarantee
phrasings and found none.

**Vacuity is stated, not flattered.** Figure 15 is captioned "Every one of these expressions is above the trivial
ceiling"; the five stacked bars are drawn at full length past a dashed line at 1 rather than clipped at it, every bar
visibly overshoots, and the text says *"The stack is drawn at full length rather than clipped at 1. Clipping would
hide exactly the thing worth seeing: how far past the point of saying anything these expressions are."* Its five raw
sums match my independent computation exactly. Figure 13 similarly refuses to draw four separately-valid one-sided
statements as a single confidence band, and says why.

**Figure 9.** The builder declined to redesign it, judging it legible with its angle arc and adjacent readout. **I
accept that judgement.** The arc is drawn, the two unit rays are visibly ~26° apart (`arccos(0.9) = 25.84°`), and
`25.8°` sits in a readout immediately beside the drawing with the sentence explaining that the inner product is the
cosine of that angle because both have length 1. The single combined label `φ(x₁), φ(x₂)` for two rays would be a
defect in a figure where the two vectors played different roles; here the quantity shown is the angle between them
and both have length 1, so which ray is which carries no information. The drawing is small relative to its column,
but small is not illegible.

Findings from the page are B1, B2, S5 and O8 below.

---

# Blocking

## B1 — Investigation 2 grades a comparison strictly tighter than the precision it prints, and returns ✗ beside two identical numbers

**Where.** `src/learn/components/lesson-labs/RademacherLabs.jsx:392` (and the same construction at `:440` and
`:214`).

```js
const difference = model.energyUpper - model.complexity;
return {
  outcome: Math.abs(difference) <= 1e-12 ? 'equal' : (difference > 0 ? 'below' : 'above'),
```

**What is wrong.** The builder fixed this defect class for the *numeric* field — `RademacherShared.jsx:345-359`
correctly sets the allowance to half a unit in the last printed place, and the caption promises "the number you read
here is a number you can type back". The *categorical* comparison beside it was left at `1e-12`, while both compared
quantities are printed to six decimals. Half a unit in the last printed place is `5e-7`. Any true gap between
`1e-12` and `5e-7` is therefore invisible on the page and graded as a miss.

**It is reachable one keystroke from a shipped preset.** Load "Perpendicular (1, 0) and (0, 1)" and change x2
coordinate 1 from `0` to `0.001` — a legal value, since the field is `decimals={3}`. I drove this in the live page.
The verdict returned is:

> ≠ You recorded **Exactly equal to the energy bound**; the calculation gives **Strictly below the energy bound**.
> You wrote 0.707107 for the complexity; the calculation gives 0.707107, **within** 5.000 × 10⁻⁷ of it. The exact
> value is **0.707107** and the energy bound is **0.707107**. The bound replaces an average length by a
> root-mean-square length, which is never smaller — and the gap is exactly the directional information it discards.

The readout directly beneath shows *three* identical values: "Exact empirical complexity 0.707107", "Feature-energy
bound 0.707107", "Largest-row bound 0.707107". So in one verdict the page tells the learner their number is right,
their category is wrong, and prints the two quantities it claims differ as the same number — then asserts a gap.
Screenshot: `scratch/rad-review/shots/grading-defect-inv2.png`.

**Why it matters.** This is precisely the defect the contract in `RademacherShared.jsx:345-352` exists to prevent
("A page that grades its own displayed value wrong is the defect this whole contract exists to prevent"), reappearing
in the field one line above the one that was fixed. It is worse than the original `0.333333` case, because here the
two halves of the same verdict contradict each other.

**How I verified it.** Independently, by computing the exact quantities for `x1 = (1,0)`, `x2 = (0.001,1)`, `B = 1`:
complexity `0.70710679...`, energy bound `0.70710688...`, gap `8.84e-8`. Then by a random sweep over legal
three-decimal coordinate/budget combinations, which found **615** reachable inputs whose two quantities print
identically at six decimals yet grade as `below` (`scratch/rad-review/gradegap.py`). Then by driving the live page
and reading the verdict out of the DOM.

**Scope.** The `:440` kernel stage is **safe** and I want to be precise about that rather than sweep it in: it prints
ten decimals, so its threshold is `5e-11`, and the smallest non-zero similarity on a three-decimal field is `0.001`,
which produces a gap of `8.8e-8` — over three orders of magnitude above the printed resolution. The `:214` case in
investigation 1 is latent rather than reachable at the default size; see O5.

## B2 — Figure 8 draws the same unit vectors and the same unit ball at two different scales, in a figure captioned "Same lengths"

**Where.** `src/learn/data/rademacher-models.js:744-750` (`ballGeometry`), rendered as figure 8 by
`RademacherFigures.jsx`.

```js
const extent = Math.max(
  radius,
  ...arrows.map(arrow => Math.max(Math.abs(arrow.signed[0]), Math.abs(arrow.signed[1]))),
  Math.abs(best.v[0]), Math.abs(best.v[1]),
  1e-6,
);
const unit = usable / extent;
```

**What is wrong.** Figure 8 is three small multiples with the caption **"Same lengths, same upper bound, different
exact answers"** and the prose "Three two-observation samples, all with unit rows and therefore the same
feature-energy bound 0.707107." Each panel autoscales independently, and `extent` includes the signed sum. Panel 1
("Two copies of (1, 0)") has signed sum `(2, 0)`, so `extent = 2`; panels 2 and 3 have signed sums of maximum
coordinate 1, so `extent = 1`. I measured the rendered SVG:

| Panel | viewBox | ball radius drawn | input arrow length drawn |
| --- | --- | --- | --- |
| 1 — two copies of (1, 0) | `0 0 150 150` | **26.5** | **26.5** (two coincident arrows) |
| 2 — perpendicular (1, 0) and (0, 1) | `0 0 150 150` | **53** | **53** |
| 3 — the perpendicular pair, rotated 90° | `0 0 150 150` | **53** | **53** |

The same unit vector is drawn at half length, and the same radius-1 budget ball at half radius, in a figure whose
entire point is that the lengths and the budget are identical across the three samples.

**Why it matters.** The failure mode is not subtle. The title also says "different exact answers" (0.5 versus
0.707107), so the natural reading of the one visibly smaller circle is that it depicts the smaller answer. It does
not — the circle is the budget, which is the same in all three. The figure teaches the wrong mechanism for the very
difference it exists to explain. And the quantity that set panel 1's scale is not drawn: an element census shows each
panel contains 4 lines, 1 circle and 2 arrowhead polygons, so panel 1 shows only its two coincident inputs and the
ball — a reader has no way to discover why the scale moved.

**Why no verifier catches it.** The falsification harness has a case for exactly the adjacent defect — case 10, "norm
ball drawn with two axis scales", which breaks the `y` term of `unit`. That asserts the x and y scales agree *within*
one panel. Nothing asserts that scales agree *across* the panels of one small-multiple figure, and `ballGeometry` is
called once per panel with no shared extent.

**How I verified it.** By reading the rendered image, then by measuring `r` on each `<circle>` and the endpoints of
each `<line>` through the live DOM (`scratch/rad-review/probe8.cjs`), then by reading `ballGeometry` and confirming
the arithmetic: `extent` 2 versus 1 gives exactly the factor of 2 measured.

---

# Should-fix

## S1 — Two verifiers write their evidence JSON *before* their final assertions, so the recorded totals are wrong and a failing run can be recorded as passing

**Where.** `scripts/verify-rademacher-data.py:700` (write), `:717` (`totalChecks`), `:754` (`"passed": not
failures`), versus `:759-761` and `:772-777`. And `scripts/verify-rademacher-examples.py:338` (write), `:371`
(`oracles`), `:388` (`passed`), versus `:393` and `:403-405`.

**What is wrong.** In both scripts the evidence file is written, and then more assertions run.

*Consequence one, cosmetic but load-bearing for the records.* The data verifier prints
`PASS: 5,785 data checks` (`:782`) but records `"totalChecks": 5782`. I ran it twice; the only difference between the
two evidence files was `checkedAt`, so this is deterministic, not flaky. The three missing checks are the floor
guards at `:759-761`. The examples verifier likewise prints `76 oracle assertions` (`:408`) but records
`"oracles": 75`, the missing one being the floor at `:393`. Every downstream number inherits one or the other: the
brief and the coordinator's summary quote 5,785 and 76; `design.md` quotes 5,782 and 75. **Both are "right"**, which
is the problem — the verifier reports one number on completion and stores a different one, and there is no way to
tell from the artefact which was meant.

*Consequence two, serious.* `"passed": not failures` is evaluated at write time. The assertions that run afterwards
include the three coverage floors **and, in the data verifier, the byte-identity check at `:772-774`** — the guard
that stops the generated module being hand-edited. If the module is not byte-identical, the process exits non-zero
but `docs/teaching/evidence/rademacher-data.json` has already been written saying `"passed": true`. Falsification
case 21 ("generated data module edited by hand") exercises exactly that path.

**Why it matters.** This is the "record asserting a state the tree contradicts" class, mechanised. It is also a
one-line fix in each file: move the evidence write below the final assertion.

**How I verified it.** By running both verifiers and diffing their output against the evidence they wrote, and by
counting the `check()`/`oracle()` calls that appear after the write line in each source.

## S2 — The falsification harness leaves `rademacher-sources.json` written from a deliberately broken source, and never re-runs that verifier clean

**Where.** `scripts/verify-rademacher-sources.py:219-250` (evidence written, including `"passed": not problems`,
*before* `raise SystemExit`), driven by `scripts/falsify-rademacher.mjs` cases 17–20 (`:193-228`).

**What is wrong.** The builder clearly understood this hazard — it protects the models verifier with `--no-evidence`
(`falsify-rademacher.mjs:32`, honoured at `verify-rademacher-models.mjs:1608-1610`), and it re-runs the *browser*
verifier once against the repaired build at `:489-496` with a loud failure if that clean run does not pass. But four
cases drive `verify-rademacher-sources.py`, which writes its evidence unconditionally before exiting, and the harness
neither passes it a `--no-evidence` flag (it has none) nor re-runs it clean. A full harness run can therefore leave
`docs/teaching/evidence/rademacher-sources.json` recording `"passed": false`, derived from a defect the harness
itself injected.

This is the same defect the builder found and fixed for the browser artefacts, surviving in one of the three writers
it did not cover. The third, `verify-rademacher-examples.py`, is safe only by luck of ordering — its served-copy
assertion at `:163-167` precedes its evidence write.

**Why it matters.** The falsification harness exists to prove the guards can fire. An artefact it corrupts on the way
is an artefact a later reader will trust.

**How I verified it.** By reading the write-then-exit ordering in the sources verifier and the case list in the
harness. I did **not** run the harness — see O11 — so I am reporting a code path, not an observed corruption. The
timestamps are consistent with it having happened: `rademacher-falsification.json` records
`checkedAt: 2026-09-19T16:02:35.559Z`, and `rademacher-sources.json` was last written `2026-09-20T04:01:27` by a
later clean re-run.

## S3 — An unanchored breakage is misreported as a concurrent edit and double-counted

**Where.** `scripts/falsify-rademacher.mjs:347`, `:350-353`, `:373`, `:379-384`; identically in the browser pass at
`:426-429` and `:456-461`.

**What is wrong.** When a breakage's anchor text does not occur exactly once, the harness sets
`outcome = 'unanchored'`, increments `failures`, and — correctly — writes nothing, leaving `brokenDigest` at `null`.
The `finally` block still calls `restore()`, which computes a real digest of the untouched file, compares it against
`null`, concludes the bytes changed underneath, and returns `concurrentEdit: true`. Line `:379-384` then overwrites
the diagnosis with `outcome = 'concurrent-edit'` and a message telling the operator that something else modified the
file and "this verdict is not trustworthy", and increments `failures` a second time.

**Why it matters.** `design.md` specifically credits the harness with catching one of its own anchors going stale and
reporting it as `unanchored`. That capability is destroyed by this bug: the next stale anchor will be reported as a
phantom concurrent edit, sending the operator to look for a process that does not exist. The fix is to guard the
branch with `brokenDigest !== null`.

## S4 — The harness has no signal handling, and the one-line "did the breakage apply" check is absent despite the value being in hand

**Where.** `scripts/falsify-rademacher.mjs` — no `process.on('SIGINT'|'SIGTERM'|'exit')` anywhere; `:342`
(`originalDigest`), `:356` (`brokenDigest`).

**What is wrong.** Restore is in a `finally` (`:372-374`, `:448-451`), which covers thrown exceptions. It does not
cover Ctrl-C or a kill. A SIGINT or an out-of-memory kill mid-case leaves a deliberate one-line defect applied to a
tracked source file, and under `--browser` also leaves a broken bundle in `dist-rad` that a running preview keeps
serving. Nothing in the harness warns the operator that `git status` and `git checkout --` are the recovery, and
`rademacher-falsification.json` — the only other tell — is written at the very end and so simply will not have moved.
On this machine, which has killed processes four times today, that is a live hazard rather than a theoretical one.

Separately, `originalDigest` is computed at `:342` and **never read again**. The exactly-one-occurrence precondition
at `:344` plus `String.replace` rules out a silent no-op unless `find === replace`, which nothing checks. The check
that would close this — comparing `brokenDigest` against `originalDigest` — is one line away with both values already
in scope, and is not made. No defined case is currently tautological; the harness would not notice if one became so.

**What is genuinely fixed.** The 709-capture near-miss is properly closed, and I want to say so plainly. The cleanup
at `:480-488` is a flat `readdirSync` of one directory with a double filter — `startsWith('rademacher-')` **and**
`endsWith('.png')` — and a non-recursive `fs.rmSync`. There is no glob, no `recursive: true`, no path joining on
external input. It cannot reach any of the 724 non-`rademacher` captures in the shared directory, and a directory
happening to be named `rademacher-x.png` would throw rather than be deleted. Restore is byte-for-byte: the original
bytes are held as a `Buffer` (`:341`), written back with no encoding argument (`:308`) and re-verified by SHA-256
(`:310-315`). The residual gaps are that the block at `:471-496` sits outside any `try/finally` and deletes the 29
good captures before re-capturing them with no backup, so a failed clean run leaves zero captures beside a
`rademacher-browser.json` that still lists their digests.

## S5 — Figure 13 draws a one-sided bound as a two-ended segment, and two of its four rows show the exact value outside the drawn bar while the table says "yes"

**Where.** Figure 13, `monteCarloLayout` (`rademacher-models.js:976`), rendered by `RademacherFigures.jsx`.

**What is wrong.** Each row draws a dot at the estimate and a bar running rightwards to the one-sided upper endpoint,
with a vertical line at the exact value 0.5. The adjacent table has a column "Covers .5?" reading `yes` for all four
rows. For `T = 16` and `T = 64` the estimate is below 0.5 and the bar crosses the line, so the picture and the column
agree. For `T = 256` (estimate 0.531250) and `T = 1024` (estimate 0.519531) the estimate is *above* 0.5, the drawn
bar lies entirely to the right of the line, and the column still says `yes`.

Both are correct for a one-sided upper bound — the covering region is `(−∞, endpoint]`, and `0.5 <= 0.607742` — but
the drawn segment is `[estimate, endpoint]`, which is not the covering region. A reader checking the two against each
other sees the exact value outside the bar and the table asserting coverage.

**Why it matters.** The confusion it invites — reading a one-sided bound as a two-sided interval — is one of the
specific confusions this lesson exists to prevent, and the figure is otherwise excellent about exactly that (its
prose correctly refuses to merge the four rows into one band). Either extending the bar leftwards off-axis, or
renaming the column to something like "endpoint above .5?", would close it.

## S6 — Three count and attribution claims in the appended `design.md` sections contradict the tree

**Where.** The Phase A / Phase C sections appended to
`docs/teaching/drafts/rademacher-complexity-generalization-bounds/design.md` (88 added lines, 0 deleted).

Most of it checks out — I confirmed 17 figures, 4 investigations, 8 practice items, 50 model-check groups, 319
grouped checks, 73 property statements, 104 hygiene checks over 16 files, 8 browser cases, 29 captures with 29
distinct digests and 29 files on disk, 25 of 25 falsification cases caught (22 non-browser plus 3 browser, every
recorded outcome `caught`), the trust-root arithmetic, the preserved original-body hash, and the untouched topic
note. Three claims do not:

1. **"1,115 drawn-geometry checks"** — `rademacher-models.json` records `sweeps.drawnGeometryChecks: 1205`, and the
   verifier's own PASS line prints "1,205 drawn-geometry checks". Every other sweep number in the same sentence is
   exact (1,005 kernel cases; 7,230 ramp comparisons; 410 movement verdicts; 16,001 loss samples; 252 hull cases).
2. **The exclusion breakdown does not sum.** design.md writes "12 declared exclusions (three library-version
   strings, two prose statements, five SLSQP iteration counts)" — that is 10. The total of 12 is right; there are
   **four** prose exclusions, not two (`/convention`, `/selection/rule`, `/type`, `/date`). The same mis-description
   was carried into the brief I was given.
3. **The BLAS-threading claim is inverted.** design.md says pinning the thread-count variables to 1 moves the `B = 4`
   coefficient from `-3.1888250733899115` to `-3.188825073389911`. But the packet's `experiment-results.json` — and
   `rademacher-data.js:733` — carry `-3.188825073389911`, the value described as the *drifted* one, while
   `rademacher-native.json` records the default-threaded regeneration as byte-identical. The direction is the wrong
   way round. The same wording appears at `scripts/verify-rademacher-examples.py:132`, so it is a consistent
   mis-statement in two places rather than a transcription slip. The substance — that a last-bit difference exists,
   that it is measured separately, and that default threading is what the byte-identity check uses — is sound.

## S7 — One Phase C fix is attributed to the wrong figure, and "the repairs are geometric" over-generalises

**Where.** The Phase C section of `design.md`; `rademacher-models.js:752-772` (`supportLabel`);
`RademacherFigures.jsx:337-338`, `:493-497`, `:438-444`.

design.md pins the `supportLabel` perpendicular-offset repair to "the `ℓ₂ best` label in figure 10". In the tree
`supportLabel` is consumed by **figure 7** (the `w*` label). Figure 10's `ℓ₂ best` label was fixed by a different
mechanism — moving it into a legend strip below the plot. And the sentence "The repairs are geometric and asserted"
covers four defects, one of which (figure 9) was repaired by relocating the angle into the readout, which is not
geometric and is not asserted by a geometric verifier check. The mechanisms all exist and all but figure 9's are
asserted; the record maps them to the wrong figures.

---

# Observations

**O1 — The mathematical layer is clean.** Across roughly a hundred independently recomputed quantities — six
enumerated classes with their full positional arrays, seven ball geometries, three kernels, three Massart values, two
growth functions, Sauer sums, the contraction step, the finite-population symmetrization example, four Monte-Carlo
rows plus the unit-ball record, and a complete refit of the real experiment with a different optimiser — **I found no
disagreement**, apart from the one-ulp case in A1 where the implementation is the more accurate of the two. I want
this stated as a finding rather than an absence: the enumeration order trap the builder flagged is genuinely closed,
and it is closed on the arrays and not merely on the aggregates.

**O2 — The sign-pattern ordering is documented as well as correct.** `rademacher-models.js:116-126` explains *why*
the order is not cosmetic and what the misalignment would have hidden. That comment is the reason a later editor will
not reintroduce the bug, and it is worth more than the test.

**O3 — The "eleven required statements and three forbidden phrasings" have no automated guard.** I could find no
assertion for them in any of the five verifiers or the harness. The substance is present in the page — I checked the
rendered text myself and it is good (see Part B) — but unlike almost everything else in this lesson, that property is
held by a manual pass with no regression guard. Given the lesson's central risk, a handful of `assert.match` calls in
the browser verifier would be cheap insurance.

**O4 — B1's latent twin in investigation 1.** `RademacherLabs.jsx:214` uses the same `1e-12` threshold with
six-decimal display. The row entries are `decimals={3}`, so the smallest non-zero value of `exact − single` is
`0.001 / (n · 2^n)`. At the default `n = 3` that is `4.2e-5`, comfortably visible; at the maximum `n = 8`
(`limits.maxLabColumns`) it is `4.88e-7`, just under the `5e-7` printed resolution. Reachable only with a very
contrived eight-column setup, so I am not raising it as a defect in its own right — but whatever fix B1 gets should
be applied here too.

**O5 — Figure 8 shows two identical inputs as one arrow.** Panel 1's two `(1, 0)` inputs are drawn as two exactly
coincident lines with two coincident arrowheads. The panel label carries the "two copies" claim, so nothing is
stated wrongly, but the picture of the duplicate-row case — the case the figure exists to contrast — shows one arrow.
An offset, a doubled stroke or a `×2` annotation would show it. Secondary to B2, same figure.

**O6 — "709 other captures" is not reproducible.** The shared screenshots directory currently holds 753 PNGs, 724 of
them non-`rademacher`; 680 are git-tracked. None of those figures is 709. The number is plausible for some
intermediate moment and the disclosure is honest about the near-miss; it just cannot be checked from the tree.

**O7 — The blueprint-registration claim is now stale, not false.** design.md says the blueprint is "not registered;
`blueprints/index.js` belongs to the increment owner". It is now registered at `blueprints/index.js:28` and `:165`,
in the same edit that registered pac-learning and calibration — i.e. by the increment owner afterwards. True when
written.

**O8 — No accessibility testing, and that is a departure from the packet, not only a limit.** design.md discloses
"no assistive-technology testing", and I confirmed there is no `axe-core`, `pa11y` or `lighthouse` anywhere in
`package.json` or `scripts/`, and no accessibility assertion in any rademacher verifier. What is not disclosed is
that `visual-specifications.md:99` explicitly asked for it ("test invalidation, input bounds and **accessibility** in
the real reader"). The gap is stated; its status as an unmet spec item is not. The single-browser-engine limit is
accurately disclosed (`verify-rademacher-browser.cjs:162`, `chromium.launch({ channel: 'msedge' })`).

**O9 — The investigation contract holds.** I exercised it directly. Nothing graded is revealed on first paint; each
stage requires a radio selection and, where required, a numeric commitment before "Apply and check" enables; grading
is against `committed` state with the pending-draft banner appearing when fields diverge; ties are handled by
`answer.acceptable` at `RademacherShared.jsx:343` and the verdict names the other correct answers; the numeric
allowance is half a unit in the last printed place. Investigation 2's stage-1 question even handles the
genuinely-undefined case ("a maximising coefficient does not exist when the signed sum is zero") as neither a near
miss nor zero. Figure 2 marks *all* tied winners in its highlighted cells, not just the first — I checked columns σ3
and σ6, which are the two tied patterns in that class. The two fixed defects are real fixes, not claims.

**O10 — The destination note: the builder's reading is correct and its recommendation is right.** I read
`docs/teaching/topic-notes/rademacher-complexity-generalization-bounds.md` claim by claim against the tree. It is
git-clean and untouched, as claimed. Its status line — "production implementation and independent verification are
pending" — is stale in exactly the one direction the builder describes. Every substantive instruction in it is
honoured: the no-absolute-value `1/n` convention is preserved (and the absolute convention used exactly once, to show
a convention is not typography); the loss class is named in the theorem (figure 4, "twice the complexity of the LOSS
class"); empirical complexity, its expectation and sign simulation versus new sampling are distinguished (figures 4
and 13); the disjoint 80/240/80/80 roles are implemented and I verified the source-row ids myself; the vacuous bounds
sit beside improving measured validation performance and are not replaced by an invented curve; and the fixed-corpus
diagnostic is not presented as an iid deployment certificate. The recommendation — keep it open, restate the status
as implemented with independent review pending, and add a line about the duplicate-feature disclosure now travelling
in the served attribution — is correct, and I verified the last part directly: that disclosure is in
`public/learn-assets/rademacher/ATTRIBUTION.txt` and matches my own recomputation exactly. I did not edit the note.

**O11 — I did not run the falsification harness, deliberately.** The brief asked me to. I judged against it and want
the reasoning on the record rather than silently skipped. The harness writes tracked source files with no signal
handler (S4) on a machine that has killed processes four times today for memory and spend limits, so a kill during my
run would have left a deliberate defect in the tree for the next reader. More decisively, running it would have
written `docs/teaching/evidence/rademacher-falsification.json` and could have corrupted
`docs/teaching/evidence/rademacher-sources.json` (S2) — both records I was instructed not to edit, and I had no
permitted way to repair the second afterwards. Instead I verified the harness statically in full: every mutated file,
the backup and restore path, the byte-for-byte guarantee, the screenshot cleanup scoping, the case list, and the
attribution requirement that each case's verifier output must contain the expected guard text (`:362-366`). I also
confirmed the recorded clean run is internally consistent — `breakagesApplied: 22`, `caught: 22`, `skipped: 0`, three
`browserResults` all `caught`, every entry carrying a populated `guardNamed`, and no `concurrent-edit`,
`misattributed`, `survived`, `restore-failed` or `unanchored` outcome anywhere in it. That is as far as I can
responsibly take "the 25/25 is real" without running it; a reviewer with a quiet machine and permission to rewrite
evidence should close the gap.

**O12 — What the verifier suite does not assert.** Worth recording, since two of my findings live in these gaps.
Nothing checks that small-multiple panels of one figure share a scale (B2). Nothing checks that a graded categorical
threshold is no tighter than the figure's own printed precision (B1) — the browser verifier does pin the *numeric*
tolerance regression at `:384-391`, but only for the value field. Nothing exercises the investigations at degenerate
inputs — exact ties are pinned for investigation 1 and investigation 4, but zero complexity, a vacuous bound at
exactly 1, and near-ties are not. `rademacher-falsification.json` records no `browserCaught` count and no `failures`
count, so the headline "25 of 25" is not recoverable from the artefact — `caught` and `breakagesApplied` both exclude
the browser pass, and a reader taking their ratio silently drops the three browser classes.

---

# Summary

The mathematics is the strongest part of this lesson and I could not break it. Every finite Rademacher complexity,
growth function, Sauer sum, Massart bound, ball and kernel geometry, contraction step, symmetrization example and
Monte-Carlo record reproduces exactly under independent exact-rational or 50-digit recomputation, and the real
experiment reproduces from the served CSV under a different optimiser down to the last recorded digit of the energy
factor and the confidence term. The sign-pattern ordering trap is genuinely closed, on the positional arrays and not
just the aggregates. The trust-root count is a real mechanism with empty `uncovered` lists and individually reasoned
exclusions, and the word-count disagreement was counted right and recorded for the right reasons. The central
pedagogical risk is handled better than the brief led me to expect: figure 4 separates the three randomness lanes and
says in terms that 1 − δ is not a per-prediction confidence, and figure 15 draws ten vacuous bounds at full length
past the trivial ceiling and says so.

Two blocking defects. **B1** is the lesson's own named defect class reappearing one field over: investigation 2
grades a three-way comparison at `1e-12` while printing both quantities to six decimals, so a learner one keystroke
from a shipped preset gets a ✗ whose own text prints the two "different" values as `0.707107` and `0.707107` — and
whose numeric half simultaneously says they were right. **B2** is a small-multiple figure captioned "Same lengths"
that draws the same unit vector at 26.5 px in one panel and 53 px in the next, because each panel autoscales to a
signed sum it does not draw.

Five should-fixes, of which two matter beyond tidiness: the two Python verifiers write their evidence before their
final assertions, so their recorded totals are permanently three and one short of what they print and a failed
byte-identity check can be recorded as `"passed": true` (**S1**); and the falsification harness, which the builder
correctly taught not to overwrite the browser evidence, still leaves the *sources* evidence written from a
deliberately broken file and never re-runs that verifier clean (**S2**). The screenshot near-miss that threatened 709
other lessons' captures is properly and narrowly fixed, and restore is genuinely byte-for-byte.

The records are mostly accurate — seventeen counts checked out — but three claims in the appended `design.md`
contradict the tree: a drawn-geometry count of 1,115 against a recorded 1,205, an exclusion breakdown that sums to 10
rather than 12, and a BLAS-threading statement whose direction is inverted and which is mirrored in a verifier
docstring. The destination note is untouched and the builder's reading of it is correct; its recommendation should be
followed.
