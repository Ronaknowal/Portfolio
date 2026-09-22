# PAC Learning & VC Dimension — independent phase-two review

Reviewed 20 September 2026 against the working tree plus the uncommitted PAC implementation. The reviewer
authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/`,
`docs/teaching/drafts/` or `docs/teaching/topic-notes/` was edited; this review document is the only write
outside `scratch/`. Throwaway scripts live under `scratch/pac-independent/`. No state-changing git command was
run. The build directory `dist-pac/` was deleted and the preview on 4192 stopped at the end.

Two housekeeping notes, stated up front because both bear on trust in the record:

- I ran `falsify-pac-guards.mjs --browser` **once**, never concurrently with anything, having first copied all
  ten files it mutates and all 46 `pac-*` evidence artifacts to `scratch/pac-independent/backup/` and
  `scratch/pac-independent/evidence-backup/`. Afterwards every file in the tree matched the SHA-256 baseline I
  took before starting (`scratch/pac-independent/BASELINE-HASHES.txt`), and all 40 screenshots were
  byte-identical to my pre-run copies.
- Re-running `verify-pac-data.py`, `verify-pac-examples.py` and `verify-pac-sources.py` **overwrote their
  evidence JSON**, because only `verify-pac-models.mjs` offers `--no-evidence`. I diffed the results (every
  substantive field reproduced identically; only `checkedAt` differed) and then **restored all six evidence
  files from my pre-run copies**, so the record on disk is exactly what the builder left. See S9.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/pac-independent/`, importing neither `pac-models.js` nor `pac-data.js` nor `pac-examples.js` nor any `verify-pac-*` script nor the packet's `pac-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.1, mpmath 1.3.0). **546 checks** in `exact.py` — exact `Fraction`/`int` wherever a rational answer exists, 50-digit `mpmath` for every transcendental bound. A **100,720-case** feasibility sweep written from the definitions and cross-checked by closed form. An exact **Fourier–Motzkin** decision procedure for half-plane realizability, run over every labeling of all four packet configurations and over **58,905 four-point subsets** of a rational grid. A full independent replay of the banknote learning curves from the **served** CSV, and of the seed-41 interval simulation. Reviewer-written Playwright drives (`drive.cjs`, `radicals.cjs`, `labs.cjs`, `figs.cjs`) sharing no code with `verify-pac-browser.cjs`, at 1366, 390 and 320 px on Edge. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `checked-results.json`, `pac-calculations.py`, `banknote-learning-curves.py`); the destination note; the lesson body, model layer, both generated modules, all three lab/figure/shared components and the CSS; all five verifiers and the harness; the blueprint; the served assets. |
| **Reused (declared)** | scikit-learn's estimators at the declared hyperparameters and NumPy's `default_rng`, because the protocol *is* those calls — as the builder states. Everything around them (split construction, prefixes, prediction comparison, counting, quantiles) is written here. One subagent for a fan-out read of verifier internals. **Every finding a subagent surfaced I re-derived myself before reporting it** — I read each cited line, ran the instrumented count in S4 myself, and measured S6 in the live page. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/pac-learning-vc-dimension.jsx` | `9a2499ce0c9024a81ab6f997c4b8983b9dc1688e309b2e1d584446583b40cd19` |
| `src/learn/data/pac-models.js` | `36d80a2f5b2960a57543b3f0aba2f26801fadeb8669bd11fe0ee265a967ffa9c` |
| `src/learn/data/pac-data.js` | `8754a2d5228e6f5a391df5d7bc9ca385aac4c6083ba2e6b497106d33b3131285` |
| `src/learn/data/pac-examples.js` | `2ad31dd00134354d3de0a6176bf4d4fae16cc6c39cc8ab47db3c1b72c429d05a` |
| `src/learn/components/lesson-labs/PacShared.jsx` | `eb7ccc0b50835d8452f7f809708a1bcc8a77113efeaa00c49fb55cedcc88161b` |
| `src/learn/components/lesson-labs/PacLabs.jsx` | `3aa4e4c99acab04fa4d5f2b79a4e6f1fda04b6828ec9d87150aa9f9a551f33f8` |
| `src/learn/components/lesson-labs/PacFigures.jsx` | `19ad116c585cf89becc5549ec05dbd35528a5deb32e8bf812f599bd0be4f32a6` |
| `src/learn/components/lesson-labs/pac-labs.css` | `2943d4850ec64d73571e4ff31694affc9b4c39c90abf4d20d814af6ab3858d28` |
| `scripts/verify-pac-models.mjs` | `38071bf83a3221f5eb17cb1731a8d1c58fafcc746280c91e059dae63b474e379` |
| `scripts/verify-pac-data.py` | `1fd2c28ca97caf0df03ce7fe429afba7c7ad861434698584817194baf3813608` |
| `scripts/verify-pac-examples.py` | `59a28e5b2c96571db68899b13bfc5b49d9f738a2421cc4497991bb33ccf55657` |
| `scripts/verify-pac-sources.py` | `b338509923f34f2bd77edf0c50827952afdfe0dac3812eab22d0425ab356f07e` |
| `scripts/verify-pac-browser.cjs` | `17c872624e0ed6bcf56e53e62a0023dfaa750ff602a5bf7624df8c1a4a2707d5` |
| `scripts/falsify-pac-guards.mjs` | `2314e3177d4c678e9336c145d02023230d98a959ea771e0dace3790061cf8c9c` |
| `docs/teaching/drafts/.../lesson.md` | `8fcc973b3e4960eb140b7274c8f3eeddf3b0ba2e14c7534f10297392c5dbce14` |
| `docs/teaching/drafts/.../design.md` | `9285c263839a188aac49c7b5780944d5dda2d03a445b8501e06724c7fb7067d9` |
| `docs/teaching/drafts/.../checked-results.json` | `b12b1e3ce5c67554b9fd55e5498b0b1ffa0b0d884a1e5b6e1d330a1e57be9217` |
| `docs/teaching/topic-notes/pac-learning-vc-dimension.md` | `0f5ecd803cfd9a9034abf7e9c21ab7c259c774829be6f9c6043b10cffb0c9a28` |

---

# Part A — correctness

## A1. Every bound expression, at 50 digits — **no disagreement**

`scratch/pac-independent/exact.py`. Each formula re-implemented from its statement, not from the module.

- **Finite-family radius.** `finiteRadius(K, n, δ) = sqrt(ln(2K/δ) / (2n))`. At K=25, n=500, δ=.05 the true real
  value is `0.0831129068134554962519547203748…`; the nearest IEEE double, expanded exactly, is
  `0.08311290681345549769…`, and JavaScript's `Math.sqrt(Math.log((2*25)/0.05)/(2*500))` produces **that same
  double**. Both round-trip to the shortest decimal `0.0831129068134555`. The note's `.08311291` follows.
  I confirm the arithmetic identity the note asserts: `2K/δ = 1000` and `2n = 1000` are both exact in binary
  floating point (`50/0.05 === 1000` exactly), so `r_K` **is** `sqrt(log(1000)/1000)`, not merely close to it.
- Single-rule radius `0.060736146190830516`, K=3 radius `0.069191702846382137`, practice 2's
  `sqrt(ln(1200)/1600) = 0.066567995481012175` — all reproduce, and `2*12/.02 === 1200` is likewise exact.
- **Explicit VC radius** `r = sqrt((32/n)(d ln(en/d) + ln(8/δ)))`. At d=2, δ=.05: `2.183517882763692` at
  n=100 and `0.095857778624884719` at n=100,000 — the two values §7 prints at six decimals.
- **BEHW realizable sufficient size**, base-2 logs, both conditions. At d=2, δ=.05 the two conditions evaluate
  to (106.44, 481.79), (212.88, 1123.58), (425.75, 2567.16) at ε = .2, .1, .05, giving **482, 1124, 2568** —
  the three numbers §7 prints. The second condition dominates at all three, so the "larger of the two" framing
  is doing real work rather than decorating a single binding constraint.
- **Finite realizable condition.** `(ln 32 + ln 100)/.05 = 161.418121776…`, so the prose's "about 161.42" is
  correct and the ceiling is **162**.
- Two-strip bound `2(1-ε/2)^n` at ε=.1, n=200 is `0.000070105332497658006`, printing as `0.0000701053`;
  `16e^{-1} = 5.886071` and `16e^{-6} = 0.039660` reproduce.

**No disagreement with any bound value on the page.** Two apparent mismatches in my first run were my own
harness passing `δ` as a float where an exact rational was wanted; corrected, they agree to 40 digits.

## A2. Growth functions, Sauer, and the exact constructions — **no disagreement**

Pattern counts by **three** independent routes: my own endpoint enumeration, brute force over all 2^n vectors
testing contiguity (intervals) or suffix-of-ones (thresholds), and the closed forms. All three agree for
n = 1…10. Interval counts `2, 4, 7, 11, 16, 22, 29, 37, 46, 56`; threshold counts `2…11`. The page's
7 patterns on three points, 11 on four, 16/32 at n=5 and 56/1024 at n=10 all check out. The missing pattern on
three points is exactly `101`, and the five impossible four-point patterns are exactly
`0101, 1001, 1010, 1011, 1101` — derived, not matched against a literal.

`sauerSum(100, 5) = 79,375,496` exactly. Sauer's bound genuinely dominates both classes for n = 1…12, and
`(en/d)^d >= Sauer(n, 2)` holds throughout, so the §6 inequality chain is not merely asserted.

**The sine construction.** `r` with first n bits `1-y_i` then `01`, `θ = 2πr`. For `[1,0,1,1]`, `r = 17/64`,
cycles `17/64, 17/32, 1/16, 1/8`, prediction `[1,0,1,1]` — the figure's values exactly. I checked all 16
four-bit labelings two ways: by exact rational cycle position, and against the **actual floating-point sign of
`sin(θx)`**. Both agree on all 16. I also confirmed the construction shatters all 2^n labelings for
n = 5, 6, 7, 8 — the "infinite VC dimension" claim is exercised beyond the displayed case.

**The exact four-input world.** Occupancy over seen-masks in `Fraction`, cross-checked for n ≤ 8 by brute-force
enumeration of all 4^n draw sequences. Failure probability is exactly `(1/2)^n` at n = 1, 2, 4, 8, 16, 24;
`1/16` at n=4 and `1/16777216` at n=24 are exact, as printed. The all-zero target gives exactly 0. I also
verified, over every sample of length 0–4, that the **lexicographically first consistent rule equals** the
closed form "observed bits from the target, 0 elsewhere" — the module computes one and the page describes the
other, and they are the same function.

## A3. The feasibility sweep — **no disagreement, and both verdicts are genuinely exercised**

The builder reports 100,720 cases over every ordering of 2–6 points × every labeling, 23,984 feasible and
76,736 not. I derived this independently, twice.

By **closed form**: over all `n!` orderings, thresholds admit `n+1` labelings and intervals admit
`1 + n(n+1)/2`, so feasible = `Σ n!(n+1) + Σ n!(1 + n(n+1)/2)` for n = 2…6 = `5,910 + 18,074` = **23,984**, out
of `2 · Σ n! 2^n` = **100,720**. By **explicit enumeration** in `exact.py`, sorting labels by each permuted
coordinate and testing contiguity/suffix: the same 100,720 / 23,984 / 76,736, with the per-n breakdown
`interval 8/42/264/1920/15840`, `threshold 6/24/120/720/5040`.

**Both verdicts are non-vacuous, and not only in aggregate.** Every n from 2 to 6 and both families contribute
feasible *and* infeasible cases; the smallest cell (n=2, threshold) is 6 feasible of 8. The verifier's floor
`assert(witnessCases >= 100000)` and the exact `assert(selectionCases === 256)` are real (see S5 for what is
*not* floored).

## A4. Half-planes, by exact Fourier–Motzkin — **no disagreement**

The builder replaced the packet's linear program with exact convex-hull intersection in `Fraction`. I used a
third method: realizability of `w·x + b ≥ 0` on positives and `< 0` on negatives, with the strict side
normalised to `≤ -1` by positive scaling (valid because the system is positively homogeneous), decided by
**exact Fourier–Motzkin elimination** of `w₁, w₂, b` in `Fraction`. I cross-checked it against a second route
(no direction `w` satisfies `w·(p−q) > 0` for all differences ⟺ `0 ∈ conv{p−q}`); the two agree on every case.

| configuration | recorded realizable / infeasible | my exact FM | recorded infeasible labelings | agree |
| --- | --- | --- | --- | --- |
| triangle `(0,0),(1,0),(0,1)` | 8 / 0 | 8 / 0 | — (shattered) | yes |
| square `(0,0),(1,0),(1,1),(0,1)` | 14 / 2 | 14 / 2 | `0101`, `1010` | yes |
| collinear `(0,0),(1,0),(2,0)` | 6 / 2 | 6 / 2 | `010`, `101` | yes |
| interior `(0,0),(2,0),(0,2),(.5,.5)` | 14 / 2 | 14 / 2 | `0001`, `1110` | yes |

These are the 8 / 14 / 6 the §5 prose quotes, and the two impossible square labelings really are the crossing
diagonals while the two impossible interior labelings really are "hull positive, interior negative" and its
complement — i.e. the geometric arguments the page makes are the arguments that decide these cases.

I also swept the **class-wide** claims that no finite computation can settle for the builder: over all 58,905
four-point subsets of a 6×6 rational grid, **not one is shattered**; among 7,140 triples, 6,768 are. That is
consistent with VC dimension exactly 3 and, importantly, with the page's own warning that such a sweep is
evidence and not the proof — the proof remains the §5 prose, which is correct as written.

## A5. The declared coverage limitation — **accurately characterised, not an overstatement**

This is item 3 of my brief and I treated it adversarially. The claim is 100% of 5,275 trust-root leaves, of
which 126 are "validated, not re-derived".

- The 126 paths listed in `trustRootLeavesValidatedNotReDerived` are **exactly** `w/0`, `w/1` and `b` for the
  42 recorded realizable labelings (8 + 14 + 6 + 14) across the four configurations: 42 × 3 = **126**, with
  nothing else smuggled in. I enumerated the list and checked it contains no other path shape.
- I validated **all 42 witnesses myself** in exact `Fraction` arithmetic (`scratch/pac-independent/witnesses.py`):
  every recorded `(w, b)` realizes exactly its recorded labeling under `w·x + b ≥ 0`, every recorded
  `minimum_signed_margin` matches my own minimum `|w·x + b|` to better than 1e-12, and every margin is
  strictly positive. Zero problems.
- **Which labelings are realizable at all is genuinely re-derived**, by me as well as by the builder, and by a
  different method again (A4). So the claim "we validate the witnesses but independently re-derive the
  realizability question, and that is the claim the lesson makes" is **exactly true**.

My own leaf count came to 5,274, one short. The difference is not a discrepancy: `leaf_paths`
(`verify-pac-data.py:137`) yields `prefix/[]` for an **empty list**, and there is exactly one in the trust root
— `/geometry/triangle/infeasible`, empty precisely because the triangle *is* shattered. 5,274 + 1 = 5,275.
Worth knowing when the figure is quoted (O8), but the accounting is sound.

**Verdict on item 3: describing this as 100% coverage with a stated caveat is honest.** The caveat is in the
right place, names the right leaves, and understates rather than overstates what was skipped — the skipped
quantity is one LP solution among infinitely many, and the mathematical content attached to it is re-derived.

## A6. The banknote learning curves, refit from the served CSV — **no disagreement**

`scratch/pac-independent/curves.py`, reading `public/learn-assets/pac-learning/banknote-subset.csv` (21,237
bytes, SHA-256 `d28fa993…`, matching the recorded provenance) and reusing only the scikit-learn estimators.

Reproduced exactly: 480 rows, 480 distinct `source_row` ids, class balance 275/205. **The selection claim is
independently confirmed** — the served ids are exactly `default_rng(23).permutation(1372)[:480] + 1`, in that
order. Splits 320/80/80, disjoint, covering all 480.

**All fifteen rows reproduce**, and so does **every one of the fifteen 80-element development prediction
vectors**, not merely the counts:

```
logistic_regression  20 19/20 76/80 | 40 37/40 77/80 | 80 74/80 77/80 | 160 156/160 77/80 | 320 313/320 79/80
rbf_svc              20 19/20 72/80 | 40 40/40 75/80 | 80 77/80 79/80 | 160 158/160 79/80 | 320 318/320 79/80
depth5_tree          20 20/20 63/80 | 40 40/40 64/80 | 80 80/80 74/80 | 160 160/160 76/80 | 320 317/320 76/80
```

Practice 9's numbers hold: at n=80 the tree fits **80/80** training labels and gets **74/80** development, the
SVC gets **77/80** and **79/80** — six errors against one. I read these off the rendered figure-10 table as
well, and all fifteen rows match.

**Test isolation is real.** No frozen `train_source_rows` list intersects the development split or the test
split, the frozen `development_source_rows` are exactly the 320–399 slice, and `test_evaluated` is `false`.

## A7. The retained seed-41 simulation — **no disagreement**

`scratch/pac-independent/sim.py`, with the fit, the symmetric-difference risk and the quantile rule written out
(NumPy's "linear" rule reimplemented rather than called). All five rows reproduce to 1e-12, including the
quantiles:

```
n=10  743 fails  mean .177947833  q05 .042913829  q50 .159417044  q95 .397552179
n=20  368        mean .092211845  q05 .017995293  q50 .080020198  q95 .202562067
n=50   46        mean .039486158  q05 .007319083  q50 .031992207  q95 .097127948
n=100   1        mean .019713288  q05 .003582084  q50 .016268780  q95 .047046280
n=200   0        mean .009839118  q05 .001996603  q50 .008281579  q95 .023542443
```

The observed failure fraction respects the two-strip bound at every size, and every plotted quantile is
strictly positive — so figure 9's logarithmic risk axis is admissible, as the model layer asserts rather than
assumes.

## A8. The radical fix, in the rendered page — **confirmed at all three widths**

This is the item I was told matters most, and it holds.

- The page emits **exactly three** `.katex .sqrt` nodes — §4's `r_K`, §7's `r`, and practice 2's
  `sqrt(ln(1200)/1600)` — matching the three radicals the lesson's LaTeX contains.
- At **1366, 390 and 320 px** all three paint at identical size: `78.45 × 36.39`, `161.50 × 36.39`,
  `110.8 × 20.13` CSS px. **None under 4 px. None collapsed.** I opened the images: the §4 block renders
  `r_K = √(ln(2K/δ)/(2n))` with the vinculum over the whole fraction, and the §7 block renders
  `r = √((32/n)(d ln(en/d) + ln(8/δ)))` with the vinculum over the whole parenthesised expression. Both are
  the radius, not the radius squared.
- **The fix is scoped, not blanket.** `.pac-lesson svg:is(.pac-line, .pac-plot, .pac-panel)` carries the layout
  properties. I enumerated every stylesheet selector in the live document and asked which ones match anything
  inside `.katex`: **no `.pac-*` selector reaches into KaTeX's markup at any of the three widths.** The only
  rules touching `.katex svg` are KaTeX's own (`height: inherit; width: 100%`) and the UA default. Tagging the
  element rather than a container is verified live: 24 lesson SVGs, **0 untagged**, 0 zero-sized.
- **The measurement guard fires.** I confirmed by construction rather than by reading: the harness case that
  unscopes the rule again is one of the four browser breakages, and my own run of
  `falsify-pac-guards.mjs --browser` scored **25 of 25**, meaning that case rebuilt, ran, exited non-zero and
  named its guard. `verify-pac-browser.cjs:528-541` measures `getBoundingClientRect()` on every
  `.katex .sqrt svg` and fails below 4 px, with a `radicals.length >= 3` floor so an empty scan cannot pass.

**No other place in this lesson's stylesheet reaches into KaTeX.** `.pac-lesson svg text` is lesson-root-scoped
by deliberate choice, but KaTeX emits no `<text>` — its radicals and delimiters are `<path>` — so it matches
nothing there; I verified that live rather than reasoning about it.

## A9. The other five "DOM right, picture wrong" mechanisms — **none found in this lesson**

Measured at 1366 and 320 px with a reviewer-written drive:

| mechanism | how I checked | result |
| --- | --- | --- |
| CSS beating a presentation attribute | computed style vs attribute over every shape | none |
| CSS collapsing geometry | every lesson SVG and every KaTeX SVG measured | 0 zero-sized of 24 + 3 |
| **Paint order** | `elementFromPoint` at each `<text>` centre; and every later opaque sibling overlapping a `<text>` by ≥60% | **0** |
| Contrast | WCAG ratio of every `svg <text>` fill against its painted ground | **89 nodes, 0 below 4.5:1**, 0 below 3:1, 0 under 9 px |
| `<rect>` over `<text>` | as paint order, restricted to rect/circle/path/polygon with opaque fill | **0** |
| Containment checked against the inset it guards | read the guard: `MINIMUM_INSET = 12` is a fixed constant asserted *before* any position is compared against `drawing.inset` | correctly repaired |

I opened the repaired figures. **Figure 11's two half-discs both render** — green upper, brown lower, with the
dial marks and the legend naming each; the paint-order defect is genuinely gone, and the four cycle values
`17/64, 17/32, 1/16, 1/8` printed beside them are the ones I computed exactly. **Figure 8** is five separated
rows over one shared axis with a legend, the two missed strips clearly distinguishable from the bands they used
to sit on, and the prose's "third row" description matches what is drawn. **Figure 10** defaults to all three
development curves on one tightly scaled axis and does show the comparison the section is about.

**No horizontal overflow at any width**: document `scrollWidth === clientWidth` at 1366, 390 and 320, and
**zero** `.katex-display` blocks overflow their own box at any of the three — so departure 4's claim that the
four restructured formulas no longer overflow at 320 px is true. Zero page errors at any width. The only
"clipped" elements outside KaTeX are the `<thead>` nodes of stacked tables at narrow widths, which is the
intended responsive pattern.

## A10. The investigations against their contract — **no mis-grading found**

I drove all three live (`scratch/pac-independent/labs.cjs`). Taking the four known defect shapes in turn:

1. **A tie graded as a miss.** Investigation 1 at target `0011`, observations `[0, 1, 2]`, ε = .25 gives risk
   exactly `.25`, i.e. risk **equal to** ε. Predicting "meets" is graded **a match**, and the numeric `0.25` is
   accepted. The boundary is `≤`, the lab says so in its own hint, and the model layer asserts
   `meetsTarget === (risk <= epsilon)` over all 256 target/subset pairs at six epsilons with a floor proving
   the tie branch is entered. **Not present.**
2. **A tolerance rejecting the page's own printed value.** This is the shape I expected to find, because the
   interval risks are *not* exactly representable: the closer-edges risk is `0.020000000000000017764` while the
   page prints `0.020000`, and the worked risk is `0.099999999999999866773` printed as `0.100000`. Investigation
   3 — the only place a float risk is graded numerically — uses **`tolerance: 1e-6`**, which accepts `0.02` and
   `0.1` comfortably. The two labs that use `tolerance: 0` grade quantities that *are* exactly representable: a
   risk in `{0, .25, .5, .75, 1}` (investigation 1) and an integer count (investigation 2). I typed the page's
   own printed values into all three and all three were accepted. **Not present, and the choice looks deliberate
   rather than lucky.**
3. **A verdict naming the wrong quantity.** Investigation 1 grades `run.risk` and calls it "the population
   risk"; investigation 2 grades `intervalPatterns(n).length` (or `thresholdPatterns`) and calls it "the number
   of realizable labelings"; investigation 3 grades the post-change `experiment.risk` and calls it "the
   population risk". Each matches. **Not present.**
4. **An "exact null" preset built from the draft while graded against the applied state.** The presets *are*
   built from the draft. From a pristine state the null holds — I clicked "Add two negatives far outside:
   .01 and .99", predicted "unchanged", and the verdict was a match with the two risks differing by exactly
   `0.000000000000`. From a **dirty** draft it does not. See **S6**; the grading is still correct, which is why
   this is a should-fix and not a blocking mis-grade.

Degenerate inputs all behave: an **empty observation sequence** returns the all-zero rule at risk `0.500000`
and grades correctly; a **zero-error hypothesis** (observe all four) returns `0011` at risk `0.000000` and
grades correctly; an **unshatterable request** (`101` on `.2/.5/.8`) is graded "impossible" with the
obstruction named and the count 7 accepted; the **+2 translation** is an exact null on the count, as practice 3
promises.

**The leak property holds.** On first paint, investigations 1 and 2 render no six-decimal value, no
`.pac-reveal` and no `.pac-verdict` at all. Investigation 3 renders `0.100000` and `0.000000` — but those are
the **applied** state's risk and training error, which §8 has already worked through two paragraphs above, and
its graded quantity is the risk *after* the edit. I re-measured all three builder pins on a fresh load:
`0.500000`, `7 of 8` and `0.020000` are each **absent before commitment**, and I saw each appear after. That
matches the declared departure 6 exactly.

## A11. The destination note — **every claim verified; I concur with narrowing rather than closing**

`docs/teaching/topic-notes/pac-learning-vc-dimension.md`, unedited by me.

- **`finite_radius(25, 500, .05)` = `sqrt(log(1000)/1000)` = `0.0831129068134555`.** Verified three ways: as an
  exact identity (A1), as the value the retained program computes, and as the value rendered on the page —
  `verify-pac-browser.cjs:269` pins `toFixed(8)` against the live text, and I read `0.08311291` off §4 myself.
- **Shared-sample dependence, protected selection, outcome-invented family.** All three are explicit in §4:
  "The models can all be evaluated on the same 500 examples; their prediction errors need not be independent of
  one another"; "It covers a member selected after seeing the evaluation scores, provided it belongs to that
  protected family"; and the K=3 passage naming `0.069192` as "a radius that was never earned". Practice 2 asks
  all three back. Figure 3 draws it, with `rankingCertified` computed and false (separation `0.01` against
  `2r = 0.166`), so the figure cannot accidentally certify the ranking.
- **The four accuracy reassessments all landed.** Sufficient-versus-minimum, realizable-versus-agnostic,
  statistical-versus-computational and gap-versus-excess-risk are a four-row table in §2 that the rest of the
  page refers back to; measurability is stated in §7 with the BEHW citation and the explicit refusal to extend
  to "arbitrary nonmeasurable constructions".
- **The historical error is not repeated.** I grepped the lesson body, the figures and the labs for every
  phrasing in that family. The word "minimum" appears **zero** times in the lesson body or figures; "sufficient"
  appears 23 times. The four hits my adversarial grep returned are all benign — a *negated* "tight bound", a
  geometric "tight bounding rectangle", a genuine "proves the upper bound 4", and a prediction-option label. §3
  says a sufficient size "is not a claim that 161 examples cannot work"; §7 says a vacuous radius "is not
  evidence that the actual error exceeds 1"; practice 7 is built entirely around repairing exactly this
  mistake. **No figure or investigation lets a bound read as tight or as a promise**: figure 7 shades the
  vacuous region explicitly, and figure 9's caption separates the Monte Carlo failure counts from the
  analytically computed risks.

**On the recommendation.** The builder recommends narrowing rather than closing the note, on the ground that
implementation now exists but independent verification did not. That reasoning is now spent — this review is
that verification — but I reach the same conclusion by a different route, and so I concur with **narrowing**:

- What the note actually asked for is done and verified: the finite-family bridge, the selection contract, and
  all four accuracy reassessments. That part should be recorded as closed.
- What should stay open is the note's own last sentence: *"A future author must inspect primary PAC/VC sources
  for the stronger claims."* I verified the bounds are correctly **stated and evaluated**, and that the BEHW
  and Mehta forms are the standard ones; I did **not** re-derive them from the primary papers, and neither did
  the builder. The note should retain that residue rather than have it closed by a numerical verification that
  did not address it.
- Independently, **B1 below** is a reason not to close this class of concern at the record level.

---

# Blocking

## B1. The record states the generalisation as actionable and does not run it; the identical wrong-mathematics defect is live one lesson away, in this same change set

**Where.** `docs/teaching/drafts/pac-learning-vc-dimension/design.md:246-249` and `:360-362` ("Still open").

The Phase C record says, correctly and valuably:

> **Any lesson whose stylesheet contains an unscoped `svg` rule is exposed.** The check is one line: look for a
> selector that ends in a bare `svg` under the lesson root, and ask whether KaTeX renders inside that root.

I ran that one line across `src/learn/components/lesson-labs/*.css`. It returns 46 descendant-`svg` rules
setting `height: auto`; most are scoped to a figure or plot container, where KaTeX never renders, and are
harmless. **Seven are scoped at a lesson root or a lab root**: `.bn-lesson svg`, `.cal-lesson svg`,
`.hmm-lesson svg`, `.imb-lesson svg`, `.rad-lesson svg`, `.ssl-lesson svg`, and — widest of all —
`.lesson-lab svg` in the **shared** `lessons.css`.

`rademacher-labs.css:23` is the live one. That lesson is **modified in this same uncommitted change set**, sits
at classical-ml position 36 (two after PAC), is the lesson PAC's §10 and §12 both hand off to, and its subject
is generalization bounds. Its source contains 23 `sqrt` occurrences. Driven in the same preview build:

```
rademacher-complexity-generalization-bounds   root=lesson-pilot rad-lesson
  .katex nodes: 271   radicals: 24   COLLAPSED: 24
  [{"w":83.9,"h":0.5},{"w":83.9,"h":0.5},{"w":100.8,"h":0.3},{"w":94.4,"h":0.7},
   {"w":27.8,"h":0.1},{"w":22.5,"h":0.0},{"w":20.9,"h":0.0},{"w":68.2,"h":0.2}]
```

**All 24 of its radicals paint at under one pixel of height.** That is the same defect, with the same cause,
producing the same class of error — a Rademacher bound rendered without its radical is the bound squared — in
a lesson delivered alongside this one. (`.bn-lesson` and `.cal-lesson` are latent rather than live: I checked,
and neither lesson currently contains a `\sqrt`. They will break the moment one is added.)

**Why it matters.** The PAC lesson itself is clean — I verified that directly in A8 and I am not asking for a
change to any PAC file. The blocking issue is at the **record and process** level: `design.md` identifies a
repo-wide wrong-mathematics defect class, states the detection rule, declares the PAC instance repaired, and
lists nothing about the class under "Still open". A reader of that record would reasonably conclude the class
was handled. It was not; it was handled at the lesson boundary, and the one-line check the record itself
supplies finds a live instance two topics later. This effort's single wrong-mathematics defect should not be
signed off while a sibling in the same change set exhibits it.

**How I verified.** Tree-wide regex over the lab stylesheets after stripping comments; then a live Playwright
drive of `rademacher-complexity-generalization-bounds`, `calibration-conformal-prediction` and
`bayesian-networks-causal-graphical-models` against the same `dist-pac` preview, measuring
`getBoundingClientRect()` on every `.katex .sqrt svg`; then `grep -c sqrt` on each lesson body to separate live
from latent. PAC measured 3 radicals, 0 collapsed, in the same run — so the drive is not producing false
positives.

**Scope.** Fixing `rademacher-labs.css` is outside this review's remit and outside this lesson's files. What
belongs to this deliverable is the record: the generalisation should be moved to "Still open" with the
tree-wide result attached, rather than left reading as a closed lesson-local repair.

---

# Should-fix

## S1. `falsify-pac-guards.mjs` has no crash safety: a kill mid-run leaves a mutated source file on disk with no record of it

**Where.** `scripts/falsify-pac-guards.mjs:341-397`.

The harness reads the original bytes into a Buffer (`:341`), writes the mutation (`:350`), runs a verifier —
and for the four browser cases, a full `npx vite build` (`:354`) — then restores in a `finally` (`:381`). The
original exists **only in the Node process's heap**: no backup file, no temp copy, no stash. `grep -c
'process.on\|SIGINT\|SIGTERM\|uncaughtException'` returns **0**.

So any termination that does not unwind the stack — Ctrl-C, `kill`, an OOM kill, a spend-limit kill, a reboot —
leaves the breakage applied. The window is a full verifier run, and for browser cases a full production build:
minutes, not milliseconds. The files at risk include `pac-models.js`, the lesson body, `pac-labs.css` and
`verify-pac-models.mjs` itself. One of the browser breakages is **the unscoped `svg` rule that collapses every
radical** — precisely the defect of B1. A kill during that case leaves the wrong mathematics restored to the
CSS, and nothing on disk says so.

Two related gaps in the same file:

- The evidence restore loop (`:397`) is **not** in a `finally`. If any case throws the restore-mismatch error at
  `:386`, the loop at `:340` exits by exception and `:397` never runs — leaving the broken-page evidence JSON
  and screenshots on disk, which is exactly what the comment at `:324-327` says the snapshot exists to prevent.
- `occurrencesReplaced: 1` (`:371`) is a hardcoded constant, not a measurement. `String.replace` with a string
  pattern replaces the first occurrence only, so when `occurrencesPresent > 1` the report states a substitution
  count it did not measure.

**Why it matters.** This is the root cause of the failure the builder already recorded: two concurrent
instances left `inset = 0` in `stripGeometry` and an inert containment guard that read as correct. The
concurrency was the trigger; the absence of any durable record of the pre-mutation state is why the corruption
was silent and cost a full re-verification. The record's response is a warning in prose ("Two instances must
not run at once"). A `.orig` sidecar written before the mutation and removed after, plus `process.on('SIGINT'|
'SIGTERM'|'uncaughtException')` handlers that restore from it, and a lock file, would make the warning
unnecessary.

**How I verified.** Read `:319-397` line by line; confirmed the handler count is zero by grep; confirmed the
loop is strictly sequential (`spawnSync` throughout, no `Promise.all`); confirmed the evidence snapshot is
`pac-*`-prefix-scoped and deletes nothing, so it cannot reach another lesson's captures. I then ran it once
with backups in place and confirmed every file returned to its baseline SHA-256 and all 40 screenshots were
byte-identical — so the mechanism is correct *when the process completes*. That is the whole of my point.

## S2. `design.md`'s Phase A verification table contradicts the tree, and Phase C's narrative contradicts Phase A

**Where.** `design.md:125-129` against `design.md:241-242` and `:336`.

Measured by re-running each verifier myself:

| Phase A table says | tree actually reports |
| --- | --- |
| `516 grouped checks across 91 groups` | **520 grouped checks across 95 groups** |
| `93 checks over 16 files` | **96 source-hygiene checks over 16 files** |
| `20 of 20 breakages`, four browser guards deferred | **25 of 25**, the four browser guards run |
| `5,158 checks; 5,275 leaves` | 5,158 / 5,275 — correct |

Phase C then narrates the pre-fix state as "**520 model checks** … **96 hygiene checks**" (`:241-242`) — numbers
that only became true *after* Phase C added checks, and that directly contradict the Phase A table four
paragraphs above. The three sweep counts inside the Phase A row (100,720 / 23,984 / 76,736) are correct and I
re-derived them independently (A3); it is the check tallies that went stale.

**Why it matters.** This is the record-versus-tree class the brief says appeared in seven of the last fifteen
reviews. Nothing here is *wrong* about the implementation — every verifier is green and the higher numbers are
the true ones — but a reader reconciling the record against the tree finds three mismatches and one internal
contradiction, and cannot tell which section to trust. Phase A's table should either be updated or explicitly
dated as "as of part A".

**How I verified.** Ran all four offline verifiers and the harness, once each, and read their PASS lines.

## S3. The closing callout overstates the displayed programs as "byte-exact slices"

**Where.** `src/learn/data/topics/pac-learning-vc-dimension.jsx:860` (the "Where this page's numbers come from"
callout): *"The three displayed programs are **byte-exact slices** of that program."*

They are not, quite. Each displayed program is a byte-exact slice of the function definitions **plus** material
that appears nowhere in `pac-calculations.py`:

- `finite-world` adds `from math import exp` — the program's own import line is
  `from math import comb, exp, log, log2, sqrt, ceil, pi`;
- all three append a driver block (`for n in range(1, 6): print(...)`, `for sample in (...)`, and so on) written
  for the snippet.

I checked every line of all three: **every function body line is present verbatim in the program**, and the
only non-verbatim lines are the adapted import and the drivers.

**Why this is worth fixing rather than shrugging at.** §8's own sentence is scrupulously correct — *"each one's
**algorithm** is lifted byte for byte from the complete downloadable program"* — and `design.md:127` is correct
too — *"assembled from byte-exact, SHA-256-pinned slices"*. The verifier pins the SHA of the **extracted**
span, not of the assembled snippet, which is the right thing to pin. It is only the closing callout that drops
the qualifier, and it does so on a page whose entire discipline is refusing to let a claim drift a notch
stronger than what was checked. Restoring "whose algorithms are byte-exact slices" costs two words.

**How I verified.** Line-by-line membership test of each displayed `code` field against the served
`pac-calculations.py`; then read `verify-pac-examples.py:179-190` to confirm what the SHA actually pins.

## S4. 55 of the 96 source-hygiene "checks" are literally `check(True, …)`, and that verifier has no count floor

**Where.** `scripts/verify-pac-sources.py:131, 141, 154, 186, 205, 222`; the counter at `:56-60`; the PASS line
at `:271`.

`check()` increments `checks` unconditionally, and six call sites pass the literal `True`: three inside the
per-file loop (×16 files), one per component (×4), one once, and one per component (×2).

I measured this rather than counting by hand — I copied the script to `scratch/`, wrapped `check` with a frame
inspection that counts call sites whose source text is `check(True`, and ran it:

```
LITERAL-TRUE CHECKS: 55 of 96
PASS: 96 source-hygiene checks over 16 files - no eaten escape, no raw control byte, ...
```

The **detection logic behind each of those is real** — the raw control-byte scan at `:125-130` is the strongest
guard in the whole set, and it is what would have caught the eaten-backslash failures this repo has shipped
before — but it reports by appending to `problems`, which does not increment `checks`. So the headline count is
inflated roughly 2.3×, and there is **no `check(checks >= N)` anywhere in the file**: the only floor is
`len(blocks) >= 12` at `:163`.

Related, in the same file: the escape allow-list at `:134-137` ends in `or following.isalnum()`, which admits
every letter and digit, so it can never flag `\b`, `\d`, `\s` or `\q` — the very escapes the docstring names as
the hazard. Those are caught by the byte scan, not by this check, and the check is close to inert.

**Why it matters.** "96 hygiene checks" is quoted in `design.md` and in the Phase C narrative as evidence of
coverage. Forty-one of them are assertions; fifty-five are counters wearing an assertion's name.

## S5. Every headline count is a printed tally, and the floors sit far below the current values

**Where.** `verify-pac-models.mjs:1314-1315` and `:1376`; `verify-pac-data.py:873-875`;
`verify-pac-examples.py:439`; `verify-pac-sources.py` (no floor); `verify-pac-browser.cjs:674`.

| headline | asserted? | floor |
| --- | --- | --- |
| 520 model checks / 95 groups | tally | `total >= 300`, `groups >= 45` |
| 5,158 data checks | tally | `checks >= 5000`, `leaves >= 5200`, `root_checks >= 35` |
| 44 example oracles | tally | `oracle_count >= 30` |
| 96 hygiene checks / 16 files | tally | **none** |
| 12 browser cases | tally | **none** — `records.length` is never asserted |
| 25 of 25 breakages | **pass condition is asserted all-or-nothing** | but on whatever N was assembled |

So the model suite could silently lose 220 checks and 50 groups and still print PASS; a whole browser section
could be deleted and the verifier would print `cases: 11, status: passed`.

**In fairness, three things are genuinely hard-asserted** and they are the ones that matter most:
`witnessCases >= 100000`, `riskCases >= 1000`, `selectionCases === 256` (exact equality), `planeCases >= 40`;
the leaf-coverage property is asserted in **both** directions (`not extra` and `seen >= all_leaves`), so "every
scalar leaf re-derived" is a real property rather than a ratio; and the falsification gate credits a case only
when the verifier exits non-zero **and** matches the case's `expect` regex, which is the right construction.

**Why it matters.** The builder's own comment at `:1311-1313` says "a counter that is reported but never floored
is decoration" — and then sets the floor at 58% of the current value. The floors should track the real numbers
closely enough that losing a section fails the build.

## S6. Investigation 3's presets compose with an uncommitted draft, so the button the lesson presents as the exact null can produce a move

**Where.** `src/learn/components/lesson-labs/PacLabs.jsx:368-379` (`INTERVAL_PRESETS`) and `:503-505`
(`state.suggest(apply(draft))`).

Every preset is applied to `draft`, while grading compares the committed draft against `active` (the applied
state). From a pristine state these coincide and the preset behaves as advertised. From a dirty draft they do
not. Driven live:

```
--- CLEAN: "Add two negatives far outside" from pristine state ---
VERDICT: = Your prediction matches: It does not move at all. ...
         The two risks differ by 0.000000000000, inside the tolerance 1e-12.

--- DIRTY: target right endpoint edited to 0.9 first, then the same preset ---
pending banner shown: yes
VERDICT: ≠ You recorded It does not move at all; the calculation gives It falls.
         ... risk is 0.050000, against 0.100000 before.
```

The grading is **correct** — the risk really did fall, because the committed state carries both the edit and the
preset. That is why this is a should-fix and not a mis-grade. But "Add two negatives far outside: .01 and .99"
is the button §8 and practice 8 both present as *the* demonstration that an exterior negative leaves the fit
untouched, and it silently stops demonstrating that.

**Mitigations already present**, which is why I rank this below the items above: `state.suggest` retires the
verdict, the `.pac-pending` banner appears and says in terms that "the comparison will be against the state
currently applied", and the verdict prints both risks. A learner who reads the banner is not misled.

**The fix is small:** build the comparison presets from `state.active` rather than `state.draft`, so clicking
one discards pending edits and restores the null it promises. Investigation 2's presets have the same shape but
are harmless there, because its graded quantity is absolute rather than a before/after comparison.

**How I verified.** Playwright, `scratch/pac-independent/labs.cjs` sections 6 and 7; log retained at
`scratch/pac-independent/labs-log.txt`.

## S7. The two-strip value the prose quotes to six significant figures is guarded at ±1.4%

**Where.** `scripts/verify-pac-models.mjs:556`.

```js
close(twoStripBound(0.1, 200).raw, 0.0000701053, 'the value the prose quotes beside those zero failures', 1e-6);
```

`close()` scales by `Math.max(1, |expected|)`, and `|expected| = 7.01e-5 < 1`, so the tolerance is **absolute
1e-6 against a value of 7.0e-5** — a band of roughly ±1.4%. The assertion would accept `0.0000691` or
`0.0000711`, while the page prints `0.0000701053` at ten decimals. The value itself is correct (I computed
`0.000070105332497658006` at 50 digits), so nothing is wrong today; the guard simply cannot detect the class of
change it exists to detect.

Every other `close()` in both verifiers uses the 1e-12 default on sub-unit quantities, which is tight and
appropriate — this one call passes an explicit `1e-6` and is the only loose comparison I found.

## S8. `verify-pac-examples.py` writes `"passed": true` to the evidence file before its coverage floor runs

**Where.** `scripts/verify-pac-examples.py:381` (write), `:434` (`"passed": not failures`), `:439`
(`oracle(oracle_count >= 30, …)`).

The evidence JSON is written, with its `passed` field, and only afterwards does the oracle-count floor execute.
If that floor ever trips, the process exits non-zero **and** leaves `docs/teaching/evidence/pac-native.json` on
disk asserting `passed: true`. Since the evidence files are the durable record and the console output is not,
the artifact that survives would be the one that lies. Moving the floor above the write, or writing
`passed: false` first and rewriting on success, closes it.

## S9. Three of the five verifiers cannot be re-run without overwriting the record they document

**Where.** Only `verify-pac-models.mjs` honours `--no-evidence` (`:1372`). `verify-pac-data.py`,
`verify-pac-examples.py` and `verify-pac-sources.py` write `pac-data.json`, `pac-native.json` and
`pac-sources.json` unconditionally.

This bit me directly and is worth recording as a concrete failure mode rather than a hypothetical: my
instrumented copy of the sources verifier (S4), run from `scratch/` with `ROOT` patched, wrote
`docs/teaching/evidence/pac-sources.json` with a `verifierSha256` of the **scratch copy** — a record pointing
at a file that does not exist in the tree. I caught it only because I had hashed every evidence file
beforehand, and I restored all six from backup.

An independent reviewer is exactly the person most likely to re-run these, and exactly the person who must not
disturb the record. `--no-evidence` on the other three would cost a line each.

---

# Observations

**O1. About forty figure-geometry assertions in `verify-pac-models.mjs` are self-comparisons.** The module
stores `x: scale(value)` and the verifier asserts `close(point.x, plot.x(point.n))` — the same exported scale,
called twice. Examples: `:891-892`, `:906-907`, `:913-914`, `:919-921`, `:987`, `:1089-1092`, `:1136-1138`,
`:863-868`. Two are pure tautologies: `:167` compares `growthTable[i]` with `growthRow(i+1)`, and
`growthTable` *is* `Array.from(..., growthRow)`; `:1164` compares `ghost.patternCount` with
`intervalPatterns(8).length`, which is how it was computed. None of this is *harmful* — the containment,
ordering and monotonicity assertions interleaved with them are real and are where those sections' coverage
lives, and `:167` is immediately followed by genuine packet comparisons. But the file header's claim that
nothing is compared with itself does not hold for the figure-geometry sections.

**O2. `verify-pac-examples.py:181` cannot fail.** `oracle(piece in source, …)` where `piece =
"\n".join(lines[start:end])` and `lines = source.split("\n")`: a contiguous run of lines rejoined by newline is
a substring of the source by construction. The "verbatim slice" property is actually carried by the SHA pin at
`:187` — which, note, is written `program["extractedSha256"] in ("", program["extractedDigest"])`, so an empty
pin silently disables it. All three pins are non-empty today.

**O3. `verify-pac-browser.cjs:267` is vacuous for the small growth counts.**
`assert.ok(rendered.includes(String(row.intervals)))` over `growthTable` asserts, at n=1, that the whole page
text contains `"2"`. The learning-curve assertions two lines above (`includes(\`${developmentCorrect}/${developmentN}\`)`)
and the radius assertion two lines below are specific and strong; this one is not.

**O4. The prose literal "about 161.42" is not pinned.** `verify-pac-models.mjs:531` asserts only
`161 < (ln 32 + ln 100)/.05 < 162`, which pins the ceiling claim. The quoted two-decimal figure is correct — I
get `161.418121776` — but it is the one free-standing numeral in the body that no verifier compares against a
computed value.

**O5. The install snippet's pins are not cross-checked against the recorded environment.** The prose tells the
reader `numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1` (`:606`, `:610`); `pac-examples.js` records exactly
those versions plus Python 3.12.14. They agree today, and nothing asserts they stay in step.

**O6. §5 names three of the four half-plane configurations the program checks.** The prose says "8 feasible
triangle patterns, 14 feasible square patterns and 6 feasible collinear-triple patterns"; the program and the
packet also carry the interior configuration's 14, which figure 4 draws and which the §5 four-point argument
depends on. The sentence is not false, just short of what it enumerates.

**O7. Two small things in `verify-pac-data.py`.** `:671` is a no-op —
`{path if path.startswith("#curves") else path for path in covered}` is `set(covered)`. `:652-654` asserts that
**other lessons'** `banknote-subset.csv` files exist; a failure there says nothing about this lesson. Neither
affects a verdict.

**O8. The 5,275 leaf figure includes one empty container.** `/geometry/triangle/infeasible` is an empty list,
counted as a leaf by `leaf_paths` (`:143-144`). 5,274 scalars + 1 empty list. It is a defensible convention —
an empty list *is* a claim — but the figure is quoted as "scalar leaves" in both the record and the PASS line.

**O9. `occurrencesReplaced: 1`** in `falsify-pac-guards.mjs:371` is a constant, not a measurement. Covered under
S1 but noted separately because it is the report field, not the mechanism.

**O10. Figure 11's fourth panel caption wraps differently from the other three at 1366 px.** The first three
read `r·x mod 1 = …, upper half` / `→ 1`; the fourth pushes `→` onto the first line and `1` onto the second.
Cosmetic, and it does not occur at 390 or 320.

**O11. Departures I judge justified, on inspection.** Departure 1 (figure 2 uses a different target from
investigation 1) avoids the figure answering the investigation's opening question and says so in its own
caption. Departure 2 (25 declared rows rather than a loss matrix) is the only way to carry the checked
n=500 radius honestly, and the counts are integers so every displayed error is an exact multiple of 1/500 — I
confirmed `best = 131/500`, `runnerUp = 136/500`, separation `0.01` against `2r = 0.1662`, and
`rankingCertified === false`. Departure 3 (the repeated-run question lives in figure 9, and the browser draw is
an *input generator* explicitly disclaimed as not NumPy's) is the right call; a page that implied it could
reproduce seed 41 in mulberry32 would be worse than one that says it cannot. Departure 5 (an empty observation
sequence is allowed) makes the lesson's point about the all-zero default and grades correctly, as I checked.
Departure 6 (investigation 3 draws the applied state) is correct given that its graded quantity is post-edit,
and the leak measurement confirms it.

**O12. Things I specifically looked for and did not find.** No `.katex-error` nodes. No page errors at any
width. No horizontal overflow at 320. No `url(#…)` pattern fill that could fail to resolve. No blanket `fill`
rule that could beat a `fill="none"` attribute — every shape class states its own. No coincident-coordinate
grading path (it is refused, with a message, rather than resolved by a tie rule). No use of `Math.sqrt` in the
lesson body, where `Math` is the KaTeX component — the source verifier checks this and the module header
explains why. No control bytes in any verifier: I byte-scanned all six for `0x00–0x08, 0x0b–0x1f, 0x7f` and
found none, so no `\b` has been eaten into a literal backspace anywhere in this lesson's tooling. The one
`0x08` in the tree is constructed deliberately at runtime by `falsify-pac-guards.mjs:177` as a breakage.

---

# Summary

I recomputed every number on this page from first principles — 546 exact and 50-digit checks, a 100,720-case
feasibility sweep by two independent derivations, half-plane realizability by exact Fourier–Motzkin over every
labeling of four configurations plus 58,905 grid quadruples, a full replay of the learning-curve experiment
from the served CSV including all fifteen prediction vectors, and a full replay of the seed-41 simulation —
and **found no mathematical disagreement anywhere**. The radical fix is real, scoped, and confirmed by looking
at the rendered images at 1366, 390 and 320 px. The declared coverage limitation is characterised exactly
right. The investigations do not mis-grade at ties, empty sequences, zero-error hypotheses or unshatterable
configurations, and none of the four known grading defect shapes is present as a mis-grade. The destination
note's every claim checks out, and the historical "minimum data" error is not merely avoided but is the
explicit subject of a table row, two sections and a practice task.

The blocking finding is not in this lesson's files. It is that the record identifies a repo-wide
wrong-mathematics defect class, supplies the one-line check for it, declares the local instance repaired, and
leaves the class unlisted under "Still open" — while running that one line finds all 24 radicals collapsed,
live, in the sibling lesson delivered in the same change set. The should-fix items are mostly about the
verification apparatus rather than the page: a harness that cannot survive being killed, a stale count table, a
callout that drifts one notch stronger than what was checked, and a set of headline counts that are tallies
rather than assertions.

This is a careful, unusually honest lesson. Most of what I have to say is about the scaffolding around it.
