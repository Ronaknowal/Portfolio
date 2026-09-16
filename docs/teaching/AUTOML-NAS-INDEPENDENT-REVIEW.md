# AutoML & Neural Architecture Search (NAS) — independent phase-three review

Reviewed 16 September 2026 against the working tree plus the uncommitted AutoML implementation. The reviewer authored
neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or
`docs/teaching/drafts/` was edited; this document is the only write outside `scratch/`. Throwaway scripts and captures
live under `scratch/independent-review-automl/`. No state-changing git command was run. The four verifiers were re-run
(§D) — they rewrite their own untracked evidence files and screenshots, and they regenerate identically.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/independent-review-automl/`, importing neither `automl-models.js` nor `automl-data.js` nor any `verify-automl-*` script nor `author-calculations.py` for any numerical claim. (a) `recompute_campaign.py` — my own reimplementation of the declared protocol, refitting the whole **35-fit campaign** from the **served** `public/learn-assets/automl-nas/banknote-data.csv` with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1, SciPy 1.18.1). (b) `diff2.py` — **97 field-level comparisons** against `calculated-inputs.json`, covering 10,109 out-of-fold labels, 410 inspection predictions, 1,372 role ids and 2,757 fold id entries. (c) `constructed_exact.py` — **86 re-derivations of the `constructed` block from the manuscript's declared settings alone**, in `fractions.Fraction` and `mpmath` at 60 decimal digits, every one cross-checked by a second, genuinely different route (adaptive quadrature of the EI definition over (−∞, b]; quadrature of the normal density against the erf route; central differences at ε = 10⁻²⁵ for every mixture gradient and all three bilevel derivatives; explicit halving loops; full enumeration of the configuration space; the block sum for `6h+1`). (d) `gridsweep.mjs` — **72,427 grid points** — and `probe.mjs`, exercising the shipped decision logic for drawn-rule/applied-rule agreement and for reachable grading ambiguity. (e) `drive.cjs`, `figures.cjs`, `longfloats.cjs`, `final.cjs` — my own Playwright drive (Edge, not derived from `verify-automl-browser.cjs`) at 1366, 390 and 320 px. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `data-source.json`, `calculated-inputs.json`); the lesson body, model layer, generated data and examples modules, the labs, the shared scaffolding, the figures module and the CSS; all four verifiers in full; `LESSON-TEACHING-STANDARD.md`; the served asset and its attribution; the blueprint and its registration. |
| **Reused (declared)** | scikit-learn/NumPy/mpmath as *general* libraries, which is what the manuscript's own program is written against — not lesson code. Two subagents for fan-out: a served-data/provenance audit and a verifier dead-assertion audit. **Every finding below that originated with them I re-derived myself** — I re-read each cited verifier line, recounted the math payloads (102 vs 99) with my own regex run, re-measured the SVG text floor directly in the browser, and re-tested the tautologies and the `halvingSchedule` boundary by execution. Two of their leads I checked and discarded (see O11). |
| **Looked at** | Every figure at 1366 and 320 px, plus the driven states of I1, I4 and I6. I opened the images. One apparent defect (F5 lane 1 clipped to half-height glyphs) proved to be a sticky-header artifact of my own element screenshot and is **not** a finding; I re-captured with page chrome neutralised and with an ancestor-clip probe before discarding it. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/automl-neural-architecture-search-nas.jsx` | `9bd78f891afff9da1756651c73f4acd6afd46475c86749919fc7db5986c37a78` |
| `src/learn/data/automl-models.js` | `2029738ca1c68464170a5086f393b1ddc393f43c34bd9b2f508f5494a8ab86e4` |
| `src/learn/data/automl-data.js` | `a4a40b3c2d37c21595d6c5d63ae0deff3eb1664846dcb34661de600411c4d78d` |
| `src/learn/data/automl-examples.js` | `5fac2302b73d265fd2605460ea7c80fd02e1bc85fc261ce3927a29b0117bccf9` |
| `src/learn/components/lesson-labs/AutomlShared.jsx` | `91e52bacc3d888262eb759831f5ee2d4275e39c3a0c9d17b2bf58f043e32ec82` |
| `src/learn/components/lesson-labs/AutomlLabs.jsx` | `4b36aef452a156e99ff47dda530e7c289f9eb52971b883eae3dde49727e0dd21` |
| `src/learn/components/lesson-labs/AutomlFigures.jsx` | `399a23a10419349362b850c6c735e700ec39b2927d964e388e4cdcad10042935` |
| `src/learn/components/lesson-labs/automl-labs.css` | `0f6ada2efad6d06cd4a98ecf4e7866ad61ba81c39d428fd5c67371d403fbfbbd` |
| `src/learn/data/curriculum/blueprints/automl-neural-architecture-search-nas.js` | `eef499984c5008cd4d1d7d23288e9989f4e5d4dc0bb1444eb6f4fdb541e30a89` |
| `docs/teaching/drafts/.../lesson.md` | `068d358d9ce3aaeed784056a34765dfbd9fbed63484e828451efce2f53ec59bc` |
| `docs/teaching/drafts/.../design.md` | `299acef08f6cefc796516888ac6ee330c7e921bfaab8631c81386b0420e1b71b` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `e7a839cc100331322fe856325f6068050a9f5af9fa24382ff8b5f5d87cb8fba6` |
| `scripts/verify-automl-models.mjs` | `fa5c2ef9a6d86dbf2ecd1e68f7bcf1def1dda69d2715c2e8e5cdb3407bb42b65` |
| `scripts/verify-automl-data.py` | `16618fc6b9611a85fcb60c27ddf0d577eb9de05eccfcb5e903c6ce13113d18da` |
| `scripts/verify-automl-examples.py` | `066ff1691d07d83536bcd17ea9abea5bcdc64367bd614eba4beb29d2c8c188a4` |
| `scripts/verify-automl-browser.cjs` | `01e4ce22b5554adac78a31db3437e103cbd5ce6dd08cb7acf112c415e2b8b991` |
| `public/learn-assets/automl-nas/banknote-data.csv` | `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9` |

---

# Part A — correctness

## A1. The whole 35-fit campaign, refitted from the served CSV — **no disagreement**

`scratch/independent-review-automl/recompute_campaign.py` and `diff2.py`. My own implementation of the declared
protocol, reading the file the page serves. **97 field-level comparisons, one disagreement, and that one is a
floating-point association artifact in my own arithmetic (A2 below).**

- **Grouping.** 1,372 rows, 4 features + class, 1,348 unique feature vectors, **11 repeated groups accounting for 24
  additional rows**, and **zero identical-feature groups with conflicting labels**. The packet's
  `groups.repeated_group_ids` list matches mine element for element.
- **Roles.** Development 900 groups / 919 rows / 407 class-1; inspection 200 / 205 / 91; reserved 248 / 248 / 112 —
  the table in §5 exactly. I compared the **full id lists** (1,372 ids) and the development and inspection label
  vectors, not the counts.
- **Folds.** All three fitting and validation id lists, 2,757 entries, identical. Validation sizes 314, 300, 305.
- **Eleven candidates.** Every one of the 33 fold accuracies, every mean, every pooled out-of-fold accuracy, and
  **all 10,109 out-of-fold predicted labels** (11 × 919) agree. The `warnings` array is empty for all eleven, and my
  refit produced no fitting warning either. The eleven means reproduce §5's table to six decimals:
  0.977420, 0.983883, 0.964439, 0.975266, 0.905760, 0.962198, 0.998907, 0.994629, 0.998907, **1.000000**, 0.998907.
- **Both refits.** Selected `mlp-tanh-16`: 205 of 205, confusion [[114, 0], [0, 91]]. Declared baseline
  `logistic-standard-c1` (registry index 3): 202 of 205, confusion [[111, 3], [0, 91]]. **All 410 inspection
  predictions** match. The baseline's three errors are source-file lines 108, 196 and 346 — inspection rows 107,
  195 and 345; together with development row 349 these are exactly the four rows the packet narrates.
- **Majority baseline.** Development majority class 0; 114 of 205 = 0.5560975609756098.
- **Seed-75 order.** `search_order` is `[7, 10, 6, 9, 3, 2, 1, 4, 8, 0, 5]`, which is
  `numpy.random.default_rng(75).permutation(11)` exactly — *not* `RandomState(75).permutation`, which gives a different
  order. It renders as the eleven names §5 lists, in that order, and the running best-so-far vector is the running
  maximum over it.
- **`fits` is 35** and **`reserve_scored` is `false`**.

## A2. The one disagreement, and why it is not a defect

`candidates[2].pooled_oof_accuracy` is `0.9640914036996736`; I compute `0.9640914036996735`. One unit in the last
place. The packet forms `(919 − 33)/919`; I formed `1 − 33/919`. Same rational number, different association order in
binary floating point. Recorded so the next reviewer does not spend time on it.

## A3. The packet's `constructed` trust root, re-derived from the declared settings — **no disagreement**

`scratch/independent-review-automl/constructed_exact.py`. Exact rationals and 60-digit `mpmath`, from the settings the
manuscript states and nothing else. **86 checks; 85 agree; the single failure is my own ε = 10⁻²⁵ finite difference
losing the last digit against a closed form that itself matches the packet exactly.** Every second route is genuinely
independent of the first:

- **Expected improvement and Φ.** A = 0.05004008274358261 (μ = 0.35, σ = 0.02, b = 0.4), B = 0.07978845608028655
  (μ = b, σ = 0.2, so the whole value is σφ(0)), C = 0 (σ = 0, no improvement). Each closed form agrees with
  `mpmath.quad` of E[max(b − F, 0)] over (−∞, b] to better than 10⁻²⁵ — **adaptive quadrature of the definition, not a
  fixed 40,001-point grid of the same integrand the page draws**. Φ via `erf` agrees with quadrature of the standard
  normal density at z = −3, −2.5, −1, 0, 0.5, 2.5, 5 to better than 10⁻³⁰. Practice 7's 0.05 and
  0.039894228040143274 confirmed.
- **Configuration count.** 2×2 + 2 + 2 + 3 = 11, branch by branch, and by full enumeration of the space. Practice 1's
  3×2 + 4 + 2 + 2×2 = 16 and 64 four-fold fits confirmed.
- **Mixture.** softmax([ln 2, 0, 0]) = [1/2, 1/4, 1/4] **as exact rationals** (the packet's floats are the exact
  values); ō = 0; L = 1/2; ∂L/∂α = [−0, −1/2, +1/2] confirmed twice — by the closed form (ō − t)pᵢ(oᵢ − ō) and by
  60-digit central differences of the loss itself; updated logits [ln 2, 0.2, −0.2]; updated output
  0.1993359892499117. The translation null (add any constant) leaves the weights at [1/2, 1/4, 1/4]. Practice 8's
  output 1 and gradients [+1, −1] confirmed.
- **Discretization.** Mixed loss 0, selected loss 2.
- **All three bilevel derivatives.** With L_train = ½(w − α)², L_val = ½(w − 1)², w = 0, α = 0.2, ξ = 0.1:
  w′ = 0.02, direct 0, one-step **−0.098**, exact **−0.8**, w* = α = 0.2. The one-step and exact values were each
  confirmed a second way by numerically differentiating the composed functions g(α) = L_val(w − ξ(w − α)) and
  h(α) = L_val(α). The stationary variant (w = α = 0.2) gives **−0.08** while the exact stays −0.8 — the teaching point
  is real. Practice 9's 0, −0.38, −1.5 confirmed.
- **Successive halving, round by round.** Survivors [0–8] → [0, 1, 2] → [2]; selected C finishing at 0.07;
  full-resource best D at 0.02; restart 9(1)+3(3)+1(9) = 27, resume 9(1)+3(2)+1(6) = 21, all-full 9(9) = 81. The
  offset null (add ½ to every loss) leaves the survivor sets identical. The I3 edit (D's first rung 0.14 → 0.09) makes
  D survive both cuts and win. Practice 3's survivors, 12 and 8 confirmed.
- **`6h+1` at every permitted width.** `limits.width` is {1, …, 32}. I confirmed by the block sum
  (4+1)h + (h+1) that the shortcut holds for **every h from 1 to 64**, and separately drove the shipped
  `singleLayerCount` over all 32 permitted widths. 49 at h = 8, 97 at h = 16, and the two-layer 40+72+9 = 121.
  Practice 4's blocks [36, 21, 4] = 61 and the width-12 alternative's 85 confirmed.
- **NASWOT.** Codes 110/101 → Hamming distance 2 → K = [[3,1],[1,3]], det 8, ln 8 = 2.0794415416798357; identical
  codes give det 0.
- **Portfolio.** Best single default C at 0.25; the {A, B} pair at 0.10; the new task reversing it, 0.40 against 0.20.
- **Pareto.** Frontier {A, B, C} by pairwise dominance; the inclusive 5 ms cap selects B and the 3 ms cap selects A;
  **D at exactly 5 ms is feasible under a 5 ms cap**. Practice 6's frontier {P, Q} and its 5 ms answer P confirmed.

I found **no disagreement with any number in the manuscript, the packet, the practice solutions or the lesson body.**

## A4. The lesson's central caution — **holds exactly**

This is the claim I pushed hardest on, and it survives.

- The shipped `automl-data.js` exposes `sourceRows` for **119 rows and no others**. I checked each against my own
  independently computed role partition: **116 are development rows, 3 are inspection rows (107, 195, 345), and zero
  are reserved**. The 119 is exactly the union of all eleven candidates' out-of-fold mistakes plus the three rows the
  declared baseline got wrong on the inspection partition — the two sets of rows the lesson legitimately shows.
- No candidate record carries any field matching `/insp|reserv|test|holdout/`. Only the two declared models have an
  inspection outcome, and `reserveScored` is `false`.
- `replayPrefix` returns `inspectionAvailable: false`, and I5's closing caption says the replay cannot borrow the
  width-16 network's 205-of-205 because those predictions were never made for a prefix that recommends something else.
  My own refit confirms the arithmetic behind that: the replay's recommendation at budget 2 is `mlp-tanh-8x8`, at
  budget 3 it is `neighbors-standard-k3` on a **flat** best score (the registry tie rule, stated in the investigation's
  own note), and only at budget 4 does `mlp-tanh-16` arrive.
- F1's fix is present and correct in the rendered image: every row source now sits **inside** the node it feeds, and
  the reserved box carries "no arrow in or out" inside itself. The frame is labelled `selection`
  (`AutomlFigures.jsx:91`) and renders visibly — an earlier apparent absence was my own screenshot artifact.
- F3's Phase C fix is present and correct. The page says "3 other candidates" and names them, then separately says
  "7 of the 11 candidates miss it". My refit confirms both halves independently: **exactly three** candidates tie at
  0.998907 (knn 3, net 8, net 8,8), and **exactly seven** miss development row 349. The two sets are genuinely
  different and the page now says so.

## A5. Drawn rules against applied rules, over the whole enterable grid — **no disagreement**

`scratch/independent-review-automl/gridsweep.mjs`. **72,427 grid points, zero disagreements.** Not sampled points —
the enterable grid.

- **I6's cap band.** Every 2-decimal cap in [0.10, 30.00] crossed with every 1.37 ms step of D's latency over its
  whole range: the shaded band covers exactly `latency > cap` and the applied `feasible` flag agrees at every point.
  **The boundary fix is real**: the band starts at `scaleX(cap) + 1.5` with the line drawn over it
  (`AutomlLabs.jsx:987-989`), the legend and caption both state the cap is inclusive, and D at exactly 5 ms under a
  5 ms cap is drawn outside the band and called feasible by the rule and the table. Marker shape (square iff
  dominated, circle iff on the frontier) agrees with the table at every point, and the shipped marker is always a
  feasible one.
- **I3's purchased traces.** Over D's entire enterable first-rung column (0.00 to 1.00 in hundredths): every trace's
  purchased length equals the rung at which its candidate was cut, no purchased point comes from a rung the candidate
  never reached, the dashed counterfactual always begins exactly on the solid end, and the selected candidate always
  survived every rung.
- **I2's shading.** Over 528 incumbent/deviation combinations (1,584 density checks): the shaded region is never nonzero to the right of b,
  never negative, and its trapezoidal area matches the analytic EI it is captioned with. The highlighted row is always
  the tabulated arg-max, and a tie is always reported as a tie rather than resolved by display rounding.
- **I4's gradient table.** Over a 34 × 15 × 8 grid of input, target and logit: the highlighted row is always the
  arg-min of the displayed gradient column; the minimum is always strictly negative when the gradient does not vanish
  (this is forced — Σpᵢ(oᵢ − ō) = 0 — so "the most negative gradient" is never a misnomer); and `descendId` is null
  whenever the gradient vanishes. **No tie at the arg-min anywhere on the grid**, so that particular ambiguity is not
  reachable.

## A6. Environment — **undisturbed**

NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1, SciPy 1.18.1, threadpoolctl 3.6.0 — exactly the versions the packet,
`automl-native.json` and `automl-data.json` record. The `dist-info` directories for scikit-learn and threadpoolctl date
from 2026-09-11 and SciPy from 2026-09-10, i.e. **before** this lesson's implementation window; the only site-packages
additions dated 2026-09-15 are `imbalanced_learn` + `sklearn_compat` (the immediately preceding lesson) and
shap/numba/llvmlite/slicer (an earlier one). **`flaml`, `tensorflow`, `keras` and `keras-tuner` are absent**, so the
declared trade in §8 was honoured in fact and not only in prose. The strongest evidence is A1 itself: my refit under
these exact versions reproduced every recorded number.

---

# Part B — Blocking

## B1. I4's architecture step grades the only correct answer wrong, and then states a falsehood

**Where.** `AutomlLabs.jsx:1123-1124` —

```js
const correct = applied.zeroGradient ? 'stay'
  : applied.movesTowardTarget ? 'toward' : 'away';
```

with `movesTowardTarget: Math.abs(updatedOutput - target) < Math.abs(residual)` at `automl-models.js:888`, and
`stepSize: { minimum: 0, maximum: 1 }` at `automl-models.js:47`. The step-size control at `AutomlLabs.jsx:1180`
inherits that minimum, so **0 is typeable and reachable by the spinner**.

**What is wrong.** Set the step size to 0 and leave everything else at the baseline. The stage's own prompt then
reads *"A gradient step of size **0** on the logits will move the mixed output:"* with options "toward the target",
"nowhere at all", "away from the target". The only correct answer is "nowhere at all". The gradient is not zero (it is
[0, −0.5, +0.5]), so `zeroGradient` is false; the output does not move, so
`|updatedOutput − target| < |residual|` is **false**; the ternary therefore falls through to `'away'`, and
"nowhere at all" is graded **wrong**.

Worse, the verdict it prints alongside the miss is self-refuting and contains a false statement. Driven on the real
page:

> *(verdict class `am-verdict is-miss`)* The updated logits are (0.693147, 0, 0), giving output 0 and loss 0.5 against
> the previous 0.5. **The weight on identity increases**, because descending on the logits raises whichever operation
> pulls the mixture toward the target.

The updated logits it prints are identical to the current ones. The output it prints is the previous output. The loss
it prints is the previous loss. And then it asserts that a weight increases, which at step 0 is simply false — no
weight changes at all.

**Why it matters.** This is the same class as Phase C finding 2, in the same investigation, one stage over: an answer
that is correct by the page's own displayed evidence, graded wrong. The fix for the argmin-over-zeros case introduced
`zeroGradient`/`zeroGradientReason` and a null `descendId`, but the step stage's three-way verdict was not revisited,
and it has a second route to "nothing moves" — a zero step — that the `zeroGradient` flag does not cover. A learner
who reasons correctly is told they are wrong *and* told something untrue about the mechanism.

**How verified.** Grid probe over five inputs × five targets × four step sizes found 20 such cases, all at step 0
(`scratch/independent-review-automl/probe.mjs`); then driven on the production build at 1366 px with Playwright/Edge,
verdict class and text read from the DOM, screenshot at
`scratch/independent-review-automl/shots/i4-step-zero-desktop.png`.

**Note for the fix.** A zero step is a legitimate and instructive setting — it is the cleanest demonstration that the
step size and the gradient are different things. The verdict should distinguish "the gradient vanishes" from "the step
length is zero", and the explanatory sentence must be conditional on a step actually having been taken.

## B2. I6 grades a tie wrong against a rule the page never declares, and its readout claims something the table contradicts

**Where.** `automl-models.js:812-819` (the `selected` reduce), `:825` (`reason`), `:827` (`selectionRule`),
`AutomlLabs.jsx:916` (`outcome: analysis.selectedId ?? 'none'`) and `:924` (the investigation's `note`).

**What is wrong.** `paretoAnalysis` breaks a tie among feasible candidates by greatest accuracy, then lower latency,
then earlier index. That rule exists as a string at `automl-models.js:827` — and **`selectionRule` is read by nothing**.
It is not rendered by the component, not quoted in the note, not in the legend, not in the readout. The
investigation's `note` declares only the *dominance* rule: "A candidate is dominated when another is no worse in both
objectives and strictly better in at least one. Two identical pairs therefore do not dominate each other."

The lab's own suggested-setup button **"Two identical candidates: make E match A exactly"** makes this reachable in
two clicks. With that setup and any cap in [2, 4), driven on the real page:

| candidate | latency (ms) | accuracy | dominated by | within the cap | shipped |
| --- | --- | --- | --- | --- | --- |
| A | 2 | 0.9 | nothing — on the frontier | yes | **yes** |
| … | | | | | |
| E | 2 | 0.9 | nothing — on the frontier | yes | **no** |

A and E are identical in every displayed column. Recording E returns:

> ≠ You recorded **E (2 ms, 0.9)**; the calculation gives **A (2 ms, 0.9)**.

The verdict prints the two options as the same pair of numbers and calls one of them wrong. And the readout beneath it
states:

> Under the cap the choice is A — **A has the greatest accuracy among the candidates at or under 3 ms.**

That is false. E has the same accuracy. The sentence is generated unconditionally at `automl-models.js:825` with no
tie branch.

The tie is also reachable without the suggested setup: type D's accuracy as `0.94` (a permitted 4-decimal value) and
leave the default 5 ms cap — B and D then tie at 0.94 and the page ships B with the same unqualified "greatest
accuracy" claim.

**Why it matters.** Every sibling investigation handles this correctly, which makes I6's omission an inconsistency
rather than an oversight of the whole design. I2 offers an explicit "They tie" option and `acquisitionTable` returns
`winner: null` with a `tied` list, and the readout says *"A tie is reported as a tie rather than resolved by display
rounding."* I3 states its tie rule in the prose above the control ("Ties at a rung take the earlier candidate letter,
and that rule is fixed before the task"). I5 states its tie rule in the investigation note, and the whole point of
its §5 narration is a tie being broken by a declared rule. I6 alone has no tie option, no declared tie rule, and a
readout that asserts uniqueness it has not established. A learner reasoning correctly from the page's own table is
graded wrong.

**How verified.** `probe.mjs` against the shipped `paretoAnalysis` (4 caps × identical-pair setup, plus a sweep of D's
accuracy at the default cap); then driven on the production build with Playwright/Edge — table, note, verdict and
readout read from the DOM; screenshot at `scratch/independent-review-automl/shots/i6-identical-tie-desktop.png`.
`selectionRule` confirmed unreferenced by grep across `src/` and all four verifiers.

## B3. I1 displays the graded answer before the prediction is recorded

**Where.** `AutomlLabs.jsx:136` and `:141` —

```js
const branch = draftCount?.branches.find(entry => entry.family === family);
...
<span className="am-branch-count">{branch ? `${branch.expression} = ${branch.count}` : 'invalid'}</span>
```

`draftCount` is `countConfigurations(draft.space)` (`:112`) — the **proposed** space, recomputed live on every edit.

**What is wrong.** On first paint, before any prediction is recorded, the registry shows
`2 × 2 = 4`, `2 = 2`, `1 × 2 = 2`, `1 × 1 × 3 = 3`. Click the investigation's own "Add tree depth 8" button and the
tree branch becomes `3 = 3` — still before any prediction. The prompt immediately below reads *"The applied space holds
11 valid configurations. Applying the draft, what happens to that number?"* with options "It shrinks / Exactly
unchanged / It grows" and an optional numeric field for the new total.

Both are on screen. The direction is the sign of 4+3+2+3 versus 11; the exact total, which the optional guess asks for,
is the sum of four displayed numbers.

**Why it matters.** This is the investigation whose entire teaching job is that conditional spaces *add* across
branches and *multiply* within one — and the page does that arithmetic for the learner, in the notation the answer
panel will later use, before they commit. It also contradicts three of the lesson's own statements: the shared
scaffolding's header comment (`AutomlShared.jsx:11-14`, "Nothing is revealed on first paint"), the labs module's header
(`AutomlLabs.jsx:19-22`, "only then sees a result computed from the inputs that prediction was recorded against"), and
the explicit guards its siblings carry on the same kind of input table — I2's *"The raw inputs. Nothing derived from
them is shown until a prediction is recorded"* and I6's *"The raw pairs. No feasibility flag, dominance mark or winner
appears until a prediction is recorded."* I1 has no such guard and needs one most.

I measured this across all six investigations: **I1 is the only one that exposes a derived quantity before a
prediction.** The other five show inputs and option labels only.

**Why the verifier misses it.** The Phase C record describes the replacement assertion as "every investigation shows at
least fifteen controls and exactly zero verdicts, readouts, rungs or SVGs before a prediction". `.am-branch-count` is a
plain `<span>` — not a verdict, readout, rung or SVG — so the guard passes. The guard enumerates *containers* rather
than asking whether any derived number is on screen.

**How verified.** Driven on the production build; branch counts read from the DOM before any interaction and again
after the suggested-setup click; per-investigation census of verdicts/readouts/SVGs/rungs/derived text at first paint.
Screenshot at `scratch/independent-review-automl/shots/i1-before-prediction-desktop.png`.

---

# Part C — Should-fix

## S1. Three of the 102 math payloads escape the escape audit — and they are the three highest-risk ones

**Where.** `scripts/verify-automl-models.mjs:1153-1154`.

```js
const payloads = [...lessonSource.matchAll(/<Math(?:Block)?>\{'((?:[^'\\]|\\.)*)'\}/g)].map(match => match[1]);
assert(payloads.length > 40, `only ${payloads.length} math payloads were found; the scan is inert`);
```

The regex matches **single-quoted** payloads only. I ran it against the topic body: the file contains **102**
`<Math>`/`<MathBlock>` openings; the regex captures **99**. The three it misses are template literals, at
`src/learn/data/topics/automl-neural-architecture-search-nas.jsx:246` (two) and `:339`. They look like

```jsx
<Math>{`K=\\begin{bmatrix}${kernel.matrix[0].join('&')}\\\\${kernel.matrix[1].join('&')}\\end{bmatrix}`}</Math>
```

These are the **only** payloads on the page built by string interpolation, and they carry exactly the
`\begin{bmatrix}` / `\\` escapes the audit exists to protect. The `> 40` floor cannot detect the shortfall, and
nothing ties `payloads.length` to a count of `<Math` tags.

**Why it matters.** The Phase C record presents this as "A standing escape audit over this lesson's own math payloads
… 99 payloads, 146 LaTeX commands, every backslash intact". The count reported is honest about what was scanned, but
the framing ("this lesson's own math payloads") reads as complete when it is 97% with the riskiest 3% excluded. The
packet's own gotcha list records a heredoc eating a backslash as a live hazard.

**Fix shape.** Widen the regex to template literals, and assert
`payloads.length === (lessonSource.match(/<Math(?:Block)?>/g) || []).length`.

**How verified.** `scratch/independent-review-automl/mathcount.mjs`, running the verifier's own regex verbatim.

## S2. The "8 px SVG text floor" measures the wrong quantity, records a misleading number, and passes on an empty set

**Where.** `scripts/verify-automl-browser.cjs:491-493`.

```js
const smallest = await page.locator('.am-lesson svg text').evaluateAll(items => Math.min(
  ...items.filter(item => item.getClientRects().length).map(item => Number(getComputedStyle(item).fontSize.replace('px', '')))));
assert.ok(smallest >= 8, `SVG text renders at ${smallest}px at viewport ${width}px`);
```

`getComputedStyle(text).fontSize` inside an SVG returns the size in **SVG user units**, before the viewBox-to-box
scale. Every figure here is a 340-unit viewBox rendered into a box narrower than that, so the two differ.

Two consequences:

1. **The assertion is scale-invariant and therefore cannot detect the regression it exists for.** The evidence file
   records *"SVG text at 10.5px or larger"* at **both** 390 px and 320 px — the same number at two widths, which is
   itself the giveaway. Shrink the figure container to 160 px and it would still report 10.5.
2. **It passes vacuously on an empty set.** `Math.min(...[])` is `Infinity`, and `Infinity >= 8` holds. If every SVG
   label vanished, the check passes and `records.push` at `:499` writes "SVG text at Infinity px or larger" into
   `automl-browser.json`.

I measured the rendered size directly (computed font-size × boundingRect.width ÷ viewBox.width) at 320 px:
**minimum 7.99 px, median 8.65 px**, n = 78.

**Why it matters.** The Phase C record's "Still open" entry says "SVG text in the inline figures renders at about
8.4 px at a 320 px viewport. **The browser verifier asserts a floor of 8 px**". The verifier asserts no such thing.
The actual rendered minimum is right at, and marginally below, the floor the record believes is being enforced. See
O8 for my judgement on whether the mitigation is sufficient — the mitigation is fine; the *guard on it* is not.

**Fix shape.** Multiply by `svg.getBoundingClientRect().width / svg.viewBox.baseVal.width`, and guard the set is
non-empty before `Math.min`.

**How verified.** Read the verifier line; re-ran the browser verifier and read the number it wrote into
`docs/teaching/evidence/automl-browser.json`; measured directly in my own drive at 390 and 320 px.

## S3. The cap-boundary guard cannot fail for the exact regression it was added to catch

**Where.** `scripts/verify-automl-browser.cjs:342-347`.

```js
capGeometry.marks.filter(mark => !mark.infeasible).forEach(mark => {
  assert.ok(mark.x <= capGeometry.bandLeft, ...);
});
```

The band is drawn at `scaleX(cap) + 1.5` (`AutomlLabs.jsx:987`), and candidate D sits at exactly `scaleX(cap)` under
the 5 ms cap. The bug this guard was added for is the band starting *at* the line rather than past it. Remove the
`+ 1.5` and `mark.x === bandLeft`, which `<=` accepts. The guard is silent on the one regression it names.

The comment two lines above (`:334-336`) says the band "must therefore begin **strictly** to the right of every
feasible marker" — the assertion does not say that.

Secondarily, neither partition is asserted non-empty (`:341` guards only `marks.length >= 5`), so a regression that
marked everything feasible would leave the second loop vacuous.

**Fix shape.** `<` rather than `<=`, or assert `bandLeft > scaleX(cap)` directly; and assert both partitions non-empty
for the fixtures that should produce both.

**How verified.** Read the verifier and the component; the geometry itself I swept over the whole cap grid in A5 and
it is currently correct — this is a finding about the guard, not the drawing.

## S4. F4 prints a number at full binary-float resolution that its own boxes and table print to two decimals

**Where.** `src/learn/data/automl-models.js:966` —

```js
note: `w' = w − ξ(w − α) = ${stepped}`,
```

rendered at `AutomlFigures.jsx:407`.

**What is wrong.** On first paint, at every width, the sentence under F4's second lane reads:

> w' = w − ξ(w − α) = **0.020000000000000004**. Here the current w is held fixed inside the one-step formula.

The node box in the same lane prints `w' = 0.02`; the lane table in the same figure prints `0.02`. Nothing else on the
page displays a number this way.

**Why it matters.** This is the class of Phase C finding 7 ("A logit field displayed 0.69314718055994…"), which the
record says was fixed and guarded. The guard added for it — *"the browser verifier asserts that no displayed number
carries more decimals than its control will take"* — inspects `input[type=number]` values only, so a figure's prose is
unprotected. I swept every visible text node on the page for decimals with nine or more fractional digits: this is the
only illegitimate hit. (The others are `0.050040083` and friends, which are deliberate eight-decimal displays, and
`0.9853658536585366` inside the recorded stdout of the displayed program, which is correct verbatim output.)

**Fix shape.** Format `stepped` at the figure's own resolution, as the box and table already do.

**How verified.** `scratch/independent-review-automl/longfloats.cjs` — a TreeWalker sweep over the rendered page, on
first paint and after toggling every figure alternative. Also visible in
`scratch/independent-review-automl/shots2/F4-1366.png`.

## S5. Three model-verifier checks cannot fail, and one 15-check sweep asserts nothing about any control

The Phase C record says the verifier was audited for weakened assertions and three were found. The audit was not
thorough. These are in addition to those three:

- **`verify-automl-models.mjs:304-305` compares a helper to itself.**
  ```js
  close(density.exact, expectedImprovement(0.4, candidate.mean, candidate.deviation),
    'the reported exact value is the analytic one', 1e-15);
  ```
  `improvementDensity` sets `exact: expectedImprovement(best, mean, deviation)` at `automl-models.js:414`, from the
  same three arguments passed at `:299`. This is `f(a,b,c) === f(a,b,c)`, bit-identical. The verifier's own header
  (`:6-8`) states that "an identity is checked by a second route, not by calling the same helper on both sides".
  The genuine second route is already on the next line (`:307`, trapezoid vs analytic); `:304` adds nothing.
- **`:301-303` restate a ternary.** `weighted >= 0` and `weighted === 0 for loss >= b` are precisely the two branches
  of `weighted: loss < best ? (best - loss) * density : 0` at `automl-models.js:404`. `:302` is also filter-then-assert:
  `.every()` on an empty filter is `true`.
- **`:1113-1120` and `:1122-1127` — 15 + 3 checks — assert arithmetic over two verifier literals.**
  ```js
  [[8, 1], [10, 0.01], ..., [0.4, 0.1]].forEach(([value, step]) => { assert(onStep(value, step), ...); });
  ```
  The `step` operand is hardcoded in the verifier. `limits` (`automl-models.js:31-59`) exposes no step field; the real
  steps are string literals in the JSX (`AutomlLabs.jsx:337, 339, 342, 529, 757, 929, 931, 936, 1180-1187`). The
  check's stated purpose is "every value a suggested setup or the prose asks a learner to type must land on its
  control's step" — but change `step="0.05"` to `step="0.03"` in the JSX and all of them stay green. It is a
  consistency check between two of the verifier's own transcriptions.

**Fix shape.** Delete `:304` (the second route already exists at `:307`); delete or re-derive `:301-303`; export a
`controlSteps` map from `automl-models.js`, consume it in both the JSX and the verifier, or drop the block.

**How verified.** Read each line and its target; the `improvementDensity` identity confirmed by reading
`automl-models.js:404` and `:414`; the step literals located by grep in the JSX.

## S6. "82 trust-root values re-derived" is a hand-maintained counter, not coverage; about half the leaves are re-derived

This is the item the brief asked me to state explicitly.

**What the verifier genuinely does, and it is good.** `verify-automl-data.py` imports no SciPy (only `math.erf` at
`:181`), imports nothing from `author-calculations.py`, and hand-writes its softmax (`:194-198`), its halving loop, its
pairwise dominance and its Hamming kernel. `author-calculations.py:7-8` uses `scipy.special.ndtr` and
`scipy.special.softmax`. The two sides are genuinely different code. That claim holds.

**What the number 82 is.** `trust_root_checks` is a counter incremented by literals at `:257, 263, 270, 271, 299, 309,
325, 356, 375, 390, 434`, summing to exactly 82. It is not a count of leaves covered:

- `:267-270` re-checks the same three `ei` leaves under a common offset and adds 3 again.
- `:299` adds `3 * len(gradient) + 8` = 17 for a block whose distinct gradient leaves number 3.
- `:265`/`:271`/`:406` add toward the tally for checks that read **no packet key at all**. `:406` is
  `expect((5 + 1) * 12 + (12 + 1) == 85, ...)` — pure verifier-internal arithmetic.

**Actual leaf coverage.** Enumerating every scalar in `constructed`: **149 leaves.**

| subkey | leaves | independently re-derived | consumed as a declared input and echoed | referenced by nothing |
| --- | ---: | ---: | ---: | ---: |
| `conditional_counts` | 5 | 5 | 0 | 0 |
| `ei` | 3 | 3 | 0 | 0 |
| `mixture` | 23 | 15 | 8 | 0 |
| `discretization` | 7 | 2 | 3 | 2 |
| `bilevel_scalar` | 8 | 5 | 3 | 0 |
| `halving` | 49 | 18 | 30 | 1 |
| `portfolio` | 9 | 0 | 9 | 0 |
| `pareto` | 18 | 3 | 15 | 0 |
| `additional_checks` | 27 | 27 | 0 | 0 |
| **total** | **149** | **78** | **68** | **3** |

**So: roughly 52% of the trust root's leaves are independently re-derived by the data verifier.** The other 48% are
*inputs* — the mixture's logits, operation outputs, target and step size; all 27 halving curve values and the resource
ladder; all nine portfolio losses; all fifteen Pareto labels, latencies and accuracies. They are consumed by the
Python and cross-asserted against the JS declarations by `verify-automl-models.mjs:365-367, 703-706, 731-732`, so a
wrong input passes everywhere as long as the two files agree — and **nothing checks either against `lesson.md`**
(see O5). Some are pinned indirectly (corrupting `mixture.logits` breaks the probability leaves; corrupting
`pareto.latency_ms` likely breaks `nondominated`), but `halving.curves`' 27 values are pinned only through
`selected_final_loss` and the survivor ordering.

Two leaves are asserted by **nothing in the repository**: `constructed.halving.selected` (grep-confirmed absent from
all three numerical verifiers — the Python reads `selected_final_loss`, the JS asserts `selectedId === 'C'` against
its own literal), and `constructed.discretization.probabilities` on the Python side (`:303` discards the softmax
output; it *is* covered on the JS side at `verify-automl-models.mjs:874`, so it is not orphaned overall).

**This is a documentation-and-coverage finding, not a correctness one.** I re-derived all 149 leaves myself (A3) and
found no disagreement. The numbers are right. What is not right is the record's "**82 trust-root values re-derived**"
and "all 82 trust-root values agreed", which read as complete coverage of a block that has 149 leaves.

**Fix shape.** Replace the counter with a set of packet leaf-paths actually asserted and report `len(covered)/149`;
add re-derivations for `halving.selected` and (on the Python side) `discretization.probabilities`; drop or re-point
`:406`.

**How verified.** Enumerated the leaves myself; re-read every `count +=` site; grep-confirmed the two unreferenced
keys; re-ran the verifier and read its printed "82".

## S7. F3's key caption is unreadable at desktop width without sideways scrolling

**Where.** The `<caption>` of F3's "Exact outcomes" table, inside `.am-table-scroll` (`AutomlShared.jsx:156-160`).

At 1366 px the caption measures **829 px inside a 745 px scroller**. The sentence it clips is:

> Exact outcomes. Fold sizes are 314, 300, 305 rows, so the mean of fold accuracies and the pooled out-of-fold
> accuracy **are two different rules and need not agree.**

That is the one sentence in the figure that disambiguates its two adjacent numeric columns — and my refit confirms the
distinction is live: mean-of-folds and pooled out-of-fold differ for **ten of the eleven** candidates. A desktop reader
sees the first half of the caption and has to scroll the table sideways to find out why the two columns disagree.

It is the only caption on the page in this state; the other six fit. At 320 px the table stacks and the caption fits,
which is why the narrow-width checks miss it: `verify-automl-browser.cjs:485-489` asserts caption block-display and
`caption.width > 120` only at 390 and 320 px, and `:488-489`'s `A table still scrolls sideways` check likewise runs
only at narrow widths. Nothing checks desktop.

**Fix shape.** Move the caption outside the scroll container, or let it wrap at the container's width rather than the
table's.

**How verified.** `scratch/independent-review-automl/figures.cjs` measured `clientWidth` of every
`.am-table-scroll` against its table's `scrollWidth` and its caption's rendered width at 1366 and 320 px; visible in
`scratch/independent-review-automl/shots/figure-2-desktop.png`.

## S8. `halvingSchedule` throws a bare `TypeError` for inputs its own guard admits

**Where.** `automl-models.js:437-439`.

```js
if (!Array.isArray(curves) || curves.length < keepFraction) {
  throw new RangeError(`Run at least ${keepFraction} candidates.`);
}
```

The guard admits three or more curves. With three or four curves on the three-rung ladder the second rung computes
`Math.floor(alive.length / 3) === 0`, leaves `alive` empty, and the third rung dereferences `curves[undefined]`:

```
halvingSchedule(3 curves) -> TypeError: Cannot read properties of undefined (reading 'losses')
halvingSchedule(4 curves) -> TypeError: Cannot read properties of undefined (reading 'losses')
```

**Why it matters, and why it is not blocking.** It is **not reachable from I3** — the curve count is fixed at nine and
the lab offers no add or remove. So nothing a learner can do triggers it. But the function is exported, the model
verifier's refusal suite tests only `declaredCurves.slice(0, 2)` (`verify-automl-models.mjs:420`, correctly refused
with a `RangeError`) and never 3–8, and the Phase A record claims "every entry point refuses out-of-range, duplicated,
empty and malformed input". Here the guard passes an input the body cannot handle, and the failure mode is a
`TypeError` rather than the `RangeError` the rest of the module raises — `refuses()` requires `RangeError`
specifically (`:49-52`), which is good discipline, and this entry point would fail it.

**Fix shape.** Require `curves.length >= keepFraction ** (resource.length - 1)`, or clamp `keep` to at least 1 at every
rung.

**How verified.** Executed against the shipped module.

---

# Part D — Verifier counts, re-run

All four reproduce their declared counts on my machine, against the production build I made
(`npx vite build --outDir dist-automl-review`) and previewed on 127.0.0.1:4189:

| Verifier | Declared in `design.md` | Observed |
| --- | --- | --- |
| `verify-automl-models.mjs` | 316 grouped checks / 61 groups | `PASS: 316 grouped AutoML model checks across 61 groups.` |
| `verify-automl-data.py` | 289 checks, 82 trust-root values, module byte-identical | `PASS: 289 grouped checks across 11 groups; 1372 rows, 1348 feature groups, 35 estimator fits recomputed and matched; 82 packet trust-root values re-derived independently; module byte-identical (31 KB).` |
| `verify-automl-examples.py` | 104 oracle assertions, 1 of 4 blocks executed | `PASS: 1 of 4 displayed blocks executed, 104 oracle assertions, 35 estimator fits; module current.` |
| `verify-automl-browser.cjs` | 18 cases / 31 screenshots | `{"status":"passed","cases":18,"evidencePath":...,"sourceFiles":11,"screenshots":31}` |

The counts are honest. What they do **not** assert is S1, S2, S3, S5, S6 above and O3, O4, O5, O6 below — and,
materially, none of the three blocking findings.

---

# Part E — Observations

**O1. `ATTRIBUTION.txt`'s contents are asserted by nothing.** `verify-automl-models.mjs:509` checks it exists,
`verify-automl-browser.cjs:214-215` checks it returns HTTP 200, and both hash it as an owned file. Nothing asserts that
the row count, byte count, digest, duplicate accounting or licence *inside it* agree with the CSV or with
`automl-data.js`. I checked every number in it by hand and every one is correct today — 1,372 rows, five columns,
CRLF, 46,400 bytes, the true SHA-256, 1,348 unique vectors, 11 repeated contributing 24 extra rows, no conflicting
labels, Volker Lohweg 2012, donated 15 April 2013, DOI 10.24432/C55P57, CC BY 4.0, retrieved 12 September 2026, and
the "curtosis" source spelling noted. The analogous claims in `data-source.json` **are** asserted
(`verify-automl-data.py:488-496`). A future change to the served CSV would keep all four verifiers green while leaving
the public-facing attribution silently wrong. This is the one real hole in an otherwise unusually tight provenance
chain.

**O2. The archive SHA-256 is unfalsifiable locally.** `1e2acd9a…05227` appears identically in `ATTRIBUTION.txt:8`,
`data-source.json:4`, `data-provenance.md:12` and `automl-data.js:41`, and is contradicted by nothing — but the ZIP was
deliberately not retained (`data-provenance.md:100`), so no verifier and no offline reviewer can confirm it.
Trust-on-first-ingestion. Everything downstream of the extracted member is independently checkable, and I checked it.

**O3. `verify-automl-models.mjs:1194` hides a missing source file.** `sources.filter(fs.existsSync)` silently drops any
of the eleven declared source files that no longer exists and still writes `passed: true` with ten hashes.
`verify-automl-browser.cjs:105` does the same job unfiltered and would throw. The weaker one is on the verifier that
runs more often.

**O4. The curve-through-label sweep inspects no figure shape.** `verify-automl-browser.cjs:71`'s
`if (shape.classList.contains('am-grid')) continue;` is dead: `am-grid` is only ever applied to `<line>` elements
(`AutomlShared.jsx:330`, `AutomlFigures.jsx:272, 278`) while the enclosing loop selects `path, polyline, polygon`. More
consequentially, `AutomlFigures.jsx` contains **no** `<polyline>` or `<polygon>`, and its only `<path>` elements are
three arrowhead markers inside `<marker>` defs (`:26, 29, 32`), which have no `getScreenCTM()` and are skipped at
`:76`. The sweep's stated purpose (`:47-49`) names flow arrows, but it passes its `inspectedShapes > 0` guard entirely
on lab polylines. The copied sampler did earn its place — it caught the counterfactual-trace collision the shared
inspector missed — but it is not covering the figures.

**O5. The manuscript's prose numbers are checked against nothing.** `verify-automl-examples.py:63` pins `lesson.md` by
SHA-256 but extracts only its seven fenced code blocks. The prose values that `verify-automl-models.mjs` asserts "as
the manuscript prints it" (`:246-248`, `:373-375`, and elsewhere) are hand-transcribed literals in the verifier. A
manuscript number that drifted from the packet would fail nothing. I checked every prose number by hand in A1 and A3
and found no disagreement — but the repo does not establish that.

**O6. `author_checks` and `tie_rule` are read by nothing.** `calculated-inputs.json`'s top-level `author_checks` block
is eighteen self-attested booleans (`"mixture_gradient_central_difference_verified": true`, …) with no oracle; grep
returns zero hits across all three numerical verifiers. Same for the `tie_rule` string. I did independently confirm
one of them — `local_topic_routes_found: 9` is correct (see O7).

**O7. Record off-by-one in departure 12.** `design.md` says internal links use the module-preserving route "for the
eight classical-ML destinations". There are **seven** distinct classical-ML targets (cross-validation, feature scaling,
feature selection, Gaussian processes, hidden Markov models, imbalanced learning, regularization) plus two out-of-module
(`neural-architecture-search-nas`, `automl-as-meta-learning`) — nine internal targets in all. The manuscript and the
implemented body agree on all nine, each route form is correct, and nothing is missing; the "eight" is an arithmetic
slip in the record, not a defect in the lesson.

**O8. The 320 px SVG-text mitigation is sufficient; the claim about it is not (S2).** At 320 px the figure type renders
at about 8.0 px, which is genuinely small. I judge the mitigation adequate, for three reasons I checked rather than
assumed: (i) every figure's numbers are duplicated in an adjacent stacking table — I read F2's at 320 px and it carries
all three architectures, every block equation and all three parameter counts (49, 97, 121); (ii) each SVG carries a
full `aria-label` narrating its content, so the drawing is not the only route to it; (iii) nothing on the page is
*only* in the drawing. At 320 px the document `scrollWidth` equals the viewport exactly, no element crosses the
viewport edge, and no SVG text escapes its own viewBox. The claim that needs fixing is the verifier's, not the design's.

**O9. §8's trade is sound and the page is honest about it — I endorse it.** The reasoning is right: resolving
`flaml[automl]` and TensorFlow into the shared `scratch/lesson-tools` runtime could move NumPy or scikit-learn, and
several completed lessons' recorded outputs are pinned to those exact versions. I confirmed the four packages are
genuinely absent and that the four core packages' install dates predate this lesson's window (A6), so the trade was
honoured in fact. The page says so plainly, in the section's own opening prose: *"Neither library is installed in this
site's lesson runtime, so **no output is recorded for them here**: run them yourself and record the versions you
used."* `automl-native.json` records `executed: false` with `stdoutHash: null` for both, names the four uninstalled
packages and the reason. And the substantive claim is not merely asserted — the verifier imports the extracted
`banknote_search.py` and resolves `folds[0]`, recording `firstFoldFittingRows: 605`, `firstFoldValidationRows: 314`,
`protectedRowsTouched: 0`, which my own refit reproduces exactly. A learner is told what was run and what was not, and
the one claim that could mislead is checked for real.

**O10. F4's dependency labels crowd their nodes at 1366 px.** "∂w'/∂α = ξ" and "∂w*/∂α = 1" sit flush against both the
α node's right edge and the weight node's left edge; the one-character-shorter "∂w/∂α = 0" in lane 1 keeps a visible
gap. No glyph overlaps another glyph, so this is below the bar the Phase C pass set, but it is the same crowding the
record says it fixed for lane 1 — worth a look when S4 is fixed in the same figure.

**O11. Two leads I checked and discarded, recorded so they are not re-found.** (a) My viewBox-escape probe flagged the
text "4 rows in" at x = −40 in a 340-unit viewBox in F2. It is a **rotated** label
(`AutomlFigures.jsx:209`, `transform="rotate(-90 12 58)"`) running vertically up the left gutter, correctly placed; my
probe ignored the transform. Not a defect. (b) An element screenshot of F5 showed its first lane's boxes clipped to
half-height glyphs. That was the page's sticky header painting over a Playwright element capture. I re-captured with
page chrome neutralised and ran an ancestor-clip probe over every figure SVG at 1366 and 320 px: **zero clipped**.
F5's short-label fix is real and all nine box labels render in full. I also checked for the sister lesson's
`fill="none"`-versus-CSS-`fill` failure: **zero conflicts, and zero shapes whose computed fill and stroke are both
absent**, at 1366, 390 and 320 px.

**O12. Departures 1–12 audited against the tree; eleven correct, one off by one (O7).** The blueprint **is** registered
(`blueprints/index.js:24, 151`) under `'AutoML & Neural Architecture Search (NAS)'`, matching
`track-definitions.js:136`, so departure 1 is settled as the record says. `\widehat` appears **zero** times in the body
(departure 2). The chain-rule vector `v` is introduced one sentence earlier, at topic jsx:324 (departure 3). I1 keeps
all four family branches with a "cannot empty a dimension" invariant rather than an enable toggle (5), grades direction
plus an optional total (6), and puts the sampling comparison inside a post-reveal `<details>` (7). I4's step and
commitment are separate stages, each with its own radio group, each keyed on the applied inputs so an edit remounts and
retires them (8). F2 uses a selector with the comparison table always visible (9). F3 labels by short code with full
labels in the table (10). The mistake table covers all eleven candidates — I confirmed the 119 exposed rows are exactly
the union of all eleven error sets plus the three inspection errors (11). Departure 4 is O9.

---

# Summary

The mathematics is right. I refitted the entire 35-fit campaign from the served CSV and re-derived every value in the
packet's `constructed` block from the declared settings alone, in exact rational and 60-digit arithmetic, each by a
second independent route — **and found no disagreement anywhere**: not in the eleven fold-accuracy triples, not in the
10,109 out-of-fold labels, not in either confusion matrix, not in the expected improvements, not in the halving
schedule, not in any of the three bilevel derivatives, not in the `6h+1` claim at any permitted width, not in the
configuration count, not in a single practice solution. The builder's claimed second routes are, with the one exception
at S5, genuinely independent of what they check. Across 72,427 grid points covering the learner's whole enterable
input space, every drawn rule agrees with every applied rule — the Pareto cap-boundary fix is real and correct. The
lesson's central caution holds exactly: the shipped data module exposes 119 rows, every one of which a search
legitimately saw, and **not one reserved row**.

The three blocking findings are all of one kind, and it is the kind the brief asked me to hunt: **the page grades a
learner wrong for an answer its own displayed evidence supports, or shows them the answer before asking for it.** B1
marks "nowhere at all" wrong for a zero-size step while printing identical before-and-after numbers and then asserting
a weight change that does not happen. B2 marks E wrong in favour of A while printing them as the same pair of numbers,
against a tie rule that exists in the model layer and is rendered nowhere, and captions it with a uniqueness claim its
own table refutes. B3 computes and displays the branch counts whose sum is the graded answer, live, before the
prediction is recorded — in the one investigation whose whole purpose is that arithmetic.

B1 and B2 are each a second instance of a class the Phase C pass found and fixed once. That is the useful lesson here:
the fixes were correct but were applied to the instance rather than to the class, and the guards added afterwards
enumerate containers (verdicts, readouts, rungs, SVGs) or inspect controls rather than asking the general question —
*is any graded quantity on screen before the commitment, and can any displayed state support more than one defensible
answer?* Neither B1 nor B2 nor B3 is visible to any of the 727 assertions the four verifiers currently make.
