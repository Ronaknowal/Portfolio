# ML Problem Formulation, Baselines & Data Leakage — independent phase-two review

Reviewed 21 September 2026 against the working tree plus the uncommitted problem-formulation implementation.
The reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/`,
`docs/teaching/drafts/` or `docs/teaching/topic-notes/` was edited; this review document is the only write
outside `scratch/formulation-independent/`. No state-changing git command was run. The build directory
`dist-form/` was deleted and the preview on 4195 stopped at the end.

Three housekeeping notes, stated up front because all three bear on trust in the record:

- **I did not run `falsify-formulation.mjs`.** Its browser cases run `npx vite build --outDir dist-form`
  (lines 611 and 659), which is the same directory this review's preview was serving, and the instruction was
  never to run two instances. I audited the harness by reading it line by line and cross-checking its committed
  report against the tree instead. Its counts are independently corroborated below where the tree allows.
- I re-ran the four offline verifiers with `--no-evidence`, which every one of them honours
  (`verify-formulation-sources.py:367` even asserts that they do). Nothing on disk changed:
  `sha256sum -c scratch/formulation-independent/BASELINE-HASHES.txt` is clean for all 27 lesson files
  including the six evidence artifacts. I also took copies of all six under
  `scratch/formulation-independent/backup/` before starting.
- I rebuilt `dist-form/` myself. That directory is a build artifact, not evidence, and it was deleted
  afterwards.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | `scratch/formulation-independent/refit.py`, importing neither `formulation-models.js` nor `formulation-data.js` nor `formulation-examples.js` nor any `verify-formulation-*` script nor the packet's `author-calculations.py`. Run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1). A **full refit of the study from the served CSV**, with the design matrix rebuilt by hand — median impute and population-variance standardisation fitted on the training rows only, one-hot blocks built from **sorted training categories** with unknown levels all-zero. Average precision summed over **distinct score thresholds from its definition in exact `Fraction` arithmetic**; log loss, the four confusion cells, the top-50 and top-25 selections and the full ranking all written out here. The §3 admissibility rule re-implemented as a filter-then-`max` over `(event, available, version)`. §4 and §8 arithmetic in exact rationals. Reviewer-written Playwright drives (`drive.cjs`, `overflow.cjs`, `zoom.cjs`, `nulls.cjs`) sharing no code with `verify-formulation-browser.cjs`, at 1366, 390 and 320 px on Edge. An instrumented copy of the source-hygiene verifier to count its real assertions. A balanced-paren scanner for identical ternary branches across the five lesson modules. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `data-source.json`, `calculated-inputs.json`, `author-calculations.py`, `source-description.txt`); the lesson body, model layer, both generated modules, all three lab/figure/shared components and the CSS; all five verifiers and the harness; the blueprint; the served assets; the committed evidence JSON. I did **not** open the destination note — it belongs to the integration owner and I had no reason to. |
| **Reused (declared)** | scikit-learn's `train_test_split` at the declared seeds and `LogisticRegression` at the declared hyper-parameters, because the protocol *is* those calls. Everything around them — the design matrix, every metric, every selection, every count — is written here. Two subagents for a fan-out read of verifier internals. **Every finding a subagent surfaced I re-derived myself before reporting it**: I read each cited line, ran the instrumented count in S7 myself, and reproduced every browser finding in a live page. Two of their candidate findings I checked and **downgraded** (see O1 and the note under S7). |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx` | `e78a0c2b535120027f4a2e2a71731bb79de8a5189670eea9012ec213f38b5206` |
| `src/learn/data/formulation-models.js` | `d4b248ba818eb09722dfedc90640ddfb3af77649f9d877cf9c6c65610afb9545` |
| `src/learn/data/formulation-data.js` | `72ff276251e713ef6416ed8703b00de3a74404010dc3c298cdd1792823bdc1ae` |
| `src/learn/data/formulation-examples.js` | `f4ee3741cab02b659abdecc6ec312ab058a4869ea341e435523842620f8b0f43` |
| `src/learn/components/lesson-labs/FormulationShared.jsx` | `cdf25e4cd530bafa581da3d9317bc441edceaf9d40eb93cc132379e2fc5ef95f` |
| `src/learn/components/lesson-labs/FormulationLabs.jsx` | `d95ea8ccc00a37d14c482085de528f471696b29c42a042cfd7b19dbe5bf1c1d7` |
| `src/learn/components/lesson-labs/FormulationFigures.jsx` | `da1277acc759ed72e8cf2017a47f77d9e7fc4866ced49c1befd982eb90c1569d` |
| `src/learn/components/lesson-labs/formulation-labs.css` | `32988e8767bb32b844627f7a29e2a24822b08d5cb4107d4da83bd2f442164722` |
| `scripts/verify-formulation-models.mjs` | `99b5acfb8d83179ee3ffec45feaae6186704070fbe93fe18eeac386360496cb9` |
| `scripts/verify-formulation-data.py` | `9182931fdff0bb906c4c0b30f66a6ecff27a6278bb8104fb68cd1dc5ffe88e28` |
| `scripts/verify-formulation-examples.py` | `e95e3d4f2998db42a9c694280362e403272439a6772e66d6b7403590fc3cbab0` |
| `scripts/verify-formulation-sources.py` | `68b255d16fcd43f2bcec28fc40b603ae448ee7df3cadcc66e66ea0af02ab01cd` |
| `scripts/verify-formulation-browser.cjs` | `925d99ad50c8bdddc6df53f245063519d559fb5616a75bc6fd28f446dc134a86` |
| `scripts/falsify-formulation.mjs` | `26078d1d4f1a6c758a735c99c91bb91c3c0b5d49f8d8923926e46508ca036a59` |
| `docs/teaching/drafts/.../lesson.md` | `1ff63549d7055024cdc8a4a982d1af05b6c575e305b824f88f90d8f0642b3139` |
| `docs/teaching/drafts/.../design.md` | `42051cf0286053e0fedbf245cdf107f21e6feba14d114464f7eecb4e73e3cd6b` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `d694b6546ba00e7bc6d4592d21e973a4f7faa3d81551b252762b5b7d26b405a7` |

All eight source hashes in the Phase C record match this table exactly. The record is current with the tree.

---

# Part A — correctness

## A1. The study, refitted from the served CSV — **no disagreement**

`scratch/formulation-independent/refit.py`. Nothing imported from the lesson or its verifiers. The design
matrix is rebuilt by hand rather than through `ColumnTransformer`, so the one-hot column order, the imputation
medians and the standardisation constants are all independently derived from the training rows.

**Provenance.** The served
`public/learn-assets/problem-formulation/bank-additional.csv` is **583,898 bytes**, SHA-256
`7e59cf650004d65d1c9d6b08553bad2ee9a9ad70d594f536e3c584ee6ed5df50`, and is **byte-identical to the packet's
copy**. 4,119 rows, 21 columns, so the page's "20 input columns and a binary recorded subscription target" is
right; 451 positive rows; 3,959 rows carry the `pdays` sentinel 999. Every one of these matches
`formulation-data.js`.

**Splits.** Regenerated from seeds 53 and 54: development 3,295 / reserved 824, then train 2,471 /
validation 824. Checked myself for **disjointness** (development ∩ reserved = ∅, train ∩ validation = ∅),
**exhaustiveness** (development ∪ reserved is exactly `range(4119)`, train ∪ validation is exactly
development), **size**, and **stratification** — positive fractions 451/4119 whole, 361/3295 development,
45/412 reserved, 271/2471 train, 45/412 validation. Train positives 271, validation positives 90, reserved
positives 90. `trainPrior` reproduces to the last bit: `0.10967219749089438` = 271/2471. The full 824-element
`validation.ids` array matches element for element.

**Metrics, each by a second route.** Average precision summed over distinct thresholds in exact `Fraction`
arithmetic; log loss and the confusion cells written out; the ranking by an explicit
`(-score, source row)` sort rather than `numpy.lexsort`.

| | prior | candidate | duration |
| --- | --- | --- | --- |
| average precision (mine) | `0.10922330097087378` | `0.2534403031698255` | `0.46170012981199043` |
| average precision (page) | `0.10922330097087378` | `0.2534403031698255` | `0.4617001298119905` |
| log loss (mine) | `0.34488940603783713` | `0.336527423595858` | `0.2646387134831804` |
| log loss (page) | `0.34488940603783835` | `0.3365274235941854` | `0.2646387134831803` |
| confusion | `[[734,0],[90,0]]` | `[[718,16],[75,15]]` | `[[714,20],[66,24]]` |
| correct | 734 | 733 | 738 |
| top-50 positives | 6 | 20 | 26 |
| iterations | — | 54 | 60 |
| design width | — | 39 | 40 |

Every integer agrees exactly. The average-precision and log-loss gaps are 7e-17 and at worst 1.7e-12 —
summation order and a design matrix assembled in a different order — and all three round identically at the
six decimals the page prints. The candidate's 39 and the duration model's 40 one-hot columns reproduce the
recorded `candidateColumns` and `durationColumns`.

**The prior baseline's average precision is exactly 45/412**, the validation positive fraction, in exact
rational arithmetic. The page's claim that "its validation average precision equals the validation positive
fraction" is not an approximation; it is an identity, and my exact-`Fraction` route confirms it.

**Selections.** `policyFixtures` reproduces in full: top-25 positives 12; the top-25 identity list matches
element for element beginning 589, 2122, 696, 1846, 658; the recorded exchange is **589 → 1680 giving 11**;
`reorderNullPositives` is 12. Practice 4's `practiceAtTwentyFive` — computed on the page from `rankByScore`
rather than read from `policyFixtures` — lands on the same 12, so precision 0.48 and recall 12/90 = 0.133333,
against 0.4 and 20/90 = 0.222222 at capacity 50. The page's "precision rises here while recall falls" is
correct.

## A2. The three clocks — **no disagreement, and the whole grid agrees**

I re-implemented the admissibility rule as a filter (`entity`, `event ≤ cutoff`, `event ≥ cutoff − age`,
`available ≤ cutoff`) followed by `max` on `(event, available, version)`, written before reading
`latestKnown`'s fold. All six recorded cases reproduce: default → `A·e1·v1` value **10**; arrive_earlier →
`A·e4·v1` value **20**; latest_known_revision (cutoff 7) → `A·e1·v2` value **12**; new_event_now_known
(cutoff 9) → `A·e4·v1` value **20**; too_old (cutoff 5, age 2) → **null**; unrelated_entity_null → **10**
unchanged.

The practice fixture reproduces too: at cutoff 6 with age 5 the answer is **9** (event 2's revision, known at
5, beating version 1 on the later arrival); moving the event-4 arrival to 4 gives **11**; and with age 1 the
window is [5, 6] so **nothing qualifies in either arrival case** — which is exactly what practice 1's
derived sentence prints.

**The destination note's property is asserted over the whole grid, and I confirm it.**
`verify-formulation-models.mjs:190-222` sweeps 150 arrival configurations × cutoff 0…12 × maximumAge 0…12 ×
{`sensor_A`, `sensor_B`} = **50,700 cases**, floored at 50,000 (L1256), comparing the module's fold against a
second route written locally at L108-121 that sorts a zero-padded composite key and takes the last element.
22,672 selections and 28,028 missing calibrations are both floored above 1,000 (L224-226), so neither branch
is vacuous. On **every** case it additionally asserts the four admissibility predicates directly on the
selected record (L212-218) — this is the leakage bar, and it is met at every point, not at samples. The grid
covers `sensor_B`, cutoff 0 (below every event time), `maximumAge = 0`, and `available == event` (the
per-record sweep at L169 starts at `available = item.event`). I re-ran the suite: **PASS, 130 grouped checks
across 56 groups**, matching the Phase C record.

The one grid gap is **arrival < event**, which the sweep cannot construct because its lower bound is the event
time. That case is covered instead by a single refusal at L272 and, on the page, by a named refusal I
exercised live (see A4).

The specific 8→4 claim, however, is asserted as **two single points**, not as a property. `verify-formulation-models.mjs:254-263`
pins value 10 and selected event 1 at (cutoff 5, age 5, sensor_A), and value 20 and selected event 4 after the
arrival moves. The 50,700-case grid proves the two *routes* agree; it never asserts that changing an arrival
changes the selection. I checked the property myself over the whole grid and it holds wherever it is
meaningful, so this is a gap in the guard, not in the lesson — recorded as O5.

## A3. Every other number the prose states — **no disagreement**

- §4 prevalence: 90 negative, 10 positive → 9/10 accuracy, 0 positive recall. The page renders "90% accuracy
  and 0 recall" and figure 4 renders "A class prevalence of 10% … reach 90% accuracy". I checked the float
  path specifically, because `0.9 * 100` and `0.1 * 100` are the classic artifact sites: in V8 both round to
  exactly 90 and 10, and the rendered page confirms it.
- §4/practice 6: threshold 3/12 = **0.25** exactly; at p = .2 acting costs 12/5 = **2.4** and waiting 9/5 =
  **1.8**, so the cheaper choice is to wait — the page's `preferred === 'act' ? '' : 'not '` renders "do not
  act", correct since .2 < .25.
- §8 newsvendor: critical fraction 3/4 = **0.75**, optimal stock **20**, expected costs **15** and **5**,
  mean demand 15. Figure 7's second table carries 15 and 5 and marks 20 as the cheaper. The prose's "the
  expected costs of stocking 10 and 20 are the two values in figure 7's second table" is accurate — figure 7
  does carry a second table, and the figure numbering (ContractFlow 1 … Uplift 7) is consistent throughout.
- §8 threshold-versus-quantile: `costThreshold(1, 3) = 0.25` and `criticalQuantile(3, 1) = 0.75`, complements,
  as the page says.
- Practice 3: policy A correct **9/12** at cost **21**; policy B correct **11/12** at cost **2** and
  **exceeds** capacity 3 with four escalations. All four reproduce.
- Practice 8 / figure 7: A .8 vs .75 (increment .05), B .45 vs .1 (increment .35); `leaderByCalledProbability`
  A, `leaderByIncrement` B, `reverses` true. The float residue `0.8 − 0.75 = 0.05000000000000004` prints as
  `0.05` through `round(·, 6)`, correctly.
- The displayed `latest_known.py` program: I evaluated its four queries by hand — cutoff 5 → 10, 7 → 12,
  9 → 20 at `maximum_age=9`, and `None` at cutoff 5 with `maximum_age=2`. The recorded `expected` field is
  `"5 10\n7 12\n9 20\nNone"`. Agreement. `verify-formulation-examples.py` passes with **40 oracles**, both
  programs extracted verbatim (80 lines, 0 composed) and both modules byte-identical.
- All four offline verifiers reproduce the advertised counts on re-run: **130 / 56**, **10,163 checks with
  100% of 10,099 leaves** (3,506 + 2,474 + 4,119 = 10,099 — the arithmetic closes), **40 oracles**,
  **107 hygiene checks**.

**Summary: I found no numerical disagreement anywhere on this page.** Every recorded measurement, every
derived answer, every practice solution and every figure coordinate I recomputed agrees with an independent
derivation. The defects below are about grading, labelling, and what the guards and records claim.

## A4. What the page does with degenerate inputs — mostly right

Driven live at 1366 px:

- **Arrival before its event.** Setting `A·event 4·v1`'s arrival to 0 produces a named refusal
  ("would arrive at 0, before the event it measures happened at 4 … move the arrival to 4 or later"), the
  figure is withheld entirely (0 SVGs in the investigation), the Apply button is disabled, and a separate
  pointer says why. Correct, and the specification departure that makes the state reachable (design.md
  departure 3) is the right call.
- **Cutoff before every event.** Cutoff 0 grades `no record qualifies` and names three of the four rejection
  reasons. Correct.
- **Arrival equal to its event.** Both marks are drawn and both are visible — see B-section, figure pass.
- **No admissible match at all.** Handled as a result, with the reveal saying "A missing calibration is the
  answer here. Substituting a value that had not arrived would be the leak this section is about."
- **The two designed nulls both grade as nulls.** Reversing the selected block: "It does not move at all …
  20/50 = 0.400000 against 20/50 = 0.400000 … No identity entered or left the selected set, which is why no
  set metric could move." Changing the other sensor's value to 88: the selection and value are unmoved at
  `A·event 1·v1` / 10.

---

# Part B — blocking

## B1. Investigation 2's numeric tolerance is tighter than the precision the page prints, and it tells a correct learner their answer is outside — **blocking**

`src/learn/components/lesson-labs/FormulationLabs.jsx:576-582` sets `numeric: { tolerance: 1e-9,
required: true }` on the capacity investigation. The graded quantity is `positivesFound / capacity`, and
`src/learn/components/lesson-labs/FormulationShared.jsx:323` prints it through `fixed(value, 6)` — six
decimals — and the reveal at `FormulationLabs.jsx:592` prints it the same way.

The capacity control accepts 1…100. For most capacities the precision is a fraction whose decimal expansion
does not terminate in six places, so **the page's own printed value is never within 1e-9 of the answer.**
Because the guess is `required`, the learner cannot decline.

Reproduced live at capacity 3 (`scratch/formulation-independent/shots/cap-tolerance.png`):

> `=` Your prediction matches: It rises. **You wrote 0.666667 for the precision; the calculation gives
> 0.666667, outside 1.000 × 10⁻⁹ of it.** The selected 3 contain 2 recorded subscriptions, so precision is
> 2/3 = 0.666667 against 20/50 = 0.400000 before.

The page displays the two numbers as the **same string** and then declares them different. The reveal
immediately below reads `Precision  2/3 = 0.666667`, so there is nothing else a learner could have entered.
This is the "tolerance tighter than the precision the page prints" variant, in its sharpest form.

It is not a rare corner. Of capacities 1…100 on the candidate ranking, only about a fifth
(1, 2, 4, 5, 8, 10, 15, 16, 18, 20, 22, 25, 32, 40, 45, 50, 60, 64, 80, 100) yield a precision that terminates
within six decimals; the rest — 3, 6, 7, 9, 11, 12, 13, 14, 17, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, … —
do not. The two presets happen to land on 25 and 100, which is why the defect never showed in the recorded
captures.

The disclosure "Answers within 1e-9 are accepted" is honest about the rule but does not make the rule usable:
`1e-9` is also the only place on the page where a raw exponent is printed to a learner, and the verdict then
renders the same value as `1.000 × 10⁻⁹`.

**Suggested shape of a fix** (not applied): either grade the numeric answer at the precision the page prints —
a tolerance of `5e-7`, matching `fixed(·, 6)` — or keep 1e-9 and print the answer as the exact fraction the
reveal already shows. The timeline investigation gets this right by using `tolerance: 0` on a quantity that is
always an integer.

## B2. Investigation 2 states a stale baseline before every round after the first, and grades correct reasoning wrong — **blocking**

`src/learn/components/lesson-labs/FormulationLabs.jsx:566-570`:

```jsx
{!shown && <p className="formulation-caption">
  Section 5 records the starting point for you: at capacity {formulationData.partition.capacity} the
  candidate's selected set contains {formulationData.policyFixtures.top50Positives} subscriptions, a
  precision of {round(candidateProcedure.precisionAt50, 6)}. Predict what your change does to that.
</p>}
```

The grader compares against `state.active` — the previously **committed** state — not against capacity 50.
`shown` is nulled by `retire()` on any edit (`FormulationShared.jsx:180-186`), so this caption **reappears
before every subsequent round**, still naming capacity 50 and 0.4 and still instructing "Predict what your
change does to that". After the first commitment that sentence is false, and it is the only concrete baseline
number on screen at prediction time. The `formulation-pending` banner says only "the comparison will be
against the state currently applied"; it never says what that state's precision is.

Reproduced live (`scratch/formulation-independent/shots/cap-stale-baseline.png`):

1. Set capacity 25, predict "It rises", enter 0.48, apply → correct; the applied state is now capacity 25 at
   precision 0.48.
2. Set capacity back to 50. The caption reappears verbatim: *"at capacity 50 the candidate's selected set
   contains 20 subscriptions, a precision of 0.4. Predict what your change does to that."* A learner who does
   exactly that predicts **"It does not move at all"** (0.4 → 0.4) and enters 0.4.
3. The verdict:

> `≠` You recorded It does not move at all; the calculation gives It falls. **You wrote 0.4 for the precision;
> the calculation gives 0.400000, within 1.000 × 10⁻⁹ of it.** The selected 50 contain 20 recorded
> subscriptions, so precision is 20/50 = 0.400000 against 12/25 = 0.480000 before.

The same verdict confirms the learner's *number* is right and marks their *direction* wrong, because the page
told them to compare against a baseline the grader does not use. This is the dominant defect class exactly:
the page grades a correct answer wrong, and it does so by instruction, not by accident.

Note the contrast with how carefully the presets were built: every one of them composes from `state.active`
rather than the draft (`FormulationLabs.jsx:479-497`), precisely to avoid the sibling-lesson defect where
presets were built from the draft and graded against the applied state. The same care was not extended to the
sentence that states the baseline in words.

**Suggested shape of a fix** (not applied): render the caption's numbers from `state.active` through
`selectionMetrics`, so it reads "the state you last applied" whenever one exists, and keep the section-5
wording only on the first round.

---

# Part C — should-fix

## S1. Figure 5's third metric is labelled as a count and drawn as a fraction

`src/learn/data/formulation-models.js:936`:

```js
precisionAt50: { label: 'positives among the selected 50', domain: [0, 1], better: 'higher', digits: 2 },
```

The button, the axis and the figure caption all read **"positives among the selected 50"**, the axis runs
0 → 1, and the three bars are drawn and annotated **0.12 / 0.4 / 0.52** — the *fraction*, not the count. The
caption composes to "positives among the selected 50, higher is better, drawn on 0 to 1", which is
self-evidently wrong for a count with a maximum of 50.

The same figure contradicts itself twice over: its table column "Positives in the top 50" shows **6 / 20 / 26**,
and the strips underneath read "baseline: 6 of 50", "candidate: 20 of 50", "+ duration: 26 of 50". A reader
switching to this metric sees the number they were just shown as 6 redrawn as 0.12 under an identical name.
Screenshot: `scratch/formulation-independent/shots/zoom-f5-precision.png`.

This is the "naming one quantity while showing another" variant, in a figure rather than a grader. The
model's key is `precisionAt50` and the value is a precision; only the label is wrong. "share of the selected
50 that subscribed" would fix it without touching a number.

## S2. The admissible-age band silently understates the window, and the model's own warning flag is read by nobody

`formulation-models.js:276-279` computes

```js
/* Reported, not hidden: an age limit wider than the drawn ruler reaches
   back past its left edge, and the band then understates the window. */
startsBeforeTheRuler: windowStart < start,
```

`startsBeforeTheRuler` appears **once in the entire repository** — at its own definition. No component reads
it, no verifier asserts it, and the browser check never looks for it. The same is true of `eventClipped` and
`availableClipped`.

The consequence is reachable and visible. With cutoff 2 and maximum age 12 (both inside the controls' 0–12
bounds), the figure caption reads *"admissible event window **−10 to 2**"* while the drawn band spans only
**0 → 2**, because `x` and `width` are computed from `clamp`ed values. Nothing on the figure says the band has
been truncated. Screenshot: `scratch/formulation-independent/shots/timeline-band-past-ruler.png`.

The model comment says the clamp is "reported, not hidden". It is computed, not reported. In this fixture no
event lies below 0 so the answer is unaffected, but the figure's extent *is* its claim — that is the standard
this lesson sets for itself everywhere else — and here the extent is wrong relative to the caption beside it.

## S3. A third inert guard: the identical-branch scan cannot see the one real identical-branch conditional in the lesson

`src/learn/components/lesson-labs/FormulationShared.jsx:62`:

```js
export const asInput = value => (Number.isInteger(value) ? sign(String(value)) : sign(String(value)));
```

Both branches are byte-identical. The integer test does nothing; `asInput` is `sign(String(value))`.

`scripts/verify-formulation-sources.py:256-266` exists specifically to catch this, over a `COMPONENTS` list
that **includes this file**:

```python
for match in re.finditer(r"\?\s*('[^']*'|\"[^\"]*\")\s*:\s*('[^']*'|\"[^\"]*\")", body):
    if match.group(1) == match.group(2):
```

The regex requires both branches to be quoted **string literals**, so an expression-valued pair is invisible
to it. `falsify-formulation.mjs:265-273` proves the guard fires — by injecting
`(flag ? 'same' : 'same')`, a string-literal instance — while the one genuine instance in the tree sits
unflagged in the same scan's input.

I wrote a balanced-paren scanner over all five lesson modules
(`scratch/formulation-independent/`): **exactly one** identical-branch conditional exists, and this is it.
The brief asked for a third inert guard beyond the two the builder found; this is it, and it is the more
instructive kind, because the falsification case gives it a green tick it has not earned on real code.

Nothing renders wrong as a result — both branches produce the same string — so this is should-fix, not
blocking. But the guard's domain is narrower than its message and its falsification case both imply.

## S4. A named preset silently does nothing when the entity selector is on sensor B

`FormulationLabs.jsx:42-43`:

```js
['Let the delayed event arrive at time 4 instead',
  state => withArrival(state, { event: 4, version: 1, available: 4 })],
```

`withArrival` (`formulation-models.js:1119-1127`) filters on `record.entity === fixture.entity`, i.e. the
*queried* entity, not the record's own. With the entity selector on sensor B, the preset matches
`B·event 4·v1` — whose arrival is already 4 — and leaves `A·event 4·v1` at 8.

Driven live: after switching to sensor B and committing a prediction, clicking the button **retires the
verdict, produces no pending banner (draft == active), and changes nothing**; the history table still shows
`A · event 4 · v1  4  8  1  20`. The control that carries the destination note's whole point is a no-op in a
state the page's own selector reaches in one click. Unlike investigation 2, this investigation passes no
`requireChange`, so nothing catches it.

The 50,700-case grid does cover `sensor_B` for the *rule*, but the preset is not part of the model and is not
swept.

## S5. §8's quantile routing is true but sends the reader out of a curriculum that owns the material — and the record's search claim does not hold

**The adjudication itself is correct, and I confirm it.** `lesson.md` §8's sentence, "The earlier
quantile-regression lesson owns fitting that conditional quantity," is false: I walked the catalogue
independently — **1,499 titled entries** in `docs/curriculum/curriculum-inventory.json`, each with a distinct
slug — and exactly one title matches `/quantile/i`: *Streaming Quantiles, KLL, t-Digest & Reservoir Sampling*,
in system-design, and unpublished (`lesson-manifest.json` maps it to `None`). Recording the disagreement
rather than editing the frozen manuscript was the right call, and the replacement wording is literally
accurate.

**But the replacement is materially incomplete, and the record that justifies it is wrong.** design.md:159-163
says:

> A search of all 1,460 catalogue entries finds three unrelated uses of the word … None teaches quantile
> regression.

There are considerably more than three uses, and two of them are not unrelated:

- **`calibration-conformal-prediction`** — position 35, immediately preceding this lesson, **published**
  (`src/learn/data/topics/calibration-conformal-prediction.jsx:555-565`) — carries a section headed
  *"Conformalised quantile regression"* that defines **pinball loss** by name. The page sends the reader to
  scikit-learn "for the pinball-loss implementation" when the previous lesson defines the loss.
- **`decision-theory-risk-cost-sensitive-decisions`** — published, and **earlier in the same plan file**
  (`src/learn/data/curriculum/cross-domain-expansion.js`, entry 11 against this lesson's 20) — contains the
  identical derivation §8 gives, under the heading *"Absolute error requests a median; asymmetric error
  requests a quantile"*, with `τ = c_u/(c_u+c_o)` displayed, a worked underage/overage newsvendor, the
  discrete-quantile subtlety, and a practice on non-unique optimal orders
  (`decision-theory-risk-cost-sensitive-decisions.jsx:94-102, 226`).

So the accurate statement is not "this curriculum has no dedicated quantile-regression lesson yet" full stop,
but that the fitting technique has no lesson of its own **while the decision-theoretic content §8 is teaching
already has one, earlier**. As written, the page's only route is outward, and the record's justification rests
on a search that missed the two nearest neighbours. Both the page sentence and design.md:156-167 should be
revised; the frozen manuscript should stay untouched, as the builder decided.

## S6. The model verifier's header describes a second implementation it does not contain

`scripts/verify-formulation-models.mjs:11-15`:

> A SECOND ROUTE MEANS A DIFFERENT DERIVATION. Average precision, log loss, the confusion cells and the
> ranking **are implemented here from their definitions** … a different language and a different
> implementation, **not a second call**.

They are not implemented there. `averagePrecision`, `logLoss`, `confusionAtThreshold` and `rankByScore` are
all **imported from the module under test** at L42-43 and called at L514, L518, L521 and L531. The inline
comment at L510-512 repeats the claim verbatim.

The comparisons are still sound, because the other side of each is the packet's scikit-learn output — the
"different language and different implementation" is scikit-learn's, not this file's. And the property does
hold: my own from-definition implementations reproduce every value (A1). But a reader auditing whether a
second in-repo route exists will conclude it does, and it does not. The genuinely local second routes in that
file are three: `asKnownBySortedKey` (L108), `rankBySelection` (L126, and applied only to the first 120 of 824
ranked rows at L534) and `constantLogLoss` (L144).

The same header's claim that "nothing is compared with itself" is contradicted by the geometry assertions — see O1.

## S7. The browser verifier carries inert guards, an unpaired absence, and floors below its own totals

I verified each of these against the source and the tree.

- **`.formulation-table-scroll caption` (L751) cannot match.** The only producer of `.formulation-table-scroll`
  is `Table` in `FormulationShared.jsx:153-171`, which deliberately renders its caption as a sibling `<p>`.
  There is **no `<caption` tag anywhere in the four lesson components** — the only occurrence of the word is
  the docstring at `FormulationShared.jsx:150` explaining why one is not used. The record at L755-757
  nonetheless states the property as verified: *"and every caption sits outside its scroll box."* Nothing
  establishes that a caption sits anywhere; the scope is proven non-empty by the adjacent `count() >= 8`, but
  the subject of the assertion cannot exist. Unlike the KaTeX pin at L808, this one is not documented as a
  deliberate zero.
- **`/outlines/` (L833) and foreign `learn-assets/` (L834) are unreachable.** Outlines are loaded through
  `import.meta.glob`, so in a production build no emitted filename contains `outlines` — I checked
  `dist-form/assets`: zero matches. And the lesson never `fetch`es its assets; it links to them, so no
  `learn-assets` URL reaches `page.on('request')` at all. The evidence file's own `requestedScripts` confirms
  it: nine hashed `assets/*.js`, nothing else. L833 additionally carries no message string.
- **`.formulation-reveal` absence has no presence partner.** `count === 0` at L420 and L595; a grep of the
  whole file finds **no** assertion that `.formulation-reveal` is ever present. Every other absence in the
  file is paired — the four pre-commit phrases at L446 with L465, the four at L485 with L512, the suppressed
  drawing at L621 with L628, the verdict at L418 with L458/507/539/553/574. The reveal is the one that is not,
  which is the exact pattern design.md:348-353 says was added to fix. The paired-pattern claim holds for the
  phrases it names; it does not hold universally.
- **Floors sit below the totals.** `records.length >= 13` against 14 pushed (L926) and
  `screenshots.length >= 30` against 37 (L927). A whole case can be deleted without tripping either. The
  model suite is the same: `total >= 118` against 130 and `>= 52` groups against 56 (L1254-1255), which the
  harness's own limitations list discloses honestly.

**Where I disagreed with the fan-out read:** the `form-grid` exclusion at L91 and the `getTotalLength` guard
at L92 are dead filters but harmless and not worth a finding; and the KaTeX SVG guard and the curve sampler
are both genuinely repaired — see D1.

**On the source-hygiene count.** I instrumented a copy of `verify-formulation-sources.py` and ran it:
**49 real `check()` calls and 58 `scanned()` counter bumps**, summing to the advertised 107. `scanned()` is
honestly documented ("it exists to make the count true"), and the scans behind it do report by appending to
`problems`, so they can fail. But its docstring also says *"Each scan therefore ends by evaluating whether it
found anything"* — and its return value `len(problems) == before` is **discarded at all nine call sites**. The
evaluation never gates anything, and a scan whose pattern matches nothing still increments. 54% of the
advertised checks count scans performed, not assertions that could fail.

## S8. The falsification harness's evidence snapshot cancels its own provisional record

`falsify-formulation.mjs:457-469` snapshots every `formulation-*` evidence file — **43 of them**, and the
committed report's `evidenceFilesRestored` confirms `docs/teaching/evidence/formulation-falsification.json` is
one. L540-549 then writes a provisional `status: 'running'` record before the first mutation, with a comment
saying it exists so that "a case throwing … would otherwise leave the previous report on disk claiming every
guard fired."

But the `finally` at L655-662 writes the **snapshot** back, which is the previous report, and the throw then
prevents the real write at L697. So the exact failure the provisional record was written to prevent —
a restore mismatch at L649, or a failed build at L613 — restores the previous green `guardsThatFired: 35`
record and exits non-zero. The two mechanisms cancel. Excluding the harness's own report from
`evidenceSnapshot` would resolve it.

Everything else about the harness is sound and I confirm it: a lock file plus per-file `.formulation-falsify-orig`
sidecars, an explicit **refusal to start** on either (L517, L525) registered *before* the exception handler so
a refusal cannot clear the lock it refused on, `--recover` with a byte comparison, restore from a raw `Buffer`
verified by SHA-256 against the pre-mutation digest (L638-651), and an evidence scope built by
`readdirSync(...).filter(name => name.startsWith('formulation-'))` with **no glob and no delete**. I counted
the screenshots directory: 855 files, 37 of them `formulation-*`. A sibling lesson's captures are unreachable,
as the comment at L455-456 claims. Two residual weaknesses: the lock is check-then-write rather than
`openSync(path, 'wx')`, and the browser verifier is the only one invoked without `--no-evidence` (L45), so an
evidence file it *creates* that did not exist at snapshot time is not removed.

The ordering rule the record states — run offline-only before `--browser`, never after — is correct and the
reason is visible at L696-697: a full-object `writeFileSync` with no read-modify-write.

## S9. Nothing in this lesson's verifier set parses the JSX it owns, or builds

`scripts/verify-formulation-sources.py` imports `hashlib, json, re, sys, unicodedata, datetime, pathlib` and
nothing else: no parser, no `esbuild`, no `babel`, no `subprocess`. Everything it does with a `.jsx` file is a
byte scan for control characters, an escape-sequence regex, and comment-stripped regex matching. A stray
brace, an unclosed tag, an unescaped apostrophe inside a single-quoted string — all pass. The file says so
itself at L412-413.

`verify-formulation-browser.cjs` requires a production build but **does not run one**; it reads
`${distDir}/.vite/manifest.json` (L265). A stale `dist-form` from before a breakage would carry it.

This is not hypothetical here: design.md:283-292 records that exactly this happened — a one-character JSX
error in the position-39 sibling `EndToEndLabs.jsx` broke `npx vite build` repo-wide and was invisible to
every offline verifier, leaving the falsification record understated until the sibling's fix landed. The
record narrates the episode but does not name the gap as a gap. Adding a parse step over the four owned `.jsx`
files — even `esbuild --loader=jsx` on each, which costs milliseconds — would close it for this lesson's own
files. I confirmed the tree builds today: `npx vite build --outDir dist-form` completes in 58 s with no error.

---

# Part D — what I checked and found genuinely sound

## D1. The nine visual defects are gone, and I confirm each in an image

I drove the page myself and opened the pictures rather than the DOM.

1. **Cutoff label outside its viewBox** — gone. At the default cutoff the label sits well inside
   (`zoom-timeline.png`). The `LABEL_ASCENT` floor in `timelineGeometry` (models.js:211-214) makes the
   vertical case unconstructible. (The *horizontal* case survives marginally; see O2.)
2. **Arrival marker running through its own lane label** — gone. The drawn identity is the seven-character
   compact form, the gutter is 58 units, and the constructor refuses a label that does not fit. In the image
   every label ends well before the ruler.
3. **Arrival diamond covering the event dot** — genuinely gone, and this was the one to check hardest.
   `formulation-labs.css:111` sets `.form-arrival { fill: none }`, and I confirmed the **computed** fill is
   `none` and the stroke is not, on every lane, via `getComputedStyle`. More to the point I looked:
   `zoom-timeline.png` shows lane 1 (`A·e1·v1`, event 1 = arrival 1) as a gold dot inside a green outline
   diamond, and lane 4 (`B·e4·v1`, event 4 = arrival 4) as a grey dot inside a grey diamond. Both marks
   visible in both coincident cases. After the arrive-earlier preset a third lane coincides and behaves the
   same.
4. **"An exact answer is required. Optional: leave it blank."** — gone; one sentence per case
   (`FormulationShared.jsx:294-301`), and I read both variants on screen.
5. **Three JSX expressions losing their following space** — all three repaired in the *rendered* text, which
   is where it matters: the page emits "in pdays", "is not deterministic" and "0.25 and 0.75"
   (`scratch/formulation-independent/page-text.txt`).
6. **Timeline and metric bars pinned at 300 px** — now `maxWidth` 480 and 470, and the timeline is legible at
   1366 px.
7. **Strips centring away from their labels** — the two-column grid lines all three up; 6 / 20 / 26 reads at a
   glance.
8. **Figure 1 claiming an annotation it never drew** — the caption and `<desc>` now describe the channel. The
   arrowhead points into "available records" in the image.
9. **Legend above a suppressed drawing, and "0 exchanges and a reversal"** — both gated. In the refusal state
   I measured zero `svg.form-svg` in the investigation and no legend.

**The six "DOM right, picture wrong" mechanisms, checked:** CSS beating a presentation attribute — no blanket
`fill` rule exists; every shape class states its own, and outline-only shapes state `fill: none` in CSS.
CSS collapsing geometry — the layout rule is `svg.form-svg`, never a bare descendant `svg`, and I counted
**0 KaTeX SVGs and 0 collapsed KaTeX boxes** in the live page. Paint order — item 3 above. Contrast — the
greyed other-entity lane is `#6f7b76` on `#1b2a24`; legible, and never the sole carrier of meaning (the lane
label names the entity). A `<rect>` over a `<text>` — the selected-record outline is `fill: none`, the
age-window band is painted first. A self-referential containment check — the browser verifier delegates to
`lesson-visual-layout.cjs`, which compares each label's transformed bbox against the `<svg>`'s **rendered
layout box**, an independent frame; that one is sound. The offline analogue is not (O1).

## D2. The prediction contract holds

First paint at 1366 px, measured rather than eyeballed: **zero** `.formulation-verdict` nodes, **zero**
`.formulation-reveal` nodes, **zero** radios checked, both Apply buttons disabled, **zero** six-decimal
strings anywhere inside either investigation, the string "subscribed" absent from investigation 2, and
"hidden until you apply" present 13 times (twelve outcome cells plus the investigation note). No page errors.

The contract holds under the harder tests too: editing any graded input retires the verdict *and* re-hides
every outcome (because the table's outcome column is gated on `shown?.answer`); the reveal recomputes its
figure from `shown.inputs`, not from the draft; `check` computes the answer from the draft at the moment of
commitment and only then advances `active`; and the presets compose from `active` rather than the draft. The
six-decimal convention is a real pin — `fixed` is used for graded values and nothing else, `probabilityText`
at four decimals for the 824 scores, `asInput` for inputs — and I confirmed `0.666667` is absent from the page
before commitment.

## D3. The trust root is a real mechanism, and the split is honestly described

`verify-formulation-data.py` reports 100% of 10,099 scalar leaves asserted, and the three-way split
3,506 + 2,474 + 4,119 sums exactly to 10,099. The mechanism is not a tautology: the comparison walks the
packet tree recording the path of every leaf it actually compares and fails unless that set is complete, so
the denominator is measured rather than declared.

The description is honest in the way that matters most — it names what is *not* independently derived. The
2,474 fitted probabilities are validated by refitting the same estimator, which the record states plainly and
which is the right call: an independent reimplementation of regularised logistic regression would be testing a
different object. The 4,119 split indices are regenerated from seeds and separately checked for disjointness,
exhaustiveness, size and stratification, with the record stating that "their exact order is a property of
scikit-learn's random state and is not independently reproduced. That is the whole of what is not covered."
I reproduced all four split properties myself (A1) and agree with that characterisation. The 3,506
independently re-derived leaves I re-derived a third time, independently of both the packet and the verifier,
and they agree.

## D4. The records against the tree

Almost everything in Phase A and Phase C checks out, which is unusual enough to be worth saying.

- All eight Phase C source hashes match the tree byte for byte.
- **Nine visual defects**: all nine confirmed repaired, in images (D1).
- **Two vacuous guards**: both accounts accurate. The `.katex svg` filter is now
  `assert.equal(katexSvgs.length, 0, …)` — a cardinality pin, not a filter over an empty set — and I confirmed
  the page emits exactly 0, so adding a radical will fail it. The curve sampler's guessed floor of 10 is gone,
  replaced by `assert.deepEqual(sampled, ['form-arrival' ×4, 'form-arrow is-feedback'])`; I counted the page's
  drawn paths and it does draw exactly those five.
- **Two guards added**: the rendered-prose glue scan over 471 elements is real and floored at 150. The
  paired presence/absence pattern holds for every phrase set it names — but not universally (S7).
- **Two rebuilt breakages**: both accounts are internally consistent with the harness source and with the
  model floor added in defect 1.
- **The payload table**: I measured it myself with `gzip -9` on my own build. Raw bytes for PAC (175,993),
  Calibration (254,014) and Rademacher (209,782) match the record exactly; this lesson's 170,714 against
  170,650 and the gzip figures differ by 100–300 bytes because sibling chunk hashes changed in the tree since.
  The ordering claim is what matters and it holds: Eval 43.9 KB < PAC 57.0 < **this lesson 63.4** <
  Rademacher 70.8 < Calibration 83.9. "Between two of its neighbours rather than above them" is accurate.
- **Screenshot accounting**: 37 recorded in the evidence, **37 `formulation-*.png` on disk**, 14 cases
  recorded. Verified by count.
- **"Every verifier writes a provisional record stamped `passed: false`"**: verified in all five.
- **Crash safety "proved by an actual kill"**: I did not repeat the kill, but the mechanism the account
  describes is present exactly as described (S8), including the ordering that prevents a refusal from clearing
  its own lock.

The two record claims that do **not** hold are S5 (the quantile search) and S6 (the model verifier's header),
plus the smaller items at O7 and O8.

---

# Part E — observations

**O1. The offline timeline containment assertions cannot fail for any input.**
`verify-formulation-models.mjs:396-403, 407-408, 443-444` assert each marker and tick lies within
`[geometry.labelWidth, geometry.width − geometry.inset]`. But every such coordinate is `scale(clamp(v))`,
where `clamp` maps into `TIMELINE_DOMAIN` and `scale`'s **range is exactly `[labelWidth, width − inset]`**
(models.js:215). The assertion compares positions against the endpoints of the scale that produced them:
mechanism 6 from the brief, roughly 1,800 times across the geometry sweep. Similarly
`ageWindow.width >= 0` (L436) against a `Math.max(·, 0)`; `ageWindow.x + ageWindow.width === cutoff.x` (L437)
as an algebraic identity; `ticks.length >= 5` (L441) against a constant 7; `bar.labelX < padding.left` and
`bar.valueX > bar.endX` (L1037-1038) against `padding.left − 5` and `scale(value) + 5`.

I am recording this as an observation rather than a should-fix, in fairness to the builder, for three reasons.
The file **identifies the hazard by name** at L452-455 and floors the inset and gutter separately, with
`refuses()` cases that genuinely exercise them. These assertions still catch a **source mutation** — which is
what the falsification harness tests, and case 3 fires on exactly one of them. And the real containment check,
against the rendered SVG box, lives in the browser verifier and is not self-referential. What is inaccurate is
the header's claim at L5-7 that "nothing is compared with itself"; several dozen assertions do.

**O2. The cutoff label crosses the right edge of its own viewBox at cutoff 12.** With `cutoff = 12`,
`cutoff.x = width − inset = 286` and the six-character label is centred there, needing ~15.3 units of
half-width against a 14-unit inset. Measured in the live page: the label's right edge is at **481 px in a
480 px SVG**, so the final glyph is shaved by browsers' default `overflow: hidden` on the outermost `<svg>`.
The amount is about one device pixel, which is why I am not raising it higher — but it is the same class as
repaired defect 1, it is reachable from the cutoff control, and the machinery narrowly misses it twice: lane
labels have a width floor and the cutoff label has none, and `lesson-visual-layout.cjs` allows a ±2 px
tolerance. Screenshot: `scratch/formulation-independent/shots/timeline-cutoff-12.png`.

**O3. Figure 1's feedback channel uses the wrong-coloured arrowhead.** The polyline is
`.form-arrow.is-feedback`, stroked `#b58ec0` (purple, dashed), but it reuses `marker#form-arrowhead`, whose
path is filled `#8eb9a5` (green) and is not scoped per-arrow. The result, visible in
`zoom-flow.png`, is a green triangle terminating a purple dashed line.

**O4. `metricAxes[*].digits` is dead configuration.** `digits: 6 / 6 / 2` at models.js:934-936 is never read;
`metricBarGeometry` ignores it and the figure prints `round(bar.value, 4)`. The only `.digits` consumed
anywhere is `numeric.digits` in `Prediction`, a different object.

**O5. The 8→4 property is pinned at two points, not swept.** See A2. The 50,700-case grid proves the two
routes agree; it never asserts that moving an arrival with no event time changed moves the selection. Given
how central that claim is to the destination note, a small sweep — "for every cutoff and age, moving
`A·e4·v1`'s arrival from 8 to 4 either leaves the selection alone or moves it to a later event" — would cost
nothing and would be the property rather than the instance.

**O6. `eventClipped` and `availableClipped` are unreachable.** The controls bound every time to 0…12 and
`TIMELINE_DOMAIN` is `[0, 12]`, so neither flag can ever be true. Unlike `startsBeforeTheRuler` (S2) this
costs nothing, but it means two-thirds of the clipping apparatus is dead.

**O7. The Phase A record's catalogue count and search result are both off.** It says "all 1,460 catalogue
entries"; I count 1,499 titled entries with distinct slugs. It says the search finds "three unrelated uses of
the word"; the word appears in at least seven entries' titles, outcomes or subtopics, including the two that
matter (S5). Different counting bases could explain 1,460 versus 1,499; they cannot explain three versus
seven.

**O8. Phase C slightly overstates one contract claim.** "Investigation 2 renders **no outcome for any of its
824 cases**" is true but the window renders only 12 rows at a time, so at most 12 cells were ever candidates
for the check. The underlying property — no outcome revealed before commitment — I verified and it holds.

**O9. The recorded-exchange preset bundles two changes, and it is the only one that does.**
"Load the recorded exchange at capacity 25" sets capacity **and** applies the swap, so from the default state
it is graded "It rises" (11/25 = 0.44 against 20/50 = 0.40) even though the exchange alone lowers precision
from 12/25. The button names the capacity and the verdict discloses both numbers, so this is fair rather than
wrong — but it is the preset whose name most invites single-cause reasoning, and the only one that is not a
one-variable change. Reproduced live; see `shots/cap-recorded-exchange.png`.

**O10. Practice 1 hardcodes the age limit twice.** The prose says "event times must be at least
`{practiceTimelineFixture.cutoff - 1}`" while the fixture it describes is `{...practiceTimelineFixture,
maximumAge: 1}`. The `1` is written independently in two places; every other number in that solution is
derived. Low risk, but it is the one place on the page where a changed fixture would leave the prose behind.

**O11. Narrow-width layout is clean.** At 390 px and 320 px `document.documentElement.scrollWidth` equals the
client width — no horizontal page scroll at either. The elements that exceed the viewport are the hidden
`.katex-mathml` MathML subtrees (390 px) and the shared `LessonTable` tables, which sit inside
`div.lesson-table-wrap`, an `overflow-x` scroller (320 px). Both are permitted. The lesson's own `Table`
component blockifies below 560 px and never overflows. Figures 1–7 and both investigations read correctly at
all three widths.

---

# Verdict

This is a strong implementation of a lesson whose subject is unusually easy to commit while teaching. **I found
no numerical disagreement of any kind** — every measurement, baseline, score, confusion cell, ranking,
selection, split property, figure coordinate and practice answer reproduced from an independent derivation,
several in exact rational arithmetic. The availability contract is the strongest part of the work: the rule
drawn is the rule applied, a cutoff is part of every signature, the missing-calibration case is a result
rather than a substitution, and the 50,700-case sweep checks the four admissibility predicates on every
selected record rather than at a fixture. The page does not commit the error it teaches: nothing that depends
on a future value is visible before a prediction is recorded, and I verified that by measurement rather than
by looking for a banner.

**Two blocking defects, both in investigation 2, both in the same family the effort has been chasing.** B1
prints a learner's correct answer and the page's own answer as the identical string and declares them
different, at most of the capacities the control reaches. B2 tells the learner, in words, to predict against a
baseline the grader does not use, from the second round onward, and then marks their correct reasoning wrong
while simultaneously confirming their number. Neither is subtle once triggered; both were missed because the
recorded captures only ever exercise capacities 25, 50 and 100 and only ever one round.

The should-fix list is mostly about claims rather than code: a figure axis naming a count and drawing a
fraction, a clamp the model flags and nothing reports, a hygiene guard whose domain excludes the one real
instance in its own input, a verifier header describing a second implementation it does not contain, and a
quantile-routing sentence that is true but sends the reader past two published lessons that own the material.
The harness is the best I have reviewed in this series — lock plus sidecar plus refusal-to-start, hash-verified
restore, no glob, no delete, and an explicit scope that puts a sibling's 818 other captures out of reach by
construction — with one self-cancelling interaction between its snapshot and its provisional record.

---

*Throwaway reviewer scripts and captures: `scratch/formulation-independent/` (`refit.py`, `drive.cjs`,
`overflow.cjs`, `zoom.cjs`, `nulls.cjs`, `sources-probe.py`, `shots/`, `BASELINE-HASHES.txt`,
`backup/`). `dist-form/` deleted and the preview on 4195 stopped.*
