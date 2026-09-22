# End-to-End Supervised Learning & Error Analysis — independent phase-two review

Reviewed 20 September 2026 against the working tree plus the uncommitted end-to-end implementation. The
reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/`,
`docs/teaching/drafts/` or `docs/teaching/evidence/` was edited; this review document is the only write
outside `scratch/endtoend-independent/`. No state-changing git command was run. The build directory
`dist-e2e/` was deleted and the preview on 4197 stopped at the end.

Four housekeeping notes, stated up front because all four bear on trust in the record:

- I took a SHA-256 baseline of all 31 lesson artifacts before touching anything
  (`scratch/endtoend-independent/BASELINE-HASHES.txt`) and re-checked it at the end.
- **I did not run `falsify-endtoend.mjs` at all.** The coordinator had already established that the tree was
  clean after an interruption, with no lock and no `.orig` sidecar anywhere. Re-running a harness that mutates
  eleven source files to re-confirm a count I could audit statically was not worth the risk on a machine that
  has killed processes for memory and spend limits repeatedly. Its 32 offline and 5 browser cases are counted
  from source, and its safety properties are read from source; that is stated rather than implied. See S5.
- Re-running `verify-endtoend-data.py` and `verify-endtoend-examples.py` **overwrote their evidence JSON**
  (only `verify-endtoend-models.mjs` offers `--no-evidence`). I diffed both — `endtoend-data.json` differed
  only in `checkedAt`; `endtoend-native.json` differed in `checkedAt`, `oracles` and `isolatedRun`, all
  because I ran without `--isolated` — and then **restored both from my pre-run copies**. All 27 screenshots
  were byte-identical to my pre-run copies throughout.
- `docs/teaching/evidence/endtoend-models.json` differs from the baseline I took at the start of this review.
  It was not changed by me (I used `--no-evidence`); it was changed by a verifier run between my baseline and
  my evidence backup. I have left it as found rather than reverting another party's legitimate run, and flag
  it here so the record is not mistaken for mine.

**A concurrency hazard worth recording:** during this review `src/learn/components/lesson-labs/TimeSeriesLabs.jsx`
— an untracked file belonging to a *sibling* lesson — was being edited by another process, and at one point
carried a JSX syntax error that failed `npx vite build` outright. My first build failed for that reason, not
for anything in this lesson. The lesson's own 31 files showed **zero drift** across the whole review. This is
not a defect in this lesson, but it means a clean production build of the repo is not currently reproducible
without coordination.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/endtoend-independent/`, importing neither `endtoend-models.js` nor `endtoend-data.js` nor `endtoend-examples.js` nor any `verify-endtoend-*` script nor the packet's `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, scikit-learn 1.9.1, pandas 3.0.1, mpmath 1.3.0 — the author snapshot exactly). **54 checks** in `recompute.py` and **522 field comparisons** in `datadiff.py`, with accuracy, balanced accuracy, confusion matrices and every practice quantity in exact `Fraction` arithmetic and log loss at 50-digit `mpmath`. An independent CART tree walk (`treewalk.py`, `tw2.py`) over all 100 fitted trees × 178 specimens to settle the float32 question. An independent execution of the served `wine_study.py` with a byte comparison of its stdout (`example_check.py`). Five reviewer-written Playwright drives (`drive1`–`drive5.cjs`) sharing no code with `verify-endtoend-browser.cjs`, on Edge at 1366, 390 and 320 px, including a live sweep of all 48 selection settings through the real controls. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `calculated-inputs.json`, `author-calculations.py`, `wine.csv`); the lesson body, model layer, both generated modules, all three lab/figure/shared components and the CSS; all five verifiers and the harness; the blueprint; the served assets; and the builder's 27 captures, opened as images. |
| **Reused (declared)** | scikit-learn's estimators and `train_test_split` at the declared settings and seeds, because the declared protocol *is* those calls. Everything around them — split construction, every metric, slice membership, paired counts, the Wilson interval, the acceptance ledger, the tree walk — is written here. One subagent for a fan-out read of verifier internals. **Every finding a subagent surfaced I re-derived myself before reporting it**: I read each cited line, ran the arithmetic myself, and where a subagent's claim was wrong (it reported no `SIGINT`/`SIGTERM` handlers where a registration loop does install them) I corrected it rather than repeating it. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx` | `eaac7377c4eb02ffd3488a7109a143ea06b2358a6aa43f03e7fa6b40d5d61dda` |
| `src/learn/data/endtoend-models.js` | `07b338b8c99100cbd5041c862897ba14b04ad6c9ca3602c4d9f5424e95863b6c` |
| `src/learn/data/endtoend-data.js` | `92a4d405e75ff085554a3268fac1b97b0854ef82c1addae6c27f2f2ad68b75ff` |
| `src/learn/data/endtoend-examples.js` | `f17d70ef6a313ff0d6922fa06c224cace3681170d7b69652103813d73e5c1952` |
| `src/learn/components/lesson-labs/EndToEndShared.jsx` | `d42c82cdb950759a99951b32a496511cc0deb747cc91eb54ff58abd0587ed220` |
| `src/learn/components/lesson-labs/EndToEndLabs.jsx` | `2d6f10568cb9657c2471f121dbf102b222d620b4442cd71dbda60fc863e15e08` |
| `src/learn/components/lesson-labs/EndToEndFigures.jsx` | `8feeadfa2fdaad9bff2c15cb88540e69490c685c3df9a770e892aaed8dfa9af1` |
| `src/learn/components/lesson-labs/endtoend-labs.css` | `d9eb9abd61c9fd59433d6a9ac6f01a52952d599693fa30d4b5f201bbb523a8f3` |
| `scripts/verify-endtoend-models.mjs` | `0e93255290d3dd98ac25a17290865a3269b9948135e58a2105a53a53fb22b002` |
| `scripts/verify-endtoend-data.py` | `5fabe486a95d85e9c4a2370dac5f429125d6f13954c16a9e145cb748a8527cec` |
| `scripts/verify-endtoend-examples.py` | `175d7fb11ddd81df0719efc5d4bd705502669dc61268dad0b345873d9cd233a8` |
| `scripts/verify-endtoend-sources.py` | `2e80ce490af6080cea193f3a0a46263c97d9ea0ecc240ed1e5a1ccc5632ca45b` |
| `scripts/verify-endtoend-browser.cjs` | `62e3849890a92036d091aa9a771837c20a7120f11ad09a280c33414b640769a2` |
| `scripts/falsify-endtoend.mjs` | `324eec74d58b78e154c90334fe48a40dfe2ce65b8e0d712c88684b7d98482a8e` |
| `docs/teaching/drafts/.../lesson.md` | `0b25eaad5fe45232c97f756222805e9222bb91659aeec08835c7a88e5e5338e0` |
| `docs/teaching/drafts/.../design.md` | `d75a81635deb2b44d1498264724e67f63f4f6ccff411863faea693d88a13e4a9` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `24fb758fe088b6afd4baec315429bf9554cc3936e6bff474bf4d188e2ed90688` |
| `docs/teaching/drafts/.../wine.csv` (= served copy) | `34ced17cfa0a96bf5ae5c25565da075c612145b20368fc2455e5e8f966e818be` |

---

# Part A — correctness

## A1. The whole wine study, refit from the served CSV — **no disagreement**

`scratch/endtoend-independent/recompute.py`. The CSV was read as text and parsed to exact `Fraction`s; the
splits were rebuilt from the two declared `train_test_split` calls; every metric was written here from its
definition rather than taken from `sklearn.metrics`.

**54 of 54 checks pass.** Specifically, and with the exact rational value beside each printed one:

- Split sizes **106 / 36 / 36**, an exact partition of 1…178, pairwise disjoint. Class counts
  train `[35, 43, 28]`, validation `[12, 14, 10]`, test `[12, 14, 10]`.
- Validation balanced accuracy: majority `1/3`, linear_two `111/140 = 0.7928571428571428`,
  forest_two `517/630`, linear_three `8/9`. Accuracy: `7/18`, `29/36`, `5/6`, `8/9`. Log loss at 50 digits:
  `1.0900217757972094413`, `0.52280464175493102439`, `0.51129270814958634252`, `0.21800808114277152774`.
  All twelve round to the six-decimal values §4 prints.
- The §5 confusion matrix for `linear_two` is exactly `[[9,3,0],[0,13,1],[2,1,7]]`, and its balanced accuracy
  **is** `(9/12 + 13/14 + 7/10)/3` as an exact rational identity, not merely to six places.
- Paired change: `forest_two` 1 repaired / 0 broken (specimen 39); `linear_three` 4 repaired / 1 broken,
  and the identifiers are exactly **122, 143, 146, 162** repaired and **28** broken. The lesson's "net gain
  of three, from 29/36 to 32/36" follows exactly.
- Slices at cutoff 4: `linear_two` 17/2 and 19/5; `linear_three` 17/3 and 19/1. Practice 4's class-1 slice is
  14 specimens with 1 error and 0 errors respectively.
- Selection: the eligible scores are `111/140`, `517/630`, `8/9` — I checked as **exact rationals** that the
  tie set has size one, so `linear_three` wins strictly and the tie rule is not silently doing work here.
- Held-out: accuracy exactly `35/36`, balanced accuracy exactly `(1 + 1 + 9/10)/3 = 29/30`, log loss
  `0.12870825516266741602`, confusion `[[12,0,0],[0,14,0],[0,1,9]]`.
- Wilson at k=35, n=36, z=1.96 with an exact rational centre `46151/49802` and a 50-digit radius:
  `[0.858299662649691998…, 0.995079719704430325…]`, printing as **[0.858, 0.995]**.
- Every practice number: accuracy `3/4`, balanced accuracy `32/45`, recalls `4/5, 5/6, 1/2`; practice 2's
  11/95 and 7/95; practice 5's two costs both exactly 100; §8's 8/10 coverage and 1/8 conditional error;
  §3's `-log(0.8) = 0.223` and `-log(0.2) = 1.609`.

**No disagreement with any number the prose states, at any point in the lesson.**

## A2. `endtoend-data.js`, field by field — **no disagreement in 522 of 522 comparisons**

`scratch/endtoend-independent/datadiff.py` parses the generated module as text and compares every field
against the independent refit: all four candidates' scores, roles and confusion matrices; all 36
`validationRows` with their measurements, per-candidate predictions and `probabilityOfActual` **at exact
equality, tolerance 0.0**; the paired identifier lists in split order; `colourSliceReference`; the whole
`heldOut` block including its 36 test identifiers, actuals and predictions; the deferral fixture; the
defaults; the practice block.

One comparison differed, and it is not a defect. `forest_two.validationBalancedAccuracy` is stored as
`0.8206349206349206` where my left-to-right sum and `sklearn.metrics.balanced_accuracy_score` both give
`0.8206349206349207`. The exact value is `517/630 = 0.82063492063492063…`, so **the stored double is the
correctly-rounded one and sklearn's is 1 ULP high**; the difference is summation order. Both print
`0.820635`, which is the only form the page and the program ever show. I record it as a curiosity, not a
finding.

I also confirmed the stored Wilson bounds are bit-identical to the doubles produced by evaluating
`wilsonInterval`'s expressions in its own order, so the stored record and the runtime computation agree.

## A3. The float32 trap — **the builder's diagnosis is correct, and the verifier implements it correctly**

`treewalk.py`, `tw2.py`. I wrote my own CART traversal from the prediction rule and compared it against
`tree_.apply` across all 100 trees.

- A naive **float64** walk disagrees with the fitted model at the *leaf* level on exactly **three of the 36
  validation specimens — 94, 98 and 150** — which is precisely the builder's claim. Over all 178 specimens it
  disagrees on thirteen.
- The mechanism is as described: sklearn stores float32-rounded thresholds in a float64 array and casts `X`
  to float32 before comparing. I exhibited the collisions, e.g. specimen 18, tree 16: `x = 13.83`,
  `threshold = 13.829999923706055`, so `x <= t` is **false** in float64 and **true** in float32.
- The exact emulation is `float32(x) <= float64(threshold)` — cast the **input only**. I verified this gives
  **zero** leaf disagreements over all 178 × 100 walks. **Casting the threshold as well is wrong**: only 767
  thresholds are stored and not all of them are exactly float32-representable, and a both-sides cast
  re-introduces disagreements on specimens 8, 22, 84 and 141.
- `scripts/verify-endtoend-data.py:211,218` does exactly the right thing —
  `single = np.asarray(sample, dtype=np.float32)` then `float(single[feature]) <= float(tree.threshold[node])`
  — casting the input and leaving the threshold in double. **This is correct and I confirm it independently.**

**One refinement for the record.** The disagreement is at the *leaf* and hence in the *probabilities* (up to
7.6 × 10⁻² absolute); the **predicted labels agree on all 36 validation specimens either way**. The design
record's phrase "disagreed with the fitted model on three of the thirty-six validation specimens" is accurate
about routing but could be read as a prediction disagreement, which it is not. Worth one clause.

## A4. Specimen 172 and the hundredths grid — **confirmed**

The served CSV carries `9.899999` for specimen 172, exactly as reported; it is the only measurement off the
two-decimal grid, which `verify-endtoend-data.py:451-453` pins by identifier. The cutoff control holds integer
hundredths and divides once, so `colorIntensity < cutoff` compares a stored double against `h/100`, and no
measurement is rounded. I confirm there is no boundary ambiguity at that specimen — and in any case specimen
172 is not in the development set, so it never reaches the slice explorer. The design record's account is
correct.

## A5. The runnable example — **no disagreement**

`example_check.py`. The served `public/learn-assets/end-to-end/wine_study.py` hashes to the recorded
`extractedSha256`, is byte-identical to `examples.code`, and `examples.code` is a byte-exact slice of the
manuscript's single fenced python block. I executed the served program: it exits 0 and prints exactly the
manuscript's recorded block. `developmentOutput + "\n" + heldOutOutput` reproduces that stdout exactly (the
verifier `strip("\n")`s the trailing newline before hashing, which is why `stdoutSha256` is the hash of the
stripped form — this is a convention, not a discrepancy, and the split **is** checked against the real run at
`verify-endtoend-examples.py:156`, so the hash is not self-referential).

No held-out quantity appears anywhere in `developmentOutput`; its last line is `selected linear_three`, which
is development evidence and correctly stays on the development side.

---

# Part B — the gate

## A6. The held-out gate holds. I could not break it — **no disagreement**

This was the main adversarial target. `drive1.cjs`, `drive2.cjs`, `drive5.cjs`, against a production build
previewed on 4197, Edge.

Before either commitment I swept, for fifteen textual forms of every held-out quantity (the three scores at
six and seven decimals, `35/36`, `35 of 36`, `29/30`, the Wilson bounds at three and seven decimals, the
`test …` program line, and three forms of the confusion matrix):

- **the initial HTML payload** — 0 hits (it is a 588-byte SPA shell);
- **every text node** in the rendered document — 0 hits;
- **every attribute of every element**, including `aria-*`, `title`, `alt`, `placeholder`, `data-*` — 0 hits;
- **every form control's `value` property** (which is not an attribute once set) — 0 hits;
- `document.body.innerHTML` in full — 0 hits;
- the **accessibility tree** via `ariaSnapshot()` — 0 hits.

State counts, measured live and matching the builder's claims:

| State | `[data-role]` scores | Roles present | Sealed | Held-out hits |
| --- | ---: | --- | ---: | ---: |
| First paint | **31** | validation 28, training 3 | **3** | **0** |
| After commitment 1 | 33 | + **selection** 1 | 3 | **0** |
| After commitment 2 | 39 (41 with investigation B also committed) | all four | 0 | report open |

The selection criterion that appears after the first commitment is `0.888889` badged **SELECTION** — the same
number the table above it shows badged **VALIDATION**, re-badged through `asSelectionCriterion`, which refuses
any record whose role is not `validation` and keeps `from: 'validation'`. That is exactly the teaching claim,
implemented as a type constraint rather than a caption.

The withholding is structural: `HeldOutOnly` (`EndToEndShared.jsx:108-115`) returns a placeholder paragraph
instead of its children. Nothing is hidden with CSS. I confirmed the three sealed regions are the §6 report,
the §6 compact report and the §8 Wilson check — each with a placeholder naming what it is waiting for.

**The 48-setting sweep, driven through the real controls** (`drive4.cjs`): I enumerated all 16 eligible-candidate
subsets × 3 metrics, clicking the actual checkboxes and select, committing a prediction each time. **47 refuse
and exactly 1 offers the held-out step** — mask 14 (the three declared candidates) under validation balanced
accuracy. This matches the builder's claim and my own reading of `selectionOutcome`: `heldOutAvailable`
requires the chosen set to equal the declared set *and* the metric to be the declared one, which is
satisfiable by exactly one of the 48. The three empty-set settings are blocked with a named reason rather than
defaulting to a winner.

**The one caveat, which the record already states honestly.** The built chunk
`assets/end-to-end-supervised-learning-error-analysis-*.js` contains `0.966667`, because the page computes the
report client-side once the gate opens. This is unavoidable for a static site and `design.md`'s "Still not
covered" section says so in as many words ("it does not establish that the JavaScript bundle lacks them, which
it does not and cannot"). I looked for a stronger claim being made anywhere and found none. **This is not a
defect.** No held-out number is reachable *in the DOM*, which is the property claimed and the property that
matters for a learner.

---

# Blocking

## B1 — Investigation C grades the answer to a question it did not ask

**Where.** `src/learn/components/lesson-labs/EndToEndLabs.jsx:429` (the graded value) against
`:544-545` (the question), graded by `src/learn/components/lesson-labs/EndToEndShared.jsx:337-338`.

**What is wrong.** The numeric field asks:

> **What total cost will the proposed rule have?**  · `name: 'the proposed total cost'`

but `answerFor` supplies

```js
      value: comparison.difference,
```

and the grader compares the learner's number against that:

```js
  const guessed = gradable && numeric && shown.guess !== '' && answerIsNumeric
    ? Math.abs(Number(shown.guess) - shown.answer.value) <= tolerance
```

So the question names the **proposed total cost** and the grader compares the **difference between the
proposed and the baseline cost**.

**How I verified it — in the live page, at the default state.** `drive3.cjs`. Baseline threshold .6 gives
8 answered, 1 wrong, 2 deferred, cost `10·1 + 2·2 = 14`. Proposed threshold .8 gives 4 answered, 0 wrong,
6 deferred, cost `10·0 + 2·6 = 12`. I computed both by hand first, then answered **12** — the correct proposed
total cost — and committed. The page replied:

> = Your prediction matches: Lower total cost. **You wrote 12 for the proposed total cost; the calculation
> gives −2.00, which does not match.** 8 answered at 0.6 against 4 at 0.8; total cost 14 against 12.

I then answered **−2**, which is not a total cost at all, and the page replied **"which matches."**

And the reveal table rendered immediately beneath that verdict reads:

| Quantity | Active · 0.6 | Proposed · 0.8 |
| --- | --- | --- |
| Total observed cost | 14 | **12** |

**The page displays the correct answer, in its own table, in the same viewport, while telling the learner
their correct answer is wrong.** The same sentence also contains the contradiction in full: it says the
calculation gives −2.00 and then says "total cost 14 against 12".

The null case is no better. Under the preset "A threshold no case reaches" (1.01, nothing accepted), the
proposed cost is `2·10 = 20`; answering 20 returns *"the calculation gives 6.00, which does not match."*

**Why it matters.** This is the dominant defect class of the whole effort — the page grades a correct answer
wrong — and it is the fifth distinct variant of it. The sibling investigation in the same file documents the
exact hazard and avoids it (`EndToEndLabs.jsx:68-71`):

> */\* The number graded is the one the field's label asks for — the candidate's error count inside the
> slice — not the difference. A verdict that quotes one quantity while naming another has shipped in this
> repository before. \*/*

The comment is in investigation B. The bug is in investigation C, forty lines of the same file away.

**Fix the class, not the site.** Changing `value:` to `proposed.cost` repairs this instance. The property that
should be asserted, for **every** graded question on every lesson, is:

> **The quantity a question names is the quantity its grader compares, and the reference is the value the page
> itself displays for that quantity.**

Mechanically, that is checkable: each `numeric` spec already carries a human-readable `name`; require it to
carry the *path* to the displayed value too, and have the browser verifier, for each graded question, read the
number the page prints for that path, type it into the field, commit, and assert the verdict says it matches.
A guard framed that way would have caught this, the ECE/AUC mislabelling, and the `winners[0]` tie — none of
which the current per-lesson guards catch, because each was written against the previous instance's surface
rather than the invariant.

**Why the existing evidence missed it.** I opened
`docs/teaching/evidence/screenshots/endtoend-investigation-c-after.png`. The numeric field is captured
**empty**, showing its `a number` placeholder, so the verdict in the evidence carries no numeric clause at all.
The one capture that would have exhibited the defect was taken without exercising the input. The browser
verifier's acceptance case (`verify-endtoend-browser.cjs` record 8) likewise asserts only that the proposed
ledger is withheld and then shown; it never types a number.

---

# Should-fix

## S1 — A further inert guard: the Wilson leak pins in the models verifier, at a precision nothing produces

**Where.** `scripts/verify-endtoend-models.mjs:903-918`.

```js
const heldOutQuantities = [
  packet.test.balancedAccuracy.toFixed(6), packet.test.accuracy.toFixed(6), packet.test.logLoss.toFixed(6),
  fixed(wilson.lower, 6), fixed(wilson.upper, 6),
];
nonEmpty(heldOutQuantities, 5, 'the held-out quantities that must not leak');
for (const quantity of heldOutQuantities) {
  assert.ok(!developmentText.includes(quantity), …);
  record('no held-out quantity in the development output');
}
for (const quantity of [packet.test.balancedAccuracy.toFixed(6), packet.test.accuracy.toFixed(6),
  packet.test.logLoss.toFixed(6)]) {
  assert.ok(heldOutText.includes(quantity), …);
```

The absence loop runs over **five** quantities; the paired presence loop covers only **three**. The two
uncovered ones are `fixed(wilson.lower, 6)` and `fixed(wilson.upper, 6)` — the strings `0.858300` and
`0.995080`.

**Why they can never match.** Two reasons, independently:

1. The page prints the Wilson bounds at **three** decimals
   (`…/end-to-end-supervised-learning-error-analysis.jsx:485`, `fixed(wilson.lower, 3)`), so `0.858300` is not
   a form that exists anywhere.
2. More fundamentally, the subject here is `developmentText` — the **program's stdout**. The program computes
   no Wilson interval at all, at any precision. The string cannot appear in that subject under any
   circumstances.

So two of the five absence assertions are permanently inert, and each contributes a `record()` bump, so the
grouped-check total counts them twice per run as checking performed.

**Why this one matters more than the arithmetic.** This is the *same defect* the builder found and fixed in
`verify-endtoend-browser.cjs:229-236`, where the comment reads: *"Pinning them at six … was a pin that could
never match: the absence assertion passed trivially and proved nothing. It was the PAIRED presence assertion
that caught it, which is why the two are always written together."* **The fix was never back-ported to
`verify-endtoend-models.mjs`**, and the reason it was not caught there is precisely that the models verifier
breaks the paired rule for these two entries. The lesson the builder drew from Phase C is correct; it was
applied to one file and not the other.

**Fix.** Either drop the two Wilson entries from `heldOutQuantities` (the program never produces them, so they
do not belong in a check about the program's output), or move them to a check whose subject is the page text
and pin them at three decimals with a paired presence assertion — as `browser.cjs` already does.

## S2 — The hundredths-encoding guard cannot fail for any encoding

**Where.** `scripts/verify-endtoend-data.py:454-455`.

```python
    lossy = [hundredths for hundredths in range(0, 1401) if round(hundredths / 100 * 100) != hundredths]
    check(not lossy, f"the cutoff's hundredths encoding loses {len(lossy)} of its 1401 settings")
```

`round()` re-snaps every float to the nearest integer, so the predicate is false for every input. I ran both
forms myself:

| expression | failures over 0…1400 |
| --- | ---: |
| `round(h/100*100) != h` (as written) | **0** |
| `h/100*100 != h` (without the `round`) | **143** (first: 7, 14, 28, 29, 55, 56) |

The guard passes for *any* encoding whatsoever, including one that collapsed every setting to zero.

**A note on the fix, because the obvious one is also wrong.** Simply deleting `round()` would make the check
fail on 143 legitimate settings — `h/100*100 != h` is ordinary and harmless float behaviour, not a defect. The
property that actually matters for this control is that **the 1,401 cutoff values are pairwise distinct
doubles and each compares stably against the stored measurements**. That is what should be asserted:
`len({h/100 for h in range(1401)}) == 1401`, plus the boundary sweep the models verifier already runs. As
written, the line certifies nothing at all.

## S3 — Held-out quantities are printed without a role badge, contradicting the lesson's headline invariant

**Where.** `src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx:414` and `:485`.

The lesson's central discipline, stated in its own callout — *"Every printed score carries its role beside it,
so you never have to reconstruct which one you are reading"* — and in `design.md` Phase A — *"Every number
reaches the reader through `Score` or `Count`, both of which require a role"* — and again in Phase C — *"No
score anywhere lacks a role."*

I audited the post-gate page for every held-out quantity and asked, for each, whether it sits inside a
`.ete-score` (`drive5.cjs`). **Six of nine do not.** Four are the program's `heldOutOutput` code block, which
is program stdout and defensibly unbadged. The other two are authored prose:

- **`:414`** — the compact final report: `accuracy {heldOut.correct}/{heldOut.total} and balanced
  accuracy <Score record={heldOut.balancedAccuracy} />`. This renders **`35/36` as bare text** and the
  balanced accuracy through `<Score>` **in the same sentence**. The identical fact three paragraphs above
  (`:387`) correctly uses `<Count part={heldOut.correct} whole={heldOut.total} role="held-out" />`. So the page
  badges the quantity in one place and not in the other.
- **`:485`** — the Wilson bounds: `approximately [{fixed(wilson.lower, 3)}, {fixed(wilson.upper, 3)}]`,
  rendering **`[0.858, 0.995]`** with no role. These are computed from the held-out accuracy and are the one
  place on the page where a learner is most likely to want to know what kind of evidence they are reading —
  §8's own paragraph goes on to warn against misattributing this interval.

**Why the guards miss it.** `collectScoreRoles` (`verify-endtoend-browser.cjs:137-143`) enumerates
`.ete-score` and asserts each match carries a valid role. Its domain is exactly the set of numbers that
*already* have a badge, so it is structurally incapable of finding a number that lacks one. The source-hygiene
rule refuses four-or-more-decimal *literals*; `fixed(x, 3)` is neither a literal nor four decimals, so it
passes there too. This is a guard whose domain excludes the defect it exists to prevent — in the guard
protecting the lesson's single most important claim.

**Fix.** Wrap both in `<Score>`/`<Count>` with `role="held-out"` (the Wilson pair reads naturally as two
`<Score>`s with `digits={3}`), and add the complementary guard: enumerate every numeric text node inside
`.endtoend-lesson` outside a code block and assert it is inside a `.ete-score`, with an explicit allowlist for
the program output. That is the assertion the record's claim actually corresponds to.

## S4 — The design record's verifier counts are stale against the tree it describes

**Where.** `docs/teaching/drafts/end-to-end-supervised-learning-error-analysis/design.md`, Phase A
"Verification produced" table; and `scripts/verify-endtoend-models.mjs:23` and `:1277`.

I ran the verifiers myself:

| Record says | Tree produces | Verified how |
| --- | --- | --- |
| `verify-endtoend-models.mjs`: **283** grouped checks in **51** groups | **285** grouped checks across **52** groups | `node scripts/verify-endtoend-models.mjs --no-evidence` |
| `verify-endtoend-examples.py`: 52 oracles | 52 oracles | run |
| `verify-endtoend-data.py`: 2,434 checks, 100% of 2,214 leaves, 178 validated-not-re-derived | identical | run |
| 44,832 slice comparisons | 44,832 | run |
| 32 offline + 5 browser = 37 harness cases | `cases` has 32 entries, `browserCases` 5 | counted from source |

The models count is the only mismatch, and it is small — but this is the eighth of eighteen reviews to find a
record asserting a state the tree contradicts, and the published evidence file already records
`totalGroupedChecks: 285`, so the record disagrees with both the tree *and* the evidence beside it.

Separately, **a stale number survives inside the verifier and inside the published evidence**: the header
comment at `:23` and the evidence `scope` string at `:1277` both say the sweep runs **25,218** slice
comparisons, while the same run reports and asserts **44,832**. `25,218 = 1,401 × 18`, from a version that
swept 9 candidate pairs rather than 16. The exact assertion at `:562`
(`assert.equal(sliceCases, 1401 * 2 * allPairs.length)`) does bind, so the sweep cannot silently shrink — but
the published `scope` prose under-reports it by 44%, and the floor at `:1245` (`sliceCases >= 25000`) was
evidently written against the stale figure.

## S5 — The harness protects sources across a kill but not evidence

**Where.** `scripts/falsify-endtoend.mjs:456`, `:522-535`, `:625`.

The parts that are right, which I verified by reading the source: a lock file at
`scratch/endtoend-falsification.lock` refuses a second instance; the pre-mutation bytes go to an on-disk
`.orig` sidecar *before* the mutation and are removed only after a SHA-256-verified restore; a stale sidecar
refuses the next run; `--recover` exists; signal handlers are installed for `SIGINT`, `SIGTERM`, `SIGHUP` and
`SIGBREAK` via the loop at `:536-537` (I initially mis-read this as absent and corrected myself); and — the
part a sibling got wrong — **both** evidence sweeps are prefix-scoped to `endtoend-`, for
`docs/teaching/evidence` and `docs/teaching/evidence/screenshots` alike, so no path it writes can reach
another lesson's evidence. That is done properly.

The asymmetry is this. The evidence snapshot lives **only in the process heap**:

```js
const evidenceSnapshot = Object.fromEntries(evidenceFiles.map(file => [file, fs.readFileSync(file)]));
```

and is written back only at `:625`, inside the normal `finally`. `restoreFromSidecar` — the function every
signal handler, the `uncaughtException` handler and the `exit` handler call — restores the mutated **source
file and the lock, and nothing else**. So the file's own contract at `:20-22`…

> *every evidence file the run could touch is snapshotted first and restored in a `finally`, so falsifying a
> guard never rewrites the record it is falsifying — not even when a case throws*

…is accurate for a throw and **not** for a kill. The limitations list at `:652` is candid that crash safety is
best-effort and that "the `.orig` sidecar beside the mutated file holds its original bytes" — but the sidecar
covers sources only; evidence has no on-disk protection and `--recover` cannot restore it.

This matters because `BROWSER` at `:46` is `['node', ['scripts/verify-endtoend-browser.cjs']]` with **no
`--no-evidence`** (the browser verifier has no such flag and writes evidence and screenshots
unconditionally). A forced termination during a `--browser` case — on the machine that motivated the
crash-safety work, where `Stop-Process -Force` delivers no catchable signal — therefore leaves the *broken*
page's `endtoend-browser.json` and its captures on disk as the record, with no recovery path. That is exactly
the failure the evidence snapshot exists to prevent, at exactly the moment the comment at `:470-473` identifies
as the widest kill window.

**Fix.** Write evidence sidecars to disk alongside the source sidecar and restore them in
`restoreFromSidecar`; or, more cheaply, give `verify-endtoend-browser.cjs` a `--no-evidence` flag and pass it
from the harness, since a falsification run has no business writing the record at all.

## S6 — 144 of the 2,214 "re-derived" trust-root leaves are values compared with themselves

**Where.** `scripts/verify-endtoend-data.py:426-430`.

```python
    for row_index, row in enumerate(packet["validationRows"]):
        key = row["model"]
        …
        compare(f"/validationRows/{row_index}/model", key, row["model"])
```

`key` *is* `row["model"]`. The comparison cannot fail — and because `compare()` registers its path in
`seen_leaves` (`:117`), each call also counts toward the coverage figure that the file's headline presents as
its integrity claim. The packet has **144** `validationRows` entries, so **144 of 2,214 leaves — 6.5% of the
"100% re-derived" total — are self-comparisons**.

**On the mechanism as a whole, in fairness: it is real, not a tautology.** I checked how coverage is computed.
`all_leaves` is enumerated independently from the packet JSON; `seen_leaves` is populated *only* inside
`compare()`; and both directions are asserted at `:571-573` — `not extra` and `not missing`. So a leaf nobody
compared shows up as missing and fails the run, and a compared path that is not a leaf fails too. Coverage
cannot be inflated by omission. That is a genuinely well-built trust root, and better than most in this
repository.

**And the 178 exclusions are honest and completely listed.** Despite the dead `validated_not_rederived.update()`
at `:330` (see O7), the set *is* populated at `:285` with every split index, and I confirmed the published
`endtoend-data.json` lists all **178** paths explicitly (`/splits/test/0` … `/splits/validation/9`), with
`trustRootUncoveredPaths: []`. `106 + 36 + 36 = 178` checks out. The exclusion is declared, scoped, counted
and enumerated — the record is accurate here.

The fix for the self-comparison is to compare `row["model"]` against the key derived independently from the
row's position in the packet's own ordering, or to drop the path from the leaf set and say so.

## S7 — A cluster of inert assertions in the models verifier, including the inset floor that was written to prevent exactly this

**Where.** `scripts/verify-endtoend-models.mjs`.

**(a) `:429-433` — the inset floor guards nothing.** The assertion is:

```js
assert.ok(MINIMUM_INSET >= 12,
  `the minimum inset is ${MINIMUM_INSET}, below the 12 units every containment check assumes. `
  + 'This is asserted BEFORE any position is compared against it, because a containment check that '
  + 'reads the inset it guards passes at inset zero.');
```

I grepped every occurrence of `MINIMUM_INSET` in the verifier: the import at `:46` and these two lines. **No
containment check compares any position against `MINIMUM_INSET`.** The three geometries that carry
`inset: MINIMUM_INSET` — `scatterGeometry` (`endtoend-models.js:541`), `candidateBarGeometry` (`:631`) and
`recallComparisonGeometry` (`:696`) — have their containment checked against their **`padding`** instead. The
four geometries whose containment *is* checked against an inset (`:994`, `:1005`, `:1089`, `:1091`, `:1132`)
all use their own independent literal defaults — `flavanoidStripGeometry` 16, `pairedStripGeometry` 16,
`informationLaneGeometry` 14, `acceptanceRailGeometry` 16 — and only one of the four, the flavanoid strip
(`:997`), has a floor asserted at all.

So the sentence "the 12 units every containment check assumes" is false: no containment check assumes it. The
falsification case that sets `MINIMUM_INSET = 0` fires only because `:429` tests the constant against a
literal. **Setting `pairedStripGeometry`'s, `informationLaneGeometry`'s or `acceptanceRailGeometry`'s inset to
zero would break nothing** — which is precisely the defect the comment claims has been defended against.

**(b) `:1089-1092` — `inset >= inset`.** `endtoend-models.js:722` sets `const laneX = inset` and every lane
takes `x: laneX`, so `lane.x >= lanes.inset` is a value compared with itself; `lane.y = inset + 26 + …` makes
the y conjunct `inset + 26 >= inset`.

**(c) `:1132-1133` and `:994-995` — containment inside a scale built from the inset.** The rail's scale is
`linearScale([0.5, 1], [inset + 10, width - inset - 10])` and every tile is that scale applied to a confidence
in [0.5, 0.95], so tiles land inside `[inset, width - inset]` for *any* inset. Same for the flavanoid marks.

**(d) `:1148-1154` — the threshold-segment check reads the band it guards.** `acceptanceRailGeometry`
(`endtoend-models.js:806-809`) *defines* segment 1's `y2` as `tileBand.top - 2` and segment 2's `y1` as
`tileBand.bottom + 2`, so `segment.y2 <= rail.tileBand.top || segment.y1 >= rail.tileBand.bottom` is true by
construction in both disjuncts, for any `railY` and any band. The tick-label conjunct reduces to
`railY + 3 < railY + 7`, in which `railY` cancels.

This one is worth calling out specifically because it is the guard written for the Phase C paint-order defect
— the threshold rule crossing the boundary tile's own number. **The geometry is in fact correct** (I read it:
the band is `railY-20 … railY-6`, the tiles occupy `railY-20 … railY-6` with their number at `railY-9`, and
the two segments span `railY-36 … railY-22` and `railY-4 … railY+3`, leaving the number clear), and I
confirmed the gap is visible in `endtoend-investigation-c-after.png`. But the *check* would pass no matter
what the geometry did, so nothing would catch a regression.

**(e) `:901-902` — a string-length tautology.**

```js
assert.equal(`${developmentText}\n${heldOutText}`.length,
  developmentText.length + heldOutText.length + 1, 'the two halves are separate strings');
```

`(a + "\n" + b).length === a.length + b.length + 1` holds for every pair of JS strings. The real guard for this
property is `verify-endtoend-examples.py:156`, which compares the rejoined halves against the actual stdout
and does bind.

**Fix.** Where a containment check is meant to catch a drawing escaping its frame, the bound must be a
constant the geometry does not derive the position from — a literal, or the SVG's declared `width`/`height`.
Every one of (a)–(d) becomes non-vacuous by comparing against `width`/`height` and a hard floor rather than
against the geometry's own inset.

## S8 — Absence assertions with no paired presence assertion

The suite states the paired rule explicitly (`verify-endtoend-browser.cjs:22-23`) and applies it well in the
most important place — `:473-477` asserts that every held-out pin *is* present after the gate, with the
message "…so the absence assertion in section 3 could never have failed". That is exactly right and it is why
I can trust A6. Three places break it:

- **`verify-endtoend-browser.cjs:379-382`** — `heldOutProgramPins` is derived as
  `heldOutOutput.split('\n')[0].split(' ').slice(1)` = `['0.966667', '0.972222', '0.128708']`, and the loop
  asserts the absence of `test 0.966667`, `test 0.972222`, `test 0.128708`. Only the first is a string any
  rendering can produce — the word `test` precedes exactly one number. Two of three iterations are inert, and
  none has a paired presence assertion.
- **`verify-endtoend-browser.cjs:383`** — `assert.ok(!textBefore.includes('[[12'), …)` is never paired with a
  check that `textAfter` *does* contain it. (`verify-endtoend-models.mjs:919-920` pairs the equivalent
  correctly; the browser file is the one missing it.)
- **`verify-endtoend-browser.cjs:602-605`** — `${LESSON} .ete-table caption` can never match: the `Table`
  component (`EndToEndShared.jsx:245-260`) renders the caption as a sibling `<p className="ete-caption">` and
  has no `<caption>` path at all, which `endtoend-labs.css` states as a deliberate convention. The assertion
  is structurally incapable of failing, and nothing asserts the sibling caption exists.

---

# Observations

**O1 — The declined .05 fix is the right call.** Two development specimens, **28** (colour 3.95) and **146**
(colour 4.00), sit .05 apart on opposite sides of the default cutoff of 4; I confirmed both from my own refit
and in the live specimen inspector. At the scatter's 14-unit y-domain over roughly 300 rendered pixels that is
about one pixel, and no renderable width separates them. Moving quantitative marks apart to make a picture
legible would falsify the measurement, which is the worse error. Making the slice count and the specimen
inspector the stated authority, and saying so in a caption beside the figure, is the correct resolution. I
verified the inspector reports membership unambiguously ("In the chosen slice: yes" for 28, "no" for 146) and
prints colour intensity through `asInput`, at full precision, never rounded.

**O2 — The five Phase C look-only fixes are present in the images.** I opened all the relevant captures.
Figure A at 320 px carries one short label per lane with the explanatory sentences moved to the HTML table;
the y-axis title in the scatter sits in its own top band clear of the `14` tick label; the categorical axis
title is anchored left and clear of the bar labels; the acceptance rail's threshold rule is drawn in two
segments with a visible gap at the tile band; and the investigation captures are readable end to end with no
stitching seam and no sticky header covering the heading.

**O3 — The forbidden-edge rendering is sound.** I checked whether Figure A's counterexample arrow (report →
selection) is distinguishable from the legitimate edges, since drawing a forbidden edge like a permitted one
would teach the opposite of the intent. It is dashed (`stroke-dasharray: 2 3`), given its own colour, its own
legend entry and a footnote naming it as the counterexample. Not a defect.

**O4 — Figure lettering collides with the manuscript's.** The page renders Figure A, B, C, D *and*
Investigations B, C, D. The manuscript's "**Figure D** — a net gain contains two directions" is the page's
**Figure C**, and the page's Figure D is a different figure (per-class recall). `design.md` declares this as a
page-order departure with a reason, so it is disclosed rather than drifted. It is acceptable, but a reader
moving between the frozen packet and the page will find "Figure D" naming two different pictures; one
sentence in the record mapping the two letterings would remove the ambiguity.

**O5 — Investigation B is graded correctly at every degenerate input I could construct.** I exercised: an
empty slice (cutoff 0, lower side, and cutoff 12.5 upper side) — reported as ungradable with a named reason,
*not* as a zero difference or a tie; a candidate compared with itself — an exact tie graded `same` when
answered `same` and graded a miss when answered `fewer`, so the tie rule is genuinely exercised on real ties
rather than assumed; the boundary at exactly 4.00 on the upper side, where a specimen sits exactly on the
cutoff; and the full 36-row slice. Every numeric verdict matched my independently computed error count, with
tolerance 0 against an integer count and the caption stating that an exact answer is required. `bestByRule`
takes the first extremum in declared order rather than `winners[0]` after a sort, which is the correct rule and
is swept over 162 cases containing real ties. Controls and presets disable on commit, edits retire the verdict,
and nothing about the answer renders before commitment. **Investigation B is the model; investigation C is the
one that departed from it.**

**O6 — The verdict in investigation D reads `draft.metricKey` rather than the committed
`shown.inputs.metricKey`** (`EndToEndLabs.jsx:342`). This is safe today because the controls are disabled
while a result is shown, so the draft cannot diverge. It is the "presets built from the draft while graded
against the applied state" shape one edit away from mattering, and costs nothing to make explicit.

**O7 — `validated_not_rederived.update()` at `verify-endtoend-data.py:330` is a no-op.** `set.update()` with
no arguments adds nothing. It sits in the refit loop where the docstring says the fitted coefficients and tree
structures are validated rather than re-derived, which suggests an intent that was never completed. In fairness
the headline claim is unaffected: no fitted coefficient is a packet leaf, and everything computed from the
fitted state — the logistic probabilities via an explicit standardise/dot/softmax, the forest probabilities via
the explicit tree walk — genuinely is re-derived. The line is dead code in a misleading position, not a false
claim.

**O8 — `allRestoredExactly` treats a missing field as success.** `falsify-endtoend.mjs:636` is
`results.every(result => result.restoredExactly !== false)`, and the not-applied branch pushes a result with no
`restoredExactly` field at all, so `undefined !== false` reports it as restored exactly when nothing was
touched. The run still fails such a case through the separate inert check, so this is a reporting blemish.

**O9 — Neither python verifier floors its own counter.** `verify-endtoend-data.py` (`checks`) and
`verify-endtoend-examples.py` (`oracles`) print a headline count with no assertion that it stays above a floor.
For the data verifier the leaf-coverage check compensates for `compare()` calls but not for the ~60 `check()`
calls. For the examples verifier it compounds with a real risk: the five `if` branches at `:176-203` pattern-match
scraped program text, and if the print format changed they would all silently match zero lines, run zero
oracles, and still print PASS. A floor of today's 52 would close that.

**O10 — The record is unusually honest where it matters.** `design.md`'s "What is not covered" and "Still not
covered" sections state, without prompting, that the JS bundle necessarily carries the held-out numbers, that
the painted-against-declared check cannot catch a shape covered by a later sibling (the exact class the
threshold rule fell into, found by looking rather than by a check), that no independent content review has been
done, and that the split indices are validated rather than re-derived. The Phase C narrative of the capture
artifacts — a stitching seam, then a sticky header, then the page-versus-viewport coordinate bug — is the kind
of disclosure that makes the rest of the record credible. Where I checked those claims they held.

---

# Summary

The mathematics is clean. I refit the entire study from the served CSV with my own metric implementations in
exact rational arithmetic and 50-digit floating point, and compared 576 quantities across the packet, the
generated data module, the runnable example and the prose: **I found no disagreement with any number this
lesson states**, and the one apparent mismatch was a 1-ULP summation-order artifact in which the stored value
is the more accurate of the two. The float32 trap is real, correctly diagnosed and correctly implemented. The
held-out gate is the strongest I have tested in this effort: I attacked it through the initial payload, every
text node, every attribute, every form-control value, the serialized `innerHTML` and the accessibility tree,
and drove all 48 selection settings through the real controls, and **no held-out quantity is reachable in the
DOM before it is earned**.

One blocking defect. Investigation C asks for the proposed rule's total cost and grades the learner against
the cost *difference*, so a learner who reads the correct answer off the page's own table and types it is told
they are wrong, while the difference is accepted as a total cost. It is the fifth variant of this effort's
dominant defect class, and it shipped forty lines from a comment describing the same hazard. The fix for the
instance is one word; the fix for the class is an assertion that the quantity a question names is the quantity
its grader compares, referenced against the value the page itself displays.

The should-fix list is dominated by guards that cannot fail: a Wilson pin at a precision nothing produces
(the identical defect the builder fixed in one verifier and did not back-port to the other), an encoding check
whose `round()` swallows the failure it names, and an inset floor asserted with visible care that no
containment check ever reads. In each case the reasoning in the surrounding comment is right and the code does
not implement it. That, plus the role badge missing from the two held-out quantities the lesson's own
invariant most loudly promises to badge, is where the remaining work is.
