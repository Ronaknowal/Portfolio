# Time-Series Validation & Forecasting Baselines — independent phase-two review

Reviewed 21 September 2026 against the working tree plus the uncommitted time-series implementation. The
reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/`,
`docs/teaching/drafts/` or `docs/teaching/topic-notes/` was edited; this review document is the only write
outside `scratch/timeseries-independent/`. No state-changing git command was run. The build directory
`dist-ts/` was deleted and the preview on 4196 stopped at the end.

Three housekeeping notes, stated up front because all three bear on trust in the record:

- **Nothing on disk moved.** I took a SHA-256 baseline of all 5,516 files under `src/`, `public/`, `scripts/`
  and `docs/teaching/` before starting (`scratch/timeseries-independent/BASELINE-HASHES.txt`) and re-took it
  at the end. **Zero `timeseries-*` / `time-series-*` files differ.** All eight `timeseries-*.json` evidence
  artifacts and all 35 `timeseries-*` screenshots are byte-identical to the copies I took before starting.
- Every offline verifier was re-run with `--no-evidence`, which all five support, so no evidence file was
  rewritten and no restore-from-backup was needed. This is a genuine improvement on sibling efforts, where
  the reviewer had to restore evidence by hand.
- **I did not run `falsify-timeseries.mjs`.** An offline-only run replaces the report unconditionally
  (line 666, outside the `try`/`finally`), so it would have demoted the builder's `mode: "offline and browser"`,
  32-of-32 record to a 26-of-26 offline-only one. A `--browser` run would have rebuilt the tree repeatedly
  while a concurrent agent was actively editing sibling lessons (see below) — the exact condition that killed
  two of the builder's own runs. I verified the harness's safety properties by reading it and by one live,
  non-mutating test of the single-instance refusal instead. See S3 and O10.

A fourth note, on the environment: a **concurrent agent was editing the `end-to-end` and `problem-formulation`
lessons throughout this review**. Every file that differs between my two baselines belongs to one of those two
lessons (their verifiers, components, models, evidence and captures, plus a new
`PROBLEM-FORMULATION-INDEPENDENT-REVIEW.md`). None of it touches this lesson. I mention it because it is the
same hazard the builder's record describes, and because it is why I declined the harness run.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/timeseries-independent/`, importing neither `timeseries-models.js` nor `timeseries-data.js` nor `timeseries-examples.js` nor any `verify-timeseries-*` script nor the packet's `forecast-experiments.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1, mpmath 1.3.0 — the exact versions the packet declares). A complete replay of the forecasting study from the **served** CSV: all four baselines in exact `Fraction` arithmetic, and the direct-ridge study by **two** independent routes. A **50-digit `mpmath` LU** solve of all 616 ridge fits on incrementally accumulated Gram matrices. A **3,240,636-assertion** probe of the model layer (`probe.mjs`) over 7,644 fit configurations, 1,176 eligibility grid points and 16,380 donor-index cases, with all reference values written from the prose. Reviewer-written Playwright drives (`drive.cjs`, `fig5.cjs`, `zoom.cjs`, `hatch.cjs`, `paintorder.cjs`, `grade.cjs`, `fig4states.cjs`) sharing no code with `verify-timeseries-browser.cjs`, at 1366, 390 and 320 px on Edge, plus pixel sampling of the rendered PNGs with Pillow. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `data-source.json`, `source-description.txt`, `calculated-inputs.json`, `forecast-experiments.py`); the lesson body, model layer, both generated modules, all three lab/figure/shared components and the CSS; all six verifiers and the harness; the blueprint; the served assets; all eight evidence artifacts. |
| **Reused (declared)** | scikit-learn's `Ridge` and `StandardScaler` at the declared hyperparameters for **one** of my three ridge routes, because the protocol *is* those calls; the other two routes are written here. One subagent for a fan-out read of `verify-timeseries-sources.py`, `verify-timeseries-render.cjs` and `falsify-timeseries.mjs`. **Every finding the subagent surfaced I re-derived myself before reporting it** — I read each cited line, ran each verifier myself, and tested the lock refusal myself. Two of its claims I did not confirm and have therefore not reported. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/time-series-validation-forecasting-baselines.jsx` | `882b361f075140d99d787e6911e53fa98297d05fc8125fc2764bddadd7e0fc37` |
| `src/learn/data/timeseries-models.js` | `2b54cd79270c99c20937e358dade6142e17be878071fdcf774b81d5b90551a0c` |
| `src/learn/data/timeseries-data.js` | `fee13a6bba9101d234993a55ddc48747460034ca1c43c002a9446dac16fd691d` |
| `src/learn/data/timeseries-examples.js` | `52a67945accc52adf019c60dac7f83cd532340e00b0e42666b8d49043cd375e3` |
| `src/learn/components/lesson-labs/TimeSeriesShared.jsx` | `f1390908e08db2329be5f2c3d822672f6280560983dfd2858eac16bdcf941df1` |
| `src/learn/components/lesson-labs/TimeSeriesLabs.jsx` | `c9a844bcb308d6a00f8044935fd6d09b36969be712a74b639f58fef9bee5c826` |
| `src/learn/components/lesson-labs/TimeSeriesFigures.jsx` | `7663a731e967e791f64ff23b27180514f4dc4c1dfff4f142f8ec332713319fe1` |
| `src/learn/components/lesson-labs/timeseries-labs.css` | `9a88ddbe9577963279e89fac3415126a7487bc1200e9285108d316cd1f501e3f` |
| `src/learn/data/curriculum/blueprints/time-series-validation-forecasting-baselines.js` | `8cafe786f2fee809b1fd4fad84e14bf051fff3d6383ace91aaa0c631e4eab675` |
| `scripts/verify-timeseries-models.mjs` | `830f6815a6cce9cf27931e8a88132c51af40d35eaa69796fb418354cc1f9feb9` |
| `scripts/verify-timeseries-data.py` | `8c1f0df86fe2a882ea827ed85dee1f428dd74983109065d3eb5da2aa44b6e395` |
| `scripts/verify-timeseries-examples.py` | `892f7737914170cad234e7db4614ac2396d793402f9fafa605d2d2060ea7dc04` |
| `scripts/verify-timeseries-sources.py` | `7b9fa90e5c9a7b7c9b0d2a9f9e4e6e03daec280ce41dfa4584cdd571671c6461` |
| `scripts/verify-timeseries-render.cjs` | `6e3c17606b5ab858550e6b92ab347813aa7fe292f7d88d7dceec7f90f16ed223` |
| `scripts/verify-timeseries-browser.cjs` | `ac5dbf67e1685bd54d65873f05b31845703e675ed825829cdf103a0bfbdbe2f9` |
| `scripts/falsify-timeseries.mjs` | `fa779136cdd7f161d195829a3e0c6205fb971127ec43143424531a5081024477` |
| `docs/teaching/drafts/.../lesson.md` | `5e672c664079e62eaac0c33893bf95a2db90532fc1973f5128c463303e0baab6` |
| `docs/teaching/drafts/.../design.md` | `41cee48d56ae322ec723676e24cb073ed7f64b1403e68a6d5887183ed3758491` |

---

# Part A — correctness

## A1. The forecasting study, replayed from the served CSV — **no disagreement**

`scratch/timeseries-independent/replay.py`, `exactmae.py`, `ridge.py`, `ridge50.py`, `compare.py`.

**The data.** The served CSV (`public/learn-assets/.../bike-sharing-daily.csv`) is byte-identical to the packet
member — SHA-256 `a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be`, 57,569 bytes, 731 rows.
The calendar is complete and consecutive from 2011-01-01 to 2012-12-31 with no gap, and
`casual + registered == cnt` holds **exactly on all 731 rows** — which is both the lesson's stated reason for
excluding those two columns and a real internal-consistency check on the file.

**The schedule.** `np.arange(364, 731-7, 7)` gives **52 origins**, 364 through 721. I confirmed independently
that **every one of the 52 is a Saturday**, that index 364 is 2011-12-31 and index 721 is 2012-12-22, that the
first 36 origins target 2012-01-01 through 2012-09-08 and the last 16 target 2012-09-09 through 2012-12-29,
and that exactly two source days (2012-12-30, 2012-12-31) remain unscored because they do not form another
complete seven-day block. Every date in §5's contract is correct.

**The baselines, in exact rational arithmetic.** Errors accumulated as `Fraction`, never as floats:

| | Development MAE | Development RMSE | Page |
| --- | --- | --- | --- |
| Historical mean | `2083.7024130118101` | `2383.2453` | 2083.70 / 2383.25 ✓ |
| Naive | `1198.2976190476190` | `1532.4513` | 1198.30 / 1532.45 ✓ |
| Seasonal naive, 7 | `987.32142857142857` | `1348.1035` | 987.32 / 1348.10 ✓ |
| Drift | `1203.0007392504937` | `1538.2780` | 1203.00 / 1538.28 ✓ |

Final: naive `1310.3392857142857` → 1310.34 ✓, seasonal `1390.9107142857142` → 1390.91 ✓. The drift slope is
`(y_T − y_1)/(T−1)` as stated; I also ran it **without** the declared zero-clipping and got identical results,
so clipping never binds on this series — the rule is honest but inert here, which is fine and worth knowing.

**The direct ridge, by three routes.** I wrote the 14-feature construction from §5's prose only
(`y_s, y_{s−1}, y_{s−6}`, the past-seven-day mean, seven target-weekday indicators, sine/cosine of target
day-of-year on the disclosed 365.25 approximation, elapsed calendar years), fitted a per-horizon
`StandardScaler` on the eligible rows and `Ridge(alpha=1, fit_intercept=True)`, and clipped negatives:

| | My sklearn route | My 50-digit LU route | Page |
| --- | --- | --- | --- |
| Expanding, development | MAE 772.3324, RMSE 1045.4507 | identical to 4 dp | **772.33 / 1045.45** ✓ |
| 90 eligible rows, development | MAE 883.8012, RMSE 1169.2027 | identical | **883.80 / 1169.20** ✓ |
| Expanding, final | MAE 1167.6228 | identical | **1167.62** ✓ |

The final per-horizon curve came out `1134.02, 1118.55, 1530.54, 965.26, 931.64, 926.52, 1566.82` — **every one
of the seven values in §5's table, to the printed precision, from an implementation that shares no line with
the builder's.**

**The third route.** The builder re-derived ridge by normal-equations/LU against sklearn's SVD pipeline with a
worst gap of 7 × 10⁻¹⁵ over 616 predictions. I verified this by a **third** route: 50-digit `mpmath` LU on Gram
matrices accumulated incrementally in `mpf` (nested for the expanding window, rebuilt for the sliding one),
with standardisation written from its definition including the zero-variance → scale-one rule.

- worst relative gap, **my sklearn SVD vs my 50-digit LU**, over all 616 predictions: **6.382 × 10⁻¹⁵**
- worst relative gap, **the packet's recorded `calculated-inputs.json` vs my 50-digit LU**: **4.766 × 10⁻¹⁵**

The builder's 7 × 10⁻¹⁵ claim is confirmed, and confirmed against a route neither of us shares. I also
reproduced every `summary` block in `calculated-inputs.json` — all six development candidates' MAE, RMSE, bias
and both per-horizon profiles, and all three final ones.

**The fit counts.** 36 origins × 7 horizons × 2 ridge variants = **504** development fits and 16 × 7 = **112**
final fits, exactly as §5 and `/protocol` state, 616 in total. The model layer's separate **728** is
52 × 7 × 2 — a different, larger sweep over the whole contract, not a restatement of 616. Both are correct;
they are simply different objects, and the record keeps them apart properly.

## A2. Every toy fixture, practice solution and investigation number — **no disagreement**

`scratch/timeseries-independent/inv.py`, all in exact `Fraction`.

- §2 toy history `10,20,10,20,12,22`: mean `47/3`, naive 22, seasonal-2 `12,22,12,22`, drift
  `122/5, 134/5, 146/5, 158/5` = 24.4, 26.8, 29.2, 31.6, slope `12/5` = 2.4 ✓. Against outcomes
  `12,22,12,22`: MAE 5, 5, **0**, 11 ✓.
- Investigation 1's edit (fifth value 12 → 18): seasonal-2 becomes `18,22,18,22`, naive and drift unchanged
  because neither endpoint moved ✓.
- All six season lengths through eight horizons on the toy history reproduce the module's
  `all_period_forecasts` exactly, including the wrap beyond one cycle.
- §3 recursive `24,26,28,30` and updated `24,14,24,14`, **both MAE 10** ✓; changed continuation `22,24,26,28`
  gives recursive MAE **2** and updated `24,24,26,28` MAE **1/2** ✓.
- §3 eligibility: at cutoff 12, h 3, d 2 the eligible origins are `[6, 7]`; at d 0, `[6, 7, 8, 9]`; at cutoff
  10, d 2, `[]` ✓. Splitter gap `g ≥ h + d − 1` verified as arithmetic for h 1–10 × d 0–10: the last training
  row at `t − g − 1` has its label arrive at exactly `t`, never after ✓.
- Practices 1–8: origins `4,5,6` then `4…8` ✓; `4,10,8,4,10` and `4,12,8,4,12` with horizon three unchanged ✓;
  MAE 2 both / RMSE √8 ≈ 2.828 vs 2 ✓; latest eligible origin **17** with targets overlapping on days 19–20 ✓;
  scale `5/3`, scaled error exactly `9/10` ✓. §7's `0.9⁷ = 0.4782969` ✓.
- Investigation 3's two named origins: at 2011-12-31 seasonal-7 MAE is `7298/7` = 1042.571429 against naive
  `5527/7` = 789.571429; at 2012-01-07 it is exactly **978** against `10267/7` = 1466.714286 ✓ — all four
  numbers in §5's prose.
- Figure 5 and §5's **"history position 617 of 617"**: the first final origin is index 616, so its history is
  617 values. Correct. This was one of the builder's own nine fixes (it previously read "400 of 400") and the
  replacement is right, not merely different.

**Not one number in the manuscript, the model layer, the generated modules or the rendered page disagreed with
my independent computation.** That is a real result and I want it stated plainly: across the whole study, every
figure, every table, every practice solution and every investigation readout reproduced.

One rounding note, in O7 — three per-horizon values are exact ties at the third decimal and the page rounds
them half-away-from-zero. It is self-consistent and not an error.

## A3. The leak property, checked exhaustively rather than at samples — **no disagreement**

`scratch/timeseries-independent/probe.mjs`. Reference values written in that file from the definitions; the
module imported only as the subject under test.

The builder splits the claim into three that fail separately — feature causality against the row's own origin,
label maturity against the fit's origin, and closure — and decides eligibility by an availability-set route.
I verified that route is **not a tautology**: `knownByDay` builds, for each day, the explicit set of
observations that have arrived (`observation + delay <= day`), and `scheduleBySimulation` then decides each row
by **set membership** (`arrived.has(origin + horizon)`). It never writes `s + h + d ≤ t`. That is a genuinely
different computational path to the same answer, and the two are asserted to agree on the exact list of
training origins, not merely on how many.

I then built my own third route and swept far wider than either:

- **7,644 fit configurations**: all 52 contract origins × 7 horizons × 3 window settings (`null`, 90, **1**) ×
  7 reporting delays (0, 1, 2, 3, 4, 7, 10).
- For every one: the module's `trainOrigins` equals my forward availability simulation **exactly**; the audit
  reports **zero violations**; the largest observation index touched is ≤ the issue origin **and ≤ origin − delay**
  (a strictly tighter bound than the module asserts, which also holds everywhere); every training row's feature
  set equals my independently constructed `{s−d−6 … s−d}` and every label satisfies `label + delay ≤ issueOrigin`.
- **1,176 eligibility grid points**: cutoff 0–20 × horizon 1–8 × delay 0–6, against my own availability sets.
- **16,380 donor-index cases** and 282 baseline-rule combinations: history lengths 2–40 × every season length
  1–T × horizons 1–20, with mean/naive/seasonal/drift each checked against my own formula.

**Total: 3,240,636 assertions, zero failures.** This covers every origin at every admitted delay, where the
builder's own delay sweep covers 4 sampled origins (O6). The sampling hides nothing — but the property is now
established exhaustively rather than by sample.

**The baselines' information sets are exactly what the page claims.** `forecastRequest` hands the rules
`counts.slice(0, origin + 1)` and attaches outcomes separately, only for scoring; I confirmed by construction
that no baseline reads an index above its origin, and the seasonal and drift rules in particular use only
`history`, never any test-period statistic. The horizon-7 identity between naive and seven-day seasonal naive
is mechanical, not coincidental: I verified `seasonal7[h=7] === naive[h=7]` at **all 52 origins**, and in the
recorded data module the two final values are the byte-identical double `1362.4375`, which is why the tie
detection below genuinely fires.

## A4. The trust root — the mechanism is real, the counts are right, the exclusions are fair

`scratch/timeseries-independent/leaves.py`.

I wrote my own leaf walker over `calculated-inputs.json` and counted **exactly 6,659 scalar leaves** — the
denominator the record claims, confirmed independently.

The mechanism is not a tautology, and the reason is structural: `derived_paths` and `validated_paths` are
populated **as a side effect of `compare()` actually comparing a re-derived value against the packet's**
(`verify-timeseries-data.py:140`). Coverage cannot be claimed without a comparison having happened. Crucially
the verifier then asserts **both directions** (lines 531–534): `missing = every_leaf − covered` must be empty,
*and* `stray = covered − every_leaf` must be empty. The `stray` assertion is what makes inflation impossible —
you cannot pad coverage with paths that are not leaves. A third guard (`len(every_leaf) > 6000`) stops the
walker silently failing to reach the records.

The **12 declarations** are exactly the genuinely non-computable free choices, which I enumerated
independently before looking at the verifier's list and got the same set: `ridge_alpha`, `ridge_solver`, the
two `ridge_windows` entries, the four `/versions` strings, and the four prose policy strings (`selection`,
`final_policy`, `count_availability_assumption`, `prediction_postprocessing`). 8 + 4 = 12. Everything a machine
could derive **is** derived, including `source_sha256`, `selected_method`, both origin-index lists, the dates,
the unused tail and the 504/112 fit counts. This classification is principled, not convenient.

**Judging the two exclusions.** Both are honest and genuinely uncomputable from the tree: the provider
archive's own digest (the zip deliberately not retained) and that the retained CSV is what the provider served
on the recorded date. I accept both. I would note only that the first was made permanently uncheckable by a
choice — retaining the zip would have cost a few megabytes — and that what remains is still substantial: the
CSV is pinned by digest, is byte-identical between packet and served copy, carries CC BY 4.0 attribution, and
satisfies `casual + registered == cnt` on every row. That is about as much as can be had without the archive.

---

# Part B — the rendered page

Built with `npx vite build --outDir dist-ts` and previewed on `127.0.0.1:4196 --strictPort`; driven at 1366,
390 and 320 px on Edge with reviewer-written Playwright. **I opened and read every image**, at up to 6× device
scale, and sampled painted pixels out of the PNGs with Pillow where the eye was not enough.

The page renders cleanly: no page errors, no console errors, no error boundary, 5 figures, 3 investigations,
7 tagged diagrams at all three widths, and **no horizontal overflow at any width** — `scrollWidth` equals
`clientWidth` at 1366, 390 and 320, and no element inside a figure or investigation extends past the right
edge. At 320 px the tables reflow into stacked labelled blocks and stay readable. I also confirmed
**nothing is revealed on first paint**: zero verdict nodes, zero reveal panels, zero preselected controls.

**The builder's nine screenshot-found fixes, confirmed in the images.** Figure 2 now prints
`target 11 + delay 2 = 13 > 12` beside "No" and `latest feature day 6 + delay 2 = 8 ≤ 12` beside "Yes" — both
relations true, and the operator is computed from the same boolean that decides the verdict
(`TimeSeriesFigures.jsx:215, 219`), so they cannot drift apart again. The "features" label clears its squares.
Figure 3's third lane marks three arrows, not four, and its row reads "3 of its 4 inputs are dated after day 0".
Figure 4 is two panels. Figure 5 prints "617 of 617". Investigation 3's verdict prints six decimals
(`Seasonal MAE 789.571429 against naive MAE 789.571429`), not a raw double. No table clips a column.

**Figure 4's drawn boundaries, checked against the schedule they depict, in all six control states.** For every
combination of {expanding, sliding} × {h = 1, 4, 7} I computed the training window myself and compared it with
what the page prints and draws: rows used `358, 365, 372` / `355, 362, 369` / `352, 359, 366` expanding and
`90, 90, 90` sliding — **all six agree**. The boundary table's `Last training origin` equals `t − h` for all
seven horizons, `Rows eligible` counts down 358→352, `Target day` counts up 365→371, and `Latest day read` is
**364 at every horizon in both modes**, which is the lesson's central claim rendered as a column. Panel A keeps
one left edge when expanding and moves it (268 / 275 / 282) when sliding, carries no boundary or target marks
as the record claims, and draws not-yet-reached rehearsals as dashed outlines. Panel B places the boundary at
−h and the targets at +1…+7, magnifying the boundary-to-issue gap by 18× at h = 1 and 5.6× at h = 7, both well
above the claimed 3×. **No drawn split boundary contradicts the audited contract.**

**Figure 1's calendar** is correct: 14 real days from 2011-12-25 to 2012-01-07, weekday letters right, the
Saturday lane observing 7 of 14 days and the Friday lane 13 of 14, arrows of length 7 and 1, and the target
square identical in both lanes. Its hatching, however, is B1.

**Figure 5's tie** is named for *both* rules — "Naive and Seasonal naive, 7 days" at h = 7 — and the chart draws
an explicit ring for "a point two rules share", which is the right mitigation for the coincident-point paint
hazard. Every value in both tables matches my independent computation.

**The six "DOM right, picture wrong" mechanisms.** CSS beating a presentation attribute: not found. CSS
collapsing geometry: the SVG layout rule is correctly scoped to `svg.ts-diagram` and both KaTeX radicals
measure 111 × 60 and 21 × 17 px identically at all three widths, far above the 4 px floor — no collapse.
Contrast: **found, B1**. Paint order: **found, B1**. A `<rect>` over a `<text>`: the axis-title-over-target-rects
case the builder fixed is gone; I found no new instance. A containment check comparing positions against the
inset it guards: **not present** — `verify-timeseries-browser.cjs:181–187` measures each label's
`getBoundingClientRect()` against the **SVG's own** `getBoundingClientRect()`, which is the correct subject.

---

# Blocking

## B1 — Figure 1's "hatched" cells carry no hatching; the legend and caption name an encoding the picture does not contain

**Where.** `src/learn/components/lesson-labs/TimeSeriesFigures.jsx:78–91` (draw order) and
`src/learn/components/lesson-labs/timeseries-labs.css:91` (`.ts-cell`).

**What is wrong.** Figure 1's legend says **"not yet observed (hatched)"** and its caption says **"hatched cells
had not happened yet at that moment."** There is no visible hatching anywhere in the figure, at any width, at
any magnification.

The hatch lines exist in the DOM and every attribute reads correctly — 31 `<line class="ts-hatch-line
is-unknown">` elements, `stroke: rgb(74, 86, 80)`, `stroke-opacity: 1`, `opacity: 1`, `visibility: visible`,
each with a real bounding box 16 units tall, correctly positioned inside the cells they mark. They are simply
**painted over**. `<Hatch>` is emitted first (line 82), then the `ts-known-band` rect (line 85), then a
`ts-cell` rect for **every** day including the unobserved ones (lines 87–92). SVG has no `z-index`; document
order is paint order. The bare `.ts-cell` class — the one used for exactly the unobserved days — is
`fill: #141a19`, **fully opaque**, so each unobserved cell's own rectangle covers the hatching that cell exists
to display.

**How I verified it.** Four ways, because an attribute check cannot see this:

1. Read the picture at deviceScaleFactor 6: no hatching visible in any unobserved cell.
2. Sampled the painted PNG with Pillow along the middle row of the cells. Inside an unobserved cell the only
   colours present are the fill `rgb(20,26,25)`, the cell stroke `rgb(47,55,53)`, and two faint values
   `rgb(22,27,27)` and `rgb(27,32,31)` — sub-pixel antialiasing bleed at the rect's edge. **The declared
   `rgb(74,86,80)` is nowhere on the canvas.**
3. Enumerated the painted children in document order: the hatch lines occupy indices 0–26 and the cell rects
   27–41 and 63–67, with `fill-opacity: 1` and `opacity: 1` on every rect. No ancestor carries opacity, a
   filter or a blend mode.
4. Contrast, computed both ways:

   | | ratio |
   | --- | --- |
   | hatch **as declared** vs unobserved fill | 2.30 : 1 |
   | hatch **as painted** vs unobserved fill | **1.07 : 1** |
   | observed vs unobserved cell **fill** | 1.30 : 1 |
   | observed vs unobserved cell **stroke** | 2.06 : 1 |

**Why it matters.** Three reasons, in increasing order.

First, the page states something about its own drawing that is false — the same category as the `13 ≤ 12` the
builder found, and the reason this review exists.

Second, the fallback encodings are weak. With the hatch gone, "observed" and "not yet observed" are
distinguished only by a 1.30 : 1 fill difference and a 2.06 : 1 stroke difference. Both are below the 3 : 1
floor for non-text contrast, so the distinction this figure exists to draw — *what the forecaster had actually
seen* — rests entirely on differences a reader may not resolve. The hatch was presumably the redundant channel
that made it safe, and it is not there.

Third, and most tellingly, **the same helper works correctly forty lines away**. At
`TimeSeriesFigures.jsx:458`, figure 4's "training continues past this window" marker emits `<Hatch>` *after*
its `ts-train-band` rect, and that hatching **is** plainly visible in the rendered panel B — I confirmed it in
the sliding-mode capture. So the helper is sound and the ordering is the defect.

The code even anticipates the hazard and then implements the opposite. The comment immediately above the call
reads: *"Unknown region first, so the known band and the target cell paint over it rather than under it. Paint
order decides what a reader sees, and no attribute check can see paint order."* The intent is clear — hatch the
whole span, then let the **known** band paint over the part that should not be hatched. What defeats it is that
a `ts-cell` rect is drawn for the unknown days too, and its base fill is opaque rather than `none`. And
`Hatch`'s own docstring says it draws real lines rather than a `url(#pattern)` fill precisely because a pattern
"fails silently and leaves the band unpainted, which is how a figure's regions once became invisible while
every offline assertion passed." **The same outcome has recurred through a different mechanism.**

**Not a capture artifact.** I separately confirmed that the apparent clipping in my first element screenshots
of figures 3–5 *was* a Playwright scroll-stitch artifact and is not a page defect — no element on the page has
`overflow: hidden` with content exceeding its box. B1 is not in that category: it reproduces in full-page
captures, in element captures at a 3,200 px viewport, and in direct pixel sampling.

---

# Should-fix

## S1 — Figure 3 styles each lane's first arrow as carrying an input it does not carry

**Where.** `src/learn/components/lesson-labs/TimeSeriesFigures.jsx:302` —
`kind={invalid ? 'invalid' : lane.kind}`.

**What is wrong.** Arrow style is chosen **per lane**, not per arrow. The model layer already computes the
right answer and the figure never asks for it: `recursiveTrace` sets
`inputKind: lead === 1 ? 'observation' : 'prediction'` (`timeseries-models.js:485`) and `updatedTrace` sets
`inputKind: lead === 1 ? 'observation' : 'newly observed outcome'` (`:516`). `row.inputKind` is read nowhere in
the components.

The consequence is visible in the rendered figure:

- **Lane 1, "fixed origin, recursive".** All four arrows are drawn in the `prediction-fed` style, whose legend
  entry reads **"input is this chain's own prediction"**. The first arrow's input is the *observed* count 22.
  The chain cannot have fed itself a prediction it had not yet made.
- **Lane 2, "advancing origin, updated".** All four are drawn `observation-fed`, legend **"input is a newly
  observed outcome"**. The first arrow's input is the origin's own count, already in hand at the origin — not
  a newly observed outcome.
- **Lane 3** is correct: its first arrow is `observation-fed` and only the three genuinely late ones are marked
  invalid.

**Why it matters.** This is the residue of the defect the builder just fixed in lane 3, left standing in lanes 1
and 2 — a blanket lane style applied to an arrow whose actual kind differs. It matters more here than a generic
styling slip because the distinction it erases *is the subject of the figure*: the whole reason recursive and
updated forecasting differ is which arrow carries a prediction and which carries an observation, and the first
arrow — identical in all three lanes — is the fixed point that makes the comparison legible. The page even
contradicts itself: its own prose beneath the table says "the first one, which reads the origin's own count, is
not at fault", acknowledging exactly the input kind that lanes 1 and 2 mis-style.

The legend wording is unambiguous that the style is a claim about the arrow, not the lane: "input **is** this
chain's own prediction". So there is no defensible reading under which the current drawing is right.

**How I verified it.** Read the rendered figure at 1366 px; read `recursiveTrace`/`updatedTrace` and confirmed
`inputKind` is computed per row; grepped the components and confirmed `inputKind` has no reader; read line 302
and confirmed `lane.kind` is a single value per lane.

## S2 — `design.md`'s Status section reports 111 source-hygiene checks; the verifier and the evidence both say 113

**Where.** `docs/teaching/drafts/time-series-validation-forecasting-baselines/design.md:342` — "sources
(111 checks over 17 files)".

**What is wrong.** The same document says **113** eighty lines earlier, at line 312: "12 files parsed; 113
hygiene checks in total." I ran the verifier: it prints **`PASS: 113 source-hygiene checks over 17 files`**, and
`docs/teaching/evidence/timeseries-sources.json` records `"checks": 113`. The Status line is simply wrong.

**Why it matters.** This is the record asserting a state the tree contradicts — the failure mode the brief
records in eight of the last nineteen reviews. It is a small number and nothing downstream depends on it, but
the Status section is the part of the record a later reader trusts without re-running anything, and it is
internally inconsistent with the same document's own narrative section. Every other count in that section I
checked is right (see the table in Part C), which makes the one wrong entry more likely to be believed.

**How I verified it.** Ran `verify-timeseries-sources.py --no-evidence` and read the printed total; read the
`checks` field of the pre-existing evidence file; read both lines of `design.md`.

## S3 — A crashed falsification run leaves the previous report on disk, still reading green

**Where.** `scripts/falsify-timeseries.mjs:619–627` (the `finally`) and `:665–666` (the report write).

**What is wrong.** The harness snapshots every `timeseries-*` evidence artifact before it starts and restores
them in a `finally`. That snapshot **includes `docs/teaching/evidence/timeseries-falsification.json` itself**,
because it matches the `startsWith('timeseries-') && endsWith('.json')` filter at line 461. The new report is
written at line 666, which is **after** the `try`/`finally`.

So if anything throws inside the `try` — a breakage that fails to restore byte-for-byte (the explicit throw at
line 613), a build broken mid-run by a concurrent agent, a verifier subprocess dying — the `finally` restores
the **old** report and line 666 never executes. The run dies, and the record on disk still says
`"passed": true, "mode": "offline and browser", "guardsThatFired": 32`, with nothing anywhere to indicate that
a later run started and failed.

**Why it matters.** Every one of the five verifiers writes a provisional `"passed": False` record *before* it
starts and stamps success only after its final assertion — the record says so explicitly, and
`verify-timeseries-sources.py:409` even asserts that they all do. The harness, whose entire purpose is to
demonstrate that the guards can fail, is the one artifact in the suite that does the opposite. And this is not
hypothetical: `design.md:314–318` records **two** runs dying mid-way on a concurrent agent's build break. On
both occasions the source tree was correctly protected by the sidecar, but the falsification record on disk
would have been the pre-crash one.

**Why it is Should-fix and not Blocking.** The source tree is genuinely safe — that protection is real and I
tested part of it. The exposure is to the *record*, and only in the window between a crash and the next
successful run. The fix is small: write a provisional failing report before entering the `try`, as the five
verifiers already do, and exclude the harness's own report from the restore snapshot.

**How I verified it.** Read lines 455–470, 500–520 and 616–632 and 660–670 directly. Confirmed that
`timeseries-falsification.json` matches the snapshot filter. Did **not** trigger a crash to observe it, because
doing so would have required running the harness.

## S4 — Investigation 2's verdict prints an internal token, `none-qualify`

**Where.** `src/learn/components/lesson-labs/TimeSeriesShared.jsx:294` (`label`) with
`TimeSeriesLabs.jsx:257, 259`.

**What is wrong.** When the eligible set is empty and the learner correctly ticks "none of them qualify", the
verdict reads:

> Your prediction matches: **none-qualify**.

`label(key)` resolves a key through the `options` array; investigation 2 uses a set-valued custom control and
passes `options={[]}`, so `label` falls through to `?? key` and prints the raw internal identifier
`NONE_QUALIFY = 'none-qualify'`. The control the learner clicked says "none of them qualify".

**Why it matters.** It is the only place on the page where an internal identifier surfaces in learner-facing
prose, and it lands on the one branch the lesson makes a point of — §3's "An empty eligible set is a meaningful
result". The grading itself is **correct**, and the explanation beside it is excellent ("No offered origin
satisfies s + 3 + 2 ≤ 10: the earliest offered origin is 6, whose label would arrive on day 11"), which is why
this is cosmetic rather than a grading defect. But it is exactly the shape of the "verdict quoting one quantity
while naming another" family, and it reads as unfinished.

**How I verified it.** Drove the page, clicked the lesson's own "Cutoff 10, h 3, delay 2" preset, ticked "none
of them qualify", submitted, and read the rendered verdict text out of the DOM
(`scratch/timeseries-independent/grade.cjs`, case C). Confirmed the fall-through in `label`.

---

# Observations

## O1 — Figure 4's overview axis carries only two ticks in expanding mode

`timeseries-models.js:758–773`. `axisTicks({low: 6, high: 385, count: 4})` returns **`[6, 200]`**: the raw step
is 126.3, the chosen step from `[1, 2, 2.5, 5, 10] × 10²` is 200, the first multiple at or above 6 is 200, and
400 exceeds the domain, so one tick plus the unshifted low. The bars end at days 364–378 — past the last label.

I checked whether this misleads: linear extrapolation from the two labels recovers 364 essentially exactly, and
in sliding mode the domain narrows and three ticks appear (268, 300, 350), so the sparse case is specific to
expanding mode. The panel's stated job is the left edge, which *is* labelled, and each lane is named by its own
origin. So this is legibility rather than falsity — but an axis whose entire informative region lies beyond its
last tick is doing less work than it could, in the figure the builder split in two precisely for legibility.

## O2 — One source-hygiene check cannot fail for one of its five subjects

`verify-timeseries-sources.py:403–410` loops over five verifiers asserting each contains `--no-evidence` and
writes a provisional `"passed": False`. The list includes **`scripts/verify-timeseries-sources.py` itself**, and
both search literals appear in the assertion's own source — `"--no-evidence"` at line 407 and `'"passed": False'`
at line 409. For that one file the substring test is satisfied by the check, not by the behaviour.

The behaviour is genuinely present (the flag is handled at line 123, the provisional record written at line
214), so nothing is actually wrong. But the guard proves nothing for one of its five subjects, and deleting
lines 123 and 214 would not make it fire. A path-anchored read or an exclusion of self would fix it. This is the
"further inert guard" the brief asks for; it is mild, and I found no inert guard with a defect behind it.

## O3 — Two CSS checks read the raw stylesheet, not the comment-stripped copy

`verify-timeseries-sources.py:368` and `:371` test `css`, while the bare-`svg` scan immediately above correctly
tests `css_without_comments`. A commented-out `svg.ts-diagram { … height: auto … }` would satisfy both. The
consequence is small — the check that actually protects KaTeX's radicals from a bare `svg` selector is the one
that *does* strip comments, and the browser suite measures both radical dimensions at all three widths — but it
is inconsistent with the file's own discipline two lines earlier.

## O4 — Three floors sit loose or exactly on today's value

`parsed >= 12` (line 416) is a floor exactly at the current count, so it can only detect a shrink, never assert
a target. `checks >= 90` (line 420) runs against an actual 113 — 23 of slack, and it is evaluated before its own
increment, so it compares 111. `models.count("assert") >= 200` (line 378) is a bare substring count over the
models verifier's text, so the word "assertion" in a comment counts toward it; today's value is 289. None of
these is wrong; all three are weaker than they read.

## O5 — "An unpaired absence assertion cannot be expressed" is stronger than the code enforces

`verify-timeseries-browser.cjs:357–375`. The two helpers `pinned` and `pinnedThroughTwoGates` do exactly what is
claimed — each asserts absence *and* presence before pushing, so a pin looking for text the page never renders
throws rather than passing silently. All 22 pins go through them, and falsification case 29 proves the helper
fires. That is real protection and it caught two genuinely inert pins.

But `pins` is an ordinary `const pins = []` in the enclosing function scope, so `pins.push({ … })` remains
writable from anywhere in that scope. The accurate claim is "every pin here is written through a helper that
asserts presence in the same call, and no unpaired pin exists"; `design.md:280–282` and the browser evidence
both say "cannot be expressed". Closing over the array and exposing only the two helpers would make the record
literally true.

## O6 — The delay sweep is exhaustive in delay and sampled in origin

`verify-timeseries-models.mjs:663–679` sweeps delays 0–3 × origins `[364, 400, 616, 721]` × horizons 1–7 = 112.
The record describes this as "the invariant under every admitted reporting delay", which is true of the delays
and not of the origins. I swept all 52 origins × 7 horizons × 7 delays × 3 windows and found no violation, so
**the sampling conceals nothing** — but the 728-fit sweep beside it *is* exhaustive over origins, and the
asymmetry is worth naming rather than leaving to a reader to discover.

## O7 — Three printed per-horizon MAEs are exact ties, rounded half-away-from-zero

In exact arithmetic, final naive h = 3 is `14001/8` = 1750.125, naive h = 6 is `9381/8` = 1172.625, and seasonal
h = 2 is `13589/8` = 1698.625. All three are exact ties at the second decimal. The page prints 1750.13, 1172.63
and 1698.63 — JavaScript's `toFixed`, which rounds halves away from zero. NumPy's `round`, which the packet's
own program would use, is round-half-to-even and would print 1750.12, 1172.62 and 1698.62.

Both are correct renderings and the page is self-consistent across all 42 printed per-horizon values. I record
it only because a reader reproducing §5 with `np.round` will see three last-digit differences and may think
something is wrong. Nothing is.

## O8 — Zero-clipping never binds, and the drift baseline is the only rule that could need it

§5 declares that negative ridge and drift predictions are clipped to zero and calls this "part of the
predeclared rule". I ran the full study with clipping disabled: **every number is identical**. The rule is
correct and honestly declared, and clipping *is* reachable through investigation 1's "A falling history, where
drift clips at zero" preset — so it is exercised in the lab even though the real series never triggers it.
Worth knowing that the published figures do not depend on it.

## O9 — A duplicated comment block in the model layer

`timeseries-models.js:931–941`: the paragraph beginning "Three marked rows plus an axis, each with its own
band…" appears twice in `arrivalGeometry`, the second copy extended. Cosmetic, no behavioural effect.

## O10 — What the suite does not assert

Having reproduced every count, the more useful question is what remains outside them.

- **No offline verifier sees the picture.** The render verifier executes the React tree but stubs CSS to empty
  (`verify-timeseries-render.cjs:82`), so by construction it cannot see a colour, a collapsed SVG, an overflow
  or a paint-order occlusion. Its own `limitations` block says so. B1 lives exactly in that gap, and so would
  any future recurrence.
- **The browser suite exercises the opening state plus its 14 case states**, not every reachable control
  combination — the record says this plainly and it is the right disclosure. The model suite's whole-grid
  sweeps cover the arithmetic those states would show, which is the correct division of labour. My own probe
  extended the arithmetic side considerably (A3); the *rendered* side remains sampled, and figure 3's arrow
  styling (S1) is a defect in that gap.
- **The falsification report is replaced unconditionally** by whichever run finishes last, so the evidence on
  disk records a mode rather than a maximum. The ordering rule — `--browser` last, never an offline-only run
  after it — is operational discipline with nothing enforcing it. Combined with S3, the report is the least
  self-protecting artifact in an otherwise careful suite.
- **The JSX parse guard is genuine and covers what it claims.** I confirmed the filter at line 160 selects
  exactly 12 files from the 17 declared, that the blueprint is among them (it is `FILES[4]`, ends in `.js`, and
  is included with an explicit comment saying why), and that an empty target set is an explicit failure rather
  than a vacuous pass (lines 161–163). The verifier's own output names all 12. This guard is sound.

---

# Verification re-run, as found

All five offline verifiers re-run with `--no-evidence`; nothing on disk was rewritten.

| Verifier | Claimed in `design.md` | Printed on re-run |
| --- | --- | --- |
| models | 8,882 grouped checks in 70 groups | **8,882 / 70** ✓ (plus 728 fits over 54 schedules, 112 eligibility points, 112 grader points, 648 donor points, 108 request setups, 282 baseline combinations) |
| data | 6,647 of 6,659 leaves derived, 12 declarations | **6,647 / 6,659 / 12, none uncompared, 616 fits, 34 named checks, module byte-identical** ✓ |
| examples | 71 oracles | **71**, 3 programs executed, author program reproducing its frozen results byte for byte ✓ |
| sources | "111 checks over 17 files" (line 342) | **113 over 17 files, 12 JS/JSX parsed** — see **S2** |
| render | 35 checks | **35**, 240,326 bytes of markup, 2 KaTeX radicals, no verdict/feedback/reveal on first paint ✓ |
| browser (evidence, not re-run) | 14 cases, 35 captures, 22 paired pins | evidence records **14 / 35 / 22**; **35 `timeseries-*` captures on disk** ✓ |
| falsification (evidence, not re-run) | 32 of 32, 6 browser guards | evidence records `breakagesApplied: 32, guardsThatFired: 32, browserCasesIncluded: 6, allRestoredExactly: true` ✓ |

Other record claims confirmed: **43 evidence artifacts** (8 JSON + 35 captures) ✓; **2 radicals at every width,
smallest 21 × 17 px** ✓ — identical measurements at 320, 390 and 1366, so the count check cannot go inert;
`timeseries-kill-test.json` exists and documents a real forced-kill method rather than asserting crash safety
from reading the code.

**Harness safety, verified directly.** The evidence-restore filter (`falsify-timeseries.mjs:455–470`) is
anchored to `startsWith('timeseries-')` on both `docs/teaching/evidence/*.json` and
`docs/teaching/evidence/screenshots/*`, so it cannot read, write or delete another lesson's artifacts — the
sibling's 709-capture near-miss cannot recur here. The sidecar is written and flushed **before** each mutation
(line 564) and removed only after a SHA-256-verified restore (lines 604–615). A stale sidecar or a stale lock
causes a **refusal to start**, not a guess.

I tested the single-instance protection live: with `scratch/timeseries-falsification.lock` present the harness
**refuses to start** with an accurate message, and — correctly — **does not delete the lock it is complaining
about**, because the throw precedes handler registration. I removed my test lock myself and confirmed it was
gone. This is the right design for Windows, where a forced kill delivers no catchable signal and the sidecar is
the only real protection.

---

# Summary judgement

The mathematics is, as far as three independent routes and 3.2 million assertions can establish, **right**.
Every number in the manuscript, the model layer, the generated modules and the rendered page reproduced from
the served CSV under exact rational arithmetic or 50-digit precision, with no disagreement anywhere. The leak
property — the reason this topic exists — holds not just at the 728 fits and 112 delay points the builder
audits but at every one of the 7,644 configurations I could construct, under a tighter bound than the module
asserts. The availability-set route is a genuinely independent decision procedure rather than a restatement.
The trust-root mechanism is real, its count of 6,659 is exactly right, its 12 declarations are precisely the
free choices and nothing more, and its two exclusions are honest. The graded investigations hold their
contract: nothing is revealed before a prediction, verdicts retire on edit, grading is against the committed
draft, and I could not reproduce any of the five sibling defect variants — the exact tie at season length 1
grades **correct**, the empty eligible set grades **correct**, tolerances are exact on integer answers, and the
per-horizon winner names **every** rule attaining the minimum rather than `winners[0]`.

What the suite still cannot see is the picture, and that is where both substantive findings live. **B1** is a
legend and a caption promising an encoding that opaque rectangles painted afterwards have entirely removed —
invisible to every attribute check, diagnosed only by sampling the canvas, and made sharper by the fact that
the identical helper works correctly in figure 4 and that the code comment above the failing call warns about
precisely this hazard. **S1** is the unfixed half of a defect the builder did fix: figure 3 still styles each
lane's first arrow by the lane rather than by the input it carries, in the one figure whose subject is which
arrow carries what, while the model layer sits on the correct per-arrow answer that nothing reads.

The remaining findings are smaller: a record line that contradicts its own document and the verifier
(**S2**), a harness that leaves a stale green report if it dies (**S3**), and an internal token in a
learner-facing verdict (**S4**). The observations are mostly about guards that are weaker than they read rather
than guards that are wrong; I found no inert guard with an actual defect hiding behind it, and the two the
builder found and fixed were real.

This is careful work. The two defects I am reporting are both in the same place — the rendered image — and both
are of the kind the builder's own record says are invisible to everything except opening the picture and
looking at it. That remains the binding constraint on this effort.
