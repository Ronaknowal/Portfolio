> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Time-series validation: content design and continuation

Stable ID: time-series-validation-forecasting-baselines. Classical ML position38; authorized batch position20. Root author, 12 September 2026. Research/write revision1 only. The central phase ledger owns current status and hashes.

## Preflight, scope and source conservation

Read the topic preflight and returned notes, the current teaching/domain/design/code policies, and full bespoke planned entry in src/learn/data/curriculum/cross-domain-expansion.js at baseline8c5da59f18516be77c29d5aeeafca3decca4f738. This is a planned topic with no existing published JSX body. No own destination-note file existed. Its plan requires origin/horizon/cutoff, causal lags, rolling windows, horizon errors, drift/overlap, naive/seasonal baseline and a time-split exercise. All are developed in the manuscript. Runtime, blueprint, catalogue, sequence and publication remain unchanged.

Retain the title and stable identity. The useful expansion is time-of-knowledge semantics from the preceding problem-formulation topic, direct/recursive/updated forecast distinctions, locked rolling-refit assessment, known calendar versus future observed weather, and overlapping tasks versus independent observations. These are central to this topic, not title-bloating tangents. No catalogue-wide audit was performed.

Readiness is core §§1–6 and practices1–6; deeper§7/practices7–8 do not become entry gates. The next actual topic is End-to-End Supervised Learning & Error Analysis, whose independent-row protocol answers a different task; the bridge says so. The advanced ARIMA/GARCH owner receives a [destination note](../../topic-notes/arima-garch-classical-time-series.md) to preserve the forecast contract while extending temporal modeling; this request does not authorize implementing it.

## Teaching design

| Learner hurdle | Explanation and original example | Visual / learner evidence |
| --- | --- | --- |
| Target date mistaken for issue date | Same Saturday target forecast from two origins | F1 two clocks; core contract |
| A baseline treated as a meaningless constant | Four rules on the same six-value operating cycle | I1 donor-cell dependencies and source-ID prediction |
| Timestamp alone assumed to prove availability | Count delay changes historical feature snapshot and label maturity separately | F2 and I2 eligibility timeline; practice1 |
| Shift/index convention changes silently | Origin-indexed versus target-indexed rolling means | §3 example and missing-day practice6 |
| Updated one-step score called a multi-step forecast | Fixed origin, moving origin, and invalid future-input arrow | F3 with equal-error and changed-error fixtures; practice3 |
| Every final test assumed to freeze one model | Rehearse a declared weekly refit policy with protected procedure selection | F4 staircase and actual code |
| An aggregate winner presumed best at every horizon | Actual seasonal wins h4/h5 and required naive/seasonal equality h7 | F5 measured chart/table; practice5 |
| A plausible seasonal story replaces observation | Opposite local weekly rankings on real counts | I3 actual requests with committed predictions, meaningful count edits and a future-edit null |
| More scored cells mistaken for more independent outcomes | Multiple forecasts of one target and overlapping sums | Deeper§7, practice7 |
| Units/target functional ignored in a metric | Exact MAE/RMSE and scaled denominator exercises | Core/deeper practice4/8 |

Three investigations use different operations, not a lab quota. F3's two static traces teach its boundary adequately without an extra interaction. Application connections are mechanism-driven: maintenance cycles, fixed battery schedules versus replanning, capacity versus recorded rentals and delayed reporting. Exact artificial cycles and counterfactual edits are never labeled as real measurements.

## Canonical-reference section-list audit

Canonical reference: Hyndman and Athanasopoulos, Forecasting: Principles and Practice3, Chapter5. The actual chapter navigation was retrieved and parsed on12 September2026 because the text retriever's toolbox page exposed only a short body. These are its complete section titles, not an invented generic outline:

| Canonical section | Disposition |
| --- | --- |
|5.1 A tidy forecasting workflow | Core§§1/4/5 reproduce task→fit→forecast→assess in original Python and origin-level evidence; R/fable syntax is optional alternate learning |
|5.2 Some simple forecasting methods | Core§2 develops all four rules with original mechanism/contrast; real§5 compares them |
|5.3 Fitted values and residuals | Core forecast-error definition; deeper§7 distinguishes fitted residuals from issued horizon errors |
|5.4 Residual diagnostics | Deeper§7 gives useful interpretation and limits; full autocorrelation tests/model-specific diagnostics routed to ARIMA/GARCH owner, with a saved note |
|5.5 Distributional forecasts and prediction intervals | Deeper§7 distinguishes point/marginal/path claims; annotated primary chapter for model-specific formulas; calibration assumptions linked conceptually to its existing owner |
|5.6 Forecasting using transformations | Deeper§7 teaches inverse-transform mean/median distinction with exact lognormal conditions and origin-safe fitting; detailed Box–Cox fitting owned by preparation/time-series modeling |
|5.7 Forecasting with decomposition | Deeper§7 explains retrospective versus origin-available decomposition; full decomposition/ETS/ARIMA fitting routed with destination note |
|5.8 Evaluating point forecast accuracy | Core MAE/RMSE/horizon aggregation, deeper units/MASE/zero denominator/MAPE limitations and changed exercises |
|5.9 Evaluating distributional forecast accuracy | Optional deeper resource; current packet makes no fitted probabilistic claim. Full quantile/interval/CRPS comparative temporal assessment is routed to time-series owner, not falsely completed by citing it |
|5.10 Time series cross-validation | Core rolling origins, explicit selection boundary, expanding/sliding, exact dates and fit schedule |
|5.11 Exercises | Eight original changed questions with optional hints/solutions, not copied book exercises |
|5.12 Further reading | Annotated book/Python/original-data alternatives, with inspected extent below |

This does not attempt to compress the whole forecasting textbook into a foundations-validation topic. The map records developed coverage and specific advanced ownership. No useful original lesson section was removed; no published body existed here.

## Research record actually inspected

- [FPP3 simple methods](https://otexts.com/fpp3/simple-methods.html): substantive full definitions, season indexing and drift, plus book examples. Source figures/data/results are not reproduced; the operating cycle and daily-bike experiment are original calculations.
- [FPP3 temporal cross-validation](https://otexts.com/fpp3/tscv.html): rolling-origin explanation, multi-step evaluation and code structure. Generic statements about errors growing with horizon are not turned into a universal monotonic law; actual weekday-linked curves are retained.
- [FPP3 point accuracy](https://otexts.com/fpp3/accuracy.html): actual error, scale-dependent/percentage/scaled definitions and training-denominator distinction. The lesson gives original changed arithmetic and avoids treating a below-one scaled error as a guaranteed future-baseline win.
- [FPP3 prediction intervals](https://otexts.com/fpp3/prediction-intervals.html): distributions, one/multi-step interval assumptions, benchmark formulas and residual-bootstrap construction read. No interval is claimed fitted here and the book's informal “uncorrelated” bootstrap sufficiency wording is not promoted into a universal independence theorem.
- [FPP3 distributional accuracy](https://otexts.com/fpp3/distaccuracy.html): quantile scores, Winkler intervals, CRPS and relative skill sections inspected. Used for optional onward study and explicit scope routing, not copied numerical results.
- [FPP3 transformations](https://otexts.com/fpp3/ftransformations.html): reverse-transform and interval sections plus actual bias-adjustment heading retrieved. The body after that heading was not exposed by the web text retriever, so full-section reading is not claimed. The lognormal mean/median statement is a standard exact identity with its distributional assumption, not a reported experiment.
- [scikit-learn lagged features](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html): substantive shift/rolling, missing-prefix, random versus forward evaluation, fit schedule, loss/quantile sections inspected. It is an alternate next-hour tutorial, not evidence that feeding observed within-block lags is valid for a block issued at one origin. No source runtime/timing/percentage results are reused.
- TimeSeriesSplit gap definition checked against primary sklearn API search excerpts: excluded sample count before the first test row. Search surfaced several versioned/dev pages; this lesson's exact t−g−1 arithmetic is stated as an explicit indexing convention, not a newly executed current-API behavior test. Real calculation uses directly constructed origins.
- [UCI source](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset): current citation, license, daily/hourly member distinction and variable definitions; complete original Readme retained/read. The actual731-row daily member is counted/checked by the author program.

No video is falsely claimed watched; substantial alternate book and executable Python resources satisfy the intended learning alternatives without a medium quota. No benchmark from an external tutorial is substituted for the lesson's observed result.

## Author calculations and self-review

Full numerical provenance and limitations are in [data-provenance.md](data-provenance.md). The complete author experiment executes616 small deterministic model fits, all reported baseline/horizon comparisons and bounded exact fixtures, preserving per-origin predictions. A subsequent toy-only edit records the changed-recursion continuation without rerunning unchanged model results.

The author caught a temporal consistency issue during writing: adding a reporting delay must also move every historical row's feature snapshot backward, not merely delay its label. Manuscript§3/F2/I2 now state both boundaries explicitly. A second useful correction keeps the equal initial recursive/updated MAE, then supplies a changed continuation with contrasting errors; no score improvement is required to establish a protocol violation. The root numerical result also prevents a generic smooth-increasing horizon illustration: actual curves are nonmonotone and horizon is confounded with weekday in this weekly schedule.

The author read the complete final manuscript, specifications, design and provenance in ordinary order. The final pass introduced MAE before its first numeric use, made the per-horizon winner emphasis consistent, and replaced an ambiguous ordered list of lab MAEs with explicit rule-to-value labels. All six offered toy periods were calculated through eight horizons; source dependencies and period-one naive equality are retained. Unchanged real fits were not repeated for these text/fixture additions.

Learning-experience findings:

1. Route: immediate first-pass path and labeled deeper branches; core readiness uses only taught core mechanisms.
2. Cautions: data limitations live with the real protocol; temporal legality has one developed explanation reused by later cases. Code prints numeric outcomes, not warning prose. Distinct fixed/updated and selection/assessment issues remain where their mechanisms are introduced.
3. Real question: Saturday staffing/rental planning returns as a licensed daily-series comparison, with actual unfavorable weeks and horizon-specific exceptions retained.
4. Investigations: donor identities, complete eligible sets and a real forecast request are different tasks; predictions are input-bound and checked. Toy periods1–6, real periods1/7/14, delay/empty-set cases, changed future and specified nulls have exact calculations or explicit finite arithmetic. Arbitrary implemented interactions remain unverified.
5. Figures: contracts specify baseline presence, unhidden maxima, weekday labels, exact equal points and text/mobile forms. Physical legibility cannot pass before rendering; phase two must inspect it.
6. Connections: the reporting-delay snapshot explicitly extends the previous lesson. The canonical section list has a disposition for every branch; advanced temporal model/interval work has a saved real destination.
7. Code: the complete main file and short displayed loop expose origin, eligibility, features, fit and prediction directly. Verification bookkeeping remains in author evidence, with no browser model-fitting campaign.
8. Practice: changed histories, delays, dates, errors and application claims provide independent transfer. Questions2/4/8 give exact new numeric results.
9. Screenshots: intentionally not created in research/write mode; informative implemented states, mobile visibility and checked-prediction feedback belong to phase two.

The inline-visual reading pass confirmed that the concept's actual objects remain visible: issue dates, observed donor days, legal labels, prediction-fed versus observation-fed arrows and measured horizon errors. F3 uses a focused static explanation; no extra lab was added merely for uniformity. Formal independent review, rendered visual quality, arbitrary runtime edits and exact displayed-program verification remain deferred.

## Phase-two continuation

Read [lesson.md](lesson.md), [visual-specifications.md](visual-specifications.md), provenance and the complete author program before implementation. Build topic-owned figures, three investigations and relevant models/assets with lazy loading; do not load this616-fit author experiment in a browser. Execute displayed programs in their stated setup and implement the small direct arithmetic from the contract. Preserve the reporting-delay feature boundary, period wrap, future-edit invariance, exact h7 baseline equality, empty eligibility state and distinct fixed/rolling update policies.

Complete formal independent content/visual/model review, necessary fixes, responsive/accessibility/browser checks, source-bound evidence and integration before marking implementation complete or publishing. Retain the pending packet and real input. The current research/write request does not authorize those actions.

## Phase two, part A: implementation record (20 September 2026)

Appended by the phase-two implementer. Everything above this heading is the frozen
research/write checkpoint and is unchanged. Browser verification is deliberately absent:
the lesson is not registered in the publication manifest, so its page is not reachable,
and that work is phase C.

### What was built

| Concern | Source |
| --- | --- |
| Every number, and every drawn coordinate | `src/learn/data/timeseries-models.js` |
| Recorded measurements and the served calendar | `src/learn/data/timeseries-data.js`, generated |
| Executed programs and their real output | `src/learn/data/timeseries-examples.js`, generated |
| Shared controls, the graded-prediction contract, the single SVG wrapper | `TimeSeriesShared.jsx` |
| F1–F5 | `TimeSeriesFigures.jsx` |
| I1–I3 | `TimeSeriesLabs.jsx` |
| Styles | `timeseries-labs.css` |
| Reader body | `src/learn/data/topics/time-series-validation-forecasting-baselines.jsx` |
| Authored plan | `blueprints/time-series-validation-forecasting-baselines.js` (not yet registered) |
| Served data, attribution and the author program | `public/learn-assets/time-series-validation/` |

The manifest entry and the blueprint registration remain with the integration owner and were
not touched.

### The one property the model layer exists to protect

Time-series validation exists because ordinary cross-validation leaks the future, so every
object representing a fit carries its own information set and `informationSetAudit` refuses
any fit touching an observation dated later than its own origin. Three claims are kept apart
because they fail separately: **feature causality** (a row issued at origin s reads indices no
later than s − d), **label maturity** (a row enters a fit at origin t only when s + h + d ≤ t),
and **fit closure** (the largest index any fit at t touches is at most t). Closure follows from
the other two, which is exactly why it is asserted in its own right: it is the claim a reader
can check against a drawing, and a drawing that satisfies it cannot be teaching the opposite of
the lesson. Every drawn split boundary is a model coordinate; none is placed by hand.

### Departures from the specification, with reasons

1. **Two displayed programs were added.** The manuscript displays one runnable program, in §5,
   which left this topic's two central mechanisms — the four baseline rules and the eligibility
   boundary — with no executable form. `baseline-rules` (§2) and `eligible-rows` (§4) are
   byte-exact slices of `forecast-experiments.py`, located by their `def` lines and pinned by
   SHA-256, with composed import and printing lines counted separately in the evidence. The
   second one prints, for each horizon, the training range and the latest day that fit reads,
   and asserts that boundary as it runs — the lesson's own invariant in executable form. No
   manuscript text was changed.

2. **Investigation 1's four-rule table, donor strip and substituted index sentence now wait for
   a recorded prediction.** As first built they rendered on first paint, which put both graded
   answers — the donor position and the forecast value — on screen beside the question. The
   common contract forbids revealing a correct answer "before the specified reveal", so only the
   learner's own inputs and the rule stated in general terms are visible before Apply.

3. **Investigation 2's timeline and per-row table likewise wait for a recorded set.** The
   timeline colours each row by its verdict and the table prints the inequality for every origin;
   both are the answer.

4. **Investigation 3 marks the copied block only after the forecast is issued.** The
   specification asks the history chart to show "the exact period-length block copied". Marking
   it before issuing points at the observations the rule is about to copy, which is the substance
   of the numeric commitment. The block is marked at the moment the forecast is drawn, where the
   requirement does its teaching work; before that the chart is the history a forecaster has.

5. **The investigation-3 future list is four editable outcomes rather than up to eight.** The
   specification permits at most eight; four keeps the unscored-horizon case reachable whenever
   the horizon count is raised above four, which is the behaviour the contract actually names.

Departures 2–4 are tightenings of the reveal boundary, not reductions in what the lesson shows.

### Numbers checked, and one thing to know about them

Every packet figure was recomputed independently and **agreed**. No disagreement with the packet
was found, so nothing was recorded as a disagreement and nothing was edited. Specifically:
the six development MAEs and RMSEs; the three final MAEs and their horizon curves; the toy
fixtures including all six season lengths through eight horizons; the four eligibility sets; the
recursive/updated pair at MAE 10 and the changed continuation at 2 against 0.5; the nine real
seasonal replays; the future-edit null; and the history edit dropping the MAE by exactly 1000/7.
`forecast-experiments.py` reproduces `calculated-inputs.json` byte for byte in the declared
runtime.

The ridge predictions were re-derived by a genuinely different algorithm — standardization
written from its definition and the normal equations solved by LU, rather than sklearn's pipeline
with an SVD solver. The worst relative gap across all 616 predictions is 7 × 10⁻¹⁵.

### Verification, and what it does not cover

Five verifiers plus a falsification harness, all runnable from a clean checkout, each writing a
provisional failing record before it starts and stamping success only after its final assertion.
Trust-root coverage is measured as asserted leaf paths: 6,647 of 6,659 leaves independently
derived, 12 validated as declarations, none uncompared. Two things are **not** covered at all and
are stated in the evidence: the provider archive's own digest, because the zip was deliberately
not retained; and that the retained CSV is what the provider served on the recorded date, which
is a claim about a download rather than a computable property.

The falsification harness was proved against an actual forced kill, recorded in
`docs/teaching/evidence/timeseries-kill-test.json`.

Not covered here: anything about the rendered page. Legibility, layout at 320 px, keyboard
reachability, painted colour and the KaTeX radical measurement are phase C, and the browser
verifier is written and waiting rather than run.

## Phase two, part C: browser verification record (20 September 2026)

The lesson is registered, so its page is reachable. Built to `dist-ts` and previewed on
`127.0.0.1:4196 --strictPort`; the browser verifier ran at 1366, 390 and 320 px in Edge.
**Every one of the 35 captures was opened and read.** That is where most of what follows
came from: the offline suite was green throughout and saw none of it.

### A sixth verifier, promoted from a scratch probe

`scripts/verify-timeseries-render.cjs` executes the React tree in Node and asserts the
first-paint contract. It exists because a sibling lesson in this effort **did not render at
all** — its body threw, the error boundary replaced the page, and three offline verifiers
stayed green, because the model layer, the recorded data and the source text were each
perfectly correct. Nothing that reads those artifacts can see that failure.

It earned its place twice during this phase: it caught a JSX comment written into an
attribute position, and it refused to leave a green record when it did.

### What reading the screenshots found

Nine defects, none of which any offline check could see:

1. **Figure 4's drawing was illegible.** On one linear axis from day 6 to day 371, the
   entire informative region — the per-horizon boundary at t − h, the issue day, the seven
   targets — was the last 2% of the width. The boundary a reader is meant to read off this
   figure could not be told from the issue mark. It is now **two panels**: an overview in
   absolute days whose only job is the left edge (fixed when expanding, moving when
   sliding), and a detail panel on days *relative to each rehearsal's own origin*, where
   the boundary sits at −h and the targets at +1…+7. Rehearsals not yet advanced to are
   drawn as dashed outlines rather than left as empty space. The models verifier now
   asserts that panel A carries **no** boundary or target marks, and that panel B magnifies
   the boundary-to-issue gap by at least three times.
2. **Figure 2 printed a false inequality.** The relational operator was hard-coded, so the
   label row rendered `13 ≤ 12` beside the answer "No" — a false statement given as the
   reason for a correct verdict. It is now computed per row.
3. **Figure 2's "features" label sat on the feature squares.**
4. **Figure 3 marked a legitimate arrow as invalid.** The third lane passed `kind: 'invalid'`
   for the whole lane, so its first arrow — which reads the origin's own observation and is
   perfectly legal — was drawn red and dashed. The figure accused an arrow that is not at
   fault, which is the opposite of what it exists to show.
5. **Figure 3 miscounted.** It reported the audit's *violation* count as a count of late
   inputs: "4 of its inputs", when three are late and the fourth violation is the fit's own
   closure.
6. **Figure 5 and section 5 printed an invented history length.** "copies history position
   400 of 400" — true of a history this experiment never has. Now the first final origin's
   real length, 617.
7. **Four tables clipped a column at desktop.** The shared `Table` now takes a `wrap` prop
   naming its prose columns; values stay unwrapped so a number never breaks across lines.
8. **Investigation 3's verdict printed a raw double**, `1042.5714285714287`, where every
   other readout on the page prints six decimals.
9. **Three label collisions and seven labels painted past their own viewBox edge**, plus a
   preset that left a replacement value from a different week beside the count it replaced.

### Checks added because a defect got past the existing ones

Each of these exists because reading a screenshot found something no check was looking for:

- **Labels against their own SVG edge.** The shared inspector has a check of this name but
  computes boxes through `getBBox`; it did not see labels overflowing by one or two pixels,
  which are exactly the ones a real page clips. Measured here with `getBoundingClientRect`
  on both, at every width. 144 labels measured.
- **Labels over point marks.** The curve sampler reads `path`, `polyline` and `polygon`; the
  shared inspector reads straight `line` elements. Neither looks at a `rect` or a `circle`,
  and this lesson draws its target, feature, arrival and issue days as exactly those. An
  axis title landed squarely on a row of target rects and every geometry check stayed green.
  100 marks checked.
- **Table columns clipped at desktop**, and the same three checks re-run **in the revealed
  state**, because the widest table on the page exists only after investigation 3 reveals
  its outcomes and the first pass ran with that investigation reset.
- **The shared layout inspector's candidates are now asserted** against a reviewed
  allow-list rather than merely recorded. The list is empty: every candidate it reported was
  a real defect and was fixed.

### Two guards that were passing on nothing

Both found by making a property structural rather than by noticing them:

- **An absence assertion with no presence partner cannot fail usefully.** Two pins in this
  lesson's browser verifier were looking for text the page never renders anywhere — one
  appended a word the page does not print beside that number, one named an `aria-label`
  attribute. Both absence assertions passed on every run and proved nothing: they could not
  distinguish "correctly withheld" from "looking in the wrong place". Pins are now written
  through a helper that asserts presence in the same call, so an unpaired absence cannot be
  expressed. **22 paired pins**, seven of them through two gates.
- **A KaTeX radical check must measure both dimensions and pin its own count.** A collapsed
  radical is also narrow. Both dimensions are floored at 4 px at all three widths, and the
  count at each narrow width must equal the desktop count, so the check cannot go inert by
  measuring an empty set. Measured: 2 radicals at every width, smallest 17 × 21 px.

### What the rendered page confirms about the lesson's own claim

No drawn split boundary contradicts the audited contract. Figure 4's detail panel draws the
boundary at −h and its table's last column reads "Latest day read = the issue origin" for
every horizon, matching the 728 audited fits. The informative failures are shown, not
cropped: figure 5's table names seasonal naive as the lowest at horizons 4 and 5, names
*both* tying rules at horizon 7, and the development-to-final degradation from 772.33 to
1167.62 is stated in the prose. Investigation 3's outcome table shows 2036 against 2036 at
horizon 7 — the same identity, on real data, inside the lab.

### A gap none of the three lessons' offline suites could see

Midway through this phase the shared `npx vite build` began failing for every lesson at
once, on an unescaped apostrophe inside a single-quoted string in a sibling lesson's
component file. **No offline verifier in any of the three lessons saw it**, and the reason
generalises: model and data verifiers import plain `.js`, source-hygiene scans read the text
without parsing it, and nothing else opens the `.jsx`. A JSX syntax error surfaces only in a
production build or a browser run — which is exactly when it is most expensive.

This suite now parses every JS and JSX file it owns, the blueprint included. The blueprint
matters most: nothing imports it at runtime, so a syntax error there would have reached the
integration owner rather than this suite. The guard cannot go inert, because a file that does
not parse cannot be made to look as though it does. It is falsified by a breakage that
reproduces the original defect exactly — an unescaped apostrophe in a single-quoted string —
and it fires. 12 files parsed; 113 hygiene checks in total.

The same episode is a reminder about the harness rather than about the lesson: two of its
browser cases died on a build that a concurrent agent had broken mid-run, and on both
occasions the `.orig` sidecar restored the mutated file and the run refused to continue with
a breakage applied. That is the second and third time the crash-safety machinery has been
exercised for real in this topic, after the deliberate `taskkill /F` in phase A.

### Final falsification result

**32 of 32 breakages make their intended guard fire and name itself**, six of them browser
guards rebuilt and run against the live preview:

| Browser breakage | Guard that fired |
| --- | --- |
| Investigation 1 renders its reveal before a prediction is committed | the leak property, pinned by the graded quantity's own text |
| Investigation 3 reveals the outcomes at the moment the forecast is issued | the second gate |
| A pinned phrase stops being rendered | the paired-pin helper, which refuses an absence assertion with no presence partner |
| The SVG layout rule is unscoped again | the KaTeX radical measurement, in both dimensions |
| A stylesheet paints a calendar band black | painted-against-declared computed styles |
| The staircase inset shrinks | every SVG label box measured against its own SVG box |

Every file was restored byte for byte and all 43 evidence artifacts were restored. The
captures were re-verified afterwards: 35 recorded, 35 on disk, every digest and byte count
matching, all 35 digests distinct, no orphans.

### Status and limits

Green: models (8,882 grouped checks in 70 groups), data (6,647 of 6,659 leaves derived),
examples (71 oracles), sources (113 checks over 17 files), render (35 checks), browser
(14 cases, 35 captures, 22 paired pins). All 35 captures match their recorded digests and
sizes, none duplicated, no orphans.

*These are the counts as at the end of phase C. The independent review found this line
saying 111 where the verifier and the evidence both said 113, and several counts rose again
during the review dispositions below; the figures that supersede them are in the disposition
section's own table, which is the one to trust.*

Not established here: that a first-time learner can follow any of this. Every judgement
above is an author's reading of rendered pixels and text, not an observed learner. The
browser verifier exercises the opening state and the states its cases reach; it is not a
proof that every reachable combination of controls renders correctly, which is what the
whole-grid sweeps in the model suite cover for the arithmetic those states would show.

## Phase two, part D: disposition of the independent review (21 September 2026)

The review is `docs/teaching/TIME-SERIES-INDEPENDENT-REVIEW.md`. It reproduced every published
number by an independent route, swept the leak property over 7,644 configurations, and could not
reproduce any of five defects found in sibling lessons. It named five defects here and ten
observations. Four of the five defects were repaired in code and one in this document; of the ten
observations, seven produced code changes, three are recorded without change and are argued below.
Two further defects were found by this pass and are recorded at the end — neither was in the
review, and both were surfaced by the checks rather than by reading.

Nothing in the frozen packet was edited. No number in it was found wrong.

### Counts, re-measured after every repair

This table supersedes the Status line above, which was written before the review.

| Verifier | What it now reports |
| --- | --- |
| `verify-timeseries-models.mjs` | 10,240 grouped checks in 72 groups; 728 fits audited with no observation later than its own origin over 54 schedules; 112 eligibility grid points by an availability-set route; 112 grader grid points; 648 donor grid points; 108 request setups; 282 baseline-rule combinations |
| `verify-timeseries-data.py` | 6,647 of 6,659 packet leaves independently re-derived and 12 validated as declarations, leaving none uncompared; 616 fits audited; 34 named checks; module byte-identical |
| `verify-timeseries-examples.py` | 3 displayed programs executed, 1 lifted whole from the manuscript; the whole author program run, reproducing its frozen results file byte for byte; 71 oracle assertions |
| `verify-timeseries-sources.py` | 113 source-hygiene checks over 17 files, and all 12 declared JS/JSX files parse |
| `verify-timeseries-render.cjs` | 35 checks over 245,204 bytes of rendered markup; no verdict, numeric feedback or reveal on first paint |
| `verify-timeseries-browser.cjs` | 14 cases at 1366, 390 and 320 px; 35 captures; 22 paired pins; 22 legend encodings; 24 arrows checked against their own input kind; 2 KaTeX radicals measured in both dimensions at every width, identical at all three |
| `falsify-timeseries.mjs` | 35 of 35 breakages made the intended guard fire and name itself — 27 offline and 8 browser — every file restored byte for byte |

The model count rose from 8,882 in 70 groups to 10,240 in 72 groups, and the falsification suite
from 27 cases to 35, because the repairs below each brought their own guard and their own breakage.

### The five named defects

**B1 — figure 1's legend named an encoding the picture did not contain.** Thirty-one hatch lines
were in the DOM with every attribute correct, emitted *before* the opaque `.ts-cell` rects that
were then painted over them. SVG has no z-index; document order is paint order. The fix is the
order: band, then cells, then hatch — which figure 4's identical helper already used. The hatch
stroke was also lightened from `#4a5650` to `#7d8c85`, measured at 5.00:1 against the calendar
ground and 3.32:1 against the training band.

The repair that matters is not the reorder but the check. `legendEncodingsPainted` now requires,
for every figure and at every width, that each encoding the legend *names* is realized by an
element with at least one point on itself that no later opaque shape covers. Occlusion is decided
geometrically in the SVG's own coordinate space. The first draft used `elementsFromPoint`, which
takes viewport coordinates — and this reader scrolls an inner container, not the window, so every
sample point for a figure below the fold landed outside the viewport and the check reported every
encoding in every figure as covered. A check that fails on everything is exactly as useless as one
that fails on nothing, and the tempting repair is to loosen it until it goes quiet.

**S1 — figure 3 styled each lane's first arrow as carrying an input it does not carry.** Fixed as
the property the review asked for, not as the instance. `ARROW_STYLE_BY_INPUT_KIND` and
`arrowStyleFor` live in the verified model layer and `demand` an entry for every input kind; the
figure derives every arrow's style through that table and emits `data-input-kind` alongside; and
`arrowStylesMatchTheirInput` asserts, for all 24 arrows, that each arrow's drawn class is the one
its own recorded input kind maps to. A new legend entry and caption say what the first arrows mean.

**S2 — this document said 111 source-hygiene checks where the verifier and the evidence said 113.**
Corrected, and the Status line now points here rather than carrying a number that will drift again.

**S3 — a crashed falsification run left the previous report on disk, still reading green.** The
harness now writes a provisional `passed: false` record carrying an explicit "in progress" status
immediately after it acquires its lock, and excludes its own report from the evidence snapshot it
restores, so a run that dies cannot leave a finished-looking record behind. This was observed
live while the final run of this pass was in flight: the report on disk read `passed: false` with
the in-progress status for the whole run.

**S4 — investigation 2's verdict printed the internal token `none-qualify`.** `Prediction` gained a
`labelFor` prop and investigation 2 supplies one, so the verdict now reads "no offered origin
qualifies" and pluralizes multi-origin sets. Confirmed in the re-captured
`timeseries-eligibility-empty-set-desktop.png`.

### The ten observations

**O1 — figure 4's overview axis carried only two ticks in expanding mode. Changed.** The tick list
now always ends at the domain's high endpoint when the last chosen tick falls more than 8% of the
span short of it, so the labelled region reaches the bars.

**O2 — one hygiene check could not fail for one of its five subjects. Changed.** The check over the
five verifiers' `--no-evidence` and provisional-record behaviour reads `verify-timeseries-sources.py`
with its own assertion region cut out, delimited by explicit sentinels; a missing sentinel is
itself a failure, so the exclusion cannot silently stop excluding.

**O3 — two CSS checks read the raw stylesheet, not the comment-stripped copy. Changed.** Both now
read `css_without_comments`, as the bare-`svg` scan two lines above already did, so a commented-out
rule can no longer satisfy them.

**O4 — three floors sat loose or exactly on today's value. Changed.** The parse floor became an
exact equality derived from the declared file list rather than a number typed in the check, so it
notices a file added and never parsed as well as a shrink. The `checks` floor is read into a
variable before its own increment so it compares the number it names. The models-assertion count
became a regex over actual call sites rather than a substring count that the word "assertion" in a
comment could satisfy.

**O5 — "an unpaired absence assertion cannot be expressed" was stronger than the code enforced.
Changed.** The pin array is closed over inside an IIFE that exposes only the two gate helpers and a
reader returning a copy, so the claim is now literally true rather than a description of current
practice.

**O6 — the delay sweep was exhaustive in delay and sampled in origin. Changed.** The reviewer swept
all 52 origins independently and found no violation, so the sampling concealed nothing — but the
728-fit sweep beside it *is* exhaustive over origins, and a record reading "the invariant under
every admitted reporting delay" should not be carried by four sampled origins. The sweep now runs
every contract origin: 4 delays × 52 origins × 7 horizons, and the label is literally true. This is
most of the rise in the model count.

**O7 — three printed per-horizon MAEs are exact ties, rounded half-away-from-zero. Recorded, not
changed.** 14001/8, 9381/8 and 13589/8 are exact ties at the second decimal. JavaScript's `toFixed`
rounds halves away from zero and NumPy's `round` rounds half to even, so a reader reproducing §5
with `np.round` will see 1750.12, 1172.62 and 1698.62 where the page prints 1750.13, 1172.63 and
1698.63. Both are correct renderings, the page is self-consistent across all 42 printed values, and
changing it would mean the page disagreeing with its own JavaScript. Named here so the difference
is not mistaken for an error.

**O8 — zero-clipping never binds on the real series. Recorded, not changed.** The reviewer ran the
full study with clipping disabled and every number was identical. The rule is correct and
predeclared, and it *is* reachable through investigation 1's falling-history preset, so it is
exercised where a learner can see it. Removing a predeclared rule because the published data never
triggers it would make the declaration less honest, not more.

**O9 — a duplicated comment block in the model layer. Changed.** Removed.

**O10 — what the suite does not assert.** Three of its four bullets are accepted as written and are
already stated in the limitations blocks of the records concerned: no offline verifier sees the
picture (the render verifier stubs CSS to empty by construction), the browser suite exercises the
opening state plus its 14 case states rather than every reachable control combination, and the JSX
parse guard is sound.

The fourth — that the falsification report is replaced unconditionally by whichever run finishes
last, so the file records a mode rather than a maximum — is **recorded, not changed**. The report
names its own `mode` and its `browserCasesDeferred` count in every run, so an offline-only record
cannot be mistaken for a fuller one by anyone who reads it; the ordering discipline (offline first,
`--browser` last) is stated here and was followed for this pass. A refusal to overwrite a
browser-mode record would break that very workflow at its first step, and a second report file
would create two artifacts that can disagree. The honest cost is named instead: **the report
describes the run that wrote it, not the best run ever performed.**

### Per-guard evidence that each repair now fails when it should

Each line is one breakage from `falsify-timeseries.mjs`, applied alone to a clean tree, with the
tree rebuilt and the verifier run against the live preview. A case counts as fired only when the
verifier exits non-zero **and** its output names the guard, so a run that fails for an unrelated
reason earns no credit.

| Repair | Breakage | Guard that fired |
| --- | --- | --- |
| B1 | figure 1 paints an opaque cell over the hatching its legend promises | `a figure legend names an encoding the painted picture does not contain` |
| S1, drawn | an arrow is drawn in a style that does not match the input it carries | `an arrow is drawn in a style that does not match the input it carries` |
| S1, source | the first arrow of a lane stops having its own style, as it did before the review | `every lane's first arrow carries the origin's own observed count` — the property in the model layer, above the figure |
| O5 / leak | a pinned phrase stops being rendered, so its absence assertion would pass on nothing | the paired-pin helper, which refuses an absence assertion that has no presence partner |
| O4 | a JSX file stops parsing, which no text scan and no JS import can see | the parse check over every owned JS and JSX file, including the blueprint, which nothing imports at runtime |
| O4 | the model suite stops counting what it checked | the floors under the grouped-check and group counts, which stop a suite that has stopped checking from still printing PASS |
| KaTeX | the SVG layout rule is unscoped again, collapsing every KaTeX radical to nothing | the measurement of every `.katex .sqrt svg` in **both** dimensions, at all three widths, with the narrow-width count required to equal the desktop count |

Two of these cases were reported **INERT** by the harness on the first attempt, and both reports
were correct:

- The first draft of the B1 breakage inserted a *second*, covered hatch and left the real one in
  place. The picture still contained visible hatching, so the guard was right to stay silent. A
  breakage that does not break the property proves nothing about the guard, and the harness said so
  rather than crediting it.
- The first draft of the S1 breakage forced every arrow to one style, which also emptied two of
  figure 3's legend encodings — so the *legend* check fired first and the per-arrow check was never
  reached. The case was credited to no guard because the guard that named itself was not the one
  under test. The breakage now permutes two input kinds instead of collapsing them, leaving all
  three styles present in the figure; the legend check stays silent and only the per-arrow check
  can see the defect. Verified directly: with the permutation applied, the legend guard does not
  fire.

### Two defects this pass found that the review did not

**The harness reported a mutation it had applied as not applied.** `occurrencesReplaced` was
computed as occurrences-before minus occurrences-after, which is zero for any breakage that *wraps*
the text it finds rather than deleting it — and reads exactly like the stale find-string failure it
exists to catch, which had already bitten this harness four times. It is now reported as
`occurrencesPresent`, `occurrencesRemaining` and `mutationApplied`, and a breakage that leaves the
file byte-identical is a hard error rather than a line in a table.

**The legend check measured hollow marks at their hole.** Sample points were taken at an element's
centre. For a hollow closed shape — figure 5's open coincidence ring — that asked two wrong
questions: whether the *hole* was covered, so a ring with something drawn through its middle would
have been reported as painted over while its outline was fully visible; and what the stroke's
contrast was against whatever sat inside the hole, which recorded the cream ring at 1.75:1 against
the green marker it *encircles*, a comparison no reader makes and a number that would have read as
a legibility finding. Hollow closed shapes are now sampled on their outline. `line` and `path`
keep the centre sample deliberately: a diagonal hatch line does not touch its own bounding-box
corners, so outline sampling would measure points its stroke never reaches.

### What is still not established

Unchanged from the Status section above: that a first-time learner can follow any of this. Every
judgement about the rendered page is an author's reading of pixels and text, not an observed
learner. The browser suite exercises the opening state and the states its 14 cases reach; the
whole-grid sweeps in the model suite cover the arithmetic those states would show, which is the
division of labour, not a proof that every reachable combination of controls renders correctly.
