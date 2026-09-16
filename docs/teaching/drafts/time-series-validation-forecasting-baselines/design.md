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
