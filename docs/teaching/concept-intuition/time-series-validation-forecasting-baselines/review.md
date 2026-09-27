# Forecasting validation: conceptual-transition review

26 September 2026. Full original JSX read (~800 lines): all seven sections, eight exercises, provenance and alternatives. Preserved all recorded bicycle results, programs, original figures and three live labs. Source-verifier files inspect the page; it is not regenerated from another prose source. Author-only status pending independent/browser/integration.

| Location and concepts | Disposition |
| --- | --- |
| §1 issue/target/horizon, error, nowcast/reconstruction | Retained explicit clock table and issue-time figure. |
| §2 mean/naive/seasonal/drift, donor index, multi-cycle, baseline error | Retained four-rule calculation, runnable implementation and live donor selection. |
| §3 row origin, label maturity, delay, gap indexing | Retained arrival diagram, eligibility lab and derived inequality including feature availability. |
| §3 rolling windows, shifts, calendar/weather knowledge, missing dates | Retained exact indexing distinctions and source-version warning. |
| §3 recursive versus updated forecasting | Retained complete traces and continuation figure. |
| §4 rolling/expanding/sliding/refit, locked policy, independence | Retained staircase and audited source-index examples. |
| §4 error aggregation | Gap fixed with exact two-horizon RMSE counterexample: average 2 versus pooled √8; avoids prose-only warning. |
| §5 real experiment, features, per-horizon fit, period/horizon tie, counterfactual edits | Retained all measurements, source rows, interpretation and request lab; no fits rerun. |
| §6 practice | All eight questions/solutions retained. |
| §7 direct/recursive/joint | Gap fixed with scalar perturbation propagation for feedback .8 versus 1.2; new arrival table relates architecture outputs to training eligibility and explains a partially mature vector. |
| §7 overlapping targets/dependence | Retained three distinct mechanisms and justified-inference conditions. |
| §7 daily versus path uncertainty | Gap fixed with native matrices: same 9/10 per-column coverage, 3/10 versus 9/10 full-row coverage; never passed off as fitted or independent. |
| §7 residuals/conformal | Retained assumptions, distinction from forecast errors, link to owner. |
| §7 scaled metrics/transformation/decomposition/drift | Scaled denominator, zero scale and causal smoothing already explicit. Gap fixed for back-transform with equally likely 1 and 9 (mean 5 versus exp mean-log 3), and distinction from lognormal adjustment. |
| Next/resources/provenance | Retained route, all programs and dataset; provenance separates new constructed arithmetic from retained live fixtures. |

## Research and visual decisions

Read FPP3's [prediction intervals](https://otexts.com/fpp3/prediction-intervals.html) and [point accuracy](https://otexts.com/fpp3/accuracy.html) main sections, including horizon-specific uncertainty, simulated paths, loss/summary choice and scaled denominator. Existing source alternatives already give a useful R/Python second route. No code rerun or linked slider/video inspection claimed. The new coverage construction specifically fills the pointwise-to-joint gap; it does not reproduce the textbook's curves or imply its residual assumptions apply here.

`TimeSeriesIntuitionFigures.jsx` uses two semantic tables that stack on phones. Dot and cross communicate status without color; both axes are explicit path/day identifiers. Arithmetic checks cover matrices, recursive differences, maturity, pooling and transform examples. Existing numerical/lab/figure hashes match the baseline. No synthetic fit, performance benchmark or new randomized experiment introduced.

Final copy-only follow-through: Replaced stale say-the-answer-first instructions in the topic-specific live lab with direct manipulation/observation wording. The changed lab is now included in source identity; its controls and calculations are unchanged.
