# Forecasting data and numerical provenance

Prepared 12 September 2026 for research/write only. The CSV and calculations are required offline inputs for the pending lesson, not disposable scratch or deployed browser assets.

## Observations and rights

[UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Hadi Fanaee-T (2013), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894), is licensed CC BY 4.0 by the repository. The current dataset/license/DOI page and complete retained provider Readme were inspected. Data concern recorded Capital Bikeshare rentals in Washington, DC, in 2011–2012. The provider's historical statements about worldwide program counts are not current statistics and are not used in the lesson.

Downloaded [provider archive](https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip) on 12 September 2026. Archive SHA-256 b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401. Retained only unchanged daily member day.csv, renamed [bike-sharing-daily.csv](bike-sharing-daily.csv), and the unchanged provider [description](source-description.txt); the unused hourly dataset and zip archive are not retained.

Daily file: 57,569 bytes; SHA-256 a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be. [data-source.json](data-source.json) preserves retrieval metadata. There are 731 distinct consecutive daily dates, 1 January 2011–31 December 2012; the author code verifies this member rather than relying on the website's mixed hourly/daily headline count. It also verifies casual+registered=cnt on every row.

Original fields are preserved. The experiment uses only cnt and dates. The target is recorded rentals, not unconstrained latent demand or required per-station inventory. The file supplies no actual report-arrival timestamps, station identifiers, archived weather forecasts or proof of stable future conditions. Day-end count availability is an explicit replay assumption. No cause is assigned to an unusual date merely from its count.

## Declared procedure and actual computation

[forecast-experiments.py](forecast-experiments.py) ran in the existing read-only lesson runtime: Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1. Serial deterministic SVD ridge; no shared environment changes or downloaded modeling services.

Zero-based origins are364,371,…,721, each forecasting indices origin+1 through origin+7. The first36 origins are development; final16 assess the locked selected procedure plus predeclared naive/seasonal baselines. The last two source dates are not scored because no complete additional seven-day block remains.

The six development candidates and fourteen direct-model features were fixed before calculating this comparison. Ridge alpha1 and zero-clipping for ridge/drift are explicit. At each horizon h and issue origin t, training origin IDs are6 through t−h inclusive; sliding mode takes the latest90 of them. Every horizon fit learns its own scaler and coefficients from those rows. Target-date weekdays, approximate annual sine/cosine and elapsed calendar time are known deterministic inputs. Future observed counts do not enter a row's historical features. There are504 development ridge fits (36×7×2).

Select expanding ridge by development MAE772.3323587; 90-row ridge MAE883.8012182. The final replay performs112 selected ridge fits (16×7), allowing new outcomes into subsequent scheduled fits only after their arrival. It does not score alternative ridge windows or use the final table for candidate reselection. Final MAE1167.6228091, RMSE1654.8896621; naive MAE1310.3392857, seasonal1390.9107143. All per-origin forecasts and per-horizon summaries are retained in [calculated-inputs.json](calculated-inputs.json).

This is a predeclared rolling update policy, not a frozen-model test and not an untouched final-period dataset that never enters later fitting. Forecast records are never revised retroactively. No superiority claim outside this dataset/protocol or confidence interval from112 independent observations is made.

## Exact fixtures and scope of evidence

Constructed history[10,20,10,20,12,22], period2 and four supplied future outcomes expose baseline dependencies. Exact seasonal output[12,22,12,22], mean15.666667, naive22, drift[24.4,26.8,29.2,31.6] and their errors are calculated. Editing the fifth history value changes seasonal and mean but not naive/drift. The recursive/updated example intentionally has equal MAE10 on its initial continuation; the changed continuation produces MAE2 versus.5 without changing the task-validity rule. These are exact scenarios, not observed rental histories.

The label-arrival fixtures enumerate the stated inequality. The main real code uses delay0. The delayed-count investigation is separately specified: historical features at issue s must stop at s−delay; it must not reuse no-delay y_s features while only changing the label cutoff. Current code records the eligibility sets, while the full delayed-time UI and arbitrary edits remain phase two.

Actual development-origin seasonal replays cover periods1,7,14 at origins364,371,476. Their differences and the future-edit null are calculated and saved. The history-edit contrast754→1754 at index358 changes only the first period7 forecast at origin364; its MAE change is exactly1000/7 under the declared unchanged future. No re-fitting is needed to verify this dependency.

The full author script was executed once for all model results. Subsequent toy-only evaluations added the changed recursive continuation and all six seasonal periods through eight horizons, updating only that JSON subtree and preserving all recorded model results. The equivalent baseline loop's values are therefore calculated, but its displayed block has not undergone a separate formal verbatim-program verification campaign. Phase two owns that check plus formal independent review, production implementation, UI behavior, responsive/accessibility inspection and publication.

No hardware-speed benchmark, fabricated timing curve, external performance number or unsourced confidence band is included. Preserve these offline inputs until the pending implementation has incorporated them.
