# Titans teaching packet: real data and execution provenance

Prepared 13 September 2026. These files are required content-phase evidence and offline teaching inputs, not disposable scratch.

[UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Hadi Fanaee-T (2013), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894), provides observations from Capital Bikeshare in Washington, DC, in 2011–2012. The provider page's dataset description, variable list, DOI and CC BY 4.0 license were inspected on 13 September 2026. The complete retained provider description was also read. Cite Fanaee-T and Gama, *Event labeling combining ensemble detectors and background knowledge*, 2013, DOI10.1007/s13748-013-0040-3, as requested in the original description. Historical counts of global bike-sharing programs are not used as current facts.

The daily CSV was already downloaded for the earlier forecasting packet on 12 September 2026 from the [provider archive](https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip). This packet reuses exactly those verified daily bytes; it does not claim a fresh download. Archive SHA256: b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401. Member day.csv, retained here as [bike-sharing-daily.csv](bike-sharing-daily.csv): 57,569 bytes, SHA256 a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be. No CSV values were modified. The unchanged [source description](source-description.txt) accompanies it. The original download record remains [in the forecasting packet](../time-series-validation-forecasting-baselines/data-source.json).

731 distinct consecutive dates are verified from 1 January 2011 through 31 December 2012; every row satisfies casual+registered=cnt. The lesson uses dates and cnt only. UCI's headline mixed/hourly count is not the row count of this retained daily member. No station identities or report-arrival timestamps are present. Day-end availability is an explicit replay assumption; recorded rentals are not unconstrained latent demand or a per-station inventory requirement. Temperature normalization differences between old Readme and current variable descriptions do not affect this experiment because no weather field is used.

## Declared question and calculation

The [protocol](experiment-protocol.md) was written before this experiment ran. The stream is appropriate for studying the forecast→observation→write boundary, and lets a learner compare actual online adaptation against a frozen copy and naive forecasts. It is not a reuse of the earlier forecasting model, and it is not a Titans benchmark reproduction.

[rental_memory_study.py](rental_memory_study.py) uses 358 training examples whose target rows are7–364 inclusive, with normalization fitted only on counts0–364. The fixed 9→8→1 SiLU network is trained for1000 full-batch Adam steps per seed3,7,19, then copied. Predict-before-write replay covers rows365–730, with a reporting boundary before548. Adaptive state carries across that boundary; no settings are chosen from the resulting reports. Update rate .005, retention .5 and decay .0001 are fixed. All predictions are retained without clipping. Initial parameters, final parameters and final momentum make the replay independent of a refit if software initialization changes later.

The observed environment is Python3.12.14, NumPy2.3.5, PyTorch2.14.0+cpu, float64, CPU, one thread. No dependency was installed or changed. The first full study execution completed under the declared protocol. No hyperparameter search or rerun for a better score occurred. [rental-results.json](rental-results.json) contains all three seeds, both windows, all individual forecasts, prewrite losses, gradients and update norms. Replaying those saved initial weights is a correctness check, not a new random experiment or a new independent dataset.

Matched adaptive assessment MAEs are823.539613,867.338835,805.499626 versus frozen1767.045292,1781.812565,1813.871956 for seeds3,7,19. The previous-day assessment MAE is843.814208; hence seed7 does not beat that baseline. The manuscript keeps that comparison and all seeds. Different seeds are not independent draws of rental histories, so their spread is not a confidence interval for future deployment performance.

## Evidence types and reproduction

- [memory_mechanisms.py](memory_mechanisms.py) and [mechanism-results.json](mechanism-results.json): exact constructed linear recurrences and payload arithmetic; floating-point softmax/gate trace; nonlinear outer derivative compared with finite differences. The gated block is a declared identity-projection MAG-topology specialization, not a published checkpoint.
- [neural_memory.py](neural_memory.py): complete functional SiLU memory read/write, initialization and copying. One scalar output per key, mean half-squared loss for a batch, local graph detachment at inference, optional differentiable write for outer derivatives.
- [check_author_packet.py](check_author_packet.py) and [author-checks.json](author-checks.json): independent arithmetic, null/contrast conditions, source/split/feature checks, all-seed metric aggregation, saved-weight replay, zero-update equality, full-state continuation, request isolation and future-outcome perturbation. Numerical comparison tolerances are explicit in the checker.

Run from this directory with an environment supplying the recorded packages:

```sh
python -B memory_mechanisms.py
python -B rental_memory_study.py
python -B check_author_packet.py
```

The exact repository author invocations used `scratch/lesson-tools/Scripts/python.exe -B` followed by each packet-relative script path. The shared runtime is retained; this work produced no temporary downloaded media, scratch images or bytecode. Browser controls, responsive diagrams, accessibility, formal independent review and website integration remain phase two.
