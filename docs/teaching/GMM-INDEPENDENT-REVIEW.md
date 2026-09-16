# Gaussian Mixture Models & EM — independent phase-two review

Reviewed 13 September 2026 against the working tree plus the uncommitted GMM implementation. Reviewer did not author the packet or the implementation. No file under `src/`, `public/`, `scripts/` or `docs/teaching/drafts/` was edited; this review document is the only write. No state-changing git command was run.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under the session scratchpad, written from the manuscript's own definitions and importing neither `gmm-models.js` nor any `verify-gmm-*` script nor `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1). One-dimensional and analytic work used **mpmath at 50 decimal places**; the Mahalanobis forms and the M-step moments used exact `fractions.Fraction`. They recomputed the whole §2 density/responsibility table, the complete first EM cycle, the trace, the changed-D cycle, the identical-start null, the log-sum-exp fixture, the four collapse values, the plane geometry and eigenpairs, the parameter totals, the AIC/BIC example, the full Q/entropy/ELBO decomposition, the k-means limit values, the conditional mixture, and practices 1–8. Separately, the **entire Iris pipeline was refitted from the served CSV**: the split, the scaler by hand at `ddof=0`, all 16 candidates with log-densities recomputed from the fitted parameters via `scipy.stats.multivariate_normal` + `logsumexp`, BIC/AIC from independently derived parameter counts, ARI from an independently written contingency table, the reserved-test outcome, the narrow component, and the three Dirichlet-process fits. The served CSV was compared byte-for-byte against the packet copy, against `sklearn.datasets.load_iris`, and against UCI's `iris.data` and `bezdekIris.data`. All three displayed programs were extracted with `node`, written to the filenames the lesson names, and executed beside the served CSV. `sha256sum` was run on every reviewed file. |
| **Read in full** | `lesson.md` (769 lines), `visual-specifications.md`, `data-provenance.md`, `checked-results.json` (structure and the `bound`, `em`, `iris` blocks), `GMM-LESSON-DESIGN.md` including its phase-two append, the published `.jsx` body, `GmmLabs.jsx`, `GmmShared.jsx`, `GmmFigures.jsx`, `gmm-labs.css`, `gmm-models.js`, `gmm-iris-data.js`, `gmm-examples.js`, `RunnableExample.jsx` and its diff, the blueprint and its registration, the four evidence JSONs, the GMM rows of `lesson-delivery-progress.json`, the GMM topic note and its diff, and the GMM paragraphs of `LESSON-AUTHORING-HANDOFF.md` and `AGENTS.md`. |
| **Looked at** | Nine of the fifteen evidence screenshots were opened and inspected: `collapse-desktop`, `covariance-gallery-desktop`, `allocation-desktop`, `bound-chain-desktop`, `candidates-desktop`, `candidates-mobile`, `responsibility-desktop`, `em-floor-desktop`, `covariance-desktop`, `covariance-mobile`, `em-step-mobile`, `selector-desktop`. Findings S5, S6, S7, S9, S12 and observations O1 and O17 come from that pass. |
| **Reused, not rerun** | `evidence/gmm-models.json` (80 grouped checks), `gmm-native.json`, `gmm-iris-data.json` and `gmm-browser.json` (10 Playwright cases). Every source hash they record equals the hash below for the same file, so the recorded results bind to the bytes reviewed here. The production build, the Playwright run and `verify-curriculum.mjs` were not repeated. |
| **Delegated** | Every external URL in the Sources block was fetched and checked against the sentence that cites it, including Bishop's PRML PDF (table of contents and body pages extracted rather than trusted to a summariser), both Stanford PDFs, the SEE lecture page and transcript, both DOIs, four scikit-learn pages and the UCI record's raw API payload. Findings S10 and observations O8, O9, O18 come from that pass. |
| **Not done** | No production build, no browser session, no keyboard pass, no 200 % zoom check, no reduced-motion check: every interaction finding below was established by reading the components and the screenshots, not by operating the page. No beginner walkthrough. Six screenshots were not opened. The library-version sensitivity of the Iris fits was not tested on any version other than the pinned one. |

## Source versions reviewed (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/gaussian-mixture-models-gmm-em-algorithm.jsx` | `90da6e19060532fe10e0b480213639d964478d4f7638bbdfbff74354f50b20a8` |
| `src/learn/data/gmm-models.js` | `fad049a3b67173781cd12905075f8aceb042f2e606f4d425607164d781f508e0` |
| `src/learn/data/gmm-iris-data.js` | `6fdb40c8cea22f86d225046a97148ede264ab6f06537a5bcce0cd47d33e7b0ae` |
| `src/learn/data/gmm-examples.js` | `1e30a7ca4a712c54185e35ae31cb7906ffc31ecabb19e0754fd94803d2b881c9` |
| `src/learn/components/lesson-labs/GmmShared.jsx` | `59c24b114a73830500a9de2107a557341f29245bf960d3cb23ed6d9e450fec66` |
| `src/learn/components/lesson-labs/GmmLabs.jsx` | `97e16bcc27508fdd4b6edea48c8431a49d7ca060bc4eaf7f3aa6fbff7c86b473` |
| `src/learn/components/lesson-labs/GmmFigures.jsx` | `52264985d333ac32baef75dc51a93f7471f60ed4e5e84e47b5f681c0ba8c9faa` |
| `src/learn/components/lesson-labs/gmm-labs.css` | `2f6b3b0ef9bb94dcb1dc6687ed97aaf8b0c869d7c3024f033c552e881efb372b` |
| `src/learn/data/curriculum/blueprints/gaussian-mixture-models-gmm-em-algorithm.js` | `374e2c6817992c0e9aadbe496e03ab25c83d41181b214c8702788b6a4ebfb60d` |
| `public/learn-assets/gmm/iris.csv` | `c6fd24e7f41dd55405cbc30f344e648c2f31eedfb24f9dcaba6290998bc26eb7` |
| `docs/teaching/drafts/.../lesson.md` | `f5af9c95de0c3fd5066e3944fa92280b2b321e31c4bc3ae3e970df168e4d76d6` |
| `docs/teaching/drafts/.../visual-specifications.md` | `1d96bb1392b3f49f6f83aea42687ef3516f7d4e7d408d5a8de5a963fc9f6a85f` |
| `docs/teaching/drafts/.../checked-results.json` | `c3f18e2bac5636b3de21fdf4e6bf5e1e77d0e4bd16fe5178e215a7d499e267c4` |
| `docs/teaching/drafts/.../data-provenance.md` | `9fe73d6ce056afe6fa940795fbc71681b291be2acf2b868b584ebba8fc6d5818` |
| `docs/teaching/drafts/.../author-calculations.py` | `8ee73509089edf90732be8b6a30558691de02879f50222579094f9f244b78204` |
| `docs/teaching/drafts/.../iris.csv` | `c6fd24e7f41dd55405cbc30f344e648c2f31eedfb24f9dcaba6290998bc26eb7` |
| `docs/teaching/GMM-LESSON-DESIGN.md` | `f0381ef89b8e5dfb69c82628186639f4c068759742f2eeb0302c5a653c50e80d` |
| `scripts/verify-gmm-models.mjs` (read only) | `9a12a5ae3ab36b03200f084e8b6cff4c2a8b3a4f221b02b7d764dbe686a4ec05` |
| `scripts/verify-gmm-examples.py` (hash only) | `ac276be104049577490804912f530f11874f88246f0c4ee08177b6a57c3aeef1` |
| `scripts/verify-gmm-iris-data.py` (hash only) | `a08e19778c981f5cd0782b9f2fca18845bba3ad5eb40e2e153fe6c4829d10659` |
| `scripts/verify-gmm-browser.cjs` (hash only) | `efb583872cad0ea43305fccdb5375ba688069fd3a2f1545b52231ef7ce721905` |

A hash proves identity, not correctness. The three evidence files that record source hashes (`gmm-models.json`, `gmm-browser.json`, `gmm-native.json`) name exactly these bytes, so their recorded results apply to the version reviewed here. The served `public/learn-assets/gmm/iris.csv` is byte-identical to the packet's `iris.csv`, and both are the file the evidence names.

## Part A — correctness

### A1. Manuscript sections, claims, cautions and tasks preserved?

All twelve manuscript sections are present in the published body, in order, with the declared first-pass route (`.jsx` L40), the three-kind covariance table, both deeper branches, all three programs in the manuscript's order, practices 1–8 with separate closed hint and solution disclosures, and a readiness-check table.

The four honesty caveats named in the brief were checked sentence by sentence and are all present:

- **Density is not probability** — the §1 `Callout` (`.jsx` L42–44) is a genuine single home ("once for the whole lesson") and carries height-per-unit versus area, probability zero at a point, and density above 1. It is not repeated as a warning later; §2's table and §8's score discussion use it as a mechanism rather than restating it.
- **A responsibility is not a category probability** — the same Callout ("neither verified species probabilities nor an uncertainty distribution over the fitted parameters"), the responsibility lab's own note (`GmmLabs.jsx` L45, "Neither is a species probability"), §7's row-34 paragraph ("The rounding hides a small second responsibility; it is not an uncertainty calculation about species identity", `.jsx` L168), and the closing paragraph.
- **BIC does not identify a true K** — `.jsx` L152, verbatim in substance; and the blueprint's misconception list carries it too.
- **The reserved test showed the baseline ahead** — `.jsx` L167, stated with the number, the direction and the refusal to re-choose a rule: "the selected mixture scores **0.123975 nats per row lower** than the one-Gaussian baseline: −3.011223 against −2.887249 … We keep the declared rule and report the outcome." Repeated honestly in practice 5 and in the design record's closing section. Nothing anywhere claims the mixture won.

Also preserved and checked: `spherical` is not k-means (three places, each making a different point); a negative log-determinant is valid; the E-step does not change the observed log-likelihood; `reg_covar` is not the variance-floor maximizer; the 39.35 % unit contour with the explicit rejection of 68 %; "not measured runtimes"; "not a field-sampling study"; "not automatic proof of a biological category count"; and the closing "None of them is a benchmark, a species probability or a claim about any future dataset."

Deviations found, each judged:

| Location | Deviation from manuscript / specification | Judgement |
| --- | --- | --- |
| `.jsx` L90–91 | The manuscript's four-row Left/Right responsibility table is compressed to a prose list of Left shares. | Acceptable: `AllocationFigure` (`GmmFigures.jsx` L93–96) carries the full two-column table one element later, and L95's "the table above" then resolves correctly. |
| `.jsx` L95 | The manuscript's "total log-likelihood rises from −7.158186977 to −6.461856301" is not in the prose. | Acceptable: `AllocationFigure`'s closing paragraph (`GmmFigures.jsx` L101) prints both values, computed live. |
| `.jsx` L93 | **Sentence shipped twice, verbatim.** | Finding **B2**. |
| `.jsx` L157–168 (§7) | The manuscript's "The author calculation run used Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1; numerical fitting can vary across versions. All 16 candidates below converged in that run" is **absent from the whole published body**. | Finding **S2**. |
| `.jsx` L197 area (§9) | The manuscript's "Zero terms are handled by their limiting values where the support permits" is dropped from the Jensen step. | Observation **O20**. |
| §10 Bayesian branch | The manuscript's three-row weight table is replaced by the program's expected output alone. | Observation **O3**: the numbers are on the page, but `gmm-iris-data.js` ships `bayesianSensitivity` for a table that is never rendered. |
| `.jsx` L226–255 | Practice tasks retain every stated answer, with two added lab pointers not in the manuscript. | One of them is wrong: finding **S1**. |
| `.jsx` L298–305 | A readiness-check table not in the manuscript. | Good addition; matches the standard's readiness check and maps each item to where it was taught. |
| Section titles 2, 3 | Reworded. | No consequence. |
| `RunnableExample.jsx` | Modified to print `Save as <file>` — a change to a component every lesson shares. | Observation **O4**; an improvement, but unrecorded. |
| `GMM-LESSON-DESIGN.md` "Departures" | Lists four departures; at least nine exist. | Finding **S11**. |
| Ledger / handoff / `AGENTS.md` | Assert a state that does not exist. | Finding **B1**. |

### A2. Independent recomputation (reviewer's own scripts, no lesson code)

Every numeric claim in the manuscript and in the published body was recomputed from first principles. **Nothing disagreed.** The table below is the full list, abbreviated where a family of values was checked together.

| Claim | Independent result | Agrees? |
| --- | --- | --- |
| §2 weighted contributions at x = 2: `0.000066915`, `0.199471140`, sum `0.199538055`, B responsibility `0.999664650` | `0.00006691511288`, `0.199471140201`, `0.199538055314`, `0.99966464987` | Yes, every digit |
| The three-row table: densities `0.053990967 / 0.199538055 / 3.0379414e−9`, B responsibilities `0.5 / 0.999664650 / 1−1.27e−14`, negative log-densities `2.918939 / 1.611750 / 19.612086` | all reproduced; the exact complement at 8 is `1.26641655e−14` | Yes |
| Quick-try: A weight 0.2 at x = 0 gives `(0.2, 0.8)` with density unchanged at `0.053990967` | identical | Yes |
| `e⁴/(1+e⁴) = 0.982013790` and the full four-row responsibility matrix | identical to 12 s.f. | Yes |
| Left weighted sum `−2.689649316`, effective count exactly 2, second moment exactly `2.5` | `−2.68964931611`, `2.0`, `5/2` exactly | Yes |
| **First M-step `±1.344824658`, variance `0.691446639`** | `±1.34482465805`, `0.691446639091` by both routes (second moment minus squared mean, and scatter around the new mean) | Yes |
| **EM trace `−7.158186977`, `−6.461856301`, `−5.724277515`**, plateau `−5.675741839` | `−7.15818697714`, `−6.4618563013`, `−5.72427751492`, `−5.67574183…` | Yes |
| Next-E-step Left responsibility for A rises to `0.999582068` | `0.999582068112` | Yes |
| **Changed D = 3 cycle: `1.844792261`, `0.503878397`, `1.582910448`**, with count `2.015513587` and `1/(1+e⁻⁶) = 0.997527377` | `1.84479226084`, `0.503878396701`, `1.58291044777`, `2.01551358681`, `0.997527376843` | Yes |
| The constrained M-step derivation `−N_k(log v + s_k/v)/2`, maximum at `s_k`, boundary otherwise | algebra confirmed; stationary point at `v = s_k`, and a scan of the objective over the feasible set confirms the boundary case | Yes |
| **Log-sum-exp fixture `a = (−1000, −1001)`: `−999.686738312`, `(0.731058579, 0.268941421)`** | identical | Yes |
| **Identical-start null `−7.508335597`**, all responsibilities 0.5, M-step returns mean 0 / variance 2.5 | `−7.50833559657`, exact fixed point | Yes |
| **Collapse `−8.122121377, −6.945323529, −4.669586147, −2.369725898`** at σ = 1, 0.1, 0.01, 0.001 | identical to nine decimals | Yes |
| **`D²(1,1) = 8/7`, `D²(1,−1) = 8`; densities `0.135882281` and `0.004407103`** | exactly `8/7` and `8` by `Fraction`; `0.135882280689` and `0.00440710274293` | Yes, exact |
| det Σ = `0.4375 = 7/16`; eigenvalues `1.75` and `0.25`; semiaxes `√1.75` and `0.5`; eigenvectors `(1, ±1)/√2` | identical, exact | Yes |
| **Unit contour mass `1 − e^(−1/2) = 39.346934 %`** | `0.3934693403`; the 2-D χ²₂ tail is correct and the 68 % rejection is right | Yes |
| **Parameter totals 17 / 11 / 14 / 11** for d = 2, K = 3 | `9+6+2`, `3+6+2`, `6+6+2`, `3+6+2` | Yes |
| **AIC 310 and 308; BIC 323.025851 and 336.656872**; penalty gap 27.631021 | identical | Yes |
| **Bound decomposition: Q `−8.069044223`, entropy `0.910857246`, ELBO `−7.158186977`, lifted `−6.799547014`**, gap `0.337690713`, `Q_new = −7.710404260` | identical; the touching equality holds to < 1e−40 and the three-term ordering is strict | Yes |
| **k-means limit `0.592667 / 0.817574 / 0.997527`**, midpoint stays `0.5` | `0.592666599954 / 0.817574476194 / 0.997527376843 / 0.5` | Yes |
| Mixture moments: mean 0, variance 5; residual outer product `[[4,−2],[−2,1]]`; `det(0.25I) = 0.0625`; `10·5000²·8 = 2e9` bytes ≈ `1.86 GiB` | identical | Yes |
| Conditional mixture at u = 1: weights `(0.5, 0.5)`, means `0.5` and `3.5`, both variances `0.75`, mixture mean 2, variance 3 | identical | Yes |
| **Practices 1–8**, every stated value: `0.241970725`, `1−log3/2 = 0.450693856`; `13/10, 13/30, 1, 41/13, 28/13`, floored to 3; eigenvalues `0.25`, Cholesky `0.5I`; `4`, `28/3`, `5`, det `3/4`; the restricted K ≤ 2 search; the F/ℓ counterexample; `0.952574127` and `0.999993856`; `0.119202922`, means 0 and 3, conditional mean `0.357608766` | every one reproduced, the exact ones by `Fraction` | Yes |
| **CSV identity** | served CSV byte-identical to the packet copy; 150 rows, IDs 1…150, 50/50/50 species, every cell equal to `load_iris()` in 1.9.1; the provenance file's stated recreation recipe regenerates the file byte-for-byte | Yes |
| **Rows 35 and 38** | versus UCI `iris.data`, exactly three cells differ (r35 petal width 0.2 vs 0.1; r38 sepal width 3.6 vs 3.1 and petal length 1.4 vs 1.5); versus UCI `bezdekIris.data`, identical. UCI's own wording matches the lesson's stated corrections exactly | Yes |
| **Split 90 / 30 / 30** from `default_rng(16).permutation(150)` | sizes and membership identical to `checked-results.json` for all three lists; disjoint, union 1…150 | Yes |
| **Scaler `(5.772222222, 3.080000000)` / `(0.829863145, 0.420898510)`** | `ddof = 0`, fit on train only; hand-computed Z equals `StandardScaler.transform` to 0.0 | Yes |
| **All 16 candidates** — 16 validation scores and 16 training BICs | every printed digit reproduced; independently derived parameter counts `(5,11,17,23 / 5,8,11,14 / 4,9,14,19 / 3,7,11,15)` equal `model._n_parameters()`; hand-computed BIC agrees with sklearn's to 1.1e−13; all 16 `converged_ = True`; iteration counts match `checked-results.json` exactly | Yes |
| Selection: full K = 2 at `−2.668800`, unique; full K = 3 second at `−2.671122`; training BIC minimum at full K = 4 (`461.595`) | identical | Yes |
| **Test `−3.011223` versus baseline `−2.887249`**, difference `0.123975` with the **baseline ahead** | `−3.0112234685906634`, `−2.8872488178783784`, `0.123974651`; the sign and the conclusion are robust (the unregularised single Gaussian still scores `−2.887253`) | Yes |
| **ARI `0.433446`** | `0.4334462099505079` from sklearn and identical to 12 dp from an independently written contingency table | Yes |
| First test ID 34, responsibilities `[1. 0.]`, log-density `−4.861128` | ID 34 (so the `first[0]+1` mapping is genuinely right, not coincidentally right); raw responsibilities `[0.9999999640, 3.599e−08]`; `−4.861127656` | Yes |
| **Narrow component at 1e−4** | full K = 4 component 0 has `cov[1][1] == 1e-4` by exact float equality; weight `0.148476513`; standardised mean `(0.245438856, −0.190069573)`; raw width SD `0.004208985` cm against a 0.1 cm recording resolution; its 14 argmax-assigned training rows all have sepal width **exactly 3.0 cm** | Yes |
| The three Dirichlet-process fits, their six weights each, and counts 3/3/4 and 2/2/2 | reproduced at 4 dp; all three converged | Yes |
| `gmm-iris-data.js` as shipped | all 150 observation rows, the scaler, the 16 candidates, `selection`, `selectedParameters`, `narrowComponent`, the 30 `testRows` and `bayesianSensitivity` reproduce the refit at the published rounding, with max absolute difference 0.000e+00 against `checked-results.json` | Yes |

**Could not verify numerically:** the asymptotic cost statements `O(nKd²+Kd³)` and `O(nKd)` (they are not computed quantities); the reading-time estimates; and version-independence of the Iris fits — every Iris number is bit-exact only in the pinned environment, which is also the environment the author used. I confirmed run-to-run determinism (three repeats, and again under `OMP_NUM_THREADS=1`) but tested no other library version. See S2.

### A3. Displayed programs versus their published output

All three programs were extracted from `gmm-examples.js` with `node`, written to `em_1d.py` and `iris_density.py` beside a copy of the served CSV, and executed. **Every one reproduces its stored `expected` block exactly** — including the 16-line candidate table, `selected full 2`, the test lines, `responsibilities [1. 0.]`, and the scientific-notation formatting of the `0.01` Bayesian row. The appendix was concatenated after `iris_density.py`, exactly as its `file` field ("iris_density.py (appendix)") and the prose instruct, and the shared prefix of the combined run is byte-identical to the standalone run. The only differences are Windows `\r\n` line endings on my run and the absent trailing newline in the stored strings — platform artefacts, not content mismatches.

No program prints an explanatory or cautionary sentence. The only guard is `raise RuntimeError(f"Unfinished fit: {kind}, K={k}")`, which is a real convergence check, not a disclaimer. Displayed code is compact: 38, 42 and 14 lines.

### A4. Model and lab logic read for defects

- `mixtureAt` (`gmm-models.js` L64–88) evaluates in log space with an explicit max shift, so responsibilities survive where both `exp` terms underflow. `checkComponents` rejects weights that do not sum to 1 within 1e−9 and any non-positive variance. The `(−1000, −1001)` claim is a property of this code path, not a lookup.
- `maximization` (L106–120) is the exact constrained maximiser: `Math.max(scatter, floor)`, never `scatter + floor`. It returns `count`, `scatter` and `atFloor` alongside the updated parameters, and refuses a component with `count <= 1e-12` using the specification's exact sentence. I confirmed by scanning the objective that the boundary case really is the maximiser. Two of those three returned fields are then never displayed — see S8.
- `emCycle` (L124–137) keeps `responsibilities` (the matrix the M-step used) separate from `freshResponsibilities` (the one computed afterwards), which is precisely what I2's specification demanded, and `EmStepLab.applyMaximization` preserves it (`GmmLabs.jsx` L183).
- `boundDecomposition` (L174–187) computes Q and H from a *supplied* responsibility matrix, so `BoundChainFigure` can evaluate the new parameters against the **old** q — the fixture the specification warned must not accidentally recompute q. It does not; `GmmFigures.jsx` L322–323 passes `start.responsibilities` to both calls.
- `symmetricEigenpairs` (L225–241) returns two *different* coordinate directions when the cross term is zero, which is what fixes the defect the design record admits (diagonal and spherical families previously drew as straight lines). I rebuilt each gallery matrix from its returned eigenpairs and every one reconstructs.
- `correlationGeometry.massInsideUnitContour` is `1 − e^(−1/2)`, a derivation rather than a stored 0.393469.
- `useInvestigation` (`GmmShared.jsx` L98–123) is the strongest part of the interaction design: `edit` clears both the result and the choice, `check` computes the answer *from the draft at the moment of commitment*, and `describeKey` makes it impossible to show a result beside inputs it was not computed from. `Prediction` starts with no radio selected and disables "Check prediction" until a choice exists.
- Defects found by reading: the stale radio selection and the stranded verdict after "Back one half-step" in `EmStepLab` (S4); `justUpdated.before` and the unclipped `scatter` computed and discarded (S8); the missing free-text explanation field in `Prediction` (S14); `weightedUpdate` exported "for the practice task" and imported by nothing (O2).

## Part B — learning-experience checklist (reviewer's heuristic run)

| # | Item | Finding |
| --- | --- | --- |
| 1 | **Route** | Present and declared: `.gm-route` (`.jsx` L40) names sections 1–8 with the three investigations and two programs, sends 9 and 10 with practices 6–8 to a later sitting, and gives both time bands. Both deeper sections carry "Deeper branch" in their heading. |
| 2 | **Cautions** | The §1 `Callout` is a genuine single home and is not restated as a warning. The density/responsibility distinction is thereafter carried as mechanism (§2's table, §7's row 34, §8's score). Each lab repeats a one-line note, which is the specification's own requirement rather than hedging drift. No code prints a caution. The one real gap is the *missing* version caveat (S2), not a repeated one. |
| 3 | **Real question** | Opens on an overlapping population and returns to it in §7 with 150 real Iris rows, a downloadable CSV, a corrected-snapshot provenance statement, a CC BY 4.0 attribution with a licence link, a split declared before the rule, and a result the reader can judge — including the fact that it went against the mixture. |
| 4 | **Labs as investigations** | **I1 responsibility**: prediction unset, committed with the inputs, retired on any edit; the identical-component null really does return the mixing weights at every x; the two-point synthesis target (both B responsibilities > 0.99, negative log-densities ≥ 5 apart) is real and reachable — the screenshot shows 18.0003. **I2 EM**: four presets each replacing the whole setup, explicit E/M half-steps, a real back step, the identical-start null and the active floor both reachable and both captured; but the prediction state leaks across iterations (S4) and the introductory state hides the allocation (S8). **I3 covariance**: prediction unset, a permanent ρ = 0 null panel, the ordering reversal at negative ρ, the exact 39.3469 % statement, and a screenshot of a *missed* prediction. Its geometry is correct; its type is too small (S7). |
| 5 | **Figures** | F1, F2 and F6 are legible and correct at desktop. F5 is the best figure here — the magnified strip is drawn at its true height, the narrow band is perceptible, and both mobile and desktop are readable. F4's ellipses are geometrically right but carry no visible numbers (S6). F3 does not deliver its own point (S5). No caption apologises for its figure; the F3 caption instead *admits* the thing that is wrong with it. |
| 6 | **Connections** | Strong. The 8-versus-2 contrast is set up in §2, used as the anomaly score in §8, and rehearsed in I1. `0.691446639` recurs across §3, F2 and I2. The unit-contour eigenpairs are explicitly tied to PCA. §10 separates the Lloyd construction from the σ² → 0 limit from the `spherical` API, and practice 7 tests exactly that separation. §6 links clustering evaluation and DBSCAN by route. Canonical facts checked and correct: the mixture density, Bayes-rule responsibility, the three M-step updates with denominator `N_k`, the multivariate density, the four covariance families and their parameter counts, AIC/BIC, `F = Q + H`, the KL gap form, and the Jensen touching equality. |
| 7 | **Code** | Compact and mechanism-dominated: 38, 42 and 14 displayed lines with one convergence guard and no printed disclaimers. The `pip install` pin is a single `bash` block with no path-escaping hazard. |
| 8 | **Practice** | Eight tasks, each changing both numbers and context, hint and solution in separate closed disclosures, every stated value reproduced. Two tasks point into a lab; one of those pointers is wrong (S1). |
| 9 | **Screenshots** | Fifteen captures covering every specification-required informative state: the decisive distant point with its log-density, a matched and a missed prediction, the identical-start null, the active variance floor, the reversed correlation with its ρ = 0 null, every inline figure, and four mobile captures. Gaps: no mobile capture of the responsibility lab, and no capture of the F3 "spike zoom" the specification required — because no zoom panel exists (S5). |

## Ranked actionable findings

Severity: **blocking** = wrong on the published page, or a record that asserts something untrue; **should-fix** = an inaccuracy, a defeated teaching mechanism, a dropped specification item with a bounded fix; **observation** = recorded for the ledger.

### Blocking

**B1. The phase ledger still says implementation has not started, and three separate records assert the opposite — including an independent review that did not exist.**

Locations: `docs/teaching/lesson-delivery-progress.json`, topic entry for `gaussian-mixture-models-gmm-em-algorithm`; `LESSON-AUTHORING-HANDOFF.md` L91, L93, L97; `AGENTS.md` L31.

Evidence:
- The ledger entry reads `"implementation": {"status": "not-started"}` with `nextAction: "… Current content is complete; implementation has not started."` `git diff docs/teaching/lesson-delivery-progress.json` shows the GMM entry was added at the *content* phase (12 September) and contains **no** implementation block and **no** implementation source hashes.
- `LESSON-AUTHORING-HANDOFF.md` L97 states: "The [phase ledger](docs/teaching/lesson-delivery-progress.json) records implementation complete with the final source hashes." That is demonstrably false against the file it links.
- `LESSON-AUTHORING-HANDOFF.md` L93 links `docs/teaching/GMM-INDEPENDENT-REVIEW.md` as "the disposition of the independent review". That file did not exist when this review began; the link was dead and the claim unsupported. `AGENTS.md` L31 makes the same claim for GMM ("their phase-two records, evidence and independent reviews are linked from those designs"), and the GMM topic note's "Implementation/verification links" bullet links it too.
- The ledger's content checkpoint binds `docs/teaching/GMM-LESSON-DESIGN.md` at `0fbfec29c051cd9b43648e83296becee1f646ea7be91250cea79858554496ca8`; the file is now `f0381ef89b8e5dfb69c82628186639f4c068759742f2eeb0302c5a653c50e80d` after the phase-two append.
- `GMM-LESSON-DESIGN.md` contradicts *itself*: its "Current scope and phase" section (L5–18) still reads "**Implementation:** not started", "**Independent phase-two review:** not started", and "No runtime, lesson JSX, blueprint, registry, navigation or publication changes are authorized in this phase" — immediately above a "Phase two: implementation, 13 September 2026" section describing all of those changes.

The standard is explicit that unknown states are recorded as unknown and that an author's own reading is not independent review. Fix: (a) update the ledger entry to `implementation: complete` with the final source hashes and a real next action, and refresh the design-record hash in the content checkpoint; (b) rewrite `GMM-LESSON-DESIGN.md` L5–18 so the phase table matches the appended section, or move it under a dated "at content-phase close" heading; (c) restate handoff L97 so it does not claim a ledger state that is not there. The handoff and `AGENTS.md` may now legitimately cite this document, but only after (a).

**B2. Two classes of duplicated text are shipped in the published body.**

Location 1: `src/learn/data/topics/gaussian-mixture-models-gmm-em-algorithm.jsx` L93 —

```jsx
<Prose>Right is the mirror image, and both weights stay 0.5. Right is the mirror image, and both weights stay 0.5.</Prose>
```

The sentence appears twice, verbatim, inside one paragraph, in the middle of §3's hand-worked EM cycle. The manuscript (`lesson.md` L149) has it once. Fix: delete the second copy.

Location 2: `.jsx` L26 — `Program` renders `<Prose><strong>Before running:</strong> {example.question}</Prose>`, but all three `question` strings already contain the phrase (`gmm-examples.js` L7, L15, L23). The three program blocks therefore render:

- "**Before running:** Before running: the two components start at −1 and +1 with variance 1. …"
- "**Before running:** Validation picks the candidate; the test rows are untouched until the end. Before running: will the selected mixture …"
- "**Before running:** Six components are available to every fit. Before running: does a larger concentration …"

Fix: strip the phrase from the three `question` strings in `gmm-examples.js` (and re-run `scripts/verify-gmm-examples.py`, which hashes the module), or drop the `<strong>` label from `Program`. Note that the `question` fields are not part of the executed-program oracle, so the numerical evidence is unaffected either way.

### Should-fix

**S1. Practice 1 sends the learner to a lab that cannot take its parameters and has no boundary to watch.**

Location: `.jsx` L231 — "Set these values in the responsibility lab to watch the boundary move."

Evidence: practice 1 uses means **0 and 2** with weights 0.25/0.75. `ResponsibilityLab` exposes exactly three controls — `Measurement x`, `Weight of A`, `Variance of A` (`GmmLabs.jsx` L48–53) — and prints "Means stay at −2 and 2" in its own caption (L56). Component means are set only by the two presets (L13–16, `[-2, 2]` and `[0, 0]`); there is no control for them and `limits` contains no entry for them. The weight 0.25 is reachable; the means are not. Separately, the lab draws no tie boundary at any setting, so "watch the boundary move" describes an affordance that does not exist. A learner who follows the instruction will set the weight, see a different tie location than `0.450693856`, and have no way to tell why.

Fix: either add a means control to `ResponsibilityLab` (the model layer already supports arbitrary means; only the UI restricts them) and draw the tie location as a marked x, or replace the sentence with one that names what the lab *can* show — e.g. "The responsibility lab shows the same effect with the weight control: the heavier component's region extends toward the lighter one." Compare practice 4's pointer at L240 ("The covariance lab reproduces all three panels"), which I checked and which is **true**: ρ = ±0.5 are inside `limits.correlation` at step 0.05, (2, 1) and (2, −1) are inside `limits.point`, and the ρ = 0 null panel is always drawn.

**S2. The Iris version-dependence caveat is dropped from the published lesson.**

Location: `.jsx` L157–168 (§7) and the whole body.

Evidence: `lesson.md` L354 reads "The author calculation run used Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1; numerical fitting can vary across versions. All 16 candidates below converged in that run." Grepping the published `.jsx`, `GmmFigures.jsx` and `GmmLabs.jsx` for `3.12`, `converged in that run`, `can vary` and `version` returns nothing relevant — the only version text on the page is the `pip install` pin in **§4** (L108), which a reader will reasonably attach to the small EM program, not to the 16-candidate table five sections later. `gmm-native.json` and `gmm-iris-data.json` both record "Numerical fitting can differ on other library versions" as an explicit limitation, and `CandidateFigure` presents 32 numbers to six and three decimals as settled fact.

This matters more than it looks. My refit confirms the fragile candidate is **full K = 4** — 198 EM iterations, with one component pinned at the `reg_covar` floor — and that is exactly the fit carrying the training-BIC story and the whole narrow-variance panel of F5. A library bump is most likely to disturb the one paragraph the lesson leans on hardest.

Fix: restore the manuscript sentence in §7, before or immediately after the program. One sentence; no verifier rerun needed.

**S3. The intro promises that every investigation records a prediction before it computes; two of the three offer a button that skips it, and the third does not require one at all.**

Location: `.jsx` L38 — "Every investigation records a prediction before it computes."

Evidence: `Prediction` renders a second button, "Calculate without recording a prediction" (`GmmShared.jsx` L151), in both I1 and I3; it calls `state.explore`, which computes and displays the full result with `graded: false`. In I2 there is no gate at all — "Compute E-step" (L258) and "Apply M-step" (L259) are enabled whenever the phase allows, independent of `recorded`; only the grading is skipped. The affordances are correct — the specification's contract item 2 explicitly permits an explore action — so the defect is the promise, not the design.

Fix: reword L38 to something the page keeps, e.g. "Every investigation asks for a prediction before it shows you the answer, and retires it the moment you change an input."

**S4. The EM lab carries a prediction and a verdict across state changes they no longer describe.**

Locations: `GmmLabs.jsx` L148 (`choice` state), L189–192 (grading), L198–203 (`back`), L256 (record button).

Evidence, two parts:

1. `choice` is cleared by `apply` and by `editDraft` but **not** after a graded cycle. In `gmm-em-floor-desktop.png` the prompt reads "Starting from **iteration 1**, what will one complete E-step and M-step do…" while the radio "It rises" is already filled — that is the learner's *iteration 0* answer, still selected, presented as an answer to the next question. "Record it" is enabled (because `recorded.iteration` is 0 and `current.iteration` is 1), so one click silently records a prediction the learner never made for this iteration. The specification's contract item 1 is explicit: "Initial **prediction is unset** … record the choice, never silently infer it from a control or preset."

2. `back()` restores the run snapshot but leaves `recorded` untouched. After record → E → M, `past` is `[iter0-ready, iter0-expected]`; one "Back one half-step" returns to iteration 0, phase `expected` — while `recorded.outcome` is still set, so the verdict "The cycle from iteration 0 moved the log-likelihood by +0.696330676, so it rose" stays on screen describing a cycle that has been undone, and "Record it" is now disabled because `recorded.iteration === current.iteration`. The specification asked for exactly this: "backward stepping restores model and appropriate prediction snapshot without labeling future predictions as already made."

Fix: clear `choice` in `applyMaximization` after grading; and in `back()`, drop `recorded.outcome` (and `recorded` itself when the restored iteration precedes the one it was recorded against), which also re-enables recording.

**S5. F3 does not deliver the contrast it exists to show, and drops two required elements.**

Location: `GmmFigures.jsx` L108–145; evidence `gmm-collapse-desktop.png`.

Evidence: each panel draws its own x-window `[−2 − 4σ, −2 + 4σ]` (L116). Because the window scales with σ, the narrow bell occupies the same fraction of every panel, and the three panels are **visually identical**. The figure's own closing paragraph admits it: "The three panels look alike because each is drawn in its own window." The specification (`visual-specifications.md`, F3) required the opposite: "Rendered peak-width contrast must be visible even for SD 0.01; use a zoom panel rather than a one-pixel spike." A local x-view was permitted *for the spike*, not for all three panels simultaneously — applying it everywhere cancels the very comparison.

Two further F3 requirements are absent:
- "Objective strip plots SD 1, 0.1, 0.01, 0.001 on a log x-axis, with log-likelihood −8.122121377, …". There is a table (correct, and I verified all four values) but no strip and no log axis.
- "Use a separate common-axis inset of the broad component to establish that it remains fixed." Each panel asserts "the broad component is unchanged throughout" in text, but the broad curve is redrawn in three different windows at three different y-scales, so the claim is stated rather than shown.

Fix (smallest version that restores the teaching point): keep one shared x-window for the three panels with a marked scale break for the σ = 0.01 spike, or add a fourth panel at a common x-scale; and add the objective strip, which is four points on a log axis and already has its data.

**S6. The covariance gallery never shows a sighted reader a single covariance number.**

Location: `GmmFigures.jsx` L165–204; evidence `gmm-covariance-gallery-desktop.png`.

Evidence: each panel renders the family name, an SVG of two contours with "solid: left" / "dashed: right", and a one-line note. The eight matrices exist only inside the `aria-label` string (L177). The specification's F4 requires "Matrix cells are linked to semiaxis directions and lengths", and separately "Spherical uses separate scalar badges 1.5 and 1 to make the API distinction unavoidable" and "tied's two matrices have a 'shared' bracket". None of the three is implemented. The manuscript's own F4 note says the caption must state "which covariance entries are shared or zero" — the per-panel notes do say it in words, which is why this is should-fix rather than blocking, but the figure's stated job is to connect *cells* to *shapes*, and there are no cells.

The contours themselves are correct: I rebuilt every matrix from `symmetricEigenpairs` and each reconstructs, tied's two shapes are congruent, diag is axis-aligned, and the spherical circles render as circles at radii √1.5 and 1 on a genuinely equal scale (`unit = 34` for both axes, L167–169). The parameter table (L191–196) gives 17 / 11 / 14 / 11 correctly and, as the specification demanded, is kept separate so the two-component drawings are not implied to contain 17 parameters.

Fix: print each panel's two matrices as small text tables beside the contours, with the two spherical scalars as badges; the numbers are already in the `gallery` array.

**S7. The covariance lab's plot type never reaches the specification's size target, and its point IDs are not legible at phone width.**

Locations: `gmm-labs.css` L65 (`.gm-plot svg text, .gm-figure svg text { font: 11px … }`); `GmmLabs.jsx` L364 (`viewBox="0 0 340 300"` inside `.gm-panels`, two panels side by side); evidence `gmm-covariance-desktop.png` and `gmm-covariance-mobile.png`.

Evidence: the specification's shared contract item 6 says "Redraw charts at their available width with **fixed readable type (target 14–16px)**, rather than scaling a large SVG down", and I3's own mobile note says "Do not compress the two plots into unreadable side-by-side thumbnails." Every GMM SVG instead uses a fixed 340- or 300-unit viewBox with `width: 100%`, so the label size is whatever the scale factor makes it. For I3 specifically: at a ~700 px reader column the two panels give each SVG about 315 px, so 11 × 315/340 ≈ **10.2 px**; at a 350 px viewport the panel gives about 311 px, so ≈ **10.1 px**. The type is therefore *never* in the 14–16 px band, at any width. In `gmm-covariance-mobile.png` the axis numerals −3…3 and the "P" and "Q" point labels are not readable at all, which also defeats contract item 5 ("Points have text IDs **and** shapes as well as color") — the shapes survive, the IDs do not.

For contrast, the three full-width figures (`.gm-figure svg { max-width: 620px }`, CSS L94) render at 620/340 → ≈ 20 px and are comfortably legible, and `CandidateFigure`'s 300-unit panels land near 11 px and remain readable because they carry few, short labels. The problem is specific to the densest panel in the set.

Fix: give `.gm-plot svg text` a viewBox-relative size that resolves to ≥ 14 px at the panel's rendered width (or set the I3 panels to full width and stack them at every breakpoint, which the specification offers as the alternative: "or stack both full-width"). Then re-capture `gmm-covariance-mobile.png`.

**S8. The EM lab computes the three things that would show the mechanism, then throws them away.**

Locations: `GmmLabs.jsx` L153–160 (`startState`), L187 (`justUpdated.before`), L307–312 (parameter table); `gmm-models.js` L118 (`scatter`).

Evidence, three parts:
1. `startState` calls `expectation(...)` for the iteration-0 log-likelihood and **discards the responsibility matrix**, setting `responsibilities: null`. The responsibility table is therefore absent until the learner presses "Compute E-step" — visible in `gmm-em-step-mobile.png`. The specification's I2 opens with "The introductory displayed state must expose **both allocation and geometry** before interaction." Showing the initial allocation leaks nothing: the prediction is about the direction of the objective, not the allocation.
2. `applyMaximization` stores `justUpdated.before = current.components` and nothing ever renders it. The specification asked for "M-step feedback names the **old**/new counts, means, variances and objective" and for an "old/new parameter table [appearing] immediately under the consequence". The readout names the gain and the counts; the old means and variances are gone the moment the M-step runs.
3. `maximization` returns `scatter` — the *unclipped* weighted scatter — and the component table shows only "at the floor? yes, held at 0.25". In the `repeated` preset the true scatter is 0.0053638, which is the number that makes the floor's activity legible; the learner sees only that a floor was applied, not by how much. The manuscript's I2 blurb asks the learner to "explain which update hits the boundary".

Fix: keep `startState`'s responsibilities; render `justUpdated.before` as an old/new column pair in the existing parameter table; add a "weighted scatter" column, shown when `atFloor`.

**S9. Observation IDs in the EM lab plot are struck through by the component mean stems, and collide with each other in the repeated-measurements preset.**

Location: `GmmLabs.jsx` L285–292 — observation labels are drawn at `scaleX(value)` with `textAnchor="middle"`, and the mean stems at `scaleX(component.mean)` run from the axis to `peak * 0.9`.

Evidence: in the standard preset the observations are at −2, −1, 1, 2 and the initial means at −1 and +1, so the "B" and "C" labels sit exactly under the two stems — visible in `gmm-em-step-mobile.png`, where only "A" and "D" are readable. In the `repeated` preset the observations are `[-2, -2, 2, 2]`, so "A" and "B" are drawn at the same coordinate and "C" and "D" likewise — visible in `gmm-em-floor-desktop.png` as an unreadable composite glyph at each location. The specification's F2 verification note ("No component label is confused with observation A or B") is about naming, but contract item 5's "Points have text IDs and shapes as well as color" is defeated here in fact.

Fix: offset duplicate labels horizontally (or stack them as "A, B"), and draw the mean stems with a small x-offset or below the label baseline. The responsibility table already disambiguates the rows, so this is legibility rather than ambiguity of meaning.

**S10. The row 35/38 corrections are sourced to a page that documents neither the rows nor the values.**

Location: `.jsx` L276 — "The supplied file is the corrected [scikit-learn snapshot](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), which fixes two historical rows: row 35 is 4.9, 3.1, 1.5, 0.2 and row 38 is 4.9, 3.6, 1.4, 0.1."

Evidence: the linked `load_iris` page says, in full, "*Changed in version 0.20: Fixed two wrong data points according to Fisher's paper. The new version is the same as in R, but not as in the UCI Machine Learning Repository.*" It never names a row, never gives a value, and never uses the words "corrected" or "bezdek". The row-level facts come from the **UCI** record, whose `additional_info.summary` reads verbatim: "*The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa" where the error is in the fourth feature. The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa", where the errors are in the second and third features.*" UCI is cited earlier in the same bullet, for the licence.

This is an implementation-only slip: the manuscript (`lesson.md` L769) does not state the rows at all, and the packet's `data-provenance.md` L7 attributes them correctly — "UCI flags historical differences in the 35th and 38th samples; the supplied rows 35 and 38 are respectively `4.9,3.1,1.5,0.2,setosa` and `4.9,3.6,1.4,0.1,setosa`." The published page pulled the facts forward from the provenance file and dropped the source with them.

**The facts themselves are correct** — I diffed the served CSV against `load_iris()` (identical), against UCI `iris.data` (exactly three cells differ, precisely as UCI describes) and against UCI `bezdekIris.data` (identical). Only the attribution of the claim is misplaced. A secondary wrinkle from the same sentence: the CC BY attribution points at the UCI record while the shipped bytes follow the R/scikit-learn correction, which sklearn's own page says is "not as in the UCI Machine Learning Repository".

Fix, one clause: "…which fixes the two rows UCI's own record identifies: row 35 …". The manuscript needs no edit.

**S11. The design record's "Departures from the manuscript" list is materially incomplete.**

Location: `GMM-LESSON-DESIGN.md`, "Departures from the manuscript" (four bullets).

Not recorded, at least: (a) the version-dependence caveat dropped from §7 (S2); (b) the §9 "Zero terms are handled by their limiting values" sentence dropped; (c) the §3 four-row responsibility table moved into `AllocationFigure`; (d) the §7 16-candidate table moved into `CandidateFigure` as a scroll-capped region; (e) the §10 Bayesian weights table replaced by program output alone; (f) the readiness-check table added in §12; (g) `RunnableExample.jsx` — a component every lesson shares — changed to print `Save as <file>`; (h) three F3 elements and three F4 elements not implemented (S5, S6); (i) the free-text explanation field the interaction contract asked for, not implemented (S14). Most are improvements or harmless; the record is what lets the next author tell a deliberate choice from a slip.

**S12. The EM lab's objective-history plot prints its y-axis values outside the plot area, over the x tick labels.**

Location: `GmmLabs.jsx` L314–325 — the `Plot` is `height={160}` with `padding.bottom = 30`, so the axis sits at y = 130 and the tick labels at y = 144; the two value labels are hard-coded at `y="12"` and `y="150"`.

Evidence: in `gmm-em-floor-desktop.png` the lower label "−6.447001" sits below the axis on the same line as the "0" tick and touches it; the upper label "−3.675769" floats above the plot frame rather than beside its own point. The design record notes that three other label overlaps were found and repaired by screenshot inspection; this one survived.

Fix: place both labels inside the frame using the plot's own `scaleY` (the component already receives it), or extend `padding.left` and left-align them against the y axis.

**S13. F5's third panel reduces the required covariance geometry to one horizontal line, and its accessible text omits the training IDs the specification required.**

Location: `GmmFigures.jsx` L264–299.

Evidence: the specification says "Overlay full-K4 component 0 covariance geometry transformed back using train means/scales", and separately that the text equivalent must include "the narrow covariance's values and **actual train IDs**". What is drawn is a single horizontal rule at the raw mean width (L272) plus, in the zoom panel, a band of ±1 fitted SD. The component's other axis — standardised variance 1.313807522, i.e. ≈ 0.95 cm of sepal-length spread — is never shown, so the reader sees a line rather than an extremely eccentric ellipse. The panel `aria-label` (L266) gives only the coordinate ranges and the width; `gmm-iris-data.js` ships all 150 rows with their split codes and `selectedParameters`, neither of which reaches any accessible table.

What is right, and worth keeping: the band is drawn at its **true** height with an explicit source comment refusing to pad it (L285–290), which is exactly what the specification demanded; the two vertical scales are labelled and contrasted in the caption; species colours are absent from every panel and the text says so; and the "component 0" naming is explicitly disclaimed. I verified the 14 training rows that component 0 takes by argmax all have sepal width exactly 3.0 cm, so "a narrow line of rounded measurements" is literally true.

Fix: draw the transformed ellipse in the full-scale panel (clipped or annotated where it leaves the strip), and add the in-strip rows' IDs to the zoom panel's accessible description.

**S14. No investigation offers the explanation field the interaction contract asks for twice.**

Location: `GmmShared.jsx` L127–166 (`Prediction`) and the hand-rolled prediction block in `EmStepLab` (`GmmLabs.jsx` L241–270).

Evidence: contract item 1 requires "a labeled radio group with no selected option **and an optional short explanation field**", and item 3 explains why: "A correct choice accompanied by a wrong explanation still invites the learner to inspect the mechanism; free text is not automatically graded." No such field exists in any of the three investigations. I3 partly compensates with its "Two transfers worth recording separately" disclosure (`GmmLabs.jsx` L432–439), which asks for explanations but provides nowhere to put them.

Fix: add an optional `<textarea>` to `Prediction`, carried in the committed snapshot and echoed back beside the verdict, ungraded. One component change covers I1 and I3; `EmStepLab` needs the same two lines.

### Observation

**O1. Trailing-zero stripping makes published numbers differ from the manuscript's.** `GmmShared.jsx` L22 does `value.toFixed(digits).replace(/0+$/, '')`, so `CandidateFigure`'s table prints the selected candidate's validation score as **−2.6688** where the manuscript, the program output and practice 5 all write **−2.668800**; likewise "528.98" for "528.980" and, in `BoundChainFigure`, "−7.71040426" beside "−8.069044223" in the same column. Nothing is wrong, but a table whose decimal places vary row to row reads as noise, and the headline number no longer matches the sentence that cites it three screens away.

**O2. Dead exports.** `gmm-iris-data.js` ships `selectedParameters`, `testRows` (30 rows with per-row densities and responsibilities), `bayesianSensitivity`, `speciesNames`, `splitNames` and `splitCounts`; none reaches the page. `gmm-models.js` exports `emTrace`, `conditionalMixture`, `mixtureMoments`, `sphericalResponsibility`, `parameterCount`, `criteria` and `quadraticForm`, all of which are exercised only by `scripts/verify-gmm-models.mjs` — so §6's AIC/BIC example, §10's k-means table and §10's conditional mixture are hard-coded prose beside a verified model that could have generated them. `GmmLabs.jsx` L444 exports `weightedUpdate` "for the practice task" and nothing imports it; `GmmFigures.jsx` L359 re-exports `scaler` and `selection` for no consumer.

**O3. The Bayesian weights table is not rendered.** `lesson.md` L616–618 has a three-row table; the page carries the numbers only inside the third program's expected output, while the module ships `bayesianSensitivity` ready to render it.

**O4. A shared component was changed for this lesson without being recorded.** `RunnableExample.jsx` now emits `Save as <code>{example.file}</code>` for every lesson that sets `file` — which includes the anomaly lesson, whose own review raised the absence as O7. Good change; it belongs in the design record's "What was built".

**O5. The "Before running" question precedes the heading it belongs to.** `Program` (`.jsx` L26) emits a `<Prose>` before `<RunnableExample>`, which emits the `<H3>` title. Reading order is question → title → code, the same ordering as the sibling anomaly lesson.

**O6. F1's source comment names the wrong section.** `GmmFigures.jsx` L13 says `/** §1 · F1. …` while the specification places F1 in §2 and the JSX renders it there (L53). The other five comments are right.

**O7. `narrowComponent.covariance` ships a signed negative zero.** `gmm-iris-data.js` L217: `[[1.313807522156, -0.0], [0.0, 0.0001]]`. The true off-diagonals are ≈ −9.2e−34 and 1.1e−34, so the matrix is asymmetric in its published rounding and a future renderer could print `-0`.

**O8. The Wu (1983) hedge is stale.** `.jsx` L273 and `lesson.md` L766 say "full-text retrieval failed during authoring, so no theorem-number guidance is claimed." The Project Euclid full text and PDF are openly served today. The caveat is honest about the authoring process but now invites the question of why no theorem numbers were checked when the paper is one click away.

**O9. The manuscript does not hyperlink the CC BY licence; the published page does.** `lesson.md` L769 writes "CC BY 4.0" as plain text, and the deed explicitly requires "provide a link to the license". `.jsx` L276 links `https://creativecommons.org/licenses/by/4.0/` and, with the creator, title, DOI source link and a statement of the modification, complies. Only the packet source diverges.

**O10. The ARI rests on six setosa rows.** The 30 test rows split 6 / 9 / 15 by species, and the selected two-component argmax produces a pure "setosa versus the rest" partition. `0.433446` is correct and §7 describes it correctly as a separate diagnostic on exactly those rows, but the imbalance driving it is not mentioned.

**O11. The split seed appears only in the code block.** `.jsx` L160 describes "one fixed, label-blind permutation: 90 training, 30 validation and 30 test" without naming `default_rng(16)` or the slice convention; the program two elements later is exact. A reader working from the prose alone cannot reproduce the split.

**O12. The "one Gaussian" baseline carries `reg_covar=1e-4`.** Full K = 1 fits covariance `[[1.0001, −0.029584], [−0.029584, 1.0001]]`, not the plain MLE. The test difference is 4e−6 nats and changes nothing, but "one Gaussian" is doing slightly less work than it sounds.

**O13. Component identity is carried by colour alone in three places.** `gmm-labs.css` L70–71 gives `.gm-curve.is-left` and `.is-right` the **same** `stroke-dasharray: 5 3`, differing only in hue (#8eb9a5 against #91aecf); `.gm-stem.is-left`/`.is-right` and the allocation bar's `.gm-bar-left`/`.gm-bar-right` (L86–87) are likewise hue-only. In I1 and I2 the two component curves are therefore distinguishable only by colour. Adjacent numeric readouts carry the same information in text, which is why this is an observation rather than a finding, but a second dash pattern costs one line.

**O14. I1's responsibility bar has neither percentages nor patterns.** The specification says "The responsibility bar is [0,1] with **percentages** and fixed A/B patterns"; `GmmLabs.jsx` L96 prints "A 0.9997 · B 0.0003" and the fills are flat colours. The adjacent table and the `aria-label` cover the content.

**O15. The saved two-point pair survives a model round-trip.** `GmmLabs.jsx` L36 filters `saved` by `modelKey(active)` rather than clearing it, so editing the weight and editing it back re-displays the earlier pair. The design record claims "Changing the model clears the pair." The values are still correct for that model, so nothing false is shown — but the record overstates the mechanism.

**O16. The intro slightly overstates what is fitted.** `.jsx` L38 says "fit 150 real flower measurements"; 90 rows are fitted, 30 select and 30 are reserved. §7 is precise about this.

**O17. F5's 16-row table is capped at 19 rem.** `gmm-labs.css` L56 (`.gm-rows { max-height: 19rem }`) means about six rows are visible; the full K = 4 row that the same figure marks as the training-BIC winner is below the fold, reachable only by scrolling the region. It is a keyboard-operable `role="region"` with `tabIndex={0}`, so this is friction, not inaccessibility.

**O18. Three scikit-learn citations pin a version against a moving URL.** `.jsx` L274 and `lesson.md` L767 say "checked against documentation identifying itself as 1.9.1" while linking `/stable/` pages. The claim is accurate today — all three pages self-identify as 1.9.1 — and will falsify silently.

**O19. Control ranges are not visible until you break them.** Contract item 5 asks that "Every control has a persistent label, unit, range and visible focus." `NumberField` sets `min`/`max` attributes and surfaces "Keep it between −10 and 10" only as an error after an invalid entry. Similarly, I2 and I3 carry no "numerical limits of the browser model … described once near its controls" note; I1 has one (`GmmLabs.jsx` L55–58) and it is good.

**O20. One precision sentence dropped from §9.** `lesson.md` L509's "Zero terms are handled by their limiting values where the support permits" does not appear in the published Jensen step. `boundDecomposition` (`gmm-models.js` L181) does implement it (`if (share <= 0) return;`).

**O21. Screenshot gaps.** No mobile capture of the responsibility lab, the selector figure, the collapse figure, the gallery or the bound chain; and no capture of the F3 "spike zoom" that the specification listed as required, because no zoom panel exists (S5). `gmm-browser.json` asserts 320 px and 390 px passes structurally; the only narrow captures are at 350 px.

## Closure

The mathematics and the data handling of this lesson are, as far as I can establish, flawless. Every number I recomputed from first principles agrees to the full stated precision — the EM trace, the first M-step, the changed-D cycle, the identical-start null, the log-sum-exp fixture, all four collapse values, `8/7` against `8` and their densities, the `39.346934 %` contour, the parameter totals, the AIC/BIC example, the complete Q / entropy / ELBO decomposition with its touching equality and `0.337690713` gap, the k-means limit, and all eight practice answers, several of them exactly by rational arithmetic. The Iris pipeline reproduces bit-exactly from the raw CSV: the split, the `ddof=0` scaler, all 16 candidates with independently derived parameter counts, the narrow component's `1e-4` variance by exact float equality, the reserved test at `−3.011223` against `−2.887249` with the baseline ahead, and an ARI of `0.433446` from my own contingency table. The served CSV is the corrected snapshot it says it is, verified against `load_iris`, UCI `iris.data` and UCI `bezdekIris.data`. All three displayed programs reproduce their published output. Every external URL resolves to the document it claims, and the Bishop section range — which looked like the most likely mis-citation in the set — is correct, because the k-means *relationship* really does sit at §9.3.2 inside the cited 9.2–9.4 range.

What needs work is presentation and record-keeping. Two records assert an implementation state and an independent review that did not exist (B1); a sentence and a program label are shipped twice (B2); one practice task points at a lab that cannot do what it says (S1); the one caveat that would protect the lesson's most fragile numbers is missing (S2); the EM lab lets a prediction and a verdict outlive the states they describe (S4); the collapse figure cancels its own comparison (S5); the covariance gallery hides its numbers from sighted readers (S6); and the covariance lab's labels never reach a readable size (S7).

Recheck plan after the fixes: B1, S11 and most observations are documentation and need no verifier. B2 location 2 changes `gmm-examples.js`, so re-run `scratch/lesson-tools/Scripts/python.exe scripts/verify-gmm-examples.py` (it hashes the module) — the executed output is unaffected. S1, S3, S4, S8, S9, S12, S14 are component changes; after them re-run `node scripts/verify-gmm-models.mjs` and `node scripts/verify-gmm-browser.cjs`, and capture two pictures that would have caught the defects: the EM lab immediately after a completed cycle (showing an unset radio for the next iteration) and again after one "Back one half-step". S5, S6, S7 and S13 are figure changes needing fresh screenshots at desktop and at 350 px, including the F3 zoom panel the specification asked for. S2, S10 and O20 are prose edits in both the JSX and `lesson.md` and need no verifier rerun. **No numerical verifier needs to be repeated for any finding in this review.**
