# Rademacher lesson — data and author calculations

Prepared on 12 September 2026. All programs and small data files are pending instructional assets for the later implementation phase. No browser or production integration is claimed.

## Constructed mathematical evidence

`complexity_calculations.py` generates `checked-results.json`. Its prediction matrices, sign patterns, vectors, kernels and margins are explicitly constructed examples. Exact enumeration means every finite sign pattern is evaluated; displayed floating decimals still round irrational or fractional values. The Monte Carlo branch is labeled separately and records seed, draw count, estimate, deterministic per-draw range and one-sided Hoeffding correction.

Key checks are singleton0, constants1/2, positive thresholds2/3, both orientations5/6 and all labels1 on three distinct inputs; loss complexity1/3; absolute singleton1/2; duplicate/translation/convex-average nulls; Euclidean duplicate versus perpendicular geometry; rotation and input-sign invariance; PSD kernel similarity comparison; margin scale invariance and changed practice. The finite-population proof example has exactly two equiprobable population points, four [0,1]-valued loss functions and every size-two sample with replacement. It checks expectation inequalities as a small author demonstration, not an empirical learning benchmark.

## Real data source

- Dataset: [UCI Banknote Authentication267](https://archive.ics.uci.edu/dataset/267/banknote+authentication), Volker Lohweg2012, DOI10.24432/C55P57, [CC BY4.0](https://creativecommons.org/licenses/by/4.0/).
- Original: 1,372 rows; four wavelet-derived statistics—variance, skewness, source-spelled `curtosis`, entropy—and class code0/1. The retained CSV uses its established schema. No unverified code-to-authenticity interpretation is imposed.
- Actual archive retrieved earlier in this assigned range on 12 September 2026: https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip . Archive SHA256 `1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227`; member `data_banknote_authentication.txt` SHA256 `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9`.
- Selection: `np.random.default_rng(23).permutation(1372)[:480]`, preserving one-based source row IDs. This packet copies the existing PAC packet's CSV byte-for-byte. It does not claim another download.
- Retained `banknote-subset.csv` SHA256 `d28fa993ed459d2f706816395475af08eebd2f394be67f2ad42dd9b511fc6b5a`.
- The historical split field labels320pool/80dev/80test. This experiment's explicit role allocation is first80 representation design, next240 coefficient fit, next80 validation and last80 assessment. Source IDs in `experiment-results.json` are the authority for those disjoint roles.

The fixed subset makes neighboring lessons comparable without reusing their fitted estimators or silently giving a row two roles within this experiment. Drawing rows without replacement from one corpus is different from drawing new iid acquisitions. The teaching manuscript calls the bound expression a diagnostic under a stated iid model, not a proof that this source collection certifies future deployment risk.

Root also checked exact equality of all four numeric features in the retained CSV. There are 476 distinct feature vectors among 480 rows. Repeated source-ID pairs are 346 (representation) / 108 (fit), 658 / 618 (both fit), 428 (fit) / 285 (validation), and 405 (fit) / 477 (validation). No repeated feature group involves assessment. The manuscript now distinguishes row-role disjointness from feature uniqueness and states the two fitting/validation overlaps. Original measurements remain a disclosed finite-corpus diagnostic; they were not retroactively relabeled as a grouped or iid experiment. A future independent-input study needs a newly declared grouping/split protocol and new evidence, not a cosmetic renaming of these results.

## Experiment specification and actual execution

`bounded_norm_experiment.py` fits a `StandardScaler` using only the first80 rows, then freezes it. Standardized features are divided by3 and clipped to[−1,1], followed by a constant1 coordinate. That defines a globally bounded feature representation with row norm at most√5; clipping and the constrained intercept are part of the model.

The coefficient objective is mean logistic margin loss subject to a Euclidean ball with B in {.25,.5,1,2,4}. SciPy SLSQP receives analytic gradients, an explicit radius inequality and fixed solver tolerances. A small inward projection handles roundoff at the boundary. The returned vector's objective, norm and KKT stationarity residual are checked and recorded; this is numerical author verification, not a formal optimizer proof. The bounded ramp losses for rho in {.5,1} are evaluated separately from the logistic fitting objective.

Model selection uses validation errors, then validation log loss, then smaller B. It occurs before assessment computations. All five candidates were specified before results; all ten theory comparisons receive the union-bound allowance delta/K with delta=.05. No validation-selected model is represented as being chosen by the bound, and no assessment-selected candidate is hidden.

Author execution: Python3.12.14, NumPy2.3.5, SciPy1.18.1, scikit-learn1.9.1. Both complete programs ran successfully using existing shared packages; no packages were installed there. The first real run returned coefficient norms larger than their radii by roughly10⁻¹² or less. A deterministic inward projection fixed the feasibility boundary, and the complete experiment was rerun. The final selected budget remains4, with2/80 assessment errors against29/80 for the fit-majority baseline. Every simple predeclared bound expression remains above1. Exact final values are in the JSON, rather than inferred from rounded tables.

Retained outputs include all fitted weights, representation parameters, source IDs, train/validation/assessment margins and errors, raw and clipped bound components, Monte Carlo estimates and solver records. No wall-clock benchmark, neural fit, stability experiment, learned kernel fit or deployment certificate is claimed. The chapter's citations to those methods support explained alternative theory routes.

## Retention and later verification

Keep the CSV, two programs, two result files, full manuscript, visual contracts and design/provenance records. They support offline downloads and independently reproducible calculations in phase two. A local Python `__pycache__` is disposable and should be removed once author checks finish. Do not delete pending source data or replace them with a network-only demonstration.

Later work must verify the mathematical calculations independently, reconstruct real features and predictions, implement accessible diagrams/interactions, check download execution and review rendered content. These author calculations establish prepared content evidence only.
