# Yeast observed-data provenance

## Original observations and permission

Nakai, K. (1991), **Yeast**, UCI Machine Learning Repository, DOI [10.24432/C5KG68](https://doi.org/10.24432/C5KG68). [Dataset landing page](https://archive.ics.uci.edu/dataset/110/yeast); [official archive](https://archive.ics.uci.edu/static/public/110/yeast.zip). The UCI page's substantive dataset description, feature table and CC BY4.0 license were inspected on12 September2026. Attribution and a description of analysis changes must accompany a future downloadable extract. The ZIP's full `yeast.names` was also read in memory; it identifies Kenta Nakai as creator and Paul Horton as the1996 donor, describes the original localization task and cites the earlier1991/1992 and1996 work. The UCI citation year and donation date are different metadata fields.

Retained `yeast.data` is the unchanged original94,976-byte data member. It has1,484 rows, a sequence/protein identifier, eight numerical scores and an original location label. SHA-256:

`7cf61776fc04f527f93bf57a327b863893a1225d82df02d457e8950173218258`

Downloaded archive SHA-256:

`8fe9a4490edfa67532f1dee7ec11f1804437ad43f9f8b7536e32d5be8bce0bb5`

The archive was read in memory and not retained. No large corpus, image copy, scraped model output or private user data is involved. `calculated-inputs.json` is an author-produced derivative with split identities, scaler parameters, fitted scores and synthetic-data lineage; it does not replace the original observed file.

## What the columns mean

| Source column after identifier | Name | Source meaning | Used in this study |
| --- | --- | --- | --- |
|1|mcg|McGeoch signal-sequence recognition score|yes|
|2|gvh|von Heijne signal-sequence recognition score|yes|
|3|alm|ALOM membrane-spanning prediction score|yes|
|4|mit|Discriminant score using the first20 amino acids for mitochondrial versus nonmitochondrial sequences|yes|
|5|erl|Binary HDEL-substring indicator|no|
|6|pox|C-terminal peroxisomal-targeting-signal feature|no|
|7|vac|Discriminant score for vacuolar versus extracellular proteins|yes|
|8|nuc|Nuclear-localization-signal score|yes|

The six selected source-score columns are fixed indices[0,1,2,3,6,7] within the eight numerical features. This predeclared modeling choice avoids treating the excluded targeting/indicator fields as ordinary continuous interpolation coordinates. It is not a feature-subset search. No physical unit is supplied for these scores; label axes as source-score units or standardized-score units. An interpolated score vector is synthetic numerical training data, not a synthesized amino-acid sequence or demonstrated physically valid protein.

The original location counts are CYT463, NUC429, MIT244, ME3163, ME251, ME144, EXC35, VAC30, POX20, ERL5. The lesson defines a binary positive as original classME2, the membrane-protein class with an uncleaved signal sequence; all other original labels are negative for this one-vs-rest task. This is a teaching task definition, not a clinical diagnosis.

## Identity repair before the frozen study

The original file contains1,462 distinct protein IDs. Twenty-two IDs each occur twice, and each repeated record has exactly the same eight scores and original label. All repeated records belong to CYT or NUC, so all51 ME2 observations remain after identity repair. The analysis verifies identical feature/label content before retaining the first original-file occurrence per ID. It would stop on conflicting repeats instead of guessing how to merge them. The unchanged input is retained; kept/removed source indices and duplicate identifier names are saved in the JSON.

This was a necessary correction discovered after an initial five-fit author calculation had used the1,484 rows directly. That initial row split could place identical proteins across roles. Its outputs were discarded and the exact same five methods, penalty, target, feature columns, seeds, split sizes for development/tuning/inspection and cost criterion were rerun once after identity repair. The initial reserve had484 rows; the corrected reserve has462. No method or hyperparameter was changed to favor a result. There were ten native author fits in total across this correction, five in the final retained experiment. The manuscript and visual contracts use only the corrected results. No initial or corrected reserve predictions were computed.

Identity deduplication prevents this exact-record leakage. It does not supply unknown homolog/family/experimental-group metadata, and therefore does not establish independent assessment of a new protein family or organism. Source dataset age and task construction also limit deployment interpretation.

## Fixed protocol and derived artifacts

Sort retained first-occurrence indices into source order. Stratified `train_test_split` with seed61 gives1,000 development and462 reserve. From development, seed62 gives600 fitting and400 remaining. Seed63 splits that remainder into200 tuning and200 inspection. Positive counts21/7/7/16 respectively. The JSON stores all exact source indices, not only seeds, so later implementations need not reproduce a library's shuffle to identify a case.

StandardScaler is fitted on the original600 fitting rows of six features. No scaler fits on tuning, inspection, reserve, duplicated or synthetic rows. The five declared procedures use the same weighted-mean logistic loss plus λ||w||²/2, λ=.01, unpenalized intercept. SciPy L-BFGS-B starts at zero with analytic gradient, maxiter1000, gtol1e−9 and ftol1e−13. The records retain objective, convergence status, iteration count and gradient infinity norm.

- Original:600 fitting rows,21 positive/579 negative.
- Balanced weights:n/(2classCount), so total weight per class is equal; weights normalize by their total in the objective.
- Random oversampling: seed64 draws558 positive copies, making1,158 training rows.
- Random undersampling: seed65 retains21 of579 negative rows plus all21 positive rows,42 total.
- SMOTE: seed66 draws558 anchor/neighbor/fraction triples among21 positives, with k3, standardized Euclidean distance, self excluded and stable distance ties. One scalar fraction interpolates an entire vector. Result1,158 training rows. All anchor source IDs, neighbor indices and fractions are retained, plus ten derived vectors in both standardized and source-score coordinates.

The final fitted models are held fixed while their thresholds are selected on the200 tuning records. Candidate thresholds are each distinct score plus no-alerts; costsFP1/FN12 are hypothetical and fixed. Tied minimum cost uses the higher threshold. Final200 inspection records receive those thresholds once for the recorded comparison. Actual scores, labels and source IDs are retained for future bounded investigations;462 reserve records receive no predictions. The code's no-alert threshold infinity is serialized as the string `above-all-scores`, not invalid JSON Infinity.

`author-calculations.py` reproduces the final five-fit evidence and exact constructed fixtures. `calculated-inputs.json` contains actual results, not estimated stdout. Python3.12.14, NumPy2.3.5, SciPy1.18.1 and sklearn1.9.1 were used. There was no imbalanced-learn installation or execution; the optional displayed API example remains unexecuted. Browser plots, native displayed-program equivalence, UI interactions and formal phase-two review remain pending.

When publishing later, retain this attribution, source hash, class definition, identity repair, fitting/assessment ownership and the distinction between observed proteins and generated vectors. Do not publish an unexplained balanced copy as if it were the original population.
