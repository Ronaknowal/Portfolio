# Palmer Penguins input and calculated results

Downloaded 12 September 2026 from the dataset creators' public file:
https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv

The project https://allisonhorst.github.io/palmerpenguins/ identifies the dataset as CC0. Credit Allison Horst, Alison Hill, and Kristen Gorman; observations originate with Gorman and colleagues' Palmer Station penguin research. The variable reference https://allisonhorst.github.io/palmerpenguins/reference/penguins.html was inspected for acquisition context, row count and measurement units. The educational prediction task is our own use, not the original study's asserted research question.

`penguins.csv` is the unchanged 15,241-byte public CSV. SHA-256:
`f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`

All 344 rows are included in their original order; no rows were selected for favorable outcomes. Variables: species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year. Units are explicit in the numeric column names. Missing values remain NA in the CSV. Each numeric measurement column has two missing cells, and sex has eleven. Row identifiers in the calculated file are zero-based positions in this CSV, not data features or original animal identifiers.

The fitted feature matrix uses four numeric measurements and sex. Species is the target; island/year/row ID are excluded from X. A stratified train/test split with test_size .25 and random_state20 yields258 training and86 held-out rows. Exact indices are recorded. This is a controlled random-row comparison within this dataset, not a demonstrated temporal, island, repeated-animal, or deployment-population generalization claim.

`author-calculations.py` is a compact bounded author calculation. It reads only this local CSV, fits four small KNN pipelines and a majority baseline, and retains actual preparation statistics, feature names, held-out transformed records, predictions, and confusion matrices. It also calculates constructed scaler, target-encoding and KNN-imputation fixtures. These synthetic tables are explanatory examples, not claimed real observations. Environment: Python3.12.14, NumPy2.3.5, pandas3.0.1, scikit-learn1.9.1, using the shared read-only lesson-tools interpreter. No dependencies were installed or mutated. One initial serialization failure was corrected by converting missing raw-table cells to JSON null; subsequent execution completed without warnings. This was a content calculation, not the deferred formal native-example campaign.

`calculated-inputs.json` stores the actual output with no NaN JSON values. Expected correct counts: majority38, raw67, standard84, minmax85, robust85, all out of86. The packet does not choose a deployment winner from this same inspected holdout. All result values, original test rows and transformed coordinates remain available for phase-two figures without remote fetching. Do not regenerate data with a different split and retain the old narrative numbers.

No screenshots, downloaded reference PDFs, environment copies or disposable logs are retained. Keep the small CSV and calculation/result pair while the packet awaits implementation; they are required offline inputs and quantitative provenance.
