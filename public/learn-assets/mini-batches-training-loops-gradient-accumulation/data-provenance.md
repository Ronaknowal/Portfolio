# Offline Iris observations

Prepared 13 September 2026 for the training-loop and accumulation experiment. This is real observed data; the scalar and normalization investigation fixtures are explicitly constructed examples.

- Dataset attribution: Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76). [UCI page](https://archive.ics.uci.edu/dataset/53/iris), metadata and license inspected 13 September 2026.
- Underlying data license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as stated by UCI. Retain attribution and this adaptation notice with the downloadable CSV.
- Exact local variant: `sklearn.datasets.load_iris()` in scikit-learn 1.9.1, already installed in the shared lesson runtime. Exported once on 13 September 2026; no new package or dataset download occurred. [Official load_iris documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) states that two incorrect values were corrected in version 0.20 according to Fisher's paper; its bundled variant matches R and differs from the older UCI copy. Do not claim our file is byte-identical to the old UCI `iris.data` file.
- Export transformations: preserve all 150 rows and their order; retain all four measured features and class IDs; add one-based `row_id` and descriptive header; serialize measurement floats to one decimal and class IDs as integers. No observations were synthesized, imputed or removed.
- File: [iris.csv](iris.csv), UTF-8 ASCII content, CSV header plus 150 observations. SHA-256 `777674aa8e8ed6bc0a6758822616f02bceac793e0044a2676d9ee328316bf8b0`.
- Columns: `row_id`, `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`, `class_id`. Classes 0=setosa, 1=versicolor, 2=virginica; 50 observations per species. The new row ID is an export locator, not a newly discovered collector identifier.
- Original educational/scientific question: distinguish three Iris species using flower measurements. Our chosen neural network is a new instructional experiment, not a reproduction of Fisher's discriminant analysis or a model recommended for field identification.

## Reproduction protocol

[train_iris.py](train_iris.py) reads only the supplied CSV. It stratifies the fixed rows into 40 training/10 validation per class using NumPy generator seed 17; training indices define feature means and population standard deviations. The validation partition never enters normalization fitting or backpropagation. Model initialization uses torch seed 17. Twenty epoch orders are generated once using NumPy generator seed 29, shared between the two runs.

Both models are a 4→8→3 tanh network, CPU float64, one Torch thread, SGD(lr=0.05,momentum=0.9), with no clipping, scheduler, dropout, batch normalization, weight decay or AMP. Effective groups per epoch are 32,32,32,24. The compared microbatch limits are 32 versus 12; the changed practice additionally executes limit 7 with the same effective groups. All labels are integer class targets, so cross-entropy sum divided by group row count is the intended objective.

The single fixed twenty-epoch experiment was selected to test loop equivalence, with no hyperparameter or seed sweep. Its validation rows are an observed diagnostic split, not an untouched final test set. No performance timing, peak-memory measurement, external deployment or multi-seed uncertainty estimate was collected. Do not convert the displayed counts into a universal accuracy or speed claim. Preserve the full epoch history, including accuracy regressions, in [author-results.json](author-results.json).

The final source-bound record identifies Python 3.12.14, Torch 2.14.0+cpu and NumPy 2.3.5 on Windows. The runtime's scikit-learn is needed only to explain/export the provenance; learners running the saved program need NumPy and Torch, not scikit-learn or internet access.
