# Iris data carried with this draft

Collected data: four morphological measurements of iris specimens used to study discrimination between species. Source attribution: Fisher, R. (1936), *Iris* [dataset], UCI Machine Learning Repository, DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76). [UCI's dataset page](https://archive.ics.uci.edu/dataset/53/iris) states **CC BY 4.0**; [license](https://creativecommons.org/licenses/by/4.0/). Retrieved metadata 12 September 2026.

The supplied `iris.csv` was exported on that date from the offline `sklearn.datasets.load_iris` copy installed with **scikit-learn 1.9.1**. It contains the loader's corrected Fisher-version values, not a fresh download of the legacy UCI `iris.data` file; UCI documents differences in rows 35 and 38. [Loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) describes the corrected copy. Phase two must retain this exact provided file and show its download link beside every example that needs it.

- 150 data rows; no omitted observations and no generated values.
- `row_id`: 0 through 149, assigned in loader order; this is an identifier, **not a feature**.
- `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`: four original measurements, in centimetres.
- `species`: 0=setosa, 1=versicolor, 2=virginica; reference annotation, **not a fitting feature**.
- CSV SHA256: `13c9255444bce09fd9df60cccfec34d5b0a0eeebc06008fb4e2b836aa2caf78a`.
- Export transformation: added zero-based row identifier, numeric class codes and descriptive header; measured values unchanged from the named loader.

The lesson asks whether geometric groups under a declared representation agree with these recorded categories. The dataset is a small historical benchmark, not a representative modern field-sampling study. The read-only descriptive comparison uses all rows. The separate deployment-style example explicitly creates development/selection/report subsets; those are different evaluation questions, not conflicting split protocols.

`author-calculations.json` contains small calculations supporting manuscript values and proposed investigation fixtures. It is not a benchmark, population uncertainty estimate, complete displayed-program run or runtime/browser verification. All future outputs must identify which representation, k, seed and data population produced them.

Additional reproducible author inputs:

- [author-input-probes.py](author-input-probes.py) and [visual-input-calculations.json](visual-input-calculations.json): the explicit ring construction, frozen-feature-weight contrasts, exact row-aligned partition comparison and saved transformation/view data. The ring points are generated teaching data; Iris coordinates remain the real licensed observations.
- [report-author-calculation.py](report-author-calculation.py) and [report-author-calculation.json](report-author-calculation.json): the prespecified fit/selection/report calculation and actual row-ID lists. Its standardized distortion is reported per specimen, against a fit-mean one-center baseline; it is not a population confidence statement.

Neither script downloads data. Both use the supplied CSV and declared installed versions. They record content-stage calculations separately from future execution of the six displayed programs.
