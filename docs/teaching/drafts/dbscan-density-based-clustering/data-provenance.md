# Offline Iris input and author calculation provenance

Prepared 12 September 2026 for `dbscan-density-based-clustering`, content stages 1–2 only.

**Attribution:** Fisher, R. (1936), *Iris* [Dataset], UCI Machine Learning Repository, [DOI 10.24432/C56C76](https://doi.org/10.24432/C56C76). The [UCI dataset page](https://archive.ics.uci.edu/dataset/53/iris) identifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as its license. The distributed file is an adaptation into named CSV columns with a row-ID column and numeric species codes; no endorsement by the dataset creator is implied.

The immediate source is `sklearn.datasets.load_iris()` from installed scikit-learn **1.9.1**, exported by [author-calculations.py](author-calculations.py). The [loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) states that the copy corrects two historical data points to match Fisher's paper, unlike the older UCI file. UCI itself identifies the corrections. This CSV retains those corrected values: zero-based ID 34 is `[4.9,3.1,1.5,.2]`, ID 37 is `[4.9,3.6,1.4,.1]`. Do not silently swap in a different Iris file when comparing exact outputs.

## File contract

- [iris.csv](iris.csv): 150 observations, original loader order, no missing feature values. IDs 0–149 are stable local row identities, not chronological timestamps or global specimen identifiers.
- Ordered columns: `row_id,sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,species`.
- Each feature is measured in centimeters. Species codes are 0=setosa, 1=versicolor, 2=virginica, with 50 rows each. Species is a retrospective reference, never a clustering input or hidden parameter-selection target.
- Features are exported with one decimal place, preserving every value in this supplied copy. CSV uses UTF-8 and LF records.
- SHA256: `387c9585511b1d4ab2513075377d2f07c5cbf05e7a04fd934fcb254290a7a651`.

The preceding Clustering Evaluation draft deliberately uses the same source, IDs and column order. Keeping a copy here makes the DBSCAN program runnable offline without requiring that other draft directory. Data may be centrally deduplicated later only if download/export independence and the exact input contract remain clear.

## Analytic question and limitations

The task is to inspect density groupings of this finite measured collection and account for rows left unassigned. All four feature rows are available to the descriptive scaler; this is not a held-out estimate of future performance. Parameters are a stated sensitivity grid, with species revealed afterward. Store retained row IDs whenever noise is excluded from a metric. Four-dimensional clustering and a two-feature plotting projection are different objects. This familiar dataset is small and selected; the lesson does not infer ecological population density from its class-balanced sample.

The invented trail, variable-spacing groups, unit-conversion corners and concentric rings are separate **constructed fixtures**, not claimed observations. Their exact coordinates/generators are supplied in the manuscript and author JSON. The stability tree is an abstract finite tree with declared lifetimes, not an HDBSCAN fit to Iris.

## Calculations and reproducibility boundary

`author-calculations.py` generated the CSV and [author-calculations.json](author-calculations.json) using NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1. It records five trail radii, order reversal, density-variation contrast/null, duplicate and m=1 nulls, OPTICS, selected mutual-reachability values, HDBSCAN, and seven Iris settings. Legitimate undefined OPTICS distances are stored as JSON `null`, while sklearn uses infinity. This serialization choice is explicit; it is not a numerical repair.

`supplementary-author-calculations.py` and its [JSON](supplementary-author-calculations.json) record the small ring comparison, five-row unit-conversion contrast, uniform/restored-metric nulls, four-row accidental null and stability arithmetic. The draft's original four-row practice did not expose the claimed unit-conversion failure; the fifth row fixes that actual content error, while the unchanged four-row case is retained as a lesson in counterexample selection.

Both scripts are modest author calculations. Complete learner programs are written in `lesson.md` with imports, inputs, target results and run instructions. Their exact whole-program stdout execution, implemented diagrams/controls, browser review and independent phase-two verification are **not performed or claimed** in this research-and-writing checkpoint. The JSON supplies inspectable numeric targets; the later stage must bind executions to the exact implemented code.
