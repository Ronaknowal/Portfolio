# Offline Wine data for the PCA lesson

- Dataset: **Wine**, credited by UCI to **Stefan Aeberhard and M. Forina (1992)**. [UCI record](https://archive.ics.uci.edu/dataset/109/wine), [DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).
- License: **Creative Commons Attribution 4.0 International**, as shown on the UCI record when inspected on 12 September 2026. [License terms](https://creativecommons.org/licenses/by/4.0/). Credit the creators and source, link the license and indicate changes when distributing this mirror.
- Purpose and scope: chemical measurements for comparing wines from three cultivars in one Italian region. There are 178 observations, 13 numeric features, three class labels and no missing values in the supplied data. UCI's donation date and dataset citation year differ; the citation above follows its supplied citation rather than inferring a collection year.
- Source actually used: the offline dataset bundled with **scikit-learn 1.9.1**, through [`sklearn.datasets.load_wine`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). A direct UCI ZIP download was attempted but network access from the shell was blocked; no UCI ZIP bytes were used or claimed to have been compared. The browser-accessible UCI metadata establishes attribution/license, and the bundled scikit-learn dataset is the data source for every calculation here.
- Local file: [wine.csv](wine.csv). Changes: add a header from scikit-learn feature names; put cultivar first; map scikit-learn labels 0,1,2 back to 1,2,3; serialize numeric values with 17 significant digits to preserve floating-point round trips. Preserve all observations and their original bundled order. No filtering, imputation or rescaling is applied in this file.
- Column order: cultivar, alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline. “Alcalinity” and “flavanoids” follow the dataset's naming. The header's slash is part of a feature name, not a path.
- Class counts: 59,71,48. Cultivar labels are codes and must be excluded from PCA features. The source metadata does not provide complete measurement units/protocols for every field. Do not invent units, vintage, tasting quality, collection dates or a clinical interpretation from column names.
- Deterministic calculations: [author-calculations.json](author-calculations.json) stores this CSV's SHA256, software versions, split row indices, fitted values and results. Row indices there are zero-based data-row indices, excluding the CSV header. The full-collection exploratory fit and the 133/45 train/validation fit are different analyses and must stay labeled separately.

The reader's main programs use the bundled loader for a simple offline setup. To run the same analysis from this CSV, use:

```python
import numpy as np

table = np.loadtxt("wine.csv", delimiter=",", skiprows=1)
X = table[:, 1:]
cultivar = table[:, 0].astype(int) - 1
```

This file is a required pending handoff input, not a disposable scratch download. In phase two, provide the data and attribution through a real download/runtime asset path. Do not make the browser fetch UCI on every lesson visit.
