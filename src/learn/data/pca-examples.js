// Complete displayed programs for the PCA lesson, executed by scripts/verify-pca-examples.py.
// Programs with `continues` extend the named program's namespace, as the lesson instructs.
export const pcaExamples = {
  "svd": {
    "title": "Center, decompose and reconstruct the four points with NumPy",
    "question": "Which shape does `directions` have, and why does the forward multiplication need `.T` while the reconstruction does not?",
    "code": "import numpy as np\n\nX = np.array([[1., 1.], [2., 0.], [4., 4.], [5., 3.]])\nmean = X.mean(axis=0)\ncentered = X - mean\n_, singular_values, directions = np.linalg.svd(centered, full_matrices=False)\n\nk = 1\nkept_directions = directions[:k]       # shape: (k, d)\nscores = centered @ kept_directions.T # shape: (n, k)\nreconstructed = scores @ kept_directions + mean\nvariances = singular_values**2 / (len(X) - 1)\n\nprint(np.round(variances, 4))\nprint(np.round(variances / variances.sum(), 4))\nprint(np.round(reconstructed, 4))\nprint(round(np.mean((X - reconstructed)**2), 4))",
    "expected": "[6.     0.6667]\n[0.9 0.1]\n[[1.5 0.5]\n [1.5 0.5]\n [4.5 3.5]\n [4.5 3.5]]\n0.25",
    "language": "python"
  },
  "library": {
    "title": "The library version exposes the same operations",
    "question": "The new observation (6, 4) was not part of the fit. Which fitted quantities does `transform` reuse, and what would change if you refitted with it included?",
    "continues": "svd",
    "code": "# Continue the NumPy program above: X, mean and directions are already defined.\nfrom sklearn.decomposition import PCA\n\npca = PCA(n_components=1, svd_solver=\"full\")\nscores = pca.fit_transform(X)\nreconstructed = pca.inverse_transform(scores)\n\nnew_observation = np.array([[6., 4.]])\nnew_scores = pca.transform(new_observation)\nnew_reconstruction = pca.inverse_transform(new_scores)\n\nprint(pca.components_.shape, scores.shape)\nprint(np.round(new_reconstruction, 4))",
    "expected": "(1, 2) (4, 1)\n[[5.5 4.5]]",
    "language": "python"
  },
  "wineScaling": {
    "title": "Raw and standardized PCA on all 178 wines",
    "question": "Will the raw first component retain more or less than half of the variance, and which single measurement do you expect to dominate it?",
    "code": "import numpy as np\nfrom sklearn.datasets import load_wine\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\nwine = load_wine()  # included with scikit-learn; no dataset download\nX, cultivar = wine.data, wine.target\n\nraw_pca = PCA(svd_solver=\"full\").fit(X)\nscaler = StandardScaler().fit(X)\nstandardized = scaler.transform(X)\nscaled_pca = PCA(svd_solver=\"full\").fit(standardized)\nscores = scaled_pca.transform(standardized)\n\nprint(X.shape, np.bincount(cultivar))\nprint(np.round(raw_pca.explained_variance_ratio_[:2], 4))\nprint(np.round(scaled_pca.explained_variance_ratio_[:3], 4))\nprint(round(scaled_pca.explained_variance_ratio_[:2].sum(), 4))",
    "expected": "(178, 13) [59 71 48]\n[0.9981 0.0017]\n[0.362  0.1921 0.1112]\n0.5541",
    "language": "python"
  },
  "budget": {
    "title": "Choose the smallest component count that meets a validation error budget",
    "question": "Which rows fit the scaler and the PCA, which rows are scored, and what does a ratio of 0.10 mean relative to always returning the training mean?",
    "code": "import numpy as np\nfrom sklearn.datasets import load_wine\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\nwine = load_wine()\ntrain, validation = train_test_split(\n    np.arange(len(wine.data)), test_size=0.25,\n    random_state=42, stratify=wine.target\n)\nscaler = StandardScaler().fit(wine.data[train])\nA = scaler.transform(wine.data[train])\nB = scaler.transform(wine.data[validation])\npca = PCA(svd_solver=\"full\").fit(A)\n\nbaseline_mse = np.mean((B - pca.mean_)**2)\nratios = []\nfor k in range(14):\n    directions = pca.components_[:k]\n    scores = (B - pca.mean_) @ directions.T\n    reconstruction = scores @ directions + pca.mean_\n    ratios.append(np.mean((B - reconstruction)**2) / baseline_mse)\n\nprint(len(train), len(validation))\nprint(round(baseline_mse, 4))\nfor k in [0, 2, 7, 8, 10, 13]:\n    print(k, round(ratios[k], 4))\nprint(next(k for k, ratio in enumerate(ratios) if ratio <= 0.10))",
    "expected": "133 45\n1.0963\n0 1.0\n2 0.4281\n7 0.1265\n8 0.096\n10 0.0521\n13 0.0\n8",
    "language": "python"
  },
  "originalUnits": {
    "title": "Recover approximate measurements in their original units",
    "question": "The output has thirteen columns. How many independently stored numbers per wine produced them?",
    "continues": "budget",
    "code": "# Continue the training/validation program above.\nk = 8\ndirections = pca.components_[:k]\nscores = (B - pca.mean_) @ directions.T\nreconstruction = scores @ directions + pca.mean_\noriginal_units = scaler.inverse_transform(reconstruction)\n\nprint(original_units.shape)",
    "expected": "(45, 13)",
    "language": "python"
  },
  "eigen": {
    "title": "Check that covariance eigenvectors and SVD give the same PCA",
    "question": "Why does the comparison use reconstructions rather than the direction vectors themselves?",
    "continues": "svd",
    "code": "# Continue the four-point NumPy example in section 4.\ncovariance = centered.T @ centered / (len(X) - 1)\neigenvalues, eigenvectors = np.linalg.eigh(covariance)\nleading = eigenvectors[:, -1:]\neigen_reconstruction = (centered @ leading) @ leading.T + mean\n\nprint(np.round(eigenvalues[::-1], 4))\nprint(np.allclose(eigen_reconstruction, reconstructed))\nprint(round(float(np.sum(singular_values[1:]**2)), 4))",
    "expected": "[6.     0.6667]\nTrue\n2.0",
    "language": "python"
  },
  "gaussian": {
    "title": "A noise-only sample still has a leading component",
    "question": "The population covariance is the identity, so every two directions hold exactly 10% of population variance. Will the first two sample components hold exactly 10%?",
    "code": "import numpy as np\nfrom sklearn.decomposition import PCA\n\nrng = np.random.default_rng(23)\nnoise = rng.normal(size=(40, 20))\npca = PCA(svd_solver=\"full\").fit(noise)\nprint(round(pca.explained_variance_ratio_[:2].sum(), 4))",
    "expected": "0.2459",
    "language": "python"
  },
  "pipeline": {
    "title": "Evaluate PCA inside a prediction pipeline with a no-PCA baseline",
    "question": "Where are the scaler and PCA fitted in each fold, and what would be wrong with fitting them once on all 178 rows first?",
    "code": "import numpy as np\nfrom sklearn.datasets import load_wine\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\n\nwine = load_wine()\nfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nfor k in [None, 2, 8]:\n    reduction = \"passthrough\" if k is None else PCA(\n        n_components=k, svd_solver=\"full\"\n    )\n    pipeline = make_pipeline(\n        StandardScaler(), reduction, LogisticRegression(max_iter=2000)\n    )\n    accuracy = cross_val_score(\n        pipeline, wine.data, wine.target,\n        cv=folds, scoring=\"accuracy\", n_jobs=1\n    )\n    print(k, np.round(accuracy, 4), round(accuracy.mean(), 4))",
    "expected": "None [0.9722 0.9722 0.9722 1.     1.    ] 0.9833\n2 [0.9444 0.9722 0.9167 0.9714 0.9714] 0.9552\n8 [0.9722 0.9722 0.9722 0.9714 1.    ] 0.9776",
    "language": "python"
  }
};
