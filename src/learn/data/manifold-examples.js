// Actual executed programs; regenerate with scripts/verify-manifold-examples.py --write.
export const manifoldExamples = {
  "digitsAudit": {
    "title": "Audit the complete optical-digits collection",
    "question": "Will a two-dimensional nonlinear map retain more original pixel neighbors than PCA?",
    "file": "inspect_digits.py",
    "code": "import numpy as np\nfrom sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE, trustworthiness\nfrom sklearn.metrics import pairwise_distances\n\ndata = np.loadtxt(\"digits-300.csv\", delimiter=\",\", skiprows=1)\nsource_row, labels = data[:, 0].astype(int), data[:, 1].astype(int)\nX = data[:, 2:] / 16.0\n\ndef neighbors(points, k):\n    distances = pairwise_distances(points)\n    np.fill_diagonal(distances, np.inf)\n    return np.argsort(distances, axis=1, kind=\"stable\")[:, :k]\n\ndef retention(X, Y, k):\n    return np.mean([len(set(a) & set(b)) / k\n                    for a, b in zip(neighbors(X, k), neighbors(Y, k))])\n\nlayouts = {\"PCA\": PCA(n_components=2, svd_solver=\"full\").fit_transform(X)}\nfor perplexity in (5, 30, 80):\n    layouts[f\"t-SNE p={perplexity}\"] = TSNE(\n        perplexity=perplexity, init=\"pca\", learning_rate=\"auto\",\n        max_iter=1000, random_state=7, method=\"barnes_hut\",\n        angle=0.5, n_jobs=1).fit_transform(X)\n\nfor name, Y in layouts.items():\n    print(name, f\"R10={retention(X, Y, 10):.4f}\",\n          f\"T10={trustworthiness(X, Y, n_neighbors=10):.4f}\")\n    np.savetxt(name.replace(\" \", \"_\") + \".csv\",\n               np.column_stack((source_row, labels, Y)), delimiter=\",\")",
    "language": "python",
    "expected": "PCA R10=0.3763 T10=0.8676\nt-SNE p=5 R10=0.7103 T10=0.9874\nt-SNE p=30 R10=0.7703 T10=0.9901\nt-SNE p=80 R10=0.7623 T10=0.9894"
  },
  "umapFit": {
    "title": "Fit actual UMAP coordinates",
    "question": "How many ten-neighbor selections survive this fitted UMAP map?",
    "file": "inspect_umap.py",
    "code": "import numpy as np\nimport umap\nfrom sklearn.manifold import trustworthiness\nfrom sklearn.metrics import pairwise_distances\n\ndata = np.loadtxt(\"digits-300.csv\", delimiter=\",\", skiprows=1)\nX = data[:, 2:] / 16.0\nreducer = umap.UMAP(n_neighbors=15, min_dist=0.1, spread=1.0,\n                    n_components=2, metric=\"euclidean\", init=\"spectral\",\n                    random_state=7, transform_seed=7, n_jobs=1)\nY = reducer.fit_transform(X)\ndistances = pairwise_distances(X)\nnp.fill_diagonal(distances, np.inf)\ninput_neighbors = np.argsort(distances, axis=1, kind=\"stable\")[:, :10]\nmap_distances = pairwise_distances(Y)\nnp.fill_diagonal(map_distances, np.inf)\nmap_neighbors = np.argsort(map_distances, axis=1, kind=\"stable\")[:, :10]\nR10 = np.mean([len(set(a) & set(b)) / 10\n               for a, b in zip(input_neighbors, map_neighbors)])\nprint(umap.__version__, Y.shape)\nprint(f\"R10={R10:.4f} T10={trustworthiness(X, Y, n_neighbors=10):.4f}\")\nnp.savetxt(\"umap-digits.csv\", np.column_stack((data[:, :2], Y)), delimiter=\",\")",
    "language": "python",
    "expected": "0.5.12 (300, 2)\nR10=0.7143 T10=0.9890"
  },
  "heldoutTransform": {
    "title": "Transform held-out images in a fitted reference",
    "question": "Will ten UMAP coordinates improve this split’s classifier over the pixel and PCA baselines?",
    "file": "compare_transforms.py",
    "code": "import numpy as np\nimport umap\nfrom sklearn.decomposition import PCA\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.neighbors import KNeighborsClassifier\n\ndata = np.loadtxt(\"digits-300.csv\", delimiter=\",\", skiprows=1)\nX, labels = data[:, 2:] / 16.0, data[:, 1].astype(int)\nX_train, X_valid, y_train, y_valid = train_test_split(\n    X, labels, test_size=0.25, stratify=labels, random_state=7)\nrepresentations = {\"pixels\": (X_train, X_valid)}\nfor name, reducer in {\n    \"PCA-10\": PCA(n_components=10, svd_solver=\"full\"),\n    \"UMAP-10\": umap.UMAP(n_components=10, n_neighbors=15,\n                         min_dist=0.1, random_state=7,\n                         transform_seed=7, n_jobs=1)\n}.items():\n    train_coordinates = reducer.fit_transform(X_train)\n    valid_coordinates = reducer.transform(X_valid)\n    representations[name] = (train_coordinates, valid_coordinates)\nfor name, (train, valid) in representations.items():\n    model = KNeighborsClassifier(n_neighbors=5).fit(train, y_train)\n    print(name, f\"accuracy={model.score(valid, y_valid):.4f}\")",
    "language": "python",
    "expected": "pixels accuracy=0.9733\nPCA-10 accuracy=0.9733\nUMAP-10 accuracy=0.9467"
  },
  "tinyTsne": {
    "title": "Follow a tiny exact t-SNE optimizer",
    "question": "Which way will A move on the first update, and will the final normalized KL decrease?",
    "file": "tiny_tsne.py",
    "code": "import numpy as np\n\nX = np.array([0., 1., 3., 7.])[:, None]\nn, perplexity = len(X), 2.0\nconditional = np.zeros((n, n))\nfor i in range(n):\n    others = np.arange(n) != i\n    distances = np.sum((X[others] - X[i])**2, axis=1)\n    lower, upper, beta = 0.0, np.inf, 1.0\n    for _ in range(80):\n        weights = np.exp(-beta * (distances - distances.min()))\n        p = weights / weights.sum()\n        positive = p > 0\n        entropy = -np.sum(p[positive] * np.log(p[positive]))\n        if abs(entropy - np.log(perplexity)) < 1e-10:\n            break\n        if entropy > np.log(perplexity):\n            lower = beta\n            beta = 2 * beta if np.isinf(upper) else (lower + upper) / 2\n        else:\n            upper = beta\n            beta = (lower + upper) / 2\n    conditional[i, others] = p\nP = (conditional + conditional.T) / (2 * n)\n\ndef cost_and_gradient(Y):\n    offsets = Y[:, None, :] - Y[None, :, :]\n    kernel = 1 / (1 + np.sum(offsets**2, axis=2))\n    np.fill_diagonal(kernel, 0)\n    Q = kernel / kernel.sum()\n    mask = P > 0\n    cost = np.sum(P[mask] * np.log(P[mask] / Q[mask]))\n    gradient = 4 * np.sum(((P - Q) * kernel)[:, :, None] * offsets, axis=1)\n    return cost, gradient\n\nY = np.array([-1.5, -0.5, 0.5, 1.5])[:, None]\nprint(f\"initial KL={cost_and_gradient(Y)[0]:.6f}\")\nfor _ in range(200):\n    _, gradient = cost_and_gradient(Y)\n    Y -= 0.5 * gradient\n    Y -= Y.mean(axis=0)\nprint(f\"final KL={cost_and_gradient(Y)[0]:.6f}\")\nprint(np.round(Y.ravel(), 6))",
    "language": "python",
    "expected": "initial KL=0.053522\nfinal KL=0.018227\n[-1.775811 -0.759392  0.42237   2.112833]"
  }
};
