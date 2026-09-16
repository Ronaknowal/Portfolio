// Complete displayed programs for the ICA lesson with their actual executed
// output. Regenerate with scripts/verify-ica-examples.py --write; running the
// script without --write proves the recorded output matches a fresh run.
export const icaExamples = {
  "handSeparation": {
    "title": "Separate the exact four-state mixture",
    "question": "the four rows enumerate a designed distribution, not a sample. Will the whitened covariance be the identity, and will each source match exactly one recovered component?",
    "file": "ica_by_hand.py",
    "code": "import numpy as np\n\nS = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])\nA = np.array([[2., 1.], [1., 2.]])\nX = S @ A.T\nmean = X.mean(axis=0)\ncentered = X - mean\nvalues, vectors = np.linalg.eigh(centered.T @ centered / len(X))\nK = (vectors / np.sqrt(values)).T\nZ = centered @ K.T\n\ndef separate(Z, seed=12, max_iter=200, tol=1e-10):\n    rng = np.random.default_rng(seed)\n    W = np.zeros((Z.shape[1], Z.shape[1]))\n    for j in range(len(W)):\n        w = rng.normal(size=Z.shape[1])\n        w -= W[:j].T @ (W[:j] @ w)\n        w /= np.linalg.norm(w)\n        for _ in range(max_iter):\n            y = Z @ w\n            r = Z.T @ (y**3) / len(Z) - (3*y*y).mean() * w\n            r -= W[:j].T @ (W[:j] @ r)\n            new = r / np.linalg.norm(r)\n            change = 1 - abs(new @ w)\n            w = new\n            if change < tol:\n                break\n        else:\n            raise RuntimeError(\"Iteration limit reached\")\n        W[j] = w\n    return W\n\nW = separate(Z)\nestimated = Z @ W.T\nB = W @ K\nC = np.corrcoef(S.T, estimated.T)[:2, 2:]\nreconstructed = estimated @ np.linalg.inv(B).T + mean\nprint(np.round(Z.T @ Z / len(Z), 6))\nprint(np.round(np.max(np.abs(C), axis=1), 6))\nprint(np.max(np.abs(reconstructed - X)) < 1e-10)",
    "language": "python",
    "expected": "[[1. 0.]\n [0. 1.]]\n[1. 1.]\nTrue"
  },
  "realRecording": {
    "title": "Fit blindly, select on development data, evaluate later",
    "question": "predict which of raw channels, PCA coordinates or ICA coordinates will give the largest held-out absolute correlation.",
    "file": "ica_recording.py",
    "code": "import numpy as np\nfrom sklearn.decomposition import FastICA, PCA\n\ndigital = np.loadtxt(\"r01-first20s.csv\", delimiter=\",\", skiprows=1)\nmicrovolts = (digital + 32768) * (6553.6 / 65535) - 3276.8\nreference = microvolts[:, 0]\nX = microvolts[:, 1:]\ntrain = slice(0, 12000)\ndevelopment = slice(12000, 16000)\ntest = slice(16000, 20000)\n\ndef correlation_with_reference(values, target):\n    joined = np.column_stack([values, target])\n    return np.corrcoef(joined, rowvar=False)[-1, :-1]\n\npca = PCA(n_components=4, svd_solver=\"full\").fit(X[train])\nica = FastICA(n_components=4, whiten=\"unit-variance\",\n              whiten_solver=\"svd\", algorithm=\"parallel\", fun=\"logcosh\",\n              random_state=7, max_iter=1000, tol=1e-5).fit(X[train])\nrepresentations = [(\"channel\", X), (\"PCA\", pca.transform(X)),\n                   (\"ICA\", ica.transform(X))]\nfor name, values in representations:\n    dev = correlation_with_reference(values[development], reference[development])\n    chosen = int(np.argmax(np.abs(dev)))\n    result = correlation_with_reference(values[test], reference[test])[chosen]\n    print(name, chosen + 1, f\"{abs(dev[chosen]):.6f}\", f\"{abs(result):.6f}\")\nprint(\"ICA iterations\", ica.n_iter_)",
    "language": "python",
    "expected": "channel 3 0.201203 0.119806\nPCA 4 0.184580 0.450178\nICA 2 0.169804 0.343966\nICA iterations 14"
  }
};
