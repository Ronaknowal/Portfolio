// Complete displayed programs for the NMF lesson, executed by
// scripts/verify-nmf-examples.py. `file` is the filename the lesson asks the
// learner to save the block as; the digit program runs beside the served CSV.
export const nmfExamples = {
  "multiplicativeStep": {
    "title": "Forty multiplicative sweeps on three constructed observations",
    "question": "the initial factors are not the ones that built X, and the initial loss is 17.06. Will forty sweeps drive the loss to zero, and will the factors return to the ones from section 1?",
    "file": "nmf_step.py",
    "code": "import numpy as np\n\nX = np.array([[2., 1., 3.], [1., 2., 3.], [3., 3., 6.]])\nW = np.array([[1., .5], [.5, 1.], [1., 1.]])\nH = np.array([[1., .2, .8], [.2, 1., .8]])\n\ndef objective(X, W, H):\n    return np.sum((X - W @ H) ** 2) / 2\n\nprint(0, round(objective(X, W, H), 8))\nfor iteration in range(1, 41):\n    H *= (W.T @ X) / ((W.T @ W) @ H)\n    W *= (X @ H.T) / (W @ (H @ H.T))\n    if iteration in (1, 2, 10, 40):\n        print(iteration, round(objective(X, W, H), 8))",
    "expected": "0 17.06\n1 0.03944744\n2 0.02823377\n10 0.00180608\n40 5e-08",
    "language": "python"
  },
  "digitDictionary": {
    "title": "The complete offline digit experiment",
    "question": "eight candidate fits, then a fixed eight-component panel against PCA and the training-mean image. Does the additive constraint cost anything on the reserved images?",
    "file": "nmf_digits.py",
    "code": "import numpy as np\nfrom sklearn.decomposition import NMF, PCA\nfrom sklearn.model_selection import train_test_split\n\ndata = np.loadtxt('digits-300.csv', delimiter=',', skiprows=1)\nsource_rows, labels = data[:, 0].astype(int), data[:, 1].astype(int)\nX = data[:, 2:] / 16\ntrain, rest = train_test_split(np.arange(len(X)), test_size=.4,\n                               random_state=19, stratify=labels)\nvalidation, test = train_test_split(rest, test_size=.5,\n                                    random_state=19, stratify=labels[rest])\n\nfor k in (1, 4, 8, 16):\n    for seed in (7, 19):\n        model = NMF(n_components=k, init='random', solver='cd',\n                    random_state=seed, max_iter=2000, tol=1e-5)\n        W_train = model.fit_transform(X[train])\n        W_validation = model.transform(X[validation])\n        H = model.components_\n        train_mse = np.mean((X[train] - W_train @ H) ** 2)\n        validation_mse = np.mean((X[validation] - W_validation @ H) ** 2)\n        print(k, seed, round(train_mse, 6), round(validation_mse, 6))\n\n# A fixed eight-component comparison chosen for inspectable image panels.\nnmf = NMF(n_components=8, init='random', solver='cd', random_state=19,\n          max_iter=2000, tol=1e-5).fit(X[train])\nW_test = nmf.transform(X[test])\nnmf_reconstruction = W_test @ nmf.components_\npca = PCA(n_components=8, svd_solver='full').fit(X[train])\npca_reconstruction = pca.inverse_transform(pca.transform(X[test]))\nmean_reconstruction = np.repeat(X[train].mean(axis=0)[None, :], len(test), axis=0)\nfor name, reconstruction in [('mean', mean_reconstruction),\n                             ('PCA8', pca_reconstruction),\n                             ('NMF8', nmf_reconstruction)]:\n    print(name, round(np.mean((X[test] - reconstruction) ** 2), 6))\nprint('first held-out source row', source_rows[test[0]])\nprint('first reconstructed image')\nprint(nmf_reconstruction[0].reshape(8, 8).round(2))",
    "expected": "1 7 0.072144 0.072803\n1 19 0.072144 0.072803\n4 7 0.040409 0.041899\n4 19 0.040409 0.0419\n8 7 0.02475 0.025548\n8 19 0.025147 0.026315\n16 7 0.012295 0.013548\n16 19 0.012729 0.015289\nmean 0.069995\nPCA8 0.019678\nNMF8 0.025381\nfirst held-out source row 242\nfirst reconstructed image\n[[0.   0.02 0.19 0.68 0.76 0.37 0.08 0.01]\n [0.   0.06 0.5  0.78 0.87 0.53 0.14 0.  ]\n [0.   0.1  0.38 0.74 0.94 0.58 0.12 0.  ]\n [0.   0.03 0.2  0.92 0.97 0.47 0.07 0.  ]\n [0.   0.08 0.53 1.02 0.98 0.3  0.1  0.  ]\n [0.   0.12 0.72 0.89 0.81 0.26 0.18 0.  ]\n [0.   0.05 0.54 0.81 0.6  0.51 0.29 0.01]\n [0.   0.02 0.2  0.61 0.8  0.7  0.18 0.  ]]",
    "language": "python"
  },
  "wordPatterns": {
    "title": "A constructed five-document transfer",
    "question": "two components, five documents and a generalized-KL objective. Will the two patterns divide the vocabulary, and how does the mixed document use them?",
    "file": "nmf_words.py",
    "code": "import numpy as np\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.decomposition import NMF\n\ndocuments = ['orbit rocket orbit', 'rocket orbit rocket',\n             'goal team goal', 'team goal team',\n             'orbit rocket goal team']\nvectorizer = CountVectorizer()\nX = vectorizer.fit_transform(documents)\nmodel = NMF(n_components=2, init='nndsvda', solver='mu',\n            beta_loss='kullback-leibler', max_iter=1000,\n            tol=1e-5, random_state=19)\nW = model.fit_transform(X)\nH = model.components_\nmass = H.sum(axis=1)\nnormalized_patterns = H / mass[:, None]\namounts = W * mass\nprint(vectorizer.get_feature_names_out())\nprint(normalized_patterns.round(3))\nprint(amounts.round(3))\nnew = vectorizer.transform(['rocket team'])\nprint((model.transform(new) @ H).round(3))",
    "expected": "['goal' 'orbit' 'rocket' 'team']\n[[0.5 0.  0.  0.5]\n [0.  0.5 0.5 0. ]]\n[[0. 3.]\n [0. 3.]\n [3. 0.]\n [3. 0.]\n [2. 2.]]\n[[0.5 0.5 0.5 0.5]]",
    "language": "python"
  }
};
