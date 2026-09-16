// Complete displayed programs for the bias-variance and learning-curve
// lesson, executed by scripts/verify-bias-variance-examples.py. `file` is the
// filename the lesson asks the learner to save the block as; `appendedTo`
// marks a block that runs inside an earlier program's namespace.
export const biasVarianceExamples = {
  "finiteWorlds": {
    "title": "Every possible tiny training set, enumerated",
    "question": "eight equally likely training sets, three fitted degrees and two probe inputs. Which fit has zero squared bias at the probe, and does the lowest expected error belong to it?",
    "code": "from itertools import product\nimport numpy as np\n\ntrain_x = np.array([-1., 0., 1.])\nprobe = np.array([0., .5])\ncurvature, sigma = 1., .5\nsigns = np.array(list(product([-1., 1.], repeat=len(train_x))))\ntargets = 1 + train_x + curvature * train_x**2 + sigma * signs\ntruth = 1 + probe + curvature * probe**2\n\nfor degree in (0, 1, 2):\n    design = np.vander(train_x, degree + 1, increasing=True)\n    coefficients = np.linalg.lstsq(design, targets.T, rcond=None)[0]\n    predicted = (\n        np.vander(probe, degree + 1, increasing=True) @ coefficients\n    ).T\n    mean = predicted.mean(axis=0)\n    bias_squared = (mean - truth)**2\n    variance = predicted.var(axis=0)\n    expected_error = ((predicted - truth)**2).mean(axis=0) + sigma**2\n    print(degree, np.round(mean, 6), np.round(bias_squared, 6),\n          np.round(variance, 6), np.round(expected_error, 6))",
    "expected": "0 [1.666667 1.666667] [0.444444 0.006944] [0.083333 0.083333] [0.777778 0.340278]\n1 [1.666667 2.166667] [0.444444 0.173611] [0.083333 0.114583] [0.777778 0.538194]\n2 [1.   1.75] [0. 0.] [0.25     0.179688] [0.5      0.429688]",
    "language": "python",
    "file": "bias_variance_worlds.py"
  },
  "airfoilLearningCurve": {
    "title": "Learning curves for four prespecified procedures",
    "question": "four procedures, five fitted sizes and five folds. Which of them improves most as training size grows, and does requiring twenty items per leaf help at every size?",
    "code": "import numpy as np\nfrom sklearn.dummy import DummyRegressor\nfrom sklearn.linear_model import Ridge\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import KFold, train_test_split, learning_curve\n\ndata = np.loadtxt(\"airfoil-self-noise.dat\")\nx, y = data[:, :5], data[:, 5]\ndevelopment, untouched = train_test_split(\n    np.arange(len(y)), train_size=1200, random_state=41\n)\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\nmodels = {\n    \"mean\": DummyRegressor(strategy=\"mean\"),\n    \"ridge\": make_pipeline(StandardScaler(), Ridge(alpha=1.)),\n    \"tree_leaf1\": DecisionTreeRegressor(\n        min_samples_leaf=1, random_state=43),\n    \"tree_leaf20\": DecisionTreeRegressor(\n        min_samples_leaf=20, random_state=43),\n}\nfor name, model in models.items():\n    sizes, train_scores, valid_scores = learning_curve(\n        model, x[development], y[development],\n        train_sizes=[60, 120, 240, 480, 900], cv=cv,\n        scoring=\"neg_mean_squared_error\",\n        shuffle=True, random_state=44, n_jobs=1,\n    )\n    print(name)\n    for n, tr, va in zip(sizes, train_scores, valid_scores):\n        print(n, round(-tr.mean(), 4), round(-va.mean(), 4))",
    "expected": "mean\n60 43.1969 45.4545\n120 41.0584 45.081\n240 44.2372 45.2469\n480 43.4767 45.2065\n900 45.0987 45.222\nridge\n60 17.7898 24.9831\n120 20.0128 23.8044\n240 21.2317 23.6478\n480 22.0263 23.4994\n900 22.7528 23.372\ntree_leaf1\n60 -0.0 41.8405\n120 -0.0 24.4041\n240 -0.0 20.1754\n480 -0.0 13.7464\n900 -0.0 8.1412\ntree_leaf20\n60 32.8389 39.9286\n120 23.75 32.3389\n240 21.9108 28.1945\n480 16.1018 21.6109\n900 12.5438 16.6298",
    "language": "python",
    "file": "airfoil_learning_curve.py"
  },
  "settingAndTrajectory": {
    "title": "One changed setting, then one training trajectory",
    "question": "the same development pool, a restriction grid and a boosting run. Does restricting the leaf size improve the score here, and where does the monitoring error reach its lowest inspected value?",
    "code": "from sklearn.model_selection import validation_curve\nfrom sklearn.ensemble import GradientBoostingRegressor\n\nleaves = [1, 2, 5, 10, 20, 40]\ntr, va = validation_curve(\n    DecisionTreeRegressor(random_state=43), x[development], y[development],\n    param_name=\"min_samples_leaf\", param_range=leaves, cv=cv,\n    scoring=\"neg_mean_squared_error\", n_jobs=1,\n)\nfor size, train_score, valid_score in zip(leaves, tr, va):\n    print(\"leaf\", size, round(-train_score.mean(), 4),\n          round(-valid_score.mean(), 4))\n\nfit_ids, monitor_ids = train_test_split(\n    development, test_size=240, random_state=45\n)\nboosted = GradientBoostingRegressor(\n    n_estimators=120, max_depth=2, learning_rate=.1, random_state=46\n).fit(x[fit_ids], y[fit_ids])\ntrain_mse = [np.mean((y[fit_ids] - p)**2)\n             for p in boosted.staged_predict(x[fit_ids])]\nmonitor_mse = [np.mean((y[monitor_ids] - p)**2)\n               for p in boosted.staged_predict(x[monitor_ids])]\nfor i in [0, 9, 29, 59, 119]:\n    print(\"round\", i + 1, round(train_mse[i], 4),\n          round(monitor_mse[i], 4))\nprint(\"best inspected round\", int(np.argmin(monitor_mse)) + 1)",
    "expected": "leaf 1 -0.0 8.2079\nleaf 2 1.267 8.5475\nleaf 5 4.5952 10.1516\nleaf 10 8.2474 12.3799\nleaf 20 12.2123 15.7762\nleaf 40 18.8711 21.8071\nround 1 41.5026 42.9204\nround 10 26.9413 28.0979\nround 30 17.5512 20.0168\nround 60 13.0405 15.3584\nround 120 9.1042 12.3338\nbest inspected round 120",
    "language": "python",
    "appendedTo": "airfoil_learning_curve.py"
  }
};
