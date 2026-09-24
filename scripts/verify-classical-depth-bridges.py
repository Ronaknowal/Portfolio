"""Native checks and displayed output for the scoped implementation-depth bridges."""
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/teaching/evidence/classical-depth-bridges-native.json"


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_manifold():
    module = load("public/learn-assets/manifold-learning/umap-mechanism.py")
    x = np.array([[0.], [0.], [1.], [2.], [4.]])
    ids, distances = module.exact_neighbors(x, 4)
    for i in range(len(x)):
        expected = [i] + sorted((j for j in range(len(x)) if j != i), key=lambda j: (abs(x[j, 0]-x[i, 0]), j))[:3]
        assert ids[i].tolist() == expected
    graph, _, sigma, _ = module.fuzzy_graph(ids, distances)
    np.testing.assert_allclose(graph.toarray(), graph.T.toarray())
    assert np.all((graph.data >= 0) & (graph.data <= 1)) and np.all(sigma > 0)
    ids[0, 1:] = -1
    distances[0, 1:] = np.inf
    disconnected = module.fuzzy_graph(ids, distances)[1]
    assert disconnected.getrow(0).nnz == 0
    coincident = module.fuzzy_graph(*module.exact_neighbors(np.zeros((5, 1)), 4))
    assert np.isfinite(coincident[0].data).all()
    for magnitude in [1e308, 1e45, 1e-50, 1e-170]:
        try:
            module.exact_neighbors(np.array([[magnitude], [-magnitude], [0.]]), 2)
        except ValueError as error:
            assert "Rescale" in str(error)
        else:
            raise AssertionError("Unrepresentable distances must not produce a graph")
    y = np.array([[-1., 0.2], [-0.4, -0.3], [0.2, 0.1], [0.9, -0.2], [1.5, 0.4]])
    loss, gradient = module.full_pair_loss_gradient(y, graph)
    numerical = np.empty_like(y)
    for index in np.ndindex(y.shape):
        plus, minus = y.copy(), y.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (module.full_pair_loss_gradient(plus, graph)[0] - module.full_pair_loss_gradient(minus, graph)[0]) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=3e-8)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-13)
    assert module.full_pair_loss_gradient(y - 0.001 * gradient, graph)[0] < loss
    np.testing.assert_array_equal(module.sampled_edge_step(y, 0, 1, [2], 0), y)
    np.testing.assert_allclose(module.sampled_edge_step(np.array([[0.], [2.]]), 0, 1, [], 0.1), [[0.08], [1.92]])
    return module, ["deterministic duplicate/tie neighbors", "symmetric bounded sparse graph", "removed-neighbor and coincident support", "squared and float32 distance underflow/overflow rejection", "finite-difference full-pair gradient", "translation invariant gradient and descent", "zero-rate and scalar sampled-step oracle"]


def verify_crf():
    module = load("public/learn-assets/crf/crfsuite-bridge.py")
    model = module.fit()
    for words in [["Unknown"], ["Taylor", "enjoys", "running"]]:
        items = module.features(words)
        labels, emissions, transitions = module.chain_arrays(model, items)
        logz, nodes, path = module.infer(emissions, transitions)
        paths = list(itertools.product(range(len(labels)), repeat=len(words)))
        scores = np.array([sum(emissions[t, v] for t, v in enumerate(p)) + sum(transitions[a, b] for a, b in zip(p, p[1:])) for p in paths])
        weights = np.exp(scores - scores.max()); weights /= weights.sum()
        exact = np.array([[sum(w for p, w in zip(paths, weights) if p[t] == v) for v in range(len(labels))] for t in range(len(words))])
        np.testing.assert_allclose(nodes, exact, atol=2e-14)
        assert tuple(path) == paths[int(scores.argmax())]
        np.testing.assert_allclose(logz, scores.max() + np.log(np.exp(scores - scores.max()).sum()), atol=2e-14)
        model.tagger_.set(items)
        for p, probability in zip(paths, weights):
            np.testing.assert_allclose(probability, model.tagger_.probability([labels[i] for i in p]), atol=3e-6)
        changed = transitions.copy(); changed[0, 0] += 1
        if len(words) > 1:
            assert not np.allclose(nodes, module.infer(emissions, changed)[1])
    return module, ["enumerated path partition/marginals/Viterbi", "all package path probabilities from dumped weights", "one-token boundary", "changed transition sensitivity"]


def verify_active():
    module = load("public/learn-assets/active-learning/query-library-bridge.py")
    np.testing.assert_allclose(module.entropy(np.array([[1., 0.], [.5, .5], [.2, .8]])), [0, np.log(2), -(.2*np.log(.2)+.8*np.log(.8))])
    class NoAccess:
        def __getattr__(self, name):
            raise AssertionError("A closed acquisition must not access model or oracle")
        def __call__(self, *args):
            raise AssertionError("A closed acquisition must not call oracle")
    known = np.array([0., np.nan, 1.])
    x = np.arange(3)[:, None]
    _, budget, result = module.acquire_one(x, known, NoAccess(), NoAccess(), NoAccess(), 0)
    assert budget == 0 and result is None
    _, budget, result = module.acquire_one(x, np.array([0., 1., 1.]), NoAccess(), NoAccess(), NoAccess(), 2)
    assert budget == 2 and result is None
    return module, ["entropy zero/mixed/uniform analytic oracle", "zero-budget does not query or reveal", "empty pool does not query or reveal", "main: same-state utilities, acquisition IDs and final refit"]


def old_example(family, index):
    filename, symbol = {"gp": ("gaussian-process", "gaussianProcessExamples"), "lr": ("linear-logistic", "linearLogisticExamples")}[family]
    code = f"import {{{symbol} as examples}} from './src/learn/data/{filename}-examples.js'; console.log(JSON.stringify(examples[{index}].code));"
    source = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", code], cwd=ROOT, text=True, encoding="utf-8"))
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(source, namespace)
    return namespace


def verify_conventions():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.linear_model import LogisticRegression
    gp = old_example("gp", 0)
    x, y, targets = gp["x"], gp["y"], gp["targets"]
    for variance in [.01, .25, 1.]:
        mean, covariance, log_marginal = gp["predict"](x, y, targets, variance)
        library = GaussianProcessRegressor(kernel=RBF(1.0), alpha=variance, optimizer=None, normalize_y=False).fit(x[:, None], y)
        other_mean, other_cov = library.predict(targets[:, None], return_cov=True)
        np.testing.assert_allclose(mean, other_mean, atol=2e-14)
        np.testing.assert_allclose(covariance, other_cov, atol=2e-14)
        np.testing.assert_allclose(log_marginal, library.log_marginal_likelihood(), atol=2e-14)
        observations = GaussianProcessRegressor(kernel=RBF(1.) + WhiteKernel(variance), alpha=0., optimizer=None, normalize_y=False).fit(x[:, None], y)
        obs_mean, obs_cov = observations.predict(targets[:, None], return_cov=True)
        np.testing.assert_allclose(obs_mean, mean, atol=2e-14)
        np.testing.assert_allclose(obs_cov, covariance + variance*np.eye(len(targets)), atol=2e-14)
    lr = old_example("lr", 3)
    model = LogisticRegression(C=1/(len(lr["y"])*lr["penalty"]), l1_ratio=0, solver="lbfgs", max_iter=2000, tol=1e-12).fit(lr["x"][:, None], lr["y"])
    weights = np.r_[model.intercept_, model.coef_[0]]
    np.testing.assert_allclose(weights, lr["weights"], atol=2e-7)
    assert np.linalg.norm(lr["objective"](weights)[1], np.inf) < 1e-7
    doubled = LogisticRegression(C=1/(2*len(lr["y"])*lr["penalty"]), l1_ratio=0, solver="lbfgs", max_iter=2000, tol=1e-12).fit(np.tile(lr["x"], 2)[:, None], np.tile(lr["y"], 2))
    np.testing.assert_allclose(doubled.coef_, model.coef_, atol=2e-7)
    return ["GP: mean/covariance/log evidence at three noise settings", "GP: alpha versus WhiteKernel observation variance", "logistic: lambda = 1/(n C), unpenalized intercept and gradient residual", "logistic: duplicate rows with adjusted C preserve mean objective"]


def main():
    selection = sys.argv[1] if len(sys.argv) > 1 else "conventions"
    old = json.loads(EVIDENCE.read_text()) if EVIDENCE.exists() else {"date": "2026-09-22", "records": {}}
    old["records"][selection] = {"status": "in-progress"}
    EVIDENCE.write_text(json.dumps(old, indent=2) + "\n")
    if selection == "conventions":
        checks, output, sources = verify_conventions(), "", ["src/learn/data/gaussian-process-examples.js", "src/learn/data/linear-logistic-examples.js"]
    else:
        module, checks = {"manifold": verify_manifold, "crf": verify_crf, "active": verify_active}[selection]()
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            module.main()
        output = capture.getvalue().rstrip()
        sources = [str(Path(module.__file__).relative_to(ROOT)).replace("\\", "/")]
        metadata = {"source": "/" + sources[0].removeprefix("public/"), "output": output}
        target = ROOT / f"src/learn/data/{selection}-mechanism-program.js"
        target.write_text("// Generated by scripts/verify-classical-depth-bridges.py.\nexport default " + json.dumps(metadata, indent=2) + ";\n", encoding="utf-8")
        sources.append(str(target.relative_to(ROOT)).replace("\\", "/"))
    versions = {}
    for package in ["numpy", "scipy", "scikit-learn", "umap-learn", "sklearn-crfsuite", "python-crfsuite", "scikit-activeml"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources.append("scripts/verify-classical-depth-bridges.py")
    old["records"][selection] = {"status": "passed", "python": sys.version.split()[0], "versions": versions, "checks": checks, "output": output,
                                 "sourceHashes": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    EVIDENCE.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
    print(selection, "passed", len(checks), "groups")


if __name__ == "__main__":
    main()
