"""Complementary, source-bound review probes; never reruns MCMC/ADVI fitting."""
import ast
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/teaching/implementation-depth"
checks = []


def passed(name, detail):
    checks.append(dict(name=name, detail=detail, status="passed"))


def load(folder, filename):
    path = ROOT / "public/learn-assets" / folder / filename
    spec = importlib.util.spec_from_file_location(folder.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def topology():
    module = load("topology-tda", "gudhi_rips.py")
    rng = np.random.default_rng(944)
    # Independent graph oracle: beta1 = E - V + components, H0 deaths = MST edges.
    for n in range(1, 9):
        points = rng.uniform(-2, 2, size=(n, 2))
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        edges = sorted((distances[a, b], a, b) for a in range(n) for b in range(a))
        for cutoff in (0., .7, 2., 8.):
            parent = list(range(n))
            def find(i):
                while parent[i] != i:
                    i = parent[i]
                return i
            deaths, edge_count = [], 0
            for weight, a, b in edges:
                if weight > cutoff:
                    break
                edge_count += 1
                a, b = find(a), find(b)
                if a != b:
                    parent[a] = b
                    deaths.append(weight)
            components = len({find(i) for i in range(n)})
            bars = module.compare(points, cutoff, dimension=1)
            assert sum(d == 1 and math.isinf(e) for d, _, e in bars) == edge_count-n+components
            assert sum(d == 0 and math.isinf(e) for d, _, e in bars) == components
            actual = sorted(e for d, _, e in bars if d == 0 and math.isfinite(e))
            np.testing.assert_allclose(actual, sorted(deaths), rtol=1e-12, atol=1e-12)
    passed("topology-graph-and-MST", "32 graph filtrations checked against Euler cycle rank and Kruskal H0 deaths")
    points = np.array([[0., 0.], [3., 0.], [3., 1.], [0., 1.], [1., .4]])
    base = module.compare(points, 5.)
    shifted = module.compare(points[[4, 2, 0, 3, 1]] + [7., -5.], 5.)
    np.testing.assert_allclose(base, shifted, atol=1e-12, rtol=1e-12)
    scaled = module.compare(2*points, 10.)
    np.testing.assert_allclose(np.asarray(scaled)[:, 1:], 2*np.asarray(base)[:, 1:], atol=1e-12)
    passed("topology-invariance-and-scaling", "Permutation/translation preserve interval multiplicity; coordinate/cutoff scaling scales finite endpoints")
    assert module.compare([[0, 0]]*6, 0.) == [(0, 0., math.inf)]
    for points, cutoff, dim in [([], 1., 2), ([[101, 0]], 1., 2), ([[0, 0]], -1., 2), ([[0, 0]], 1., 3)]:
        rejects(lambda p=points, c=cutoff, d=dim: module.compare(p, c, d))
    passed("topology-contract", "Coincident points, empty input, bounded-coordinate/cutoff/dimension contracts")
    print(json.dumps(dict(checks=checks, python=sys.version.split()[0], gudhi=importlib.metadata.version('gudhi'))))


def numerical():
    from scipy.linalg import cho_factor, cho_solve
    from scipy.stats import beta
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.utils.extmath import randomized_svd
    from scipy.sparse.linalg import eigsh
    svd = load("randomized-linear-algebra", "randomized_svd_library.py")
    rng = np.random.default_rng(231)
    for shape, rank, extra in [((7, 4), 1, 0), ((7, 4), 4, 0), ((7, 4), 2, 2), ((4, 4), 3, 1)]:
        a = rng.normal(size=shape)
        before = a.copy()
        for rounds in (0, 1, 3):
            u, s, vt = svd.range_svd(a, rank, extra, rounds, 33)
            u2, s2, vt2 = randomized_svd(a, n_components=rank, n_oversamples=extra,
                n_iter=rounds, power_iteration_normalizer="QR", transpose=False,
                flip_sign=False, random_state=33)
            np.testing.assert_allclose((u*s)@vt, (u2*s2)@vt2, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(s, s2, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(u.T@u, np.eye(rank), atol=1e-12)
            assert np.linalg.norm(a-(u*s)@vt) + 1e-11 >= np.linalg.norm(np.linalg.svd(a, compute_uv=False)[rank:])
        np.testing.assert_array_equal(a, before)
    passed("randomized-SVD-rank-and-iteration-boundaries", "12 exact-route comparisons include p=0, full rank, full probe width, square inputs and three iteration counts; inputs unchanged")
    a = rng.normal(size=(9, 5))
    for bad in [lambda: svd.range_svd(a, 0, 0, 0, 1), lambda: svd.range_svd(a, 3, 3, 0, 1),
                lambda: svd.range_svd(a, 2, 0, -1, 1), lambda: svd.range_svd(a.T, 2, 0, 0, 1)]:
        rejects(bad)
    passed("randomized-SVD-input-contract", "Rank, probe width, iteration and orientation errors rejected")

    ridge = load("functional-analysis", "kernel_ridge_library.py")
    x = np.array([-1., -.4, -.4, .2, 1.1, 2.])
    y = np.array([1., -2., 2., .1, 4., -1.])
    query = np.array([-.7, -.4, 0., 1.7])
    for gamma in (.01, .7, 20.):
        gram = ridge.kernel(x, x, gamma)
        np.testing.assert_allclose(np.diag(gram), 1., atol=0.)
        for penalty in (.001, .3, 10.):
            shift = len(x)*penalty
            coefficient = np.linalg.solve(gram+shift*np.eye(len(x)), y)
            fit = KernelRidge(alpha=shift, kernel="rbf", gamma=gamma).fit(x[:, None], y)
            np.testing.assert_allclose(fit.dual_coef_, coefficient, atol=1e-10, rtol=1e-11)
            np.testing.assert_allclose(fit.predict(query[:, None]), ridge.kernel(query, x, gamma)@coefficient, atol=1e-10, rtol=1e-11)
            np.testing.assert_allclose((gram+shift*np.eye(len(x)))@coefficient, y, atol=1e-10)
    passed("RKHS-varied-regularization-and-kernel", "Nine changed gamma/average-penalty contracts with conflicting duplicates; solve coefficients, held-out query predictions and normal equation agree")

    spectral = load("spectral-graph", "spectral_library.py")
    affinity = np.zeros((6, 6))
    affinity[:3, :3] = 1.; affinity[3:, 3:] = 1.; np.fill_diagonal(affinity, 0.)
    lap, degree = spectral.laplacian(affinity)
    changed, _ = spectral.laplacian(affinity*1e4)
    np.testing.assert_allclose(lap.toarray(), changed.toarray(), atol=1e-15)
    values, vectors = eigsh(lap, k=2, which="SM", v0=np.arange(1., 7.), tol=1e-12)
    expected = np.zeros((6, 2)); expected[:3, 0] = 1/math.sqrt(3); expected[3:, 1] = 1/math.sqrt(3)
    np.testing.assert_allclose(values, 0., atol=1e-12)
    np.testing.assert_allclose(vectors@vectors.T, expected@expected.T, atol=1e-12)
    assert spectral.same_partition(np.array([0, 0, 1, 1]), np.array([7, 7, 9, 9]))
    assert not spectral.same_partition(np.array([0, 0, 1, 1]), np.array([7, 9, 7, 9]))
    assert spectral.normalized_cut(affinity, np.array([0, 0, 0, 1, 1, 1])) == 0.
    passed("spectral-disconnected-subspace-and-scale", "Two repeated zero eigenvalues checked as a full projector, global affinity scaling cancels, pairwise partition relabeling and disconnected cut verified")
    for bad in [affinity+np.eye(6), -affinity, np.pad(affinity, (0, 1)), np.array([[0., 1.], [0., 0.]])]:
        rejects(lambda a=bad: spectral.laplacian(a))
    passed("spectral-input-contract", "Self-loops, negative weights, isolation and directed affinity rejected")

    # Extract only this exact source function; avoid importing or rerunning the inference engine.
    source = ROOT / "public/learn-assets/variational-inference/pymc_variational.py"
    function = next(item for item in ast.parse(source.read_text()).body if isinstance(item, ast.FunctionDef) and item.name == "gaussian_kl")
    namespace = dict(np=np, cho_factor=cho_factor, cho_solve=cho_solve)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    gaussian_kl = namespace["gaussian_kl"]
    for dimension in (1, 2, 4):
        basis = rng.normal(size=(dimension, dimension)); covariance = basis@basis.T + .2*np.eye(dimension)
        target = rng.normal(size=dimension); mean = target+rng.normal(size=dimension); variance = rng.uniform(.05, 2., size=dimension)
        delta = mean-target
        exact = .5*(np.trace(np.linalg.solve(covariance, np.diag(variance))) + delta@np.linalg.solve(covariance, delta)
                    - dimension + np.linalg.slogdet(covariance)[1]-np.log(variance).sum())
        np.testing.assert_allclose(gaussian_kl(mean, variance, target, covariance), exact, atol=1e-12)
        optimum = 1/np.diag(np.linalg.solve(covariance, np.eye(dimension)))
        floor = gaussian_kl(target, optimum, target, covariance)
        assert gaussian_kl(target, 1.3*optimum, target, covariance) > floor
        assert gaussian_kl(target+.1, optimum, target, covariance) > floor
    passed("VI-exact-KL-and-family-optimum", "Exact source Cholesky KL independently compared with solve/slogdet formula in dimensions 1,2,4; mean and variance perturbations raise reverse KL")
    for a, b in ((10, 4), (3, 7)):
        n = a+b-1
        independent_tail = sum(math.comb(n, j)*.7**j*.3**(n-j) for j in range(a))
        np.testing.assert_allclose(beta.sf(.7, a, b), independent_tail, atol=1e-14)
    passed("MCMC-changed-target-reference", "Original and changed Beta posterior tails independently checked through finite binomial-sum identity; no new chain run claimed")


def optimizer_families():
    import torch
    module = load("gradient-variants", "optimizer_library_bridge.py")
    gradients = [None, np.zeros(3), np.array([1e-12, -2e-12, 3e-12]),
                 np.array([.4, -.2, 2.]), None, np.zeros(3),
                 np.array([-1., 0., .5]), np.array([2., -3., 0.]),
                 None, np.array([1e-12, 1e-12, -1e-12]), np.zeros(3)]
    for name in ("sgd", "adagrad", "rmsprop", "adam", "adamw"):
        for initial in (np.zeros(3), np.array([1., -2., .3])):
            parameter = torch.tensor(initial.copy(), dtype=torch.float64, requires_grad=True)
            optimizer = module.make_optimizer(parameter, name)
            expected = initial.copy(); theta = initial.copy()
            first, second, step = np.zeros(3), np.zeros(3), 0
            history = []
            for gradient in gradients:
                saved = [array.copy() for array in (theta, first, second)]
                theta, first, second, step = module.manual_step(theta, first, second, step, gradient, name)
                if gradient is not None:
                    history.append(gradient + (0. if name == "adamw" else .1*expected))
                    # Recompute full weighted histories, independent of the maintained recurrence.
                    g = history[-1]; age = np.arange(len(history)-1, -1, -1)[:, None]
                    h = np.asarray(history)
                    if name == "sgd":
                        direction = np.sum(.9**age*h, axis=0)
                    elif name == "adagrad":
                        moment2 = np.sum(h*h, axis=0)
                        direction = g/(np.sqrt(moment2)+1e-8)
                    elif name == "rmsprop":
                        moment2 = .05*np.sum(.95**age*h*h, axis=0)
                        direction = g/(np.sqrt(moment2)+1e-8)
                    else:
                        moment1 = .1*np.sum(.9**age*h, axis=0)
                        moment2 = .05*np.sum(.95**age*h*h, axis=0)
                        direction = (moment1/(1-.9**len(history)))/(np.sqrt(moment2/(1-.95**len(history)))+1e-8)
                    expected = expected*(.995 if name == "adamw" else 1.) - .05*direction
                else:
                    for actual, before in zip((theta, first, second), saved):
                        np.testing.assert_array_equal(actual, before)
                assert step == len(history)
                np.testing.assert_allclose(theta, expected, atol=1e-12, rtol=1e-12)
                parameter.grad = None if gradient is None else torch.tensor(gradient, dtype=torch.float64)
                optimizer.step()
                np.testing.assert_allclose(parameter.detach().numpy(), expected, atol=1e-12, rtol=1e-12)
                if not history:
                    continue
                state = optimizer.state[parameter]
                if name == "sgd":
                    np.testing.assert_allclose(state['momentum_buffer'].numpy(), direction, atol=1e-12, rtol=1e-12)
                else:
                    assert int(state['step']) == len(history)
                    key = 'sum' if name == 'adagrad' else 'square_avg' if name == 'rmsprop' else 'exp_avg_sq'
                    np.testing.assert_allclose(state[key].numpy(), moment2, atol=1e-12, rtol=1e-12)
                    if name in ('adam', 'adamw'):
                        np.testing.assert_allclose(state['exp_avg'].numpy(), moment1, atol=1e-12, rtol=1e-12)
        passed('optimizer-'+name+'-weighted-history', 'Two three-coordinate initializations and eleven events; leading/interior None, active zero, epsilon-sensitive tiny gradients, coupled/decoupled decay, parameter clock and full weighted-history state oracle')


def main():
    if "--topology" in sys.argv:
        topology()
        return
    numerical()
    optimizer_families()
    executable = ROOT / "scratch/math-library-runtime/Scripts/python.exe"
    child = subprocess.run([str(executable), __file__, "--topology"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if child.returncode:
        raise RuntimeError(child.stdout + child.stderr)
    topology_result = json.loads(child.stdout)
    checks.extend(topology_result['checks'])
    sources = ["public"+topic["programUrl"] for topic in json.loads((REPORT/"math-library-remediation.json").read_text())["topics"]]
    sources.append(Path(__file__).relative_to(ROOT).as_posix())
    result = dict(status="passed", scope="complementary native/source review; no UI and no repeated MCMC/ADVI fitting", checks=checks,
                  runtime=dict(python=sys.version.split()[0], packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch')},
                               topologyPython=topology_result['python'], gudhi=topology_result['gudhi']),
                  sourceHashes={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources})
    (REPORT/"math-library-independent-models.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Passed {len(checks)} complementary groups")


if __name__ == "__main__":
    main()
