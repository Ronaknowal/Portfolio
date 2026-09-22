"""Execute topic-owned bridges, retain exact output, and bind author evidence.

Run with the existing lesson Python. Missing comparison packages can be installed
in scratch/math-depth-remediation/extra-dependencies; the shared runtime is not
modified. --refresh-manifest updates body/doc bindings only after checking that
all previously executed program hashes still match.
"""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ROOT / "scratch/math-depth-remediation/extra-dependencies"
if EXTRA.exists():
    sys.path.insert(0, str(EXTRA))
    os.environ["PYTHONPATH"] = str(EXTRA) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Stable ID, legacy body filename, local example owner, new program namespace/path.
TOPICS = [
    ("matrix-calculus-jacobians", None, "matrix-calculus", "matrix-calculus", "derivative-library-bridge.py"),
    ("multivariate-calculus-gradients", None, "multivariate-calculus", None, None),
    ("second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient", None, "second-order-methods", None, None),
    ("constrained-multi-objective-optimization", None, "constrained-multiobjective", "constrained-optimization", "simplex-solver-bridge.py"),
    ("optimal-transport-wasserstein-distance-sinkhorn", None, "optimal-transport", "optimal-transport", "sinkhorn-library-bridge.py"),
    ("causal-inference-do-calculus", None, "causal-inference", "causal-inference", "causal-estimation-bridge.py"),
    ("mutual-information-information-bottleneck", None, "mutual-information", "mutual-information", "mutual-information-library-bridge.py"),
    ("f-divergences-integral-probability-metrics", None, "divergence-ipm", "divergences", "mmd-library-bridge.py"),
    ("graph-fundamentals-adjacency-laplacian-connectivity", None, "graph-fundamentals", "graph-fundamentals", "laplacian-library-bridge.py"),
    ("stochastic-processes-markov-chains-brownian-motion-poisson", None, "stochastic-processes", None, None),
    ("queueing-theory-m-m-1-m-g-1-little-s-law", "queueing-theory-m-m-1-m-g-1-littles-law", "queueing", "queueing", "fcfs-simpy-bridge.py"),
    ("it-calculus-stochastic-differential-equations", "ito-calculus-stochastic-differential-equations", "ito-sde", "ito-calculus", "sde-library-bridge.py"),
    ("numerical-pdes-grids-finite-elements-stability", None, "numerical-pde", "numerical-pdes", "finite-element-banded-bridge.py"),
]
EVIDENCE = "docs/teaching/evidence/math-depth-remediation-author.json"
MANIFEST = "docs/teaching/implementation-depth/MATH-DEPTH-REMEDIATION.json"


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load_program(namespace, filename):
    path = f"public/learn-assets/{namespace}/{filename}"
    spec = importlib.util.spec_from_file_location(namespace.replace("-", "_"), ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(receipt):
    rows = []
    for stable_id, alias, owner, namespace, filename in TOPICS:
        body = f"src/learn/data/topics/{alias or stable_id}.jsx"
        reviewed = [body, f"src/learn/data/{owner}-examples.js"]
        changed = [body]
        assets = []
        if namespace:
            assets = [f"public/learn-assets/{namespace}/{filename}"]
            changed += [f"src/learn/data/{namespace}-mechanism-program.js"] + assets
            reviewed += changed[1:]
            anchor = ("constrained" if namespace == "constrained-optimization" else namespace) + "-code-route"
        else:
            anchor = {"multivariate-calculus": "multivariate-calculus", "second-order-methods": "second-order",
                      "stochastic-processes": "stochastic-processes"}[owner] + "-code-route"
        for path in reviewed:
            if path.startswith("public/"):
                assert receipt["programs"][namespace]["sha256"] == digest(path), "Executed source changed"
        rows.append({"id": stable_id, "body": body, "status": "changed",
                     "disposition": "new executable bridge" if namespace else "retained mechanism with explicit reuse/boundary",
                     "anchor": anchor, "changed": changed, "assets": assets,
                     "authorEvidence": [EVIDENCE], "hashes": {path: digest(path) for path in reviewed},
                     "reusedProgram": "public/learn-assets/matrix-calculus/derivative-library-bridge.py" if owner == "multivariate-calculus" else None})
    result = {"schemaVersion": 1, "date": "2026-09-22", "topics": rows,
              "verifier": {"path": "scripts/verify-math-depth-remediation.py", "sha256": digest("scripts/verify-math-depth-remediation.py")},
              "sharedUiReused": {path: digest(path) for path in ["src/learn/components/lesson-labs/MechanismProgram.jsx", "src/learn/components/lesson-labs/mechanism-program.css"]},
              "authorEvidence": {"path": EVIDENCE, "sha256": digest(EVIDENCE)},
              "integration": "Root owns independent review and final production/browser checks; no author build or server."}
    report = "docs/teaching/implementation-depth/MATH-DEPTH-REMEDIATION.md"
    if (ROOT / report).exists():
        result["report"] = {"path": report, "sha256": digest(report)}
    (ROOT / MANIFEST).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if "--refresh-manifest" in sys.argv:
    manifest(json.loads((ROOT / EVIDENCE).read_text(encoding="utf-8")))
    print("Refreshed thirteen source bindings; executed program hashes unchanged")
    raise SystemExit

import numpy as np
import scipy
import sklearn
import torch
torch.set_num_threads(1)
checks = []
programs = {}
modules = {}
for _, _, _, namespace, filename in TOPICS:
    if not namespace:
        continue
    path = f"public/learn-assets/{namespace}/{filename}"
    run = subprocess.run([sys.executable, str(ROOT/path)], capture_output=True, text=True, check=True, timeout=120)
    output = run.stdout.strip()
    metadata = {"source": path.removeprefix("public"), "output": output}
    (ROOT/f"src/learn/data/{namespace}-mechanism-program.js").write_text(
        "// Exact stdout from the topic-owned Python program; generated by verify-math-depth-remediation.py.\n"
        + "export const mechanismProgram = " + json.dumps(metadata, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    programs[namespace] = {"path": path, "sha256": digest(path), "stdout": output, "stderr": run.stderr.strip()}
    modules[namespace] = load_program(namespace, filename)
    checks.append(namespace + ": complete learner program and its assertions executed")
    print(checks[-1], flush=True)


def close(name, actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    checks.append(name)


def rejects(name, action):
    try:
        action()
    except (ValueError, FloatingPointError):
        checks.append(name)
    else:
        raise AssertionError(name)


affine = modules["matrix-calculus"].affine_loss_pullback
rng = np.random.default_rng(22)
x, w, b, target = rng.normal(size=(5, 3)), rng.normal(size=(3, 4)), rng.normal(size=4), rng.normal(size=(5, 4))
original = affine(x, w, b, target)
repeat = affine(np.tile(x, (2, 1)), w, b, np.tile(target, (2, 1)))
close("affine duplicated batch: objective", repeat[0], original[0])
close("affine duplicated batch: each input derivative halves", repeat[1][:5], original[1]/2)
close("affine duplicated batch: parameter derivatives invariant", repeat[2], original[2])
close("affine duplicated batch: bias derivative invariant", repeat[3], original[3])
rejects("affine rejects accidental broadcast target", lambda: affine(x, w, b, target[:1]))
simplex = modules["constrained-optimization"]
for budget in [.2, 1., 7.]:
    v = rng.normal(size=13)
    p = simplex.project_simplex(v, budget)
    close(f"simplex budget {budget}: conservation", p.sum(), budget)
    close(f"simplex budget {budget}: idempotence", simplex.project_simplex(p, budget), p)
    close(f"simplex budget {budget}: translation invariance", simplex.project_simplex(v+100, budget), p)
    assert np.all(p >= 0)
transport = modules["optimal-transport"]
a, b = np.array([.2, .8]), np.array([.3, .4, .3])
cost = np.array([[0., 1., 2.], [2., .5, 1.]])
p, state = transport.sinkhorn_log(a, b, cost, .5)
q, _ = transport.sinkhorn_log(a, b, 3*cost, 1.5)
close("Sinkhorn joint cost/epsilon scaling preserves plan", p, q)
close("Sinkhorn rectangular row conservation", p.sum(axis=1), a)
close("Sinkhorn rectangular column conservation", p.sum(axis=0), b)
rejects("Sinkhorn zero-mass contract", lambda: transport.sinkhorn_log([0., 1.], b, cost, .5))
causal = modules["causal-inference"]
manual, fitted, _ = causal.estimate([30, 60, 70, 40], [3, 18, 21, 20])
close("causal changed target, exact .2 stratum contrast", manual, [.2]*3)
close("causal changed target GLM parity", fitted, manual)
rejects("causal no-overlap rejection", lambda: causal.scores(np.array([1]), np.array([1]), np.array([1.]), np.array([0.]), np.array([1.])))
mi = modules["mutual-information"]
counts = np.array([[8, 0, 3], [2, 7, 1]])
close("MI unused category invariance", mi.count_mi(np.pad(counts, ((0, 1), (0, 1)))), mi.count_mi(counts))
rejects("MI rejects normalized probabilities at count boundary", lambda: mi.count_mi(counts/counts.sum()))
mmd = modules["divergences"]
sample = np.array([[0.], [1.]])
for block in [1, 2, 7]:
    close(f"MMD block {block}: identical biased samples", mmd.mmd2(sample, sample, block_size=block), 0)
    close(f"MMD block {block}: negative unbiased estimator retained", mmd.mmd2(sample, sample, block_size=block, unbiased=True), np.exp(-.5)-1)
graph = modules["graph-fundamentals"]
adjacency = graph.adjacency(4, [(0, 1, 4.), (1, 2, 1.), (2, 3, 0.)])
close("graph changed weighted harmonic value", graph.harmonic_extension(adjacency, [0, 2, 3], [0., 3., 7.]), [0., .6, 3., 7.])
rejects("graph unanchored zero-edge isolate", lambda: graph.harmonic_extension(adjacency, [0, 2], [0., 3.]))
queue = modules["queueing"]
arrivals = np.sort(rng.integers(0, 20, 40)).astype(float)
service = rng.integers(0, 4, 40).astype(float)
close("queue forty jobs including ties and zero service", queue.fcfs(arrivals, service), queue.simulate(arrivals, service))
rejects("queue rejects unsorted arrivals", lambda: queue.fcfs([1, 0], [1, 1]))
sde = modules["ito-calculus"]
noise = rng.normal(0, .1, (20, 2)); grid = np.linspace(0, .2, 21)
path = sde.euler_maruyama(lambda x,t: np.zeros(2), lambda x,t: np.array([[1., 0.], [0., 0.]]), [0., 2.], grid, noise)
close("SDE unused driver leaves second coordinate fixed", path[:, 1], np.full(21, 2.))
close("SDE first coordinate sums matching driver", path[:, 0], np.r_[0., noise[:, 0].cumsum()])
rejects("SDE rejects nonuniform comparison grid", lambda: sde.euler_maruyama(lambda x,t: x, lambda x,t: np.eye(1), [1.], [0., .1, .3], [[0.], [0.]]))
fem = modules["numerical-pdes"]
nodes = np.array([0., .1, .35, .7, 1.])
diag, off, rhs = fem.assemble(nodes, np.ones(4), np.zeros(4), point_load=(.35, 1.))
close("FEM on-node point source load", rhs, [0., 1., 0.])
close("FEM on-node exact Green nodal values", fem.solve_ldl(fem.factor_ldl(diag, off), rhs), [.065, .2275, .105])
rejects("FEM rejects nonpositive conductivity", lambda: fem.assemble(nodes, [1., 0., 1., 1.], np.zeros(4)))
rejects("LDL rejects indefinite pivot", lambda: fem.factor_ldl([1., 1.], [2.]))

# Retained complete programs in the two assessment-without-new-algorithm topics.
legacy = {}
for owner in ["second-order-methods", "stochastic-processes"]:
    path = f"src/learn/data/{owner}-examples.js"
    javascript = f"import * as m from './{path}'; console.log(JSON.stringify(Object.values(m)[0]));"
    exported = json.loads(subprocess.run(["node", "--input-type=module", "-e", javascript], cwd=ROOT,
                                          check=True, text=True, capture_output=True).stdout)
    for name, example in exported.items():
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(example["code"], f"{path}:{name}", "exec"), {"__name__": "__main__"})
        actual = output.getvalue().strip()
        assert actual == example["expected"].strip(), (owner, name, actual, example["expected"])
        checks.append(f"retained {owner}/{name}: exact expected output replayed")
    legacy[path] = {"sha256": digest(path), "programCount": len(exported)}

receipt = {"date": "2026-09-22", "kind": "author checks; independent review remains separate",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "torch": torch.__version__},
           "checks": checks, "count": len(checks), "programs": programs, "retainedPrograms": legacy}
(ROOT/EVIDENCE).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
manifest(receipt)
print(f"Passed {len(checks)} author groups; ten outputs recorded and thirteen topics bound")
