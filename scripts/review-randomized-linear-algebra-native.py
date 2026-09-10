"""Independent semantic checks of the actual lesson helper, not saved stdout."""

import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/learn/data/randomized-linear-algebra-examples.js"
serialized = subprocess.check_output(
    ["node", "--input-type=module", "-e",
     "import { randomizedLinearAlgebraExamples as e } from "
     "'./src/learn/data/randomized-linear-algebra-examples.js';"
     "process.stdout.write(JSON.stringify(e));"],
    cwd=ROOT, text=True, encoding="utf-8",
)
examples = json.loads(serialized)
helper_trees = []
for key in ("gaussianBaseline", "spectrumBudget", "errorDecomposition", "latentStructurePractice"):
    tree = ast.parse(examples[key]["code"])
    definitions = [part for part in tree.body if isinstance(part, (ast.Import, ast.FunctionDef))]
    helper_trees.append(ast.Module(body=definitions, type_ignores=[]))
assert len({ast.dump(tree) for tree in helper_trees}) == 1, "Standalone helpers have drifted"
scope = {}
exec(compile(helper_trees[0], str(SOURCE), "exec"), scope)
randomized_svd = scope["randomized_svd"]

counts = {"rectangular_helper_cases": 0, "full_range_optimality_cases": 0,
          "zero_or_rank_deficient_cases": 0, "invalid_inputs": 0,
          "embedding_proof_cases": 0, "exhaustive_trace_matrices": 0}
max_decomposition_error = 0.
fixtures = np.random.default_rng(809)

for m, n in ((1, 1), (1, 7), (7, 1), (3, 8), (8, 3), (5, 5), (9, 6)):
    dimension = min(m, n)
    left, _ = np.linalg.qr(fixtures.normal(size=(m, dimension)))
    right, _ = np.linalg.qr(fixtures.normal(size=(n, dimension)))
    for true_rank in sorted({0, 1, dimension // 2, dimension}):
        spectrum = np.zeros(dimension)
        spectrum[:true_rank] = np.geomspace(9., .3, true_rank)
        unit_A = (left * spectrum) @ right.T
        for scale, k, p, q in itertools.product(
            (1.e-80, 1., 1.e80), sorted({1, min(2, dimension), dimension}),
            sorted({0, 2, dimension}), (0, 1, 3)):
            A = scale * unit_A
            U, values, Vt, Q, B = randomized_svd(A, k, p, q, np.random.default_rng(31))
            observed = Q.shape[1]
            output_rank = min(k, observed)
            assert U.shape == (m, output_rank)
            assert values.shape == (output_rank,) and Vt.shape == (output_rank, n)
            assert observed <= min(k + p, dimension, true_rank)
            np.testing.assert_allclose(Q.T @ Q, np.eye(observed), atol=2.e-12)
            np.testing.assert_allclose(U.T @ U, np.eye(output_rank), atol=2.e-12)
            np.testing.assert_allclose(Vt @ Vt.T, np.eye(output_rank), atol=2.e-12)
            np.testing.assert_allclose(B / scale, Q.T @ unit_A, atol=2.e-12)
            reconstruction = (U * (values / scale)) @ Vt
            projection = Q @ (B / scale)
            residual = unit_A - reconstruction
            total = np.sum(residual ** 2)
            outside = np.sum((unit_A - projection) ** 2)
            inside = np.sum((projection - reconstruction) ** 2)
            discrepancy = abs(total - outside - inside)
            max_decomposition_error = max(max_decomposition_error, discrepancy)
            assert discrepancy <= 5.e-10
            # Exact prescribed singular values give the global rank-k floor.
            floor = np.sum(spectrum[k:] ** 2)
            assert total >= floor - 5.e-10
            assert total <= np.sum(unit_A ** 2) + 5.e-10
            # Independently factor the projected full rectangle, not the helper's B.
            projected_spectrum = np.linalg.svd(projection, compute_uv=False)
            np.testing.assert_allclose(values / scale, projected_spectrum[:output_rank], atol=2.e-11)
            if min(k + p, dimension) >= true_rank:
                np.testing.assert_allclose(projection, unit_A, atol=3.e-11)
                np.testing.assert_allclose(total, floor, atol=5.e-10)
                counts["full_range_optimality_cases"] += 1
            if true_rank < dimension:
                counts["zero_or_rank_deficient_cases"] += 1
            counts["rectangular_helper_cases"] += 1

# A deliberately uninformative probe generator is an edge contract, not a Gaussian guarantee.
class ZeroProbes:
    def normal(self, size):
        return np.zeros(size)

U, values, Vt, Q, B = randomized_svd(np.eye(3), 2, 1, 3, ZeroProbes())
assert U.shape == (3, 0) and Vt.shape == (0, 3) and Q.shape == (3, 0)
assert B.shape == (0, 3) and not values.size
assert np.all((U * values) @ Vt == 0)

bad_cases = [([], 1, 0, 0), ([1, 2], 1, 0, 0), (np.empty((3, 0)), 1, 0, 0),
             (np.full((2, 2), np.nan), 1, 0, 0), (np.full((2, 2), np.inf), 1, 0, 0),
             (np.eye(2, dtype=complex), 1, 0, 0), (np.eye(2), 0, 0, 0),
             (np.eye(2), 3, 0, 0), (np.eye(2), 1, -1, 0), (np.eye(2), 1, 0, -1),
             (np.eye(2), 1.5, 0, 0), (np.eye(2), 1, .5, 0), (np.eye(2), 1, 0, .5)]
for A, k, p, q in bad_cases:
    try:
        randomized_svd(A, k, p, q, np.random.default_rng(31))
    except ValueError:
        counts["invalid_inputs"] += 1
    else:
        raise AssertionError(("accepted invalid input", A, k, p, q))

# Verify the proof with arbitrary residual spaces, including dependent features.
for m, d, deficient, epsilon in itertools.product((7, 11), (2, 4), (False, True), (.1, .4, .8)):
    A = fixtures.normal(size=(m, d))
    if deficient:
        A[:, -1] = A[:, 0]
    b = fixtures.normal(size=m)
    residual_columns = np.column_stack((A, b))
    basis, singular, _ = np.linalg.svd(residual_columns, full_matrices=False)
    basis = basis[:, singular > 1.e-12]
    distortion = np.linspace(1 - epsilon, 1 + epsilon, basis.shape[1])
    S = np.diag(np.sqrt(distortion)) @ basis.T
    original, *_ = np.linalg.lstsq(A, b, rcond=None)
    sketched, *_ = np.linalg.lstsq(S @ A, S @ b, rcond=None)
    original_residual = A @ original - b
    sketched_residual = A @ sketched - b
    lower = (1 - epsilon) * (sketched_residual @ sketched_residual)
    middle1 = np.linalg.norm(S @ sketched_residual) ** 2
    middle2 = np.linalg.norm(S @ original_residual) ** 2
    upper = (1 + epsilon) * (original_residual @ original_residual)
    assert lower <= middle1 + 1.e-10 <= middle2 + 2.e-10 <= upper + 3.e-10
    counts["embedding_proof_cases"] += 1

# Enumerate every sign vector to test trace expectation/variance without Monte Carlo noise.
for dimension in (2, 3, 5):
    signs = np.array(list(itertools.product((-1., 1.), repeat=dimension)))
    for _ in range(8):
        raw = fixtures.integers(-5, 6, size=(dimension, dimension)).astype(float)
        for matrix in (raw, (raw + raw.T) / 2):
            probes = np.array([z @ matrix @ z for z in signs])
            np.testing.assert_allclose(np.mean(probes), np.trace(matrix), atol=1.e-12)
            if np.array_equal(matrix, matrix.T):
                variance = 4 * np.sum(np.triu(matrix, 1) ** 2)
                np.testing.assert_allclose(np.var(probes), variance, atol=1.e-12)
            counts["exhaustive_trace_matrices"] += 1

assert np.mean([4, 12]) == 8 and np.mean([4, 12, 12]) > 8
assert all(np.mean(sequence) != 8 for sequence in itertools.product((4, 12), repeat=3))
result = {
    "status": "passed", "checkedAt": datetime.now(timezone.utc).isoformat(),
    "numpy": np.__version__, "examplesSha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    **counts, "identical_standalone_helpers": len(helper_trees),
    "maximum_error_decomposition_discrepancy": max_decomposition_error,
    "practice_fourth_sample_counterexample": "Three 4-or-12 samples cannot have mean 8; use the third sample after [4,12].",
    "bounds": "Small finite fixtures verify implementation and identities, not probabilistic tail guarantees.",
}
output = ROOT / "scratch/randomized-independent-review"
output.mkdir(parents=True, exist_ok=True)
(output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
