import contextlib
import io
import itertools
import json
from pathlib import Path
import runpy
import sys
import numpy as np

directory = Path(sys.argv[1])
cases = json.loads((directory / "model-cases.json").read_text())
for case in cases["contractions"]:
    result = case["result"]
    operands = [np.array(value) for value in case["operands"]]
    reference = np.einsum(case["expression"], *operands)
    assert list(reference.shape) == result["outputShape"], case["expression"]
    np.testing.assert_array_equal(reference.ravel(), [cell["value"] for cell in result["cells"]])
    for cell in result["cells"]:
        expected = reference[tuple(cell["coordinates"])] if reference.ndim else reference.item()
        assert cell["value"] == expected
        # Every depicted address is checked against native indexing, separately
        # from comparing the aggregate einsum result.
        for term in cell["terms"]:
            product = 1
            for source in term["sources"]:
                native = operands[source["operand"]][tuple(source["coordinates"])]
                assert source["value"] == native
                product *= native
            assert term["product"] == product

for case in cases["attention"]:
    fixture = case["fixture"]
    q = np.array(fixture["queries"][case["query"]])
    keys, values = np.array(fixture["keys"]), np.array(fixture["values"])
    scores = keys @ q / (np.sqrt(q.size) if case["scaled"] else 1)
    mask = np.array(case["allowed"])
    # Direct exponentials are independently safe for these tiny score ranges.
    weights = np.where(mask, np.exp(scores), 0)
    weights /= weights.sum()
    result = case["result"]
    np.testing.assert_allclose([row["score"] for row in result["rows"]], scores)
    np.testing.assert_allclose([row["weight"] for row in result["rows"]], weights)
    np.testing.assert_allclose(result["context"], weights @ values)
    np.testing.assert_allclose([row["contribution"] for row in result["rows"]], values * weights[:, None])

for case in cases["bases"]:
    result = case["result"]
    S, v = np.array(result["S"]), np.array(case["vector"])
    coords = np.linalg.solve(S, v)
    f = np.array([2, -1])
    A = np.array([[2, 1], [0, 1]])
    np.testing.assert_allclose(result["coordinates"], coords)
    np.testing.assert_allclose(result["newFunctional"], S.T @ f)
    np.testing.assert_allclose(result["newMap"], np.linalg.solve(S, A @ S))
    np.testing.assert_allclose(result["metric"], S.T @ S)
    np.testing.assert_allclose(result["reconstructed"], v)
    np.testing.assert_allclose(result["mappedNew"], np.linalg.solve(S, A @ v))
    assert result["measurement"] == result["newMeasurement"] == f @ v
    np.testing.assert_allclose(result["normSquared"], v @ v)
    np.testing.assert_allclose(result["metricNormSquared"], v @ v)

def multiply_count(rows, inner, columns):
    # Enumerate the products in the straightforward output-cell algorithm.
    return sum(1 for _ in itertools.product(range(rows), range(columns), range(inner)))

for case in cases["orders"]:
    a, b, c, d = case["dimensions"]
    left, right = case["result"]["left"], case["result"]["right"]
    assert left["total"] == multiply_count(a, b, c) + multiply_count(a, c, d)
    assert right["total"] == multiply_count(b, c, d) + multiply_count(a, b, d)
    assert left["firstSize"] == np.empty((a, c)).size
    assert right["firstSize"] == np.empty((b, d)).size

with contextlib.redirect_stdout(io.StringIO()):
    attention = runpy.run_path(str(directory / "attention.py"))["scaled_attention"]
rng = np.random.default_rng(819)
unseen = 0
for b, t, s, d, f in itertools.product([1, 2], [1, 3], [1, 4], [1, 3], [1, 2]):
    q = rng.normal(size=(b, t, d))
    k = rng.normal(size=(b, s, d))
    v = rng.normal(size=(b, s, f))
    allowed = rng.random((b, t, s)) > .4
    allowed[:, :, 0] = True
    weights, actual = attention(q, k, v, allowed)
    expected = np.zeros((b, t, f))
    for batch in range(b):
        for query in range(t):
            permitted = [key for key in range(s) if allowed[batch, query, key]]
            scores = [sum(q[batch, query, j] * k[batch, key, j] for j in range(d)) / np.sqrt(d) for key in permitted]
            raw = np.exp(scores)
            probabilities = raw / raw.sum()
            for probability, key in zip(probabilities, permitted):
                expected[batch, query] += probability * v[batch, key]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=-1), 1)
    assert np.all(weights[~allowed] == 0)
    unseen += 1

invalid_attention = [
    (np.ones((1, 2)), np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.ones((1, 2, 2))),
    (np.ones((1, 2, 2)), np.ones((1, 2, 3)), np.ones((1, 2, 2)), np.ones((1, 2, 2))),
    (np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.zeros((1, 2, 2))),
    (np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.ones((1, 2, 3))),
    (np.full((1, 2, 2), np.nan), np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.ones((1, 2, 2))),
    (np.ones((1, 2, 0)), np.ones((1, 2, 0)), np.ones((1, 2, 2)), np.ones((1, 2, 2))),
]
for args in invalid_attention:
    try:
        attention(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid attention input to fail")

extra = 0
for count in [2, 3, 7]:
    for features in [2, 4]:
        data = rng.normal(size=(2, count, features))
        centered = data - data.mean(axis=1, keepdims=True)
        actual = np.einsum("bni,bnj->bij", centered, centered) / (count - 1)
        np.testing.assert_allclose(actual, np.stack([np.cov(batch, rowvar=False) for batch in data]), atol=1e-12)
        np.testing.assert_allclose(actual, actual.transpose(0, 2, 1))
        assert np.linalg.eigvalsh(actual).min() >= -1e-12
        extra += 1
for _ in range(50):
    S = rng.normal(size=(2, 2)) + 3*np.eye(2)
    u, v = rng.normal(size=(2, 2))
    inverse = np.linalg.inv(S)
    transformed = inverse @ np.outer(u, v) @ inverse.T
    np.testing.assert_allclose(transformed, np.outer(np.linalg.solve(S, u), np.linalg.solve(S, v)), atol=1e-12)
    extra += 1
# Changed practice fixtures, not copies of the lab defaults.
np.testing.assert_array_equal(np.einsum("ij,j->i", [[2,-1],[0,3]], [2,3]), [1,9])
np.testing.assert_array_equal(np.cov([[0,1],[2,3]], rowvar=False), [[2,2],[2,2]])
np.testing.assert_array_equal(np.array([[0,1],[2,3]]).T @ np.array([[0,1],[2,3]]), [[4,6],[6,10]])
np.testing.assert_array_equal(.5*np.array([4,-2])+.5*np.array([0,6]), [2,2])
assert (np.array([1,2]) @ np.array([4,1])) == (np.array([1,3]) @ np.array([3,1])) == 6
# Zero, singleton, empty reduction and API boundaries in the actual runtime.
assert np.einsum("i->", np.array([], dtype=float)) == 0
np.testing.assert_array_equal(np.einsum("ii->i", np.eye(0)), [])
assert np.einsum("i,i->", np.zeros(3), np.ones(3)) == 0
np.testing.assert_array_equal(np.einsum("bi,bi->b", np.ones((1,3)), np.arange(6).reshape(2,3)), [3,12])
for expression, operand in [("ii->i", np.ones((1,2))), ("ij->ii", np.ones((2,2))), ("ij->k", np.ones((2,2)))]:
    try:
        np.einsum(expression, operand)
    except ValueError:
        pass
    else:
        raise AssertionError("NumPy boundary accepted")
print(json.dumps({
    "python": sys.version.split()[0], "numpy": np.__version__,
    "contractionModels": len(cases["contractions"]), "attentionModels": len(cases["attention"]),
    "basisModels": len(cases["bases"]), "costModels": len(cases["orders"]),
    "unseenAttentionPrograms": unseen, "attentionValidationGroups": len(invalid_attention),
    "covarianceOuterProductFixtures": extra, "independentPracticeGroups": 5, "runtimeBoundaryGroups": 7
}))

