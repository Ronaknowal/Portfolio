"""NumPy/least-squares references for exact lesson models and derived claims."""
import json
import sys
from pathlib import Path
import numpy as np

cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = {}
for case in cases["projections"]:
    vector = np.array(case["vector"], dtype=float)
    direction = np.array(case["direction"], dtype=float)
    result = case["result"]
    assert result["product"] == np.dot(vector, direction)
    assert np.isclose(result["length"], np.linalg.norm(vector))
    if not np.any(direction):
        assert result["projection"] is None and result["coefficient"] is None
        assert result["residual"] is None and result["cosine"] is None
        continue
    # A library least-squares solve independently finds the closest coefficient.
    coefficient = np.linalg.lstsq(direction.reshape(2, 1), vector, rcond=None)[0][0]
    projection = coefficient * direction
    np.testing.assert_allclose(result["projection"], projection, atol=1e-12)
    np.testing.assert_allclose(result["residual"], vector - projection, atol=1e-12)
    assert np.isclose(np.dot(result["residual"], direction), 0, atol=1e-12)
    assert np.isclose(np.linalg.norm(vector) ** 2, np.linalg.norm(result["projection"]) ** 2 + np.linalg.norm(result["residual"]) ** 2)
    if np.any(vector):
        assert np.isclose(result["cosine"], np.dot(vector / np.linalg.norm(vector), direction / np.linalg.norm(direction)))
    else:
        assert result["cosine"] is None
counts["projection_cases_against_lstsq"] = len(cases["projections"])

for case in cases["maps"]:
    matrix, vector = np.array(case["matrix"]), np.array(case["vector"])
    np.testing.assert_array_equal(case["output"], np.matmul(matrix, vector))
    np.testing.assert_array_equal(matrix @ vector, matrix[:, 0] * vector[0] + matrix[:, 1] * vector[1])
counts["map_cases"] = len(cases["maps"])

partial_states = 0
for case in cases["products"]:
    left, right = np.array(case["left"]), np.array(case["right"])
    expected = left @ right
    np.testing.assert_array_equal(case["output"], expected)
    for cell in case["cells"]:
        row, column, terms = cell["row"], cell["column"], cell["terms"]
        result = cell["result"]
        partial = np.dot(left[row, :terms], right[:terms, column])
        assert result["partial"] == partial
        np.testing.assert_array_equal(result["result"], expected)
        np.testing.assert_array_equal([term["product"] for term in result["contributions"]], left[row] * right[:, column])
        partial_states += 1
counts["matrix_products_against_numpy"] = len(cases["products"])
counts["product_partial_states"] = partial_states

source_cells = 0
for case in cases["reductions"]:
    tensor, axis, result = np.array(case["tensor"]), case["axis"], case["result"]
    expected = tensor.mean(axis=axis)
    assert result["shape"] == list(expected.shape)
    np.testing.assert_allclose([[cell["mean"] for cell in row] for row in result["cells"]], expected)
    remaining = [index for index in range(3) if index != axis]
    for row, cells in enumerate(result["cells"]):
        for column, cell in enumerate(cells):
            # Independent coordinate enumeration finds the exact source set.
            coordinates = [coord for coord in np.ndindex(tensor.shape)
                           if coord[remaining[0]] == row and coord[remaining[1]] == column]
            assert [source["coordinate"] for source in cell["sources"]] == [list(coord) for coord in coordinates]
            assert [source["value"] for source in cell["sources"]] == [tensor[coord] for coord in coordinates]
            assert cell["sum"] == sum(tensor[coord] for coord in coordinates)
            source_cells += len(coordinates)
counts["tensor_reductions_against_numpy"] = len(cases["reductions"])
counts["source_value_correspondences"] = source_cells

# Independently check the worked and transfer values, structure and shape claims.
assert np.linalg.norm(np.array([1, 2]) * 2 - [3, -1]) == np.sqrt(26)
matrix = np.array([[1, 2], [2, 4]])
assert np.linalg.matrix_rank(matrix) == 1
for parameter in [-10, -.5, 0, 1, 10]:
    np.testing.assert_allclose(matrix @ [3 - 2 * parameter, parameter], [3, 6])
assert np.linalg.matrix_rank(np.column_stack([matrix, [3, 7]])) == 2
np.testing.assert_array_equal(matrix @ [-2, 1], [0, 0])
X = np.array([[1, 0, 1], [0, 2, 0]])
R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
np.testing.assert_array_equal(X @ R.T, [[1, 0, 1, 2], [0, 2, 0, 2]])
assert (np.zeros((5, 1, 2, 3)) @ np.zeros((7, 3, 4))).shape == (5, 7, 2, 4)
tensor = np.arange(12, dtype=float).reshape(2, 2, 3)
correct = tensor - tensor.mean(axis=1, keepdims=True)
wrong = tensor - tensor.mean(axis=1)
np.testing.assert_allclose(correct.mean(axis=1), 0)
assert not np.allclose(wrong.mean(axis=1), 0)
np.testing.assert_array_equal(np.arange(6).reshape(2, 3).T, [[0, 3], [1, 4], [2, 5]])
np.testing.assert_array_equal(np.arange(6).reshape(3, 2), [[0, 1], [2, 3], [4, 5]])
assert np.arange(3).T.shape == (3,)
counts["worked_transfer_and_shape_contract_groups"] = 11
print(json.dumps({"python": sys.version.split()[0], "numpy": np.__version__, "independentChecks": counts}))
