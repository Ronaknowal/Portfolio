"""Independent NumPy/LAPACK and algebraic checks for the bounded teaching models."""
import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np

directory = Path(sys.argv[1])
programs = json.loads((directory / "programs.json").read_text(encoding="utf-8"))
models = json.loads((directory / "models.json").read_text(encoding="utf-8"))
namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(programs["pivotedLu"]["code"], namespace)
factor_lu = namespace["factor_lu"]
solve_factored = namespace["solve_factored"]
rng = np.random.default_rng(8042)
lu_cases = 0
for size in range(1, 9):
    for _ in range(50):
        matrix = rng.normal(size=(size, size))
        target = rng.normal(size=size)
        ordering, lower, upper = factor_lu(matrix)
        assert np.allclose(matrix[ordering], lower @ upper, atol=1e-11, rtol=1e-11)
        assert np.allclose(np.triu(lower, 1), 0)
        assert np.allclose(np.diag(lower), 1)
        assert np.allclose(np.tril(upper, -1), 0)
        answer = solve_factored(ordering, lower, upper, target)
        assert np.allclose(answer, np.linalg.solve(matrix, target), atol=1e-9, rtol=1e-9)
        assert np.allclose(matrix @ answer, target, atol=1e-10, rtol=1e-10)
        lu_cases += 1
for invalid in [[], [[1, 2]], [[np.inf]], [[np.nan]], np.array([[1+2j]])]:
    try:
        factor_lu(invalid)
        raise AssertionError("Invalid LU input accepted")
    except ValueError:
        pass
for singular in [[[0.]], [[1., 2.], [2., 4.]]]:
    try:
        factor_lu(singular)
        raise AssertionError("Zero-pivot input accepted")
    except np.linalg.LinAlgError:
        pass
for trace in models["elimination"]:
    matrix = np.array(trace["matrix"])
    target = np.array(trace["target"])
    for step in trace["steps"]:
        augmented = np.array(step["augmented"])
        permutation = np.array(step["permutation"])
        lower = np.array(step["lower"])
        assert np.allclose(permutation @ matrix, lower @ augmented[:, :2])
        assert np.allclose(permutation @ target, lower @ augmented[:, 2])
    if trace["solution"] is not None:
        assert np.allclose(trace["solution"], np.linalg.solve(matrix, target))
    else:
        assert np.linalg.matrix_rank(matrix) < 2
for model in models["qr"]:
    first, second = np.array(model["firstColumn"]), np.array(model["secondColumn"])
    projection = first * ((first @ second) / (first @ first))
    remainder = second - projection
    assert np.allclose(projection, model["projection"])
    assert np.allclose(remainder, model["perpendicular"])
    assert np.allclose(first @ remainder, 0)
    if not model["dependent"]:
        orthogonal, triangular = np.array(model["orthogonal"]), np.array(model["triangular"])
        assert np.allclose(orthogonal.T @ orthogonal, np.eye(2))
        assert np.allclose(orthogonal @ triangular, np.column_stack([first, second]))
    else:
        assert np.linalg.matrix_rank(np.column_stack([first, second])) == 1
for model in models["covariance"]:
    covariance, lower = np.array(model["covariance"]), np.array(model["lower"])
    assert np.allclose(lower @ lower.T, covariance)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12
    assert np.isclose(np.linalg.det(covariance), model["determinant"])
    if model["positiveDefinite"]:
        assert np.allclose(lower, np.linalg.cholesky(covariance))
    else:
        assert np.linalg.matrix_rank(covariance) == 1
    assert np.allclose(np.array(model["circle"]) @ lower.T, model["transformed"])
for model in models["svd"]:
    matrix, approximation = np.array(model["matrix"]), np.array(model["approximation"])
    left, singular_values, right_transpose = np.linalg.svd(matrix)
    assert np.allclose(singular_values, model["singularValues"])
    assert np.allclose(matrix @ np.array(model["input"]), model["output"])
    assert np.allclose(approximation @ np.array(model["input"]), model["approximationOutput"])
    error = matrix - approximation
    assert np.isclose(np.linalg.norm(error, "fro"), model["frobeniusError"], atol=1e-12)
    assert np.isclose(np.linalg.norm(error, 2), model["spectralError"], atol=1e-12)
    assert np.linalg.matrix_rank(matrix) == model["rank"]
    assert np.allclose(np.array(model["circle"]) @ matrix.T, model["transformed"])
    # Repeated singular values permit different optimal factors. Compare the
    # objective with LAPACK's best approximation, not one arbitrary basis.
    retained = model["retained"]
    reference = (left[:, :retained] * singular_values[:retained]) @ right_transpose[:retained]
    assert np.isclose(np.linalg.norm(matrix - reference, "fro"), np.linalg.norm(error, "fro"), atol=1e-11)
result = {"numpy": np.__version__, "randomLuSystems": lu_cases,
          "eliminationTraces": len(models["elimination"]), "qrStates": len(models["qr"]),
          "covarianceStates": len(models["covariance"]), "svdStates": len(models["svd"])}
(directory / "native-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result))
