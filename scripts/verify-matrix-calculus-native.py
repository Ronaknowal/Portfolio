import contextlib
import io
import json
from pathlib import Path
import platform
import runpy
import sys

import numpy as np

directory = Path(sys.argv[1])
cases = json.loads((directory / 'model-cases.json').read_text())
counts = {}

def close(actual, expected, **options):
    np.testing.assert_allclose(actual, expected, rtol=options.get('rtol', 1e-10), atol=options.get('atol', 1e-10))

def complex_jacobian(function, point):
    point = np.asarray(point, dtype=float)
    columns = []
    for index in range(point.size):
        perturbed = point.astype(complex)
        perturbed[index] += 1e-25j
        columns.append(np.imag(function(perturbed)) / 1e-25)
    return np.stack(columns, axis=-1)

def polynomial(x):
    return np.array([x[0] * x[0] + x[1], x[0] * x[1]])

for case in cases['local']:
    x, v, step = np.array(case['input']), np.array(case['direction']), case['step']
    jacobian = complex_jacobian(polynomial, x)
    close(case['jacobian'], jacobian)
    close(case['base'], polynomial(x))
    close(case['shifted'], polynomial(x + step * v))
    close(case['rate'], jacobian @ v)
    close(case['predicted'], step * (jacobian @ v))
    close(case['actual'], polynomial(x + step * v) - polynomial(x))
    close(case['error'], [step ** 2 * v[0] ** 2, step ** 2 * v[0] * v[1]])
counts['local_cases_against_complex_step'] = len(cases['local'])

A = np.array([[1., 2.], [-1., 1.]])
for case in cases['chain']:
    x, v, q = (np.array(case[key]) for key in ['input', 'direction', 'outputWeights'])
    vector_function = lambda value: (A @ value) ** 2
    scalar_function = lambda value: q @ vector_function(value)
    jacobian = complex_jacobian(vector_function, x)
    gradient = complex_jacobian(scalar_function, x)
    close(case['value'], scalar_function(x))
    close(case['squaredJacobian'], jacobian)
    close(case['inputGradient'], gradient)
    close(case['intermediateTangent'], A @ v)
    close(case['squaredTangent'], jacobian @ v)
    close(case['lossTangent'], gradient @ v)
    close(np.sum(case['contributions'], axis=1), gradient)
    # Independent transpose pairing with two further test vectors, not only the UI seed.
    for test in [np.array([2., -3.]), np.array([-1., 4.])]:
        close(q @ jacobian @ test, gradient @ test)
counts['chain_cases_against_complex_step'] = len(cases['chain'])

W = np.array([[1., -1.], [2., 1.]])
b = np.array([0., 1.])
affine_coordinates = 0
for case in cases['affine']:
    X, T = np.array(case['inputs']), np.array(case['targets'])
    divisor = len(X) if case['reduction'] == 'mean' else 1
    def objective(inputs, weights, bias):
        residual = inputs @ weights + bias - T
        return np.sum(residual * residual) / (2 * divisor)
    close(case['loss'], objective(X, W, b))
    for variable, original, field in [('X', X, 'inputGradient'), ('W', W, 'weightGradient'), ('b', b, 'biasGradient')]:
        numeric = np.zeros_like(original, dtype=float)
        for index in np.ndindex(original.shape):
            perturbation = original.astype(complex)
            perturbation[index] += 1e-25j
            arguments = {'inputs': X, 'weights': W, 'bias': b}
            arguments[{'X': 'inputs', 'W': 'weights', 'b': 'bias'}[variable]] = perturbation
            numeric[index] = objective(**arguments).imag / 1e-25
            affine_coordinates += 1
        close(case[field], numeric)
    for feature in range(2):
        for output in range(2):
            for term in case['weightContributions'][feature][output]:
                observation = term['observation']
                # Isolate that observation's loss; perturb the parameter independently.
                perturbed = W.astype(complex)
                perturbed[feature, output] += 1e-25j
                residual = X[observation] @ perturbed + b - T[observation]
                contribution = (np.sum(residual * residual) / (2 * divisor)).imag / 1e-25
                close(term['product'], contribution)
counts['affine_coordinate_gradients'] = affine_coordinates

for case in cases['differences']:
    x, h = case['point'], case['step']
    if case['kind'] == 'cubic':
        close(case['exact'], 3 * x * x)
        # Exact-arithmetic expansions plus a scale-aware floating subtraction bound.
        rounding_bound = 32 * np.finfo(float).eps * max(abs(x) + h, 1e-20) ** 3 / h
        close(case['forward'], 3 * x * x + 3 * x * h + h * h, atol=rounding_bound + 1e-25)
        close(case['backward'], 3 * x * x - 3 * x * h + h * h, atol=rounding_bound + 1e-25)
        close(case['central'], 3 * x * x + h * h, atol=rounding_bound + 1e-25)
    elif x == 0:
        assert case['exact'] is None and case['centralError'] is None
        close([case['backward'], case['forward'], case['central']], [-1, 1, 0])
    elif h < abs(x) and h > 1e-10:
        close([case['backward'], case['forward'], case['central']], [np.sign(x)] * 3, atol=1e-7)
    assert case['roundedInput'] == (x + h == x or x - h == x)
counts['difference_contract_cases'] = len(cases['differences'])

# Execute learner functions, then challenge them on unseen fixtures.
namespaces = {}
for name in ['dual', 'softmax', 'hessian']:
    with contextlib.redirect_stdout(io.StringIO()):
        namespaces[name] = runpy.run_path(str(directory / (name + '.py')))
Dual, program = namespaces['dual']['Dual'], namespaces['dual']['program']
dual_cases = 0
for x in [(-2., 3.), (0., 0.), (1.5, -2.), (4., 1.)]:
    for v in [(0., 0.), (1., 0.), (0., 1.), (-1., 2.)]:
        result = program(Dual(x[0], v[0]), Dual(x[1], v[1]))
        close([item.value for item in result], polynomial(np.array(x)))
        close([item.tangent for item in result], complex_jacobian(polynomial, x) @ v)
        dual_cases += 1
counts['dual_unseen_cases'] = dual_cases

rng = np.random.default_rng(902613)
matrix_cases = 0
softmax_cases = 0
for _ in range(100):
    matrix = rng.normal(size=(3, 3)) + 5 * np.eye(3)
    direction = rng.normal(size=(3, 3))
    rhs, rhs_direction = rng.normal(size=(2, 3))
    square_derivative = matrix @ direction + direction @ matrix
    square_numeric = np.imag((matrix + 1e-25j * direction) @ (matrix + 1e-25j * direction)) / 1e-25
    close(square_derivative, square_numeric)
    inverse = np.linalg.inv(matrix)
    inverse_numeric = np.imag(np.linalg.inv(matrix + 1e-25j * direction)) / 1e-25
    close(-inverse @ direction @ inverse, inverse_numeric)
    solution = np.linalg.solve(matrix, rhs)
    solution_rate = np.linalg.solve(matrix, rhs_direction - direction @ solution)
    solution_numeric = np.linalg.solve(matrix + 1e-25j * direction, rhs + 1e-25j * rhs_direction).imag / 1e-25
    close(solution_rate, solution_numeric)
    matrix_cases += 3
    scores = rng.normal(size=4)
    probabilities = namespaces['softmax']['softmax'](scores)
    reference = lambda value: np.exp(value - scores.max()) / np.exp(value - scores.max()).sum()
    jacobian = complex_jacobian(reference, scores)
    close(np.diag(probabilities) - np.outer(probabilities, probabilities), jacobian)
    close(jacobian @ np.ones(4), np.zeros(4))
    close(probabilities, namespaces['softmax']['softmax'](scores + 300))
    softmax_cases += 1
counts['matrix_operator_cases_against_complex_step'] = matrix_cases
counts['softmax_unseen_cases'] = softmax_cases

# Independent hand-transfer answers and the partial-derivative counterexample.
close(complex_jacobian(lambda x: np.array([x[0] + 3*x[1], x[0]*x[1]**2]), [-1, 2]), [[1, 3], [4, -4]])
close(complex_jacobian(lambda x: (x[0]+x[1])**2 + 3*x[0] - 6*x[1], [2, 1]), [9, 0])
for t in [1., 0.1, -0.1, 1e-6]:
    value = t*t / np.sqrt(2*t*t)
    close(abs(value) / np.linalg.norm([t, t]), 0.5)
for point in [[1, -1], [0, 0], [-2, 3]]:
    close(complex_jacobian(namespaces['hessian']['gradient'], point), [[2, 1], [1, 4]])
counts['independent_transfer_and_counterexample_groups'] = 4
print(json.dumps({'python': platform.python_version(), 'numpy': np.__version__, 'independentChecks': counts}))
