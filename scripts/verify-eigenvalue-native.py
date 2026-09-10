"""Check displayed programs and bounded JS states using independent NumPy work."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np

directory = Path(sys.argv[1])
programs = json.loads((directory / 'programs.json').read_text(encoding='utf-8'))
fixtures = json.loads((directory / 'models.json').read_text(encoding='utf-8'))
for name, example in programs.items():
    filename = directory / (name + '.py')
    filename.write_text(example['code'], encoding='utf-8')
    result = subprocess.run([sys.executable, '-I', str(filename)], capture_output=True, text=True, encoding='utf-8', timeout=20)
    assert result.returncode == 0, (name, result.stderr)
    assert result.stdout.strip() == example['expected'].strip(), (name, result.stdout, example['expected'])

def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

for fixture in fixtures['directions']:
    state = fixture['state']
    matrix = np.array(state['matrix'])
    angle = np.deg2rad(fixture['angle'])
    vector = np.array([np.cos(angle), np.sin(angle)])
    output = matrix @ vector
    coefficient = np.linalg.lstsq(vector.reshape(2, 1), output, rcond=None)[0][0]
    perpendicular = output - coefficient * vector
    close(state['input'], vector)
    close(state['output'], output)
    close(state['alongFactor'], coefficient)
    close(state['along'], coefficient * vector)
    close(state['perpendicular'], perpendicular)
    close(state['residualNorm'], np.linalg.norm(perpendicular))
    assert state['isEigenDirection'] == bool(np.linalg.norm(perpendicular) < 1e-10)
    assert state['isZeroOutput'] == bool(np.linalg.norm(output) < 1e-10)
    if state['isEigenDirection']:
        assert min(abs(np.linalg.eigvals(matrix) - coefficient)) < 1e-10
    if fixture['preset'] == 'rotation':
        close(state['residualNorm'], 1)
        assert not state['isEigenDirection']

starts = {'mixed': [1, 1], 'horizontal': [1, 0], 'vertical': [0, 1]}
repeated_states = 0
for fixture in fixtures['repetitions']:
    matrix = np.array(fixture['trace']['matrix'])
    initial = np.array(starts[fixture['start']])
    for state in fixture['trace']['states']:
        k = state['step']
        expected = np.linalg.matrix_power(matrix, k) @ initial
        close(state['vector'], expected)
        close(state['norm'], np.linalg.norm(expected))
        if fixture['preset'] == 'shear':
            close(expected, np.array([[1, k], [0, 1]]) @ initial)
        if fixture['preset'] == 'transient':
            coupling = 0 if k == 0 else 2 * k * 0.6 ** (k - 1)
            close(expected, np.array([[0.6 ** k, coupling], [0, 0.6 ** k]]) @ initial)
        if fixture['preset'] == 'rotation':
            close(np.linalg.norm(expected), np.linalg.norm(initial))
        repeated_states += 1

for fixture in fixtures['pca']:
    state = fixture['state']
    points = np.array(state['points'])
    mean = points.mean(axis=0)
    centered = points - mean
    angle = np.deg2rad(fixture['angle'])
    q = np.array([np.cos(angle), np.sin(angle)])
    scores = centered @ q
    projected = np.outer(scores, q) + mean
    covariance = np.cov(points, rowvar=False, ddof=1)
    values = np.linalg.eigvalsh(covariance)
    variance = np.var(scores, ddof=1)
    error = np.sum((points - projected) ** 2)
    for key, expected in [('mean', mean), ('centered', centered), ('direction', q), ('scores', scores),
                          ('projections', projected), ('covariance', covariance), ('variance', variance),
                          ('totalVariance', np.trace(covariance)), ('squaredReconstructionError', error)]:
        close(state[key], expected)
    close(state['retainedFraction'], variance / values.sum())
    close((points - projected) @ q, 0)
    close(error, (len(points) - 1) * (values.sum() - variance))
    assert values[0] - 1e-10 <= variance <= values[-1] + 1e-10

with contextlib.redirect_stdout(io.StringIO()):
    namespace = runpy.run_path(str(directory / 'powerIteration.py'))
power_iteration = namespace['power_iteration']
rng = np.random.default_rng(73)
power_cases = 0
for dimension in range(2, 7):
    for _ in range(20):
        Q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        values = np.linspace(0.1, 0.9, dimension)
        values[-1] = rng.choice([-3.0, 3.0])
        A = (Q * values) @ Q.T
        start = Q @ np.ones(dimension)
        vector, value, residual, _ = power_iteration(A, start)
        close(abs(value), 3)
        assert residual <= 1e-10
        # The stopping residual is 1e-10, so compare the line using its actual
        # residual/separation bound, not a tighter unrelated entrywise tolerance.
        separation = np.min(np.abs(values[:-1] - value))
        projector_error = np.linalg.norm(np.outer(vector, vector) - np.outer(Q[:, -1], Q[:, -1]))
        assert projector_error <= np.sqrt(2) * residual / separation + 3e-14
        power_cases += 1
for matrix, start in [(np.eye(2), [0, 0]), (np.ones((2, 3)), [1, 2, 3]), (np.eye(2), [1]),
                      (np.array([[np.inf, 0], [0, 1]]), [1, 1]), (np.eye(2, dtype=complex), [1, 1])]:
    try:
        power_iteration(matrix, start)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid power-iteration input accepted')
for tolerance in [0, -1, float('nan'), float('inf')]:
    try:
        power_iteration(np.eye(2), [1, 1], tolerance=tolerance)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid tolerance accepted')
try:
    power_iteration(np.array([[0, -1], [1, 0]]), [1, 0], maximum_steps=5)
except RuntimeError:
    pass
else:
    raise AssertionError('Real quarter-turn falsely converged to a real pair')

# Independent checks of the graph/application and changed-condition hand tasks.
laplacian = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
for vector, value in [([1, 1, 1], 0), ([1, 0, -1], 1), ([1, -2, 1], 3)]:
    vector = np.array(vector)
    close(laplacian @ vector, value * vector)
    close((np.eye(3) - 0.25 * laplacian) @ vector, (1 - 0.25 * value) * vector)
changed_mixing = np.array([[0.9, 0.4], [0.1, 0.6]])
close(changed_mixing @ [400, 100], [400, 100])
close(np.linalg.matrix_power(changed_mixing, 2) @ [500, 0], [425, 75])
close(np.sort(np.linalg.eigvals(changed_mixing)), [0.5, 1])
for k in range(31):
    close(np.linalg.matrix_power(np.array([[1, 2], [0, 1]]), k) @ [0, 1], [2*k, 1])
close(np.sort(np.linalg.eigvals([[1, 1], [1e-8, 1]])), [1-1e-4, 1+1e-4])
close((8+1)/(8+1+1), 0.9)
close((11-1) * 1, 10)

results = {'numpy': np.__version__, 'standalonePrograms': len(programs),
           'directionStates': len(fixtures['directions']), 'repeatedStates': repeated_states,
           'pcaStates': len(fixtures['pca']), 'independentPowerCases': power_cases,
           'additionalHandChecks': 'three graph modes/smoothing; changed mixing; 31 shear powers; defective perturbation; variance budget'}
(directory / 'native-results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
print(json.dumps(results, indent=2))
