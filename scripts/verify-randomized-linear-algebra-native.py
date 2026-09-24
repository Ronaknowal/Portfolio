"""Independent NumPy SVD/lstsq and exhaustive sign oracles for browser fixtures."""
import json
from pathlib import Path
import numpy as np

fixtures = json.loads(Path('scratch/randomized-linear-algebra-review/model-fixtures.json').read_text())
check = np.testing.assert_allclose

for case in fixtures['probes']:
    matrix = np.array([[3., 0., 3.], [0., 1., 1.]])
    probe = matrix @ case['weights']
    check(case['output'], probe, atol=1e-12)
    projector = np.outer(probe, probe) / (probe @ probe) if probe @ probe else np.zeros((2, 2))
    check(np.array(case['projected']).T, projector @ matrix, atol=1e-12)
    check(case['residualSquared'], np.linalg.norm(matrix - projector @ matrix)**2, atol=1e-11)

def basis_from_svd(matrix):
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = np.count_nonzero(singular > (1e-11 * singular[0])) if singular.size else 0
    return left[:, :rank]

largest_projector_difference = 0.
for case in fixtures['spectra']:
    matrix = np.array(case['matrix'])
    singular = np.linalg.svd(matrix, compute_uv=False)
    check(singular, case['spectrum'], atol=1e-12)
    basis = np.array(case['basis'])
    observed = basis.shape[1]
    check(basis.T @ basis, np.eye(observed), atol=2e-10)
    expected_basis = basis_from_svd(matrix @ np.array(case['omega']))
    for _ in range(case['iterations']):
        if expected_basis.shape[1]:
            expected_basis = basis_from_svd(matrix @ basis_from_svd(matrix.T @ expected_basis))
    projector_difference = np.linalg.norm(basis @ basis.T - expected_basis @ expected_basis.T)
    largest_projector_difference = max(largest_projector_difference, projector_difference)
    check(basis @ basis.T, expected_basis @ expected_basis.T, atol=2e-8)
    compressed = basis.T @ matrix
    if observed:
        small_left, small_singular, small_right = np.linalg.svd(compressed, full_matrices=False)
        rank = min(case['rank'], observed)
        exact_small = (small_left[:, :rank] * small_singular[:rank]) @ small_right[:rank]
        check(case['compressed'], compressed, atol=1e-10)
        check(case['smallSpectrum'], small_singular, atol=2e-8)
        small_vectors = np.array(case['smallVectors'])
        check(small_vectors.T @ small_vectors, np.eye(observed), atol=1e-10)
        gram = compressed @ compressed.T
        check(gram @ small_vectors, small_vectors * np.array(case['smallSpectrum'])**2, atol=2e-9)
        # Tied singular vectors need not be unique; the optimal residual is.
        check(np.linalg.norm(matrix - np.array(case['approximation'])), np.linalg.norm(matrix - basis @ exact_small), atol=2e-9)
    approximation = np.array(case['approximation'])
    check(case['projected'], basis @ compressed, atol=1e-10)
    check(case['errorSquared'], np.linalg.norm(matrix - approximation)**2, atol=1e-9)
    check(case['rangeErrorSquared'] + case['truncationErrorSquared'], case['errorSquared'], atol=1e-9)
    check(case['floorSquared'], np.sum(singular[case['rank']:]**2), atol=1e-9)
    assert case['errorSquared'] >= case['floorSquared'] - 1e-9
    assert case['passes'] == 2 + 2 * case['iterations']
    assert np.linalg.matrix_rank(approximation, tol=1e-8) <= case['rank']

for case in fixtures['rows']:
    observations = case['observations']
    matrix = np.array([[1., point['x']] for point in observations])
    response = np.array([point['y'] for point in observations])
    exact, *_ = np.linalg.lstsq(matrix, response, rcond=None)
    check([case['fullFit']['intercept'], case['fullFit']['slope']], exact, atol=1e-11)
    check(case['leverage'], np.diag(matrix @ np.linalg.pinv(matrix)), atol=1e-11)
    check(sum(case['leverage']), 2., atol=1e-12)
    selected = case['selectedRows']
    if len(selected) < 2:
        assert case['sketchFit'] is None
        continue
    sketched, *_ = np.linalg.lstsq(matrix[selected], response[selected], rcond=None)
    check([case['sketchFit']['intercept'], case['sketchFit']['slope']], sketched, atol=1e-10)
    check(case['originalResidualSquared'], np.linalg.norm(matrix @ sketched - response)**2, atol=1e-9)
    check(case['sketchResidualSquared'], np.linalg.norm(matrix[selected] @ sketched - response[selected])**2, atol=1e-9)
    assert case['originalResidualSquared'] >= case['fullResidualSquared'] - 1e-9

for case in fixtures['traces']:
    matrix = np.array(case['matrix'])
    all_signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    values = np.einsum('bi,ij,bj->b', all_signs, matrix, all_signs)
    check(values.mean(), np.trace(matrix))
    check(values.var(), case['variancePerProbe'])
    total = 0.
    for sample in case['samples']:
        vector = np.array(sample['input'])
        check(sample['output'], matrix @ vector)
        check(sample['value'], vector @ matrix @ vector)
        total += sample['value']
        check(sample['mean'], total / sample['count'])
    if not case['samples']:
        assert case['mean'] is None and case['absoluteError'] is None

print(json.dumps({'numpy': np.__version__, 'probes': len(fixtures['probes']), 'spectral_cases': len(fixtures['spectra']), 'row_subsets': len(fixtures['rows']), 'trace_states': len(fixtures['traces']), 'maximum_projector_difference': largest_projector_difference}))
