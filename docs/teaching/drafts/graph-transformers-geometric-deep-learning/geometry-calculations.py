"""Constructed mechanisms with analytical or exhaustive numerical cross-checks."""
import json
from pathlib import Path
import numpy as np


def cycle(n):
    adjacency = np.zeros((n, n))
    for i in range(n):
        adjacency[i, (i + 1) % n] = adjacency[(i + 1) % n, i] = 1
    return adjacency


def geometric_update(points, step=.1):
    difference = points[:, None, :] - points[None, :, :]
    square = np.sum(difference ** 2, axis=-1)
    weights = 1 / (1 + square)
    np.fill_diagonal(weights, 0)
    return points + step * np.sum(difference * weights[..., None], axis=1)


def run():
    path_values = np.array([1., 2., 4.])
    probabilities = np.array([1., .5, .25]) / 1.75
    original = float(probabilities @ path_values)
    assert np.allclose(probabilities, [4/7, 2/7, 1/7])
    assert np.isclose(original, 12/7)
    graphs = {'six_cycle': cycle(6), 'two_triangles': np.kron(np.eye(2), cycle(3))}
    spectra = {}
    for name, adjacency in graphs.items():
        values, vectors = np.linalg.eigh(np.diag(adjacency.sum(-1)) - adjacency)
        transition = adjacency / adjacency.sum(-1, keepdims=True)
        spectra[name] = {'eigenvalues': values.tolist(), 'rw_three_returns': np.diag(np.linalg.matrix_power(transition, 3)).tolist()}
    assert np.allclose(spectra['six_cycle']['rw_three_returns'], 0)
    assert np.allclose(spectra['two_triangles']['rw_three_returns'], .25)
    values, vectors = np.linalg.eigh(2 * np.eye(4) - cycle(4))
    basis = vectors[:, np.isclose(values, 2)]
    angle = .37
    basis_rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_basis = basis @ basis_rotation
    projector_error = float(np.max(np.abs(basis @ basis.T - rotated_basis @ rotated_basis.T)))
    assert projector_error < 1e-12
    rotation = np.roll(np.eye(4), 1, axis=0)
    reflection = np.eye(4)[[0, 3, 2, 1]]
    group = [np.linalg.matrix_power(rotation, k) @ s for s in [np.eye(4), reflection] for k in range(4)]
    tied = sum(g.T @ rotation @ g for g in group) / 8
    assert all(np.allclose(tied @ g, g @ tied) for g in group)
    signal = np.array([1., 2., 4., 8.])
    square = {'input': signal.tolist(), 'rotation': rotation.tolist(), 'reflection': reflection.tolist(),
              'shift_then_reflect': (reflection @ rotation @ signal).tolist(),
              'reflect_then_shift': (rotation @ reflection @ signal).tolist(), 'tied_matrix': tied.tolist(),
              'tied_output': (tied @ signal).tolist(),
              'constant_input_false_null': float(np.max(np.abs((reflection @ rotation - rotation @ reflection) @ np.ones(4))))}
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
    updated = geometric_update(points)
    transform = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([3., -2., 1.])
    rotated = geometric_update(points @ transform.T + translation)
    discrepancy = float(np.max(np.abs(rotated - (updated @ transform.T + translation))))
    assert discrepancy < 1e-12
    errors = []
    rng = np.random.default_rng(25)
    for _ in range(12):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        t = rng.normal(size=3)
        errors.append(float(np.max(np.abs(geometric_update(points @ q.T + t) - (updated @ q.T + t)))))
    assert max(errors) < 1e-12
    vector = np.array([1., -2., 0.])
    relu_defect = float(np.max(np.abs(np.maximum(transform @ vector, 0) - transform @ np.maximum(vector, 0))))
    difference = points[:, None, :] - points[None, :, :]
    force = -difference.sum(1)
    def energy(x):
        return float(sum(.5 * np.sum((x[i] - x[j]) ** 2) for i in range(3) for j in range(i)))
    numerical_force = np.zeros_like(points)
    for i in range(3):
        for j in range(3):
            plus, minus = points.copy(), points.copy()
            plus[i, j] += 1e-5; minus[i, j] -= 1e-5
            numerical_force[i, j] = -(energy(plus) - energy(minus)) / 2e-5
    assert np.allclose(force, numerical_force, atol=1e-9)
    tetrahedron = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    mirror = np.diag([-1., 1., 1.])
    def volume(x):
        return float(np.linalg.det(x[1:] - x[0]))
    record = {'attention': {'weights': probabilities.tolist(), 'output': original,
                            'changed_remote_value_output': float(probabilities @ [1., 2., 11.]),
                            'masked_output': float(np.array([2/3, 1/3]) @ path_values[:2])},
              'graphs': spectra, 'basis': {'vectors': basis.tolist(), 'rotated_vectors': rotated_basis.tolist(),
                                          'projector': (basis @ basis.T).tolist(), 'projector_error': projector_error},
              'square': square, 'geometry': {'points': points.tolist(), 'updated': updated.tolist(),
                                            'rotation': transform.tolist(), 'translation': translation.tolist(),
                                            'transformed_updated': rotated.tolist(), 'equivariance_error': discrepancy,
                                            'orthogonal_probe_errors': errors, 'relu_defect': relu_defect,
                                            'energy': energy(points), 'forces': force.tolist(),
                                            'finite_difference_force_error': float(np.max(np.abs(force - numerical_force))),
                                            'oriented_volume': volume(tetrahedron), 'mirrored_volume': volume(tetrahedron @ mirror.T)} }
    (Path(__file__).resolve().parent / 'geometry-results.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    run()
