"""Small F2 boundary reduction versus GUDHI, with matching filtration contracts.
Install: python -m pip install numpy==2.3.5 gudhi==3.13.0
Run: python gudhi_rips.py
The explicit reducer is a <=30-point teaching oracle, not a scalable engine.
"""
from itertools import combinations
import math
import numpy as np
import gudhi


def checked_points(points, cutoff, dimension):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 30:
        raise ValueError("Expected 1 to 30 planar points")
    if not np.isfinite(points).all() or np.abs(points).max() > 100:
        raise ValueError("Coordinates must be finite and within [-100,100]")
    if not math.isfinite(cutoff) or cutoff < 0 or dimension not in (1, 2):
        raise ValueError("Expected nonnegative finite cutoff and dimension 1 or 2")
    return points


def reduce_rips(points, cutoff, dimension):
    points = checked_points(points, cutoff, dimension)
    # Cache distances once; every edge and triangle then reuses the same metric.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    simplices = []
    for size in range(1, dimension + 2):
        for vertices in combinations(range(len(points)), size):
            birth = max((distances[a, b] for a, b in combinations(vertices, 2)), default=0.)
            if birth <= cutoff:
                simplices.append((float(birth), vertices))
    simplices.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    index = {vertices: i for i, (_, vertices) in enumerate(simplices)}
    pivots, creators, killed, bars = {}, set(), set(), []
    for column, (death, vertices) in enumerate(simplices):
        boundary = ({index[face] for face in combinations(vertices, len(vertices) - 1)}
                    if len(vertices) > 1 else set())
        while boundary and max(boundary) in pivots:
            boundary ^= pivots[max(boundary)]
        if not boundary:
            creators.add(column)
        else:
            pivot = max(boundary)
            pivots[pivot] = boundary
            killed.add(pivot)
            birth, face = simplices[pivot]
            if death > birth:
                bars.append((len(face) - 1, birth, death))
    bars.extend((len(simplices[i][1]) - 1, simplices[i][0], math.inf)
                for i in creators - killed)
    return sorted(bar for bar in bars if bar[0] <= 1)


def compare(points, cutoff, dimension=2):
    points = checked_points(points, cutoff, dimension)
    manual = reduce_rips(points, cutoff, dimension)
    tree = gudhi.RipsComplex(points=points, max_edge_length=cutoff).create_simplex_tree(max_dimension=dimension)
    # F2, positive persistence only; include top dimension for graph-only H1.
    result = tree.persistence(homology_coeff_field=2, min_persistence=0,
                              persistence_dim_max=True)
    tool = sorted((d, float(b), float(e)) for d, (b, e) in result if d <= 1)
    assert len(manual) == len(tool)  # preserve multiplicity, not set equality
    assert np.allclose(np.asarray(manual), np.asarray(tool), atol=1e-12, rtol=1e-12)
    return tool


def main():
    square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    full = compare(square, 3.)
    h1 = [bar[1:] for bar in full if bar[0] == 1]
    assert np.allclose(h1, [(2, math.sqrt(8))])
    print(f"Square, triangles included: H1 {[(round(b, 6), round(d, 6)) for b, d in h1]}")
    truncated = compare(square, 2.1)
    assert (1, 2., math.inf) in truncated
    graph = compare(square, 3., dimension=1)
    assert sum(d == 1 and math.isinf(end) for d, _, end in graph) == 3
    print("Cutoff 2.1: one H1 bar still alive; graph-only cutoff 3: three H1 bars still alive")
    rectangle = compare([(0, 0), (3, 0), (3, 1), (0, 1)], 4.)
    assert np.allclose([bar[1:] for bar in rectangle if bar[0] == 1], [(3, math.sqrt(10))])
    assert compare([(0, 0), (0, 0)], 1.) == [(0, 0., math.inf)]
    assert compare([(0, 0)], 0.) == [(0, 0., math.inf)]
    try:
        compare([(math.nan, 0)], 1.)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite point accepted")
    print("Full interval multisets match for square, cutoff, graph, rectangle, duplicates and singleton; invalid data rejected")


if __name__ == "__main__":
    main()
