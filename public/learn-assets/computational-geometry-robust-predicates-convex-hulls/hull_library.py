"""Run beside computational_geometry_mechanisms.py; NumPy 2.3.5, SciPy 1.18.1."""
from math import isclose
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from computational_geometry_mechanisms import convex_hull, signed_area_twice


def main():
    points = [(1, 1), (3, 1), (7, 1), (7, 6), (4, 6),
              (1, 6), (3, 3), (5, 4), (1, 1)]
    # Compare corner-only contracts on small exactly representable integers.
    exact = convex_hull(points)
    coordinates = np.array(sorted(set(points)), dtype=np.float64)
    hull = ConvexHull(coordinates)
    corners = {tuple(coordinates[i]) for i in hull.vertices}
    assert corners == set(exact)
    exact_area = abs(signed_area_twice(exact)) / 2
    assert isclose(hull.volume, exact_area, rel_tol=1e-12, abs_tol=1e-12)
    print('corners agree:', True)
    print('area / perimeter:', round(hull.volume, 6), round(hull.area, 6))
    print('keep boundary records:', convex_hull(points, include_boundary=True))
    line = [(1, 2), (3, 3), (5, 4), (7, 5)]
    print('exact collinear hull:', convex_hull(line))
    try:
        ConvexHull(np.array(line, dtype=np.float64))
    except QhullError:
        print('2D library on collinear input:', 'QhullError')
    large = 2**53
    print('distinct integer inputs collapse:', float(large) == float(large + 1))


if __name__ == '__main__':
    main()
