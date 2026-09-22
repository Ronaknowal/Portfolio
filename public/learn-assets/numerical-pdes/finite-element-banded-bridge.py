"""Nonuniform P1 assembly, reusable tridiagonal LDL, and SciPy banded Cholesky.

Run: python finite-element-banded-bridge.py
Dependencies: numpy==2.3.5, scipy==1.18.1.
Equation -(a*u')'=f with Dirichlet endpoints; a and f constant per element.
Assembly, factorization and each solve are O(N) time/storage in one dimension.
This specializes the lesson's existing LDL mechanism, retaining the bandwidth.
"""
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded


def assemble(nodes, conductivity, forcing, boundary=(0., 0.), point_load=None):
    nodes, conductivity, forcing = map(lambda a: np.asarray(a, float), (nodes, conductivity, forcing))
    if (nodes.ndim != 1 or len(nodes) < 3 or conductivity.shape != (len(nodes)-1,)
            or forcing.shape != conductivity.shape or not np.isfinite(nodes).all()
            or not np.isfinite(conductivity).all() or not np.isfinite(forcing).all()
            or np.any(np.diff(nodes) <= 0) or np.any(conductivity <= 0)
            or np.asarray(boundary).shape != (2,) or not np.isfinite(boundary).all()):
        raise ValueError("Increasing nodes, positive element coefficients and finite data required")
    widths = np.diff(nodes)
    stiffness = conductivity / widths
    diagonal = stiffness[:-1] + stiffness[1:]
    off_diagonal = -stiffness[1:-1]
    rhs = (forcing[:-1]*widths[:-1] + forcing[1:]*widths[1:]) / 2
    rhs[0] += stiffness[0]*boundary[0]
    rhs[-1] += stiffness[-1]*boundary[1]
    if point_load is not None:
        location, strength = point_load
        if not nodes[0] < location < nodes[-1] or not np.isfinite(strength):
            raise ValueError("Point load must be finite and strictly inside the domain")
        element = np.searchsorted(nodes, location, side="right") - 1
        right_weight = (location - nodes[element]) / widths[element]
        for index, weight in ((element, 1-right_weight), (element+1, right_weight)):
            if 0 < index < len(nodes)-1:
                rhs[index-1] += strength*weight
    return diagonal, off_diagonal, rhs


def factor_ldl(diagonal, off_diagonal):
    pivots = np.array(diagonal, float, copy=True)
    off_diagonal = np.asarray(off_diagonal, float)
    if pivots.ndim != 1 or not pivots.size or off_diagonal.shape != (len(pivots)-1,):
        raise ValueError("Tridiagonal shapes must agree")
    multipliers = np.empty_like(off_diagonal)
    for i in range(len(off_diagonal)):
        if pivots[i] <= 0 or not np.isfinite(pivots[i]):
            raise ValueError("Positive finite pivots required; no pivoting is implemented")
        multipliers[i] = off_diagonal[i]/pivots[i]
        pivots[i+1] -= multipliers[i]*off_diagonal[i]
    if pivots[-1] <= 0 or not np.isfinite(pivots[-1]):
        raise ValueError("Positive finite pivots required")
    return pivots, multipliers


def solve_ldl(factor, rhs):
    pivots, multipliers = factor
    solution = np.array(rhs, float, copy=True)
    if solution.shape != pivots.shape:
        raise ValueError("This solver accepts one RHS vector per call")
    for i in range(1, len(solution)):
        solution[i] -= multipliers[i-1]*solution[i-1]
    solution /= pivots
    for i in range(len(solution)-2, -1, -1):
        solution[i] -= multipliers[i]*solution[i+1]
    return solution


def main():
    nodes = np.array([0., .1, .35, .7, 1.])
    diagonal, off, rhs = assemble(nodes, np.ones(4), np.full(4, 2.), boundary=(1., 2.))
    factor = factor_ldl(diagonal, off)
    solution = solve_ldl(factor, rhs)
    band = np.zeros((2, len(diagonal))); band[0] = diagonal; band[1, :-1] = off
    library_factor = cholesky_banded(band, lower=True)
    reference = cho_solve_banded((library_factor, True), rhs)
    exact_nodes = 1 + nodes + nodes*(1-nodes)
    np.testing.assert_allclose(solution, exact_nodes[1:-1], atol=1e-13)
    np.testing.assert_allclose(solution, reference, atol=1e-13)
    # Same stiffness, second load: reuse both factors, not a new matrix inverse.
    _, _, point_rhs = assemble(nodes, np.ones(4), np.zeros(4), point_load=(.4, 1.))
    point_solution = solve_ldl(factor, point_rhs)
    point_reference = cho_solve_banded((library_factor, True), point_rhs)
    np.testing.assert_allclose(point_solution, point_reference, atol=1e-13)
    exact_green_nodes = np.minimum(nodes, .4)*(1-np.maximum(nodes, .4))
    np.testing.assert_allclose(point_solution, exact_green_nodes[1:-1], atol=1e-13)
    print("quadratic nodal values", np.round(solution, 9).tolist())
    print("off-node point-load vector", np.round(point_rhs, 9).tolist())
    print("point-source nodal values", np.round(point_solution, 9).tolist())
    print("banded library max_error", f"{max(np.max(abs(solution-reference)), np.max(abs(point_solution-point_reference))):.2e}")
    print("quadratic interpolation is exact between nodes", False)


if __name__ == "__main__":
    main()
