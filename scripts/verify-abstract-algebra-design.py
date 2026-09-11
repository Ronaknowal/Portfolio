"""Exact design fixtures, not verification of an implemented lesson or browser."""

from fractions import Fraction
from itertools import product, permutations
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
GROUP = tuple((k, b) for b in range(2) for k in range(4))
IDENTITY = (0, 0)
ROTATE = (1, 0)
REFLECT = (0, 1)


def compose(g, h):
    """g*h means apply h, then g; element (k,b) means r**k s**b."""
    return ((g[0] + (-1) ** g[1] * h[0]) % 4, g[1] ^ h[1])


def inverse(g):
    return ((-(-1) ** g[1] * g[0]) % 4, g[1])


def vertex_permutation(g):
    return tuple((g[0] + (-1) ** g[1] * i) % 4 for i in range(4))


def act(g, values):
    result = [None] * 4
    for i, destination in enumerate(vertex_permutation(g)):
        result[destination] = values[i]
    return tuple(result)


def coset(g, subgroup):
    return frozenset(compose(g, h) for h in subgroup)


def orbit(gset, x):
    return frozenset(act(g, x) for g in gset)


rotation_matrix = sp.Matrix([[0, -1], [1, 0]])
reflection_matrix = sp.diag(1, -1)
matrices = {g: rotation_matrix ** g[0] * reflection_matrix ** g[1] for g in GROUP}
vertices = [sp.Matrix(v) for v in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
assert len({tuple(m) for m in matrices.values()}) == 8
for g, h in product(GROUP, repeat=2):
    assert matrices[compose(g, h)] == matrices[g] * matrices[h]
for g, h, k in product(GROUP, repeat=3):
    assert compose(compose(g, h), k) == compose(g, compose(h, k))
for g in GROUP:
    assert compose(g, inverse(g)) == compose(inverse(g), g) == IDENTITY
    for i, v in enumerate(vertices):
        assert matrices[g] * v == vertices[vertex_permutation(g)[i]]

# An independently generated symmetry set: all vertex permutations preserving distances.
distance_preserving = []
for p in permutations(range(4)):
    if all((vertices[i] - vertices[j]).dot(vertices[i] - vertices[j]) ==
           (vertices[p[i]] - vertices[p[j]]).dot(vertices[p[i]] - vertices[p[j]])
           for i, j in product(range(4), repeat=2)):
        distance_preserving.append(p)
assert set(distance_preserving) == {vertex_permutation(g) for g in GROUP}

rotations = GROUP[:4]
reflection_subgroup = (IDENTITY, REFLECT)
normal_cosets = {coset(g, rotations) for g in GROUP}
assert len(normal_cosets) == 2
assert coset(IDENTITY, reflection_subgroup) == coset(REFLECT, reflection_subgroup)
assert coset(ROTATE, reflection_subgroup) != coset(compose(REFLECT, ROTATE), reflection_subgroup)
for g, h in product(GROUP, repeat=2):
    assert matrices[compose(g, h)].det() == matrices[g].det() * matrices[h].det()
    for a, b in product(rotations, repeat=2):
        assert coset(compose(compose(g, a), compose(h, b)), rotations) == coset(compose(g, h), rotations)

counts = {}
for colors in (2, 3, 4):
    states = tuple(product(range(colors), repeat=4))
    representative_orbits = {min(orbit(GROUP, x)): orbit(GROUP, x) for x in states}
    fixed = [sum(act(g, x) == x for x in states) for g in GROUP]
    closed_form = [colors ** 4, colors, colors ** 2, colors,
                   colors ** 3, colors ** 2, colors ** 3, colors ** 2]
    assert fixed == closed_form
    assert Fraction(sum(fixed), 8) == len(representative_orbits)
    for x in states:
        assert len(orbit(GROUP, x)) * sum(act(g, x) == x for g in GROUP) == 8
    counts[str(colors)] = {
        'states': len(states), 'fixed': fixed, 'orbits': len(representative_orbits),
        'rotationOnlyOrbits': len({min(orbit(rotations, x)) for x in states}),
        'orbitSizes': sorted(len(o) for o in representative_orbits.values()),
    }

# Parameter tying is derived independently by a symbolic commutator nullspace.
permutation_matrices = {}
for g in GROUP:
    p = sp.zeros(4)
    for source, destination in enumerate(vertex_permutation(g)):
        p[destination, source] = 1
    permutation_matrices[g] = p
symbols = sp.symbols('w0:16')
unknown = sp.Matrix(4, 4, symbols)
equations = []
for generator in (ROTATE, REFLECT):
    equations.extend(unknown * permutation_matrices[generator] - permutation_matrices[generator] * unknown)
constraint, _ = sp.linear_eq_to_matrix(equations, symbols)
assert 16 - constraint.rank() == 3
weights = sp.Matrix([[2, 2, 4, 8], [3, 5, 7, 9], [6, 10, 11, 12], [13, 14, 15, 16]])
average = sum((p.T * weights * p for p in permutation_matrices.values()), sp.zeros(4)) / 8
a = sum(weights[i, i] for i in range(4)) / 4
b = sum(weights[i, (i + shift) % 4] for i in range(4) for shift in (1, 3)) / 8
c = sum(weights[i, (i + 2) % 4] for i in range(4)) / 4
distance_tied = sp.Matrix([[a, b, c, b], [b, a, b, c], [c, b, a, b], [b, c, b, a]])
assert average == distance_tied
sensor = sp.Matrix([1, 2, 4, 8])
for p in permutation_matrices.values():
    assert average * p == p * average
    assert average * p * sensor == p * average * sensor
assert weights * permutation_matrices[ROTATE] * sensor != permutation_matrices[ROTATE] * weights * sensor
assert sum((p.T * average * p for p in permutation_matrices.values()), sp.zeros(4)) / 8 == average
# Orthogonal projection certificate for Frobenius distance, exact arithmetic.
changed_tied = sp.Matrix([[2, 1, -1, 1], [1, 2, 1, -1], [-1, 1, 2, 1], [1, -1, 1, 2]])
inner = lambda a, b: sum(x * y for x, y in zip(a, b))
assert inner(weights - average, changed_tied) == 0
assert inner(weights - changed_tied, weights - changed_tied) == (
    inner(weights - average, weights - average) + inner(average - changed_tied, average - changed_tied))

# A rotation-equivariant circulant can fail reflection equivariance.
one_way = permutation_matrices[ROTATE]
assert one_way * permutation_matrices[ROTATE] == permutation_matrices[ROTATE] * one_way
assert one_way * permutation_matrices[REFLECT] != permutation_matrices[REFLECT] * one_way
# Pointwise nonlinearity commutes with coordinate permutations, not all linear representations.
relu = lambda x: x.applyfunc(lambda t: max(t, 0))
assert all(relu(p * sensor) == p * relu(sensor) for p in permutation_matrices.values())
signed_input = sp.Matrix([-1, 2])
assert relu(reflection_matrix * signed_input) != reflection_matrix * relu(signed_input)

modular = {}
for modulus in (5, 6, 7, 8):
    units = [a for a in range(modulus) if any(a * b % modulus == 1 for b in range(modulus))]
    # Independent integer Bezout criterion via exact gcd.
    assert units == [a for a in range(modulus) if sp.gcd(a, modulus) == 1]
    modular[str(modulus)] = {'units': units, 'twoXEqualsFour': [x for x in range(modulus) if 2*x % modulus == 4 % modulus]}
assert modular['6']['twoXEqualsFour'] == [2, 5]
assert modular['5']['twoXEqualsFour'] == [2]

results = {
    'checkedAt': datetime.now(timezone.utc).isoformat(),
    'status': 'design fixtures independently checked; no production lesson implemented',
    'd4': {'matrixCompositionPairs': 64, 'associativeTriples': 512,
           'distancePreservingPermutations': len(distance_preserving),
           'rs': compose(ROTATE, REFLECT), 'sr': compose(REFLECT, ROTATE),
           'rsOnVertex1': vertex_permutation(compose(ROTATE, REFLECT))[1],
           'srOnVertex1': vertex_permutation(compose(REFLECT, ROTATE))[1]},
    'colorings': counts,
    'quotient': {'normalRotationCosets': 2, 'nonnormalRepresentativeWitness': {
        'eH': sorted(coset(IDENTITY, reflection_subgroup)),
        'sH': sorted(coset(REFLECT, reflection_subgroup)),
        'rH': sorted(coset(ROTATE, reflection_subgroup)),
        'srH': sorted(coset(compose(REFLECT, ROTATE), reflection_subgroup))}},
    'equivariance': {'commutantDimension': 3, 'coefficients': list(map(str, [a, b, c])),
        'projectedWeights': [[str(average[i,j]) for j in range(4)] for i in range(4)],
        'input': list(map(str, sensor)), 'output': list(map(str, average * sensor)),
        'rawRotationGap': list(map(str, weights*permutation_matrices[ROTATE]*sensor - permutation_matrices[ROTATE]*weights*sensor)),
        'exactProjectionCertificate': True},
    'modular': modular,
    'originalPlanSha256': hashlib.sha256((ROOT/'docs/teaching/evidence/abstract-algebra-original-plan.json').read_bytes()).hexdigest(),
}
out = ROOT/'scratch/abstract-algebra-design/results.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
print(json.dumps(results, indent=2))
