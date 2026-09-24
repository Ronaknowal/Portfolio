"""Independent finite witnesses, nonrepresentable Yoneda families and changed helpers."""
import ast
import contextlib
from datetime import datetime, timezone
from fractions import Fraction as Q
from itertools import combinations, product
import io
import json
import math
from pathlib import Path

ROOT = Path('scratch/category-theory-independent-review')
examples = json.loads((ROOT / 'actual-examples.json').read_text(encoding='utf-8'))
fixtures = json.loads((ROOT / 'fixtures.json').read_text(encoding='utf-8'))
counts = {}
maximum_error = 0.0


def close(actual, expected):
    global maximum_error
    error = abs(actual - float(expected))
    maximum_error = max(maximum_error, error)
    assert math.isclose(actual, float(expected), rel_tol=5e-13, abs_tol=5e-13), (actual, expected)


namespaces = {}
for name, example in examples.items():
    namespace = {'__name__': '__review__'}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], '<actual-' + name + '>', 'exec'), namespace)
    assert output.getvalue().rstrip() == example['expected']
    namespaces[name] = namespace
counts['actual_programs'] = len(examples)

# Exact rectangular channels: sum disjoint path-event masses, not copied UI code.
for case in fixtures['channels']:
    a, b = case['a'], case['b']
    for x in range(3):
        for z in range(3):
            expected = sum(Q(a[x][y] * b[y][z], 56) for y in range(2))
            close(case['result'][x][z], expected)
    assert all(abs(sum(row) - 1) < 1e-14 for row in case['result'])
counts['rectangular_channel_states'] = len(fixtures['channels'])

for case in fixtures['pullbacks']:
    expected = {(x, y) for x in range(3) for y in range(5) if case['a'][x] == case['b'][y]}
    assert {tuple(pair) for pair in case['result']} == expected
counts['unequal_size_pullbacks'] = len(fixtures['pullbacks'])

for case in fixtures['tangents']:
    x, v, w = Q(str(case['x'])), Q('-1.37'), Q('.83')
    coefficients = {'shiftedSquare': [1, 2, 1], 'cubicAffine': [-1, 0, 0, 2], 'squareSquare': [0, 0, 0, 0, 1]}[case['choice']]
    value = sum(c * x ** i for i, c in enumerate(coefficients))
    derivative = sum(i * c * x ** (i - 1) for i, c in enumerate(coefficients) if i)
    for field, expected in [('secondValue', value), ('composedDerivative', derivative), ('outputTangent', derivative * v), ('inputCotangent', derivative * w), ('inputPairing', derivative * v * w)]:
        close(case['result'][field], expected)
counts['off_grid_tangent_pairings'] = len(fixtures['tangents'])

# Enumerate ALL mediating functions from a two-point Z. Products of witness
# counts predict the result independently; checks existence and uniqueness.
universal = namespaces['universal']
mediator_cases = 0
for mode, points in universal['candidates'].items():
    for rows in product(range(2), repeat=2):
        for columns in product(range(2), repeat=2):
            maps = [f for f in product(points, repeat=2)
                    if all(f[z][1:] == (rows[z], columns[z]) for z in range(2))]
            expected = math.prod(len(universal['witnesses'](points, rows[z], columns[z])) for z in range(2))
            assert len(maps) == expected
            assert (len(maps) == 1) == all(expected_count == 1 for expected_count in [len(universal['witnesses'](points, rows[z], columns[z])) for z in range(2)])
            mediator_cases += 1
counts['two_point_mediator_problems'] = mediator_cases

# F is now contravariant powerset, rather than another representable. Native
# maps/compose helpers enumerate arrows, but truth-set preimages implement F.
yoneda = namespaces['yoneda']
maps, compose = yoneda['maps'], yoneda['compose']
objects = range(3)
probes = {n: maps(n, 2) for n in objects}
indices = {n: {u: i for i, u in enumerate(probes[n])} for n in objects}


def preimage_mask(arrow, mask):
    return sum(1 << x for x, y in enumerate(arrow) if mask & (1 << y))


yoneda_records = []
for kind in ['powerset', 'constant_two']:
    outputs = {n: tuple(range(1 << n)) if kind == 'powerset' else (0, 1) for n in objects}
    components = {n: tuple(product(outputs[n], repeat=len(probes[n]))) for n in objects}
    compatible = []
    candidates = 0
    for family in product(*(components[n] for n in objects)):
        candidates += 1
        valid = True
        for source in objects:
            for target in objects:
                for arrow in maps(source, target):
                    for probe in probes[target]:
                        lhs = family[source][indices[source][compose(probe, arrow)]]
                        target_value = family[target][indices[target][probe]]
                        rhs = preimage_mask(arrow, target_value) if kind == 'powerset' else target_value
                        if lhs != rhs:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            recovered = family[2][indices[2][(0, 1)]]
            for n in objects:
                for i, probe in enumerate(probes[n]):
                    assert family[n][i] == (preimage_mask(probe, recovered) if kind == 'powerset' else recovered)
            compatible.append(recovered)
    assert sorted(compatible) == list(outputs[2])
    yoneda_records.append({'functor': kind, 'candidate_families': candidates, 'compatible': compatible})
counts['nonrepresentable_yoneda_candidates'] = sum(r['candidate_families'] for r in yoneda_records)

# Actual homology helper versus separate binary linear algebra: every subset
# of tetrahedron faces, all six edges. No duplicate/noninjective vertex map
# is asserted by the lesson's injective relabelling helper.
def binary_rank(columns):
    pivots = {}
    for column in columns:
        value = column
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


edges = list(combinations(range(4), 2))
faces = list(combinations(range(4), 3))
d1_rank = binary_rank([(1 << a) ^ (1 << b) for a, b in edges])
for mask in range(16):
    selected_faces = [face for i, face in enumerate(faces) if mask & (1 << i)]
    columns = [sum(1 << edges.index(edge) for edge in combinations(face, 2)) for face in selected_faces]
    beta = len(edges) - d1_rank - binary_rank(columns)
    _, _, classes = namespaces['homology']['homology_classes'](edges, selected_faces)
    assert len(classes) == 2 ** beta
counts['changed_tetrahedron_quotients'] = 16

# Shared-parameter reverse mode: rerun the actual displayed calculation with
# changed assignments and compare its gradients to exact cubic coefficients
# and the algebraic derivative of the loss.
tangent_source = ast.parse(examples['tangent']['code'])
target_assignment = next(node for node in tangent_source.body if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Tuple)
                         and [x.id for x in node.targets[0].elts] == ['w', 'x', 'b', 'target'])
original_value = target_assignment.value
gradient_cases = 0
for values in [(-2, 1, 3, 4), (0, -3, 2, -1), (1, 0, -2, 5), (3, -2, 1, 0), (Q(1, 3), Q(-2, 5), Q(7, 4), Q(9, 2))]:
    literals = ','.join('Q(' + str(v.numerator) + ',' + str(v.denominator) + ')' if isinstance(v, Q) else 'Q(' + str(v) + ')' for v in values)
    target_assignment.value = ast.parse('(' + literals + ')', mode='eval').body
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(tangent_source), '<changed-tangent>', 'exec'), namespace)
    w, x, b, target = map(Q, values)
    residual = w * w * x + w * b - target
    assert namespace['w_gradient'] == residual * (2 * w * x + b)
    assert namespace['b_gradient'] == residual * w
    assert namespace['x_gradient'] == residual * w * w
    gradient_cases += 1
target_assignment.value = original_value
counts['actual_shared_parameter_changes'] = gradient_cases

# Changed schema migrations: an injective site relabelling reflects
# compatibility; a merging map can add pairs. All component squares use the
# actual native commute helper, with independently constructed relations.
migrations = 0
commute = namespaces['capstone']['commute']
for sensor_values in product(range(3), repeat=2):
    for tech_values in product(range(3), repeat=3):
        for target_values in product(range(2), repeat=3):
            old_s = dict(enumerate(sensor_values))
            old_t = dict(enumerate(tech_values))
            rename_s = {x: 'S' + str(x) for x in old_s}
            rename_t = {x: 'T' + str(x) for x in old_t}
            site = dict(enumerate(target_values))
            new_s = {rename_s[x]: site[y] for x, y in old_s.items()}
            new_t = {rename_t[x]: site[y] for x, y in old_t.items()}
            assert not commute(old_s, new_s, rename_s, site)
            assert not commute(old_t, new_t, rename_t, site)
            old_pairs = {(s, t) for s in old_s for t in old_t if old_s[s] == old_t[t]}
            new_pairs = {(s, t) for s in old_s for t in old_t if site[old_s[s]] == site[old_t[t]]}
            assert old_pairs <= new_pairs
            assert new_pairs - old_pairs == {(s, t) for s in old_s for t in old_t if old_s[s] != old_t[t] and site[old_s[s]] == site[old_t[t]]}
            migrations += 1
counts['changed_compatible_migrations'] = migrations

result = {'passed': True, 'checkedAt': datetime.now(timezone.utc).isoformat(), 'counts': counts,
          'maximumAbsoluteFloatError': maximum_error, 'yoneda': yoneda_records,
          'limits': 'Finite changed inputs support implementation; universal statements use the independent source proof review.'}
(ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
