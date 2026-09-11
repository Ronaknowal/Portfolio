"""Separate review of actual displayed helpers and larger model domains."""
import contextlib
from datetime import datetime, timezone
import io
import itertools
import json
from pathlib import Path

directory = Path('scratch/sets-logic-independent')
fixtures = json.loads((directory / 'fixtures.json').read_text(encoding='utf-8'))
counts = {'largerRelations': 0, 'largerDiagonalBoards': 0, 'programs': 0,
          'truthFunctionArguments': 0, 'changedDiagonals': 0, 'changedOrders': 0}
for row in fixtures['relations']:
    n, pairs, result = row['size'], set(map(tuple, row['pairs'])), row['result']
    nodes = set(range(n))
    paths = {(a, c) for a, b in pairs for other, c in pairs if other == b}
    expected = {'reflexive': all((a, a) in pairs for a in nodes),
                'symmetric': pairs == {(b, a) for a, b in pairs},
                'transitive': paths <= pairs,
                'antisymmetric': not any(a != b and (b, a) in pairs for a, b in pairs)}
    assert result['properties'] == expected
    if result['partialOrder']:
        # A reported cover is indispensable to recovering all strict comparisons.
        covers = set(map(tuple, result['covers']))
        def reach(edges):
            closure = set(edges)
            while True:
                changed = closure | {(a, c) for a, b in closure for other, c in closure if other == b}
                if changed == closure:
                    return closure
                closure = changed
        strict = {(a, b) for a, b in pairs if a != b}
        assert reach(covers) == strict
        for edge in covers:
            assert edge not in reach(covers - {edge})
        assert set(result['minimal']) == nodes - {b for a, b in strict}
        assert set(result['maximal']) == nodes - {a for a, b in strict}
        assert result['least'] == sorted(a for a in nodes if sum(x == a for x, y in pairs) == n)
        assert result['greatest'] == sorted(b for b in nodes if sum(y == b for x, y in pairs) == n)
    counts['largerRelations'] += 1
for row in fixtures['diagonals']:
    sets = [{j for j, value in enumerate(values) if value} for values in row['matrix']]
    missing = set(row['result']['members'])
    assert all((i in missing) != (i in values) and missing != values for i, values in enumerate(sets))
    counts['largerDiagonalBoards'] += 1

contexts = {}
for name, example in fixtures['examples'].items():
    context, stdout = {}, io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(example['code'], name, 'exec'), context)
    assert stdout.getvalue().strip() == example['expected'].strip(), name
    contexts[name] = context
    counts['programs'] += 1
# All truth functions are supplied through the actual helper's callable API.
for first_mask in range(16):
    for second_mask in range(16):
        for conclusion_mask in [0, 1, 6, 8, 15]:
            truth = lambda mask: lambda p, q: bool(mask & (1 << (2 * int(p) + int(q))))
            admitted_mask = first_mask & second_mask
            rejected_mask = admitted_mask & ~conclusion_mask & 15
            result = contexts['truth']['inspect']([truth(first_mask), truth(second_mask)], truth(conclusion_mask))
            assert result['valid'] == (rejected_mask == 0)
            assert result['consistent'] == (admitted_mask != 0)
            assert {(2 * int(p) + int(q)) for p, q in result['countermodels']} == {i for i in range(4) if rejected_mask & (1 << i)}
            counts['truthFunctionArguments'] += 1
for n in range(9):
    for seed in range(12):
        rows = [{j for j in range(n) if (i * 7 + j * 3 + seed) % 5 < 2} for i in range(n)]
        missing = contexts['diagonal']['missing_subset'](rows)
        assert missing <= set(range(n)) and all(missing != row for row in rows)
        counts['changedDiagonals'] += 1
for values in [[12, 4, 24, 6], [30, 6, 10, 15, 2, 3, 5, 1], [7, 49, 343, 2401], [11, 13, 17], []]:
    result = contexts['orders']['order_summary'](values)
    relation = {(a, b) for a in values for b in values if b % a == 0}
    strict = {(a, b) for a, b in relation if a != b}
    composed = {(a, c) for a, b in strict for other, c in strict if other == b}
    assert set(map(tuple, result['covers'])) == strict - composed
    assert set(result['minimal']) == set(values) - {b for a, b in strict}
    counts['changedOrders'] += 1
assert all(result['rejected'] for result in fixtures['inputResults'])
result = {'reviewedAt': datetime.now(timezone.utc).isoformat(), 'passed': True, 'counts': counts,
          'malformedRegressions': fixtures['inputResults'],
          'limits': 'Complementary finite checks, not a substitute for the separately completed full proof/source review.'}
(directory / 'native-results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
