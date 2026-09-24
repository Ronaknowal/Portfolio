import contextlib
import io
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

directory = Path('scratch/sets-logic-verification')
examples = json.loads((directory / 'examples.json').read_text(encoding='utf-8'))
contexts = {}
for name, example in examples.items():
    stream = io.StringIO()
    context = {}
    with contextlib.redirect_stdout(stream):
        exec(compile(example['code'], name, 'exec'), context)
    assert stream.getvalue().strip() == example['expected'].strip(), name
    contexts[name] = context
counts = {'completePrograms': len(examples), 'parityInputs': 0, 'quantifierBoards': 0,
          'relationChecks': 0, 'changedOrders': 0, 'diagonalBoards': 0,
          'finiteCounterexamples': 0, 'changedPolicy': 0, 'setAndFunctionProofCases': 0}
for n in range(-101, 102, 2):
    result = contexts['parity']['odd_square_witness'](n)
    # The function's returned witness must reconstruct the original square.
    assert n == 2 * result[0] + 1
    assert n * n == 2 * result[1] + 1
    counts['parityInputs'] += 1
for jobs_count in range(4):
    for reviewer_count in range(4):
        jobs, reviewers = list(range(jobs_count)), list(range(reviewer_count))
        for mask in range(2 ** (jobs_count * reviewer_count)):
            approvals = {(j, r) for j in jobs for r in reviewers if mask & (1 << (j * reviewer_count + r))}
            result = contexts['quantifiers']['evaluate'](jobs, reviewers, approvals)
            each = all(sum((j, r) in approvals for r in reviewers) > 0 for j in jobs)
            common = [r for r in reviewers if sum((j, r) in approvals for j in jobs) == jobs_count]
            assert result[0] == each and result[1] == bool(common)
            assert result[3] == common
            counts['quantifierBoards'] += 1
for mask in range(512):
    elements = list(range(3))
    relation = {(a, b) for a in elements for b in elements if mask & (1 << (3 * a + b))}
    result = contexts['relations']['inspect_relation'](elements, relation)
    closure = set(relation)
    for middle in elements:
        closure |= {(a, b) for a in elements for b in elements if (a, middle) in closure and (middle, b) in closure}
    expected = (all((a, a) in relation for a in elements)
                and relation == {(b, a) for a, b in relation} and closure == relation)
    assert (result[1] is not None) == expected
    counts['relationChecks'] += 1
for values in [[2, 5, 10, 20], [3, 9, 27], [5, 7, 11], [8], []]:
    result = contexts['orders']['order_summary'](values)
    for a, b in result['covers']:
        assert b % a == 0 and a != b
        assert not any(a != c != b and c % a == 0 and b % c == 0 for c in values)
    assert result['least'] == [a for a in values if all(b % a == 0 for b in values)]
    counts['changedOrders'] += 1
for size in range(4):
    for mask in range(2 ** (size * size)):
        rows = [{column for column in range(size) if mask & (1 << (row * size + column))} for row in range(size)]
        subset = contexts['diagonal']['missing_subset'](rows)
        assert all(subset != row for row in rows)
        counts['diagonalBoards'] += 1
for last in range(20):
    f = contexts['finiteTesting']['vanishes_on_tested_values']
    assert all(f(n, last) == 0 for n in range(last + 1))
    assert f(last + 1, last) == math.factorial(last + 1)
    counts['finiteCounterexamples'] += 1
policy = contexts['policy']
policy['badged'].discard('Bo')
strict = {job: people & (policy['trained'] & policy['badged']) for job, people in policy['changed'].items()}
result = policy['audit'](strict)
assert result['uncovered'] == ['Scan', 'Label', 'Audit']
assert result['unqualified assignments'] == {} and result['common'] == []
counts['changedPolicy'] += 1
universe = set(range(3))
subsets = [set(x for x in universe if mask & (1 << x)) for mask in range(8)]
for a, b, c in itertools.product(subsets, repeat=3):
    assert a - (b | c) == (a - b) & (a - c)
    assert a - (b & c) == (a - b) | (a - c)
    counts['setAndFunctionProofCases'] += 1
for images in itertools.product(range(3), repeat=3):
    for c, d in itertools.product(subsets, repeat=2):
        preimage = lambda target: {x for x in universe if images[x] in target}
        assert preimage(c | d) == preimage(c) | preimage(d)
        assert preimage(c & d) == preimage(c) & preimage(d)
        counts['setAndFunctionProofCases'] += 1
(directory / 'native-results.json').write_text(json.dumps({'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True, 'counts': counts}, indent=2), encoding='utf-8')
print(json.dumps(counts))
