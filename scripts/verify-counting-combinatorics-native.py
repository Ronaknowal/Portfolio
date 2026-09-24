"""Independent exact finite constructions, actual programs and changed practice."""

import contextlib
import io
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

directory = Path('scratch/counting-combinatorics-verification')
fixtures = json.loads((directory / 'model-fixtures.json').read_text(encoding='utf-8'))
examples = json.loads((directory / 'examples.json').read_text(encoding='utf-8'))
contexts = {}
for key, example in examples.items():
    output = io.StringIO()
    context = {}
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], key, 'exec'), context)
    assert output.getvalue().rstrip() == example['expected'], key
    contexts[key] = context

counts = {'actualPrograms': len(examples), 'choiceStates': 0, 'allocationStates': 0,
          'overlapBoards': 0, 'inductionStates': 0, 'pathWords': 0,
          'coefficientRows': 0, 'rotationLengths': 0, 'binomialCases': 0,
          'changedPracticeTasks': 0, 'nativePartitions': 0, 'nativeDerangements': 0}

for fixture in fixtures['choices']:
    n, length = fixture['labels'], fixture['length']
    all_words = list(itertools.product('ABCDE'[:n], repeat=length))
    if not fixture['repeats']:
        all_words = [word for word in all_words if len(set(word)) == len(word)]
    result = fixture['result']
    assert {tuple(word) for word in result['descriptions']} == set(all_words)
    expected = Counter(''.join(sorted(word)) for word in all_words)
    assert {group['key']: len(group['members']) for group in result['fibers']} == dict(expected)
    assert result['uniform'] == (len(set(expected.values())) == 1)
    counts['choiceStates'] += 1

for fixture in fixtures['allocations']:
    capacities, lows, total = fixture['capacities'], fixture['minimums'], fixture['total']
    direct = list(itertools.product(*(range(low, cap + 1) for low, cap in zip(lows, capacities))))
    expected = {values for values in direct if sum(values) == total}
    assert int(fixture['count']) == len(expected)
    assert {tuple(values) for values in fixture['values']} == expected
    assert contexts['allocations']['bounded_count'](total, capacities, lows) == len(expected)
    counts['allocationStates'] += 1

for fixture in fixtures['overlaps']:
    board = fixture['memberships']
    result = fixture['result']
    assert result['union'] == [index + 1 for index, row in enumerate(board) if sum(row)]
    for phase in range(1, 4):
        expected = [sum((-1) ** (selected + 1) * math.comb(sum(row), selected)
                        for selected in range(1, phase + 1) if selected <= sum(row))
                    for row in board]
        assert result['stages'][phase - 1]['weights'] == expected
        assert result['stages'][phase - 1]['total'] == sum(expected)
    for term in result['terms']:
        assert term['members'] == [index + 1 for index, row in enumerate(board)
                                   if all(row[column] for column in term['sets'])]
    counts['overlapBoards'] += 1

for fixture in fixtures['induction']:
    result = fixture['result']
    small, large, lower, target = result['small'], result['large'], result['lower'], fixture['target']
    # Independent forward closure from the enabled base certificates.
    reachable = {lower + index for index, present in enumerate(fixture['enabled']) if present}
    for value in range(lower, target + 1):
        if value in reachable:
            reachable.add(value + small)
    assert result['supported'] == (target in reachable)
    candidates = [(first, second) for first in range(target // small + 1)
                  for second in range(target // large + 1) if first * small + second * large == target]
    assert bool(result['actualWitness']) == bool(candidates)
    assert tuple(result['actualWitness']) in candidates
    if result['witness'] is not None:
        assert tuple(result['witness']) in candidates
    assert result['chain'][0] == target and result['chain'][-1] == result['base']
    assert all(a - b == small for a, b in zip(result['chain'], result['chain'][1:]))
    counts['inductionStates'] += 1

for fixture in fixtures['catalan']:
    n, result = fixture['pairs'], fixture['result']
    # Dynamic ballot-prefix counts provide an oracle independent of reflection and the closed form.
    states = {(0, 0): 1}
    for _ in range(2 * n):
        next_states = defaultdict(int)
        for (opens, closes), count in states.items():
            if opens < n:
                next_states[opens + 1, closes] += count
            if closes < opens:
                next_states[opens, closes + 1] += count
        states = next_states
    assert len(result['valid']) == states.get((n, n), 1 if n == 0 else 0)
    assert int(result['catalan'][-1]) == len(result['valid']) == contexts['catalan']['catalan'](n)
    images = set()
    for path in fixture['paths']:
        word = path['word']
        heights = [0]
        for symbol in word:
            heights.append(heights[-1] + (1 if symbol == '(' else -1))
        assert heights == path['heights']
        if min(heights) >= 0:
            assert path['valid']
            if word:
                assert '(' + path['inside'] + ')' + path['after'] == word
                assert contexts['catalan']['first_height'](path['inside'], -1) is None
                assert contexts['catalan']['first_height'](path['after'], -1) is None
        else:
            reflected = tuple(path['reflected'])
            assert sum(reflected) == 2
            image = ''.join('(' if step == 1 else ')' for step in reflected)
            boundary = contexts['catalan']['first_height'](image, 1)
            assert contexts['catalan']['flip_prefix'](image, boundary) == word
            images.add(image)
        counts['pathWords'] += 1
    if n:
        target_family = {''.join(')' if index in downs else '(' for index in range(2*n))
                         for downs in itertools.combinations(range(2*n), n-1)}
        assert images == target_family
    assert int(result['total']) == math.comb(2*n, n)

for fixture in fixtures['coefficients']:
    capacities = fixture['capacities']
    for included, row in enumerate(fixture['rows']):
        expected = Counter(sum(values) for values in itertools.product(*(range(cap + 1) for cap in capacities[:included])))
        assert [int(value) for value in row] == [expected[degree] for degree in range(sum(capacities[:included]) + 1)]
        counts['coefficientRows'] += 1
    assert contexts['coefficients']['coefficient_rows'](capacities) == [[int(value) for value in row] for row in fixture['rows']]

for fixture in fixtures['rotations']:
    n, result = fixture['length'], fixture['result']
    # Permutation cycles determine independent fixed counts; no enumeration of the words is used here.
    fixed_counts = []
    for shift in range(n):
        unseen = set(range(n))
        cycles = 0
        while unseen:
            vertex = min(unseen)
            while vertex in unseen:
                unseen.remove(vertex)
                vertex = (vertex + shift) % n
            cycles += 1
        fixed_counts.append(2 ** cycles)
    assert result['fixed'] == fixed_counts
    assert sum(fixed_counts) == n * len(result['orbits'])
    grouped = [word for group in result['orbits'] for word in group['members']]
    assert len(grouped) == len(set(grouped)) == 2 ** n
    for group in result['orbits']:
        word = group['key']
        assert set(group['members']) == {word[shift:] + word[:shift] for shift in range(n)}
        stabilizer = sum(word[shift:] + word[:shift] == word for shift in range(n))
        assert len(group['members']) * stabilizer == n
    counts['rotationLengths'] += 1

for fixture in fixtures['binomials']:
    n, k = fixture['n'], fixture['k']
    expected = math.comb(n, k) if 0 <= k <= n else 0
    assert int(fixture['count']) == expected
    counts['binomialCases'] += 1

for n in range(7):
    # Restricted-growth labels independently canonicalize set partitions.
    labels = [()] if n == 0 else [(0,)]
    for _ in range(1, n):
        labels = [prefix + (new,) for prefix in labels for new in range(max(prefix) + 2)]
    expected = Counter((max(word) + 1 if word else 0) for word in labels)
    direct = contexts['recurrences']['partitions'](tuple(range(n)))
    assert Counter(map(len, direct)) == expected
    assert len({tuple(sorted(tuple(sorted(block)) for block in grouping)) for grouping in direct}) == len(direct)
    for k in range(n + 2):
        assert contexts['recurrences']['stirling'](n, k) == expected[k]
        counts['nativePartitions'] += 1
for n in range(8):
    direct = sum(all(value != position for position, value in enumerate(order)) for order in itertools.permutations(range(n)))
    assert contexts['overlap']['derangements'](n) == direct
    counts['nativeDerangements'] += 1

# Independently execute the numerical promises made in all twelve changed tasks.
allowed = [(a, b) for a, b in itertools.permutations('ABCD', 2) if (a, b) not in [('A', 'D'), ('B', 'C')]]
assert len(allowed) == 10 and len({frozenset(pair) for pair in allowed}) == 6
assert len(set(itertools.permutations('AABBB'))) == 10 and math.factorial(5) == 120
assert contexts['allocations']['bounded_count'](7, [4,3,3], [1,0,2]) == 7
assert 8+7+6-3-2-2+1 == 15
assert sum(math.comb(j,3) for j in range(3,7)) == 35 == math.comb(7,4)
assert (37+5)//6 == 7 and sum([7,6,6,6,6,6]) == 37
assert contexts['induction']['witness'](29,3,5,8,{8:(1,1),9:(3,0),10:(0,2)}) == (8,1)
assert contexts['catalan']['flip_prefix']('())(', 3) == ')((('
assert len({frozenset([frozenset(first), frozenset(set('ABCDE')-set(first))])
            for length in range(1,5) for first in itertools.combinations('ABCDE',length)}) == 15
assert contexts['coefficients']['coefficient_rows']([1,2,2])[-1] == [1,3,5,5,3,1]
assert len(contexts['rotations']['orbit_data'](5)[0]) == 8
assert contexts['capstone']['count_plan'](3,2,5,[3,3,3]) == {'teams':7,'allocations':12,'combined':84}
assert contexts['capstone']['count_plan'](2,1,4,[2,2,2]) == {'teams':9,'allocations':6,'combined':54}
assert contexts['capstone']['count_plan'](3,4,5,[3,3,3]) == {'teams':0,'allocations':12,'combined':0}
counts['changedPracticeTasks'] = 12
record = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True, 'counts': counts,
          'scope': 'Exact independent finite constructions plus all actual displayed programs and changed tasks; general mathematical proofs require separate source review.'}
(directory / 'native-results.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2))
