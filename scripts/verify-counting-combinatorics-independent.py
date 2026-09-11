"""Reviewer checks using coefficient DP, exact moments and period decomposition."""
import contextlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

directory = Path('scratch/counting-combinatorics-independent-review')
fixtures = json.loads((directory / 'fixtures.json').read_text(encoding="utf-8"))
counts = Counter()
contexts = {}
for name, example in fixtures['examples'].items():
    context = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], name, 'exec'), context)
    assert output.getvalue().rstrip() == example['expected'], name
    contexts[name] = context
    counts['completeDisplayedPrograms'] += 1

for case in fixtures['allocations']:
    # Direct coefficient construction uses only addition; it does not use IE.
    row = [1]
    for low, high in zip(case['minimums'], case['capacities']):
        next_row = [0] * (len(row) + high)
        for amount in range(low, high + 1):
            for index, coefficient in enumerate(row):
                next_row[index + amount] += coefficient
        row = next_row
    total = case['total']
    expected = row[total] if total < len(row) else 0
    assert int(case['count']) == expected
    actual_native = contexts['allocations']['bounded_count'](total, case['capacities'], case['minimums'])
    assert actual_native == expected
    counts['ChangedAllocationModelAndNativeCases'] += 1

for case in fixtures['coefficients']:
    for length, raw in enumerate(case['rows']):
        row = list(map(int, raw))
        caps = case['capacities'][:length]
        cardinality = math.prod(cap + 1 for cap in caps)
        mean = Fraction(sum(caps), 2)
        variance = sum((Fraction(cap * (cap + 2), 12) for cap in caps), Fraction(0))
        assert row == row[::-1]
        assert sum(row) == cardinality
        assert sum(index * count for index, count in enumerate(row)) == cardinality * mean
        assert sum((index - mean) ** 2 * count for index, count in enumerate(row)) == cardinality * variance
        counts['CoefficientRowsExactMassMeanVarianceSymmetry'] += 1

for case in fixtures['choices']:
    n, k = case['n'], case['length']
    data = case['data']
    expected_descriptions = n ** k if case['repeats'] else (math.factorial(n) // math.factorial(n-k) if k <= n else 0)
    expected_groups = (math.comb(n+k-1, k) if n else int(k == 0)) if case['repeats'] else (math.comb(n, k) if k <= n else 0)
    assert len(data['descriptions']) == expected_descriptions
    assert len(data['fibers']) == expected_groups
    for group in data['fibers']:
        expected_fiber = math.factorial(k) // math.prod(math.factorial(v) for v in Counter(group['key']).values())
        assert len(group['members']) == expected_fiber
        assert len(set(map(tuple, group['members']))) == expected_fiber
        counts['ExactMultinomialFiberCases'] += 1
    counts['ChoiceStates'] += 1

images = set()
good = 0
for data in fixtures['paths']:
    word = data['word']
    # Stack recognizer supplies an independent syntax validity condition.
    stack = []
    valid = True
    for symbol in word:
        if symbol == '(':
            stack.append(symbol)
        elif stack:
            stack.pop()
        else:
            valid = False
            break
    valid = valid and not stack
    assert valid == data['valid']
    if valid:
        assert '(' + data['inside'] + ')' + data['after'] == word
        good += 1
    else:
        reflected = data['reflected']
        assert sum(reflected) == 2
        boundary = next(i for i in range(1, len(reflected)+1) if sum(reflected[:i]) == 1)
        recovered = [-step if i < boundary else step for i, step in enumerate(reflected)]
        assert ''.join('(' if step == 1 else ')' for step in recovered) == word
        images.add(tuple(reflected))
    counts['FourteenSymbolPathBoundaryCases'] += 1
assert good == 429
assert len(images) == math.comb(14, 6) == len(fixtures['paths']) - good

def primitive_count(length):
    # All words have a unique least period dividing length. Subtract smaller
    # primitive-period families; no rotation canonicalization or fixed sums.
    primitive = {}
    for n in range(1, length + 1):
        primitive[n] = 2**n - sum(primitive[d] for d in range(1, n) if n % d == 0)
    return primitive

for n in range(1, 13):
    primitive = primitive_count(n)
    by_period = {d: primitive[d] // d for d in range(1, n+1) if n % d == 0}
    native_groups, native_fixed = contexts['rotations']['orbit_data'](n)
    assert Counter(map(len, native_groups.values())) == by_period
    assert sum(native_fixed) == n * sum(by_period.values())
    for shift, fixed in enumerate(native_fixed):
        assert fixed == 2 ** math.gcd(n, shift)
    if n <= 8:
        browser = fixtures['rings'][n-1]['data']
        assert Counter(len(group['members']) for group in browser['orbits']) == by_period
        assert browser['fixed'] == native_fixed
    counts['PrimitivePeriodModelNativeComparisons'] += 1

for case in fixtures['binomial']:
    n, k = case['n'], case['k']
    expected = math.comb(n, k) if 0 <= k <= n else 0
    assert int(case['value']) == expected
    counts['LargeExactBinomialCases'] += 1

for n in range(10):
    # Onto maps by complement; different from the native Stirling recurrence.
    for k in range(11):
        onto = sum((-1)**j * math.comb(k, j) * (k-j)**n for j in range(k+1))
        assert contexts['recurrences']['stirling'](n, k) * math.factorial(k) == onto
        counts['ChangedNativeStirlingOntoIdentity'] += 1

report = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
          'counts': dict(counts), 'scope': 'Exact finite complementary checks; the arbitrary proofs were read separately.'}
(directory / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
