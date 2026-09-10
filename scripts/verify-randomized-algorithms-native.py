"""Independent finite oracles for the actual authored Python and browser models."""
import contextlib
from collections import Counter
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from functools import cache
import io
from itertools import combinations, permutations, product
import json
from math import comb, factorial, isclose
from pathlib import Path
from random import Random

root = Path(__file__).resolve().parents[1]
directory = root / 'scratch/randomized-algorithms-verification'
examples = json.loads((directory / 'examples.json').read_text(encoding='utf-8'))
fixtures = json.loads((directory / 'model-fixtures.json').read_text(encoding='utf-8'))
functions = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'], name, 'exec'), namespace)
    functions[name] = namespace


class DrawTape:
    def __init__(self, choices):
        self.choices = iter(choices)

    def randrange(self, stop):
        value = next(self.choices)
        assert 0 <= value < stop
        return value

    def getrandbits(self, bits):
        value = next(self.choices)
        assert 0 <= value < 2 ** bits
        return value


mapping_cases = 0
for fixture in fixtures['mappings']:
    source, target = fixture['source'], fixture['target']
    accepted = source - source % target if fixture['reject'] else source
    counts = [sum(raw % target == result for raw in range(accepted)) for result in range(target)]
    assert counts == fixture['counts']
    assert accepted == fixture['accepted']
    assert isclose(source / accepted, fixture['expectedAttempts'])
    mapping_cases += 1
for limit in range(1, 32):
    for accepted in range(limit):
        bits = (limit - 1).bit_length()
        choices = [2 ** bits - 1, accepted] if limit < 2 ** bits else [accepted]
        assert functions['rejection']['uniform_below'](limit, DrawTape(choices)) == accepted

shuffle_paths = 0
for size in range(0, 7):
    values = list(range(size))
    counts = Counter()
    for draws in product(*(range(stop) for stop in range(size, 1, -1))):
        result = functions['shuffle']['shuffled'](values, DrawTape(draws))
        counts[tuple(result)] += 1
        assert values == list(range(size))
        shuffle_paths += 1
    assert set(counts) == set(permutations(values))
    assert set(counts.values()) == {1}

reservoir_paths = 0
for size in range(1, 8):
    for capacity in range(1, size + 1):
        counts = Counter()
        for draws in product(*(range(stop) for stop in range(capacity + 1, size + 1))):
            sample = functions['reservoir']['reservoir'](iter(range(size)), capacity, DrawTape(draws))
            counts[tuple(sorted(sample))] += 1
            reservoir_paths += 1
        assert set(counts) == set(combinations(range(size), capacity))
        assert len(set(counts.values())) == 1
for fixture in fixtures['reservoirs']:
    size, capacity = fixture['size'], fixture['capacity']
    actual = {tuple(entry['sample']): entry['probability'] for entry in fixture['distribution']}
    assert set(actual) == set(combinations(range(size), capacity))
    assert all(isclose(value, 1 / comb(size, capacity), abs_tol=1e-13) for value in actual.values())

weighted_tickets = 0
for weights in product(range(4), repeat=4):
    if not sum(weights):
        continue
    picker = functions['weighted']['WeightedPicker'](list(weights))
    counts = Counter(picker.pick(DrawTape([ticket])) for ticket in range(sum(weights)))
    assert [counts[index] for index in range(4)] == list(weights)
    weighted_tickets += sum(weights)

filtered_paths = 0
for size in range(1, 6):
    for values in product((0, 1), repeat=size):
        for target in set(values):
            eligible = [index for index, value in enumerate(values) if value == target]
            counts = Counter()
            for draws in product(*(range(stop) for stop in range(1, len(eligible) + 1))):
                selected = functions['filtered']['pick_matching_index'](values, target, DrawTape(draws))
                counts[selected] += 1
                filtered_paths += 1
            assert set(counts) == set(eligible)
            assert len(set(counts.values())) == 1

invalid_calls = [
    lambda: functions['rejection']['uniform_below'](0, Random(0)),
    lambda: functions['weighted']['WeightedPicker']([0, 0]),
    lambda: functions['weighted']['WeightedPicker']([1, -1]),
    lambda: functions['reservoir']['reservoir']([], -1, Random(0)),
    lambda: functions['filtered']['pick_matching_index']([1, 2], 3, Random(0)),
    lambda: functions['selection']['quickselect']([], 0, Random(0)),
    lambda: functions['selection']['quickselect']([1], 1, Random(0)),
    lambda: functions['verification']['probably_product']([[1.0]], [[1]], [[1]], 1, Random(0)),
    lambda: functions['verification']['probably_product']([[1]], [[1, 2]], [[1]], 1, Random(0)),
    lambda: functions['amplification']['majority_error'](2, Fraction(1, 4)),
    lambda: functions['estimation']['sample_budget'](0.1, 0),
]
for call in invalid_calls:
    try:
        call()
    except (ValueError, TypeError, LookupError):
        pass
    else:
        raise AssertionError('Invalid input unexpectedly accepted')


def untouched_stream():
    raise AssertionError('A zero-capacity call consumed its input')
    yield


assert functions['reservoir']['reservoir'](untouched_stream(), 0, Random(0)) == []

selection_cases = 0
for size in range(1, 7):
    for values in product((-1, 0, 1), repeat=size):
        for rank in range(size):
            for seed in (0, 1, 7):
                value, scanned = functions['selection']['quickselect'](values, rank, Random(seed))
                assert value == sorted(values)[rank]
                assert size <= scanned <= size * (size + 1) // 2
                selection_cases += 1


@cache
def rank_tree_distribution(values, wanted):
    """Enumerate full pivot trees with rational mass, not the model expectation recurrence."""
    distribution = Counter()
    for pivot in values:
        lower = tuple(value for value in values if value < pivot)
        upper = tuple(value for value in values if value > pivot)
        if pivot == wanted:
            distribution[len(values)] += Fraction(1, len(values))
        else:
            branch = lower if wanted < pivot else upper
            for work, probability in rank_tree_distribution(branch, wanted).items():
                distribution[len(values) + work] += probability / len(values)
    return distribution


expectation_cases = 0
for fixture in fixtures['expectations']:
    size, rank = fixture['size'], fixture['rank']
    expected = functions['expectedWork']['expected_scans'](size, rank)
    assert isclose(float(expected), fixture['expectation'], abs_tol=1e-12)
    if size <= 8:
        distribution = rank_tree_distribution(tuple(range(size)), rank)
        assert sum(distribution.values()) == 1
        assert sum(work * mass for work, mass in distribution.items()) == expected
    assert expected <= 4 * size
    expectation_cases += 1


def exact_product(left, right):
    return [[sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
             for column in range(len(right))] for row in range(len(left))]


probe_cases = 0
for entries in product((-1, 0, 1), repeat=4):
    error = [list(entries[:2]), list(entries[2:])]
    detections = 0
    for bits in product((0, 1), repeat=2):
        passed = functions['verification']['probe']([[1, 0], [0, 1]], error, [[0, 0], [0, 0]], bits)
        assert passed == all(sum(a * b for a, b in zip(row, bits)) == 0 for row in error)
        detections += not passed
        probe_cases += 1
    assert detections >= 2 if any(entries) else detections == 0
for fixture in fixtures['probes']:
    exact = exact_product(fixture['left'], fixture['right'])
    residual = [sum((value - claim) * bit for value, claim, bit in zip(row, claimed, fixture['bits']))
                for row, claimed in zip(exact, fixture['claimed'])]
    assert fixture['residual'] == residual
    assert fixture['passes'] == (not any(residual))

majority_cases = 0
for fixture in fixtures['majorities']:
    rounds, failure = fixture['rounds'], Fraction(fixture['failure'])
    total = Fraction(0)
    distribution = Counter()
    for outcomes in product((0, 1), repeat=rounds):
        errors = sum(outcomes)
        mass = failure ** errors * (1 - failure) ** (rounds - errors)
        distribution[errors] += mass
        if errors > rounds // 2:
            total += mass
    assert isclose(float(total), fixture['majorityError'], abs_tol=1e-13)
    for outcome in fixture['distribution']:
        assert isclose(float(distribution[outcome['errors']]), outcome['probability'], abs_tol=1e-13)
    assert functions['amplification']['majority_error'](rounds, failure) == total
    majority_cases += 1

for fixture in fixtures['budgets']:
    with localcontext() as context:
        context.prec = 60
        epsilon, delta = Decimal(str(fixture['epsilon'])), Decimal(str(fixture['delta']))
        exact_count = ((2 / delta).ln() / (2 * epsilon ** 2)).to_integral_value(rounding=ROUND_CEILING)
        assert int(exact_count) == fixture['budget']

hand = {
    'six_sided_rejection_expected': str(Fraction(16, 12)),
    'three_attempt_timeout': str(Fraction(1, 4) ** 3),
    'five_record_pair_probability': str(Fraction(1, comb(5, 2))),
    'three_vote_error_at_one_third': str(functions['amplification']['majority_error'](3, Fraction(1, 3))),
    'five_independent_probe_misses': str(Fraction(1, 2) ** 5),
    'estimation_005_001': functions['estimation']['sample_budget'](0.05, 0.01),
    'two_hundred_checks_rounds': next(rounds for rounds in range(1, 30) if Fraction(200, 2 ** rounds) <= Fraction(1, 100)),
}
assert hand == {'six_sided_rejection_expected': '4/3', 'three_attempt_timeout': '1/64', 'five_record_pair_probability': '1/10', 'three_vote_error_at_one_third': '7/27', 'five_independent_probe_misses': '1/32', 'estimation_005_001': 1060, 'two_hundred_checks_rounds': 15}
result = {'mappingCases': mapping_cases, 'shufflePaths': shuffle_paths, 'reservoirPaths': reservoir_paths, 'weightedTickets': weighted_tickets, 'filteredSamplePaths': filtered_paths, 'invalidNativeInputs': len(invalid_calls), 'zeroCapacityDoesNotConsume': True, 'selectionCases': selection_cases, 'exactExpectationCases': expectation_cases, 'errorMatrixProbes': probe_cases, 'majorityDistributions': majority_cases, 'sampleBudgets': len(fixtures['budgets']), 'handExercises': hand}
print(json.dumps(result))
