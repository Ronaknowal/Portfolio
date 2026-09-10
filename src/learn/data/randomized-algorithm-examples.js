export const randomizedAlgorithmExamples = {
  rejection: {
    title: 'Turn unbiased bits into an unbiased bounded integer',
    code: String.raw`from collections import Counter
from random import Random


def uniform_below(limit, rng):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    bits = (limit - 1).bit_length()
    while True:
        candidate = rng.getrandbits(bits)
        if candidate < limit:
            return candidate


print("modulo:", sorted(Counter(raw % 3 for raw in range(8)).items()))
print("accepted:", sorted(Counter(raw % 3 for raw in range(6)).items()))
rng = Random(17)
print("draws:", [uniform_below(3, rng) for _ in range(8)])
print("one outcome:", uniform_below(1, rng))`,
    expected: String.raw`modulo: [(0, 3), (1, 3), (2, 2)]
accepted: [(0, 2), (1, 2), (2, 2)]
draws: [2, 1, 1, 1, 1, 0, 2, 2]
one outcome: 0`
  },
  shuffle: {
    title: 'Shuffle occurrence identities and check every small path',
    code: String.raw`from collections import Counter
from itertools import product
from random import Random


def shuffled(values, rng):
    result = list(values)
    for end in range(len(result) - 1, 0, -1):
        chosen = rng.randrange(end + 1)
        result[end], result[chosen] = result[chosen], result[end]
    return result


def all_three_paths():
    counts = Counter()
    for first, second in product(range(3), range(2)):
        values = list("ABC")
        for end, chosen in [(2, first), (1, second)]:
            values[end], values[chosen] = values[chosen], values[end]
        counts["".join(values)] += 1
    return sorted(counts.items())


original = [10, 20, 30, 40]
print("one shuffle:", shuffled(original, Random(17)))
print("original:", original)
print("all paths:", all_three_paths())
print("empty:", shuffled([], Random(0)))`,
    expected: String.raw`one shuffle: [10, 30, 20, 40]
original: [10, 20, 30, 40]
all paths: [('ABC', 1), ('ACB', 1), ('BAC', 1), ('BCA', 1), ('CAB', 1), ('CBA', 1)]
empty: []`
  },
  weighted: {
    title: 'Route integer tickets through prefix ends',
    code: String.raw`from bisect import bisect_right
from itertools import accumulate
from random import Random


class WeightedPicker:
    def __init__(self, weights):
        if not weights or any(type(w) is not int or w < 0 for w in weights):
            raise ValueError("weights must be nonnegative integers")
        self.ends = list(accumulate(weights))
        if self.ends[-1] == 0:
            raise ValueError("total weight must be positive")

    def pick(self, rng):
        ticket = rng.randrange(self.ends[-1])
        return bisect_right(self.ends, ticket)


picker = WeightedPicker([1, 3, 2])
print("ends:", picker.ends)
print("every ticket:", [bisect_right(picker.ends, t) for t in range(6)])
print("one draw:", picker.pick(Random(17)))
with_zero = WeightedPicker([0, 2, 0, 1])
print("zero weights:", [bisect_right(with_zero.ends, t) for t in range(3)])`,
    expected: String.raw`ends: [1, 4, 6]
every ticket: [0, 1, 1, 1, 2, 2]
one draw: 2
zero weights: [1, 1, 3]`
  },
  reservoir: {
    title: 'Keep a uniform k-subset in one pass',
    code: String.raw`from random import Random


def reservoir(stream, capacity, rng):
    if type(capacity) is not int or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer")
    sample = []
    if capacity == 0:
        return sample
    for seen, record in enumerate(stream, start=1):
        if seen <= capacity:
            sample.append(record)
        else:
            chosen = rng.randrange(seen)
            if chosen < capacity:
                sample[chosen] = record
    return sample


records = list(enumerate(["same", "other", "same", "last"]))
print("occurrences:", reservoir(iter(records), 2, Random(17)))
print("short stream:", reservoir(iter("AB"), 5, Random(0)))
print("empty stream:", reservoir(iter([]), 2, Random(0)))
print("zero capacity:", reservoir(iter("ABC"), 0, Random(0)))`,
    expected: String.raw`occurrences: [(0, 'same'), (1, 'other')]
short stream: ['A', 'B']
empty stream: []
zero capacity: []`
  },
  filtered: {
    title: 'Sample matching positions rather than all records',
    code: String.raw`from random import Random


def pick_matching_index(values, target, rng):
    matching_count = 0
    chosen = None
    for index, value in enumerate(values):
        if value == target:
            matching_count += 1
            if rng.randrange(matching_count) == 0:
                chosen = index
    if chosen is None:
        raise LookupError("target is absent")
    return chosen


values = [9, 2, 9, 5, 9]
print("matching index:", pick_matching_index(values, 9, Random(17)))
print("unique match:", pick_matching_index(values, 2, Random(17)))
try:
    pick_matching_index(values, 7, Random(17))
except LookupError as error:
    print(type(error).__name__ + ": " + str(error))`,
    expected: String.raw`matching index: 0
unique match: 1
LookupError: target is absent`
  },
  selection: {
    title: 'Select exactly with three-way randomized partitions',
    code: String.raw`from random import Random


def quickselect(values, rank, rng):
    active = list(values)
    if type(rank) is not int or not 0 <= rank < len(active):
        raise ValueError("rank must index a nonempty input")
    scanned = 0
    while active:
        pivot = active[rng.randrange(len(active))]
        lower, equal, upper = [], [], []
        for value in active:
            scanned += 1
            if value < pivot:
                lower.append(value)
            elif value > pivot:
                upper.append(value)
            else:
                equal.append(value)
        if rank < len(lower):
            active = lower
        elif rank < len(lower) + len(equal):
            return pivot, scanned
        else:
            rank -= len(lower) + len(equal)
            active = upper


values = [8, 1, 6, 3, 9, 2, 7, 4, 5]
print("median and scans:", quickselect(values, 4, Random(17)))
print("duplicates:", quickselect([4, 4, 1, 4, 2], 3, Random(17)))
print("singleton:", quickselect([12], 0, Random(0)))
print("original:", values)`,
    expected: String.raw`median and scans: (5, 9)
duplicates: (4, 8)
singleton: (12, 1)
original: [8, 1, 6, 3, 9, 2, 7, 4, 5]`
  },
  expectedWork: {
    title: 'Average over every pivot, using exact fractions',
    code: String.raw`from fractions import Fraction
from functools import cache


@cache
def expected_scans(size, rank):
    total = Fraction(size)
    for pivot_rank in range(size):
        if pivot_rank < rank:
            rest = expected_scans(size - pivot_rank - 1, rank - pivot_rank - 1)
        elif pivot_rank > rank:
            rest = expected_scans(pivot_rank, rank)
        else:
            rest = 0
        total += Fraction(rest, size)
    return total


print("three ranks:", [str(expected_scans(3, rank)) for rank in range(3)])
print("nine-value median:", expected_scans(9, 4))
print("extreme pivots, maximum:", sum(range(1, 10)))`,
    expected: String.raw`three ranks: ['25/6', '14/3', '25/6']
nine-value median: 6367/315
extreme pivots, maximum: 45`
  },
  verification: {
    title: 'Verify an integer product without forming it',
    code: String.raw`from itertools import product
from random import Random


def matvec(matrix, vector):
    return [sum(value * coordinate for value, coordinate in zip(row, vector))
            for row in matrix]


def probe(left, right, claimed, vector):
    return matvec(left, matvec(right, vector)) == matvec(claimed, vector)


def probably_product(left, right, claimed, rounds, rng):
    size = len(left)
    if type(rounds) is not int or rounds < 1 or size < 1:
        raise ValueError("positive size and number of rounds required")
    for matrix in (left, right, claimed):
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("matrices must be square and equally sized")
        if any(type(value) is not int for row in matrix for value in row):
            raise TypeError("this contract requires exact integers")
    for _ in range(rounds):
        vector = [rng.randrange(2) for _ in range(size)]
        if not probe(left, right, claimed, vector):
            return False
    return True


left = [[1, 2], [0, 1]]
right = [[2, 0], [1, 3]]
correct = [[4, 6], [1, 3]]
wrong = [[5, 5], [1, 3]]
print("wrong product probes:", [(r, probe(left, right, wrong, r))
                                for r in product((0, 1), repeat=2)])
print("correct product:", probably_product(left, right, correct, 5, Random(17)))
print("wrong product:", probably_product(left, right, wrong, 5, Random(17)))`,
    expected: String.raw`wrong product probes: [((0, 0), True), ((0, 1), False), ((1, 0), False), ((1, 1), True)]
correct product: True
wrong product: False`
  },
  amplification: {
    title: 'Count the actual failure event before raising a probability',
    code: String.raw`from fractions import Fraction
from math import comb


def majority_error(rounds, failure):
    if rounds < 1 or rounds % 2 == 0:
        raise ValueError("use a positive odd number of binary votes")
    return sum((Fraction(comb(rounds, errors)) * failure ** errors
                * (1 - failure) ** (rounds - errors)
                for errors in range(rounds // 2 + 1, rounds + 1)), Fraction(0))


print("three independent false passes:", Fraction(1, 2) ** 3)
print("one random vector reused:", Fraction(1, 2))
print("three-vote error:", majority_error(3, Fraction(1, 4)))
tests = 1000
rounds = 20
print("union upper bound:", Fraction(tests, 2 ** rounds))`,
    expected: String.raw`three independent false passes: 1/8
one random vector reused: 1/2
three-vote error: 5/32
union upper bound: 125/131072`
  },
  estimation: {
    title: 'Choose a sufficient budget for a bounded estimate',
    code: String.raw`from math import ceil, exp, log
from random import Random


def sample_budget(epsilon, delta):
    if not 0 < epsilon <= 1 or not 0 < delta < 1:
        raise ValueError("require 0 < epsilon <= 1 and 0 < delta < 1")
    return ceil(log(2 / delta) / (2 * epsilon * epsilon))


epsilon, delta = 0.1, 0.05
count = sample_budget(epsilon, delta)
rng = Random(17)
# The population has exactly three marked tickets among ten.
estimate = sum(rng.randrange(10) < 3 for _ in range(count)) / count
print("budget:", count)
print("estimate:", round(estimate, 6))
print("failure upper bound:", round(2 * exp(-2 * count * epsilon ** 2), 6))
print("half tolerance:", sample_budget(epsilon / 2, delta))`,
    expected: String.raw`budget: 185
estimate: 0.259459
failure upper bound: 0.049447
half tolerance: 738`
  },
  replay: {
    title: 'Replay a checkpoint without resetting every operation',
    code: String.raw`from random import Random


rng = Random(17)
first = [rng.randrange(10) for _ in range(4)]
saved = rng.getstate()
later = [rng.randrange(10) for _ in range(4)]
rng.setstate(saved)
replayed = [rng.randrange(10) for _ in range(4)]
print("first:", first)
print("later:", later)
print("replayed:", replayed)
print("same checkpoint:", later == replayed)
print("reseeded every time:", [Random(17).randrange(10) for _ in range(4)])`,
    expected: String.raw`first: [8, 6, 4, 5]
later: [4, 2, 8, 4]
replayed: [4, 2, 8, 4]
same checkpoint: True
reseeded every time: [8, 8, 8, 8]`
  }
};
