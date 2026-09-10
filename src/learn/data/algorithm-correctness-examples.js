export const algorithmCorrectnessExamples = {
  search: {
    title: 'Find the first occurrence under an explicit contract',
    code: `def first_index(values, target):
    # Requires: a finite list of integers, unchanged by other code while scanning.
    # Ensures: first matching index, or -1 exactly when no match exists.
    index = 0
    while index < len(values):
        if values[index] == target:
            return index
        index += 1
    return -1


values = [4, 9, 2, 9]
print("first:", first_index(values, 9))
print("absent:", first_index(values, 7))
print("empty:", first_index([], 9))
print("input:", values)`,
    expected: 'first: 1\nabsent: -1\nempty: -1\ninput: [4, 9, 2, 9]',
  },
  assertions: {
    title: 'Use a checked invariant to localize a faulty transition',
    code: `def checked_scan(values, target, step=1):
    index = 0
    while True:
        assert 0 <= index <= len(values), "index bounds"
        assert all(value != target for value in values[:index]), "rejected prefix contains a match"
        if index == len(values):
            return -1
        if values[index] == target:
            return index
        index = min(index + step, len(values))


print("checked:", checked_scan([4, 9, 2, 9], 9))
try:
    checked_scan([4, 9, 2, 9], 9, step=2)
except AssertionError as error:
    print("fault:", error)`,
    expected: 'checked: 1\nfault: rejected prefix contains a match',
  },
  assignments: {
    title: 'Separate old values from sequential assignments',
    code: `left, right = 3, 8
left = right
right = left
print("sequential overwrite:", left, right)

left, right = 3, 8
saved_left = left
left = right
right = saved_left
print("saved old value:", left, right)

left, right = 3, 8
left, right = right, left
print("parallel assignment:", left, right)

# To ensure new_x >= 10 after x = x + 3, old_x must be >= 7.
for old_x in [6, 7]:
    x = old_x
    x = x + 3
    print("old, new, meets goal:", old_x, x, x >= 10)`,
    expected: 'sequential overwrite: 8 8\nsaved old value: 8 3\nparallel assignment: 8 3\nold, new, meets goal: 6 9 False\nold, new, meets goal: 7 10 True',
  },
  compaction: {
    title: 'Keep a stable logical prefix without allocating a result list',
    code: `def remove_stably(values, removed):
    write = 0
    for read in range(len(values)):
        if values[read] != removed:
            values[write] = values[read]
            write += 1
    return write


values = [5, 0, 5, 2, 0, 7]
count = remove_stably(values, 0)
print("kept count:", count)
print("logical result:", values[:count])
print("physical storage:", values)
print("empty count:", remove_stably([], 0))`,
    expected: 'kept count: 4\nlogical result: [5, 5, 2, 7]\nphysical storage: [5, 5, 2, 7, 0, 7]\nempty count: 0',
  },
  zeroFill: {
    title: 'Compose compaction with a second loop that finishes the contract',
    code: `def move_zeroes(values):
    write = 0
    for read in range(len(values)):
        if values[read] != 0:
            values[write] = values[read]
            write += 1
    # The first loop establishes the second loop's entry requirement.
    for fill in range(write, len(values)):
        values[fill] = 0


values = [0, 4, 0, -2, 4]
move_zeroes(values)
print("complete result:", values)
empty = []
move_zeroes(empty)
print("empty:", empty)`,
    expected: 'complete result: [4, -2, 4, 0, 0]\nempty: []',
  },
  partition: {
    title: 'Classify an unknown region while preserving every occurrence',
    code: `def partition_three(values):
    # Check before mutating: failed validation leaves the list unchanged.
    if any(type(value) is not int or value not in (0, 1, 2) for value in values):
        raise ValueError("Use integer categories 0, 1 and 2")
    low = middle = 0
    high = len(values)  # Exclusive boundary.
    while middle < high:
        if values[middle] == 0:
            values[low], values[middle] = values[middle], values[low]
            low += 1
            middle += 1
        elif values[middle] == 1:
            middle += 1
        else:
            high -= 1
            values[middle], values[high] = values[high], values[middle]
            # Do not advance middle: its incoming item is still unknown.


for original in ([1, 2, 0], [2, 0, 1, 2, 0, 1], []):
    values = list(original)
    partition_three(values)
    print("classified:", values)
invalid = [2, 3, 0]
try:
    partition_three(invalid)
except ValueError as error:
    print("invalid:", error, invalid)`,
    expected: 'classified: [0, 1, 2]\nclassified: [0, 0, 1, 1, 2, 2]\nclassified: []\ninvalid: Use integer categories 0, 1 and 2 [2, 3, 0]',
  },
  gcd: {
    title: 'Preserve common divisors while decreasing the second operand',
    code: `def euclid(first, second):
    if any(type(value) is not int or value < 0 for value in (first, second)):
        raise ValueError("Use nonnegative integers")
    a, b = first, second
    while b != 0:
        a, b = b, a % b
    return a


for pair in [(84, 30), (30, 84), (0, 12), (12, 0), (0, 0)]:
    print(pair, "->", euclid(*pair))
try:
    euclid(-1, 3)
except ValueError as error:
    print("invalid:", error)`,
    expected: '(84, 30) -> 6\n(30, 84) -> 6\n(0, 12) -> 12\n(12, 0) -> 12\n(0, 0) -> 0\ninvalid: Use nonnegative integers',
  },
  lexicographic: {
    title: 'Terminate even when a later coordinate resets upward',
    code: `def visit_rectangle(rows, columns):
    if any(type(value) is not int or value < 0 for value in (rows, columns)):
        raise ValueError("Use nonnegative integer dimensions")
    row = column = 0
    visited = []
    measures = [(rows - row, columns - column)]
    while row < rows:
        if column < columns:
            visited.append((row, column))
            column += 1
        else:
            row += 1
            column = 0
        measures.append((rows - row, columns - column))
    return visited, measures


visited, measures = visit_rectangle(2, 2)
print("visited:", visited)
print("measures:", measures)
print("strict lexicographic descent:", all(after < before for before, after in zip(measures, measures[1:])))
print("no columns:", visit_rectangle(2, 0)[0])`,
    expected: 'visited: [(0, 0), (0, 1), (1, 0), (1, 1)]\nmeasures: [(2, 2), (2, 1), (2, 0), (1, 2), (1, 1), (1, 0), (0, 2)]\nstrict lexicographic descent: True\nno columns: []',
  },
  powers: {
    title: 'Prove a recursive result and a conserved iterative product',
    code: `def recursive_power(base, exponent):
    if type(base) is not int or type(exponent) is not int or exponent < 0:
        raise ValueError("Use an integer base and a nonnegative integer exponent")
    if exponent == 0:
        return 1
    half = recursive_power(base, exponent // 2)
    squared = half * half
    return squared if exponent % 2 == 0 else base * squared


def iterative_power(base, exponent):
    if type(base) is not int or type(exponent) is not int or exponent < 0:
        raise ValueError("Use an integer base and a nonnegative integer exponent")
    result, factor, remaining = 1, base, exponent
    while remaining > 0:
        if remaining % 2 == 1:
            result *= factor
        factor *= factor
        remaining //= 2
    return result


for base, exponent in [(3, 5), (-2, 6), (0, 0), (4, 0)]:
    print((base, exponent), recursive_power(base, exponent), iterative_power(base, exponent))`,
    expected: '(3, 5) 243 243\n(-2, 6) 64 64\n(0, 0) 1 1\n(4, 0) 1 1',
  },
  modularPower: {
    title: 'Adapt the invariant to residues without constructing a huge power',
    code: `def modular_power(base, exponent, modulus):
    if (any(type(value) is not int for value in (base, exponent, modulus))
            or exponent < 0 or modulus <= 0):
        raise ValueError("Use integers with exponent >= 0 and modulus > 0")
    result, factor = 1 % modulus, base % modulus
    remaining = exponent
    while remaining:
        if remaining % 2:
            result = (result * factor) % modulus
        factor = (factor * factor) % modulus
        remaining //= 2
    return result


print("residue:", modular_power(3, 100, 7))
print("negative base:", modular_power(-2, 5, 11))
print("modulus one:", modular_power(9, 0, 1))
print("native reference:", modular_power(3, 100, 7) == pow(3, 100, 7))`,
    expected: 'residue: 4\nnegative base: 1\nmodulus one: 0\nnative reference: True',
  },
  certificate: {
    title: 'Check both sortedness and conservation of multiplicities',
    code: `from collections import Counter


def is_sorted_permutation(original, candidate):
    ordered = all(left <= right for left, right in zip(candidate, candidate[1:]))
    same_occurrences = Counter(candidate) == Counter(original)
    return ordered and same_occurrences


original = [2, 1, 2]
for candidate in ([1, 2, 2], [1, 2], [1, 1, 2], [2, 1, 2]):
    print(candidate, "valid:", is_sorted_permutation(original, candidate))
print("sets miss a lost copy:", set([1, 2]) == set(original))`,
    expected: '[1, 2, 2] valid: True\n[1, 2] valid: False\n[1, 1, 2] valid: False\n[2, 1, 2] valid: False\nsets miss a lost copy: True',
  },
  counterexample: {
    title: 'Find a small counterexample and state the limit of the search',
    code: `from itertools import product


def skips_positions(values, target):
    index = 0
    while index < len(values):
        if values[index] == target:
            return index
        index = min(index + 2, len(values))
    return -1


def find_counterexample():
    for length in range(5):
        for values in product((0, 1), repeat=length):
            expected = next((i for i, value in enumerate(values) if value == 1), -1)
            actual = skips_positions(values, 1)
            if actual != expected:
                return values, expected, actual
    return None


print("values, expected, actual:", find_counterexample())`,
    expected: 'values, expected, actual: ((0, 1), 1, -1)',
  },
  squareRoot: {
    title: 'Independent solution: preserve a feasible and an infeasible boundary',
    code: `def integer_square_root(number):
    if type(number) is not int or number < 0:
        raise ValueError("Use a nonnegative integer")
    low, high = 0, number + 1
    # Invariant: 0 <= low < high and low*low <= number < high*high.
    while high - low > 1:
        middle = (low + high) // 2
        if middle * middle <= number:
            low = middle
        else:
            high = middle
    return low


for number in [0, 1, 2, 15, 16, 17, 10**20]:
    result = integer_square_root(number)
    print(number, "->", result, "contract:", result * result <= number < (result + 1) * (result + 1))`,
    expected: '0 -> 0 contract: True\n1 -> 1 contract: True\n2 -> 1 contract: True\n15 -> 3 contract: True\n16 -> 4 contract: True\n17 -> 4 contract: True\n100000000000000000000 -> 10000000000 contract: True',
  },
};
