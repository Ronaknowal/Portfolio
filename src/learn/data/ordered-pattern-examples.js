export const orderedPatternExamples = {
  boundaries: {
    title: 'One boundary invariant answers several sorted-array questions',
    code: `from bisect import bisect_left, bisect_right

def lower_bound(values, target):
    low, high = 0, len(values)
    while low < high:
        middle = low + (high - low) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low

def upper_bound(values, target):
    low, high = 0, len(values)
    while low < high:
        middle = low + (high - low) // 2
        if values[middle] <= target:
            low = middle + 1
        else:
            high = middle
    return low

values = [2, 4, 4, 4, 7, 9]
for target in [4, 5, 10]:
    first = lower_bound(values, target)
    after = upper_bound(values, target)
    match = first if first < len(values) and values[first] == target else -1
    print(target, "boundary:", first, after, "first match:", match,
          "count:", after - first)
    assert (first, after) == (bisect_left(values, target), bisect_right(values, target))
print("empty:", lower_bound([], 4))`,
    expected: `4 boundary: 1 4 first match: 1 count: 3
5 boundary: 4 4 first match: -1 count: 0
10 boundary: 6 6 first match: -1 count: 0
empty: 0`,
  },
  recordOrder: {
    title: 'Sort by a declared key while retaining record identity',
    code: `from bisect import bisect_left

records = [("Ada", 4), ("Bo", 2), ("Cy", 4), ("Di", 1)]
ordered = sorted(records, key=lambda record: record[1])
print("by score:", ordered)
print("original:", records)
print("first score >= 3:", ordered[bisect_left(ordered, 3, key=lambda row: row[1])])
names_first = sorted(records, key=lambda row: row[0])
print("score descending, name ascending:", sorted(names_first, key=lambda row: row[1], reverse=True))
print("sort return:", records.sort(key=lambda row: row[1]))`,
    expected: `by score: [('Di', 1), ('Bo', 2), ('Ada', 4), ('Cy', 4)]
original: [('Ada', 4), ('Bo', 2), ('Cy', 4), ('Di', 1)]
first score >= 3: ('Ada', 4)
score descending, name ascending: [('Ada', 4), ('Cy', 4), ('Bo', 2), ('Di', 1)]
sort return: None`,
  },
  stableSorting: {
    title: 'Grow a sorted prefix, or merge sorted runs',
    code: `def insertion_sort(records, key=lambda item: item):
    result = list(records)
    for position in range(1, len(result)):
        saved = result[position]
        gap = position
        while gap > 0 and key(saved) < key(result[gap - 1]):
            result[gap] = result[gap - 1]
            gap -= 1
        result[gap] = saved
    return result

def merge_sort(records, key=lambda item: item):
    result = list(records)
    size = len(result)
    buffer = [None] * size
    width = 1
    while width < size:
        for start in range(0, size, 2 * width):
            middle = min(start + width, size)
            stop = min(start + 2 * width, size)
            left, right = start, middle
            for output in range(start, stop):
                take_left = right == stop or (
                    left < middle and not key(result[right]) < key(result[left])
                )
                if take_left:
                    buffer[output] = result[left]
                    left += 1
                else:
                    buffer[output] = result[right]
                    right += 1
        result, buffer = buffer, result
        width *= 2
    return result

records = [(4, "A"), (2, "B"), (4, "C"), (1, "D"), (4, "E")]
for algorithm in [insertion_sort, merge_sort]:
    answer = algorithm(records, key=lambda record: record[0])
    print(algorithm.__name__ + ":", answer)
    assert answer == sorted(records, key=lambda record: record[0])
print("empty merge:", merge_sort([]))`,
    expected: `insertion_sort: [(1, 'D'), (2, 'B'), (4, 'A'), (4, 'C'), (4, 'E')]
merge_sort: [(1, 'D'), (2, 'B'), (4, 'A'), (4, 'C'), (4, 'E')]
empty merge: []`,
  },
  partitionSort: {
    title: 'Three-way partitioning keeps equal values out of recursive work',
    code: `from random import Random

def quicksort_three_way(values, seed=7):
    result = list(values)
    random = Random(seed)

    def sort(low, high):
        if high - low < 2:
            return
        pivot = result[random.randrange(low, high)]
        less, scan, greater = low, low, high
        while scan < greater:
            if result[scan] < pivot:
                result[less], result[scan] = result[scan], result[less]
                less += 1
                scan += 1
            elif result[scan] > pivot:
                greater -= 1
                result[scan], result[greater] = result[greater], result[scan]
            else:
                scan += 1
        sort(low, less)
        sort(greater, high)

    sort(0, len(result))
    return result

values = [4, 1, 4, 6, 2, 4, 1]
print("sorted:", quicksort_three_way(values))
print("all equal:", quicksort_three_way([4] * 6))
print("original:", values)`,
    expected: `sorted: [1, 1, 2, 4, 4, 4, 6]
all equal: [4, 4, 4, 4, 4, 4]
original: [4, 1, 4, 6, 2, 4, 1]`,
  },
  heapSort: {
    title: 'Reuse a max-heap to sort with constant auxiliary storage',
    code: `def heapsort_in_place(values):
    def sink(root, stop):
        while 2 * root + 1 < stop:
            child = 2 * root + 1
            if child + 1 < stop and values[child] < values[child + 1]:
                child += 1
            if values[root] >= values[child]:
                break
            values[root], values[child] = values[child], values[root]
            root = child

    for root in range(len(values) // 2 - 1, -1, -1):
        sink(root, len(values))
    for stop in range(len(values) - 1, 0, -1):
        values[0], values[stop] = values[stop], values[0]
        sink(0, stop)

values = [4, -2, 4, 1, 0]
heapsort_in_place(values)
print(values)`,
    expected: `[-2, 0, 1, 4, 4]`,
  },
  countingSort: {
    title: 'A small integer domain permits counting instead of comparing pairs',
    code: `def counting_sort(values):
    if not values:
        return []
    if any(type(value) is not int for value in values):
        raise ValueError("Use integer values")
    smallest, largest = min(values), max(values)
    width = largest - smallest + 1
    if width > 100_000:
        raise ValueError("Teaching example limits the integer range")
    counts = [0] * width
    for value in values:
        counts[value - smallest] += 1
    result = []
    for offset, count in enumerate(counts):
        result.extend([smallest + offset] * count)
    return result

print(counting_sort([3, -1, 0, -1, 3, 2]))
print(counting_sort([]))`,
    expected: `[-1, -1, 0, 2, 3, 3]
[]`,
  },
  pairSum: {
    title: 'Discard an endpoint only after ruling out all its remaining partners',
    code: `def sorted_pair(values, target):
    left, right = 0, len(values) - 1
    while left < right:
        total = values[left] + values[right]
        if total == target:
            return left, right
        if total < target:
            left += 1
        else:
            right -= 1
    return None

values = [1, 2, 4, 5, 7, 9]
for target in [11, 3, 20]:
    pair = sorted_pair(values, target)
    print(target, pair, None if pair is None else [values[index] for index in pair])
print("one element:", sorted_pair([5], 10))
print("two equal elements:", sorted_pair([5, 5], 10))`,
    expected: `11 (1, 5) [2, 9]
3 (0, 1) [1, 2]
20 None None
one element: None
two equal elements: (0, 1)`,
  },
  compactMerge: {
    title: 'Read/write pointers retain a useful prefix; a tail merge protects unread data',
    code: `def unique_prefix(values):
    write = 0
    for read in range(len(values)):
        if write == 0 or values[read] != values[write - 1]:
            values[write] = values[read]
            write += 1
    return write

def merge_into_tail(first, count, second):
    if len(first) != count + len(second) or not 0 <= count <= len(first):
        raise ValueError("Provide exactly enough spare slots")
    left, right, write = count - 1, len(second) - 1, len(first) - 1
    while right >= 0:
        if left >= 0 and first[left] > second[right]:
            first[write] = first[left]
            left -= 1
        else:
            first[write] = second[right]
            right -= 1
        write -= 1

values = [1, 1, 2, 2, 2, 5]
length = unique_prefix(values)
print("retained:", values[:length], "length:", length)
first = [1, 4, 6, None, None, None]
merge_into_tail(first, 3, [2, 4, 7])
print("merged:", first)`,
    expected: `retained: [1, 2, 5] length: 3
merged: [1, 2, 4, 4, 6, 7]`,
  },
  triples: {
    title: 'Fix one value, then solve a smaller two-pointer problem',
    code: `def zero_triples(values):
    ordered = sorted(values)
    result = []
    for fixed in range(len(ordered) - 2):
        if fixed > 0 and ordered[fixed] == ordered[fixed - 1]:
            continue
        left, right = fixed + 1, len(ordered) - 1
        while left < right:
            total = ordered[fixed] + ordered[left] + ordered[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append((ordered[fixed], ordered[left], ordered[right]))
                left += 1
                right -= 1
                while left < right and ordered[left] == ordered[left - 1]:
                    left += 1
                while left < right and ordered[right] == ordered[right + 1]:
                    right -= 1
    return result

print(zero_triples([-3, 1, 2, -1, 0, 1, 2]))
print(zero_triples([0, 0, 0, 0]))`,
    expected: `[(-3, 1, 2), (-1, 0, 1)]
[(0, 0, 0)]`,
  },
  intervals: {
    title: 'Merge closed intervals while preserving their complete union',
    code: `def merge_closed(intervals):
    if any(len(interval) != 2 or interval[0] > interval[1] for interval in intervals):
        raise ValueError("Use closed intervals with start <= end")
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

windows = [(6, 8), (1, 4), (2, 3), (4, 5), (10, 10)]
print(merge_closed(windows))
print("original:", windows)
print("empty:", merge_closed([]))`,
    expected: `[[1, 5], [6, 8], [10, 10]]
original: [(6, 8), (1, 4), (2, 3), (4, 5), (10, 10)]
empty: []`,
  },
  fixedWindow: {
    title: 'Move a fixed-width sum without re-adding the overlapping values',
    code: `def maximum_fixed_sum(values, width):
    if not 1 <= width <= len(values):
        raise ValueError("Width must be between 1 and the array length")
    total = 0
    for index in range(width):
        total += values[index]
    best, best_start = total, 0
    for right in range(width, len(values)):
        total += values[right] - values[right - width]
        if total > best:
            best, best_start = total, right - width + 1
    return best, best_start

values = [3, -4, 5, 2, -1, 4]
total, start = maximum_fixed_sum(values, 3)
print("best sum:", total, "range:", (start, start + 3))
print("values:", values[start:start + 3], "average:", total / 3)
print("all negative:", maximum_fixed_sum([-5, -2, -8], 2))`,
    expected: `best sum: 6 range: (2, 5)
values: [5, 2, -1] average: 2.0
all negative: (-7, 0)`,
  },
  positiveWindow: {
    title: 'Find a shortest qualifying window under the nonnegative contract',
    code: `def shortest_at_least(values, target):
    if target <= 0 or any(value < 0 for value in values):
        raise ValueError("Use a positive target and nonnegative values")
    left, total = 0, 0
    best = None
    for right, value in enumerate(values):
        total += value
        while total >= target:
            stop = right + 1
            if best is None or stop - left < best[1] - best[0]:
                best = (left, stop)
            total -= values[left]
            left += 1
    return best

for values, target in [([2, 1, 3, 2, 4], 6), ([0, 0, 5], 5), ([1, 2], 9), ([], 1)]:
    answer = shortest_at_least(values, target)
    print(values, "->", answer, "length:", 0 if answer is None else answer[1] - answer[0])`,
    expected: `[2, 1, 3, 2, 4] -> (3, 5) length: 2
[0, 0, 5] -> (2, 3) length: 1
[1, 2] -> None length: 0
[] -> None length: 0`,
  },
  distinctWindow: {
    title: 'A uniqueness window moves past the last conflicting position',
    code: `def longest_unique(text):
    last_seen = {}
    left = 0
    best = (0, 0)
    for right, character in enumerate(text):
        left = max(left, last_seen.get(character, -1) + 1)
        last_seen[character] = right
        if right + 1 - left > best[1] - best[0]:
            best = (left, right + 1)
    return best

for text in ["abcaefb", "abba", ""]:
    start, stop = longest_unique(text)
    print(repr(text), "->", repr(text[start:stop]), "length:", stop - start)`,
    expected: `'abcaefb' -> 'bcaef' length: 5
'abba' -> 'ab' length: 2
'' -> '' length: 0`,
  },
  prefixSuffix: {
    title: 'Cancel a common prefix and retain a suffix summary when subtraction is unavailable',
    code: `def prefix_sums(values):
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return prefix

def outside_products(values):
    result = [1] * len(values)
    product = 1
    for index, value in enumerate(values):
        result[index] = product
        product *= value
    product = 1
    for index in range(len(values) - 1, -1, -1):
        result[index] *= product
        product *= values[index]
    return result

values = [4, -2, 3, 1]
prefix = prefix_sums(values)
print("prefix:", prefix)
print("sum [1,3):", prefix[3] - prefix[1])
print("empty [2,2):", prefix[2] - prefix[2])
print("outside products:", outside_products([2, 0, 3, 4]))`,
    expected: `prefix: [0, 4, 2, 5, 6]
sum [1,3): 1
empty [2,2): 0
outside products: [0, 24, 0, 0]`,
  },
  signedCounts: {
    title: 'Count nonempty signed segments using earlier prefix frequencies',
    code: `def count_sum(values, target):
    frequencies = {0: 1}
    prefix = 0
    count = 0
    for value in values:
        prefix += value
        count += frequencies.get(prefix - target, 0)
        frequencies[prefix] = frequencies.get(prefix, 0) + 1
    return count

for values, target in [([2, -1, 2, -1], 1), ([0, 0, 0], 0), ([], 0)]:
    print(values, "target", target, "count:", count_sum(values, target))`,
    expected: `[2, -1, 2, -1] target 1 count: 3
[0, 0, 0] target 0 count: 6
[] target 0 count: 0`,
  },
  answerSearch: {
    title: 'Search an integer rate using a proven monotone work budget',
    code: `def minimum_rate(jobs, budget):
    if not jobs or any(type(job) is not int or job <= 0 for job in jobs):
        raise ValueError("Use at least one positive integer job")
    if type(budget) is not int or budget < 0:
        raise ValueError("Use a nonnegative integer budget")
    if budget < len(jobs):
        return None

    def feasible(speed):
        slots = sum((job + speed - 1) // speed for job in jobs)
        return slots <= budget

    low, high = 1, max(jobs)
    while low < high:
        middle = low + (high - low) // 2
        if feasible(middle):
            high = middle
        else:
            low = middle + 1
    return low

jobs = [3, 6, 7]
for budget in [6, 4, 3, 2]:
    print("budget:", budget, "minimum rate:", minimum_rate(jobs, budget))`,
    expected: `budget: 6 minimum rate: 3
budget: 4 minimum rate: 6
budget: 3 minimum rate: 7
budget: 2 minimum rate: None`,
  },
  rangeReport: {
    title: 'Independent synthesis: time-range lookup with signed totals',
    code: `from bisect import bisect_left

def prepare_report(records):
    ordered = sorted(records, key=lambda row: row[0])
    times = [row[0] for row in ordered]
    prefix = [0]
    for _, amount in ordered:
        prefix.append(prefix[-1] + amount)
    return times, prefix

def query_report(times, prefix, start, stop):
    if start > stop:
        raise ValueError("Use start <= stop")
    left = bisect_left(times, start)
    right = bisect_left(times, stop)
    return right - left, prefix[right] - prefix[left]

records = [(5, 4), (2, 7), (5, -1), (9, 3), (1, -2)]
times, prefix = prepare_report(records)
for start, stop in [(2, 9), (5, 6), (5, 5), (10, 12)]:
    print((start, stop), "count and total:", query_report(times, prefix, start, stop))`,
    expected: `(2, 9) count and total: (3, 10)
(5, 6) count and total: (2, 3)
(5, 5) count and total: (0, 0)
(10, 12) count and total: (0, 0)`,
  },
};
