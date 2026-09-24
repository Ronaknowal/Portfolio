export const dynamicProgrammingExamples = {
  memo: {
    title: 'Solve each free suffix once',
    code: `from functools import cache

def best_reward(rewards):
    values = tuple(rewards)  # Keep this problem fixed for this cache.

    @cache
    def solve(index):
        if index >= len(values):
            return 0
        skip = solve(index + 1)
        take = values[index] + solve(index + 2)
        return max(skip, take)

    return solve(0)

print(best_reward([4, 7, 2, 9]))
print(best_reward([-5, -2]))
print(best_reward([]))`,
    expected: '16\n0\n0',
  },
  rewardWitness: {
    title: 'Fill suffixes backward, then recover selected sessions',
    code: `def reward_plan(rewards):
    count = len(rewards)
    best = [0] * (count + 2)
    for index in range(count - 1, -1, -1):
        best[index] = max(best[index + 1], rewards[index] + best[index + 2])

    chosen = []
    index = 0
    while index < count:
        # Skip on ties: a deterministic policy, not earliest-index selection.
        if best[index + 1] >= rewards[index] + best[index + 2]:
            index += 1
        else:
            chosen.append(index)
            index += 2
    return best[0], chosen

print(reward_plan([4, 7, 2, 9]))
print(reward_plan([2, 2]))
print(reward_plan([]))`,
    expected: '(16, [1, 3])\n(2, [1])\n(0, [])',
  },
  state: {
    title: 'Keep the previous-choice constraint in the key',
    code: `from functools import cache

def constrained_reward(rewards, initially_blocked=False):
    values = tuple(rewards)

    @cache
    def solve(index, blocked):
        if index == len(values):
            return 0
        skip = solve(index + 1, False)
        if blocked:
            return skip
        return max(skip, values[index] + solve(index + 1, True))

    return solve(0, initially_blocked)

print(constrained_reward([9], False))
print(constrained_reward([9], True))
print(constrained_reward([4, 7, 2, 9]))`,
    expected: '9\n0\n16',
  },
  weightedIntervals: {
    title: 'Recover the weighted schedule that earliest finish can miss',
    code: `from bisect import bisect_right

def weighted_schedule(jobs):
    # (start, finish, value); original input indices identify jobs.
    if any(start >= finish for start, finish, value in jobs):
        raise ValueError("Jobs must have positive duration")
    ordered = sorted(enumerate(jobs), key=lambda entry: (entry[1][1], entry[0]))
    finishes = [job[1] for original, job in ordered]
    best = [0] * (len(jobs) + 1)
    compatible = []
    for index, (original, (start, finish, value)) in enumerate(ordered):
        prefix = bisect_right(finishes, start, 0, index)
        compatible.append(prefix)
        best[index + 1] = max(best[index], value + best[prefix])

    chosen = []
    count = len(jobs)
    while count:
        original, (start, finish, value) = ordered[count - 1]
        if best[count] == best[count - 1]:
            count -= 1  # Skip ties; preserve earlier finish-order solution.
        else:
            chosen.append(original)
            count = compatible[count - 1]
    chosen.reverse()
    return best[-1], chosen

print(weighted_schedule([(0, 5, 10), (0, 2, 4), (2, 5, 4)]))
print(weighted_schedule([(0, 2, 4), (2, 5, 7), (0, 5, 10)]))
print(weighted_schedule([]))`,
    expected: '(10, [0])\n(11, [0, 1])\n(0, [])',
  },
  grid: {
    title: 'Find a cheapest right/down route and a concrete witness',
    code: `def grid_plan(grid):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("Use a nonempty rectangular grid")
    rows, columns = len(grid), len(grid[0])
    best = [[None] * columns for _ in range(rows)]
    parent = [[None] * columns for _ in range(rows)]
    for row in range(rows):
        for column in range(columns):
            if grid[row][column] is None:  # Blocked cell.
                continue
            if row == column == 0:
                best[row][column] = grid[row][column]
                continue
            candidates = []
            for prior_row, prior_column in [(row - 1, column), (row, column - 1)]:
                if prior_row >= 0 and prior_column >= 0:
                    cost = best[prior_row][prior_column]
                    if cost is not None:
                        candidates.append((cost, prior_row, prior_column))
            if candidates:
                # Cost first, then upper predecessor on ties.
                cost, prior_row, prior_column = min(candidates)
                best[row][column] = cost + grid[row][column]
                parent[row][column] = (prior_row, prior_column)

    if best[-1][-1] is None:
        return None, []
    path = []
    cell = (rows - 1, columns - 1)
    while cell is not None:
        path.append(cell)
        cell = parent[cell[0]][cell[1]]
    path.reverse()
    return best[-1][-1], path

grid = [[1, 6, 2, 1], [2, 1, 5, 2], [4, 1, 1, 1]]
print(grid_plan(grid))
print(grid_plan([[0, None], [None, 0]]))
print(grid_plan([[2, -5], [3, 1]]))`,
    expected: '(7, [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (2, 3)])\n(None, [])\n(-2, [(0, 0), (0, 1), (1, 1)])',
  },
  gridCount: {
    title: 'Reuse one row while counting blocked-grid paths',
    code: `def count_paths(blocked):
    if not blocked or not blocked[0] or any(len(row) != len(blocked[0]) for row in blocked):
        raise ValueError("Use a nonempty rectangular grid")
    ways = [0] * len(blocked[0])
    ways[0] = 1
    for row in blocked:
        for column, is_blocked in enumerate(row):
            if is_blocked:
                ways[column] = 0
            elif column:
                # ways[column]: above, old row. ways[column-1]: left, new row.
                ways[column] += ways[column - 1]
    return ways[-1]

print(count_paths([[False, False, False], [False, True, False], [False, False, False]]))
print(count_paths([[True]]))
print(count_paths([[False]]))`,
    expected: '2\n0\n1',
  },
  lcs: {
    title: 'Recover a longest common subsequence from prefix lengths',
    code: `def longest_common_subsequence(first, second):
    rows, columns = len(first), len(second)
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if first[row - 1] == second[column - 1]:
                lengths[row][column] = 1 + lengths[row - 1][column - 1]
            else:
                lengths[row][column] = max(lengths[row - 1][column], lengths[row][column - 1])

    result = []
    row, column = rows, columns
    while row and column:
        if first[row - 1] == second[column - 1]:
            result.append(first[row - 1])
            row -= 1
            column -= 1
        elif lengths[row - 1][column] >= lengths[row][column - 1]:
            row -= 1  # Prefer up on ties; no lexicographic guarantee.
        else:
            column -= 1
    return lengths[rows][columns], "".join(reversed(result))

print(longest_common_subsequence("CABAC", "ABC"))
print(longest_common_subsequence("AB", "BA"))
print(longest_common_subsequence("", "ABC"))`,
    expected: "(3, 'ABC')\n(1, 'A')\n(0, '')",
  },
  edit: {
    title: 'Change the sequence objective to unit-cost edits',
    code: `def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for row, first_character in enumerate(first, start=1):
        current = [row] + [0] * len(second)
        for column, second_character in enumerate(second, start=1):
            replace_or_match = previous[column - 1] + (first_character != second_character)
            delete = previous[column] + 1
            insert = current[column - 1] + 1
            current[column] = min(replace_or_match, delete, insert)
        previous = current
    return previous[-1]

print(edit_distance("CAT", "CUT"))
print(edit_distance("", "ABC"))
print(edit_distance("kitten", "sitting"))`,
    expected: '1\n3\n3',
  },
  knapsack: {
    title: 'Allocate a capacity budget and reconstruct the selected items',
    code: `def knapsack_plan(items, capacity):
    if not isinstance(capacity, int) or capacity < 0:
        raise ValueError("Capacity must be a nonnegative integer")
    if any(not isinstance(weight, int) or weight <= 0 for weight, value in items):
        raise ValueError("Weights must be positive integers")
    best = [[0] * (capacity + 1) for _ in range(len(items) + 1)]
    for count, (weight, value) in enumerate(items, start=1):
        for budget in range(capacity + 1):
            best[count][budget] = best[count - 1][budget]
            if weight <= budget:
                best[count][budget] = max(best[count][budget], value + best[count - 1][budget - weight])

    selected = []
    budget = capacity
    for count in range(len(items), 0, -1):
        if best[count][budget] != best[count - 1][budget]:
            selected.append(count - 1)
            budget -= items[count - 1][0]
    selected.reverse()
    return best[-1][capacity], selected

print(knapsack_plan([(2, 3), (3, 4)], 6))
print(knapsack_plan([(2, 3), (2, 3)], 2))
print(knapsack_plan([(2, -3)], 3))
print(knapsack_plan([(10, 60), (20, 100), (30, 120)], 50))`,
    expected: '(7, [0, 1])\n(3, [0])\n(0, [])\n(220, [1, 2])',
  },
  compression: {
    title: 'The direction chooses whether the current item can be reused',
    code: `def capacity_value(items, capacity, reusable=False):
    if capacity < 0 or any(weight <= 0 for weight, value in items):
        raise ValueError("Use nonnegative capacity and positive integer weights")
    best = [0] * (capacity + 1)
    for weight, value in items:
        budgets = range(weight, capacity + 1) if reusable else range(capacity, weight - 1, -1)
        for budget in budgets:
            best[budget] = max(best[budget], value + best[budget - weight])
    return best[capacity]

items = [(2, 3), (3, 4)]
print("once:", capacity_value(items, 6))
print("reusable:", capacity_value(items, 6, True))
print("one item, once/reusable:", capacity_value([(2, 3)], 4), capacity_value([(2, 3)], 4, True))`,
    expected: 'once: 7\nreusable: 9\none item, once/reusable: 3 6',
  },
  subsetSum: {
    title: 'Exact reachability needs a different initial state',
    code: `def subset_sum(values, target):
    if target < 0 or any(value < 0 for value in values):
        raise ValueError("Use nonnegative integer values and target")
    reachable = [False] * (target + 1)
    reachable[0] = True
    for value in values:
        for total in range(target, value - 1, -1):
            reachable[total] = reachable[total] or reachable[total - value]
    return reachable[target]

print(subset_sum([3, 5], 6))
print(subset_sum([3, 5, 3], 6))
print(subset_sum([], 0))`,
    expected: 'False\nTrue\nTrue',
  },
  coins: {
    title: 'Fewest coins and two different meanings of counting',
    code: `def coin_answers(coins, amount):
    if amount < 0 or any(coin <= 0 for coin in coins):
        raise ValueError("Use a nonnegative integer amount and positive denominations")
    denominations = sorted(set(coins))
    minimum = [None] * (amount + 1)
    ordered = [0] * (amount + 1)
    minimum[0], ordered[0] = 0, 1
    for total in range(1, amount + 1):
        for coin in denominations:
            if coin <= total:
                ordered[total] += ordered[total - coin]
                if minimum[total - coin] is not None:
                    candidate = 1 + minimum[total - coin]
                    if minimum[total] is None or candidate < minimum[total]:
                        minimum[total] = candidate

    combinations = [0] * (amount + 1)
    combinations[0] = 1
    for coin in denominations:
        for total in range(coin, amount + 1):
            combinations[total] += combinations[total - coin]
    return minimum[amount], ordered[amount], combinations[amount]

print(coin_answers([1, 3], 4))
print(coin_answers([2], 3))
print(coin_answers([], 0))`,
    expected: '(2, 3, 2)\n(None, 0, 0)\n(0, 1, 1)',
  },
  masks: {
    title: 'A finite set represented by membership bits',
    code: `names = ["A", "B", "C", "D"]
full = (1 << len(names)) - 1
mask = (1 << 0) | (1 << 2)  # A and C.
print(format(mask, "04b"), [name for index, name in enumerate(names) if mask & (1 << index)])
print("B present:", bool(mask & (1 << 1)))
mask |= 1 << 1  # Add B; adding twice has the same effect.
mask &= ~(1 << 2)  # Remove C, even if C was already absent.
print(format(mask, "04b"), format(full & ~mask, "04b"))
print("bitwise/logical:", 5 & 2, 5 and 2)
print("unbounded complement:", ~5)

subset = 5
subsets = []
while True:
    subsets.append(subset)
    if subset == 0:
        break
    subset = (subset - 1) & 5
print("submasks:", subsets)`,
    expected: "0101 ['A', 'C']\nB present: False\n0011 1100\nbitwise/logical: 0 2\nunbounded complement: -6\nsubmasks: [5, 4, 1, 0]",
  },
  route: {
    title: 'Cheapest visit-once path: subset plus endpoint',
    code: `def visit_once(costs):
    count = len(costs)
    if not count or any(len(row) != count for row in costs):
        raise ValueError("Use a nonempty square cost matrix; None means no edge")
    full = (1 << count) - 1
    best = [[None] * count for _ in range(full + 1)]
    parent = [[None] * count for _ in range(full + 1)]
    best[1][0] = 0  # Start at vertex 0; no edge traveled.
    for mask in range(1, full + 1):
        for endpoint in range(count):
            if best[mask][endpoint] is None:
                continue
            for next_vertex in range(count):
                edge = costs[endpoint][next_vertex]
                if mask & (1 << next_vertex) or edge is None:
                    continue
                next_mask = mask | (1 << next_vertex)
                candidate = best[mask][endpoint] + edge
                if best[next_mask][next_vertex] is None or candidate < best[next_mask][next_vertex]:
                    best[next_mask][next_vertex] = candidate
                    parent[next_mask][next_vertex] = endpoint

    candidates = [(cost, endpoint) for endpoint, cost in enumerate(best[full]) if cost is not None]
    if not candidates:
        return None, []
    cost, endpoint = min(candidates)  # Smaller endpoint on final ties.
    path = []
    mask = full
    while endpoint is not None:
        path.append(endpoint)
        previous = parent[mask][endpoint]
        mask &= ~(1 << endpoint)
        endpoint = previous
    return cost, path[::-1]

costs = [[0, 1, 1, 8], [1, 0, 1, 9], [1, 1, 0, 1], [8, 9, 1, 0]]
print(visit_once(costs))
print(visit_once([[0, None], [None, 0]]))
print(visit_once([[0]]))`,
    expected: '(3, [0, 1, 2, 3])\n(None, [])\n(0, [0])',
  },
  lis: {
    title: 'A quadratic state definition and a smaller dominance summary',
    code: `from bisect import bisect_left

def lis_quadratic(values):
    ending = [1] * len(values)
    for index, value in enumerate(values):
        for previous in range(index):
            if values[previous] < value:
                ending[index] = max(ending[index], ending[previous] + 1)
    return max(ending, default=0)

def lis_length(values):
    tails = []
    for value in values:
        position = bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)

for values in [[3, 5, 6, 2], [2, 2, 2], [], [4, 1, 3, 2, 5]]:
    print(lis_quadratic(values), lis_length(values))`,
    expected: '3 3\n1 1\n0 0\n3 3',
  },
  segment: {
    title: 'Independent application: split a stream into approved commands',
    code: `def command_plan(text, words):
    vocabulary = set(words)
    if "" in vocabulary:
        raise ValueError("Commands must be nonempty")
    previous = [None] * (len(text) + 1)
    previous[0] = 0
    for end in range(1, len(text) + 1):
        for start in range(end):
            if previous[start] is not None and text[start:end] in vocabulary:
                previous[end] = start
                break
    if previous[-1] is None:
        return None
    result = []
    end = len(text)
    while end:
        start = previous[end]
        result.append(text[start:end])
        end = start
    return result[::-1]

print(command_plan("scanreset", ["scan", "reset", "sc", "an"]))
print(command_plan("scanrest", ["scan", "reset"]))
print(command_plan("", ["scan"]))`,
    expected: "['scan', 'reset']\nNone\n[]",
  },
};
