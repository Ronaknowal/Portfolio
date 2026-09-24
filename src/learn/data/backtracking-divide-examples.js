export const backtrackingDivideExamples = {
  subsets: {
    title: 'Enumerate subsets without sharing the mutable answer buffer',
    code: `def subsets(values):
    answers, path = [], []

    def search(index):
        if index == len(values):
            answers.append(path.copy())
            return
        path.append(values[index])
        search(index + 1)
        path.pop()
        search(index + 1)

    search(0)
    return answers

print(subsets(["red", "blue"]))
print(subsets([]))
answers = subsets([1, 2])
answers[0].append(99)
print("other answer unchanged:", answers[1])
`,
    expected: "[['red', 'blue'], ['red'], ['blue'], []]\n[[]]\nother answer unchanged: [1]",
  },
  targetSubsets: {
    title: 'Prune only when every completion is impossible',
    code: `def target_subsets(values, target, prune=True):
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("values must be positive integers")
    if type(target) is not int or target < 0:
        raise ValueError("target must be a nonnegative integer")
    suffix = [0] * (len(values) + 1)
    for index in range(len(values) - 1, -1, -1):
        suffix[index] = values[index] + suffix[index + 1]
    answers, path = [], []
    entered = 0

    def search(index, total):
        nonlocal entered
        entered += 1
        if index == len(values):
            if total == target:
                answers.append(path.copy())  # positions, not deduplicated values
            return
        if prune and (total > target or total + suffix[index] < target):
            return
        path.append(index)
        search(index + 1, total + values[index])
        path.pop()
        search(index + 1, total)

    search(0, 0)
    return answers, entered

for pruning in [False, True]:
    answers, calls = target_subsets([2, 4, 5], 5, pruning)
    print("pruning:", pruning, "positions:", answers, "calls:", calls)
print("repeated values:", target_subsets([2, 2], 2)[0])
print("empty target:", target_subsets([], 0)[0])
`,
    expected: 'pruning: False positions: [[2]] calls: 15\npruning: True positions: [[2]] calls: 13\nrepeated values: [[0], [1]]\nempty target: [[]]',
  },
  permutations: {
    title: 'Unique permutations use remaining multiplicities',
    code: `from collections import Counter

def unique_permutations(values):
    remaining = Counter(values)
    choices = sorted(remaining)  # contract: orderable, hashable values
    path, answers = [], []

    def search():
        if len(path) == len(values):
            answers.append(path.copy())
            return
        for value in choices:
            if remaining[value] == 0:
                continue
            remaining[value] -= 1
            path.append(value)
            search()
            path.pop()
            remaining[value] += 1

    search()
    return answers

print(unique_permutations([2, 1, 1]))
print(unique_permutations([]))
print("distinct count:", len(unique_permutations([1, 2, 3])))
`,
    expected: '[[1, 1, 2], [1, 2, 1], [2, 1, 1]]\n[[]]\ndistinct count: 6',
  },
  combinations: {
    title: 'Allow reuse while choosing each combination once',
    code: `def combination_sum(candidates, target):
    if any(type(value) is not int or value <= 0 for value in candidates):
        raise ValueError("candidates must be positive integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be distinct")
    if type(target) is not int or target < 0:
        raise ValueError("target must be nonnegative")
    candidates = sorted(candidates)
    path, answers = [], []

    def search(start, remaining):
        if remaining == 0:
            answers.append(path.copy())
            return
        for index in range(start, len(candidates)):
            value = candidates[index]
            if value > remaining:
                break
            path.append(value)
            search(index, remaining - value)  # same index permits reuse
            path.pop()

    search(0, target)
    return answers

print(combination_sum([2, 3, 5], 8))
print(combination_sum([4, 6], 5))
print(combination_sum([], 0))
try:
    combination_sum([0, 2], 4)
except ValueError as error:
    print(error)
`,
    expected: '[[2, 2, 2, 2], [2, 3, 3], [3, 5]]\n[]\n[[]]\ncandidates must be positive integers',
  },
  queens: {
    title: 'Solve N-Queens with three restored constraint sets',
    code: `def queens_solutions(n):
    if type(n) is not int or n < 0:
        raise ValueError("n must be nonnegative")
    columns, descending, ascending = set(), set(), set()
    path, answers = [], []

    def search(row):
        if row == n:
            answers.append(path.copy())
            return
        for column in range(n):
            if column in columns or row - column in descending or row + column in ascending:
                continue
            path.append(column)
            columns.add(column)
            descending.add(row - column)
            ascending.add(row + column)
            search(row + 1)
            ascending.remove(row + column)
            descending.remove(row - column)
            columns.remove(column)
            path.pop()

    search(0)
    return answers

print("4 queens:", queens_solutions(4))
print("2 queens:", queens_solutions(2))
print("5 queens count:", len(queens_solutions(5)))
print("empty placement:", queens_solutions(0))
`,
    expected: '4 queens: [[1, 3, 0, 2], [2, 0, 3, 1]]\n2 queens: []\n5 queens count: 10\nempty placement: [[]]',
  },
  wordSearch: {
    title: 'A word path must restore used cells, even after success',
    code: `def find_word(board, word):
    rows = len(board)
    columns = len(board[0]) if rows else 0
    if any(len(row) != columns for row in board):
        raise ValueError("board must be rectangular")
    if not word:
        return []  # empty word has an empty witness
    used, path = set(), []

    def search(row, column, index):
        if not (0 <= row < rows and 0 <= column < columns):
            return None
        if (row, column) in used or board[row][column] != word[index]:
            return None
        used.add((row, column))
        path.append((row, column))
        try:
            if index == len(word) - 1:
                return path.copy()
            for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                answer = search(row + dr, column + dc, index + 1)
                if answer is not None:
                    return answer
            return None
        finally:
            path.pop()
            used.remove((row, column))

    for row in range(rows):
        for column in range(columns):
            answer = search(row, column, 0)
            if answer is not None:
                return answer
    return None

board = [["A", "B"], ["C", "A"]]
before = [row.copy() for row in board]
print(find_word(board, "ABA"))
print(find_word(board, "ABAB"))
print(find_word([], ""))
print("input preserved:", board == before)
`,
    expected: '[(0, 0), (0, 1), (1, 1)]\nNone\n[]\ninput preserved: True',
  },
  inversions: {
    title: 'Return a sorted region and its inversion count together',
    code: `def count_inversions(values):
    data = list(values)
    buffer = [None] * len(data)

    def solve(low, high):
        if high - low <= 1:
            return 0
        middle = (low + high) // 2
        count = solve(low, middle) + solve(middle, high)
        left, right, output = low, middle, low
        while left < middle and right < high:
            if data[left] <= data[right]:
                buffer[output] = data[left]
                left += 1
            else:
                buffer[output] = data[right]
                right += 1
                count += middle - left
            output += 1
        while left < middle:
            buffer[output] = data[left]
            left += 1
            output += 1
        while right < high:
            buffer[output] = data[right]
            right += 1
            output += 1
        for index in range(low, high):
            data[index] = buffer[index]
        return count

    count = solve(0, len(data))
    return count, data

values = [3, 1, 2, 1]
print(count_inversions(values))
print("original:", values)
print(count_inversions([]))
print(count_inversions([2, 2, 2]))
`,
    expected: '(4, [1, 1, 2, 3])\noriginal: [3, 1, 2, 1]\n(0, [])\n(0, [2, 2, 2])',
  },
  summary: {
    title: 'Four summaries make the combine step constant work',
    code: `from typing import NamedTuple

class Summary(NamedTuple):
    total: int
    prefix: int
    suffix: int
    best: int

def summarize(values):
    if not values:
        raise ValueError("a nonempty subarray requires nonempty input")

    def solve(low, high):
        if high - low == 1:
            value = values[low]
            return Summary(value, value, value, value)
        middle = (low + high) // 2
        left = solve(low, middle)
        right = solve(middle, high)
        return Summary(
            left.total + right.total,
            max(left.prefix, left.total + right.prefix),
            max(right.suffix, left.suffix + right.total),
            max(left.best, right.best, left.suffix + right.prefix),
        )

    return solve(0, len(values))

print(summarize([-2, 4, -1, 3, -5, 2]))
print(summarize([-8, -3, -6]))
print(summarize([5]))
try:
    summarize([])
except ValueError as error:
    print(error)
`,
    expected: 'Summary(total=1, prefix=4, suffix=3, best=6)\nSummary(total=-17, prefix=-8, suffix=-6, best=-3)\nSummary(total=5, prefix=5, suffix=5, best=5)\na nonempty subarray requires nonempty input',
  },
  parentheses: {
    title: 'Exercise solution: generate balanced parentheses from a prefix invariant',
    code: `def balanced_parentheses(n):
    if type(n) is not int or n < 0:
        raise ValueError("n must be nonnegative")
    path, answers = [], []

    def search(opened, closed):
        if closed == n:
            answers.append("".join(path))
            return
        if opened < n:
            path.append("(")
            search(opened + 1, closed)
            path.pop()
        if closed < opened:
            path.append(")")
            search(opened, closed + 1)
            path.pop()

    search(0, 0)
    return answers

print(balanced_parentheses(3))
print(balanced_parentheses(0))
`,
    expected: "['((()))', '(()())', '(())()', '()(())', '()()()']\n['']",
  },
};
