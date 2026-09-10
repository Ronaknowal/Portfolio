"""Test actual lesson Python against enumeration oracles with different algorithms."""
import contextlib
import io
import itertools
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    examples = json.load(source)
programs = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name, "exec"), namespace)
    programs[name] = namespace

cases = {"subsets": 0, "target_subsets": 0, "permutations": 0, "combinations": 0, "queens": 0, "words": 0, "arrays": 0, "parentheses": 0}
# Exercise D: enumerate every possible two-cell continuation independently.
memo_board = [list("ABX"), list("BAY")]
memo_prefixes = [
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1)],
]
memo_outcomes = []
for prefix in memo_prefixes:
    assert "".join(memo_board[row][column] for row, column in prefix) == "ABA"
    continuations = []
    remaining = set(itertools.product(range(2), range(3))) - set(prefix)
    for tail in itertools.permutations(remaining, 2):
        route = prefix + list(tail)
        if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(route, route[1:])):
            if "".join(memo_board[row][column] for row, column in route) == "ABABX":
                continuations.append(tail)
    memo_outcomes.append(bool(continuations))
assert memo_outcomes == [False, True]
assert programs["wordSearch"]["find_word"](memo_board, "ABABX") == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)]

for length in range(7):
    expected = sorted(tuple(subset) for size in range(length + 1) for subset in itertools.combinations(range(length), size))
    assert sorted(map(tuple, programs["subsets"]["subsets"](list(range(length))))) == expected
    cases["subsets"] += 1

for length in range(5):
    for values in itertools.product(range(1, 4), repeat=length):
        for target in range(9):
            expected = sorted(subset for size in range(length + 1) for subset in itertools.combinations(range(length), size) if sum(values[item] for item in subset) == target)
            for pruning in [False, True]:
                result, _ = programs["targetSubsets"]["target_subsets"](values, target, pruning)
                assert sorted(map(tuple, result)) == expected
                cases["target_subsets"] += 1
        actual = programs["permutations"]["unique_permutations"](values)
        assert sorted(map(tuple, actual)) == sorted(set(itertools.permutations(values)))
        cases["permutations"] += 1

for size in range(5):
    for candidates in itertools.combinations(range(1, 5), size):
        for target in range(9):
            expected = []
            for counts in itertools.product(range(target + 1), repeat=size):
                if sum(count * value for count, value in zip(counts, candidates)) == target:
                    expected.append(tuple(value for value, count in zip(candidates, counts) for _ in range(count)))
            actual = programs["combinations"]["combination_sum"](candidates, target)
            assert sorted(map(tuple, actual)) == sorted(expected)
            cases["combinations"] += 1

for n in range(8):
    expected = [columns for columns in itertools.permutations(range(n)) if all(abs(columns[a] - columns[b]) != b - a for a in range(n) for b in range(a + 1, n))]
    assert sorted(map(tuple, programs["queens"]["queens_solutions"](n))) == sorted(expected)
    cases["queens"] += 1

coordinates = list(itertools.product(range(2), repeat=2))
for letters in itertools.product("AB", repeat=4):
    board = [list(letters[:2]), list(letters[2:])]
    all_words = {""}
    for length in range(1, 5):
        for route in itertools.permutations(coordinates, length):
            if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(route, route[1:])):
                all_words.add("".join(board[row][column] for row, column in route))
    for length in range(6):
        for text in itertools.product("AB", repeat=length):
            word = "".join(text)
            before = [row.copy() for row in board]
            witness = programs["wordSearch"]["find_word"](board, word)
            assert (witness is not None) == (word in all_words)
            assert board == before
            if witness is not None:
                assert len(set(witness)) == len(witness)
                assert "".join(board[row][column] for row, column in witness) == word
                assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(witness, witness[1:]))
            cases["words"] += 1

for length in range(7):
    for values in itertools.product([-2, 0, 3], repeat=length):
        count, ordered = programs["inversions"]["count_inversions"](values)
        assert count == sum(values[a] > values[b] for a in range(length) for b in range(a + 1, length))
        assert ordered == sorted(values)
        if length:
            result = programs["summary"]["summarize"](values)
            assert result.total == sum(values)
            assert result.prefix == max(sum(values[:end]) for end in range(1, length + 1))
            assert result.suffix == max(sum(values[start:]) for start in range(length))
            assert result.best == max(sum(values[start:end]) for start in range(length) for end in range(start + 1, length + 1))
        cases["arrays"] += 1

for n in range(6):
    expected = []
    for symbols in itertools.product("()", repeat=2 * n):
        balance = 0
        valid = True
        for symbol in symbols:
            balance += 1 if symbol == "(" else -1
            if balance < 0:
                valid = False
        if valid and balance == 0:
            expected.append("".join(symbols))
    assert programs["parentheses"]["balanced_parentheses"](n) == expected
    cases["parentheses"] += 1

for function, arguments in [
    (programs["targetSubsets"]["target_subsets"], ([0], 1)),
    (programs["combinations"]["combination_sum"], ([1, 1], 2)),
    (programs["combinations"]["combination_sum"], ([-1, 2], 1)),
    (programs["queens"]["queens_solutions"], (-1,)),
    (programs["wordSearch"]["find_word"], ([["A"], []], "A")),
    (programs["summary"]["summarize"], ([],)),
]:
    try:
        function(*arguments)
        raise AssertionError("invalid input accepted")
    except ValueError:
        pass
print("Native independent oracle cases:", cases)
