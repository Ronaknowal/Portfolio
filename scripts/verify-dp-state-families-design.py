"""Finite design checks; these are proposed fixtures, not production verification."""
from collections import defaultdict
from datetime import datetime, timezone
from functools import cache
from itertools import permutations, product
import json
from pathlib import Path


def matrix_trees(dimensions):
    """Enumerate every ordered full parenthesization and its literal merge cost."""
    def visit(left, right):
        if right == left + 1:
            return [(0, f"A{left}")]
        answers = []
        for split in range(left + 1, right):
            for first_cost, first in visit(left, split):
                for second_cost, second in visit(split, right):
                    cost = first_cost + second_cost + dimensions[left] * dimensions[split] * dimensions[right]
                    answers.append((cost, f"({first}{second})"))
        return answers
    return sorted(visit(0, len(dimensions) - 1))


def balloon_permutations(values):
    """Replay current neighboring objects, with original indices as identities."""
    results = []
    for order in permutations(range(len(values))):
        live = list(range(len(values)))
        total = 0
        steps = []
        for original in order:
            slot = live.index(original)
            first = values[live[slot - 1]] if slot else 1
            second = values[live[slot + 1]] if slot + 1 < len(live) else 1
            earned = first * values[original] * second
            total += earned
            steps.append([original, first, values[original], second, earned])
            live.pop(slot)
        results.append((total, order, steps))
    return sorted(results, key=lambda entry: (-entry[0], entry[1]))


def independent_sets(weights, edges, blocked=None):
    results = []
    for selected in product([False, True], repeat=len(weights)):
        if blocked is not None and selected[blocked]:
            continue
        if any(selected[first] and selected[second] for first, second in edges):
            continue
        witness = [index for index, included in enumerate(selected) if included]
        results.append((sum(weights[index] for index in witness), witness))
    return sorted(results, key=lambda entry: (-entry[0], entry[1]))


def unique_digits(value):
    return len(set(str(value))) == len(str(value))


def digit_count(bound):
    digits = str(bound)
    @cache
    def visit(position, tight, started, used):
        if position == len(digits):
            return int(started)
        limit = int(digits[position]) if tight else 9
        answer = 0
        for digit in range(limit + 1):
            next_tight = tight and digit == int(digits[position])
            if not started and digit == 0:
                answer += visit(position + 1, next_tight, False, used)
            elif not used & (1 << digit):
                answer += visit(position + 1, next_tight, True, used | (1 << digit))
        return answer
    return visit(0, True, False, 0)


def main():
    matrix = {','.join(map(str, dims)): matrix_trees(dims) for dims in [
        [8, 2, 12, 3], [8, 2, 12, 3, 6], [3, 7, 2, 5], [2, 2, 2, 2], [4, 3],
    ]}
    assert matrix['8,2,12,3'][0][0] == 120
    balloon = {','.join(map(str, values)): balloon_permutations(values) for values in [
        [2, 4, 3], [2, 0, 3], [3, 1, 2, 4], [1, 1, 1], [],
    ]}
    weights = [5, 9, 2, 4, 1, 6, 3]
    edges = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
    tree = independent_sets(weights, edges)
    assert tree[0] == (19, [0, 3, 4, 5, 6])
    tree_blocked = independent_sets(weights, edges, blocked=0)
    assert tree_blocked[0][0] == 18
    digit_checks = 0
    cumulative = 0
    for bound in range(2501):
        if bound > 0 and unique_digits(bound):
            cumulative += 1
        assert digit_count(bound) == cumulative
        digit_checks += 1
    selected = [0, 9, 10, 99, 100, 101, 102, 120, 213, 999]
    counts = {str(bound): digit_count(bound) for bound in selected}
    prefix_groups = defaultdict(list)
    for value in range(1, 214):
        if unique_digits(value):
            prefix_groups[str(value).zfill(3)[:2]].append(value)
    assert len(prefix_groups['21']) == 2 and len(prefix_groups['12']) == 8
    adjacent = [value for value in range(100, 131) if all(a != b for a, b in zip(str(value), str(value)[1:]))]
    distinct = [value for value in range(100, 131) if unique_digits(value)]
    record = {
        'checkedAt': datetime.now(timezone.utc).isoformat(),
        'scope': 'Design-only exact fixtures. No production or browser implementation is asserted.',
        'matrixAllParenthesizations': matrix,
        'balloonAllRemovalOrders': balloon,
        'tree': {'weights': weights, 'edges': edges, 'best': tree[0], 'parentSelectedBest': tree_blocked[0]},
        'digitChecks': digit_checks,
        'uniqueCounts': counts,
        'prefix21': prefix_groups['21'], 'prefix12': prefix_groups['12'],
        'changedProperty100Through130': {'adjacentUnequal': adjacent, 'allDistinct': distinct},
        'matrixTies': matrix['2,2,2,2'],
    }
    out = Path('docs/teaching/evidence/dp-state-families-design-fixtures.json')
    out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'matrixDefault': matrix['8,2,12,3,6'][0],
        'balloonDefault': balloon['2,4,3'][0], 'tree': tree[0],
        'digitChecks': digit_checks, 'counts': counts,
        'changedCounts': [len(adjacent), len(distinct)],
    }, indent=2))


if __name__ == '__main__':
    main()
