"""Execute each complete displayed program and save its actual stdout."""

import contextlib
import io
import json
import textwrap
from pathlib import Path

import black


examples = {}


def add(key, title, question, code):
    formatted = black.format_str(textwrap.dedent(code).strip() + '\n', mode=black.Mode(line_length=80))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(formatted, key, 'exec'), {})
    examples[key] = {
        'title': title,
        'question': question,
        'code': formatted.rstrip(),
        'expected': stream.getvalue().rstrip(),
        'language': 'python',
    }


add('choices', 'Count legal branches',
    'If Ada can work only with Bo, while Bo and Cam each have two eligible partners, why does multiplying 3 by 2 overcount?', r'''
    from itertools import permutations

    def legal_assignments(partners):
        return [(lead, helper) for lead, helpers in partners.items()
                for helper in helpers]

    partners = {'Ada': ['Bo'], 'Bo': ['Ada', 'Cam'], 'Cam': ['Ada', 'Bo']}
    assignments = legal_assignments(partners)
    print('branch sizes:', [len(helpers) for helpers in partners.values()])
    print('assignments:', assignments)
    print('sum:', len(assignments), 'naive product:', len(partners) * 2)
    print('four people, two different roles:', len(list(permutations('ABCD', 2))))
    print('one empty assignment:', list(permutations('ABCD', 0)))
    ''')

add('fibers', 'Inspect what forgetting order merges',
    'Which unordered A/B outcome has twice as many ordered descriptions as the others? What probability does that produce under two independent fair draws?', r'''
    from collections import defaultdict
    from fractions import Fraction
    from itertools import permutations, product

    def group_descriptions(descriptions):
        groups = defaultdict(list)
        for description in descriptions:
            groups[''.join(sorted(description))].append(''.join(description))
        return dict(groups)

    committees = group_descriptions(permutations('ABCD', 2))
    draws = group_descriptions(product('AB', repeat=2))
    print('committee fibers:', committees)
    print('repeated-draw fibers:', draws)
    print('fair ordered draws -> grouped probabilities:',
          {key: str(Fraction(len(members), 4)) for key, members in draws.items()})
    print('three distinct multisets; 4 / 2 is not their count')
    ''')

add('multiset', 'Temporarily label the repeated symbols',
    'Why do 24 permutations of the four positions in NOON produce only six different strings, and what is the fiber size for each string?', r'''
    from collections import Counter
    from itertools import permutations
    from math import factorial, prod

    def multiset_count(word):
        counts = Counter(word)
        return factorial(len(word)) // prod(factorial(count) for count in counts.values())

    indexed = list(permutations(range(4)))
    displayed = Counter(''.join('NOON'[index] for index in ordering)
                        for ordering in indexed)
    print('indexed descriptions:', len(indexed))
    print('displayed strings and fiber sizes:', dict(sorted(displayed.items())))
    print('formula:', multiset_count('NOON'))
    print('BANANA:', multiset_count('BANANA'))
    print('empty word:', multiset_count(''))
    ''')

add('binomial', 'Count a chaired committee in two orders',
    'For six people and a three-person committee, will choosing the chair first or last describe the same 60 objects?', r'''
    from itertools import combinations
    from math import comb

    people = tuple('ABCDEF')
    chair_last = {(committee, chair) for committee in combinations(people, 3)
                  for chair in committee}
    chair_first = {(tuple(sorted((chair,) + others)), chair) for chair in people
                   for others in combinations(tuple(p for p in people if p != chair), 2)}
    print('same chaired committees:', chair_last == chair_first)
    print('count:', len(chair_last), 'formulas:', 3 * comb(6, 3), 6 * comb(5, 2))
    print('Pascal:', comb(6, 3), '=', comb(5, 3), '+', comb(5, 2))
    print('sum C(j, 2), j=2..6:', sum(comb(j, 2) for j in range(2, 7)))
    print('same as C(7, 3):', comb(7, 3))
    print('C(60, 30), exact integer:', comb(60, 30))
    ''')

add('allocations', 'Count bounded allocations two different ways',
    'After shifting the lower bounds, which allocations must be excluded for exceeding a capacity? Predict the changed case before running it.', r'''
    from itertools import combinations, product
    from math import comb

    def bounded_count(total, capacities, minimums=None):
        minimums = [0] * len(capacities) if minimums is None else minimums
        if not isinstance(total, int) or total < 0:
            raise ValueError('total must be a nonnegative integer')
        if len(capacities) != len(minimums):
            raise ValueError('bounds must have equal lengths')
        if any(not isinstance(v, int) or v < 0 for v in [*capacities, *minimums]):
            raise ValueError('bounds must be nonnegative integers')
        if not capacities:
            return int(total == 0)
        if any(low > high for low, high in zip(minimums, capacities)):
            return 0
        shifted = total - sum(minimums)
        answer = 0
        for size in range(len(capacities) + 1):
            for excluded in combinations(range(len(capacities)), size):
                remaining = shifted - sum(capacities[i] - minimums[i] + 1 for i in excluded)
                if remaining >= 0:
                    answer += (-1) ** size * comb(remaining + len(capacities) - 1,
                                                 len(capacities) - 1)
        return answer

    def enumerate_allocations(total, capacities, minimums):
        return [values for values in product(*(range(low, high + 1)
                for low, high in zip(minimums, capacities))) if sum(values) == total]

    for total, capacities, minimums in [(5, [5, 5, 5], [0, 0, 0]),
                                      (5, [5, 5, 5], [1, 1, 1]),
                                      (5, [3, 3, 3], [0, 0, 0]),
                                      (7, [4, 3, 3], [1, 0, 2])]:
        values = enumerate_allocations(total, capacities, minimums)
        count = bounded_count(total, capacities, minimums)
        assert count == len(values)
        print(total, capacities, minimums, '->', count)
    print('changed allocations:', values)
    print('zero containers:', bounded_count(0, []), bounded_count(2, []))
    ''')

add('overlap', 'Correct overlap and forbidden fixed points',
    'What net weight does 12 receive in the union of multiples of 2, 3 and 4? Why do derangements use the same alternating correction?', r'''
    from itertools import combinations, permutations
    from math import comb, factorial

    def union_terms(sets):
        terms = []
        for size in range(1, len(sets) + 1):
            for selected in combinations(range(len(sets)), size):
                common = set.intersection(*(sets[index] for index in selected))
                terms.append((selected, (-1) ** (size + 1) * len(common)))
        return terms

    def derangements(n):
        return sum((-1) ** chosen * comb(n, chosen) * factorial(n - chosen)
                   for chosen in range(n + 1))

    sets = [{value for value in range(1, 13) if value % divisor == 0}
            for divisor in (2, 3, 4)]
    print('signed terms:', [count for _, count in union_terms(sets)])
    print('union:', sorted(set.union(*sets)))
    for size in (0, 4, 5):
        direct = sum(all(position != value for position, value in enumerate(order))
                     for order in permutations(range(size)))
        assert direct == derangements(size)
        print('derangements:', size, '->', direct)
    ''')

add('pigeonhole', 'Count the available codewords',
    'Does allowing variable-length outputs create enough shorter binary strings to encode every four-bit input without collisions?', r'''
    from collections import Counter
    from itertools import product

    def shorter_strings(length):
        return [''.join(bits) for size in range(length)
                for bits in product('01', repeat=size)]

    for length in range(1, 5):
        print('length', length, 'inputs', 2**length,
              'shorter outputs', len(shorter_strings(length)))
    boxes = [index % 3 for index in range(7)]
    print('balanced placement of 7 items:', dict(Counter(boxes)))
    print('forced largest box:', (7 + 3 - 1) // 3)
    print('shorter outputs for two-bit inputs:', shorter_strings(2))
    ''')

add('recurrences', 'Split structures by what happens to the last object',
    'For four distinct items, why are there seven partitions into two nonempty unlabeled blocks, but fourteen assignments onto two labeled boxes?', r'''
    def tilings(length):
        if length == 0:
            return [()]
        return [(first,) + rest for first in (1, 2) if first <= length
                for rest in tilings(length - first)]

    def partitions(items):
        if not items:
            return [()]
        last = items[-1]
        result = []
        for old in partitions(items[:-1]):
            result.append(old + ((last,),))
            for index in range(len(old)):
                result.append(old[:index] + (old[index] + (last,),) + old[index + 1:])
        return result

    def stirling(objects, blocks):
        row = [1] + [0] * blocks
        for _ in range(objects):
            row = [0] + [count * row[count] + row[count - 1]
                         for count in range(1, blocks + 1)]
        return row[blocks]

    print('tilings, lengths 0..6:', [len(tilings(n)) for n in range(7)])
    two_blocks = [grouping for grouping in partitions(tuple('ABCD')) if len(grouping) == 2]
    print('partitions of ABCD into two blocks:', two_blocks)
    print('recurrence:', stirling(4, 2), 'onto two labeled boxes:', 2 * stirling(4, 2))
    print('S(0,0), S(3,0), S(3,4):', stirling(0, 0), stirling(3, 0), stirling(3, 4))
    ''')

add('induction', 'Construct the witness promised by the proof',
    'Which of 18, 19, 20 or 21 does the subtract-four chain from 38 reach? What fails in that proof if this base certificate is removed?', r'''
    def witness(target, step, other, lower, bases):
        if target < lower:
            raise ValueError('target is below the theorem range')
        base = lower + (target - lower) % step
        if base not in bases:
            return None  # Unsupported by these certificates, not proved impossible.
        first, second = bases[base]
        assert step * first + other * second == base
        return first + (target - base) // step, second

    bases = {18: (1, 2), 19: (3, 1), 20: (5, 0), 21: (0, 3)}
    for target in (18, 21, 38, 57):
        counts = witness(target, 4, 7, 18, bases)
        assert 4 * counts[0] + 7 * counts[1] == target
        print(target, '->', counts, '(counts of 4 and 7)')
    without_twenty = {base: counts for base, counts in bases.items() if base != 20}
    print('40 without base 20:', witness(40, 4, 7, 18, without_twenty))
    print('40 is still representable:', 10 * 4)
    changed = {8: (1, 1), 9: (3, 0), 10: (0, 2)}
    print('changed 3/5 target 29:', witness(29, 3, 5, 8, changed))
    ''')

add('catalan', 'Reflect bad paths and verify the inverse',
    'For four pairs, are there 70 unrestricted paths and 56 bad ones? Check that every bad path has exactly one reflected image and can be recovered.', r'''
    from itertools import combinations
    from math import comb

    def all_words(pairs):
        return [''.join('(' if index in openings else ')' for index in range(2 * pairs))
                for openings in combinations(range(2 * pairs), pairs)]

    def first_height(word, target):
        height = 0
        for length, symbol in enumerate(word, start=1):
            height += 1 if symbol == '(' else -1
            if height == target:
                return length
        return None

    def flip_prefix(word, length):
        return ''.join(')' if symbol == '(' else '(' for symbol in word[:length]) + word[length:]

    def catalan(pairs):
        return comb(2 * pairs, pairs) // (pairs + 1)

    for pairs in range(5):
        words = all_words(pairs)
        good = [word for word in words if first_height(word, -1) is None]
        bad = [word for word in words if first_height(word, -1) is not None]
        images = {flip_prefix(word, first_height(word, -1)) for word in bad}
        assert len(images) == len(bad)
        for image in images:
            original = flip_prefix(image, first_height(image, 1))
            assert original in bad
        assert len(good) == catalan(pairs)
        print('pairs', pairs, 'all', len(words), 'bad', len(bad), 'balanced', len(good))
    word = '())('
    image = flip_prefix(word, first_height(word, -1))
    print('bad path:', word, 'image:', image, 'inverse:', flip_prefix(image, first_height(image, 1)))
    ''')

add('coefficients', 'Build the count one factor at a time',
    'Which coefficient counts three tokens with capacities 2, 3 and 1? What changes for capacities 1, 2 and 2 with total four?', r'''
    from itertools import product

    def coefficient_rows(capacities):
        rows = [[1]]
        for capacity in capacities:
            if not isinstance(capacity, int) or capacity < 0:
                raise ValueError('capacity must be a nonnegative integer')
            previous = rows[-1]
            next_row = [0] * (len(previous) + capacity)
            for degree, count in enumerate(previous):
                for extra in range(capacity + 1):
                    next_row[degree + extra] += count
            rows.append(next_row)
        return rows

    for capacities, target in [([2, 3, 1], 3), ([1, 2, 2], 4)]:
        rows = coefficient_rows(capacities)
        direct = [values for values in product(*(range(cap + 1) for cap in capacities))
                  if sum(values) == target]
        assert rows[-1][target] == len(direct)
        print('capacities:', capacities, 'coefficient rows:', rows)
        print('target:', target, 'count:', len(direct), 'allocations:', direct)
    ''')

add('rotations', 'Group patterns and count fixed rotations',
    'Why does the pattern 0101 have only two distinct rotations? Compare direct orbits with the average number of fixed patterns.', r'''
    from collections import defaultdict
    from itertools import product

    def rotate(word, shift):
        return word[shift:] + word[:shift]

    def orbit_data(length):
        if not isinstance(length, int) or not 1 <= length <= 12:
            raise ValueError('enumerate rings of 1 through 12 sites')
        words = [''.join(bits) for bits in product('01', repeat=length)]
        groups = defaultdict(list)
        for word in words:
            groups[min(rotate(word, shift) for shift in range(length))].append(word)
        fixed = [sum(rotate(word, shift) == word for word in words) for shift in range(length)]
        assert sum(fixed) == length * len(groups)
        return dict(groups), fixed

    for length in (4, 5, 6):
        groups, fixed = orbit_data(length)
        print('sites', length, 'orbits', len(groups), 'fixed counts', fixed)
    word = '0101'
    images = {rotate(word, shift) for shift in range(len(word))}
    stabilizers = [shift for shift in range(len(word)) if rotate(word, shift) == word]
    print('0101 images:', sorted(images), 'fixing shifts:', stabilizers)
    print('orbit × stabilizer:', len(images) * len(stabilizers))
    ''')

add('capstone', 'Select a team and allocate its resources',
    'A committee needs at least two of three trained people, plus five identical tokens across stations with capacities 3, 3, 3. How many combined outcomes exist? Change both constraints before trusting the result.', r'''
    from itertools import combinations, product
    from math import comb

    def count_plan(team_size, trained_needed, token_total, capacities):
        trained = set('ABC')
        people = tuple('ABCDE')
        teams = [team for team in combinations(people, team_size)
                 if len(set(team) & trained) >= trained_needed]
        allocations = [amounts for amounts in product(*(range(cap + 1) for cap in capacities))
                       if sum(amounts) == token_total]
        by_training = sum(comb(3, selected) * comb(2, team_size - selected)
                          for selected in range(trained_needed, min(3, team_size) + 1)
                          if 0 <= team_size - selected <= 2)
        assert by_training == len(teams)
        # Every legal team may use every legal allocation: this is the product justification.
        return {'teams': len(teams), 'allocations': len(allocations),
                'combined': len(teams) * len(allocations)}

    print('original:', count_plan(3, 2, 5, [3, 3, 3]))
    print('changed:', count_plan(2, 1, 4, [2, 2, 2]))
    ''')


root = Path(__file__).resolve().parents[1]
target = root / 'src/learn/data/counting-combinatorics-examples.js'
target.write_text('export const countingCombinatoricsExamples = ' + json.dumps(examples, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
directory = root / 'scratch/counting-combinatorics-verification'
directory.mkdir(parents=True, exist_ok=True)
(directory / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Executed and captured {len(examples)} complete programs.')
