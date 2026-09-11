"""Execute and serialize the lesson's complete standalone Python examples."""

from pathlib import Path
import contextlib
import io
import json
import textwrap


examples = {}


def add(name, title, question, source):
    code = textwrap.dedent(source).strip()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, name + '.py', 'exec'), {})
    examples[name] = {
        'title': title,
        'question': question,
        'language': 'python',
        'code': code,
        'expected': output.getvalue().rstrip('\n'),
    }


add('sets', 'A roster is a set of distinct people',
    'Which people are selected, and how does a set containing a set differ from the inner element?', '''
    universe = {'Ada', 'Bo', 'Cam', 'Dee'}
    trained = {'Ada', 'Bo'}
    badged = {'Bo', 'Cam'}
    for name, result in [
        ('both', trained & badged),
        ('either', trained | badged),
        ('trained only', trained - badged),
        ('not trained in U', universe - trained),
        ('exactly one', trained ^ badged),
    ]:
        print(name, sorted(result))

    inner = frozenset({2})
    outer = frozenset({1, inner})
    print('singleton is an element', inner in outer)
    print('2 is an element', 2 in outer)
    print('singleton is a subset', inner <= outer)
    print('empty is a subset', frozenset() <= outer)
    print('empty is an element', frozenset() in outer)
''')

add('truth', 'Find an admitted countermodel',
    'Why does Q fail to establish P even when P implies Q?', '''
    from itertools import product

    def inspect(premises, conclusion):
        admitted = []
        failures = []
        for p, q in product((False, True), repeat=2):
            if all(rule(p, q) for rule in premises):
                admitted.append((p, q))
                if not conclusion(p, q):
                    failures.append((p, q))
        return {'valid': not failures, 'consistent': bool(admitted),
                'countermodels': failures}

    implies = lambda p, q: (not p) or q
    cases = [
        ('affirm consequent', [implies, lambda p, q: q], lambda p, q: p),
        ('modus ponens', [implies, lambda p, q: p], lambda p, q: q),
        ('modus tollens', [implies, lambda p, q: not q], lambda p, q: not p),
        ('inconsistent', [lambda p, q: p, lambda p, q: not p], lambda p, q: q),
    ]
    for name, premises, conclusion in cases:
        print(name, inspect(premises, conclusion))
''')

add('quantifiers', 'A witness per job versus one common witness',
    'How do the two claims change after an assignment edit or an empty domain?', '''
    def evaluate(jobs, reviewers, approvals):
        per_job = {
            job: [person for person in reviewers if (job, person) in approvals]
            for job in jobs
        }
        common = [person for person in reviewers
                  if all((job, person) in approvals for job in jobs)]
        return all(per_job.values()), bool(common), per_job, common

    jobs = ['Scan', 'Label', 'Audit']
    reviewers = ['Ada', 'Bo', 'Cam']
    approvals = set(zip(jobs, reviewers))
    print('diagonal', evaluate(jobs, reviewers, approvals))
    approvals.update((job, 'Bo') for job in jobs)
    print('shared Bo', evaluate(jobs, reviewers, approvals))
    print('no jobs', evaluate([], reviewers, set()))
    print('no reviewers', evaluate(jobs, [], set()))
    print('both empty', evaluate([], [], set()))
''')

add('parity', 'Inspect the arbitrary odd-number algebra',
    'What integer witness makes the square of an odd integer odd, including a negative input?', '''
    def odd_square_witness(n):
        if not isinstance(n, int) or n % 2 != 1:
            raise ValueError('n must be an odd integer')
        k = (n - 1) // 2
        witness = 2 * k * k + 2 * k
        assert n == 2 * k + 1
        assert n * n == 2 * witness + 1
        return k, witness

    for n in [-5, -3, 1, 7]:
        k, witness = odd_square_witness(n)
        print(n, 'k =', k, 'square =', n * n, 'odd witness =', witness)
''')

add('images', 'Forward images can merge different inputs',
    'Why can two disjoint input sets have overlapping images, while a preimage intersection law still holds?', '''
    domain = {-2, -1, 0, 1, 2}
    codomain = {0, 1, 4}
    function = {x: x * x for x in domain}

    def image(values):
        return {function[x] for x in values}

    def preimage(values):
        return {x for x in domain if function[x] in values}

    first, second = {-2}, {2}
    print('image of intersection', sorted(image(first & second)))
    print('intersection of images', sorted(image(first) & image(second)))
    target_a, target_b = {0, 4}, {1, 4}
    print('preimage of intersection', sorted(preimage(target_a & target_b)))
    print('intersection of preimages', sorted(preimage(target_a) & preimage(target_b)))
    print('preimage of empty', sorted(preimage(set())))
    print('preimage of codomain', sorted(preimage(codomain)))
''')

add('relations', 'A failed property must have a witness',
    'Why is being close different from belonging to the same remainder class?', '''
    from itertools import product

    def inspect_relation(elements, relation):
        reflexive = next(((a,) for a in elements if (a, a) not in relation), None)
        symmetric = next(((a, b) for a, b in product(elements, repeat=2)
                          if (a, b) in relation and (b, a) not in relation), None)
        transitive = next(((a, b, c) for a, b, c in product(elements, repeat=3)
                           if (a, b) in relation and (b, c) in relation
                           and (a, c) not in relation), None)
        failures = {'reflexive': reflexive, 'symmetric': symmetric, 'transitive': transitive}
        if any(value is not None for value in failures.values()):
            return failures, None
        classes = []
        remaining = set(elements)
        for a in elements:
            if a in remaining:
                members = [b for b in elements if (a, b) in relation]
                classes.append(members)
                remaining.difference_update(members)
        return failures, classes

    universe = list(range(6))
    close = {(a, b) for a, b in product(universe, repeat=2) if abs(a - b) <= 1}
    same_remainder = {(a, b) for a, b in product(universe, repeat=2) if (a - b) % 3 == 0}
    print('nearby', inspect_relation(universe, close))
    print('modulo 3', inspect_relation(universe, same_remainder))
    print('empty universe', inspect_relation([], set()))
''')

add('orders', 'Covers omit implied comparisons',
    'Which divisors are minimal, and when is one element below every other element?', '''
    from itertools import product

    def order_summary(elements):
        if any(not isinstance(x, int) or x <= 0 for x in elements):
            raise ValueError('Use distinct positive integers')
        if len(set(elements)) != len(elements):
            raise ValueError('Use distinct positive integers')
        def precedes(a, b):
            return b % a == 0
        covers = [(a, b) for a, b in product(elements, repeat=2)
                  if a != b and precedes(a, b)
                  and not any(c not in (a, b) and precedes(a, c) and precedes(c, b)
                              for c in elements)]
        minimal = [a for a in elements if not any(b != a and precedes(b, a) for b in elements)]
        least = [a for a in elements if all(precedes(a, b) for b in elements)]
        return {'covers': covers, 'minimal': minimal, 'least': least}

    for elements in [[1, 2, 3, 6], [2, 3, 6], [4, 6, 12, 24], []]:
        print(elements, order_summary(elements))
''')

add('induction', 'Count the border before using the formula',
    'How many new tiles turn an n-by-n square into an (n+1)-by-(n+1) square?', '''
    total_odds = 0
    for n in range(5):
        old = {(row, column) for row in range(n) for column in range(n)}
        new = {(row, column) for row in range(n + 1) for column in range(n + 1)}
        border = new - old
        total_odds += len(border)
        print('n =', n, 'old =', len(old), 'border =', len(border),
              'new =', len(new), 'odd sum =', total_odds)
        assert len(border) == 2 * n + 1
        assert total_odds == (n + 1) ** 2
    print('The general proof still needs an arbitrary n, not these five checks.')
''')

add('diagonal', 'Build a subset that is missing from every proposed row',
    'Which membership decision forces the constructed subset to differ from row i?', '''
    def missing_subset(proposed):
        universe = set(range(len(proposed)))
        if any(not subset <= universe for subset in proposed):
            raise ValueError('Every proposed subset must use the declared universe')
        return {i for i, subset in enumerate(proposed) if i not in subset}

    proposed = [{0, 2}, {0}, {1, 2, 3}, {0, 1}]
    missing = missing_subset(proposed)
    print('D =', sorted(missing))
    for i, subset in enumerate(proposed):
        print('row', i, 'at element', i, ': proposed =', i in subset, 'D =', i in missing)
        assert subset != missing
    print('empty domain has missing subset', sorted(missing_subset([])))
''')

add('finiteTesting', 'A finite run can miss the next counterexample',
    'Can four successful cases establish that this polynomial is always zero?', '''
    from math import prod

    def vanishes_on_tested_values(n, last_tested):
        return prod(n - k for k in range(last_tested + 1))

    for n in range(5):
        print('n =', n, 'value =', vanishes_on_tested_values(n, 3))
    print('changed test range')
    for n in [6, 7]:
        print('n =', n, 'value =', vanishes_on_tested_values(n, 6))
''')

add('policy', 'Choose the claim before fixing the data',
    'Does adding one qualified reviewer per job also make every assigned reviewer qualified?', '''
    universe = {'Ada', 'Bo', 'Cam', 'Dee'}
    trained = {'Ada', 'Bo'}
    badged = {'Bo', 'Cam'}
    jobs = ['Scan', 'Label', 'Audit']
    assignments = {'Scan': {'Ada'}, 'Label': {'Bo'}, 'Audit': {'Bo', 'Cam'}}

    def audit(assignments):
        qualified = trained & badged
        if set(assignments) != set(jobs) or any(not values <= universe for values in assignments.values()):
            raise ValueError('Use exactly these jobs and people from the roster')
        witnesses = {job: sorted(assignments[job] & qualified) for job in jobs}
        uncovered = [job for job in jobs if not witnesses[job]]
        unqualified = {job: sorted(assignments[job] - qualified) for job in jobs
                       if assignments[job] - qualified}
        common = sorted(qualified.intersection(*(assignments[job] for job in jobs)))
        return {'witnesses': witnesses, 'uncovered': uncovered,
                'unqualified assignments': unqualified, 'common': common}

    print('before', audit(assignments))
    changed = {job: people | {'Bo'} for job, people in assignments.items()}
    print('after adding Bo', audit(changed))
    strict = {job: people & (trained & badged) for job, people in changed.items()}
    print('after removing unqualified', audit(strict))
''')

destination = Path('src/learn/data/sets-logic-examples.js')
destination.write_text('export const setsLogicExamples = ' + json.dumps(examples, indent=2, ensure_ascii=False) + ';\n', encoding='utf-8')
print(f'Executed and serialized {len(examples)} complete Python programs.')
