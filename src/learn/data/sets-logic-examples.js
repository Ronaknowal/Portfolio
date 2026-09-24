export const setsLogicExamples = {
  "sets": {
    "title": "A roster is a set of distinct people",
    "question": "Which people are selected, and how does a set containing a set differ from the inner element?",
    "language": "python",
    "code": "universe = {'Ada', 'Bo', 'Cam', 'Dee'}\ntrained = {'Ada', 'Bo'}\nbadged = {'Bo', 'Cam'}\nfor name, result in [\n    ('both', trained & badged),\n    ('either', trained | badged),\n    ('trained only', trained - badged),\n    ('not trained in U', universe - trained),\n    ('exactly one', trained ^ badged),\n]:\n    print(name, sorted(result))\n\ninner = frozenset({2})\nouter = frozenset({1, inner})\nprint('singleton is an element', inner in outer)\nprint('2 is an element', 2 in outer)\nprint('singleton is a subset', inner <= outer)\nprint('empty is a subset', frozenset() <= outer)\nprint('empty is an element', frozenset() in outer)",
    "expected": "both ['Bo']\neither ['Ada', 'Bo', 'Cam']\ntrained only ['Ada']\nnot trained in U ['Cam', 'Dee']\nexactly one ['Ada', 'Cam']\nsingleton is an element True\n2 is an element False\nsingleton is a subset False\nempty is a subset True\nempty is an element False"
  },
  "truth": {
    "title": "Find an admitted countermodel",
    "question": "Why does Q fail to establish P even when P implies Q?",
    "language": "python",
    "code": "from itertools import product\n\ndef inspect(premises, conclusion):\n    admitted = []\n    failures = []\n    for p, q in product((False, True), repeat=2):\n        if all(rule(p, q) for rule in premises):\n            admitted.append((p, q))\n            if not conclusion(p, q):\n                failures.append((p, q))\n    return {'valid': not failures, 'consistent': bool(admitted),\n            'countermodels': failures}\n\nimplies = lambda p, q: (not p) or q\ncases = [\n    ('affirm consequent', [implies, lambda p, q: q], lambda p, q: p),\n    ('modus ponens', [implies, lambda p, q: p], lambda p, q: q),\n    ('modus tollens', [implies, lambda p, q: not q], lambda p, q: not p),\n    ('inconsistent', [lambda p, q: p, lambda p, q: not p], lambda p, q: q),\n]\nfor name, premises, conclusion in cases:\n    print(name, inspect(premises, conclusion))",
    "expected": "affirm consequent {'valid': False, 'consistent': True, 'countermodels': [(False, True)]}\nmodus ponens {'valid': True, 'consistent': True, 'countermodels': []}\nmodus tollens {'valid': True, 'consistent': True, 'countermodels': []}\ninconsistent {'valid': True, 'consistent': False, 'countermodels': []}"
  },
  "quantifiers": {
    "title": "A witness per job versus one common witness",
    "question": "How do the two claims change after an assignment edit or an empty domain?",
    "language": "python",
    "code": "def evaluate(jobs, reviewers, approvals):\n    per_job = {\n        job: [person for person in reviewers if (job, person) in approvals]\n        for job in jobs\n    }\n    common = [person for person in reviewers\n              if all((job, person) in approvals for job in jobs)]\n    return all(per_job.values()), bool(common), per_job, common\n\njobs = ['Scan', 'Label', 'Audit']\nreviewers = ['Ada', 'Bo', 'Cam']\napprovals = set(zip(jobs, reviewers))\nprint('diagonal', evaluate(jobs, reviewers, approvals))\napprovals.update((job, 'Bo') for job in jobs)\nprint('shared Bo', evaluate(jobs, reviewers, approvals))\nprint('no jobs', evaluate([], reviewers, set()))\nprint('no reviewers', evaluate(jobs, [], set()))\nprint('both empty', evaluate([], [], set()))",
    "expected": "diagonal (True, False, {'Scan': ['Ada'], 'Label': ['Bo'], 'Audit': ['Cam']}, [])\nshared Bo (True, True, {'Scan': ['Ada', 'Bo'], 'Label': ['Bo'], 'Audit': ['Bo', 'Cam']}, ['Bo'])\nno jobs (True, True, {}, ['Ada', 'Bo', 'Cam'])\nno reviewers (False, False, {'Scan': [], 'Label': [], 'Audit': []}, [])\nboth empty (True, False, {}, [])"
  },
  "parity": {
    "title": "Inspect the arbitrary odd-number algebra",
    "question": "What integer witness makes the square of an odd integer odd, including a negative input?",
    "language": "python",
    "code": "def odd_square_witness(n):\n    if not isinstance(n, int) or n % 2 != 1:\n        raise ValueError('n must be an odd integer')\n    k = (n - 1) // 2\n    witness = 2 * k * k + 2 * k\n    assert n == 2 * k + 1\n    assert n * n == 2 * witness + 1\n    return k, witness\n\nfor n in [-5, -3, 1, 7]:\n    k, witness = odd_square_witness(n)\n    print(n, 'k =', k, 'square =', n * n, 'odd witness =', witness)",
    "expected": "-5 k = -3 square = 25 odd witness = 12\n-3 k = -2 square = 9 odd witness = 4\n1 k = 0 square = 1 odd witness = 0\n7 k = 3 square = 49 odd witness = 24"
  },
  "images": {
    "title": "Forward images can merge different inputs",
    "question": "Why can two disjoint input sets have overlapping images, while a preimage intersection law still holds?",
    "language": "python",
    "code": "domain = {-2, -1, 0, 1, 2}\ncodomain = {0, 1, 4}\nfunction = {x: x * x for x in domain}\n\ndef image(values):\n    return {function[x] for x in values}\n\ndef preimage(values):\n    return {x for x in domain if function[x] in values}\n\nfirst, second = {-2}, {2}\nprint('image of intersection', sorted(image(first & second)))\nprint('intersection of images', sorted(image(first) & image(second)))\ntarget_a, target_b = {0, 4}, {1, 4}\nprint('preimage of intersection', sorted(preimage(target_a & target_b)))\nprint('intersection of preimages', sorted(preimage(target_a) & preimage(target_b)))\nprint('preimage of empty', sorted(preimage(set())))\nprint('preimage of codomain', sorted(preimage(codomain)))",
    "expected": "image of intersection []\nintersection of images [4]\npreimage of intersection [-2, 2]\nintersection of preimages [-2, 2]\npreimage of empty []\npreimage of codomain [-2, -1, 0, 1, 2]"
  },
  "relations": {
    "title": "A failed property must have a witness",
    "question": "Why is being close different from belonging to the same remainder class?",
    "language": "python",
    "code": "from itertools import product\n\ndef inspect_relation(elements, relation):\n    reflexive = next(((a,) for a in elements if (a, a) not in relation), None)\n    symmetric = next(((a, b) for a, b in product(elements, repeat=2)\n                      if (a, b) in relation and (b, a) not in relation), None)\n    transitive = next(((a, b, c) for a, b, c in product(elements, repeat=3)\n                       if (a, b) in relation and (b, c) in relation\n                       and (a, c) not in relation), None)\n    failures = {'reflexive': reflexive, 'symmetric': symmetric, 'transitive': transitive}\n    if any(value is not None for value in failures.values()):\n        return failures, None\n    classes = []\n    remaining = set(elements)\n    for a in elements:\n        if a in remaining:\n            members = [b for b in elements if (a, b) in relation]\n            classes.append(members)\n            remaining.difference_update(members)\n    return failures, classes\n\nuniverse = list(range(6))\nclose = {(a, b) for a, b in product(universe, repeat=2) if abs(a - b) <= 1}\nsame_remainder = {(a, b) for a, b in product(universe, repeat=2) if (a - b) % 3 == 0}\nprint('nearby', inspect_relation(universe, close))\nprint('modulo 3', inspect_relation(universe, same_remainder))\nprint('empty universe', inspect_relation([], set()))",
    "expected": "nearby ({'reflexive': None, 'symmetric': None, 'transitive': (0, 1, 2)}, None)\nmodulo 3 ({'reflexive': None, 'symmetric': None, 'transitive': None}, [[0, 3], [1, 4], [2, 5]])\nempty universe ({'reflexive': None, 'symmetric': None, 'transitive': None}, [])"
  },
  "orders": {
    "title": "Covers omit implied comparisons",
    "question": "Which divisors are minimal, and when is one element below every other element?",
    "language": "python",
    "code": "from itertools import product\n\ndef order_summary(elements):\n    if any(not isinstance(x, int) or x <= 0 for x in elements):\n        raise ValueError('Use distinct positive integers')\n    if len(set(elements)) != len(elements):\n        raise ValueError('Use distinct positive integers')\n    def precedes(a, b):\n        return b % a == 0\n    covers = [(a, b) for a, b in product(elements, repeat=2)\n              if a != b and precedes(a, b)\n              and not any(c not in (a, b) and precedes(a, c) and precedes(c, b)\n                          for c in elements)]\n    minimal = [a for a in elements if not any(b != a and precedes(b, a) for b in elements)]\n    least = [a for a in elements if all(precedes(a, b) for b in elements)]\n    return {'covers': covers, 'minimal': minimal, 'least': least}\n\nfor elements in [[1, 2, 3, 6], [2, 3, 6], [4, 6, 12, 24], []]:\n    print(elements, order_summary(elements))",
    "expected": "[1, 2, 3, 6] {'covers': [(1, 2), (1, 3), (2, 6), (3, 6)], 'minimal': [1], 'least': [1]}\n[2, 3, 6] {'covers': [(2, 6), (3, 6)], 'minimal': [2, 3], 'least': []}\n[4, 6, 12, 24] {'covers': [(4, 12), (6, 12), (12, 24)], 'minimal': [4, 6], 'least': []}\n[] {'covers': [], 'minimal': [], 'least': []}"
  },
  "induction": {
    "title": "Count the border before using the formula",
    "question": "How many new tiles turn an n-by-n square into an (n+1)-by-(n+1) square?",
    "language": "python",
    "code": "total_odds = 0\nfor n in range(5):\n    old = {(row, column) for row in range(n) for column in range(n)}\n    new = {(row, column) for row in range(n + 1) for column in range(n + 1)}\n    border = new - old\n    total_odds += len(border)\n    print('n =', n, 'old =', len(old), 'border =', len(border),\n          'new =', len(new), 'odd sum =', total_odds)\n    assert len(border) == 2 * n + 1\n    assert total_odds == (n + 1) ** 2\nprint('The general proof still needs an arbitrary n, not these five checks.')",
    "expected": "n = 0 old = 0 border = 1 new = 1 odd sum = 1\nn = 1 old = 1 border = 3 new = 4 odd sum = 4\nn = 2 old = 4 border = 5 new = 9 odd sum = 9\nn = 3 old = 9 border = 7 new = 16 odd sum = 16\nn = 4 old = 16 border = 9 new = 25 odd sum = 25\nThe general proof still needs an arbitrary n, not these five checks."
  },
  "diagonal": {
    "title": "Build a subset that is missing from every proposed row",
    "question": "Which membership decision forces the constructed subset to differ from row i?",
    "language": "python",
    "code": "def missing_subset(proposed):\n    universe = set(range(len(proposed)))\n    if any(not subset <= universe for subset in proposed):\n        raise ValueError('Every proposed subset must use the declared universe')\n    return {i for i, subset in enumerate(proposed) if i not in subset}\n\nproposed = [{0, 2}, {0}, {1, 2, 3}, {0, 1}]\nmissing = missing_subset(proposed)\nprint('D =', sorted(missing))\nfor i, subset in enumerate(proposed):\n    print('row', i, 'at element', i, ': proposed =', i in subset, 'D =', i in missing)\n    assert subset != missing\nprint('empty domain has missing subset', sorted(missing_subset([])))",
    "expected": "D = [1, 3]\nrow 0 at element 0 : proposed = True D = False\nrow 1 at element 1 : proposed = False D = True\nrow 2 at element 2 : proposed = True D = False\nrow 3 at element 3 : proposed = False D = True\nempty domain has missing subset []"
  },
  "finiteTesting": {
    "title": "A finite run can miss the next counterexample",
    "question": "Can four successful cases establish that this polynomial is always zero?",
    "language": "python",
    "code": "from math import prod\n\ndef vanishes_on_tested_values(n, last_tested):\n    return prod(n - k for k in range(last_tested + 1))\n\nfor n in range(5):\n    print('n =', n, 'value =', vanishes_on_tested_values(n, 3))\nprint('changed test range')\nfor n in [6, 7]:\n    print('n =', n, 'value =', vanishes_on_tested_values(n, 6))",
    "expected": "n = 0 value = 0\nn = 1 value = 0\nn = 2 value = 0\nn = 3 value = 0\nn = 4 value = 24\nchanged test range\nn = 6 value = 0\nn = 7 value = 5040"
  },
  "policy": {
    "title": "Choose the claim before fixing the data",
    "question": "Does adding one qualified reviewer per job also make every assigned reviewer qualified?",
    "language": "python",
    "code": "universe = {'Ada', 'Bo', 'Cam', 'Dee'}\ntrained = {'Ada', 'Bo'}\nbadged = {'Bo', 'Cam'}\njobs = ['Scan', 'Label', 'Audit']\nassignments = {'Scan': {'Ada'}, 'Label': {'Bo'}, 'Audit': {'Bo', 'Cam'}}\n\ndef audit(assignments):\n    qualified = trained & badged\n    if set(assignments) != set(jobs) or any(not values <= universe for values in assignments.values()):\n        raise ValueError('Use exactly these jobs and people from the roster')\n    witnesses = {job: sorted(assignments[job] & qualified) for job in jobs}\n    uncovered = [job for job in jobs if not witnesses[job]]\n    unqualified = {job: sorted(assignments[job] - qualified) for job in jobs\n                   if assignments[job] - qualified}\n    common = sorted(qualified.intersection(*(assignments[job] for job in jobs)))\n    return {'witnesses': witnesses, 'uncovered': uncovered,\n            'unqualified assignments': unqualified, 'common': common}\n\nprint('before', audit(assignments))\nchanged = {job: people | {'Bo'} for job, people in assignments.items()}\nprint('after adding Bo', audit(changed))\nstrict = {job: people & (trained & badged) for job, people in changed.items()}\nprint('after removing unqualified', audit(strict))",
    "expected": "before {'witnesses': {'Scan': [], 'Label': ['Bo'], 'Audit': ['Bo']}, 'uncovered': ['Scan'], 'unqualified assignments': {'Scan': ['Ada'], 'Audit': ['Cam']}, 'common': []}\nafter adding Bo {'witnesses': {'Scan': ['Bo'], 'Label': ['Bo'], 'Audit': ['Bo']}, 'uncovered': [], 'unqualified assignments': {'Scan': ['Ada'], 'Audit': ['Cam']}, 'common': ['Bo']}\nafter removing unqualified {'witnesses': {'Scan': ['Bo'], 'Label': ['Bo'], 'Audit': ['Bo']}, 'uncovered': [], 'unqualified assignments': {}, 'common': ['Bo']}"
  }
};
