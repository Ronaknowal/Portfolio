const satCore = `from itertools import product, combinations

def validate_formula(n, clauses):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if any(type(lit) is not int or not 1 <= abs(lit) <= n
           for clause in clauses for lit in clause):
        raise ValueError("Literals are signed variable indices 1..n")

def satisfies(n, clauses, assignment):
    validate_formula(n, clauses)
    if len(assignment) != n or any(type(value) is not bool for value in assignment):
        raise ValueError("One Boolean per variable")
    return all(any(assignment[abs(lit) - 1] == (lit > 0)
                   for lit in clause) for clause in clauses)

def find_assignment(n, clauses):
    validate_formula(n, clauses)
    for assignment in product((False, True), repeat=n):
        if satisfies(n, clauses, assignment):
            return list(assignment)
    return None
`;
const graphCore = `from itertools import combinations

def simple_graph(n, edges):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative integer")
    seen = set()
    result = []
    for u, v in edges:
        if type(u) is not int or type(v) is not int or not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("Distinct endpoints in 0..n-1 required")
        pair = (min(u, v), max(u, v))
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result

def covers(edges, chosen):
    selected = set(chosen)
    return all(u in selected or v in selected for u, v in edges)
`;
export const intractabilityExamples = {
  certificate: {
    title: 'Check one certificate, then search the finite assignment space',
    setup: 'Python 3.12+, standard library. A signed integer literal -2 means not x2. Empty conjunction is true; an empty clause is false.',
    code: `${satCore}
formula = [[1, 2, 3], [-1, 2, -3], [1, -2, -3]]
candidate = [False, False, False]
print("candidate:", satisfies(3, formula, candidate))
print("witness:", find_assignment(3, formula))
print("contradiction:", find_assignment(1, [[1], [-1]]))
print("empty formula:", find_assignment(0, []))
print("empty clause:", find_assignment(0, [[]]))`,
    expected: 'candidate: False\nwitness: [False, False, True]\ncontradiction: None\nempty formula: []\nempty clause: None',
  },
  clique: {
    title: 'Convert clauses to literal occurrences and recover an assignment',
    setup: 'Python 3.12+, standard library. Clauses have at most three literals; repeated literals are separate occurrences. The tiny clique oracle is exhaustive.',
    code: `${satCore}
def to_clique(n, clauses):
    validate_formula(n, clauses)
    if any(len(clause) > 3 for clause in clauses):
        raise ValueError("At most three literals per clause")
    vertices = [(c, j, lit) for c, clause in enumerate(clauses)
                for j, lit in enumerate(clause)]
    edges = {(i, j) for i, j in combinations(range(len(vertices)), 2)
             if vertices[i][0] != vertices[j][0]
             and vertices[i][2] != -vertices[j][2]}
    return vertices, edges, len(clauses)

def clique_witness(vertex_count, edges, k):
    for chosen in combinations(range(vertex_count), k):
        if all((i, j) in edges for i, j in combinations(chosen, 2)):
            return list(chosen)
    return None

def recover_assignment(n, clauses, chosen):
    vertices, edges, k = to_clique(n, clauses)
    if chosen is None or len(chosen) != k or len(set(chosen)) != k:
        raise ValueError("One k-clique witness required")
    if any(type(i) is not int or not 0 <= i < len(vertices) for i in chosen):
        raise ValueError("Invalid occurrence ID")
    if any(tuple(sorted(pair)) not in edges for pair in combinations(chosen, 2)):
        raise ValueError("Selected occurrences are not a clique")
    assignment = [False] * n
    for i in chosen:
        literal = vertices[i][2]
        assignment[abs(literal) - 1] = literal > 0
    assert satisfies(n, clauses, assignment)
    return assignment

formula = [[1, 2, 3], [-1, 2, -3], [1, -2, -3]]
vertices, edges, k = to_clique(3, formula)
chosen = clique_witness(len(vertices), edges, k)
print("occurrences, target:", len(vertices), k)
print("chosen IDs:", chosen)
print("chosen literals:", [vertices[i][2] for i in chosen])
print("assignment:", recover_assignment(3, formula, chosen))
v, e, k = to_clique(1, [[1], [-1]])
print("contradictory clique:", clique_witness(len(v), e, k))`,
    expected: 'occurrences, target: 9 3\nchosen IDs: [0, 4, 6]\nchosen literals: [1, 2, 1]\nassignment: [True, True, False]\ncontradictory clique: None',
  },
  complement: {
    title: 'The same subset becomes independent in the complement',
    setup: 'Python 3.12+. Simple undirected graph; duplicate orientations are collapsed. Self-loops are rejected because the stated complement identities use distinct vertex pairs.',
    code: `${graphCore}
def complement(n, edges):
    original = set(simple_graph(n, edges))
    return [pair for pair in combinations(range(n), 2) if pair not in original]

n = 4
edges = simple_graph(n, [(0, 1), (0, 2), (1, 2), (2, 3)])
clique = [0, 1, 2]
other_edges = complement(n, edges)
remaining = [u for u in range(n) if u not in clique]
print("complement edges:", other_edges)
print("independent:", not any(u in clique and v in clique for u, v in other_edges))
print("cover:", remaining, covers(other_edges, remaining))`,
    expected: 'complement edges: [(0, 3), (1, 3)]\nindependent: True\ncover: [3] True',
  },
  selfReduction: {
    title: 'Recover a SAT witness using only yes/no SAT questions',
    setup: 'Python 3.12+. The supplied decision oracle is exhaustive, so this executable demonstration is not a polynomial-time SAT solver.',
    code: `${satCore}
def decision_oracle(n, clauses):
    return find_assignment(n, clauses) is not None

def search_with_oracle(n, clauses, oracle):
    validate_formula(n, clauses)
    current = [list(clause) for clause in clauses]
    questions = 1
    if not oracle(n, current):
        return None, questions
    assignment = []
    for variable in range(1, n + 1):
        questions += 1
        false_possible = oracle(n, current + [[-variable]])
        assignment.append(not false_possible)
        current.append([-variable] if false_possible else [variable])
    assert satisfies(n, clauses, assignment)
    return assignment, questions

print(search_with_oracle(3, [[1, 2], [-1, 3], [-2, -3]], decision_oracle))
print(search_with_oracle(1, [[1], [-1]], decision_oracle))`,
    expected: '([False, True, False], 4)\n(None, 1)',
  },
  threeCnf: {
    title: 'Split long clauses using fresh variables',
    setup: 'Python 3.12+. Produces clauses of length at most three. Auxiliary variables are existentially chosen; they are not arbitrary fixed constants.',
    code: `${satCore}
def at_most_three_cnf(n, clauses):
    validate_formula(n, clauses)
    next_variable = n
    result = []
    for clause in clauses:
        remaining = list(clause)
        while len(remaining) > 3:
            next_variable += 1
            auxiliary = next_variable
            result.append([remaining[0], remaining[1], auxiliary])
            remaining = [-auxiliary] + remaining[2:]
        result.append(remaining)
    return next_variable, result

count, formula = at_most_three_cnf(5, [[1, 2, 3, 4, 5]])
print("variables:", count)
print("clauses:", formula)
original = [False, False, False, False, True]
extensions = [list(extra) for extra in product((False, True), repeat=count - 5)
              if satisfies(count, formula, original + list(extra))]
print("valid extensions:", extensions)`,
    expected: 'variables: 7\nclauses: [[1, 2, 6], [-6, 3, 7], [-7, 4, 5]]\nvalid extensions: [[True, True]]',
  },
  subset: {
    title: 'Return subset indices while counting numerical DP states',
    setup: 'Python 3.12+. Nonnegative integer values and target. Each item can be used once; duplicate values keep distinct input indices. O(target) allocation is real.',
    code: `def subset_sum(values, target):
    if type(target) is not int or target < 0 or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Nonnegative integers required")
    reachable = [False] * (target + 1)
    parent = [None] * (target + 1)
    reachable[0] = True
    for index, value in enumerate(values):
        for total in range(target, value - 1, -1):
            if not reachable[total] and reachable[total - value]:
                reachable[total] = True
                parent[total] = (total - value, index)
    if not reachable[target]:
        return None
    chosen = []
    total = target
    while total:
        total, index = parent[total]
        chosen.append(index)
    return chosen[::-1]

values = [4, 7, 9, 12]
chosen = subset_sum(values, 16)
print("indices:", chosen)
print("values:", [values[i] for i in chosen])
print("absent:", subset_sum([4, 7], 6))
print("zero target:", subset_sum([0, 5], 0))
for exponent in [10, 20, 30]:
    target = 2 ** exponent
    print("target bits, DP slots:", target.bit_length(), target + 1)`,
    expected: 'indices: [1, 2]\nvalues: [7, 9]\nabsent: None\nzero target: []\ntarget bits, DP slots: 11 1025\ntarget bits, DP slots: 21 1048577\ntarget bits, DP slots: 31 1073741825',
  },
  parameter: {
    title: 'Branch on an uncovered edge using only the remaining cover budget',
    setup: 'Python 3.12+. Simple undirected unweighted vertex cover; budget k is nonnegative. Returns one cover of size at most k or None.',
    code: `${graphCore}
def bounded_cover(n, edges, k):
    edges = simple_graph(n, edges)
    if type(k) is not int or k < 0:
        raise ValueError("Nonnegative budget required")
    calls = 0
    def visit(remaining, left):
        nonlocal calls
        calls += 1
        if not remaining:
            return []
        if left == 0:
            return None
        u, v = remaining[0]
        for chosen in (u, v):
            rest = [edge for edge in remaining if chosen not in edge]
            answer = visit(rest, left - 1)
            if answer is not None:
                return [chosen] + answer
        return None
    answer = visit(edges, min(k, n))
    return answer, calls

triangle = [(0, 1), (0, 2), (1, 2)]
print("triangle budget 1:", bounded_cover(3, triangle, 1))
print("triangle budget 2:", bounded_cover(3, triangle, 2))
print("star budget 1:", bounded_cover(5, [(0, 1), (0, 2), (0, 3), (0, 4)], 1))
print("empty budget 0:", bounded_cover(0, [], 0))`,
    expected: 'triangle budget 1: (None, 3)\ntriangle budget 2: ([0, 1], 3)\nstar budget 1: ([0], 2)\nempty budget 0: ([], 1)',
  },
  approximation: {
    title: 'Return a feasible cover and a matching lower-bound certificate',
    setup: 'Python 3.12+. Unweighted simple undirected graph. This is a 2-approximation, not an exact solver; a maximal matching suffices.',
    code: `${graphCore}
def matching_cover(n, edges):
    edges = simple_graph(n, edges)
    chosen = set()
    matching = []
    for u, v in edges:
        if u not in chosen and v not in chosen:
            chosen.update((u, v))
            matching.append((u, v))
    return sorted(chosen), matching

edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
cover, matching = matching_cover(5, edges)
print("cover:", cover)
print("matching:", matching)
print("feasible:", covers(edges, cover))
print("optimum interval:", (len(matching), len(cover)))
print("empty:", matching_cover(0, []))`,
    expected: 'cover: [0, 1]\nmatching: [(0, 1)]\nfeasible: True\noptimum interval: (1, 2)\nempty: ([], [])',
  },
  path: {
    title: 'Use path structure for an exact weighted independent set',
    setup: 'Python 3.12+. Vertices are array positions; only consecutive positions conflict. Integer weights may be negative, and selecting nothing is allowed.',
    code: `def path_independent_set(weights):
    if any(type(weight) is not int for weight in weights):
        raise ValueError("Integer weights required")
    n = len(weights)
    best = [0] * (n + 1)
    take = [False] * n
    for count in range(1, n + 1):
        include = weights[count - 1] + best[max(0, count - 2)]
        exclude = best[count - 1]
        take[count - 1] = include > exclude
        best[count] = max(include, exclude)
    chosen = []
    count = n
    while count:
        if take[count - 1]:
            chosen.append(count - 1)
            count = max(0, count - 2)
        else:
            count -= 1
    return best[n], chosen[::-1]

print(path_independent_set([5, 1, 6, 8, 4]))
print(path_independent_set([-4, -1]))
print(path_independent_set([]))`,
    expected: '(15, [0, 2, 4])\n(0, [])\n(0, [])',
  },
};
