"""Root review: operator-level, homomorphism and changed native contracts."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from datetime import datetime, timezone
import contextlib
import hashlib
import io
import json
import math
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/abstract-algebra-independent'
data = json.loads((DIRECTORY / 'payload.json').read_text(encoding='utf8'))
counts = {}

def check(condition, label):
    assert condition, label
    counts[label] = counts.get(label, 0) + 1

def exact(rows):
    return sp.Matrix([[sp.Rational(str(value)) for value in row] for row in rows])

namespaces = {}
for example in data['examples']:
    namespace, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id'], 'exec'), namespace)
    check(output.getvalue().strip() == example['expected'], 'actual_program_stdout')
    namespaces[example['id']] = namespace

# Independent D4 representation as affine permutations, evaluated at all vertices.
permutations = [tuple((offset + direction * vertex) % 4 for vertex in range(4)) for direction in [1, -1] for offset in range(4)]
def comp(g, h):
    return permutations.index(tuple(permutations[g][permutations[h][i]] for i in range(4)))

# The full averaging operator acts on the sixteen matrix entries. Its symmetric
# idempotence and rank certify the claimed orthogonal projection as an operator.
for record in data['projections']:
    columns = [sp.Matrix([sp.Rational(str(v)) for v in column]) for column in record['columns']]
    operator = sp.Matrix.hstack(*columns)
    rank = 4 if record['rotationsOnly'] else 3
    check(operator == operator.T and operator * operator == operator, 'full_orthogonal_projection_operator')
    check(operator.rank() == rank and operator.trace() == rank, 'projection_rank_and_trace')
    check(operator.eigenvals() == {sp.Integer(0): 16-rank, sp.Integer(1): rank}, 'projection_eigenvalue_multiplicities')
    def relation(i, j):
        distance = (j-i) % 4
        return distance if record['rotationsOnly'] else min(distance, 4-distance)
    for record_matrix in record['changed']:
        matrix = exact(record_matrix['matrix'])
        expected = sp.Matrix(4, 4, lambda i, j: sum(matrix[a,b] for a,b in product(range(4), repeat=2) if relation(a,b)==relation(i,j)) / sum(relation(a,b)==relation(i,j) for a,b in product(range(4), repeat=2)))
        actual = exact(record_matrix['averaged'])
        check(actual == expected, 'changed_pair_relation_averages')
        check(operator * sp.Matrix(list(matrix)) == sp.Matrix(list(actual)), 'changed_operator_application')
        native = namespaces['averaging']['averaged']([[F(value) for value in row] for row in matrix.tolist()], range(4 if record['rotationsOnly'] else 8))
        check(sp.Matrix(native) == expected, 'changed_actual_native_average')

# Classify every homomorphism into C2 and then all subgroup quotients. Normality
# is checked from conjugation; representative independence is separately checked.
homomorphism_kernels = set()
for images in product(range(2), repeat=8):
    if all(images[comp(g,h)] == (images[g]+images[h]) % 2 for g,h in product(range(8), repeat=2)):
        homomorphism_kernels.add(frozenset(g for g in range(8) if images[g] == 0))
check(len(homomorphism_kernels) == 4, 'all_C2_homomorphisms')
normal_count = 0
for record in data['subgroups']:
    H = set(record['subgroup'])
    inverse = [next(h for h in range(8) if comp(g,h)==comp(h,g)==0) for g in range(8)]
    normal = all(comp(comp(g,h),inverse[g]) in H for g in range(8) for h in H)
    cosets = [set(row['members']) for row in record['cosets']]
    class_of = {g: index for index, members in enumerate(cosets) for g in members}
    well_defined = all(len({class_of[comp(g,h)] for g in A for h in B})==1 for A in cosets for B in cosets)
    check(normal == well_defined, 'every_subgroup_quotient_contract')
    check(namespaces['quotients']['representative_independent'](H) == normal, 'changed_actual_native_quotient')
    normal_count += normal
check(normal_count == 6, 'normal_subgroup_count')
check(all(any(set(record['subgroup']) == set(kernel) for record in data['subgroups']) for kernel in homomorphism_kernels), 'kernels_have_subgroup_owner')

for record, group_size in zip(data['fourColorCounts'], [8, 4]):
    classes = namespaces['orbits']['orbits'](4, range(group_size))
    check(len(classes) == record['orbitCount'], 'changed_four_color_native_enumeration')
    check(sum(map(len, classes)) == 256, 'changed_four_color_partition')

for state in data['states']:
    matrix, x = exact(state['matrix']), sp.Matrix([sp.Rational(str(value)) for value in state['values']])
    p = permutations[state['g']]
    P = sp.Matrix(4,4,lambda j,i: int(p[i] == j))
    upper, lower = matrix*P*x, P*matrix*x
    vector = lambda values: sp.Matrix([sp.Rational(str(value)) for value in values])
    check(upper == vector(state['transformThenMap']) and lower == vector(state['mapThenTransform']), 'changed_exact_sensor_routes')
    check(max(abs(v) for v in upper-lower) == sp.Rational(str(state['inputSpecificDefect'])), 'changed_exact_sensor_defect')
    check(sum(upper)/4 == sp.Rational(str(state['poolAfter'])) and sum(matrix*x)/4 == sp.Rational(str(state['poolBefore'])), 'changed_exact_sensor_pooled_values')

solve = namespaces['modular']['solve_congruence']
for n in [13, 25, 64, 97, 100]:
    for a,b in [(-14,7),(0,0),(0,1),(15,30),(12,7),(51,17)]:
        solutions = solve(a,b,n)
        d = math.gcd(a,n)
        check(len(solutions) == (d if b%d == 0 else 0), 'changed_native_congruence_solution_count')
        check(all((a*x-b)%n == 0 for x in solutions) and len(set(solutions)) == len(solutions), 'changed_native_congruence_witnesses')

for source in data['sources']:
    check(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256'], 'final_source_unchanged')
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'reviewer': 'root', 'sources': data['sources'], 'authorPacket': data['authorPacket'], 'checks': counts, 'limits': ['Complementary exact finite examples and complete production projection operators support the source-read proofs; they do not prove arbitrary infinite-group statements or empirical learning outcomes.']}
(DIRECTORY/'native-results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
print(json.dumps({'status': result['status'], 'checkedAt': result['checkedAt'], 'checks': counts}, indent=2))
