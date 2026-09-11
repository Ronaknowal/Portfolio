"""Independent permutation/matrix/enumeration oracles for actual production exports."""

from pathlib import Path
from fractions import Fraction as F
from itertools import permutations, product
from datetime import datetime, timezone
import contextlib
import io
import json
import math
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/abstract-algebra-verification'
data = json.loads((DIRECTORY/'payload.json').read_text())
counts = {}
exact_matrix = lambda rows: sp.Matrix([[sp.Rational(str(v)) for v in row] for row in rows])
exact_vector = lambda values: sp.Matrix([sp.Rational(str(v)) for v in values])


def check(condition, label):
    assert condition, label
    counts[label] = counts.get(label, 0) + 1


namespaces = {}
for example in data['examples']:
    namespace, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id'], 'exec'), namespace)
    check(output.getvalue().strip() == example['expected'], 'actual_program_stdout')
    namespaces[example['id']] = namespace

# Independent group model: exact geometric matrices, not the encoded pair law.
R, S = sp.Matrix([[0,-1],[1,0]]), sp.diag(1,-1)
matrices = [R**k * S**b for b in range(2) for k in range(4)]
vertices = [sp.Matrix([1,0]),sp.Matrix([0,1]),sp.Matrix([-1,0]),sp.Matrix([0,-1])]
perms = [tuple(vertices.index(matrix*point) for point in vertices) for matrix in matrices]
Ps = [sp.Matrix(4,4,lambda j,i:int(p[i]==j)) for p in perms]
def comp(g,h): return matrices.index(matrices[g]*matrices[h])
def move(g,state): return tuple(state[perms[g].index(j)] for j in range(4))

distance = lambda i,j: (vertices[i]-vertices[j]).dot(vertices[i]-vertices[j])
geometric = {p for p in permutations(range(4)) if all(distance(i,j)==distance(p[i],p[j]) for i,j in product(range(4),repeat=2))}
check(geometric==set(perms), 'independent_distance_symmetries')
for g in range(8):
    check(data['permutations'][g]==list(perms[g]), 'geometric_vertex_maps')
    check(data['matrices'][g]==matrices[g].tolist(), 'geometric_representation')
    check(matrices[data['inverse'][g]]==matrices[g].inv(), 'exact_inverse')
for row in data['compositions']:
    g,h,v=row['g'],row['h'],row['vertex']
    check(row['product']==comp(g,h) and row['reversed']==comp(h,g), 'actual_composition_matrix_oracle')
    check(row['firstRoute']==[v,perms[h][v],perms[comp(g,h)][v]], 'actual_tracked_route')
    check(row['firstStates'][-1]==list(move(comp(g,h),'ABCD')), 'label_source_destination')

# Solve subgroup membership by independent permutation closure, then derive cosets.
valid_subgroups=[]
for row in data['subgroups']:
    H=set(row['subset'])
    valid=bool(H) and 0 in H and all(comp(g,h) in H for g in H for h in H)
    check((row['cosets'] is not None)==valid, 'all_256_subset_classifications')
    generated={0}
    while True:
        expanded=generated|{comp(g,h) for g in H for h in generated}
        if generated==expanded: break
        generated=expanded
    check(set(row['generated'])==generated, 'changed_generated_subgroups')
    if valid:
        valid_subgroups.append(H)
        expected={frozenset(comp(g,h) for h in H) for g in range(8)}
        actual={frozenset(c['members']) for c in row['cosets']}
        check(actual==expected, 'all_subgroup_cosets')
check(len(valid_subgroups)==10, 'D4_subgroup_count')

for row in data['orbits']:
    x=tuple(row['colors']); group=row['group']
    expected={move(g,x) for g in group}
    check({tuple(s['colors']) for s in row['states']}==expected, 'all_ternary_orbits')
    stabilizer={g for g in group if move(g,x)==x}
    check(set(row['stabilizer'])==stabilizer, 'all_ternary_stabilizers')
    check(len(expected)*len(stabilizer)==len(group), 'orbit_stabilizer_identity')
    for state in row['states']:
        check(set(state['transformations'])=={g for g in group if move(g,x)==tuple(state['colors'])}, 'exact_reaching_transformations')
for row in data['fixed']:
    states=list(product(range(row['colorCount']),repeat=4)); group=row['group']
    for entry in row['rows']:
        check(entry['fixed']==sum(move(entry['g'],state)==state for state in states), 'fixed_counts_by_enumeration')
    classes={min(move(g,state) for g in group) for state in states}
    check(row['orbitCount']==len(classes), 'Burnside_vs_full_enumeration')

# Matrix orbit averages directly pool three geometric relation classes, rather than
# computing conjugates again. Also independently solve the commutant equations.
symbols=sp.symbols('w0:16'); symbolic=sp.Matrix(4,4,symbols)
equations=[]
for p in [Ps[1],Ps[4]]: equations.extend(list(symbolic*p-p*symbolic))
constraint,_=sp.linear_eq_to_matrix(equations,symbols)
check(len(constraint.nullspace())==3,'independent_commutant_dimension')
for row in data['averages']:
    W=sp.Matrix([[sp.Rational(str(v)) for v in line] for line in row['matrix']])
    coefficient={kind:sum(W[i,j] for i in range(4) for j in range(4) if (min((j-i)%4,(i-j)%4))==kind)/sum(min((j-i)%4,(i-j)%4)==kind for i in range(4) for j in range(4)) for kind in range(3)}
    full=sp.Matrix(4,4,lambda i,j:coefficient[min((j-i)%4,(i-j)%4)])
    offsets={k:sum(W[i,(i+k)%4] for i in range(4))/4 for k in range(4)}
    rotation=sp.Matrix(4,4,lambda i,j:offsets[(j-i)%4])
    check(exact_matrix(row['full'])==full,'changed_average_relation_oracle')
    check(exact_matrix(row['rotations'])==rotation,'changed_rotation_offset_oracle')
    T=sp.Matrix(4,4,lambda i,j:[2,3,5,3][(j-i)%4])
    norm=lambda A:sum(v*v for v in A)
    check(norm(W-T)==norm(W-full)+norm(full-T),'exact_projection_pythagoras')
    actual=namespaces['averaging']['averaged']([[F(str(v)) for v in line] for line in row['matrix']])
    check(sp.Matrix(actual)==full,'actual_native_changed_averages')

for row in data['equivariance']:
    W=exact_matrix(row['matrix']); x=exact_vector(row['values']); p=Ps[row['g']]
    check(W*p*x==exact_vector(row['transformThenMap']),'actual_two_route_matrix_oracle')
    check(p*W*x==exact_vector(row['mapThenTransform']),'actual_two_route_matrix_oracle')
    check(float(max(abs(v) for v in W*p*x-p*W*x))==row['inputSpecificDefect'],'actual_input_defect')
    defects=[max(abs(v) for v in W*q-q*W) for q in Ps]
    check(float(max(defects))==row['allInputDefect'],'actual_full_matrix_certificate')
    check(float(sum(W*x)/4)==row['poolBefore'] and float(sum(W*p*x)/4)==row['poolAfter'],'actual_input_to_pooled_output')

for row in data['modular']:
    n,a,b=row['modulus'],row['multiplier'],row['target']
    solutions=[x for x in range(n) if (a*x-b)%n==0]
    check(row['solutions']==solutions,'all_bounded_modular_solutions')
    check(row['injective']==(math.gcd(a,n)==1),'unit_gcd_oracle')

# Execute the specifically changed independent practice capstone.
capstone=next(example for example in data['examples'] if example['id']=='capstone')['code']
assert '[2, 2, 4, 8]' in capstone and 'x = (1, 2, 4, 8)' in capstone
changed=capstone.replace('[2, 2, 4, 8]','[6, 2, 4, 8]').replace('x = (1, 2, 4, 8)','x = (-1, 0, 2, 3)')
namespace,output={},io.StringIO()
with contextlib.redirect_stdout(output): exec(changed,namespace)
check(namespace['matvec'](namespace['A'],namespace['x'])==(F(133,4),F(67,2),F(37),F(149,4)),'actual_changed_capstone_output')
check(namespace['means']=={F(141,4)},'actual_changed_capstone_invariant_mean')
check(move(4,(0,0,1,2)) not in {move(g,(0,0,1,2)) for g in range(4)},'ternary_chiral_practice')
check(namespaces['modular']['solve_congruence'](3,4,8)==[4],'changed_modular_practice')
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'status':'passed','productionSources':data['productionSources'],'checks':counts,'invalidModelCases':data['rejectionCount'],'changedCapstoneStdout':output.getvalue().strip(),'environment':{'sympy':sp.__version__},'limits':'Finite exact arithmetic, enumeration and separately formulated matrix oracles check actual production exports; mathematical proofs and actual-font browser reading are recorded separately.'}
(DIRECTORY/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
