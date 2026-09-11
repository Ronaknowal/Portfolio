"""Complementary reviewer checks: LP minima, augmented operators, exact stored inputs."""
import contextlib
import hashlib
import io
import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
import sympy as sp

folder = Path('scratch/conditioning-independent-review')
fixture = json.loads((folder/'fixtures.json').read_text(encoding='utf-8'))
counts = Counter()
max_difference = 0.0


def check_close(actual, expected, name, tolerance=2e-10):
    global max_difference
    difference = abs(float(actual)-float(expected)) / max(1, abs(float(expected)))
    assert difference <= tolerance, (name, actual, expected, difference)
    max_difference = max(max_difference, difference)
    counts[name] += 1


namespaces = {}
for example in fixture['examples']:
    output, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id'], 'exec'), namespace)
    assert output.getvalue().strip() == example['expected'].strip(), example['id']
    namespaces[example['id']] = namespace
    counts['actual_complete_programs'] += 1


def optimal_backward_error(a, b, x, componentwise):
    """Minimize eta over actual perturbations, without using the quotient formula."""
    a, b, x = np.array(a, float), np.array(b, float), np.array(x, float)
    m, n = a.shape
    entries = m*n
    # Variables are dA, db, nonnegative |dA| auxiliaries, and eta.
    length, eta = 2*entries+m+1, 2*entries+m
    objective = np.zeros(length)
    objective[eta] = 1
    equality = np.zeros((m, length))
    for i in range(m):
        equality[i, i*n:(i+1)*n] = x
        equality[i, entries+i] = -1
    constraints = []
    for i in range(m):
        for j in range(n):
            index = i*n+j
            for sign in [-1, 1]:
                row = np.zeros(length)
                row[index] = sign
                if componentwise:
                    row[eta] = -abs(a[i,j])
                else:
                    row[entries+m+index] = -1
                constraints.append(row)
        if not componentwise:
            row = np.zeros(length)
            row[entries+m+i*n:entries+m+(i+1)*n] = 1
            row[eta] = -np.linalg.norm(a, np.inf)
            constraints.append(row)
        for sign in [-1,1]:
            row = np.zeros(length)
            row[entries+i] = sign
            row[eta] = -abs(b[i]) if componentwise else -np.linalg.norm(b, np.inf)
            constraints.append(row)
    result = linprog(objective, A_ub=constraints, b_ub=np.zeros(len(constraints)),
                     A_eq=equality, b_eq=b-a@x,
                     bounds=[(None,None)]*(entries+m)+[(0,None)]*(entries+1),
                     method='highs', options={'primal_feasibility_tolerance':1e-9})
    assert result.success, result.message
    assert np.max(abs(equality@result.x-(b-a@x))) < 1e-8
    return result.fun


rng = np.random.default_rng(7139)
diagnostics = namespaces['backward']['diagnostics']
for m, n in [(2,2), (2,3), (3,2), (3,3)]:
    for k in range(8):
        a = rng.integers(-4,5,(m,n)).tolist()
        b = rng.integers(-3,4,m).tolist()
        x = rng.integers(-2,3,n).tolist()
        if k == 0: x = [0]*n
        if k == 1: a[0] = [0]*n; b[0] = 0
        if k == 2: a = [[0]*n for _ in range(m)]; b=[0]*m
        _, normwise, componentwise = diagnostics(a,b,x)
        check_close(normwise, optimal_backward_error(a,b,x,False),'native_normwise_lp_minimum')
        check_close(componentwise,optimal_backward_error(a,b,x,True),'native_componentwise_lp_minimum')

# Broader native helper inputs: the oracle is a two-state matrix power, not a loop.
for q in [F(-3,4),F(0),F(2,3),F(5,4)]:
    for n in [0,1,3,7,11]:
        for alternating in [False,True]:
            sign = -1 if alternating else 1
            transform = sp.Matrix([[sp.Rational(q.numerator,q.denominator),1],[0,sign]])
            expected = (transform**n * sp.Matrix([0,sp.Rational(sign,100)]))[0]
            actual,bound = namespaces['propagation']['propagation'](q,n,alternating)
            assert sp.Rational(actual.numerator,actual.denominator) == expected
            assert abs(actual) <= bound
            counts['changed_native_augmented_matrix_power'] += 1
for n in [3,5,7,13,31,63]:
    start,error = namespaces['consistency']['constant_trajectory'](n)
    state = sp.Matrix([[3,-2],[1,0]])**(n-1)*sp.Matrix([sp.Rational(1,n*n),0])
    assert sp.Rational(error.numerator,error.denominator) == state[0]
    counts['changed_native_companion_matrix_power'] += 1

for row in fixture['propagation']:
    q = F(str(row['q']))
    for state in row['state']['frames']:
        n=state['step']
        disturbances=[F(1,100)*(0 if row['mode']=='pulse' and j>0 else (-1)**(j+1) if row['mode']=='alternating' else 1) for j in range(n)]
        signed=sum((q**(n-1-j)*d for j,d in enumerate(disturbances)),F(0))
        envelope=sum((abs(q)**(n-1-j)*abs(d) for j,d in enumerate(disturbances)),F(0))
        check_close(state['error'],signed,'model_disturbance_operator')
        check_close(state['bound'],envelope,'model_absolute_operator')

for row in fixture['stored']:
    assert F(*map(int,row['fraction'])) == F(row['value'])
    counts['binary64_fraction_identity'] += 1
with localcontext() as context:
    context.prec = 180
    for row in fixture['cancellation']:
        exponent,sign,state=row['k'],row['sign'],row['state']
        actual = (Decimal(1)+Decimal(sign)/(Decimal(2)**exponent)).sqrt()-1
        low, high = F(state['referenceLower']),F(state['referenceUpper'])
        decimal = lambda value: Decimal(value.numerator)/Decimal(value.denominator)
        assert decimal(low) <= actual <= decimal(high)
        # Independently check the defining squared inequalities, including x=-1.
        target = F(1)+F(sign,2**exponent)
        assert (low+1)**2 <= target <= (high+1)**2
        counts['reference_polynomial_and_decimal_enclosure'] += 1

sum_methods = namespaces['summation']['sum_methods']
for exponent in [30,40,50,53,60]:
    for middle in [1,3,-1]:
        for values in itertools.permutations([2.0**exponent,middle,-2.0**exponent]):
            outputs = sum_methods(values)
            exact = sum(map(F,values),F(0))
            gamma = 2*F(1,2**53)/(1-2*F(1,2**53))
            assert abs(F(outputs[0])-exact) <= gamma*sum(map(lambda v:abs(F(v)),values))
            assert F(outputs[3]) == exact and F(outputs[4]) == exact
            counts['changed_native_sum_bound_and_recovery'] += 1

# Exercise the real factor/solve helper at odd exponents and every correction count.
for exponent in [9,13,17,21,23]:
    for corrections in [0,1,3,5]:
        exact,states=namespaces['refinement']['refine_two_channel'](exponent,corrections)
        epsilon=2.0**-exponent
        a=np.array([[1,1],[1,1+epsilon]],dtype=float)
        b=a@np.array([1/3,2/3])
        rational_a=sp.Matrix([[sp.Rational(F(v).numerator,F(v).denominator) for v in row] for row in a])
        rational_b=sp.Matrix([sp.Rational(F(v).numerator,F(v).denominator) for v in b])
        independent=rational_a.inv()*rational_b
        assert list(map(lambda v:sp.Rational(v.numerator,v.denominator),exact)) == list(independent)
        for _,iterate,error,residual in states:
            oracle=max(abs(sp.Rational(F(v).numerator,F(v).denominator)-truth) for v,truth in zip(iterate,independent))
            assert sp.Rational(error.numerator,error.denominator)==oracle
        counts['changed_native_refinement_rational_inverse'] += 1

for source in fixture['productionSources']:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'passed':True,'productionSources':fixture['productionSources'],
        'checks':dict(counts),'maxScaledDifference':max_difference,'numpy':np.__version__,'scipy':scipy.__version__,
        'scope':'Complementary finite reviewer oracles; production algorithms untouched. LP feasibility is numerical; rational and symbolic checks are identified separately.'}
(folder/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
