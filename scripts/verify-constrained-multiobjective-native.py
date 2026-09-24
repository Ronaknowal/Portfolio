import contextlib
import io
import itertools
import json
import math
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

payload = json.load(open(sys.argv[1], encoding='utf-8'))
cases, examples = payload['cases'], payload['examples']
namespaces = {}
for key, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'], key, 'exec'), namespace)
    namespaces[key] = namespace

def close(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-9)

def scalar_projection(target, budget):
    # A scalar bounded minimizer plus explicit endpoints, independent of the formula.
    objective = lambda t: ((t-target[0])**2 + (budget-t-target[1])**2) / 2
    result = minimize_scalar(objective, bounds=(0, budget), method='bounded',
                             options={'xatol': 1e-13})
    first = min([0, budget, result.x], key=objective)
    return np.array([first, budget-first])

for state in cases['projections']:
    target, budget = np.array(state['target']), state['budget']
    expected = scalar_projection(target, budget)
    close(state['exact'], expected, 1e-7)
    close(namespaces['projection']['simplex_projection'](target, budget), expected, 1e-7)
    p = np.array(state['exact'])
    for t in np.linspace(0, budget, 41):
        assert np.dot(target-p, np.array([t,budget-t])-p) <= 1e-9
    assert min(p) >= 0
    close(sum(p), budget)
    close(state['exactDistanceSquared'], np.sum((p-target)**2))
    first, second = state['frames'][1:]
    for frame in state['frames']:
        point = np.array(frame['point'])
        close(frame['nonnegativeViolation'], max(0, -min(point)))
        close(frame['equalityResidual'], point.sum()-budget)
        close(frame['distanceSquared'], np.sum((point-target)**2))
    if state['order'] == 'orthant-first':
        close(first['point'], np.maximum(target, 0))
        close(sum(second['point']), budget)
    else:
        close(sum(first['point']), budget)
        assert min(second['point']) >= 0

for state in cases['penalties']:
    mode, strength = state['mode'], state['strength']
    def objective(x):
        cost = (x-3)**2/2
        if mode == 'quadratic': return cost + strength * max(0,x-1)**2/2
        if mode == 'hinge': return cost + strength * max(0,x-1)
        return cost - strength * np.log(1-x)
    upper = 1-1e-12 if mode == 'barrier' else 4
    result = minimize_scalar(objective, bounds=(-4, upper), method='bounded',
                             options={'xatol': 1e-13})
    candidates = [result.x] + ([1] if mode != 'barrier' else [])
    expected = min(candidates, key=objective)
    close(state['optimum'], expected, 2e-7)
    close(state['modifiedCost'], objective(state['optimum']))

frame_count = 0
for state in cases['admm']:
    c, budget, rho = np.array(state['target']), state['budget'], state['rho']
    oracle = scalar_projection(c, budget)
    close(state['exact'], oracle, 1e-7)
    previous = state['frames'][0]
    assert previous['dualNorm'] is None and not previous['stoppingPassed']
    for frame in state['frames'][1:]:
        x,z,u = (np.array(frame[key]) for key in ['x','z','u'])
        old_z, old_u = np.array(previous['z']), np.array(previous['u'])
        # Check each subproblem's first-order conditions, not only recurrence agreement.
        derivative = x-c + rho*(x-old_z+old_u)
        assert np.all(derivative[x == 0] >= -1e-10)
        close(derivative[x > 0], np.zeros(np.count_nonzero(x > 0)))
        z_derivative = z-x-old_u
        close(z_derivative[0], z_derivative[1])
        close(z.sum(), budget)
        close(frame['primalResidual'], x-z)
        close(frame['dualResidual'], -rho*(z-old_z))
        close(u, old_u+x-z)
        close(frame['primalNorm'], np.linalg.norm(x-z))
        close(frame['dualNorm'], rho*np.linalg.norm(z-old_z))
        assert frame['stoppingPassed'] == (frame['primalNorm'] <= frame['primalTolerance'] and frame['dualNorm'] <= frame['dualTolerance'])
        repair = np.array(frame['repair'])
        assert min(repair) >= 0
        close(sum(repair), budget)
        assert frame['repairedCost'] >= state['exactCost'] - 1e-12
        previous = frame
        frame_count += 1
    history = namespaces['admm']['consensus'](c, budget, rho, 3000, 1e-9, 1e-8)
    assert history[-1][-1], (c,budget,rho)
    close(history[-1][1], oracle, 1e-6)
    for row, js in zip(history, state['frames'][1:]):
        close(row[1], js['x'])
        close(row[2], js['z'])
        close(row[3], js['u'])

for state in cases['decisions']:
    feasible = [item for item in state['candidates'] if item['latencyMs'] <= state['latencyLimit'] and item['memoryMb'] <= state['memoryLimit']]
    # Sort by one coordinate and scan the best preceding error for this fixed 2-D table.
    expected_front = []
    best_error = math.inf
    for item in sorted(feasible, key=lambda row: (row['latencyMs'], row['error'])):
        if item['error'] < best_error:
            expected_front.append(item['id'])
            best_error = item['error']
    assert set(state['front']) == set(expected_front)
    if not feasible:
        assert state['selected'] is None
        continue
    key = 'error' if state['method'] == 'error' else 'latencyMs' if state['method'] == 'latency' else 'score'
    scores = {item['id']: (100*item['error'] + state['price']*item['latencyMs'] if key == 'score' else item[key]) for item in feasible}
    best = min(scores.values())
    ties = [item for item in feasible if abs(scores[item['id']]-best) <= 1e-10]
    expected = min(ties, key=lambda row: (row['error'],row['latencyMs'],row['id']))
    assert state['selected'] == expected['id']
    assert set(state['ties']) == {item['id'] for item in ties}

for state in cases['continuous']:
    if state['method'] == 'weighted':
        a = state['alpha']
        objective = lambda x: a*x*x+(1-a)*(x-2)**2
        result = minimize_scalar(objective, bounds=(0,2), method='bounded')
        expected = min([0,2,result.x], key=objective)
    else:
        result = minimize(lambda x: x[0]**2, [state['optimum']], bounds=[(0,2)],
                          constraints=[{'type':'ineq', 'fun':lambda x:state['epsilon']-(x[0]-2)**2}],
                          method='SLSQP', options={'ftol':1e-12})
        expected = result.x[0]
        assert state['objectives'][1] <= state['epsilon'] + 1e-10
    close(state['optimum'], expected, 1e-6)
    close(state['objectives'], [state['optimum']**2,(state['optimum']-2)**2])
for state in cases['units']:
    close(state['ms'], state['seconds'])
    close(state['wrong'], 100*state['item']['error'] + state['price']*state['item']['latencyMs']/1000)

rng = np.random.default_rng(4201)
native_projection = 0
for _ in range(120):
    target = rng.uniform(-8,8,2)
    budget = rng.uniform(0.1,7)
    close(namespaces['projection']['simplex_projection'](target,budget), scalar_projection(target,budget), 3e-7)
    native_projection += 1
native_penalties = 0
for _ in range(80):
    bound = rng.uniform(-3,3)
    center = bound + rng.uniform(0.2,4)
    strength = rng.uniform(0.01,7)
    values = namespaces['penalties']['softened_optima'](center,bound,strength)
    for index, x in enumerate(values):
        if index == 0:
            close(x-center+strength*max(0,x-bound), 0)
        elif index == 1:
            if abs(x-bound) < 1e-10: assert strength >= center-bound-1e-10
            else: close(x-center+strength, 0)
        else:
            assert x < bound
            close(x-center+strength/(bound-x),0)
    native_penalties += 1

native_pareto = 0
for width in [1,2,3,5]:
    for _ in range(50):
        points = rng.integers(-3,4,(12,width))
        mask = np.all(points[:,None,:] <= points[None,:,:],axis=2) & np.any(points[:,None,:] < points[None,:,:],axis=2)
        expected = np.flatnonzero(~np.any(mask,axis=0)).tolist()
        assert namespaces['pareto']['pareto_indices'](points.tolist()) == expected
        native_pareto += 1
directions = 0
for dimension in [1,2,3,8]:
    for _ in range(40):
        first,second = rng.normal(size=(2,dimension))
        alpha,d = namespaces['commonDescent']['common_direction'](first,second)
        objective = lambda a: np.linalg.norm(a*first+(1-a)*second)**2
        result = minimize_scalar(objective,bounds=(0,1),method='bounded')
        expected = min([0,1,result.x],key=objective)
        close(objective(alpha),objective(expected),1e-8)
        assert first@d <= -d@d+1e-9 and second@d <= -d@d+1e-9
        directions += 1

# Changed exercise results are calculated outside the browser's model fixtures.
close(scalar_projection([0.5,3],2),[0,2])
close(namespaces['penalties']['softened_optima'](4,1,4)[2],0)
changed = namespaces['admm']['consensus']([1.5,-0.5],max_steps=2)
close(changed[0][1],[1,0]); close(changed[0][4],0); close(changed[0][5],math.sqrt(0.5))
close(changed[1][1],[1.25,0]); close(changed[1][2],[1.125,-0.125])
close(0.08+2*0.018,0.116)
close(namespaces['continuous']['weighted_choice'](0.75),0.5)
close(namespaces['continuous']['epsilon_choice'](2.25),0.5)
assert namespaces['measurement']['nearest_rank']([10]*98+[200]*2,0.99)==200

print(json.dumps({'projectionOracleCases':len(cases['projections']), 'admmSubproblemFrames':frame_count,
                  'nativeProjectionChanged':native_projection,'nativePenaltyChanged':native_penalties,
                  'nativeParetoChanged':native_pareto,'nativeCommonDescentChanged':directions,
                  'changedPracticeChecks':7,'numpy':np.__version__}))
