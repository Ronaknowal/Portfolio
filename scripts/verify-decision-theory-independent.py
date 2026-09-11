"""Independent LP, exact arithmetic, and actual-program review of Decision Theory."""

import contextlib
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import linprog

root = Path(__file__).resolve().parents[1]
directory = root / 'scratch/decision-theory-independent'
packet = json.loads((directory / 'fixtures.json').read_text(encoding='utf-8'))
fixtures = packet['fixtures']
counts = {}


def check(actual, expected, label):
    assert math.isclose(float(actual), float(expected), rel_tol=2e-11, abs_tol=2e-11), (label, actual, expected)
    counts[label] = counts.get(label, 0) + 1


def lp_allocation(probabilities, capacity, handling=10, damage=80):
    probabilities = np.array(probabilities, dtype=float)
    marginal = handling - damage * probabilities
    result = linprog(marginal, A_ub=[np.ones(len(probabilities))], b_ub=[capacity], bounds=[(0, 1)] * len(probabilities), method='highs')
    assert result.success
    # Unit capacities and an integer slot budget give an integral polytope.
    assert np.all(np.abs(result.x - np.round(result.x)) < 1e-10)
    return damage * sum(probabilities) + result.fun


namespaces = {}
for key, example in packet['examples'].items():
    output = io.StringIO()
    namespace = {'__name__': '__main__'}
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], f'published:{key}', 'exec'), namespace)
    assert output.getvalue().strip() == example['expected'].strip(), key
    namespaces[key] = namespace
counts['actual_program_stdout'] = len(namespaces)

for case in fixtures['tails']:
    values, masses, alpha = case['values'], case['weights'], case['alpha']
    result = linprog(-np.array(values, dtype=float), A_eq=[np.ones(len(values))], b_eq=[1-alpha], bounds=list(zip(np.zeros(len(values)), masses)), method='highs')
    assert result.success
    expected = -result.fun / (1-alpha)
    check(case['result']['cvar'], expected, 'tail_probability_allocation_lp')
    check(sum(atom['usedMass'] for atom in case['result']['tail']), 1-alpha, 'tail_mass_conservation')
    native, _ = namespaces['utilityTail']['tail_mean'](list(map(F, values)), list(map(lambda x:F(str(x)), masses)), F(str(alpha)))
    check(native, expected, 'changed_actual_tail_helper')
    for scale, offset in [(2, -3)]:
        changed, _ = namespaces['utilityTail']['tail_mean']([scale*F(x)+offset for x in values], [F(str(x)) for x in masses], F(str(alpha)))
        check(changed, scale*expected+offset, 'tail_affine_transfer')
    if alpha > 0:
        positive = sorted((x, F(str(p))) for x,p in zip(values,masses) if p > 0)
        accumulated = F(0)
        for value, mass in positive:
            accumulated += mass
            if accumulated >= F(str(alpha)):
                assert case['result']['valueAtRisk'] == value
                break

for case in fixtures['allocations']:
    expected = lp_allocation(case['probabilities'], case['capacity'], case['handling'], case['damage'])
    check(case['result']['risk'], expected, 'capacity_integral_lp')
    chosen = case['result']['selected']
    assert len(chosen) <= case['capacity'] and len(set(chosen)) == len(chosen)
    risk = sum(case['handling'] if i in chosen else case['damage']*p for i,p in enumerate(case['probabilities']))
    check(risk, expected, 'selected_policy_feasibility')
    native, _ = namespaces['capacity']['best_allocation']([F(str(p)) for p in case['probabilities']],case['capacity'],F(case['handling']),F(case['damage']))
    check(native, expected, 'changed_actual_capacity_helper')

for case in fixtures['information']:
    p, sensitivity, false_positive = (F(str(case[key])) for key in ['p','sensitivity','falsePositive'])
    joint = [[(1-p)*(1-false_positive),p*(1-sensitivity)],[(1-p)*false_positive,p*sensitivity]]
    baseline_release = sum(80*row[1] for row in joint)
    costs = [10*sum(row)-80*row[1] for row in joint]
    optimum = linprog([float(x) for x in costs], bounds=[(0,1)]*2, method='highs')
    assert optimum.success
    after = float(baseline_release) + optimum.fun
    check(case['result']['afterSignal'], after, 'signal_policy_lp')
    assert case['degraded']['value'] <= case['result']['value']+1e-12
    assert 0 <= case['result']['value'] <= case['result']['perfectValue']+1e-12
    with contextlib.redirect_stdout(io.StringIO()):
        _, native_after, _, total = namespaces['information']['test_value'](p,sensitivity,false_positive,F(3,4))
    check(native_after, after, 'changed_actual_information_helper')
    check(total, after+0.75, 'information_price_timing')

payoff=np.array([[0,6],[4,0],[2.5,2.5]],dtype=float)
primal=linprog([0,0,0,1],A_ub=np.column_stack([payoff.T,-np.ones(2)]),b_ub=[0,0],A_eq=[[1,1,1,0]],b_eq=[1],bounds=[(0,None)]*3+[(None,None)],method='highs')
dual=linprog([0,0,-1],A_ub=np.column_stack([-payoff,np.ones(3)]),b_ub=[0,0,0],A_eq=[[1,1,0]],b_eq=[1],bounds=[(0,None)]*2+[(None,None)],method='highs')
assert primal.success and dual.success
check(primal.fun,-dual.fun,'minimax_primal_dual_certificate')
check(primal.fun,F(12,5),'minimax_exact_value')
for case in fixtures['mixtures']:
    risk=np.array([case['weight'],1-case['weight'],0]) @ payoff
    check(case['result']['worst'],max(risk),'mixture_state_risk')
    assert case['result']['worst'] >= primal.fun-1e-12

for case in fixtures['provisioning']:
    quantity=F(str(case['quantity']))
    expected=sum(F(str(p))*(case['under']*max(F(value)-quantity,0)+case['over']*max(quantity-F(value),0)) for value,p in zip(case['values'],case['weights']))
    check(case['result']['risk'],expected,'provisioning_exact_changed')
    actual=namespaces['summaries']['provisioning'](quantity,list(map(F,case['values'])),[F(str(p)) for p in case['weights']],F(case['under']),F(case['over']))
    check(actual,expected,'changed_actual_provisioning_helper')

for case in fixtures['rules']:
    p=F(str(case['p']))
    risks=[p,1-p,F(1,5),F(4,5)]
    for actual,expected in zip(case['result']['bayesRisks'],risks): check(actual,expected,'procedure_conditional_average')
    assert case['result']['actions']==[i for i,r in enumerate(risks) if r==min(risks)]
    with contextlib.redirect_stdout(io.StringIO()): namespaces['procedureRisk']['inspect'](p)

for case in fixtures['contingent']:
    probabilities=case['probabilities']; index=case['index']; capacity=case['capacity']
    expected=0.8
    for state in [0,1]:
        updated=probabilities.copy();updated[index]=state
        mass=probabilities[index] if state else 1-probabilities[index]
        expected+=mass*lp_allocation(updated,capacity)
    check(case['result']['total'],expected,'contingent_branch_lp')
    ns=namespaces['capstone']; original=ns['probabilities'];ns['probabilities']=[F(str(p)) for p in probabilities]
    actual,_=ns['inspect'](index,F(4,5),capacity)
    ns['probabilities']=original
    check(actual,expected,'changed_actual_contingent_worlds')

for case in fixtures['binary']:
    p=F(str(case['p'])); risks=[(1-p)*a+p*b for a,b in case['losses']]
    actual=namespaces['lossTable']['action_risks'](p,case['losses'])
    for a,b,c in zip(actual,risks,case['result']['risks']):
        assert a==b;check(c,b,'binary_exact_changed')
for case in fixtures['fallback']:
    p=F(str(case['p'])); risks=[8*p,2*(1-p),F(3,5)]
    assert case['result']['actions']==[i for i,r in enumerate(risks) if r==min(risks)]
    counts['fallback_exact_ties']=counts.get('fallback_exact_ties',0)+1
mp.mp.dps=70
for case in fixtures['utility']:
    p=mp.mpf(str(case['p'])); expected=((1-p)*4+p*11)**2
    check(case['result']['certaintyEquivalent'],expected,'utility_exact_square_roots')

for source in packet['sources']:
    assert hashlib.sha256((root/source['path']).read_bytes()).hexdigest()==source['sha256']
report={'checkedAt':datetime.now(timezone.utc).isoformat(),'passed':True,'counts':counts,'sources':packet['sources'],'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__},'scope':'Independent review with LP formulations, exact changed calls and actual standalone program outputs; finite fixtures supplement separately read proofs.'}
(directory/'native-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'counts':counts}))
