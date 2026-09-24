"""Complementary exact laws, centered-roundoff regressions and actual programs."""
import contextlib
import datetime
import io
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path

directory = Path('scratch/sampling-measurement-independent')
payload = json.loads((directory / 'cases.json').read_text(encoding='utf-8'))
cases = payload['cases']

def mean(values):
    return sum(values, F()) / len(values)

def moments(values, weights=None):
    weights = weights if weights is not None else [F(1, len(values))] * len(values)
    total = sum(weights)
    weights = [w / total for w in weights]
    mu = sum((x*w for x, w in zip(values, weights)), F())
    return mu, sum((w*(x-mu)**2 for x, w in zip(values, weights)), F())

def close(actual, expected, *, strict=False):
    expected = float(expected)
    if expected == 0:
        assert actual == 0 if strict else abs(actual) < 2e-10, (actual, expected)
    elif strict:
        assert math.isclose(actual, expected, rel_tol=3e-12, abs_tol=0), (actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=3e-11, abs_tol=2e-10), (actual, expected)

for case in cases['moments']:
    mu, variance = moments(list(map(F, case['values'])), list(map(F, case['weights'])))
    actual = case['state']
    assert abs(actual['mean']-float(mu)) <= max(8*math.ulp(float(mu)), 1e-14)
    close(actual['variance'], variance, strict=True)

for state in cases['samples']:
    population = list(map(F, state['population']))
    estimates = [mean([population[i] for i in subset]) for subset in itertools.combinations(state['eligible'], state['size'])]
    mu, variance = moments(estimates)
    close(state['expectation'], mu); close(state['variance'], variance)
    close(state['formulaVariance'], variance)
    close(state['mse'], mean([(x-mean(population))**2 for x in estimates]))

for state in cases['inclusion']:
    values = list(map(F, state['values']))
    subsets = list(itertools.combinations(range(4), 2))
    raw_weights = {'equal':[1]*6,'unequal':[4,2,1,1,1,1],'uncovered':[1,0,0,0,0,0]}[state['mode']]
    weights = [F(w,sum(raw_weights)) for w in raw_weights]
    pi = [sum(p for group,p in zip(subsets,weights) if i in group) for i in range(4)]
    raw = [mean([values[i] for i in group]) for group in subsets]
    mu, variance = moments(raw, weights)
    close(state['raw']['mean'],mu);close(state['raw']['variance'],variance)
    if 0 in pi:
        assert state['ht'] is None and state['ratio'] is None
    else:
        totals = [sum(values[i]/pi[i] for i in group) for group in subsets]
        for key, estimates in [('ht',[total/4 for total in totals]),('ratio',[total/sum(1/pi[i] for i in group) for total,group in zip(totals,subsets)])]:
            mu, variance = moments(estimates,weights)
            close(state[key]['mean'],mu);close(state[key]['variance'],variance)

for state in cases['assignments']:
    y0,y1 = list(map(F,state['baseline'])), list(map(F,state['treated']))
    groups = itertools.combinations(range(6),3) if state['blocks'] is None else itertools.product(*state['blocks'])
    estimates = [mean([y1[i] for i in group])-mean([y0[i] for i in range(6) if i not in group]) for group in groups]
    mu, variance = moments(estimates)
    close(state['expectation'],mu);close(state['variance'],variance)
    close(mu,mean([b-a for a,b in zip(y0,y1)]))

for state in cases['readings']:
    units,repeats = state['units'],state['repeats']
    coordinates=list(itertools.product(range(units),range(repeats)))
    covariance=sum((F(str(state['unitVariance'])) if a[0]==b[0] else 0)+(F(str(state['readingVariance'])) if a==b else 0) for a in coordinates for b in coordinates)
    close(state['variance'],covariance/len(coordinates)**2)

for case in cases['missing']:
    values=case['values']; places=[i for i,x in enumerate(values) if x is None]
    candidates=[]
    for replacements in itertools.product([-4,8],repeat=len(places)):
        changed=values.copy()
        for i,value in zip(places,replacements): changed[i]=value
        candidates.append(mean(list(map(F,changed))))
    close(case['state']['lower'],min(candidates));close(case['state']['upper'],max(candidates))

scopes={}
for example in payload['examples']:
    output=io.StringIO(); namespace={'__name__':'__main__'}
    with contextlib.redirect_stdout(output): exec(compile(example['code'],example['id'],'exec'),namespace)
    assert output.getvalue().strip()==example['expected'].strip(), example['id']
    scopes[example['id']]=namespace
changed_helpers=0
for size in range(2,10):
    population=[F(i*i-2*i,3) for i in range(size)]
    for n in [1,(size+1)//2,size]:
        actual=scopes['finite-samples']['describe_sampling'](population,list(range(size)),n)
        estimates=[mean([population[i] for i in group]) for group in itertools.combinations(range(size),n)]
        mu,variance=moments(estimates)
        assert actual==(len(estimates),mu,variance,F(0),variance)
        changed_helpers+=1
for y0 in itertools.product([-2,3],repeat=3):
    y1=[y0[i]+[3,-1,2][i] for i in range(3)]
    for n in [1,2]:
        actual=scopes['assignments']['assignment_distribution'](y0,y1,n)
        estimates=[mean([F(y1[i]) for i in group])-mean([F(y0[i]) for i in range(3) if i not in group]) for group in itertools.combinations(range(3),n)]
        mu,variance=moments(estimates)
        assert actual[1:]==(F(4,3),mu,variance)
        changed_helpers+=1
protocol=next(example['code'] for example in payload['examples'] if example['id']=='protocol')
changed_protocol=protocol.replace('random.Random(49)','random.Random(50)').replace('for error in [-1, 1]','for error in [0]')
assert changed_protocol != protocol
namespace={'__name__':'__main__'}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(changed_protocol,'changed-protocol','exec'),namespace)
assert [unit for unit,arm in namespace['assignment'].items() if arm=='A']==['C2','C4','C6']
assert namespace['contrasts']==[F(5)]*3
assert sum(w*c for w,c in zip([F(1,6),F(1,3),F(1,2)],namespace['contrasts']))==5
changed_helpers+=1
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,
        'replayedAuthorStates':payload['replayedStates'],'conservedNumericFields':payload['comparedNumbers'],
        'largestConservationDifference':payload['largestDifference'],'invalidInputsRejected':payload['invalidInputs'],
        'complementaryCases':{k:len(v) for k,v in cases.items()},'actualPrograms':len(scopes),
        'changedActualNativeHelpers':changed_helpers,'sources':payload['sources'],
        'scope':'Exact complementary laws plus bounded regression replay; not a claim that finite execution proves general design theorems.'}
(directory/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
