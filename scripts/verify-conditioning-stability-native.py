"""Independent numerical oracles against actual displayed programs and exported model payloads."""
import contextlib
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform
import warnings

import numpy as np
import scipy
from scipy.linalg import LinAlgWarning

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/conditioning-stability-native'
data = json.loads((DIRECTORY / 'production-payload.json').read_text())
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def close(a, b, message, tolerance=2e-12):
    check(math.isclose(a, float(b), rel_tol=tolerance, abs_tol=1e-300), f'{message}: {a} vs {b}')


programs = {}
for example in data['examples']:
    namespace = {'__name__': '__main__'}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(example['code'], example['id']+'.py', 'exec'), namespace)
    check(stream.getvalue().rstrip() == example['expected'], 'Actual displayed stdout '+example['id'])
    programs[example['id']] = namespace

for row in data['rounding']:
    spacing = F(2**row['bin'], 8)
    grid = [j*spacing for j in range(8, 17)]
    target = F(16+row['half'], 2)*spacing
    distance = min(abs(point-target) for point in grid)
    candidates = [point for point in grid if abs(point-target)==distance]
    expected = candidates[0] if len(candidates)==1 else next(point for point in candidates if (point/spacing).numerator % 2 == 0)
    check(F(row['state']['rounded']) == expected, 'Independently selected rounding neighbor')

for row in data['exactFloats']:
    check(F(int(row['fraction'][0]), int(row['fraction'][1])) == F(row['value']), 'Exact IEEE decoder including subnormal')

with localcontext() as context:
    context.prec = 150
    for row in data['cancellation']:
        state = row['state']
        x = Decimal(row['sign']) / Decimal(2)**row['exponent']
        truth = (1+x).sqrt()-1
        low, high = F(state['referenceLower']), F(state['referenceUpper'])
        low_decimal = Decimal(low.numerator)/low.denominator
        high_decimal = Decimal(high.numerator)/high.denominator
        check(low_decimal <= truth <= high_decimal, 'Exact model enclosure contains independent Decimal150')
        check(high-low <= F(1,2**160), 'Enclosure width')
        for name in ['direct', 'repaired']:
            actual = abs(Decimal.from_float(state[name])-truth)
            error = state[name+'Error']
            # Display conversion to Number rounds the certified rational bound.
            slack = Decimal('1e-14')*max(actual, Decimal('1e-100'))
            check(Decimal.from_float(error['absoluteLow'])-slack <= actual <= Decimal.from_float(error['absoluteHigh'])+slack,
                  'Displayed machine error interval '+name)
        native_low, native_high = programs['cancellation']['reference_enclosure'](row['exponent'],row['sign'])
        check(native_low == low and native_high == high, 'Actual Python enclosure agrees, separately checked by Decimal')

for row in data['sensitivity']:
    state = row['state']
    if row['singular']:
        check(state['solution'] is None and state['status'] == ('nonunique' if row['change']==0 else 'inconsistent'), 'Singular equation contract')
        continue
    epsilon, delta = F(1,2**row['exponent']), F(row['change'],256)
    # Cramer rule with exact coefficients, rather than model's shift construction.
    a,b,c,d = F(1),F(1),F(1),1+epsilon
    rhs1,rhs2 = F(2),2+epsilon+delta
    determinant = a*d-b*c
    expected = [(rhs1*d-b*rhs2)/determinant, (a*rhs2-rhs1*c)/determinant]
    check(list(map(F,state['solution'])) == expected, 'Cramer sensitivity solution')
    for answer in [programs['sensitivity']['measurement_solution'](epsilon,delta)[0]]:
        check(list(answer)==expected,'Changed actual measurement helper')
    close(state['bound'], (2+epsilon)*abs(delta/epsilon), 'Sensitivity finite bound')

for row in data['backward']:
    state = row['state']
    scale = F(1) if row['scaled'] else F(1,10**row['power'])
    residual,eta,component = programs['backward']['diagnostics']([[1,0],[0,scale]],[1,scale],[1,0])
    close(state['normwise'],eta,'Normwise backward')
    check(component==1 and state['componentwise']==1,'Componentwise backward')
    close(state['changedEntry'],scale-state['changedRhs'],'Attaining actual witness')
    close(state['displayedReadingUncertainty']/state['row'],F(1,100),'Rescaled physical uncertainty')

# Changed dense rational systems: the constructed residual-based perturbation must solve the changed system.
diagnostics = programs['backward']['diagnostics']
for entries in itertools.product([-2,-1,0,1,2],repeat=4):
    a,b,c,d = map(F,entries)
    if a*d-b*c == 0:
        continue
    matrix = [[a,b],[c,d]]
    rhs,approximate = [F(1,3),F(-2,5)],[F(1,7),F(2,7)]
    residual,eta,component = diagnostics(matrix,rhs,approximate)
    anorm=max(abs(a)+abs(b),abs(c)+abs(d)); bnorm=max(map(abs,rhs)); xnorm=max(map(abs,approximate))
    denom=anorm*xnorm+bnorm
    for i in range(2):
        changed_a=[F(0),residual[i]*anorm/denom]
        changed_b=-residual[i]*bnorm/denom
        check(sum(changed_a[j]*approximate[j] for j in range(2))-changed_b==residual[i],'Changed normwise witness exact')
        check(sum(map(abs,changed_a))<=eta*anorm and abs(changed_b)<=eta*bnorm,'Changed perturbation bounds')

for row in data['summation']:
    state=row['state']
    exact=sum(map(F,state['values']),F(0))
    check(F(state['exact'])==exact,'Exact summation')
    native=programs['summation']['sum_methods'](state['values'])
    for actual,expected in zip([state['naive'],state['balanced']['value'],state['kahan'],state['neumaier']],native):
        check(actual==expected,'Actual method agrees with JS order')
    for step in state['steps']:
        check(F(step['lost']) == F(step['before'])+F(step['value'])-F(step['updated']), 'Exact lost local contribution')
for large in [2.**40,2.**53,1e16]:
    for little in [1.,3.,7.]:
        for values in itertools.permutations([large,little,-large]):
            native=programs['summation']['sum_methods'](values)
            check(F(native[-1])==sum(map(F,values),F(0)),'Changed actual fsum reference')

for row in data['propagation']:
    q=F(str(row['q']))
    for frame in row['state']['frames']:
        step=frame['step']
        disturbances=[F(1,100)*(0 if row['mode']=='pulse' and j>1 else (-1)**j if row['mode']=='alternating' else 1) for j in range(1,step+1)]
        direct=sum((q**(step-j)*delta for j,delta in enumerate(disturbances,start=1)),F(0))
        envelope=sum((abs(q)**(step-j)*abs(delta) for j,delta in enumerate(disturbances,start=1)),F(0))
        close(frame['error'],direct,'Independent convolution oracle')
        close(frame['bound'],envelope,'Independent absolute convolution')
for row in data['refinement']:
    start,final=programs['consistency']['constant_trajectory'](row['steps'])
    check(F(row['finalError'])==final,'Native exact fixed-time recurrence')

for exponent in range(8,24):
    exact,states=programs['refinement']['refine_two_channel'](exponent,3)
    epsilon=F(1,2**exponent)
    b1=exact[0]+exact[1]; b2=exact[0]+(1+epsilon)*exact[1]
    for step,x,error,residual in states:
        truth=max(abs(F(x[0])-exact[0]),abs(F(x[1])-exact[1]))
        check(error==truth,'Changed actual refinement exact forward reference')
    check(states[-1][2]<=states[0][2], 'Observed changed refinement improves bounded fixtures')
try:
    programs['refinement']['refine_two_channel'](24)
    raise AssertionError('Expected singular low-precision factor')
except LinAlgWarning:
    checks+=1

rejections=0
for action in [lambda:programs['cancellation']['reference_enclosure'](-1),lambda:programs['cancellation']['reference_enclosure'](10,bits=50),lambda:programs['sensitivity']['measurement_solution'](0,1),lambda:programs['backward']['diagnostics']([],[],[]),lambda:programs['summation']['sum_methods']([]),lambda:programs['summation']['sum_methods']([math.inf]),lambda:programs['refinement']['refine_two_channel'](25),lambda:programs['propagation']['propagation'](F(1,2),-1),lambda:programs['consistency']['constant_trajectory'](True)]:
    try:action()
    except (ValueError,TypeError):rejections+=1
    else:raise AssertionError('Expected native rejection')

result={'scope':'Actual current exported model states and actual displayed native programs; no browser/build claim',
        'completedAt':datetime.now(timezone.utc).isoformat(),'checks':checks,'nativePrograms':len(programs),
        'javascriptRejections':data['javascriptRejections'],'nativeRejections':rejections,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'sourceHashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['src/learn/data/conditioning-stability-models.js','src/learn/data/conditioning-stability-examples.js']}}
(DIRECTORY/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
