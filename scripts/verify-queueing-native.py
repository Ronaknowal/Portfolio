"""Independent event, exact area, probability-flow and integration checks."""
import contextlib
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as F
import heapq
import io
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma, geom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/queueing-verification'
data = json.loads((OUT / 'cases.json').read_text(encoding='utf-8'))
counts = {}

def close(a, b, rtol=2e-9, atol=2e-10):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

functions = {}
for example in data['examples']:
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(example['code'], example['id'] + '.py', 'exec'), namespace)
    assert stdout.getvalue().strip() == example['expected'].strip(), example['id']
    functions[example['id']] = namespace
counts['complete_programs_exact_stdout'] = len(functions)

def event_trace(jobs):
    # Independent priority-event dispatch, not a vectorized copy of max recursion.
    events = [(F(job['arrival']), 1, i) for i, job in enumerate(jobs)]
    heapq.heapify(events)
    waiting = []
    busy = False
    result = [{} for _ in jobs]
    while events:
        time = events[0][0]
        batch = []
        while events and events[0][0] == time:
            batch.append(heapq.heappop(events))
        for _, kind, i in sorted(batch):
            if kind == 0:
                busy = False
                result[i]['departure'] = time
            else:
                waiting.append(i)
        while waiting and not busy:
            i = waiting.pop(0)
            result[i]['start'] = time
            duration = F(jobs[i]['service'])
            if duration == 0:
                result[i]['departure'] = time
            else:
                busy = True
                heapq.heappush(events, (time + duration, 0, i))
    for row, job in zip(result, jobs):
        row['wait'] = row['start'] - F(job['arrival'])
        row['total'] = row['departure'] - F(job['arrival'])
    return result

windows = 0
for case in data['traces']:
    expected = event_trace(case['jobs'])
    actual_native = functions['trace']['trace_jobs']([(job['id'], job['arrival'], job['service']) for job in case['jobs']])
    for actual, oracle, native in zip(case['trace'], expected, actual_native):
        native = {'start': native[2], 'departure': native[3], 'wait': native[2]-native[1], 'total': native[3]-native[1]}
        for name in ['start', 'departure', 'wait', 'total']:
            assert F(actual[name]) == oracle[name]
            assert F(native[name]) == oracle[name]
    for window in case['windows']:
        h = F(window['horizon'])
        intervals = [(F(job['arrival']), row['departure' if window['boundary'] == 'system' else 'start']) for job, row in zip(case['jobs'], expected)]
        points = sorted({F(0), h} | {t for interval in intervals for t in interval if 0 < t < h})
        area = sum((right-left) * sum(a <= (left+right)/2 < b for a, b in intervals) for left, right in zip(points, points[1:]))
        done = sum(b-a for a, b in intervals if b <= h)
        close(window['stepArea'], float(area))
        close(window['clippedArea'], float(area))
        close(window['completedResidence'], float(done))
        close(window['pendingArea'], float(area-done))
        close(window['meanOccupancy'], float(area/h))
        if not case['jobs']:
            assert window['completedMean'] is None
        windows += 1
counts['independent_priority_event_traces'] = len(data['traces'])
counts['exact_fraction_windows'] = windows

for case in data['mm1']:
    lam, mu, p, state = case['lambda'], case['mu'], case['p'], case['state']
    rho = lam/mu
    close(state['probabilities'], geom.pmf(np.arange(1, 13), 1-rho))
    close(state['tail'], geom.sf(12, 1-rho))
    close(sum(state['probabilities']) + state['tail'], 1)
    close(state['meanNumber'], geom.mean(1-rho)-1)
    native = functions['tails']['stationary_mm1'](lam, mu, p)
    close(native['queue_quantile'], state['waitQuantile'])
    close(native['total_quantile'], state['totalQuantile'])
    assert state['waitQuantile'] >= 0
    # Independent Erlang mixture; omitted probability has a rigorous rho**cut bound.
    cut = max(100, math.ceil(math.log(1e-12)/math.log(rho)))
    ns = np.arange(cut)
    masses = (1-rho)*rho**ns
    for row in case['survival']:
        expected_total = float(np.dot(masses, gamma.sf(row['t'], ns+1, scale=1/mu)))
        expected_wait = float(np.dot(masses[1:], gamma.sf(row['t'], ns[1:], scale=1/mu)))
        close(row['total'], expected_total, atol=3e-12)
        close(row['queue'], expected_wait, atol=3e-12)
    close(gamma.cdf(state['totalQuantile'], 1, scale=1/(mu-lam)), p)
counts['stationary_states_and_erlang_mixtures'] = len(data['mm1'])

for state in data['mixtures']:
    mean = state['mean']
    if state['atoms']:
        busy_area = sum(atom['probability'] * quad(lambda t: atom['duration']-t, 0, atom['duration'])[0] for atom in state['atoms'])
        close(sum(atom['busyShare'] for atom in state['atoms']), 1)
        native = functions['mixtures']['mixture_metrics']([(atom['probability'], atom['duration']) for atom in state['atoms']], state['arrivalRate'])
        close(native[-1], state['meanWait'])
    else:
        busy_area = quad(lambda s: s*s/2 * math.exp(-s/mean)/mean, 0, np.inf)[0]
    close(state['busyResidual'], busy_area/mean)
    close(state['timeResidual'], state['arrivalRate']*busy_area)
    close(state['meanWait'], state['timeResidual'] + state['arrivalRate']*state['meanWait']*mean)
counts['residual_quadrature_states'] = len(data['mixtures'])

changed_mixtures = 0
for probability, short, long, arrival in itertools.product([F(1,5), F(1,2), F(4,5)], [F(1,50), F(1,20)], [F(9,50), F(3,10)], [F(1), F(3)]):
    atoms = [(probability,short),(1-probability,long)]
    mean = sum(p*s for p,s in atoms)
    second = sum(p*s*s for p,s in atoms)
    if arrival*mean >= 1:
        continue
    oracle = arrival*second/(2*(1-arrival*mean))
    native = functions['mixtures']['mixture_metrics']([(float(p),float(s)) for p,s in atoms],float(arrival))
    close(native[-1],float(oracle))
    close(native[3],float(second/(2*mean)))
    changed_mixtures += 1
counts['changed_native_rational_mixtures'] = changed_mixtures

for state in data['pools']:
    c, lam, mu = state['servers'], state['arrivalRate'], state['serviceRate']
    a, rho = lam/mu, lam/(c*mu)
    # Erlang-B loss recurrence, then B-to-C relation: independent normalizer route.
    B = 1.0
    for n in range(1, c+1):
        B = a*B/(n+a*B)
    C = B/(1-rho+rho*B)
    close(state['waitProbability'], C)
    close(state['meanWait'], C/(c*mu-lam))
    native = functions['pooling']['erlang_c'](lam, mu, c)
    close(native, [C, state['meanWait'], state['meanTotal']])
counts['erlang_b_to_c_and_native_comparisons'] = len(data['pools'])

for state in data['buffers']:
    lam, mu, K = state['arrivalRate'], state['serviceRate'], state['capacity']
    with localcontext() as ctx:
        ctx.prec = 90
        ratio = Decimal(str(lam))/Decimal(str(mu))
        weights = [ratio**n for n in range(K+1)]
        probs = [w/sum(weights) for w in weights]
        admitted = Decimal(str(lam))*sum(probs[:-1])
        mean_number = sum(n*p for n, p in enumerate(probs))
        mean_queue = sum(max(0,n-1)*p for n, p in enumerate(probs))
        close(state['probabilities'], [float(p) for p in probs], atol=0, rtol=3e-14)
        close(state['admittedRate'], float(admitted), atol=0, rtol=3e-14)
        close(state['meanTotal'], float(mean_number/admitted), atol=0, rtol=3e-14)
        close(state['meanWait'], float(mean_queue/admitted), atol=0, rtol=3e-14)
    if max(lam/mu, mu/lam) < 50:
        Q = np.zeros((K+1,K+1))
        for n in range(K+1):
            if n < K: Q[n,n+1] = lam
            if n: Q[n,n-1] = mu
            Q[n,n] = -Q[n].sum()
        A = Q.T.copy()
        A[-1,:] = 1
        b = np.zeros(K+1); b[-1] = 1
        oracle = np.linalg.solve(A,b)
        close(state['probabilities'], oracle)
    native = functions['finite-capacity']['finite_queue'](lam, mu, K)
    close(native[0], state['probabilities'], atol=0, rtol=3e-14)
    close(native[1:], [state['admittedRate'], state['meanNumber'], state['meanTotal'], state['meanWait']])
counts['finite_generators_decimal_and_native'] = len(data['buffers'])

for cap in [0.05,0.1,0.3,1,3,10,100]:
    mean, second = functions['heavy-tail']['truncated_pareto_moments'](cap)
    minimum = 1/30
    survival = lambda t: 1 if t < minimum else (minimum/t)**1.5
    close(mean, quad(survival,0,cap,points=[minimum])[0])
    close(second,quad(lambda t:2*t*survival(t),0,cap,points=[minimum])[0])
counts['heavy_tail_survival_integrals'] = 14

vacation_cases = 0
for arrival, service, vacation in itertools.product([1,3,5], [F(1,20),F(1,10)], [F(1,10),F(1,5),F(2,5)]):
    rho = F(arrival)*service
    # Deterministic cycle residual is the integral of (v-t)/v over [0,v].
    residual = quad(lambda t:(float(vacation)-t)/float(vacation),0,float(vacation))[0]
    expected = F(arrival)*service*service/(2*(1-rho))
    actual = functions['vacations']['repeated_vacation_wait'](arrival,float(service),float(service)**2,float(vacation),float(vacation)**2)
    close(actual,[float(expected),residual,float(expected)+residual])
    vacation_cases += 1
counts['changed_vacation_residual_cases'] = vacation_cases

invalid_native = [
    ('trace', 'trace_jobs', ([('a',0,-1)],)),
    ('trace', 'trace_jobs', ([('a',1,1),('b',0,1)],)),
    ('trace', 'trace_jobs', ([('a',1e6,1e-20)],)),
    ('tails', 'stationary_mm1', (1,1,.9)),
    ('tails', 'stationary_mm1', (1,2,1)),
    ('mixtures','mixture_metrics', ([(.5,.1)],1)),
    ('pooling','erlang_c',(1,2,True)),
    ('pooling','erlang_c',(1,2,2.5)),
    ('finite-capacity','finite_queue',(1,2,0)),
    ('finite-capacity','finite_queue',(1,2,True)),
    ('simulation','simulate_fcfs',(1,1,2)),
    ('heavy-tail','truncated_pareto_moments',(float('inf'),)),
    ('vacations','repeated_vacation_wait',(8,.1,.005,.4,.16)),
    ('vacations','repeated_vacation_wait',(1e100,1e-101,1e300,.4,.16)),
]
for group, function, args in invalid_native:
    try:
        functions[group][function](*args)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError((group,function,args))
counts['native_rejections'] = len(invalid_native)
counts['model_rejections'] = len(data['invalid'])

archive = json.loads((ROOT/'docs/teaching/evidence/queueing-original-content.json').read_text(encoding='utf-8'))
# Original conservation checked using the archived code-block sequence, independently of generator.
blocks = archive['blocks']
for original_id, index in [('original-mm1',0),('original-mg1',2)]:
    example = next(x for x in data['examples'] if x['id']==original_id)
    assert example['code'] == blocks[index]['text']
    assert example['expected'] == blocks[index+1]['text']
counts['original_code_and_output_blocks_conserved'] = 4

result = {'at':datetime.now(timezone.utc).isoformat(),'passed':True,'counts':counts,
          'limits':'Finite independent checks and exact deterministic examples; simulation is not a proof of stationarity or a production benchmark.'}
(OUT/'native-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
