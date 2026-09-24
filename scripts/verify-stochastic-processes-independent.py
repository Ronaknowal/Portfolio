"""Complementary reviewer checks of actual Stochastic Processes lesson programs/models."""
import contextlib
import hashlib
import io
import itertools
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.stats import norm, poisson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/stochastic-processes-independent-review'
OUT.mkdir(parents=True, exist_ok=True)
requests = []


def request(name, *arguments):
    index = len(requests)
    requests.append({'name': name, 'args': arguments})
    return index


markov_requests = [request('markovState', {'a': a, 'b': b, 'initialSunny': .375, 'steps': 7})
                   for a, b in [(.15, .45), (.7, .6), (1, 1), (0, 0), (0, .25)]]
reserve_requests = [request('absorptionState', {'boundary': boundary, 'start': start,
                                              'upward': p, 'steps': 8})
                    for boundary in [3, 5, 7] for p in [.2, .4, .5, .6, .8]
                    for start in range(boundary + 1)]
clock_requests = [request('jumpClockState', {'alpha': a, 'beta': b, 'initial': initial,
                                           'horizon': horizon, 'seed': 223})
                  for a, b in [(.3, 1.7), (2.3, .15)] for initial in [0, 1]
                  for horizon in [.2, 2.7, 7.3]]
bridge_requests = [request('bridgeState', {'left': left, 'right': right, 'duration': duration,
                                         'scale': scale, 'fraction': fraction, 'barrier': barrier})
                   for left, right, duration, scale, barrier in [(-.3, .25, .8, .9, .7),
                                                                 (.4, -.2, 1.3, .6, .9),
                                                                 (0, 0, 2, 1.4, 1.1)]
                   for fraction in [.2, .35, .7]]
nested_requests = [request('brownianState', {'seed': 251, 'horizon': 1.7, 'drift': -.4,
                                           'scale': 1.3, 'level': level, 'pathIndex': path})
                   for level in [1, 3, 5, 8] for path in [0, 3, 7]]
split_requests = [request('splitCountLaw', mean, first, total-first, probability)
                  for mean in [0, 3] for total in [0, 2, 5]
                  for probability in [0, .2, .65, 1] for first in range(total+1)]
arrival_requests = [request('arrivalState', {'rates': rates, 'interval': interval, 'seed': 113,
                                            'routing': routing, 'splitProbability': split,
                                            'maxEvents': cap})
                    for rates in [[.3, 5.1], [0, 3.7], [2.1, 0]]
                    for interval in [[0, 3], [.5, 2.25], [2, 2+2**-49]]
                    for routing, split, cap in [('independent', .3, 120), ('alternating', .3, 120),
                                                ('independent', 1, 1)]]
node_source = """
import fs from 'node:fs';
import * as model from './src/learn/data/stochastic-processes-models.js';
import {stochasticProcessesExamples as examples} from './src/learn/data/stochastic-processes-examples.js';
const calls=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify({examples, values:calls.map(call=>model[call.name](...call.args))}));
"""
run = subprocess.run(['node', '--input-type=module', '-e', node_source], cwd=ROOT,
                     input=json.dumps(requests), capture_output=True, text=True, check=True)
payload = json.loads(run.stdout)
values = payload['values']
counts = {}
errors = []


def close(actual, expected, tolerance=3e-10):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    discrepancy = np.max(np.abs(actual-expected), initial=0)
    assert np.allclose(actual, expected, atol=tolerance, rtol=tolerance), (actual, expected)
    errors.append(float(discrepancy))


spaces = {}
for name, example in payload['examples'].items():
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], name + '.py', 'exec'), namespace)
    assert output.getvalue().strip() == example['expected'].strip(), name
    spaces[name] = namespace
counts['actualPrograms'] = len(spaces)
original = (ROOT / 'scratch/stochastic-processes-design/original-lesson.jsx').read_text(encoding='utf-8')
codes = re.findall(r'<CodeBlock language="python">\{`(.*?)`\}</CodeBlock>', original, re.S)
outputs = re.findall(r'<CodeBlock language="output">\{`(.*?)`\}</CodeBlock>', original, re.S)
for name, code, output in zip(['weather', 'arrivalProbability', 'brownianOriginal'], codes, outputs):
    assert payload['examples'][name]['code'].rstrip() == code.rstrip()
    assert payload['examples'][name]['expected'] == output
counts['originalProgramsAndOutputs'] = len(codes)

# Exact path weights, including a reward accrued along the path, do not use matrix powers.
for index in markov_requests:
    state = values[index]
    a, b = F(str(state['a'])), F(str(state['b']))
    matrix = [[1-a, a], [b, 1-b]]
    for steps in range(8):
        endpoint = [F(0), F(0)]
        for path in itertools.product([0, 1], repeat=steps+1):
            weight = [F(3, 8), F(5, 8)][path[0]]
            for before, after in zip(path, path[1:]):
                weight *= matrix[before][after]
            endpoint[path[-1]] += weight
        close(state['history'][steps]['distribution'], [float(x) for x in endpoint])
counts['changedTwoStatePathLaws'] = len(markov_requests)*8
matrix = [[F(1, 3), F(2, 3), F(0)], [F(0), F(1, 4), F(3, 4)], [F(2, 5), F(0), F(3, 5)]]
for steps in range(6):
    endpoint = [F(0)]*3
    for tail in itertools.product(range(3), repeat=steps):
        path, weight = (1,)+tail, F(1)
        for before, after in zip(path, path[1:]):
            weight *= matrix[before][after]
        endpoint[path[-1]] += weight
    assert spaces['markov']['propagate'](matrix, [F(0), F(1), F(0)], steps) == endpoint
counts['nativeThreeStatePathLaws'] = 6

# Enumerate every eight-coin history and stop at the first boundary. Check a stopped
# drift-martingale identity as well as the finite distribution and survival integral.
reserve_coin_histories = 0
for index in reserve_requests:
    state = values[index]
    boundary, start, p = state['boundary'], state['start'], F(str(state['upward']))
    endpoint = [F(0)]*(boundary+1)
    first_lower, first_upper = F(0), F(0)
    expected_stopped_time = F(0)
    for moves in itertools.product([-1, 1], repeat=8):
        weight = p**moves.count(1)*(1-p)**moves.count(-1)
        position, stop = start, 0
        for move in moves:
            if position in [0, boundary]:
                break
            position += move
            stop += 1
        endpoint[position] += weight
        expected_stopped_time += weight*stop
        if stop == 8 and position == 0:
            first_lower += weight
        if stop == 8 and position == boundary:
            first_upper += weight
        reserve_coin_histories += 1
    close(state['history'][8]['distribution'], [float(x) for x in endpoint])
    close([state['history'][8]['firstLower'], state['history'][8]['firstUpper']],
          [float(first_lower), float(first_upper)])
    close(sum(row['surviving'] for row in state['history'][:8]), float(expected_stopped_time))
    assert sum(F(i)*mass for i, mass in enumerate(endpoint))-(2*p-1)*expected_stopped_time == start
    if p == F(1, 2):
        h = [F(i, boundary) for i in range(boundary+1)]
        durations = [F(i*(boundary-i)) for i in range(boundary+1)]
    else:
        ratio = (1-p)/p
        h = [(1-ratio**i)/(1-ratio**boundary) for i in range(boundary+1)]
        durations = [(i-boundary*h[i])/(1-2*p) for i in range(boundary+1)]
    close(state['success'], [float(x) for x in h])
    close(state['meanSteps'], [float(x) for x in durations])
    assert spaces['absorption']['reserve'](boundary, p) == (h, durations)
counts['changedReserveStates'] = len(reserve_requests)
counts['exactStoppedCoinHistories'] = reserve_coin_histories

# Uniformization expands a CTMC into a discrete jump-opportunity chain with a
# Poisson number of opportunities; this oracle does not use the author's scalar solution.
for index in clock_requests:
    state = values[index]
    generator = np.array(state['generator'])
    rate = max(state['alpha'], state['beta'])
    opportunity = np.eye(2)+generator/rate
    initial = np.eye(2)[state['initial']]

    def law(t):
        limit = int(poisson.ppf(1-1e-14, rate*t))+5
        return sum(poisson.pmf(k, rate*t)*(initial @ np.linalg.matrix_power(opportunity, k))
                   for k in range(limit+1))

    for row in state['history'][::10]:
        close(row['probabilityOn'], law(row['time'])[0])
    integral = quad(lambda t: law(t)[0], 0, state['horizon'], epsabs=1e-10)[0]
    close(state['expectedOnExposure'], integral)
    close(sum(state['exposure']), state['horizon'])
    close(state['stationaryOn'], (1/state['alpha'])/(1/state['alpha']+1/state['beta']))
counts['uniformizedCTMCAndExposure'] = len(clock_requests)

# Independent Gaussian conditioning via the covariance matrix, plus the killed
# heat-kernel semigroup integrated over a hidden midpoint below the barrier.
for index in bridge_requests:
    state = values[index]
    left, right, duration, scale, fraction, barrier = [state[name] for name in
        ['left', 'right', 'duration', 'scale', 'fraction', 'barrier']]
    drift = .43
    covariance = scale**2*duration*np.array([[fraction, fraction], [fraction, 1]])
    gain = np.linalg.solve(covariance[1:, 1:], covariance[1:, :1])[0, 0]
    conditional_mean = left+drift*fraction*duration + gain*(right-left-drift*duration)
    conditional_variance = covariance[0, 0]-gain*covariance[1, 0]
    close([state['mean'], state['variance']], [conditional_mean, conditional_variance])
    native = spaces['bridgeVariation']['bridge'](left, right, duration, scale, fraction, barrier)
    close(native, [state['mean'], state['variance'], state['logCrossing']])

    def killed(t, x, y):
        sd = scale*math.sqrt(t)
        return norm.pdf(y, loc=x, scale=sd)-norm.pdf(y, loc=2*barrier-x, scale=sd)

    joint_survival = quad(lambda midpoint: killed(fraction*duration, left, midpoint)
                          *killed((1-fraction)*duration, midpoint, right),
                          -np.inf, barrier, epsabs=1e-11, epsrel=1e-11)[0]
    conditional_survival = joint_survival/norm.pdf(right, loc=left, scale=scale*math.sqrt(duration))
    close(state['crossingProbability'], 1-conditional_survival)
counts['bridgeConditioningAndKilledKernelIntegrals'] = len(bridge_requests)

# Fixed-node Gauss-Hermite integrates these degree-four Gaussian quantities exactly.
# It also checks covariance between coarse and fine realized quadratic variations.
nodes, weights = hermgauss(3)
nodes, weights = nodes*math.sqrt(2), weights/math.sqrt(math.pi)
quadrature_cases = 0
for horizon, drift, scale in [(1.7, -.4, 1.3), (.6, .7, .45)]:
    averages = np.zeros(5)
    for choices in itertools.product(range(3), repeat=4):
        normals = [nodes[i] for i in choices]
        weight = math.prod(weights[i] for i in choices)
        path = spaces['brownianGrid']['brownian_path'](normals, horizon, drift, scale)
        centered = np.diff(path)-drift*horizon/4
        raw = np.sum(np.diff(path)**2)
        qfine = np.sum(centered**2)
        qcoarse = (centered[0]+centered[1])**2+(centered[2]+centered[3])**2
        averages += weight*np.array([raw, raw**2, qfine, qcoarse, qfine*qcoarse])
        quadrature_cases += 1
    mean = scale**2*horizon+drift**2*horizon**2/4
    variance = 2*scale**4*horizon**2/4+4*drift**2*scale**2*horizon**3/16
    close([averages[0], averages[1]-averages[0]**2], [mean, variance])
    close(averages[4]-averages[2]*averages[3], 2*scale**4*horizon**2/4)
for path in [0, 3, 7]:
    candidates = [values[i] for i in nested_requests if values[i]['pathIndex'] == path]
    finest = candidates[-1]['selected']
    for state in candidates:
        close(state['selected'], finest[::256//state['count']], tolerance=1e-14)
counts['exactGaussianDegreeFourQuadratureStates'] = quadrature_cases
counts['coupledNestedPaths'] = len(nested_requests)

split_diagnostics = []
for index in split_requests:
    mean, first, second, probability = requests[index]['args']
    state = values[index]
    total = first+second
    mark_kernel = math.comb(total, first)*probability**first*(1-probability)**second
    close(state.get('markKernel', state['conditional']), mark_kernel)
    possible = mean > 0 or total == 0
    if not possible:
        # An impossible conditioning event has no uniquely identified conditional law.
        assert state['conditional'] is None, state
    else:
        close(state['conditional'], mark_kernel)
    close(state['joint'], poisson.pmf(total, mean)*mark_kernel)
    close(state['product'], poisson.pmf(first, mean*probability)*poisson.pmf(second, mean*(1-probability)))
for total in [0, 2, 5]:
    for probability in [F(0), F(1, 5), F(13, 20), F(1)]:
        masses = [F(math.comb(total, k))*probability**k*(1-probability)**(total-k) for k in range(total+1)]
        mean_a = sum(k*mass for k, mass in enumerate(masses))
        covariance = sum(k*(total-k)*mass for k, mass in enumerate(masses))-mean_a*(total-mean_a)
        assert covariance == -total*probability*(1-probability)
        split_diagnostics.append({'total': total, 'routing': str(probability), 'covariance': str(covariance),
                                  'dependent': covariance != 0})
counts['changedSplittingProbabilities'] = len(split_requests)

for index in arrival_requests:
    state = values[index]
    first, second = state['rates']
    left, right = state['interval']
    # Partition the physical interval at the schedule change before multiplying.
    edges = sorted(set([left, right]+([2] if left < 2 < right else [])))
    mean = sum((b-a)*(first if (a+b)/2 < 2 else second) for a, b in zip(edges, edges[1:]))
    close(state['intervalMean'], mean, tolerance=1e-14)
    for event in state['events']:
        close(event['unitTime'], first*min(event['time'], 2)+second*max(0, event['time']-2))
    if state['complete']:
        assert state['intervalCount'] == sum(left < event['time'] <= right for event in state['events'])
    elif right > state['observedUntil']:
        assert state['intervalCount'] is None
counts['changedArrivalClocksAndCensoring'] = len(arrival_requests)

production_paths = [
    'src/learn/data/topics/stochastic-processes-markov-chains-brownian-motion-poisson.jsx',
    'src/learn/data/stochastic-processes-models.js', 'src/learn/data/stochastic-processes-examples.js',
    'src/learn/components/lesson-labs/StochasticProcessesLabs.jsx',
    'src/learn/components/lesson-labs/stochastic-processes-labs.css',
    'src/learn/data/curriculum/blueprints/stochastic-processes-markov-chains-brownian-motion-poisson.js',
]
result = {'at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'counts': counts,
          'maximumNumericalDiscrepancy': max(errors), 'conditionalSplittingBoundaries': split_diagnostics,
          'production': [{'path': path, 'sha256': hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
                         for path in production_paths]}
(OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ['production', 'conditionalSplittingBoundaries']}, indent=2))
