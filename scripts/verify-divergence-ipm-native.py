"""Independent numerical/optimization and actual-program evidence for the lesson."""
import contextlib
import io
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import cdist, jensenshannon
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/divergence-ipm-native-verification'
data = json.loads((DIRECTORY / 'model-fixtures.json').read_text())
errors = []


def close(actual, expected, atol=2e-10, rtol=2e-10):
    actual = float(actual)
    if math.isinf(expected):
        assert actual == expected
    else:
        assert math.isclose(actual, float(expected), abs_tol=atol, rel_tol=rtol), (actual, expected)
        errors.append(abs(actual - expected))


def divergences(p, q):
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi = np.where(q > 0, (p - q) ** 2 / q, np.where(p > 0, np.inf, 0)).sum()
    return {
        'kl': np.sum(rel_entr(p, q)),
        'reverse': np.sum(rel_entr(q, p)),
        'js': jensenshannon(p, q) ** 2,
        'hellinger': np.linalg.norm(np.sqrt(p) - np.sqrt(q)) ** 2 / 2,
        'chi': chi,
        'tv': sum(max(a - b, 0) for a, b in zip(p, q)),
    }


for item in data['divergence']:
    state = item['state']
    expected = divergences(state['p'], state['q'])
    for name, value in expected.items():
        close(state['values'][name], value)
    event_max = max(abs(sum((a - b) * take for a, b, take in zip(state['p'], state['q'], mask))) for mask in itertools.product([0, 1], repeat=len(state['p'])))
    close(state['values']['tv'], event_max)
    assert float(state['values']['kl']) >= 2 * event_max ** 2 - 1e-12

relative_kl = []
for item in data['closeLaws']:
    with localcontext() as ctx:
        ctx.prec = 90
        p = [Decimal.from_float(float(v)) for v in item['state']['p']]
        q = [Decimal.from_float(float(v)) for v in item['state']['q']]
        expected = sum(a * (a.ln() - b.ln()) for a, b in zip(p, q))
        relative = abs(item['state']['values']['kl'] - float(expected)) / float(expected)
        assert relative < 1e-12
        relative_kl.append(relative)

for state in data['processing']:
    p, q, channel = np.array(state['p']), np.array(state['q']), np.array(state['channel'])
    assert np.allclose(p @ channel, state['outputP'])
    assert np.allclose(q @ channel, state['outputQ'])
    expected = divergences(p @ channel, q @ channel)
    for name, value in expected.items():
        close(state['after'][name], value)
        assert float(state['after'][name]) <= float(state['before'][name]) + 1e-12

linear_programs = 0
for state in data['observers']:
    p, q, positions = np.array(state['p']), np.array(state['q']), np.array(state['positions'])
    delta = p - q
    close(state['linearValue'], abs(np.dot(delta, positions)))
    close(state['value'], np.dot(delta, state['scores']))
    accuracy = max(0.5 * (np.dot(p, mask) + np.dot(q, 1 - np.array(mask))) for mask in itertools.product([0, 1], repeat=len(p)))
    close(state['optimalAccuracy'], accuracy)
    if state['kind'] != 'lipschitz':
        continue
    n = len(p)
    cost = np.abs(positions[:, None] - positions[None, :])
    rows = []
    for index in range(n):
        row = np.zeros((n, n)); row[index, :] = 1; rows.append(row.ravel())
    for index in range(n):
        row = np.zeros((n, n)); row[:, index] = 1; rows.append(row.ravel())
    transport = linprog(cost.ravel(), A_eq=rows, b_eq=np.r_[p, q], bounds=(0, None), method='highs')
    assert transport.success
    close(state['w1'], transport.fun)
    constraints, bounds = [], []
    for i, j in itertools.permutations(range(n), 2):
        row = np.zeros(n); row[i] = 1; row[j] = -1
        constraints.append(row); bounds.append(cost[i, j])
    critic = linprog(-delta, A_ub=constraints, b_ub=bounds, bounds=[(0, 0)] + [(None, None)] * (n - 1), method='highs')
    assert critic.success
    close(state['w1'], -critic.fun)
    linear_programs += 2


def gram(values, kind, bandwidth):
    values = np.array(values)
    if kind == 'linear':
        return np.outer(values, values)
    if kind == 'quadratic':
        features = np.column_stack((values, values ** 2))
        return features @ features.T
    return np.exp(-cdist(values[:, None], values[:, None], 'sqeuclidean') / (2 * bandwidth ** 2))


def mmd_reference(x, y, kind, bandwidth):
    n, m = len(x), len(y)
    matrix = gram(list(x) + list(y), kind, bandwidth)
    weights = np.r_[np.full(n, 1 / n), np.full(m, -1 / m)]
    biased = weights @ matrix @ weights
    mask_x = ~np.eye(n, dtype=bool)
    mask_y = ~np.eye(m, dtype=bool)
    unbiased = matrix[:n, :n][mask_x].mean() + matrix[n:, n:][mask_y].mean() - 2 * matrix[:n, n:].mean()
    return biased, unbiased, matrix


for state in data['kernel']:
    x, y = state['x'], state['y']
    biased, unbiased, matrix = mmd_reference(x, y, state['kind'], state['bandwidth'])
    close(state['biasedSquared'], biased)
    close(state['unbiasedSquared'], unbiased)
    close(state['witnessGap'], biased)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-10
    n = len(x)
    assert np.allclose(matrix[:n, :n], state['xx'])
    assert np.allclose(matrix[n:, n:], state['yy'])
    assert np.allclose(matrix[:n, n:], state['xy'])
    for point in state['witness'][::10]:
        all_points = x + y + [point['position']]
        similarities = gram(all_points, state['kind'], state['bandwidth'])[:-1, -1]
        expected = similarities[:n].mean() - similarities[n:].mean()
        close(point['value'], expected)

# Unbiasedness checked by exact expectation over every 2+2 sample on a finite law.
unbiased_expectations = 0
for p, q in [([0.2, 0.5, 0.3], [0.2, 0.5, 0.3]), ([0.1, 0.3, 0.6], [0.5, 0.25, 0.25])]:
    positions = [-1, 0, 2]
    for bandwidth in [0.4, 1, 2]:
        expectation = 0
        biased_expectation = 0
        for indices in itertools.product(range(3), repeat=4):
            x = [positions[i] for i in indices[:2]]
            y = [positions[i] for i in indices[2:]]
            probability = p[indices[0]] * p[indices[1]] * q[indices[2]] * q[indices[3]]
            biased, unbiased = mmd_reference(x, y, 'rbf', bandwidth)[:2]
            expectation += probability * unbiased
            biased_expectation += probability * biased
        difference = np.array(p) - np.array(q)
        target = difference @ gram(positions, 'rbf', bandwidth) @ difference
        close(expectation, target)
        kernel_matrix = gram(positions, 'rbf', bandwidth)
        p_array, q_array = np.array(p), np.array(q)
        diagonal = np.diag(kernel_matrix)
        bias = (p_array @ diagonal - p_array @ kernel_matrix @ p_array) / 2
        bias += (q_array @ diagonal - q_array @ kernel_matrix @ q_array) / 2
        close(biased_expectation - target, bias)
        assert bias >= -1e-12
        unbiased_expectations += 1

permutation_allocations = 0
for state in data['permutations']:
    pool = state['pool']
    values = []
    for indices in itertools.combinations(range(8), 4):
        x = [pool[i] for i in indices]
        y = [pool[i] for i in range(8) if i not in indices]
        values.append(mmd_reference(x, y, 'rbf', state['bandwidth'])[0])
    assert len(values) == 70
    for actual, expected in zip(state['allocations'], values):
        close(actual['statistic'], expected)
    tail = sum(value >= values[0] - 1e-12 for value in values)
    close(state['pValue'], tail / 70)
    ranks = np.array([sum(value >= observed - 1e-12 for value in values) / 70 for observed in values])
    # Conditional validity under uniform label allocation, including ties.
    for numerator in range(1, 71):
        alpha = numerator / 70
        assert np.mean(ranks <= alpha + 1e-14) <= alpha + 1e-14
    permutation_allocations += 70

for state in data['variational']:
    p, q = np.array(state['p']), np.array(state['q'])
    truth = np.sum(rel_entr(p, q))
    close(state['truth'], truth)
    scores = np.array([row['score'] for row in state['rows']])
    bound = np.dot(p, scores) - np.dot(q, np.exp(scores - 1))
    close(state['bound'], bound)
    close(state['gap'], truth - bound)
    assert bound <= truth + 1e-12
    if state['scale'] == 1:
        close(state['gap'], state['shiftedOptimumGap'])
optimum = minimize(lambda t: np.dot(q, np.exp(t - 1)) - np.dot(p, t), np.zeros(3), jac=lambda t: q * np.exp(t - 1) - p, method='BFGS', options={'gtol': 1e-11})
close(-optimum.fun, truth, atol=2e-9)

for state in data['atoms']:
    h, bandwidth = state['displacement'], state['bandwidth']
    with localcontext() as ctx:
        ctx.prec = 80
        ratio = Decimal.from_float(h) / Decimal.from_float(bandwidth)
        expected = 2 * (1 - (-ratio * ratio / 2).exp())
        close(state['mmdSquared'], float(expected), atol=1e-30, rtol=2e-14)
    close(state['w1'], h)
    close(state['tv'], 0 if h == 0 else 1)
    close(state['kl'], 0 if h == 0 else math.inf)

programs = []
namespaces = {}
for name, example in data['examples'].items():
    path = DIRECTORY / f'verified-{name}.py'
    path.write_text(example['code'], encoding='utf-8')
    completed = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, check=True, timeout=30)
    assert completed.stdout.strip() == example['expected'], name
    assert not completed.stderr
    variables = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(example['code'], variables)
    namespaces[name] = variables
    programs.append(name)

# Verify the native MMD helper independently on changed unequal-size inputs.
for state in data['kernel'][:15]:
    x, y = state['x'], state['y']
    actual = namespaces['kernel']['mmd_squared'](x, y, state['bandwidth'])
    expected = mmd_reference(x, y, 'rbf', state['bandwidth'])
    close(actual[0], expected[0]); close(actual[1], expected[1])

# Recompute the complete native selection rule with independent Gram matrices
# on changed sizes, repeated observations and a selected bandwidth family.
project_cases = 0
for x, y in [([-1, -0.5, 0], [0.5, 1, 1.5]), ([-2, 0], [-1, 0, 1]),
             ([-1, -1], [-1, -1]), ([-2, -1, 0, 1], [0.5, 2])]:
    for bandwidths in [(0.75,), (0.3, 1, 3)]:
        pool = x + y
        reference = []
        for allocation in itertools.combinations(range(len(pool)), len(x)):
            left = [pool[i] for i in allocation]
            right = [pool[i] for i in range(len(pool)) if i not in allocation]
            reference.append(max(mmd_reference(left, right, 'rbf', sigma)[0] for sigma in bandwidths))
        tail = sum(score >= reference[0] - 1e-12 for score in reference)
        expected = (reference[0], tail, len(reference), tail / len(reference))
        actual = namespaces['project']['exact_test'](x, y, bandwidths)
        for left, right in zip(actual, expected):
            close(left, right)
        close(namespaces['project']['exact_test'](y, x, bandwidths)[-1], expected[-1])
        project_cases += 1

project_rejections = 0
for x, y, bandwidths in [([0], [1, 2], (1,)), (list(range(6)), list(range(5)), (1,)),
                        ([0, 1], [2, 3], ()), ([0, 1], [2, 3], (0,)),
                        ([0, 1], [2, 3], (math.inf,)), ([math.nan, 1], [2, 3], (1,))]:
    try:
        namespaces['project']['exact_test'](x, y, bandwidths)
    except ValueError:
        project_rejections += 1
    else:
        raise AssertionError((x, y, bandwidths))

# Direct independent calculations for every changed numerical practice answer.
changed = divergences([0.5, 0.3, 0.2], [0.25, 0.5, 0.25])
close(changed['tv'], 0.25)
close(changed['kl'], 0.14869719288733346)
close((1 + changed['tv']) / 2, 0.625)
support = divergences([1, 0], [0.5, 0.5])
close(support['kl'], math.log(2))
close(support['reverse'], math.inf)
close(support['hellinger'], 1 - 1 / math.sqrt(2))
positions = np.array([-2., 0., 4.])
delta = np.array([0.5, -1, 0.5])
close(delta @ positions, 1)
close(delta @ np.array([0., -2., 2.]), 3)
feature_x = np.array([-1., -1., 1., 1.])
feature_y = np.array([-math.sqrt(2), 0., 0., math.sqrt(2)])
close(np.mean(feature_x), np.mean(feature_y))
close(np.mean(feature_x ** 2), np.mean(feature_y ** 2))
assert mmd_reference(feature_x, feature_y, 'rbf', 1)[0] > 0
close(mmd_reference([-1, 1], [-1, 1], 'rbf', 1)[1], -(1 - math.exp(-2)))
close(math.expm1(-0.5) + 0.5, 0.10653065971263342)
close(0.98 ** 30, 0.5454843193824369)

result = {
    'at': datetime.now(timezone.utc).isoformat(),
    'nativePrograms': programs,
    'python': sys.version.split()[0],
    'knownLawPairs': len(data['divergence']),
    'decimalNearEqualityCases': len(relative_kl),
    'maximumRelativeNearKlError': max(relative_kl),
    'sharedChannels': len(data['processing']),
    'observerStates': len(data['observers']),
    'independentTransportAndCriticLinearPrograms': linear_programs,
    'kernelStates': len(data['kernel']),
    'exactFiniteSampleUnbiasedExpectations': unbiased_expectations,
    'exactPermutationAllocationsAndRankValidity': permutation_allocations,
    'variationalStates': len(data['variational']),
    'pointMassDecimalCases': len(data['atoms']),
    'rejectedInputs': data['invalidInputs'],
    'changedNativeMmdInputs': 15,
    'changedCompleteNativePermutationRules': project_cases,
    'nativeProjectInvalidInputs': project_rejections,
    'changedNumericalPracticeChecks': 14,
    'numericalComparisons': len(errors),
    'maximumAbsoluteDifference': max(errors),
}
(DIRECTORY / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
