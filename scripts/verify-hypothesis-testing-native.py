import contextlib
import io
import itertools
import json
import math
import sys
from fractions import Fraction
from functools import lru_cache
import numpy as np
from scipy import stats

payload = json.load(open(sys.argv[1], encoding='utf-8'))
cases = payload['cases']

def close(actual, expected, rtol=2e-11, atol=2e-12):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

for row in cases['distributions']:
    close(row['density'], stats.t.pdf(row['t'], 4), atol=0)
    close(row['tTail'], stats.t.sf(row['t'], 4), atol=0)
    if row['normalTail'] is not None:
        # logsf avoids SciPy's ordinary sf underflow before all subnormals vanish.
        close(row['normalTail'], np.exp(stats.norm.logsf(row['t'])), rtol=8e-11, atol=1e-315)
for row in cases['quantiles']:
    close(row['value'], stats.t.ppf(row['p'], 4))
for row in cases['tails']:
    data = np.array(row['values'])
    result = stats.ttest_1samp(data, row['reference'], alternative=row['alternative'])
    ci = result.confidence_interval(1 - row['alpha'])
    close([row['mean'], row['sd'], row['se'], row['t'], row['p']], [data.mean(), data.std(ddof=1), stats.sem(data), result.statistic, result.pvalue])
    low = -math.inf if row['low'] is None else row['low']
    high = math.inf if row['high'] is None else row['high']
    close([low, high], ci)
    assert row['reject'] == (result.pvalue < row['alpha'])

def normal_stream(seed):
    state = seed % 2**32
    while True:
        state = (1664525 * state + 1013904223) % 2**32
        u = (state + .5) / 2**32
        state = (1664525 * state + 1013904223) % 2**32
        v = (state + .5) / 2**32
        yield math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

for row in cases['coverage']:
    n, confidence = row['n'], row['level'] / 100
    known = row['varianceMode'] == 'known'
    critical = stats.norm.ppf((1 + confidence) / 2) if known else stats.t.ppf((1 + confidence) / 2, n - 1)
    close(row['critical'], critical)
    means = normal_stream(407 + row['batch'] * 97)
    residuals = normal_stream(93017 + row['batch'] * 193)
    for interval in row['intervals']:
        mean = 100 + 10 / math.sqrt(n) * next(means)
        sd = 10 if known else 10 * math.sqrt(sum(next(residuals)**2 for _ in range(n - 1)) / (n - 1))
        margin = critical * sd / math.sqrt(n)
        close([interval['mean'], interval['sd'], interval['low'], interval['high']], [mean, sd, mean - margin, mean + margin])
        assert interval['covers'] == (mean - margin <= 100 <= mean + margin)

for row in cases['power']:
    se = row['sigma'] / math.sqrt(row['n'])
    q = stats.norm.isf(row['alpha'] / (2 if row['alternative'] == 'two-sided' else 1))
    boundary = q * se
    right = stats.norm.sf(boundary, loc=row['effect'], scale=se)
    left = stats.norm.cdf(-boundary, loc=row['effect'], scale=se) if row['alternative'] == 'two-sided' else 0
    close([row['se'], row['critical'], row['boundary'], row['power']], [se, q, boundary, right + left])
    close(row['power'] + row['beta'], 1)

@lru_cache(None)
def binomial_limits(k, n):
    estimate = k / n
    margin = stats.norm.ppf(.975) * math.sqrt(estimate * (1 - estimate) / n)
    return {'wald': (estimate - margin, estimate + margin),
            'wilson': tuple(stats.binomtest(k, n).proportion_ci(method='wilson')),
            'exact': tuple(stats.binomtest(k, n).proportion_ci(method='exact'))}

for row in cases['proportions']:
    n, p = row['n'], row['truth']
    probabilities = stats.binom.pmf(np.arange(n + 1), n, p)
    close(sum(part['mass'] for part in row['rows']), 1)
    for part in row['rows']:
        close(part['mass'], probabilities[part['k']])
        for method, limits in binomial_limits(part['k'], n).items():
            close(part['intervals'][method], limits)
            assert part['covers'][method] == (limits[0] <= p <= limits[1]), (n, p, part['k'], method)
    for method in ['wald', 'wilson', 'exact']:
        expected = sum(probabilities[k] for k in range(n + 1) if binomial_limits(k, n)[method][0] <= p <= binomial_limits(k, n)[method][1])
        close(row['coverage'][method], expected)
    assert row['coverage']['exact'] >= .95 - 1e-12

for row in cases['family']:
    alpha = Fraction(str(row['alpha']))
    expected = 1 - (1 - alpha) ** row['count']
    close(row['independentFamilyError'], float(expected))
    close(row['independentBonferroniError'], float(1 - (1 - alpha / row['count'])**row['count']))

@lru_cache(None)
def fair_p(n, k):
    return Fraction(sum(math.comb(n, j) for j in range(n + 1) if abs(j - n/2) >= abs(k - n/2)), 2**n)

for row in cases['looks']:
    n, alpha = row['maximum'], Fraction(str(row['alpha']))
    fixed = ever = 0
    for bits in itertools.product([0, 1], repeat=n):
        fixed += fair_p(n, sum(bits)) < alpha
        ever += any(fair_p(index, sum(bits[:index])) < alpha for index in range(1, n + 1))
    close([row['fixedError'], row['cumulativeError']], [fixed/2**n, ever/2**n])

for row in cases['signFlips']:
    values = row['values']
    sums = sorted(sum(sign*x for sign, x in zip(signs, values)) for signs in itertools.product([-1, 1], repeat=5))
    assert sorted(part['sum'] for part in row['assignments']) == sums
    extreme = sum(abs(value) >= abs(sum(values)) for value in sums)
    assert row['extremeCount'] == extreme
    close(row['p'], extreme / 32)
    close(row['current']['mean'], sum(row['current']['transformed'])/5)
for row in cases['clusters']:
    m, r, rho = row['clusters'], row['repeats'], row['correlation']
    actual_variance = rho/m + (1-rho)/(m*r)
    close(row['designEffect'], actual_variance * m * r)
    close(row['effectiveN'], 1/actual_variance)
for row in cases['predictions']:
    q = stats.norm.ppf((1 + row['level']/100)/2)
    close([row['meanHalfWidth'], row['predictionHalfWidth']], [q*row['sigma']/math.sqrt(row['n']), q*row['sigma']*math.sqrt(1+1/row['n'])])
for row in cases['effects']:
    data, delta = np.array(row['values']), row['tolerance']
    point = stats.ttest_1samp(data, 0)
    lower = stats.ttest_1samp(data, -delta, alternative='greater')
    upper = stats.ttest_1samp(data, delta, alternative='less')
    close([row['pZero'], row['lowerP'], row['upperP']], [point.pvalue, lower.pvalue, upper.pvalue])
    close([row['low95'], row['high95']], point.confidence_interval(.95))
    close([row['low90'], row['high90']], point.confidence_interval(.90))
    assert row['equivalent'] == (max(lower.pvalue, upper.pvalue) < .05)

# Execute the actual lesson helpers with changed inputs, not rewritten copies.
namespaces = {}
for key in ['paired', 'multiplicity', 'wilson']:
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(payload['examples'][key]['code'], f'<lesson-{key}>', 'exec'), namespace)
    namespaces[key] = namespace
rng = np.random.default_rng(9817)
changed_pairs = 0
for n in [2, 3, 5, 8, 31, 100]:
    for confidence in [.8, .9, .95, .99]:
        old = rng.normal(50, 12, n)
        new = old - rng.normal(-.3, 4, n)
        result = namespaces['paired']['paired_report'](old, new, confidence)
        oracle = stats.ttest_1samp(old-new, 0)
        close(result[-2], oracle.confidence_interval(confidence))
        close(result[-1].pvalue, oracle.pvalue)
        changed_pairs += 1
for old, new in [([1], [1]), ([1,2],[1]), ([math.nan,1],[1,1]), ([2,2],[1,1]), ([1,2],[[1,2]])]:
    try:
        namespaces['paired']['paired_report'](old,new)
        raise AssertionError('invalid paired input accepted')
    except ValueError:
        pass
holm_cases = 0
for size in [1,2,5,20]:
    for _ in range(20):
        p = rng.choice([0,.001,.01,.04,.05,.2,1],size=size)
        adjusted = namespaces['multiplicity']['holm_adjust'](p)
        # Independent step-down decision oracle at several thresholds.
        for alpha in [.001,.01,.05,.2]:
            expected = np.zeros(size,dtype=bool)
            for rank,index in enumerate(np.argsort(p,kind='stable')):
                if p[index] < alpha/(size-rank): expected[index]=True
                else: break
            np.testing.assert_array_equal(adjusted<alpha,expected)
        holm_cases += 1
wilson_cases = 0
for n in [1,2,5,10,100]:
    for k in range(n+1):
        for confidence in [.8,.95,.99]:
            close(namespaces['wilson']['wilson'](k,n,confidence), stats.binomtest(k,n).proportion_ci(confidence,method='wilson'))
            wilson_cases += 1

# Changed practice calculations, independent of the browser fixture constraints.
practice = {
    'shifted95': list(stats.ttest_1samp([4,6,1,5,4],0).confidence_interval()),
    'reference3': float(stats.ttest_1samp([2,4,-1,3,2],3,alternative='greater').pvalue),
    'equivalence90': list(stats.ttest_1samp([.1,.3,-.2,.2,.1],0).confidence_interval(.90)),
    'allSuccessExact95': list(stats.binomtest(10,10).proportion_ci()),
    'fiveFamily': 1-.95**5,
}
close(practice['reference3'], .8509925941343948)
close(practice['equivalence90'], [-.078363, .278363], atol=5e-7)
print(json.dumps({'status':'passed','scipyVersion':__import__('scipy').__version__,'changedPairedHelpers':changed_pairs,'changedHolmHelpers':holm_cases,'changedWilsonHelpers':wilson_cases,'practice':practice}))
