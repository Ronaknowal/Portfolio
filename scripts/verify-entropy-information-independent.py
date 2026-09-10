"""Complementary reviewer checks; no implementation is copied from the visual models."""
import contextlib
import hashlib
import io
import itertools
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import xlogy, logsumexp
from scipy.stats import norm

root = Path(__file__).resolve().parents[1]
directory = root / 'scratch/entropy-independent-review'
fixtures = json.loads((directory / 'fixtures.json').read_text())
relative_errors = []
for item in fixtures['information']:
    # Exact float-to-Decimal conversion preserves the actual binary inputs.
    # The underflow-safe case uses actual normalized masses to isolate the log operation.
    with localcontext() as ctx:
        ctx.prec = 90
        p = [Decimal.from_float(float(x)) for x in item['state']['p']]
        q = [Decimal.from_float(float(x)) for x in item['state']['q']]
        p = [x / sum(p) for x in p]
        q = [x / sum(q) for x in q]
        scale = Decimal(2).ln() if item['base'] == 2 else Decimal(1)
        expected = sum(x * (x.ln() - y.ln()) for x, y in zip(p, q) if x) / scale
        actual = item['state']['kl']
        relative = abs(actual - float(expected)) / float(expected)
        assert relative < 3e-12, (item['p'], item['q'], actual, expected, relative)
        relative_errors.append(relative)

def entropy(probabilities):
    return -float(np.sum(xlogy(probabilities, probabilities))) / math.log(2)

maximum_checks = []
for state in fixtures['maximum']:
    mean = state['mean']
    q = np.array(state['q'])
    p = np.array(state['optimum'])
    assert abs(q.sum() - 1) < 1e-14
    assert abs(np.dot(q, [0, 1, 2]) - mean) < 1e-14
    if mean in (0, 2):
        assert state['gap'] == state['entropy'] == state['maximumEntropy'] == 0
        maximum_checks.append({'mean': mean, 'boundary': True})
        continue
    # Independently solve the moment equation in natural-parameter space.
    x = np.arange(3)
    def law(eta):
        scores = eta * x
        return np.exp(scores - logsumexp(scores))
    eta = brentq(lambda value: np.dot(law(value), x) - mean, -60, 60, xtol=1e-13)
    oracle = law(eta)
    assert np.max(np.abs(oracle - p)) < 3e-12
    lower, upper = max(0, mean - 1), mean / 2
    def at(t):
        return np.maximum([1 - mean + t, mean - 2 * t, t], 0)
    # Scale the narrow feasible interval to [0,1] so the optimizer's relative
    # stopping rule near t=1 does not dwarf the entire permitted interval.
    optimum = minimize_scalar(lambda u: -entropy(at(lower + u * (upper - lower))), bounds=(0, 1), method='bounded', options={'xatol': 1e-13})
    assert abs(-optimum.fun - state['maximumEntropy']) < 2e-8, (mean, optimum, state['maximumEntropy'])
    deficit = entropy(oracle) - entropy(q)
    assert abs(deficit - state['gap']) < 3e-12
    maximum_checks.append({'mean': mean, 'fraction': state['fraction'], 'gap': state['gap'], 'independentEntropyDeficit': deficit})

# Check the nonuniform-base certificate on different moments and weights.
base_certificates = []
for h in ([1, 2, 1], [0.4, 2, 3]):
    h = np.array(h)
    for mean in [0.5, 1, 1.6]:
        x = np.arange(3)
        def law(eta):
            scores = np.log(h) + eta * x
            return np.exp(scores - logsumexp(scores))
        eta = brentq(lambda value: np.dot(law(value), x) - mean, -30, 30)
        p = law(eta)
        best = entropy(p) + np.dot(p, np.log2(h))
        for fraction in np.linspace(0, 1, 101):
            lower, upper = max(0, mean - 1), mean / 2
            t = lower + fraction * (upper - lower)
            q = np.maximum([1 - mean + t, mean - 2 * t, t], 0)
            objective = entropy(q) + np.dot(q, np.log2(h))
            kl = np.sum(xlogy(q, q / p)) / math.log(2)
            assert abs(best - objective - kl) < 2e-12
            assert best - objective > -2e-12
            scaled = entropy(q) + np.dot(q, np.log2(7 * h))
            assert abs(scaled - objective - math.log2(7)) < 2e-12
            base_certificates.append(float(kl))

def namespace(name):
    variables = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(fixtures['examples'][name]['code'], variables)
    return variables

decode = namespace('prefix')['decode']
changed_code = dict(zip('ABCDE', ['0', '100', '101', '110', '111']))
decoded_messages = 0
for length in range(1, 5):
    for symbols in itertools.product('ABCDE', repeat=length):
        message = ''.join(symbols)
        assert decode(''.join(changed_code[s] for s in message), changed_code) == message
        decoded_messages += 1
for bits, code in [('11', changed_code), ('010', {'A': '0', 'B': '00', 'C': '1'})]:
    try:
        decode(bits, code)
        raise AssertionError('Invalid code/stream was accepted')
    except ValueError:
        pass

normal_kl = namespace('continuous')['normal_kl']
normal_integrals = []
for mp, sp, mq, sq in [(0, 1, 1, 2), (-2, 0.5, 1, 1.5), (3, 2, -1, 0.75)]:
    expected = quad(lambda t: norm.pdf(t, mp, sp) * (norm.logpdf(t, mp, sp) - norm.logpdf(t, mq, sq)), -np.inf, np.inf, epsabs=1e-10)[0]
    actual = normal_kl(mp, sp, mq, sq)
    assert abs(expected - actual) < 3e-10
    normal_integrals.append({'parameters': [mp, sp, mq, sq], 'integral': expected, 'native': actual})

joint = np.array([[0.6, 0.15], [0.05, 0.2]])
hy = entropy(joint.sum(axis=0))
conditional = entropy(joint.flatten()) - entropy(joint.sum(axis=1))
assert abs(hy - 0.934068055375491) < 1e-12
assert abs(conditional - 0.7219280948873623) < 1e-12
sources = json.loads((root / 'scratch/entropy-verification/final-source-hashes.json').read_text())
for source in sources['files']:
    assert hashlib.sha256((root / source['file']).read_bytes()).hexdigest() == source['sha256']
result = {
    'at': datetime.now(timezone.utc).isoformat(),
    'sourceSnapshot': sources,
    'decimalKlCases': len(relative_errors),
    'maximumRelativeKlError': max(relative_errors),
    'rejectedUnderflowAndRangeInputs': fixtures['rejections'],
    'maximumCases': maximum_checks,
    'nonuniformBaseCertificatesAndScaleChecks': len(base_certificates),
    'changedFiveSymbolMessages': decoded_messages,
    'rejectedCodeStreams': 2,
    'normalKlIndependentIntegrals': normal_integrals,
    'unequalContextPractice': {'hy': hy, 'conditional': conditional, 'reduction': hy - conditional},
}
(directory / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key not in ['sourceSnapshot', 'maximumCases']}, indent=2))
