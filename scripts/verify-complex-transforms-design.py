"""Independent exact/numerical checks of proposed teaching fixtures, not production QA."""
import cmath
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import quad

assert (1 + 2j) * (2 - 1j) == 4 + 3j
assert (4 + 3j) / (2 - 1j) == 1 + 2j
values = np.array([1, 2, 0, -1])
assert np.allclose(np.fft.fft(values), [2, 1 - 3j, 0, 1 + 3j])
assert np.allclose(np.fft.ifft(np.fft.fft(values)), values)
assert np.sum(values ** 2) == (4 + 10 + 0 + 10) / 4
assert abs(np.sum(values ** 2) - np.sum(abs(np.fft.fft(values)) ** 2) / 4) < 1e-13
times = np.arange(16) / 16
assert np.max(abs(np.cos(2 * np.pi * 13 * times + np.pi / 3) - np.cos(2 * np.pi * 3 * times - np.pi / 3))) < 2e-14
linear = np.convolve([1, 2, 0, -1], [1, 1])
circular = np.fft.ifft(np.fft.fft(values) * np.fft.fft([1, 1, 0, 0])).real
assert np.array_equal(linear, [1, 3, 2, -1, -1])
assert np.allclose(circular, [0, 3, 2, -1])
t, s = sp.symbols('t s', positive=True)
response = 1 - sp.exp(-2 * t)
assert sp.simplify(sp.diff(response, t) + 2 * response - 2) == 0
assert sp.simplify(sp.laplace_transform(response, t, s, noconds=True) - 2 / (s * (s + 2))) == 0
def signal(t):
    return 2 * math.cos(2 * math.pi * t) + math.cos(6 * math.pi * t + math.pi / 2)
def steady(t):
    return math.sqrt(2) * math.cos(2 * math.pi * t - math.pi / 4) + math.cos(6 * math.pi * t + math.pi / 2 - math.atan(3)) / math.sqrt(10)
response_errors = []
for at in np.linspace(0, 2, 41):
    convolution = quad(lambda tau: 2 * math.pi * math.exp(-2 * math.pi * (at - tau)) * signal(tau), 0, at, epsabs=1e-12)[0]
    closed = steady(at) - steady(0) * math.exp(-2 * math.pi * at)
    response_errors.append(abs(convolution - closed))
assert max(response_errors) < 2e-12
weighted_errors = []
for decay in [-1, 0, 2]:
    for sigma in [-2, 0, 3]:
        for omega in [0, .5, 2]:
            for horizon in [.5, 2, 4]:
                exponent = complex(decay + sigma, omega)
                expected = horizon if exponent == 0 else (1 - cmath.exp(-exponent * horizon)) / exponent
                actual = complex(
                    quad(lambda time: math.exp(-(decay + sigma) * time) * math.cos(omega * time), 0, horizon)[0],
                    -quad(lambda time: math.exp(-(decay + sigma) * time) * math.sin(omega * time), 0, horizon)[0],
                )
                weighted_errors.append(abs(actual - expected) / max(1, abs(expected)))
assert max(weighted_errors) < 2e-12
report = {
    'at': datetime.now(timezone.utc).isoformat(),
    'status': 'design-fixtures-passed; no production implementation or browser claim',
    'checks': {'complexProductAndDivision': 2, 'exactFourPointDftAndInverseEnergy': 3, 'phaseAwareAliasSamples': 16, 'linearAndCircularConvolutions': 2, 'symbolicStepResponseIdentities': 2, 'independentTimeDomainFilterIntegrals': 41, 'finiteWeightedLaplaceIntegrals': len(weighted_errors)},
    'maximumResponseAbsoluteDifference': max(response_errors),
    'maximumWeightedIntegralRelativeDifference': max(weighted_errors),
    'libraries': {'numpy': np.__version__, 'sympy': sp.__version__},
}
Path('docs/teaching/evidence/complex-transforms-design-checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
