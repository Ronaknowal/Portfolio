"""Complementary source/program checks: exact and independently formulated oracles."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import io
import json
import math

import mpmath as mp
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/complex-transforms-independent-review'
data = json.loads((DIRECTORY / 'payload.json').read_text())
mp.mp.dps = 90
counts = {}
maximum_error = 0.0


def near(actual, expected, label, tolerance=2e-11):
    global maximum_error
    error = abs(complex(actual) - complex(expected)) / max(1.0, abs(complex(expected)))
    assert error < tolerance, (label, actual, expected, error)
    maximum_error = max(maximum_error, error)
    counts[label] = counts.get(label, 0) + 1


namespaces = {}
for name, example in data['examples'].items():
    output, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], name, 'exec'), namespace)
    assert output.getvalue().strip() == example['expected'].strip(), name
    namespaces[name] = namespace
counts['actual_programs_and_stdout'] = len(namespaces)

for row in data['integrals']:
    q = mp.mpc(*row['q'])
    time = mp.mpf(row['time'])
    expected = time if q == 0 else -mp.expm1(-q * time) / q
    near(complex(*row['result']), expected, 'model_integral_mp90')
    actual = namespaces['laplace']['truncated'](0, row['q'][0], row['q'][1], row['time'])
    near(actual, expected, 'actual_native_integral_mp90')

# The filter is one coordinate of a five-dimensional constant-coefficient ODE.
# This oracle does not use its gain/lag formulas or convolution quadrature.
for row in data['filters']:
    matrix = np.zeros((5, 5))
    for offset, frequency in [(0, 1), (2, 3)]:
        omega = 2 * math.pi * frequency
        matrix[offset, offset + 1] = -omega
        matrix[offset + 1, offset] = omega
    rate, phase = row['rate'], row['phase']
    matrix[4] = [2 * rate, 0, rate, 0, -rate]
    state = np.array([1, 0, math.cos(phase), math.sin(phase), row['initial']])
    for point in row['points']:
        expected = (expm(matrix * point['time']) @ state)[4]
        near(point['total'], expected, 'filter_augmented_matrix_exponential')
        near(point['steady'] + point['transient'], point['total'], 'filter_visible_decomposition')

# Build every finite-frequency point as a polynomial evaluated at roots of unity.
# Compare no-detrending energy with a separate time-weighted calculation.
for row in data['windows']:
    n = np.arange(row['count'])
    x = np.cos(2 * np.pi * row['tone'] * n / 64)
    w = np.ones(row['count']) if row['window'] == 'rectangular' else np.sin(np.pi*n/row['count']) ** 2
    for k in [0, 13, 49, 83, 256]:
        z = np.exp(-2j*np.pi*k/512)
        value = np.polynomial.polynomial.polyval(z, w*x)
        near(complex(*row['bins'][k]['value']), value, 'window_polynomial_evaluation')
        expected = abs(value)**2 / (64*np.dot(w,w)) * (1 if k in (0,256) else 2)
        near(row['bins'][k]['density'], expected, 'window_density_scaling')
    target = np.dot(w*x, w*x)/np.dot(w,w)
    near(row['weightedMeanSquare'], target, 'no_detrending_weighted_energy')
    near(row['densityIntegral'], target, 'zero_padding_integrated_density')

# Changed jump levels 2 and 5, rather than the author's -1/+1 fixture.
# Integrate the Dirichlet derivative kernel, not the odd-harmonic sum.
gibbs_ratio = mp.si(mp.pi)/mp.pi - mp.mpf('.5')
for row in data['square']:
    m = row['terms']
    kernel_value = 4/mp.pi * mp.quad(lambda u: m if u == 0 else mp.sin(2*m*u)/(2*mp.sin(u)), [0, mp.pi/(2*m)])
    near(row['firstPeak'], kernel_value, 'square_integrated_dirichlet_peak')
    changed_peak = mp.mpf('3.5') + mp.mpf('1.5') * kernel_value
    near((changed_peak-5)/3, (row['firstPeak']-1)/2, 'changed_jump_fraction')
    assert changed_peak > 5
    if m == 64:
        assert abs((changed_peak-5)/3-gibbs_ratio) < mp.mpf('.00002')

# An unrelated periodic piecewise-smooth example: exp(t) on [-1/2,1/2].
# Check coefficients and the finite projection energy identity, with the x∈L2 condition.
for k in range(-5, 6):
    coefficient = 2*math.sinh(.5)*(-1)**k / complex(1, -2*math.pi*k)
    integral = complex(quad(lambda t: math.exp(t)*math.cos(2*math.pi*k*t), -.5, .5)[0],
                       -quad(lambda t: math.exp(t)*math.sin(2*math.pi*k*t), -.5, .5)[0])
    near(coefficient, integral, 'changed_periodic_coefficients')
for degree in [0, 1, 3]:
    coefficients = {k: 2*math.sinh(.5)*(-1)**k/complex(1,-2*math.pi*k) for k in range(-degree, degree+1)}
    def approximation(t):
        return sum(c*np.exp(2j*np.pi*k*t) for k,c in coefficients.items()).real
    error = quad(lambda t: (math.exp(t)-approximation(t))**2, -.5,.5,epsabs=1e-13)[0]
    exact_energy = math.sinh(1)
    near(error, exact_energy-sum(abs(c)**2 for c in coefficients.values()), 'changed_finite_projection_pythagoras')

# Displayed direct DFT/FFT helpers: exact changed cosine/sine/DC basis mixtures.
for count in [4, 8, 16]:
    samples = [1.25+2*math.cos(2*math.pi*n/count)-.5*math.sin(2*math.pi*n/count) for n in range(count)]
    expected = np.zeros(count, complex)
    expected[0], expected[1], expected[-1] = 1.25*count, count*(1+.25j), count*(1-.25j)
    for helper in ['dft','fft']:
        actual = namespaces[helper][helper](samples)
        for value, reference in zip(actual, expected): near(value, reference, f'actual_{helper}_known_harmonics')

changed = data['filteredDft']
expected = np.fft.fft([4,-4,4,4,4,4,4,-4]); expected[[1,7]]=0
for actual, reference in zip(changed['reconstructed'], np.fft.ifft(expected)):
    near(complex(*actual), reference, 'changed_extreme_filtered_inverse')
near(changed['reconstructed'][0][0], 4+2*math.sqrt(2), 'filtered_peak_exact_value')

# Changed display capstone run and an independent augmented-system check.
code = data['examples']['capstone']['code']
code = code.replace('32, 64, 4*math.pi, -.5', '48, 96, 9, 1.25')
code = code.replace('[(2, 1.5, math.pi/4), (5, .75, -math.pi/3)]', '[(3, 1.2, -.4), (6, .9, .7)]')
namespace, output = {}, io.StringIO()
with contextlib.redirect_stdout(output): exec(code, namespace)
matrix = np.zeros((5,5)); initial = []
for offset, (frequency, amplitude, phase) in zip([0,2], namespace['modes']):
    matrix[offset,offset+1],matrix[offset+1,offset] = -2*np.pi*frequency,2*np.pi*frequency
    matrix[4,offset] = 9*amplitude
    initial.extend([math.cos(phase),math.sin(phase)])
matrix[4,4]=-9; initial.append(1.25)
near(namespace['total'], (expm(matrix*.25)@initial)[4], 'actual_changed_capstone_matrix_ode')

result = {'checkedAt':datetime.now(timezone.utc).isoformat(),'status':'passed',
          'productionSources':data['productionSources'],'checks':counts,'maxScaledError':maximum_error,
          'gibbsFractionOfJump':str(gibbs_ratio),'changedCapstoneStdout':output.getvalue().strip(),
          'limits':'Finite complementary checks of actual programs/models and independently formulated analytic fixtures; final browser, source freeze and general proof review separately recorded.'}
(DIRECTORY/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
