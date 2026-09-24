"""Independent finite oracles for displayed transforms; no browser formula imports."""
import cmath
import json
import math
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.signal import periodogram

directory = Path('scratch/complex-transforms-verification')
data = json.loads((directory / 'fixtures.json').read_text(encoding='utf-8'))
numeric_paths = ['src/learn/data/complex-transforms-models.js', 'src/learn/data/complex-transforms-examples.js']
numeric_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in numeric_paths}
checks = {}
max_scaled_error = 0.0

def close(actual, expected, label, tolerance=3e-11):
    global max_scaled_error
    actual, expected = np.asarray(actual), np.asarray(expected)
    difference = float(np.max(np.abs(actual - expected))) if actual.size else 0
    scale = max(1.0, float(np.max(np.abs(expected))) if expected.size else 0)
    assert difference <= tolerance * scale, (label, actual, expected, difference)
    max_scaled_error = max(max_scaled_error, difference / scale)
    checks[label] = checks.get(label, 0) + 1

def c(values):
    return np.asarray(values)[..., 0] + 1j * np.asarray(values)[..., 1]

environments = {}
for name, example in data['examples'].items():
    result = subprocess.run([sys.executable, '-c', example['code']], capture_output=True, text=True, encoding='utf-8', check=True)
    assert result.stdout.rstrip() == example['expected'].rstrip(), (name, result.stdout, example['expected'])
    namespace = {'__name__': 'verification_import'}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'], f'displayed-{name}', 'exec'), namespace)
    environments[name] = namespace
checks['actual_complete_programs'] = len(environments)
with mp.workdps(110):
    for sigma in [0, 1e-18, -1e-18, 1e-9, -.2, .5, -2]:
        for omega in [0, 1e-18, -.03, 1]:
            for horizon in [0, 1e-12, 1, 4]:
                q = mp.mpc(sigma, omega)
                expected = mp.mpf(horizon) if q == 0 else -mp.expm1(-q*horizon)/q
                actual = environments['laplace']['truncated'](0, sigma, omega, horizon)
                close(actual, complex(expected), 'actual_native_laplace_removable_limit')
for values in [(0, float('nan'), 0, 1), (0, 0, 0, -1), (0, 65, 0, 1)]:
    try:
        environments['laplace']['truncated'](*values)
        raise AssertionError('Invalid finite integral input accepted')
    except ValueError:
        checks['native_laplace_invalid'] = checks.get('native_laplace_invalid', 0) + 1
for count in [1, 2, 3, 5, 8, 13, 16, 32]:
    for seed in range(5):
        values = [complex((n*5+seed)%9-4, ((n+seed)%3)-1) for n in range(count)]
        actual = environments['dft']['dft'](values)
        close(actual, np.fft.fft(values), 'actual_native_dft_changed')
        close(environments['dft']['dft'](actual, True), values, 'actual_native_inverse_changed')
        if count & (count-1) == 0:
            close(environments['fft']['fft'](values), np.fft.fft(values), 'actual_native_fft_changed')
for count in [0, 3, 6, 9]:
    try:
        environments['fft']['fft']([1]*count)
        raise AssertionError('Invalid radix-two size accepted')
    except ValueError:
        checks['native_fft_invalid'] = checks.get('native_fft_invalid', 0) + 1
for width in [.25, .7, 3]:
    for frequency in [0, -.7, .02, 1/width, 2.3]:
        close(environments['pulse']['pulse_transform'](frequency,width), quad(lambda t: math.cos(2*math.pi*frequency*t),-width/2,width/2)[0], 'actual_native_pulse_changed')
for initial in [-2, .7, 3]:
    for time in [0, .03, .4, 1.2]:
        expected = initial*math.exp(-2*math.pi*time)+quad(lambda u: 2*math.pi*math.exp(-2*math.pi*(time-u))*(2*math.cos(2*math.pi*u)+math.cos(6*math.pi*u+math.pi/2)),0,time)[0]
        close(environments['filter']['total'](time, initial), expected, 'actual_native_filter_changed')
for initial in [-2, .3, 2]:
    for time in [0, .5, 1, 1.4, 2, 2.7]:
        expected = initial*math.exp(-2*time)
        if time > 1:
            expected += quad(lambda u: 2*math.exp(-2*(time-u)),1,min(time,2))[0]
        close(environments['delayed']['response'](time, initial), expected, 'actual_native_delayed_changed')
# Independent changed practice: a shifted-ramp IVP, conjugate breaking and folding.
import sympy as sp
t = sp.symbols('t', real=True)
after = 2*sp.exp(-t)+(t-2)-1+sp.exp(-(t-2))
assert sp.simplify(sp.diff(after,t)+after-(t-2)) == 0
assert sp.simplify(after.subs(t,2)-2*sp.exp(-2)) == 0
close(np.fft.ifft([0,0,0,4]), [1,-1j,-1,1j], 'changed_practice_conjugate')
close(np.convolve([2,-1,1],[1,2]), [2,3,-1,2], 'changed_practice_convolution')
checks['changed_practice_symbolic_ramp'] = 2
for row in data['arithmetic']:
    z, w = complex(*row['z']), complex(*row['w'])
    target = {'add': lambda: z+w, 'multiply': lambda: z*w, 'divide': lambda: z/w}[row['op']]()
    close(c(row['result']), target, 'complex_arithmetic')
for row in data['synthesis']:
    first_angle = 2*math.pi*row['time']
    third_angle = 6*math.pi*row['time']+row['phase']
    first_matrix = np.array([[math.cos(first_angle),-math.sin(first_angle)],[math.sin(first_angle),math.cos(first_angle)]])
    third_matrix = np.array([[math.cos(third_angle),-math.sin(third_angle)],[math.sin(third_angle),math.cos(third_angle)]])
    close(row['first'], first_matrix@np.array([2,0]), 'phasor_rotation_matrix')
    close(row['third'], third_matrix@np.array([1,0]), 'phasor_rotation_matrix')
    close(row['signal'], 2*math.cos(first_angle)+math.cos(third_angle), 'phasor_real_projection')
for root in [sp.sqrt(3)-sp.I, 2*sp.I, -sp.sqrt(3)-sp.I]:
    assert sp.simplify(root**3+8*sp.I)==0
checks['changed_practice_cube_roots'] = 3
for row in data['roots']:
    close(c(row['result']) ** row['count'], complex(*row['z']), 'roots_power')
for row in data['projections']:
    for point in row['points']:
        signal = lambda t: -.2 + 1.3*math.cos(2*math.pi*t) + .7*math.cos(6*math.pi*t+row['phase'])
        target = complex(quad(lambda t: signal(t)*math.cos(2*math.pi*row['k']*t), 0, point['time'], epsabs=1e-12)[0], -quad(lambda t: signal(t)*math.sin(2*math.pi*row['k']*t), 0, point['time'], epsabs=1e-12)[0])
        close(c(point['accumulated']), target, 'projection_independent_quadrature')
mp.mp.dps = 70
for row in data['square']:
    m = row['terms']
    peak = mp.fsum(4*mp.sin(mp.pi*(2*j+1)/(2*m))/(mp.pi*(2*j+1)) for j in range(m))
    close(row['firstPeak'], float(peak), 'square_high_precision_peak')
    error = quad(lambda t: (1-sum(4*math.sin(2*math.pi*(2*j+1)*t)/(math.pi*(2*j+1)) for j in range(m)))**2, 0, .5, epsabs=1e-11, limit=400)[0]*2
    close(row['meanSquaredError'], error, 'square_integrated_error')
    derivative = sum(8*math.cos(2*math.pi*(2*j+1)*row['firstPeakTime']) for j in range(m))
    close(derivative, 0, 'square_peak_stationarity')
    for point in row['nearPoints'][::24]:
        reference = mp.fsum(4*mp.sin(2*mp.pi*(2*j+1)*mp.mpf(point['time']))/(mp.pi*(2*j+1)) for j in range(m))
        close(point['value'], float(reference), 'square_scaled_neighborhood')
        close(4*m*point['time'], point['scaledTime'], 'square_zoom_coordinate')
assert abs(float(2/mp.pi*mp.si(mp.pi))-1.1789797444721675) < 1e-15
for row in data['fourier']:
    values = c(row['values'])
    spectrum = np.fft.fft(values)
    close(c(row['result']['spectrum']), spectrum, 'dft_numpy')
    if row['removePair']:
        spectrum[row['bin']] = 0
        spectrum[-row['bin'] % len(values)] = 0
    close(c(row['result']['reconstructed']), np.fft.ifft(spectrum), 'inverse_changed_spectrum')
    close(row['result']['timeEnergy'], row['result']['frequencyEnergy'], 'parseval')
for row in data['aliases']:
    close([p['original'] for p in row['samples']], [p['alias'] for p in row['samples']], 'phase_aware_alias')
    assert -row['sampleRate']/2 <= row['signedAlias'] < row['sampleRate']/2
for row in data['windows']:
    values = np.array(row['samples'])
    weights = np.array(row['weights'])
    frequencies, density = periodogram(values, fs=64, window=weights, nfft=row['paddedCount'], detrend=False, scaling='density')
    close([b['frequency'] for b in row['bins']], frequencies, 'periodogram_frequencies')
    close([b['density'] for b in row['bins']], density, 'periodogram_density')
    close(row['densityIntegral'], np.sum(weights**2*values**2)/sum(weights**2), 'finite_density_identity')
    for p in row['curve']:
        transform = sum(weights*values*np.exp(-2j*np.pi*p['frequency']*np.arange(row['count'])/64))
        close(p['amplitude'], abs(transform)*(1 if p['frequency']==0 else 2)/sum(weights), 'continuous_window_curve')
for row in data['convolution']:
    full = np.convolve(row['samples'], row['kernel'])
    expected = full
    if row['circular']:
        expected = np.zeros(len(row['samples']))
        for index, value in enumerate(full): expected[index % len(expected)] += value
    close(row['output'], expected, 'convolution_direct_fold')
    close(sum(term['product'] for term in row['terms']), expected[row['outputIndex']], 'selected_convolution_terms')
for row in data['filters']:
    for p in row['points']:
        a, t = row['rate'], p['time']
        signal = lambda u: 2*math.cos(2*math.pi*u)+math.cos(6*math.pi*u+row['phase'])
        expected = row['initial']*math.exp(-a*t) + quad(lambda u: a*math.exp(-a*(t-u))*signal(u), 0, t, epsabs=1e-12)[0]
        close(p['total'], expected, 'filter_integrating_factor')
for row in data['integrals']:
    z, t = mp.mpc(row['real'], row['imaginary']), mp.mpf(row['horizon'])
    expected = t if z == 0 else -mp.expm1(-z*t)/z
    close(c(row['value']), complex(expected), 'laplace_high_precision_cancellation')
for row in data['regions']:
    q = complex(row['decay'] + row['sigma'], row['omega'])
    expected = row['horizon'] if q == 0 else -cmath.expm1(-q*row['horizon'])/q if hasattr(cmath, 'expm1') else (1-cmath.exp(-q*row['horizon']))/q
    if row['side'] == 'left': expected = -row['horizon'] if q == 0 else (1-cmath.exp(q*row['horizon']))/q
    close(c(row['value']), expected, 'sided_finite_integrals')
    assert row['converges'] == (q.real > 0 if row['side']=='right' else q.real < 0)
    checks['region_condition'] = checks.get('region_condition', 0) + 1
checks['rejected_invalid_inputs'] = data['rejected']
assert numeric_hashes == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in numeric_paths}
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'numericSourceHashes': numeric_hashes, 'checks': checks, 'maxScaledError': max_scaled_error, 'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'sympy': sp.__version__, 'limits': 'Finite arithmetic checks and complete program execution; browser and general analytic proofs are separate.'}
(directory / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
