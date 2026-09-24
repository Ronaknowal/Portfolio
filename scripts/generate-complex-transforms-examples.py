"""Execute complete learner programs and capture their exact displayed output."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

examples = {}


def add(name, title, question, interpretation, source, packages='Python 3.12 standard library'):
    code = textwrap.dedent(source).strip() + '\n'
    completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding='utf-8', check=True)
    examples[name] = {'title': title, 'question': question, 'language': 'python', 'code': code, 'expected': completed.stdout.rstrip(), 'interpretation': interpretation, 'environment': packages}


add('complex', 'Calculate with two coordinates', 'Predict the product and quotient before running. Does taking the modulus lose information?',
    'The product has length 5 and a different direction. Its quotient by the original multiplier recovers both coordinates. Zero has no defined argument in the mathematics, even if a library returns a convention.', '''
    import cmath
    import math

    z, w = 1 + 2j, 2 - 1j
    product = z * w
    print('product:', product)
    print('quotient:', product / w)
    print('modulus:', abs(product))
    print('angle degrees:', round(math.degrees(cmath.phase(product)), 6))
    roots = [cmath.exp(2j * math.pi * k / 4) for k in range(4)]
    print('fourth roots:', [(round(v.real), round(v.imag)) for v in roots])
    print('fourth powers:', [round((v ** 4).real) for v in roots])
    ''')

add('projection', 'Recover both frequency quadratures', 'The 3 Hz tone is a negative sine. What will its real and imaginary coefficients be?',
    'The imaginary coordinate carries the quarter-cycle phase. These uniformly spaced quadratures are exact in real arithmetic for this finite trigonometric fixture, because its frequency differences do not alias onto zero.', '''
    import cmath
    import math

    def signal(t):
        return 2 * math.cos(2 * math.pi * t) + math.cos(6 * math.pi * t + math.pi / 2)

    def coefficient(k, count=128):
        return sum(signal(n / count) * cmath.exp(-2j * math.pi * k * n / count)
                   for n in range(count)) / count

    for k in [0, 1, 2, 3, -3]:
        value = coefficient(k)
        real = 0.0 if abs(value.real) < 1e-12 else value.real
        imag = 0.0 if abs(value.imag) < 1e-12 else value.imag
        print(f'k={k:2}: {real:.6f} {imag:+.6f}j')
    ''')

add('series', 'Inspect the first overshoot', 'Will adding terms make the first peak tend to the square wave height 1?',
    'The peak moves toward the jump while remaining above 1. Its location follows from differentiating this finite sum; the printed values alone do not prove the limiting Gibbs result.', '''
    import math

    def partial(t, terms):
        return sum(4 * math.sin(2 * math.pi * (2*j+1) * t) / (math.pi * (2*j+1))
                   for j in range(terms))

    for terms in [1, 4, 16, 64]:
        peak_t = 1 / (4 * terms)
        error_energy = 1 - sum(8 / (math.pi**2 * (2*j+1)**2) for j in range(terms))
        print(f'{terms:2} terms: t={peak_t:.6f}, peak={partial(peak_t, terms):.6f}, MSE={error_energy:.6f}')
    print('at the jump:', partial(0, 64))
    ''')

add('pulse', 'Check a pulse transform by integration', 'If the pulse becomes twice as wide at the same height, how do its DC value and first zero change?',
    'The rectangular pulse has area equal to width. Its transform at zero equals that area; its first positive zero is the reciprocal width. The real quadrature checks the closed formula; the imaginary integral vanishes because its integrand is odd.', '''
    import math
    from scipy.integrate import quad

    def pulse_transform(frequency, width):
        if frequency == 0:
            return width
        return math.sin(math.pi * width * frequency) / (math.pi * frequency)

    for width in [1.0, 2.0]:
        for frequency in [0.0, 0.25, 1 / width]:
            integral = quad(lambda t: math.cos(2 * math.pi * frequency * t), -width/2, width/2)[0]
            print(f'width={width:g}, f={frequency:g}: closed={pulse_transform(frequency, width):.6f}, integral={integral:.6f}')
    ''', 'Python 3.12; SciPy')

add('dft', 'Compute the transform without an FFT', 'Which coefficient contains the imaginary information for [1,2,0,−1]?',
    'This direct O(N²) calculation exposes every finite sum. The inverse reconstructs the four stored numbers; the energy is 6 in either coordinate system after the 1/N factor.', '''
    import cmath
    import math

    def dft(values, inverse=False):
        n = len(values)
        if not n:
            raise ValueError('empty transform')
        direction = 1 if inverse else -1
        return [sum(value * cmath.exp(direction * 2j * math.pi * k * j / n)
                    for j, value in enumerate(values)) / (n if inverse else 1)
                for k in range(n)]

    values = [1, 2, 0, -1]
    spectrum = dft(values)
    print('coefficients:', [(round(z.real), round(z.imag)) for z in spectrum])
    print('inverse:', [round(z.real) for z in dft(spectrum, True)])
    print('time energy:', sum(x*x for x in values))
    print('frequency energy:', round(sum(abs(z)**2 for z in spectrum) / len(values), 6))
    ''')

add('fft', 'Reuse the even and odd work', 'Which two length-two transforms supply a length-four transform?',
    'The recursive algorithm is restricted to positive power-of-two lengths; that is this implementation’s contract, not a limitation of the mathematical DFT or every FFT library.', '''
    import cmath
    import math

    def fft(values):
        n = len(values)
        if n == 0 or n & (n - 1):
            raise ValueError('positive power-of-two length required')
        if n == 1:
            return [complex(values[0])]
        even, odd = fft(values[::2]), fft(values[1::2])
        rotated = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n//2)]
        return [even[k] + rotated[k] for k in range(n//2)] + [even[k] - rotated[k] for k in range(n//2)]

    values = [1, 2, 0, -1]
    print('even:', fft(values[::2]))
    print('odd:', fft(values[1::2]))
    print('combined:', [(round(z.real), round(z.imag)) for z in fft(values)])
    ''')

add('alias', 'Preserve phase when folding an alias', 'Does the 3 Hz alias retain or negate the 13 Hz cosine’s phase?',
    'At 16 Hz, 13 Hz is the signed representative −3 Hz. Turning that into a positive-frequency cosine negates the phase. At 32 Hz the two original continuous candidates no longer have identical samples.', '''
    import math

    def mismatch(sample_rate):
        return max(abs(math.cos(2*math.pi*13*n/sample_rate + math.pi/3)
                       - math.cos(2*math.pi*3*n/sample_rate - math.pi/3))
                   for n in range(sample_rate))

    print('16 Hz samples agree:', mismatch(16) < 1e-12)
    print('32 Hz samples agree:', mismatch(32) < 1e-12)
    invisible = [math.sin(math.pi*n) for n in range(16)]
    print('Nyquist sine samples zero:', max(map(abs, invisible)) < 1e-12)
    ''')

add('window', 'Separate amplitude, density and padding', 'Will padding this fixed window change its integrated finite power estimate?',
    'Both transform lengths integrate to the same window-weighted mean square. The density uses squared window weights; coherent sinusoidal amplitude uses their sum. Detrending is explicitly disabled.', '''
    import numpy as np
    from scipy.signal import periodogram

    fs, count = 64, 64
    n = np.arange(count)
    x = np.cos(2*np.pi*5.5*n/fs)
    w = .5 - .5*np.cos(2*np.pi*n/count)  # periodic Hann
    target = np.sum((w*x)**2) / np.sum(w*w)
    for padded in [64, 256]:
        frequencies, density = periodogram(x, fs=fs, window=w, nfft=padded,
                                          detrend=False, scaling='density', return_onesided=True)
        integrated = np.sum(density) * fs / padded
        print(f'M={padded}: spacing={fs/padded:.2f} Hz, integral={integrated:.6f}')
    print(f'weighted mean square={target:.6f}')
    print('window sum:', np.sum(w), 'squared-weight sum:', np.sum(w*w))
    ''', 'Python 3.12; NumPy and SciPy')

add('convolution', 'Find the wrapped tail', 'Where does the fifth linear-convolution output go in a four-point circular convolution?',
    'The −1 tail wraps onto the first value 1, producing 0. Padding to five points is already sufficient for this particular linear convolution; it need not be a power of two.', '''
    import numpy as np

    x, h = np.array([1, 2, 0, -1]), np.array([1, 1])
    linear = np.convolve(x, h)
    circular = np.fft.ifft(np.fft.fft(x, 4) * np.fft.fft(h, 4)).real
    padded = np.fft.ifft(np.fft.fft(x, 5) * np.fft.fft(h, 5)).real
    print('linear:', linear.tolist())
    print('circular:', np.rint(circular).astype(int).tolist())
    print('padded:', np.rint(padded).astype(int).tolist())
    ''', 'Python 3.12; NumPy')

add('filter', 'Verify startup by a second route', 'The steady response is nonzero at t=0. What must be added to satisfy y(0)=0?',
    'The homogeneous correction cancels the steady value at the initial instant and decays. Direct time-domain quadrature independently agrees with the transformed-frequency construction.', '''
    import math
    from scipy.integrate import quad

    rate = 2 * math.pi
    def signal(t):
        return 2*math.cos(2*math.pi*t) + math.cos(6*math.pi*t + math.pi/2)
    def steady(t):
        return math.sqrt(2)*math.cos(2*math.pi*t-math.pi/4) + math.cos(6*math.pi*t+math.pi/2-math.atan(3))/math.sqrt(10)
    def total(t, initial=0):
        return steady(t) + (initial-steady(0))*math.exp(-rate*t)

    for t in [0, .25, .5, 1]:
        independent = quad(lambda u: rate*math.exp(-rate*(t-u))*signal(u), 0, t)[0]
        print(f't={t:.2f}: steady={steady(t):.6f}, total={total(t):.6f}, integral={independent:.6f}')
    ''', 'Python 3.12; SciPy')

add('laplace', 'A finite integral does not establish convergence', 'What happens on the boundary σ=−a when ω is zero?',
    'Here the weighting exactly cancels the signal decay: the integrand is 1 and the finite integral grows like T. A finite number exists at every cutoff although the improper transform does not.', '''
    import cmath
    import math

    def truncated(decay, sigma, omega, horizon):
        if not all(math.isfinite(value) for value in (decay, sigma, omega, horizon)) or horizon < 0:
            raise ValueError('Use finite inputs and a nonnegative horizon')
        q = complex(decay+sigma, omega)
        z = q*horizon
        if not math.isfinite(abs(q)) or not math.isfinite(abs(z)) or abs(z) > 64:
            raise ValueError('This bounded teaching helper requires abs(q*T) <= 64')
        if z == 0:
            return horizon
        if abs(z) < 0.5:
            # Integral/T = sum (-z)^k/(k+1)!; retain a tiny increment near q=0.
            term = total = 1+0j
            for k in range(1, 25):
                term *= -z/(k+1)
                total += term
            return horizon*total
        return (1-cmath.exp(-z))/q

    for sigma in [0, -1, -2]:
        values = [truncated(1, sigma, 0, t).real for t in [1, 2, 4]]
        print(f'sigma={sigma}:', [round(v, 6) for v in values])
    print('right-sided convergence requires sigma > -1')
    ''')

add('initial', 'Carry the initial value through Laplace algebra', 'For y′+2y=2 and y(0)=3, is Y simply H times the input transform?',
    'The extra 3/(s+2) is the initial-state contribution. The result y=1+2e^(−2t) satisfies both the differential equation and the prescribed initial value.', '''
    import sympy as sp

    t = sp.symbols('t', nonnegative=True)
    s = sp.symbols('s', positive=True)
    Y = (2/s + 3)/(s+2)
    y = 1 + 2*sp.exp(-2*t)
    print('partial fractions:', sp.apart(Y, s))
    print('ODE residual:', sp.simplify(sp.diff(y,t)+2*y-2))
    print('initial:', y.subs(t,0))
    print('transform agrees:', sp.simplify(sp.laplace_transform(y,t,s,noconds=True)-Y) == 0)
    ''', 'Python 3.12; SymPy')

add('delayed', 'A delayed pulse is a difference of shifted steps', 'What stays continuous when the input switches on at t=1 and off at t=2?',
    'The first-order output is continuous, while its slope changes with the forcing. The expression is built from two delayed step responses and preserves the original initial response separately.', '''
    import math

    def step_response(t):
        return 0.0 if t < 0 else -math.expm1(-2*t)
    def response(t, initial=1):
        return initial*math.exp(-2*t) + step_response(t-1)-step_response(t-2)

    for t in [0, 1, 1.5, 2, 3]:
        print(f't={t:g}: y={response(t):.6f}')
    print('Y(s) = 1/(s+2) + 2*(exp(-s)-exp(-2*s))/(s*(s+2))')
    ''')

add('repeated', 'Repeated poles retain the time factor', 'Does a double pole at −1 give the same mode as a simple pole?',
    'The impulse response t e^(−t) is zero in position at the initial instant but has right derivative 1. A unit impulse in the second-order equation changes velocity, not position.', '''
    import sympy as sp

    t, s = sp.symbols('t s', positive=True)
    h = t*sp.exp(-t)
    H = 1/(s+1)**2
    print('ordinary residual after impulse:', sp.simplify(sp.diff(h,t,2)+2*sp.diff(h,t)+h))
    print('position at 0+:', sp.limit(h,t,0,dir='+'))
    print('velocity at 0+:', sp.limit(sp.diff(h,t),t,0,dir='+'))
    print('transform agrees:', sp.simplify(sp.laplace_transform(h,t,s,noconds=True)-H) == 0)
    ''', 'Python 3.12; SymPy')

add('shift', 'Rewrite the function after shifting time', 'Is the transform of u(t−1)t just e^(−s)/s²?',
    'After t=u+1, the remaining function is u+1. The additional 1/s term is essential; shifting only the lower limit does not replace t by t−1.', '''
    import math
    from scipy.integrate import quad

    for s in [1, 2, 3]:
        actual = quad(lambda t: t*math.exp(-s*t), 1, math.inf)[0]
        correct = math.exp(-s)*(1/s**2+1/s)
        wrong = math.exp(-s)/s**2
        print(f's={s}: integral={actual:.6f}, correct={correct:.6f}, shortcut={wrong:.6f}')
    ''', 'Python 3.12; SciPy')

add('hidden', 'A stable transfer can hide an unstable state', 'If the input and output see only the second state, what happens to an initial first state?',
    'The input-output transfer is 1/(s+1), but the unseen first state grows like e^t. Stable visible poles do not certify stability for every initial condition of a nonminimal realization.', '''
    import math

    for t in [0, 1, 2]:
        state = [math.exp(t), math.exp(-t)]
        output = state[1]
        print(f't={t}: hidden={state[0]:.6f}, visible={output:.6f}')
    print('A=diag(1,-1), B=(0,1), C=(0,1): H(s)=1/(s+1)')
    ''')

add('capstone', 'Check a changed acquisition and response', 'Recover the two complex coefficients and verify the startup output for the changed signal.',
    'This exact bin-centered synthetic case supports coefficient recovery. It does not justify the same accuracy for noisy, off-bin or already aliased recordings. The time-domain integral checks the filter output independently.', '''
    import math
    import numpy as np
    from scipy.integrate import quad

    fs, count, rate, initial = 32, 64, 4*math.pi, -.5
    modes = [(2, 1.5, math.pi/4), (5, .75, -math.pi/3)]
    def signal(t):
        return sum(a*math.cos(2*math.pi*f*t+p) for f,a,p in modes)
    def steady(t):
        return sum(a*rate/math.hypot(rate,2*math.pi*f)*math.cos(2*math.pi*f*t+p-math.atan2(2*math.pi*f,rate)) for f,a,p in modes)

    data = np.array([signal(n/fs) for n in range(count)])
    coefficients = np.fft.fft(data)/count
    for frequency, _, _ in modes:
        value = coefficients[int(frequency*count/fs)]
        print(f'{frequency} Hz: coefficient={value.real:.6f}{value.imag:+.6f}j')
    print('reconstruction:', np.max(abs(np.fft.ifft(coefficients*count).real-data)) < 1e-12)
    t = .25
    total = steady(t)+(initial-steady(0))*math.exp(-rate*t)
    independent = initial*math.exp(-rate*t)+quad(lambda u: rate*math.exp(-rate*(t-u))*signal(u),0,t)[0]
    print(f'output at .25 s: {total:.6f}; integral: {independent:.6f}')
    ''', 'Python 3.12; NumPy and SciPy')

target = Path('src/learn/data/complex-transforms-examples.js')
target.write_text('// Complete programs executed by scripts/generate-complex-transforms-examples.py.\nexport const complexTransformExamples = ' + json.dumps(examples, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
print(f'Executed and saved {len(examples)} complete programs.')
