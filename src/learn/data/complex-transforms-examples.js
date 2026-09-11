// Complete programs executed by scripts/generate-complex-transforms-examples.py.
export const complexTransformExamples = {
  "complex": {
    "title": "Calculate with two coordinates",
    "question": "Predict the product and quotient before running. Does taking the modulus lose information?",
    "language": "python",
    "code": "import cmath\nimport math\n\nz, w = 1 + 2j, 2 - 1j\nproduct = z * w\nprint('product:', product)\nprint('quotient:', product / w)\nprint('modulus:', abs(product))\nprint('angle degrees:', round(math.degrees(cmath.phase(product)), 6))\nroots = [cmath.exp(2j * math.pi * k / 4) for k in range(4)]\nprint('fourth roots:', [(round(v.real), round(v.imag)) for v in roots])\nprint('fourth powers:', [round((v ** 4).real) for v in roots])\n",
    "expected": "product: (4+3j)\nquotient: (1+2j)\nmodulus: 5.0\nangle degrees: 36.869898\nfourth roots: [(1, 0), (0, 1), (-1, 0), (0, -1)]\nfourth powers: [1, 1, 1, 1]",
    "interpretation": "The product has length 5 and a different direction. Its quotient by the original multiplier recovers both coordinates. Zero has no defined argument in the mathematics, even if a library returns a convention.",
    "environment": "Python 3.12 standard library"
  },
  "projection": {
    "title": "Recover both frequency quadratures",
    "question": "The 3 Hz tone is a negative sine. What will its real and imaginary coefficients be?",
    "language": "python",
    "code": "import cmath\nimport math\n\ndef signal(t):\n    return 2 * math.cos(2 * math.pi * t) + math.cos(6 * math.pi * t + math.pi / 2)\n\ndef coefficient(k, count=128):\n    return sum(signal(n / count) * cmath.exp(-2j * math.pi * k * n / count)\n               for n in range(count)) / count\n\nfor k in [0, 1, 2, 3, -3]:\n    value = coefficient(k)\n    real = 0.0 if abs(value.real) < 1e-12 else value.real\n    imag = 0.0 if abs(value.imag) < 1e-12 else value.imag\n    print(f'k={k:2}: {real:.6f} {imag:+.6f}j')\n",
    "expected": "k= 0: 0.000000 +0.000000j\nk= 1: 1.000000 +0.000000j\nk= 2: 0.000000 +0.000000j\nk= 3: 0.000000 +0.500000j\nk=-3: 0.000000 -0.500000j",
    "interpretation": "The imaginary coordinate carries the quarter-cycle phase. These uniformly spaced quadratures are exact in real arithmetic for this finite trigonometric fixture, because its frequency differences do not alias onto zero.",
    "environment": "Python 3.12 standard library"
  },
  "series": {
    "title": "Inspect the first overshoot",
    "question": "Will adding terms make the first peak tend to the square wave height 1?",
    "language": "python",
    "code": "import math\n\ndef partial(t, terms):\n    return sum(4 * math.sin(2 * math.pi * (2*j+1) * t) / (math.pi * (2*j+1))\n               for j in range(terms))\n\nfor terms in [1, 4, 16, 64]:\n    peak_t = 1 / (4 * terms)\n    error_energy = 1 - sum(8 / (math.pi**2 * (2*j+1)**2) for j in range(terms))\n    print(f'{terms:2} terms: t={peak_t:.6f}, peak={partial(peak_t, terms):.6f}, MSE={error_energy:.6f}')\nprint('at the jump:', partial(0, 64))\n",
    "expected": " 1 terms: t=0.250000, peak=1.273240, MSE=0.189431\n 4 terms: t=0.062500, peak=1.184225, MSE=0.050402\n16 terms: t=0.015625, peak=1.179305, MSE=0.012661\n64 terms: t=0.003906, peak=1.179000, MSE=0.003166\nat the jump: 0.0",
    "interpretation": "The peak moves toward the jump while remaining above 1. Its location follows from differentiating this finite sum; the printed values alone do not prove the limiting Gibbs result.",
    "environment": "Python 3.12 standard library"
  },
  "pulse": {
    "title": "Check a pulse transform by integration",
    "question": "If the pulse becomes twice as wide at the same height, how do its DC value and first zero change?",
    "language": "python",
    "code": "import math\nfrom scipy.integrate import quad\n\ndef pulse_transform(frequency, width):\n    if frequency == 0:\n        return width\n    return math.sin(math.pi * width * frequency) / (math.pi * frequency)\n\nfor width in [1.0, 2.0]:\n    for frequency in [0.0, 0.25, 1 / width]:\n        integral = quad(lambda t: math.cos(2 * math.pi * frequency * t), -width/2, width/2)[0]\n        print(f'width={width:g}, f={frequency:g}: closed={pulse_transform(frequency, width):.6f}, integral={integral:.6f}')\n",
    "expected": "width=1, f=0: closed=1.000000, integral=1.000000\nwidth=1, f=0.25: closed=0.900316, integral=0.900316\nwidth=1, f=1: closed=0.000000, integral=0.000000\nwidth=2, f=0: closed=2.000000, integral=2.000000\nwidth=2, f=0.25: closed=1.273240, integral=1.273240\nwidth=2, f=0.5: closed=0.000000, integral=0.000000",
    "interpretation": "The rectangular pulse has area equal to width. Its transform at zero equals that area; its first positive zero is the reciprocal width. The real quadrature checks the closed formula; the imaginary integral vanishes because its integrand is odd.",
    "environment": "Python 3.12; SciPy"
  },
  "dft": {
    "title": "Compute the transform without an FFT",
    "question": "Which coefficient contains the imaginary information for [1,2,0,−1]?",
    "language": "python",
    "code": "import cmath\nimport math\n\ndef dft(values, inverse=False):\n    n = len(values)\n    if not n:\n        raise ValueError('empty transform')\n    direction = 1 if inverse else -1\n    return [sum(value * cmath.exp(direction * 2j * math.pi * k * j / n)\n                for j, value in enumerate(values)) / (n if inverse else 1)\n            for k in range(n)]\n\nvalues = [1, 2, 0, -1]\nspectrum = dft(values)\nprint('coefficients:', [(round(z.real), round(z.imag)) for z in spectrum])\nprint('inverse:', [round(z.real) for z in dft(spectrum, True)])\nprint('time energy:', sum(x*x for x in values))\nprint('frequency energy:', round(sum(abs(z)**2 for z in spectrum) / len(values), 6))\n",
    "expected": "coefficients: [(2, 0), (1, -3), (0, 0), (1, 3)]\ninverse: [1, 2, 0, -1]\ntime energy: 6\nfrequency energy: 6.0",
    "interpretation": "This direct O(N²) calculation exposes every finite sum. The inverse reconstructs the four stored numbers; the energy is 6 in either coordinate system after the 1/N factor.",
    "environment": "Python 3.12 standard library"
  },
  "fft": {
    "title": "Reuse the even and odd work",
    "question": "Which two length-two transforms supply a length-four transform?",
    "language": "python",
    "code": "import cmath\nimport math\n\ndef fft(values):\n    n = len(values)\n    if n == 0 or n & (n - 1):\n        raise ValueError('positive power-of-two length required')\n    if n == 1:\n        return [complex(values[0])]\n    even, odd = fft(values[::2]), fft(values[1::2])\n    rotated = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n//2)]\n    return [even[k] + rotated[k] for k in range(n//2)] + [even[k] - rotated[k] for k in range(n//2)]\n\nvalues = [1, 2, 0, -1]\nprint('even:', fft(values[::2]))\nprint('odd:', fft(values[1::2]))\nprint('combined:', [(round(z.real), round(z.imag)) for z in fft(values)])\n",
    "expected": "even: [(1+0j), (1+0j)]\nodd: [(1+0j), (3+0j)]\ncombined: [(2, 0), (1, -3), (0, 0), (1, 3)]",
    "interpretation": "The recursive algorithm is restricted to positive power-of-two lengths; that is this implementation’s contract, not a limitation of the mathematical DFT or every FFT library.",
    "environment": "Python 3.12 standard library"
  },
  "alias": {
    "title": "Preserve phase when folding an alias",
    "question": "Does the 3 Hz alias retain or negate the 13 Hz cosine’s phase?",
    "language": "python",
    "code": "import math\n\ndef mismatch(sample_rate):\n    return max(abs(math.cos(2*math.pi*13*n/sample_rate + math.pi/3)\n                   - math.cos(2*math.pi*3*n/sample_rate - math.pi/3))\n               for n in range(sample_rate))\n\nprint('16 Hz samples agree:', mismatch(16) < 1e-12)\nprint('32 Hz samples agree:', mismatch(32) < 1e-12)\ninvisible = [math.sin(math.pi*n) for n in range(16)]\nprint('Nyquist sine samples zero:', max(map(abs, invisible)) < 1e-12)\n",
    "expected": "16 Hz samples agree: True\n32 Hz samples agree: False\nNyquist sine samples zero: True",
    "interpretation": "At 16 Hz, 13 Hz is the signed representative −3 Hz. Turning that into a positive-frequency cosine negates the phase. At 32 Hz the two original continuous candidates no longer have identical samples.",
    "environment": "Python 3.12 standard library"
  },
  "window": {
    "title": "Separate amplitude, density and padding",
    "question": "Will padding this fixed window change its integrated finite power estimate?",
    "language": "python",
    "code": "import numpy as np\nfrom scipy.signal import periodogram\n\nfs, count = 64, 64\nn = np.arange(count)\nx = np.cos(2*np.pi*5.5*n/fs)\nw = .5 - .5*np.cos(2*np.pi*n/count)  # periodic Hann\ntarget = np.sum((w*x)**2) / np.sum(w*w)\nfor padded in [64, 256]:\n    frequencies, density = periodogram(x, fs=fs, window=w, nfft=padded,\n                                      detrend=False, scaling='density', return_onesided=True)\n    integrated = np.sum(density) * fs / padded\n    print(f'M={padded}: spacing={fs/padded:.2f} Hz, integral={integrated:.6f}')\nprint(f'weighted mean square={target:.6f}')\nprint('window sum:', np.sum(w), 'squared-weight sum:', np.sum(w*w))\n",
    "expected": "M=64: spacing=1.00 Hz, integral=0.500000\nM=256: spacing=0.25 Hz, integral=0.500000\nweighted mean square=0.500000\nwindow sum: 32.0 squared-weight sum: 24.0",
    "interpretation": "Both transform lengths integrate to the same window-weighted mean square. The density uses squared window weights; coherent sinusoidal amplitude uses their sum. Detrending is explicitly disabled.",
    "environment": "Python 3.12; NumPy and SciPy"
  },
  "convolution": {
    "title": "Find the wrapped tail",
    "question": "Where does the fifth linear-convolution output go in a four-point circular convolution?",
    "language": "python",
    "code": "import numpy as np\n\nx, h = np.array([1, 2, 0, -1]), np.array([1, 1])\nlinear = np.convolve(x, h)\ncircular = np.fft.ifft(np.fft.fft(x, 4) * np.fft.fft(h, 4)).real\npadded = np.fft.ifft(np.fft.fft(x, 5) * np.fft.fft(h, 5)).real\nprint('linear:', linear.tolist())\nprint('circular:', np.rint(circular).astype(int).tolist())\nprint('padded:', np.rint(padded).astype(int).tolist())\n",
    "expected": "linear: [1, 3, 2, -1, -1]\ncircular: [0, 3, 2, -1]\npadded: [1, 3, 2, -1, -1]",
    "interpretation": "The −1 tail wraps onto the first value 1, producing 0. Padding to five points is already sufficient for this particular linear convolution; it need not be a power of two.",
    "environment": "Python 3.12; NumPy"
  },
  "filter": {
    "title": "Verify startup by a second route",
    "question": "The steady response is nonzero at t=0. What must be added to satisfy y(0)=0?",
    "language": "python",
    "code": "import math\nfrom scipy.integrate import quad\n\nrate = 2 * math.pi\ndef signal(t):\n    return 2*math.cos(2*math.pi*t) + math.cos(6*math.pi*t + math.pi/2)\ndef steady(t):\n    return math.sqrt(2)*math.cos(2*math.pi*t-math.pi/4) + math.cos(6*math.pi*t+math.pi/2-math.atan(3))/math.sqrt(10)\ndef total(t, initial=0):\n    return steady(t) + (initial-steady(0))*math.exp(-rate*t)\n\nfor t in [0, .25, .5, 1]:\n    independent = quad(lambda u: rate*math.exp(-rate*(t-u))*signal(u), 0, t)[0]\n    print(f't={t:.2f}: steady={steady(t):.6f}, total={total(t):.6f}, integral={independent:.6f}')\n",
    "expected": "t=0.00: steady=1.300000, total=0.000000, integral=0.000000\nt=0.25: steady=1.100000, total=0.829757, integral=0.829757\nt=0.50: steady=-1.300000, total=-1.356178, integral=-1.356178\nt=1.00: steady=1.300000, total=1.297572, integral=1.297572",
    "interpretation": "The homogeneous correction cancels the steady value at the initial instant and decays. Direct time-domain quadrature independently agrees with the transformed-frequency construction.",
    "environment": "Python 3.12; SciPy"
  },
  "laplace": {
    "title": "A finite integral does not establish convergence",
    "question": "What happens on the boundary σ=−a when ω is zero?",
    "language": "python",
    "code": "import cmath\nimport math\n\ndef truncated(decay, sigma, omega, horizon):\n    if not all(math.isfinite(value) for value in (decay, sigma, omega, horizon)) or horizon < 0:\n        raise ValueError('Use finite inputs and a nonnegative horizon')\n    q = complex(decay+sigma, omega)\n    z = q*horizon\n    if not math.isfinite(abs(q)) or not math.isfinite(abs(z)) or abs(z) > 64:\n        raise ValueError('This bounded teaching helper requires abs(q*T) <= 64')\n    if z == 0:\n        return horizon\n    if abs(z) < 0.5:\n        # Integral/T = sum (-z)^k/(k+1)!; retain a tiny increment near q=0.\n        term = total = 1+0j\n        for k in range(1, 25):\n            term *= -z/(k+1)\n            total += term\n        return horizon*total\n    return (1-cmath.exp(-z))/q\n\nfor sigma in [0, -1, -2]:\n    values = [truncated(1, sigma, 0, t).real for t in [1, 2, 4]]\n    print(f'sigma={sigma}:', [round(v, 6) for v in values])\nprint('right-sided convergence requires sigma > -1')\n",
    "expected": "sigma=0: [0.632121, 0.864665, 0.981684]\nsigma=-1: [1, 2, 4]\nsigma=-2: [1.718282, 6.389056, 53.59815]\nright-sided convergence requires sigma > -1",
    "interpretation": "Here the weighting exactly cancels the signal decay: the integrand is 1 and the finite integral grows like T. A finite number exists at every cutoff although the improper transform does not.",
    "environment": "Python 3.12 standard library"
  },
  "initial": {
    "title": "Carry the initial value through Laplace algebra",
    "question": "For y′+2y=2 and y(0)=3, is Y simply H times the input transform?",
    "language": "python",
    "code": "import sympy as sp\n\nt = sp.symbols('t', nonnegative=True)\ns = sp.symbols('s', positive=True)\nY = (2/s + 3)/(s+2)\ny = 1 + 2*sp.exp(-2*t)\nprint('partial fractions:', sp.apart(Y, s))\nprint('ODE residual:', sp.simplify(sp.diff(y,t)+2*y-2))\nprint('initial:', y.subs(t,0))\nprint('transform agrees:', sp.simplify(sp.laplace_transform(y,t,s,noconds=True)-Y) == 0)\n",
    "expected": "partial fractions: 2/(s + 2) + 1/s\nODE residual: 0\ninitial: 3\ntransform agrees: True",
    "interpretation": "The extra 3/(s+2) is the initial-state contribution. The result y=1+2e^(−2t) satisfies both the differential equation and the prescribed initial value.",
    "environment": "Python 3.12; SymPy"
  },
  "delayed": {
    "title": "A delayed pulse is a difference of shifted steps",
    "question": "What stays continuous when the input switches on at t=1 and off at t=2?",
    "language": "python",
    "code": "import math\n\ndef step_response(t):\n    return 0.0 if t < 0 else -math.expm1(-2*t)\ndef response(t, initial=1):\n    return initial*math.exp(-2*t) + step_response(t-1)-step_response(t-2)\n\nfor t in [0, 1, 1.5, 2, 3]:\n    print(f't={t:g}: y={response(t):.6f}')\nprint('Y(s) = 1/(s+2) + 2*(exp(-s)-exp(-2*s))/(s*(s+2))')\n",
    "expected": "t=0: y=1.000000\nt=1: y=0.135335\nt=1.5: y=0.681908\nt=2: y=0.882980\nt=3: y=0.119498\nY(s) = 1/(s+2) + 2*(exp(-s)-exp(-2*s))/(s*(s+2))",
    "interpretation": "The first-order output is continuous, while its slope changes with the forcing. The expression is built from two delayed step responses and preserves the original initial response separately.",
    "environment": "Python 3.12 standard library"
  },
  "repeated": {
    "title": "Repeated poles retain the time factor",
    "question": "Does a double pole at −1 give the same mode as a simple pole?",
    "language": "python",
    "code": "import sympy as sp\n\nt, s = sp.symbols('t s', positive=True)\nh = t*sp.exp(-t)\nH = 1/(s+1)**2\nprint('ordinary residual after impulse:', sp.simplify(sp.diff(h,t,2)+2*sp.diff(h,t)+h))\nprint('position at 0+:', sp.limit(h,t,0,dir='+'))\nprint('velocity at 0+:', sp.limit(sp.diff(h,t),t,0,dir='+'))\nprint('transform agrees:', sp.simplify(sp.laplace_transform(h,t,s,noconds=True)-H) == 0)\n",
    "expected": "ordinary residual after impulse: 0\nposition at 0+: 0\nvelocity at 0+: 1\ntransform agrees: True",
    "interpretation": "The impulse response t e^(−t) is zero in position at the initial instant but has right derivative 1. A unit impulse in the second-order equation changes velocity, not position.",
    "environment": "Python 3.12; SymPy"
  },
  "shift": {
    "title": "Rewrite the function after shifting time",
    "question": "Is the transform of u(t−1)t just e^(−s)/s²?",
    "language": "python",
    "code": "import math\nfrom scipy.integrate import quad\n\nfor s in [1, 2, 3]:\n    actual = quad(lambda t: t*math.exp(-s*t), 1, math.inf)[0]\n    correct = math.exp(-s)*(1/s**2+1/s)\n    wrong = math.exp(-s)/s**2\n    print(f's={s}: integral={actual:.6f}, correct={correct:.6f}, shortcut={wrong:.6f}')\n",
    "expected": "s=1: integral=0.735759, correct=0.735759, shortcut=0.367879\ns=2: integral=0.101501, correct=0.101501, shortcut=0.033834\ns=3: integral=0.022128, correct=0.022128, shortcut=0.005532",
    "interpretation": "After t=u+1, the remaining function is u+1. The additional 1/s term is essential; shifting only the lower limit does not replace t by t−1.",
    "environment": "Python 3.12; SciPy"
  },
  "hidden": {
    "title": "A stable transfer can hide an unstable state",
    "question": "If the input and output see only the second state, what happens to an initial first state?",
    "language": "python",
    "code": "import math\n\nfor t in [0, 1, 2]:\n    state = [math.exp(t), math.exp(-t)]\n    output = state[1]\n    print(f't={t}: hidden={state[0]:.6f}, visible={output:.6f}')\nprint('A=diag(1,-1), B=(0,1), C=(0,1): H(s)=1/(s+1)')\n",
    "expected": "t=0: hidden=1.000000, visible=1.000000\nt=1: hidden=2.718282, visible=0.367879\nt=2: hidden=7.389056, visible=0.135335\nA=diag(1,-1), B=(0,1), C=(0,1): H(s)=1/(s+1)",
    "interpretation": "The input-output transfer is 1/(s+1), but the unseen first state grows like e^t. Stable visible poles do not certify stability for every initial condition of a nonminimal realization.",
    "environment": "Python 3.12 standard library"
  },
  "capstone": {
    "title": "Check a changed acquisition and response",
    "question": "Recover the two complex coefficients and verify the startup output for the changed signal.",
    "language": "python",
    "code": "import math\nimport numpy as np\nfrom scipy.integrate import quad\n\nfs, count, rate, initial = 32, 64, 4*math.pi, -.5\nmodes = [(2, 1.5, math.pi/4), (5, .75, -math.pi/3)]\ndef signal(t):\n    return sum(a*math.cos(2*math.pi*f*t+p) for f,a,p in modes)\ndef steady(t):\n    return sum(a*rate/math.hypot(rate,2*math.pi*f)*math.cos(2*math.pi*f*t+p-math.atan2(2*math.pi*f,rate)) for f,a,p in modes)\n\ndata = np.array([signal(n/fs) for n in range(count)])\ncoefficients = np.fft.fft(data)/count\nfor frequency, _, _ in modes:\n    value = coefficients[int(frequency*count/fs)]\n    print(f'{frequency} Hz: coefficient={value.real:.6f}{value.imag:+.6f}j')\nprint('reconstruction:', np.max(abs(np.fft.ifft(coefficients*count).real-data)) < 1e-12)\nt = .25\ntotal = steady(t)+(initial-steady(0))*math.exp(-rate*t)\nindependent = initial*math.exp(-rate*t)+quad(lambda u: rate*math.exp(-rate*(t-u))*signal(u),0,t)[0]\nprint(f'output at .25 s: {total:.6f}; integral: {independent:.6f}')\n",
    "expected": "2 Hz: coefficient=0.530330+0.530330j\n5 Hz: coefficient=0.187500-0.324760j\nreconstruction: True\noutput at .25 s: -0.901760; integral: -0.901760",
    "interpretation": "This exact bin-centered synthetic case supports coefficient recovery. It does not justify the same accuracy for noisy, off-bin or already aliased recordings. The time-domain integral checks the filter output independently.",
    "environment": "Python 3.12; NumPy and SciPy"
  }
};
