// Complete Python programs; expected stdout captured in the recorded author environment.
export const singleVariableCalculusExamples = [
  {
    "id": "motion-secants",
    "title": "Approach one velocity through exact interval slopes",
    "question": "Do approaching secants give the same rate from both time directions?",
    "code": "from fractions import Fraction\n\n\ndef position(t):\n    return t * (t - 3) ** 2\n\n\nbase = Fraction(2)\nvelocity = 3 * (base - 1) * (base - 3)\nfor h in [Fraction(1), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 10)]:\n    change = position(base + h) - position(base)\n    slope = change / h\n    remainder = change - velocity * h\n    print(f\"h={h}: change={change} m, secant={slope} m/s, remainder={remainder} m\")\nprint(\"limiting velocity:\", velocity, \"m/s\")\nprint(\"times 0,1,3,4; positions:\", [position(Fraction(t)) for t in [0, 1, 3, 4]])\n",
    "expected": "h=1: change=-2 m, secant=-2 m/s, remainder=1 m\nh=1/2: change=-11/8 m, secant=-11/4 m/s, remainder=1/8 m\nh=-1/2: change=11/8 m, secant=-11/4 m/s, remainder=-1/8 m\nh=1/10: change=-299/1000 m, secant=-299/100 m/s, remainder=1/1000 m\nlimiting velocity: -3 m/s\ntimes 0,1,3,4; positions: [Fraction(0, 1), Fraction(4, 1), Fraction(0, 1), Fraction(4, 1)]"
  },
  {
    "id": "limit-guarantee",
    "title": "Check the algebra behind a closeness guarantee",
    "question": "What does a chosen delta guarantee that a finite table alone cannot?",
    "code": "from fractions import Fraction\n\nfor epsilon in [Fraction(1), Fraction(1, 10), Fraction(1, 100)]:\n    delta = min(Fraction(1), epsilon / 5)\n    supremum = delta * (4 + delta)\n    print(f\"epsilon={epsilon}, delta={delta}, supremum error={supremum}\")\n    assert supremum <= epsilon\n\nepsilon, delta = Fraction(1, 10), Fraction(1, 10)\nx = 2 + delta / 2\nerror = abs(x * x - 4)\nprint(f\"bad choice: x={x}, input distance={abs(x-2)}, output error={error}\")\nassert 0 < abs(x - 2) < delta and error >= epsilon\nprint(\"A proof needs every allowed x; this witness refutes the bad choice.\")\n",
    "expected": "epsilon=1, delta=1/5, supremum error=21/25\nepsilon=1/10, delta=1/50, supremum error=201/2500\nepsilon=1/100, delta=1/500, supremum error=2001/250000\nbad choice: x=41/20, input distance=1/20, output error=81/400\nA proof needs every allowed x; this witness refutes the bad choice."
  },
  {
    "id": "local-operations",
    "title": "Separate the first-order contribution from the cross term",
    "question": "Does a zero local prediction imply a zero finite change?",
    "code": "from fractions import Fraction\n\nfor x in [Fraction(0), Fraction(-1, 3), Fraction(-1)]:\n    h = Fraction(1, 10)\n    u, du = 3 * x + 1, 3 * h\n    exact_change = (u + du) ** 2 - u ** 2\n    linear_change = 2 * u * du\n    print(f\"x={x}: exact={exact_change}, linear={linear_change}, remainder={du**2}\")\n    assert exact_change == linear_change + du ** 2\n\na, b, da, db = Fraction(2), Fraction(3), Fraction(1, 5), Fraction(-1, 10)\nexact = (a + da) * (b + db) - a * b\nlinear = b * da + a * db\nprint(f\"product: exact={exact}, linear={linear}, cross term={da*db}\")\nassert exact == linear + da * db\n",
    "expected": "x=0: exact=69/100, linear=3/5, remainder=9/100\nx=-1/3: exact=9/100, linear=0, remainder=9/100\nx=-1: exact=-111/100, linear=-6/5, remainder=9/100\nproduct: exact=19/50, linear=2/5, cross term=-1/50"
  },
  {
    "id": "interval-extrema",
    "title": "Keep endpoints in an optimization problem",
    "question": "How does narrowing the allowed interval change the global minimum?",
    "code": "from fractions import Fraction\n\n\ndef position(t):\n    return t * (t - 3) ** 2\n\n\nfor left, right in [(Fraction(0), Fraction(4)), (Fraction(3, 2), Fraction(5, 2))]:\n    # The derivative factorization 3(t-1)(t-3) proves these are all interior candidates.\n    candidates = sorted({left, right, *[Fraction(t) for t in [1, 3] if left < t < right]})\n    values = [(t, position(t)) for t in candidates]\n    low, high = min(v for _, v in values), max(v for _, v in values)\n    print(f\"interval [{left},{right}]\")\n    print(\"candidates:\", \"; \".join(f\"t={t}: s={v}\" for t, v in values))\n    print(\"minimum at:\", \", \".join(str(t) for t, v in values if v == low))\n    print(\"maximum at:\", \", \".join(str(t) for t, v in values if v == high))\n",
    "expected": "interval [0,4]\ncandidates: t=0: s=0; t=1: s=4; t=3: s=0; t=4: s=4\nminimum at: 0, 3\nmaximum at: 1, 4\ninterval [3/2,5/2]\ncandidates: t=3/2: s=27/8; t=5/2: s=5/8\nminimum at: 5/2\nmaximum at: 3/2"
  },
  {
    "id": "signed-rectangles",
    "title": "Compute signed and absolute contributions separately",
    "question": "What do signed and absolute sums estimate, and can either match exactly by coincidence?",
    "code": "from fractions import Fraction\n\n\ndef velocity(t):\n    return 3 * (t - 1) * (t - 3)\n\n\nfor n in [3, 4, 8]:\n    width = Fraction(4, n)\n    for name, offset in [(\"left\", Fraction(0)), (\"midpoint\", Fraction(1, 2))]:\n        contributions = [velocity((j + offset) * width) * width for j in range(n)]\n        signed = sum(contributions)\n        distance = sum(abs(value) for value in contributions)\n        print(f\"n={n:2}, {name}: displacement={signed}, travel estimate={distance}\")\nprint(\"exact: displacement=4 m; travel=12 m\")\n",
    "expected": "n= 3, left: displacement=68/9, travel estimate=148/9\nn= 3, midpoint: displacement=20/9, travel estimate=92/9\nn= 4, left: displacement=6, travel estimate=12\nn= 4, midpoint: displacement=3, travel estimate=12\nn= 8, left: displacement=9/2, travel estimate=12\nn= 8, midpoint: displacement=15/4, travel estimate=12\nexact: displacement=4 m; travel=12 m"
  },
  {
    "id": "square-riemann-limit",
    "title": "Expose the finite error before taking a limit",
    "question": "Which terms disappear in a right-endpoint sum of x squared on [0,2]?",
    "code": "from fractions import Fraction\n\nfor n in [2, 4, 10]:\n    width = Fraction(2, n)\n    direct = sum((j * width) ** 2 * width for j in range(1, n + 1))\n    formula = Fraction(8, 3) + Fraction(4, n) + Fraction(4, 3 * n * n)\n    print(f\"n={n}: sum={direct}, error={direct-Fraction(8,3)}\")\n    assert direct == formula\nprint(\"limit: 8/3; finite sums are not the limiting integral\")\n",
    "expected": "n=2: sum=5, error=7/3\nn=4: sum=15/4, error=13/12\nn=10: sum=77/25, error=31/75\nlimit: 8/3; finite sums are not the limiting integral"
  },
  {
    "id": "moving-bounds",
    "title": "Differentiate an accumulation with two moving endpoints",
    "question": "Which contribution is lost if the lower endpoint is treated as fixed?",
    "code": "from fractions import Fraction\n\n\ndef primitive(t):\n    return t + t ** 3 / 3\n\n\ndef accumulated(x):\n    return primitive(x * x) - primitive(x)\n\n\nx = Fraction(2)\nderivative = (1 + x ** 4) * 2 * x - (1 + x ** 2)\nprint(\"integral from x to x^2 of (1+t^2), at x=2:\", accumulated(x))\nprint(\"upper contribution:\", (1 + x ** 4) * 2 * x)\nprint(\"lower contribution:\", -(1 + x ** 2))\nprint(\"derivative:\", derivative)\nfor h in [Fraction(1, 10), Fraction(1, 100)]:\n    secant = (accumulated(x + h) - accumulated(x)) / h\n    print(f\"h={h}: secant={float(secant):.6f}\")\n",
    "expected": "integral from x to x^2 of (1+t^2), at x=2: 62/3\nupper contribution: 68\nlower contribution: -5\nderivative: 63\nh=1/10: secant=71.450403\nh=1/100: secant=63.795320"
  },
  {
    "id": "integration-operations",
    "title": "Check three integrals by independently differentiable primitives",
    "question": "Which boundary term, transformed limit or pole belongs to each calculation?",
    "code": "import math\nfrom fractions import Fraction\n\n# u = 1+x^2 changes bounds x=0,2 into u=1,5.\nsubstitution = (Fraction(5) ** 4 - 1) / 4\n# Integration by parts: a primitive of x*exp(x) is (x-1)*exp(x).\nparts = (1 - 1) * math.exp(1) - (0 - 1) * math.exp(0)\n# 1/[x(x+1)] = 1/x - 1/(x+1), on [1,2], away from both poles.\nrational = (math.log(2) - math.log(3)) - (math.log(1) - math.log(2))\nprint(\"substitution integral:\", substitution)\nprint(\"parts integral:\", f\"{parts:.6f}\")\nprint(\"rational integral:\", f\"{rational:.9f}\")\nprint(\"same rational answer log(4/3):\", f\"{math.log(4/3):.9f}\")\n",
    "expected": "substitution integral: 156\nparts integral: 1.000000\nrational integral: 0.287682072\nsame rational answer log(4/3): 0.287682072"
  },
  {
    "id": "exponential-rate",
    "title": "Connect finite compounding with a continuous relative rate",
    "question": "Why are a twenty-percent fraction and a continuous rate of 0.2 different?",
    "code": "import math\n\nfor n in [1, 4, 16, 1024]:\n    log_compound = n * math.log1p(1 / n)\n    lower = 1 / (1 + 1 / n)\n    print(f\"n={n:4}: lower={lower:.9f}, log compound={log_compound:.9f}, upper=1\")\n    assert lower <= log_compound <= 1\nk, initial, t, period = 0.2, 10.0, 2.0, 1.0\ncurrent = initial * math.exp(k * t)\nfraction = math.expm1(k * period)\nprint(\"amount now:\", f\"{current:.6f}\")\nprint(\"instantaneous amount/time:\", f\"{k*current:.6f}\")\nprint(\"one-period fraction:\", f\"{fraction:.6f}\")\nprint(\"continuous rate matching a 20% one-period gain:\", f\"{math.log1p(0.2):.6f}\")\nprint(\"doubling time at k=0.2:\", f\"{math.log(2)/k:.6f}\")\n",
    "expected": "n=   1: lower=0.500000000, log compound=0.693147181, upper=1\nn=   4: lower=0.800000000, log compound=0.892574205, upper=1\nn=  16: lower=0.941176471, log compound=0.969993949, upper=1\nn=1024: lower=0.999024390, log compound=0.999512036, upper=1\namount now: 14.918247\ninstantaneous amount/time: 2.983649\none-period fraction: 0.221403\ncontinuous rate matching a 20% one-period gain: 0.182322\ndoubling time at k=0.2: 3.465736"
  },
  {
    "id": "taylor-remainders",
    "title": "Compare an actual approximation error with a proven bound",
    "question": "Can a higher degree become worse away from the expansion point?",
    "code": "import math\n\nx = 0.5\nfor degree in [1, 3, 5]:\n    polynomial = sum(x ** j / math.factorial(j) for j in range(degree + 1))\n    error = abs(polynomial - math.exp(x))\n    bound = math.exp(x) * abs(x) ** (degree + 1) / math.factorial(degree + 1)\n    print(f\"exp at {x}, degree={degree}: error={error:.9f}, bound={bound:.9f}\")\n    assert error <= bound\nx = 1.5\nfor degree in [2, 4, 8]:\n    polynomial = sum((-1) ** (j + 1) * x ** j / j for j in range(1, degree + 1))\n    print(f\"log(1+x), x={x}, degree={degree}: error={abs(polynomial-math.log1p(x)):.6f}\")\nprint(\"The log function exists here; its power series around zero does not converge here.\")\n",
    "expected": "exp at 0.5, degree=1: error=0.148721271, bound=0.206090159\nexp at 0.5, degree=3: error=0.002887937, bound=0.004293545\nexp at 0.5, degree=5: error=0.000023354, bound=0.000035780\nlog(1+x), x=1.5, degree=2: error=0.541291\nlog(1+x), x=1.5, degree=4: error=0.681916\nlog(1+x), x=1.5, degree=8: error=1.824368\nThe log function exists here; its power series around zero does not converge here."
  },
  {
    "id": "finite-difference-rounding",
    "title": "Do not confuse an analytic limit with floating-point subtraction",
    "question": "What happens when exp(h) rounds to the same number as exp(0)?",
    "code": "import math\n\nfor h in [1e-2, 1e-6, 1e-10, 1e-16]:\n    subtraction = (math.exp(h) - 1) / h\n    stable_quotient = math.expm1(h) / h\n    print(f\"h={h:.0e}: subtract={subtraction:.8f}, expm1={stable_quotient:.8f}\")\nprint(\"analytic derivative at zero: 1\")\nprint(\"These are recorded Python float results, not a proof of the limit.\")\n",
    "expected": "h=1e-02: subtract=1.00501671, expm1=1.00501671\nh=1e-06: subtract=1.00000050, expm1=1.00000050\nh=1e-10: subtract=1.00000008, expm1=1.00000000\nh=1e-16: subtract=0.00000000, expm1=1.00000000\nanalytic derivative at zero: 1\nThese are recorded Python float results, not a proof of the limit."
  },
  {
    "id": "improper-cutoffs",
    "title": "Compute a truncation and the part still missing",
    "question": "Does a small discarded input interval necessarily have a small integral?",
    "code": "import math\n\nfor delta in [1e-2, 1e-4, 1e-6]:\n    kept = 2 * (1 - math.sqrt(delta))\n    missing = 2 * math.sqrt(delta)\n    print(f\"x^(-1/2), delta={delta:.0e}: kept={kept:.6f}, missing={missing:.6f}\")\nfor upper in [10, 100, 1000]:\n    convergent = 1 - 1 / upper\n    divergent = math.log(upper)\n    print(f\"B={upper}: integral x^(-2)={convergent:.6f}, integral x^(-1)={divergent:.6f}\")\nprint(\"At 1/x across zero, the separate improper integrals diverge; symmetric cancellation is not convergence.\")\n",
    "expected": "x^(-1/2), delta=1e-02: kept=1.800000, missing=0.200000\nx^(-1/2), delta=1e-04: kept=1.980000, missing=0.020000\nx^(-1/2), delta=1e-06: kept=1.998000, missing=0.002000\nB=10: integral x^(-2)=0.900000, integral x^(-1)=2.302585\nB=100: integral x^(-2)=0.990000, integral x^(-1)=4.605170\nB=1000: integral x^(-2)=0.999000, integral x^(-1)=6.907755\nAt 1/x across zero, the separate improper integrals diverge; symmetric cancellation is not convergence."
  },
  {
    "id": "weighted-material",
    "title": "Accumulate mass and moment before dividing",
    "question": "Why does the center of mass differ from the midpoint of a nonuniform rod?",
    "code": "from fractions import Fraction\n\n# On 0<=x<=2 m: density rho(x) = (2+x) kg/m in these numerical units.\nlength = Fraction(2)\nmass = 2 * length + length ** 2 / 2\nmoment = length ** 2 + length ** 3 / 3\ncenter = moment / mass\n# Separate force model F(x)=(3+2x) N, same displacement interval.\nwork = 3 * length + length ** 2\nprint(\"mass:\", mass, \"kg\")\nprint(\"first moment:\", moment, \"kg m\")\nprint(\"center:\", center, \"m\")\nprint(\"geometric midpoint:\", length / 2, \"m\")\nprint(\"work:\", work, \"J\")\n",
    "expected": "mass: 6 kg\nfirst moment: 20/3 kg m\ncenter: 10/9 m\ngeometric midpoint: 1 m\nwork: 10 J"
  },
  {
    "id": "decay-steps",
    "title": "Compare a rate equation with finite tangent steps",
    "question": "Can a numerical update turn a positive decay model negative?",
    "code": "import math\n\ninitial, rate, final_time = 3.0, -2.0, 1.0\nexact = initial * math.exp(rate * final_time)\nfor steps in [1, 2, 4, 8, 32]:\n    h = final_time / steps\n    value = initial\n    for _ in range(steps):\n        value += h * rate * value\n    print(f\"steps={steps:2}: Euler={value:.6f}, exact={exact:.6f}\")\nprint(\"The rate law and initial value define the exact model; finite steps are an approximation.\")\n",
    "expected": "steps= 1: Euler=-3.000000, exact=0.406006\nsteps= 2: Euler=0.000000, exact=0.406006\nsteps= 4: Euler=0.187500, exact=0.406006\nsteps= 8: Euler=0.300339, exact=0.406006\nsteps=32: Euler=0.380366, exact=0.406006\nThe rate law and initial value define the exact model; finite steps are an approximation."
  },
  {
    "id": "changed-motion",
    "title": "Solve a different journey from its derivative and turning points",
    "question": "How far can an object travel while ending exactly where it started?",
    "code": "from fractions import Fraction\n\n\ndef position(t):\n    return t ** 3 - 3 * t ** 2\n\n\n# Its derivative is 3t(t-2), so the only interior turn on [0,3] is t=2.\ntimes = [Fraction(0), Fraction(2), Fraction(3)]\npositions = [position(t) for t in times]\ndisplacement = positions[-1] - positions[0]\ndistance = sum(abs(b - a) for a, b in zip(positions, positions[1:]))\nprint(\"turning positions:\", \", \".join(str(x) for x in positions))\nprint(\"net displacement:\", displacement, \"m\")\nprint(\"total distance:\", distance, \"m\")\nprint(\"average velocity:\", displacement / 3, \"m/s\")\nprint(\"velocity at t=1:\", 3 * (1) * (1 - 2), \"m/s\")\n",
    "expected": "turning positions: 0, -4, 0\nnet displacement: 0 m\ntotal distance: 8 m\naverage velocity: 0 m/s\nvelocity at t=1: -3 m/s"
  }
];
