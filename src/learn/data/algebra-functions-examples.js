export const algebraFunctionsExamples = [
  {
    "id": "measurement",
    "title": "Add compatible measurements exactly",
    "question": "Why must the fractional pieces have a common denominator?",
    "code": "from fractions import Fraction\n\nfirst = Fraction(3, 4)\nsecond = Fraction(1, 2)\ntotal_litres = first + second\nminutes = Fraction(5, 2)\nrate_litres_per_minute = total_litres / minutes\nprint(\"litres:\", total_litres)\nprint(\"litres per minute:\", rate_litres_per_minute)\nprint(\"up 20%, then down 20%:\", 100 * Fraction(6, 5) * Fraction(4, 5))\n",
    "expected": "litres: 5/4\nlitres per minute: 1/2\nup 20%, then down 20%: 96"
  },
  {
    "id": "equations",
    "title": "Keep exact solutions and zero cases separate",
    "question": "What should the solver return when the variable coefficient is zero?",
    "code": "from fractions import Fraction\n\n\ndef solve_linear(a, b, c):\n    a, b, c = map(Fraction, (a, b, c))\n    if a == 0:\n        return \"all real numbers\" if b == c else \"no solution\"\n    return (c - b) / a\n\n\nfor coefficients in [(3, 6, 21), (-3, 5, -7), (0, 7, 7), (0, 7, 8)]:\n    result = solve_linear(*coefficients)\n    print(coefficients, \"->\", result)\n    if isinstance(result, Fraction):\n        a, b, c = coefficients\n        assert a * result + b == c\n",
    "expected": "(3, 6, 21) -> 5\n(-3, 5, -7) -> 4\n(0, 7, 7) -> all real numbers\n(0, 7, 8) -> no solution"
  },
  {
    "id": "functions",
    "title": "Order operations and restrict the inverse",
    "question": "Why is the inverse of the nonnegative square branch different from a reciprocal?",
    "code": "import math\n\n\ndef affine(x):\n    return 2 * x + 1\n\n\ndef square(x):\n    return x * x\n\n\ndef inverse_nonnegative_square(y):\n    if not math.isfinite(y) or y < 0:\n        raise ValueError(\"The inverse input must be finite and nonnegative.\")\n    return math.sqrt(y)\n\n\nx = 2\nprint(\"square after affine:\", square(affine(x)))\nprint(\"affine after square:\", affine(square(x)))\nprint(\"inverse square branch:\", inverse_nonnegative_square(9))\nprint(\"reciprocal of square:\", 1 / square(3))\ntry:\n    inverse_nonnegative_square(-1)\nexcept ValueError as error:\n    print(type(error).__name__ + \":\", error)\n",
    "expected": "square after affine: 25\naffine after square: 9\ninverse square branch: 3.0\nreciprocal of square: 0.1111111111111111\nValueError: The inverse input must be finite and nonnegative."
  },
  {
    "id": "roots",
    "title": "Check roots in the original question",
    "question": "Which square-root candidate survives substitution, and which input remains excluded after cancellation?",
    "code": "import math\n\n\ndef vertex_roots(h, k):\n    if any(type(v) is not int or abs(v) > 100 for v in (h, k)):\n        raise ValueError(\"This teaching helper uses integer h,k from -100 to 100.\")\n    if k > 0:\n        return ()\n    if k == 0:\n        return (float(h),)\n    radius = math.sqrt(-k)\n    return (h - radius, h + radius)\n\n\nfor h, k in [(3, -4), (-1, 0), (2, 3)]:\n    print(\"vertex\", (h, k), \"roots:\", vertex_roots(h, k))\nfor x in (-1, 2):\n    valid = x >= 0 and math.isclose(math.sqrt(x + 2), x)\n    print(\"sqrt(x+2)=x candidate\", x, \"valid:\", valid)\nfor x in (0, 1, 2):\n    value = None if x == 1 else (x * x - 1) / (x - 1)\n    print(\"rational input\", x, \"output:\", value)\n",
    "expected": "vertex (3, -4) roots: (1.0, 5.0)\nvertex (-1, 0) roots: (-1.0,)\nvertex (2, 3) roots: ()\nsqrt(x+2)=x candidate -1 valid: False\nsqrt(x+2)=x candidate 2 valid: True\nrational input 0 output: 1.0\nrational input 1 output: None\nrational input 2 output: 3.0"
  },
  {
    "id": "powers",
    "title": "Keep signs, powers and roots distinct",
    "question": "Can you predict all five outputs before running?",
    "code": "from fractions import Fraction\nimport math\n\nprint(\"(-3)**2:\", (-3) ** 2)\nprint(\"-3**2:\", -(3**2))\nprint(\"2**-3 exactly:\", Fraction(2) ** -3)\nprint(\"cube root of -8:\", math.cbrt(-8))\nprint(\"sqrt((-3)**2):\", math.sqrt((-3) ** 2))\n",
    "expected": "(-3)**2: 9\n-3**2: -9\n2**-3 exactly: 1/8\ncube root of -8: -2.0\nsqrt((-3)**2): 3.0"
  },
  {
    "id": "growth",
    "title": "Invert growth and decay with the right units",
    "question": "Does a 20% increase mean a continuous rate of 0.2 per period?",
    "code": "import math\n\ninitial = 100.0\nfractional_rate = 0.2\ntarget = 300.0\nk = math.log1p(fractional_rate)\ncrossing = math.log(target / initial) / k\nprint(f\"continuous rate per period: {k:.6f}\")\nprint(f\"triple at period: {crossing:.6f}\")\nprint(f\"substitution: {initial * math.exp(k * crossing):.6f}\")\nfor n in (6, 7):\n    print(f\"period {n}: {initial * (1 + fractional_rate)**n:.6f}\")\nhalf_life = math.log(0.5) / math.log(0.8)\nprint(f\"80% retained each period, half at: {half_life:.6f}\")\n",
    "expected": "continuous rate per period: 0.182322\ntriple at period: 6.025685\nsubstitution: 300.000000\nperiod 6: 298.598400\nperiod 7: 358.318080\n80% retained each period, half at: 3.106284"
  },
  {
    "id": "doubling",
    "title": "Decide a whole-step threshold using integers",
    "question": "Why is checking powers directly safer than rounding a logarithm at an exact power of two?",
    "code": "def first_doubling(initial, target):\n    if any(type(v) is not int or v <= 0 for v in (initial, target)):\n        raise ValueError(\"Use positive integers.\")\n    steps, value = 0, initial\n    while value < target:\n        steps += 1\n        value *= 2\n    return steps, value\n\n\nfor target in (1, 8, 9, 32):\n    steps, value = first_doubling(1, target)\n    print(\"target\", target, \"steps\", steps, \"value\", value)\ndelays = [2**i for i in range(5)]\nprint(\"five retry delays:\", delays)\nprint(\"total delay:\", sum(delays))\nprint(\"closed form:\", 2**5 - 1)\n",
    "expected": "target 1 steps 0 value 1\ntarget 8 steps 3 value 8\ntarget 9 steps 4 value 16\ntarget 32 steps 5 value 32\nfive retry delays: [1, 2, 4, 8, 16]\ntotal delay: 31\nclosed form: 31"
  },
  {
    "id": "subdivision",
    "title": "Inspect finite subdivisions without mistaking them for a proof",
    "question": "What changes as one unit of nominal growth is spread over more subperiods?",
    "code": "import math\n\nfor m in (1, 2, 4, 12, 100, 10000):\n    value = math.exp(m * math.log1p(1 / m))\n    print(f\"m={m:5d} -> {value:.6f}\")\nprint(f\"e       -> {math.e:.6f}\")\n",
    "expected": "m=    1 -> 2.000000\nm=    2 -> 2.250000\nm=    4 -> 2.441406\nm=   12 -> 2.613035\nm=  100 -> 2.704814\nm=10000 -> 2.718146\ne       -> 2.718282"
  },
  {
    "id": "precision",
    "title": "Compute a tiny change without first rounding it away",
    "question": "Why can two mathematically equal expressions print different values?",
    "code": "import math\nfrom decimal import Decimal, localcontext\n\nx = 1e-16\nprint(f\"log(1+x): {math.log(1+x):.8e}\")\nprint(f\"log1p(x): {math.log1p(x):.8e}\")\nprint(f\"exp(x)-1: {math.exp(x)-1:.8e}\")\nprint(f\"expm1(x): {math.expm1(x):.8e}\")\nwith localcontext() as context:\n    context.prec = 60\n    exact_input = Decimal(\"1e-16\")\n    print(f\"decimal log: {(1 + exact_input).ln():.8E}\")\n    print(f\"decimal exp change: {exact_input.exp()-1:.8E}\")\n",
    "expected": "log(1+x): 0.00000000e+00\nlog1p(x): 1.00000000e-16\nexp(x)-1: 0.00000000e+00\nexpm1(x): 1.00000000e-16\ndecimal log: 1.00000000E-16\ndecimal exp change: 1.00000000E-16"
  }
];
