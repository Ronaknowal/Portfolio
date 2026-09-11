"""Author the complete displayed Python programs and capture real stdout.

This captures example outputs; independent numerical verification is separate.
"""

import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = []


def example(identifier, title, question, code):
    code = textwrap.dedent(code).strip() + "\n"
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=20,
    )
    EXAMPLES.append(
        {
            "id": identifier,
            "title": title,
            "question": question,
            "code": code,
            "expected": result.stdout.rstrip("\n"),
        }
    )


example(
    "motion-secants",
    "Approach one velocity through exact interval slopes",
    "Do approaching secants give the same rate from both time directions?",
    r'''
    from fractions import Fraction


    def position(t):
        return t * (t - 3) ** 2


    base = Fraction(2)
    velocity = 3 * (base - 1) * (base - 3)
    for h in [Fraction(1), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 10)]:
        change = position(base + h) - position(base)
        slope = change / h
        remainder = change - velocity * h
        print(f"h={h}: change={change} m, secant={slope} m/s, remainder={remainder} m")
    print("limiting velocity:", velocity, "m/s")
    print("times 0,1,3,4; positions:", [position(Fraction(t)) for t in [0, 1, 3, 4]])
    ''',
)

example(
    "limit-guarantee",
    "Check the algebra behind a closeness guarantee",
    "What does a chosen delta guarantee that a finite table alone cannot?",
    r'''
    from fractions import Fraction

    for epsilon in [Fraction(1), Fraction(1, 10), Fraction(1, 100)]:
        delta = min(Fraction(1), epsilon / 5)
        supremum = delta * (4 + delta)
        print(f"epsilon={epsilon}, delta={delta}, supremum error={supremum}")
        assert supremum <= epsilon

    epsilon, delta = Fraction(1, 10), Fraction(1, 10)
    x = 2 + delta / 2
    error = abs(x * x - 4)
    print(f"bad choice: x={x}, input distance={abs(x-2)}, output error={error}")
    assert 0 < abs(x - 2) < delta and error >= epsilon
    print("A proof needs every allowed x; this witness refutes the bad choice.")
    ''',
)

example(
    "local-operations",
    "Separate the first-order contribution from the cross term",
    "Does a zero local prediction imply a zero finite change?",
    r'''
    from fractions import Fraction

    for x in [Fraction(0), Fraction(-1, 3), Fraction(-1)]:
        h = Fraction(1, 10)
        u, du = 3 * x + 1, 3 * h
        exact_change = (u + du) ** 2 - u ** 2
        linear_change = 2 * u * du
        print(f"x={x}: exact={exact_change}, linear={linear_change}, remainder={du**2}")
        assert exact_change == linear_change + du ** 2

    a, b, da, db = Fraction(2), Fraction(3), Fraction(1, 5), Fraction(-1, 10)
    exact = (a + da) * (b + db) - a * b
    linear = b * da + a * db
    print(f"product: exact={exact}, linear={linear}, cross term={da*db}")
    assert exact == linear + da * db
    ''',
)

example(
    "interval-extrema",
    "Keep endpoints in an optimization problem",
    "How does narrowing the allowed interval change the global minimum?",
    r'''
    from fractions import Fraction


    def position(t):
        return t * (t - 3) ** 2


    for left, right in [(Fraction(0), Fraction(4)), (Fraction(3, 2), Fraction(5, 2))]:
        # The derivative factorization 3(t-1)(t-3) proves these are all interior candidates.
        candidates = sorted({left, right, *[Fraction(t) for t in [1, 3] if left < t < right]})
        values = [(t, position(t)) for t in candidates]
        low, high = min(v for _, v in values), max(v for _, v in values)
        print(f"interval [{left},{right}]")
        print("candidates:", "; ".join(f"t={t}: s={v}" for t, v in values))
        print("minimum at:", ", ".join(str(t) for t, v in values if v == low))
        print("maximum at:", ", ".join(str(t) for t, v in values if v == high))
    ''',
)

example(
    "signed-rectangles",
    "Compute signed and absolute contributions separately",
    "What do signed and absolute sums estimate, and can either match exactly by coincidence?",
    r'''
    from fractions import Fraction


    def velocity(t):
        return 3 * (t - 1) * (t - 3)


    for n in [3, 4, 8]:
        width = Fraction(4, n)
        for name, offset in [("left", Fraction(0)), ("midpoint", Fraction(1, 2))]:
            contributions = [velocity((j + offset) * width) * width for j in range(n)]
            signed = sum(contributions)
            distance = sum(abs(value) for value in contributions)
            print(f"n={n:2}, {name}: displacement={signed}, travel estimate={distance}")
    print("exact: displacement=4 m; travel=12 m")
    ''',
)

example(
    "square-riemann-limit",
    "Expose the finite error before taking a limit",
    "Which terms disappear in a right-endpoint sum of x squared on [0,2]?",
    r'''
    from fractions import Fraction

    for n in [2, 4, 10]:
        width = Fraction(2, n)
        direct = sum((j * width) ** 2 * width for j in range(1, n + 1))
        formula = Fraction(8, 3) + Fraction(4, n) + Fraction(4, 3 * n * n)
        print(f"n={n}: sum={direct}, error={direct-Fraction(8,3)}")
        assert direct == formula
    print("limit: 8/3; finite sums are not the limiting integral")
    ''',
)

example(
    "moving-bounds",
    "Differentiate an accumulation with two moving endpoints",
    "Which contribution is lost if the lower endpoint is treated as fixed?",
    r'''
    from fractions import Fraction


    def primitive(t):
        return t + t ** 3 / 3


    def accumulated(x):
        return primitive(x * x) - primitive(x)


    x = Fraction(2)
    derivative = (1 + x ** 4) * 2 * x - (1 + x ** 2)
    print("integral from x to x^2 of (1+t^2), at x=2:", accumulated(x))
    print("upper contribution:", (1 + x ** 4) * 2 * x)
    print("lower contribution:", -(1 + x ** 2))
    print("derivative:", derivative)
    for h in [Fraction(1, 10), Fraction(1, 100)]:
        secant = (accumulated(x + h) - accumulated(x)) / h
        print(f"h={h}: secant={float(secant):.6f}")
    ''',
)

example(
    "integration-operations",
    "Check three integrals by independently differentiable primitives",
    "Which boundary term, transformed limit or pole belongs to each calculation?",
    r'''
    import math
    from fractions import Fraction

    # u = 1+x^2 changes bounds x=0,2 into u=1,5.
    substitution = (Fraction(5) ** 4 - 1) / 4
    # Integration by parts: a primitive of x*exp(x) is (x-1)*exp(x).
    parts = (1 - 1) * math.exp(1) - (0 - 1) * math.exp(0)
    # 1/[x(x+1)] = 1/x - 1/(x+1), on [1,2], away from both poles.
    rational = (math.log(2) - math.log(3)) - (math.log(1) - math.log(2))
    print("substitution integral:", substitution)
    print("parts integral:", f"{parts:.6f}")
    print("rational integral:", f"{rational:.9f}")
    print("same rational answer log(4/3):", f"{math.log(4/3):.9f}")
    ''',
)

example(
    "exponential-rate",
    "Connect finite compounding with a continuous relative rate",
    "Why are a twenty-percent fraction and a continuous rate of 0.2 different?",
    r'''
    import math

    for n in [1, 4, 16, 1024]:
        log_compound = n * math.log1p(1 / n)
        lower = 1 / (1 + 1 / n)
        print(f"n={n:4}: lower={lower:.9f}, log compound={log_compound:.9f}, upper=1")
        assert lower <= log_compound <= 1
    k, initial, t, period = 0.2, 10.0, 2.0, 1.0
    current = initial * math.exp(k * t)
    fraction = math.expm1(k * period)
    print("amount now:", f"{current:.6f}")
    print("instantaneous amount/time:", f"{k*current:.6f}")
    print("one-period fraction:", f"{fraction:.6f}")
    print("continuous rate matching a 20% one-period gain:", f"{math.log1p(0.2):.6f}")
    print("doubling time at k=0.2:", f"{math.log(2)/k:.6f}")
    ''',
)

example(
    "taylor-remainders",
    "Compare an actual approximation error with a proven bound",
    "Can a higher degree become worse away from the expansion point?",
    r'''
    import math

    x = 0.5
    for degree in [1, 3, 5]:
        polynomial = sum(x ** j / math.factorial(j) for j in range(degree + 1))
        error = abs(polynomial - math.exp(x))
        bound = math.exp(x) * abs(x) ** (degree + 1) / math.factorial(degree + 1)
        print(f"exp at {x}, degree={degree}: error={error:.9f}, bound={bound:.9f}")
        assert error <= bound
    x = 1.5
    for degree in [2, 4, 8]:
        polynomial = sum((-1) ** (j + 1) * x ** j / j for j in range(1, degree + 1))
        print(f"log(1+x), x={x}, degree={degree}: error={abs(polynomial-math.log1p(x)):.6f}")
    print("The log function exists here; its power series around zero does not converge here.")
    ''',
)

example(
    "finite-difference-rounding",
    "Do not confuse an analytic limit with floating-point subtraction",
    "What happens when exp(h) rounds to the same number as exp(0)?",
    r'''
    import math

    for h in [1e-2, 1e-6, 1e-10, 1e-16]:
        subtraction = (math.exp(h) - 1) / h
        stable_quotient = math.expm1(h) / h
        print(f"h={h:.0e}: subtract={subtraction:.8f}, expm1={stable_quotient:.8f}")
    print("analytic derivative at zero: 1")
    print("These are recorded Python float results, not a proof of the limit.")
    ''',
)

example(
    "improper-cutoffs",
    "Compute a truncation and the part still missing",
    "Does a small discarded input interval necessarily have a small integral?",
    r'''
    import math

    for delta in [1e-2, 1e-4, 1e-6]:
        kept = 2 * (1 - math.sqrt(delta))
        missing = 2 * math.sqrt(delta)
        print(f"x^(-1/2), delta={delta:.0e}: kept={kept:.6f}, missing={missing:.6f}")
    for upper in [10, 100, 1000]:
        convergent = 1 - 1 / upper
        divergent = math.log(upper)
        print(f"B={upper}: integral x^(-2)={convergent:.6f}, integral x^(-1)={divergent:.6f}")
    print("At 1/x across zero, the separate improper integrals diverge; symmetric cancellation is not convergence.")
    ''',
)

example(
    "weighted-material",
    "Accumulate mass and moment before dividing",
    "Why does the center of mass differ from the midpoint of a nonuniform rod?",
    r'''
    from fractions import Fraction

    # On 0<=x<=2 m: density rho(x) = (2+x) kg/m in these numerical units.
    length = Fraction(2)
    mass = 2 * length + length ** 2 / 2
    moment = length ** 2 + length ** 3 / 3
    center = moment / mass
    # Separate force model F(x)=(3+2x) N, same displacement interval.
    work = 3 * length + length ** 2
    print("mass:", mass, "kg")
    print("first moment:", moment, "kg m")
    print("center:", center, "m")
    print("geometric midpoint:", length / 2, "m")
    print("work:", work, "J")
    ''',
)

example(
    "decay-steps",
    "Compare a rate equation with finite tangent steps",
    "Can a numerical update turn a positive decay model negative?",
    r'''
    import math

    initial, rate, final_time = 3.0, -2.0, 1.0
    exact = initial * math.exp(rate * final_time)
    for steps in [1, 2, 4, 8, 32]:
        h = final_time / steps
        value = initial
        for _ in range(steps):
            value += h * rate * value
        print(f"steps={steps:2}: Euler={value:.6f}, exact={exact:.6f}")
    print("The rate law and initial value define the exact model; finite steps are an approximation.")
    ''',
)

example(
    "changed-motion",
    "Solve a different journey from its derivative and turning points",
    "How far can an object travel while ending exactly where it started?",
    r'''
    from fractions import Fraction


    def position(t):
        return t ** 3 - 3 * t ** 2


    # Its derivative is 3t(t-2), so the only interior turn on [0,3] is t=2.
    times = [Fraction(0), Fraction(2), Fraction(3)]
    positions = [position(t) for t in times]
    displacement = positions[-1] - positions[0]
    distance = sum(abs(b - a) for a, b in zip(positions, positions[1:]))
    print("turning positions:", ", ".join(str(x) for x in positions))
    print("net displacement:", displacement, "m")
    print("total distance:", distance, "m")
    print("average velocity:", displacement / 3, "m/s")
    print("velocity at t=1:", 3 * (1) * (1 - 2), "m/s")
    ''',
)

target = ROOT / "src/learn/data/single-variable-calculus-examples.js"
target.write_text(
    "// Complete Python programs; expected stdout captured in the recorded author environment.\n"
    + "export const singleVariableCalculusExamples = "
    + json.dumps(EXAMPLES, ensure_ascii=False, indent=2)
    + ";\n",
    encoding="utf-8",
)
print(json.dumps({"programs": len(EXAMPLES), "target": str(target)}))
