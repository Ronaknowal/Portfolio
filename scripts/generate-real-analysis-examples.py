"""Generate complete, actually executed Python examples for Real Analysis."""

import ast
import contextlib
import io
import json
from pathlib import Path

import black


EXAMPLES = [
    ("tail-certificate", "Certify every later estimate", "At tolerance 1/10, does index 9 suffice for a strict error bound?", '''
from fractions import Fraction


def tail_start(tolerance):
    tolerance = Fraction(tolerance)
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")
    return max(1, tolerance.denominator // tolerance.numerator)


for tolerance in [Fraction(1, 10), Fraction(2, 7), Fraction(3, 2)]:
    n = tail_start(tolerance)
    error = Fraction(1, n + 1)
    print(f"epsilon={tolerance}, N={n}, first error={error}, strict={error < tolerance}")
    if n > 1:
        print("previous index fails:", Fraction(1, n) >= tolerance)
print("epsilon=1/10, N=9:", Fraction(1, 10) < Fraction(1, 10))
print("For every k >= N, 1/(k+1) <= 1/(N+1); that inequality covers the tail.")
'''),
    ("exact-bracket", "Keep the interval as exact fractions", "What does the interval width certify that a printed decimal does not?", '''
from fractions import Fraction


def square_root_bracket(target, steps, left=1, right=2):
    target, left, right = map(Fraction, (target, left, right))
    if type(steps) is not int or steps < 0 or not 0 <= left < right:
        raise ValueError("Use ordered nonnegative endpoints and a nonnegative integer step count")
    if not left * left <= target <= right * right:
        raise ValueError("The endpoint squares must enclose the target")
    for _ in range(steps):
        midpoint = (left + right) / 2
        if midpoint * midpoint < target:
            left = midpoint
        else:
            right = midpoint
    return left, right


for steps in [0, 3, 8]:
    left, right = square_root_bracket(2, steps)
    print(f"steps={steps}, interval=[{left}, {right}], width={right-left}")
    print("squared endpoints straddle 2:", left * left <= 2 <= right * right)
    print("midpoint absolute-error bound:", (right-left)/2)
print("The bound uses completeness and the invariant, not sqrt() as a test oracle.")
'''),
    ("cauchy-blocks", "Compare a small step with an entire later block", "Can a shrinking next increment coexist with a block gap of at least 1/2?", '''
from fractions import Fraction


def block_gap(n, family):
    if type(n) is not int or n < 1 or family not in ("harmonic", "telescoping"):
        raise ValueError("Use positive integer n and a known family")
    return sum((Fraction(1, k) if family == "harmonic" else Fraction(1, k*(k+1)))
               for k in range(n+1, 2*n+1))


for n in [2, 8, 32]:
    harmonic = block_gap(n, "harmonic")
    telescoping = block_gap(n, "telescoping")
    print(f"N={n}: next harmonic step={Fraction(1,n+1)}")
    print(f"  harmonic block >= 1/2: {harmonic >= Fraction(1,2)}")
    expected = Fraction(1,n+1)-Fraction(1,2*n+1)
    print(f"  telescoping block={telescoping}; identity={telescoping == expected}")
print("The harmonic lower bound holds for every N: N terms, each at least 1/(2N).")
'''),
    ("series-bounds", "Bracket a sum before trusting its digits", "Which tail estimate applies to a positive geometric sum and which to an alternating one?", '''
from fractions import Fraction


def geometric_partial(ratio, n):
    ratio = Fraction(ratio)
    if abs(ratio) >= 1 or type(n) is not int or n < 0:
        raise ValueError("Use |ratio|<1 and integer n>=0")
    partial = sum(ratio**k for k in range(n+1))
    exact_sum = 1/(1-ratio)
    tail_bound = abs(ratio)**(n+1)/(1-abs(ratio))
    return partial, exact_sum, tail_bound


partial, limit, bound = geometric_partial(Fraction(-2, 3), 6)
print("geometric partial:", partial)
print("exact limit / actual error / absolute tail bound:", limit, abs(limit-partial), bound)
n = 10
alternating = sum(Fraction((-1)**(k+1), k) for k in range(1,n+1))
next_partial = alternating + Fraction(1,n+1)
print("alternating harmonic limit bracket:", alternating, next_partial)
print("bracket width:", next_partial-alternating)
print("The decreasing next-term magnitudes prove the bracket; its decimal width is not the proof.")
'''),
    ("power-witness", "Inspect a fixed input and a moving witness", "Why does changing n while keeping x fixed answer a different question from changing x with n?", '''
from fractions import Fraction
from math import exp, log

fixed = Fraction(3, 4)
for n in [2, 8, 32]:
    witness = exp(-log(2)/n)
    print(f"n={n}: fixed (3/4)^n={float(fixed**n):.8f}")
    print(f"  moving x_n={witness:.8f}, numerical x_n^n={witness**n:.8f}")
    print(f"  [0,3/4] exact sup bound={(fixed**n)}")
print("On [0,1), sup x^n = 1, unattained. Analytically (2^(-1/n))^n = 1/2.")
print("On [0,1], the limit at 1 is 1; the near-1 error still has supremum 1.")
'''),
    ("triangle-grid", "Let exact geometry expose a missed peak", "What can a zero sampled maximum fail to tell you?", '''
from fractions import Fraction


def triangle(x, n, height):
    x, height = Fraction(x), Fraction(height)
    return height * max(Fraction(0), 1-abs(n*x-1))


n, intervals = 64, 20
for height in [Fraction(1), Fraction(n), Fraction(1,n)]:
    samples = [triangle(Fraction(j,intervals),n,height) for j in range(intervals+1)]
    area = height/n
    square_area = 2*height**2/(3*n)
    print(f"height={height}: sampled max={max(samples)}, true peak={height}")
    print(f"  area={area}, squared L2 error={square_area}, witness={Fraction(1,n)}")
    assert triangle(Fraction(1,n),n,height) == height
print("n>2*intervals puts the support before the first positive grid point.")
'''),
    ("integral-interchange", "Compare an integral with an all-domain bound", "Does a sharper formula for one family make the uniform-convergence theorem unnecessary?", '''
from math import cos

# f_n(x)=x+sin(n*x)/n on [0,2]. Integrate each term analytically.
for n in [4, 16, 64]:
    integral = 2+(1-cos(2*n))/n**2
    theorem_bound = 2/n
    sharper_bound = 2/n**2
    print(f"n={n}: integral={integral:.9f}, error={abs(integral-2):.9f}")
    print(f"  uniform theorem bound={theorem_bound:.6f}, family-specific bound={sharper_bound:.6f}")
print("Different domain: (1/n)*indicator_[0,n] on [0,infinity) has sup 1/n but integral 1.")
'''),
    ("derivative-interchange", "Keep height and slope errors separate", "What happens to the derivative at zero when the curve itself tends uniformly to zero?", '''
from math import sin, cos, pi


def value_and_derivative(n, power, x):
    if type(n) is not int or n < 1 or power not in (1,2):
        raise ValueError("Use a positive integer index and power 1 or 2")
    return sin(n*x)/n**power, cos(n*x)/n**(power-1)


for power in [1,2]:
    for n in [3,4,12]:
        _, at_zero = value_and_derivative(n,power,0)
        _, at_pi = value_and_derivative(n,power,pi)
        print(f"power={power}, n={n}: function bound={1/n**power:.6f}, derivative at 0={at_zero:.6f}, at pi={at_pi:.6f}")
print("Drifting constants f_n(x)=n have derivative 0; one convergent base value is still needed.")
'''),
    ("series-endpoint", "Check the endpoint after proving the interior", "Can the functions converge uniformly while the derivative series fails at x=1?", '''
from fractions import Fraction


def inverse_square_sums(n, x):
    x = Fraction(x)
    if type(n) is not int or n < 1 or abs(x) > 1:
        raise ValueError("Use n>=1 and |x|<=1")
    value = sum(x**k/Fraction(k*k) for k in range(1,n+1))
    derivative = sum(x**(k-1)/Fraction(k) for k in range(1,n+1))
    return value, derivative


for n in [8,32,128]:
    endpoint, derivative = inverse_square_sums(n,1)
    interior, interior_derivative = inverse_square_sums(n,Fraction(1,2))
    print(f"n={n}: f_n(1)={float(endpoint):.8f}, derivative sum at 1={float(derivative):.8f}")
    print(f"  f_n(1/2)={float(interior):.8f}, derivative at 1/2={float(interior_derivative):.8f}, uniform tail <= 1/{n}")
print("The endpoint derivative sum is harmonic. The uniform tail bound belongs to the function series.")
'''),
    ("bernstein-weights", "Build a polynomial value from exact weights", "Why does a weighted average reproduce linear functions but usually change a corner?", '''
from fractions import Fraction
from math import comb


def weights(n, x):
    x = Fraction(x)
    if type(n) is not int or n < 1 or not 0 <= x <= 1:
        raise ValueError("Use degree>=1 and x in [0,1]")
    return [comb(n,k)*x**k*(1-x)**(n-k) for k in range(n+1)]


n, x, corner = 8, Fraction(2,5), Fraction(3,10)
mass = weights(n,x)
mean = sum(p*Fraction(k,n) for k,p in enumerate(mass))
variance = sum(p*(Fraction(k,n)-x)**2 for k,p in enumerate(mass))
approximation = sum(p*abs(Fraction(k,n)-corner) for k,p in enumerate(mass))
print("weight sum:", sum(mass))
print("weighted node mean:", mean)
print("weighted squared deviation:", variance, "formula:", x*(1-x)/n)
print("target / polynomial:", abs(x-corner), approximation)
print("squared error <= variance:", (approximation-abs(x-corner))**2 <= variance)
print("A constant or linear target is reproduced exactly by the first two identities.")
'''),
    ("typewriter", "Follow the same observer through shrinking intervals", "Does small probability at each late index mean one observer is eventually never visited?", '''
from fractions import Fraction


def visit(block, position, observer):
    if type(block) is not int or block < 0:
        raise ValueError("Use a nonnegative integer block")
    observer = Fraction(observer)
    count = 2**block
    if type(position) is not int or not 0 <= position < count or not 0 <= observer < 1:
        raise ValueError("Use a valid dyadic block, position and observer in [0,1)")
    return Fraction(position,count) <= observer < Fraction(position+1,count)


for observer in [Fraction(1,3), Fraction(1,2)]:
    for block in range(1,5):
        count = 2**block
        hits = [j for j in range(count) if visit(block,j,observer)]
        print(f"observer={observer}, block={block}: hit positions={hits}, per-index probability={Fraction(1,count)}")
print("Every block partitions [0,1): one hit and, from block 1, at least one miss for each observer.")
'''),
    ("tail-mass", "A bounded mean can hide tail mass at larger values", "Can one common cutoff make the tail expectation small for every index?", '''
from fractions import Fraction


def spike_tail(n, cutoff):
    if type(n) is not int or n < 1 or cutoff < 0:
        raise ValueError("Use n>=1 and a nonnegative cutoff")
    # X_n=n on an event of probability 1/n, and zero otherwise.
    return Fraction(1) if n > cutoff else Fraction(0)


for cutoff in [2,10,100]:
    n = cutoff+1
    print(f"cutoff={cutoff}, choose n={n}: mean=1, tail mean={spike_tail(n,cutoff)}, P(X_n>0)={Fraction(1,n)}")
print("For each cutoff the supremum tail mean is 1, so the family is not uniformly integrable.")
print("A common (1+delta)-moment bound C instead gives tail mean <= C/cutoff^delta.")
'''),
    ("approximation-report", "Certify a changed polynomial-integral approximation", "If a sufficient bound exceeds the budget, has the actual approximation necessarily failed?", '''
from fractions import Fraction


def approximation_report(n, corner, slope, tolerance):
    corner, slope, tolerance = map(Fraction, (corner,slope,tolerance))
    if type(n) is not int or n < 1 or not 0 <= corner <= 1 or slope < 0 or tolerance <= 0:
        raise ValueError("Use degree>=1, corner in [0,1], slope>=0 and tolerance>0")
    node_values = [slope*abs(Fraction(k,n)-corner) for k in range(n+1)]
    # Integral of each Bernstein basis polynomial is exactly 1/(n+1).
    polynomial_integral = sum(node_values)/(n+1)
    target_integral = slope*(corner**2+(1-corner)**2)/2
    error = abs(polynomial_integral-target_integral)
    # Compare the squared Lipschitz bound to avoid rounding a square root.
    certified = slope**2 <= 4*n*tolerance**2
    return polynomial_integral, target_integral, error, certified


for n in [25,100]:
    result = approximation_report(n,Fraction(3,10),1,Fraction(1,20))
    print(f"n={n}: polynomial integral={result[0]}, target={result[1]}, exact error={result[2]}")
    print("  certified by uniform bound for tolerance 1/20:", result[3])
changed = approximation_report(100,Fraction(2,5),2,Fraction(1,10))
print("changed corner=2/5, slope=2, n=100, tolerance=1/10:", changed)
print("Exact finite sums and integrals; the uniform bound controls all x. No grid maximum is used as proof.")
''')
]


def main():
    examples = []
    for identifier, title, question, raw in EXAMPLES:
        code = black.format_str(raw.strip() + "\n", mode=black.Mode(line_length=88))
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(raw))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, f"real-analysis-{identifier}.py", "exec"), {"__name__": "__main__"})
        examples.append({"id": identifier, "title": title, "question": question, "code": code, "expected": output.getvalue().rstrip()})
    destination = Path("src/learn/data/real-analysis-examples.js")
    destination.write_text("// Complete Python programs, formatted and executed before capture.\nexport const realAnalysisExamples = " + json.dumps(examples, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    print(f"Executed and captured {len(examples)} complete real-analysis programs.")


if __name__ == "__main__":
    main()
