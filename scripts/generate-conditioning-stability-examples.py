"""Author-owned complete Math52 programs; format without changing Python AST and execute."""
import ast
import contextlib
import io
import json
from pathlib import Path

import black


EXAMPLES = [
    ("rounding", "Inspect the stored input before judging the answer", "Does converting the rounded float to Decimal restore the missing one?", '''
import math
import sys
from decimal import Decimal
from fractions import Fraction

if sys.float_info.radix != 2 or sys.float_info.mant_dig != 53:
    raise RuntimeError("This experiment requires binary64 Python floats")
u = 2.0 ** -53
print("gap above 1:", math.ulp(1.0).hex())
print("unit roundoff:", u.hex())
print("1 + u:", (1.0 + u).hex())
print("1 + 3u:", (1.0 + 3*u).hex())
intended = "10000000000000001"
stored = float(intended)
print("intended decimal:", Decimal(intended))
print("stored as Decimal:", Decimal.from_float(stored))
print("stored exactly:", Fraction.from_float(stored))
print("smallest normal:", sys.float_info.min.hex())
print("smallest subnormal:", math.ulp(0.0).hex())
'''),
    ("cancellation", "Check equivalent expressions against an exact enclosure", "At x = 2^-54, which intermediate first loses the increment?", '''
from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import isqrt, sqrt


def reference_enclosure(exponent, sign=1, bits=160):
    if type(exponent) is not int or not 0 <= exponent <= 60 or type(sign) is not int or sign not in (-1, 0, 1):
        raise ValueError("Use exponent 0..60 and sign -1, 0 or 1")
    if type(bits) is not int or not 100 <= bits <= 300:
        raise ValueError("Use 100..300 reference bits")
    scaled = (2**exponent + sign) * 2**(2*bits-exponent)
    root = isqrt(scaled)
    low = F(root, 2**bits) - 1
    high = F(root + (root*root != scaled), 2**bits) - 1
    return low, high


for sign, exponent in [(1, 26), (1, 54), (-1, 54), (-1, 0), (0, 54)]:
    x = sign * 2.0**-exponent
    direct = sqrt(1+x)-1
    repaired = x/(sqrt(1+x)+1)
    low, high = reference_enclosure(exponent, sign)
    with localcontext() as context:
        context.prec = 100
        reference = (1 + Decimal.from_float(x)).sqrt()-1
        assert Decimal(low.numerator)/low.denominator <= reference <= Decimal(high.numerator)/high.denominator
        print(f"x={x:.8e}: direct={direct:.8e}, repaired={repaired:.8e}")
        print(f"  Decimal100 reference={reference:.12e}; interval width <= 2^-160")
        print("  direct exact within enclosure:", low <= F(direct) <= high)
print("The interval encloses the exact function, not necessarily either rounded result.")
'''),
    ("small-changes", "Use APIs designed to retain a small change", "Why can exp(x)-1 and log(1+x) return zero for a nonzero x?", '''
import math
from decimal import Decimal, localcontext

x = 2.0**-54
with localcontext() as context:
    context.prec = 100
    exact_input = Decimal.from_float(x)
    references = [exact_input.exp()-1, (1+exact_input).ln()]
for name, naive, specialized, reference in [
    ("exp change", math.exp(x)-1, math.expm1(x), references[0]),
    ("log change", math.log(1+x), math.log1p(x), references[1]),
]:
    print(f"{name}: naive={naive:.10e}, specialized={specialized:.10e}")
    print(f"  same-input Decimal100 reference={reference:.10e}")
print("Specialized does not mean exact for every input or every library build.")
'''),
    ("sensitivity", "Solve the changed measurement equations exactly", "Does the condition bound predict the exact error for every perturbation?", '''
from fractions import Fraction as F


def measurement_solution(epsilon, delta):
    epsilon, delta = F(epsilon), F(delta)
    if epsilon <= 0:
        raise ValueError("This inverse formula requires positive separation")
    second = (epsilon+delta)/epsilon
    first = 2-second
    condition = (2+epsilon)**2/epsilon
    input_relative = abs(delta)/(2+epsilon)
    error = max(abs(first-1), abs(second-1))
    return (first, second), error, condition*input_relative


for epsilon, delta in [(F(1,16), F(1,256)), (F(1,128), -F(1,256)), (F(1,128), 0)]:
    answer, error, bound = measurement_solution(epsilon, delta)
    print(f"epsilon={epsilon}, delta={delta}: x={answer[0]}, {answer[1]}")
    print(f"  relative infinity error={error}; bound={bound}")
try:
    measurement_solution(0, 0)
except ValueError as error:
    print("rejected:", error)
'''),
    ("backward", "Make the nearby problem explicit", "Why can the normwise model alter an original zero while the componentwise model cannot?", '''
from fractions import Fraction as F


def diagnostics(matrix, rhs, approximate):
    if not matrix or any(len(row) != len(approximate) for row in matrix) or len(rhs) != len(matrix):
        raise ValueError("Use a nonempty compatible matrix, RHS and vector")
    a = [[F(value) for value in row] for row in matrix]
    b, x = list(map(F, rhs)), list(map(F, approximate))
    residual = [bi-sum(aij*xj for aij,xj in zip(row,x)) for row,bi in zip(a,b)]
    a_norm = max(sum(abs(value) for value in row) for row in a)
    b_norm, x_norm = max(map(abs,b)), max(map(abs,x))
    denominator = a_norm*x_norm+b_norm
    normwise = max(map(abs,residual))/denominator if denominator else F(0)
    components = []
    for row,bi,ri in zip(a,b,residual):
        di = sum(abs(aij)*abs(xj) for aij,xj in zip(row,x))+abs(bi)
        if not di and ri:
            raise ValueError("No finite componentwise perturbation explains this residual")
        components.append(abs(ri)/di if di else F(0))
    return residual, normwise, max(components)


small = F(1,10**6)
for row_scale in [small,F(1)]:
    a = [[1,0],[0,row_scale]]
    b, x = [1,row_scale], [1,0]
    residual, eta, component = diagnostics(a,b,x)
    changed_a21 = eta
    changed_b2 = row_scale-eta
    assert changed_a21*x[0]+row_scale*x[1] == changed_b2
    print(f"second row scale={row_scale}: r={residual[0]}, {residual[1]}")
    print(f"  normwise eta={eta}; componentwise eta={component}; forward error=1")
    print(f"  witness A21={changed_a21}, b2={changed_b2}")
print("Both exact problems have solution (1,1). Scaling the equations also scales uncertainty.")
'''),
    ("summation", "Trace three repairs and an order-dependent counterexample", "Does the balanced tree always beat the left fold? Does Kahan always recover the missing one?", '''
import math
from fractions import Fraction as F


def sum_methods(values):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(x) or abs(x)>1e100 for x in values):
        raise ValueError("Use nonempty finite inputs with magnitude at most 1e100")
    total = correction = kahan = kahan_correction = 0.0
    for value in values:
        updated = total+value
        if not math.isfinite(updated):
            raise ValueError("Nonfinite accumulation")
        correction += ((total-updated)+value) if abs(total)>=abs(value) else ((value-updated)+total)
        total = updated
        adjusted = value-kahan_correction
        next_kahan = kahan+adjusted
        kahan_correction = (next_kahan-kahan)-adjusted
        kahan = next_kahan
    def balanced(items):
        if len(items)==1:
            return items[0]
        middle = len(items)//2
        return balanced(items[:middle])+balanced(items[middle:])
    results = (total, balanced(values), kahan, total+correction, math.fsum(values))
    if not all(map(math.isfinite,results)):
        raise ValueError("Nonfinite result")
    return results


for values in [[1e16,1.,-1e16],[1e16,-1e16,1.],[2.**53,1.,1.,1.,1.],[1e16,3.,-1e16],[1.,-1.]]:
    exact = sum(map(F,values),F(0))
    print("inputs:",values,"exact stored-input sum:",exact)
    print("naive, balanced, Kahan, Neumaier, fsum:",sum_methods(values))
    condition = sum((abs(F(x)) for x in values),F(0))/abs(exact) if exact else None
    print("componentwise relative condition:",condition)
'''),
    ("refinement", "Reuse a low-precision factor while inspecting a more accurate residual", "What is the exact reference: the original thirds or the solution of the stored A and b?", '''
import warnings
import numpy as np
from fractions import Fraction as F
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve


def refine_two_channel(exponent, corrections=2):
    if type(exponent) is not int or not 8<=exponent<=24 or type(corrections) is not int or not 0<=corrections<=5:
        raise ValueError("Use exponent 8..24 and 0..5 corrections")
    epsilon = 2.**-exponent
    a = np.array([[1.,1.],[1.,1.+epsilon]],dtype=np.float64)
    b = a@np.array([1/3,2/3],dtype=np.float64)
    second = (F(b[1])-F(b[0]))/F(epsilon)
    exact = [F(b[0])-second,second]
    with warnings.catch_warnings():
        warnings.simplefilter("error",LinAlgWarning)
        factor = lu_factor(a.astype(np.float32))
    x = lu_solve(factor,b.astype(np.float32)).astype(np.float64)
    states = []
    for step in range(corrections+1):
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite iterate")
        error = max(abs(F(xi)-truth) for xi,truth in zip(x,exact))
        residual = b-a@x
        states.append((step,x.copy(),error,residual.copy()))
        if step<corrections:
            correction = lu_solve(factor,residual.astype(np.float32))
            assert correction.dtype==np.float32 and factor[0].dtype==np.float32
            x += correction
    return exact, states


for exponent in [12,20,24]:
    print("separation exponent:",exponent)
    try:
        exact, states = refine_two_channel(exponent)
        print("exact stored-input x:",str(exact[0]),str(exact[1]))
        for step,x,error,residual in states:
            print(f"  step {step}: ({x[0]:.12f}, {x[1]:.12f}); exact error={float(error):.5e}")
    except LinAlgWarning:
        print("rejected: the float32 factor is singular; do not continue refinement")
'''),
    ("least-squares", "Inspect the direction that Gram formation loses", "Is the mathematical design matrix rank deficient, or did forming A transpose A erase a distinction?", '''
from fractions import Fraction as F
import numpy as np

epsilon = 2.**-27
a = np.array([[1.,1.],[1.,1.+epsilon],[1.,1.-epsilon]])
b = np.array([0.,-epsilon,epsilon])
exact_gram22 = sum(F(value)**2 for value in a[:,1])
gram = a.T@a
print("exact Gram(2,2) - 3:",exact_gram22-3)
print("rounded Gram:")
print(gram)
try:
    normal_answer = np.linalg.solve(gram,a.T@b)
    print("normal equations:",normal_answer)
except np.linalg.LinAlgError:
    print("normal equations: singular rounded system")
q,r = np.linalg.qr(a,mode="reduced")
answer = np.linalg.solve(r,q.T@b)
svd_answer, _, rank, _ = np.linalg.lstsq(a,b,rcond=None)
print("QR:", np.array2string(answer,precision=8))
print("lstsq:", np.array2string(svd_answer,precision=8),"numerical rank:",rank)
print("These are observed values for this fixture; rank thresholds and BLAS builds matter.")
'''),
    ("propagation", "Compare a signed error with a worst-case envelope", "When does an alternating disturbance make the envelope loose?", '''
from fractions import Fraction as F


def propagation(q, n, alternating=False):
    q = F(q)
    if type(n) is not int or not 0<=n<=100:
        raise ValueError("Use 0..100 steps")
    error = F(0)
    for step in range(1,n+1):
        delta = F((-1)**step if alternating else 1,100)
        error = q*error+delta
    bound = F(1,100)*sum((abs(q)**j for j in range(n)),F(0))
    return error,bound


for q in [F(1,2),F(-1,2),F(1),F(11,10)]:
    for alternating in [False,True]:
        error,bound = propagation(q,8,alternating)
        print(f"q={q}, alternating={alternating}: error={error}, bound={bound}")
print("Nonnormal matrix B = [[1/2,10],[0,1/2]], initial error (0,1):")
for n in [1,2,4,8,16]:
    print(n, "steps:",10*n*F(1,2)**(n-1), F(1,2)**n)
'''),
    ("consistency", "Refine a consistent method that amplifies its start error", "Why does the error grow although the starting perturbation shrinks as h squared?", '''
from fractions import Fraction as F


def constant_trajectory(n):
    if type(n) is not int or not 2<=n<=64:
        raise ValueError("Use 2..64 steps")
    h = F(1,n)
    previous,current = F(1),1+h*h
    for _ in range(1,n):
        previous,current = current,3*current-2*previous
    error = current-1
    assert error == (2**n-1)*h*h
    return h*h,error


for n in [4,8,16,32]:
    start_error,final_error = constant_trajectory(n)
    print(f"N={n}, T=1: start error={start_error}, final error={final_error}")
print("For u'=0 the true solution is 1. The unwanted recurrence root is 2.")
'''),
    ("report", "Complete a changed accuracy report", "Can a perfect solve of the recorded central readings certify the requested amount accuracy?", '''
from fractions import Fraction as F
import numpy as np
from scipy.linalg import lu_factor, lu_solve

epsilon = 2.**-16
a = np.array([[1.,1.],[1.,1.+epsilon]],dtype=np.float64)
b = a@np.array([1/3,2/3])
second = (F(b[1])-F(b[0]))/F(epsilon)
center = [F(b[0])-second,second]
factor = lu_factor(a.astype(np.float32))
x = lu_solve(factor,b.astype(np.float32)).astype(np.float64)
print("stored central RHS:",b.tolist())
print("initial float32 solve:",x.tolist())
for _ in range(2):
    residual = b-a@x
    x += lu_solve(factor,residual.astype(np.float32))
arithmetic_error = max(abs(F(value)-target) for value,target in zip(x,center))
print("exact central solution:",str(center[0]),str(center[1]))
print("repaired exact forward infinity error:",arithmetic_error)
budget = F(1,10000)
for uncertainty in [F(1,2**24),F(1,2**30)]:
    radius = uncertainty/F(epsilon)
    combined = radius+arithmetic_error
    print(f"reading bound={uncertainty}; amount radius={radius}; total bound={combined}")
    print("certifies absolute amount budget 1/10000:",combined<=budget)
print("This is a worst-case guarantee over the declared reading interval, not the unknown actual error.")
'''),
]


def main():
    output = []
    for key, title, question, code in EXAMPLES:
        formatted = black.format_str(code.strip() + "\n", mode=black.FileMode())
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(formatted))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(formatted, f"conditioning-{key}.py", "exec"), {"__name__": "__main__"})
        output.append({"id": key, "title": title, "question": question,
                       "code": formatted, "expected": stream.getvalue().rstrip()})
    path = Path("src/learn/data/conditioning-stability-examples.js")
    path.write_text("export const conditioningStabilityExamples = " + json.dumps(output, indent=2, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Executed and captured {len(output)} complete programs: {path}")


if __name__ == "__main__":
    main()
