"""Evaluate proposed Math52 teaching fixtures, not a production lesson verifier.

Exact Fraction identities and integer sqrt enclosures complement actual binary64
and float32/library evaluations. Results are intentionally local and versioned.
"""

from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import scipy
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/conditioning-design/fixtures.json"
checks = 0


def require(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def rational(value):
    return str(F(value))


def decimal_of(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def matrix_vector(matrix, vector):
    return [sum((a * x for a, x in zip(row, vector)), F(0)) for row in matrix]


def exact_solve_2(matrix, rhs):
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("Singular exact matrix")
    return [
        (d * rhs[0] - b * rhs[1]) / determinant,
        (a * rhs[1] - c * rhs[0]) / determinant,
    ]


def naive(values):
    total = 0.0
    for value in values:
        total += value
    return total


def balanced(values):
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    middle = len(values) // 2
    return balanced(values[:middle]) + balanced(values[middle:])


def kahan(values):
    total = correction = 0.0
    for value in values:
        adjusted = value - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    return total


def neumaier(values):
    total = correction = 0.0
    for value in values:
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
    return total + correction


def rounding_fixture():
    grid = [F(i, 8) for i in range(8, 17)] + [F(i, 4) for i in range(9, 17)]
    rows = []
    for target, expected in [
        (F(17, 16), F(1)),
        (F(19, 16), F(5, 4)),
        (F(17, 8), F(2)),
        (F(19, 8), F(5, 2)),
    ]:
        distance = min(abs(value - target) for value in grid)
        tied = [value for value in grid if abs(value - target) == distance]
        chosen = [
            value for value in tied if int(value * (8 if value < 2 else 4)) % 2 == 0
        ][0]
        require(chosen == expected, "Toy tie-to-even rounding")
        rows.append(
            {"input": str(target), "tied": list(map(str, tied)), "chosen": str(chosen)}
        )
    require(
        sys.float_info.radix == 2 and sys.float_info.mant_dig == 53,
        "This fixture requires binary64 Python floats",
    )
    u = 2.0**-53
    require(1 + u == 1 and 1 + 3 * u == 1 + 4 * u, "Binary64 ties")
    require(F(float(10**16 + 1)) - (10**16 + 1) == -1, "Input rounding")
    return {
        "toy": rows,
        "unitRoundoff": u,
        "gapAboveOne": math.ulp(1.0),
        "smallestSubnormal": math.ulp(0.0),
        "intendedInteger": str(10**16 + 1),
        "storedInteger": rational(float(10**16 + 1)),
    }


def cancellation_fixture():
    rows = []
    bits = 160
    for sign in (-1, 1):
        for exponent in range(1, 61):
            x = F(sign, 2**exponent)
            scaled = (2**exponent + sign) * 2 ** (2 * bits - exponent)
            lower_integer = math.isqrt(scaled)
            lower = F(lower_integer, 2**bits) - 1
            upper = F(lower_integer + (lower_integer**2 != scaled), 2**bits) - 1
            with localcontext() as context:
                context.prec = 150
                reference = (1 + decimal_of(x)).sqrt() - 1
                require(
                    decimal_of(lower) <= reference <= decimal_of(upper),
                    "Independent Decimal reference in exact sqrt enclosure",
                )
            native_x = float(x)
            root = math.sqrt(1 + native_x)
            direct = root - 1
            repaired = native_x / (root + 1)
            row = {
                "sign": sign,
                "exponent": exponent,
                "x": str(x),
                "referenceLower": str(lower),
                "referenceUpper": str(upper),
                "referenceDecimal": str(reference),
                "direct": direct,
                "repaired": repaired,
                "directRelativeError": str(
                    abs(Decimal.from_float(direct) - reference) / abs(reference)
                ),
                "repairedRelativeError": str(
                    abs(Decimal.from_float(repaired) - reference) / abs(reference)
                ),
            }
            require(
                float(row["repairedRelativeError"]) < 4 * sys.float_info.epsilon,
                "Observed repaired error on bounded dyadic fixtures",
            )
            rows.append(row)
    require(
        next(row for row in rows if row["sign"] == 1 and row["exponent"] == 54)[
            "direct"
        ]
        == 0,
        "The selected cancellation failure must actually occur",
    )
    return {
        "referenceMethod": "Exact integer-square-root rational enclosure, width <= 2^-160; independent Decimal150 check",
        "rows": rows,
    }


def sensitivity_fixture():
    rows = []
    for epsilon in [F(1, 2), F(1, 16), F(1, 128), F(1, 100)]:
        for delta in [F(-1, 256), F(0), F(1, 256), F(1, 1000)]:
            matrix = [[F(1), F(1)], [F(1), 1 + epsilon]]
            rhs = [F(2), 2 + epsilon + delta]
            solution = exact_solve_2(matrix, rhs)
            expected = [1 - delta / epsilon, 1 + delta / epsilon]
            require(solution == expected, "Changed near-parallel exact solve")
            condition = (2 + epsilon) ** 2 / epsilon
            relative_input = abs(delta) / (2 + epsilon)
            forward_error = abs(delta / epsilon)
            require(
                forward_error <= condition * relative_input, "Finite perturbation bound"
            )
            rows.append(
                {
                    "epsilon": str(epsilon),
                    "delta": str(delta),
                    "solution": list(map(str, solution)),
                    "conditionInfinity": str(condition),
                    "forwardRelativeInfinity": str(forward_error),
                    "bound": str(condition * relative_input),
                }
            )
    return rows


def backward_fixture():
    rows = []
    for power in (3, 6, 9):
        scale = F(1, 10**power)
        matrix = [[F(1), F(0)], [F(0), scale]]
        rhs, approximate = [F(1), scale], [F(1), F(0)]
        residual = [b - a for b, a in zip(rhs, matrix_vector(matrix, approximate))]
        eta = scale / 2
        altered_matrix = [[F(1), F(0)], [eta, scale]]
        altered_rhs = [F(1), scale - eta]
        require(
            matrix_vector(altered_matrix, approximate) == altered_rhs,
            "Normwise backward witness solves perturbed problem",
        )
        denominator = [
            sum(abs(a) * abs(x) for a, x in zip(row, approximate)) + abs(b)
            for row, b in zip(matrix, rhs)
        ]
        component = max(abs(r) / d for r, d in zip(residual, denominator))
        require(component == 1, "Componentwise perturbation keeps tiny row meaningful")
        require(exact_solve_2(matrix, rhs) == [1, 1], "Forward reference")
        rows.append(
            {
                "scale": str(scale),
                "residualInfinity": str(scale),
                "forwardInfinity": "1",
                "etaNormwise": str(eta),
                "etaComponentwise": str(component),
                "scaledEtaNormwise": "1/2",
                "scaledEtaComponentwise": "1",
                "jointFiniteBound": "2",
            }
        )
    return rows


def sums_fixture():
    fixtures = [
        [1e16, 1.0, -1e16],
        [2.0**53, 1.0, 1.0, 1.0, 1.0],
        [1e16, -1e16, 1.0],
        [1e16, 3.0, -1e16],
        [1.0, -1.0],
    ]
    rows = []
    for values in fixtures:
        exact = sum(map(F, values), F(0))
        absolute_sum = sum((abs(F(value)) for value in values), F(0))
        gamma = F(len(values) - 1, 2**53 - len(values) + 1)
        calculated = {
            name: method(values)
            for name, method in [
                ("naive", naive),
                ("balanced", balanced),
                ("kahan", kahan),
                ("neumaier", neumaier),
                ("fsum", math.fsum),
            ]
        }
        require(
            abs(F(calculated["naive"]) - exact) <= gamma * absolute_sum,
            "Actual naive arithmetic respects declared bound",
        )
        require(F(calculated["fsum"]) == exact, "Chosen exact fsum references")
        rows.append(
            {
                "values": values,
                "exactStoredInputSum": str(exact),
                "conditionComponentwise": (
                    str(absolute_sum / abs(exact)) if exact else None
                ),
                "observed": calculated,
                "naiveAbsoluteBound": str(gamma * absolute_sum),
            }
        )
    require(
        rows[0]["observed"]["kahan"] == 0 and rows[0]["observed"]["neumaier"] == 1,
        "Avoid promising classic Kahan always repairs this order",
    )
    return rows


def refinement_fixture():
    rows = []
    for exponent in (12, 16, 20, 24):
        epsilon = 2.0**-exponent
        matrix = np.array([[1.0, 1.0], [1.0, 1.0 + epsilon]])
        rhs = matrix @ np.array([1 / 3, 2 / 3])
        exact_matrix = [[F(value) for value in row] for row in matrix]
        exact_rhs = list(map(F, rhs))
        exact_solution = exact_solve_2(exact_matrix, exact_rhs)
        row = {
            "exponent": exponent,
            "matrix64": matrix.tolist(),
            "rhs64": rhs.tolist(),
            "exactStoredInputSolution": list(map(str, exact_solution)),
            "states": [],
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                factor = lu_factor(matrix.astype(np.float32))
            require(factor[0].dtype == np.float32, "Factor actually uses float32")
            solution = lu_solve(factor, rhs.astype(np.float32)).astype(np.float64)
            for step in range(3):
                exact_error = max(
                    abs(F(value) - expected)
                    for value, expected in zip(solution, exact_solution)
                )
                native_residual = rhs - matrix @ solution
                exact_residual = [
                    b - a
                    for b, a in zip(
                        exact_rhs, matrix_vector(exact_matrix, list(map(F, solution)))
                    )
                ]
                row["states"].append(
                    {
                        "step": step,
                        "solution": solution.tolist(),
                        "exactForwardInfinity": str(exact_error),
                        "nativeResidual": native_residual.tolist(),
                        "exactResidual": list(map(str, exact_residual)),
                    }
                )
                correction = lu_solve(factor, native_residual.astype(np.float32))
                require(
                    correction.dtype == np.float32,
                    "Correction reuses low precision factor",
                )
                solution += correction
            require(
                F(row["states"][-1]["exactForwardInfinity"])
                < F(row["states"][0]["exactForwardInfinity"]),
                "Observed refinement improves chosen stored-input problem",
            )
        except LinAlgWarning as error:
            row["failure"] = str(error)
            require(exponent == 24, "Only intended low-precision singularity")
        rows.append(row)
    return rows


def least_squares_fixture():
    epsilon = 2.0**-27
    matrix = np.array([[1.0, 1.0], [1.0, 1.0 + epsilon], [1.0, 1.0 - epsilon]])
    rhs = np.array([0.0, -epsilon, epsilon])
    normal = matrix.T @ matrix
    require(
        np.array_equal(normal, np.full((2, 2), 3.0)),
        "Actual normal matrix loses weak direction",
    )
    exact_gram = [
        [sum(F(matrix[k, i]) * F(matrix[k, j]) for k in range(3)) for j in range(2)]
        for i in range(2)
    ]
    require(exact_gram[1][1] == 3 + F(2, 2**54), "Exact full-rank Gram correction")
    q, r = np.linalg.qr(matrix, mode="reduced")
    qr_solution = np.linalg.solve(r, q.T @ rhs)
    svd_solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    require(
        max(abs(qr_solution - [1, -1])) < 1e-7, "QR retains selected weak direction"
    )
    return {
        "epsilon": epsilon,
        "normalFloat64": normal.tolist(),
        "normalExact": [list(map(str, row)) for row in exact_gram],
        "qrSolution": qr_solution.tolist(),
        "lstsqSolution": svd_solution.tolist(),
        "condition2": float(np.linalg.cond(matrix)),
    }


def recurrence_fixture():
    propagation = []
    for q in [F(-1, 2), F(1, 2), F(9, 10), F(1), F(11, 10)]:
        for alternating in (False, True):
            error, radius = F(1, 16), F(1, 16)
            for step in range(1, 21):
                disturbance = F((-1) ** step if alternating else 1, 100)
                error = q * error + disturbance
                radius = abs(q) * radius + F(1, 100)
                explicit = abs(q) ** step * F(1, 16) + F(1, 100) * sum(
                    abs(q) ** j for j in range(step)
                )
                require(
                    abs(error) <= radius == explicit,
                    "Signed trajectory below independent geometric bound",
                )
            propagation.append(
                {
                    "q": str(q),
                    "alternating": alternating,
                    "error20": str(error),
                    "bound20": str(radius),
                }
            )
    refinement = []
    for n in (4, 8, 16, 32):
        h = F(1, n)
        previous, current = F(1), 1 + h * h
        for _ in range(1, n):
            previous, current = current, 3 * current - 2 * previous
        exact_error = (2**n - 1) * h * h
        require(
            current - 1 == exact_error,
            "Consistent unstable recurrence fixed-horizon formula",
        )
        refinement.append(
            {"steps": n, "initialError": str(h * h), "finalError": str(exact_error)}
        )
    # Taylor moments of [3*y(t+h)-2*y(t)-y(t+2h)]/h.
    require(sum([F(-2), F(3), F(-1)]) == 0, "Derivative stencil kills constants")
    require(
        sum(c * i for i, c in enumerate([-2, 3, -1])) == 1,
        "Derivative coefficient is one",
    )
    require(
        sum(F(c * i * i, 2) for i, c in enumerate([-2, 3, -1])) == F(-1, 2),
        "First-order local consistency coefficient",
    )
    nonnormal = []
    matrix = [[F(1, 2), F(10)], [F(0), F(1, 2)]]
    vector = [F(0), F(1)]
    for step in range(1, 17):
        vector = matrix_vector(matrix, vector)
        expected = [10 * step * F(1, 2) ** (step - 1), F(1, 2) ** step]
        require(vector == expected, "Stable eigenvalues allow finite transient growth")
        if step in (1, 2, 4, 8, 16):
            nonnormal.append({"step": step, "error": list(map(str, vector))})
    return {
        "propagation": propagation,
        "unstableRefinement": refinement,
        "nonnormalTransient": nonnormal,
    }


def capstone_fixture():
    epsilon = F(1, 2**16)
    matrix = [[F(1), F(1)], [F(1), 1 + epsilon]]
    rhs_float = np.array(matrix, dtype=float) @ np.array([1 / 3, 2 / 3])
    rhs = list(map(F, rhs_float))
    center = exact_solve_2(matrix, rhs)
    budget = F(1, 10000)
    intervals = []
    for uncertainty in (F(1, 2**24), F(1, 2**30)):
        radius = uncertainty / epsilon
        for sign in (-1, 1):
            perturbed = exact_solve_2(matrix, [rhs[0], rhs[1] + sign * uncertainty])
            require(
                perturbed == [center[0] - sign * radius, center[1] + sign * radius],
                "Capstone uncertainty endpoints independently solve exact equations",
            )
        intervals.append(
            {
                "readingUncertainty": str(uncertainty),
                "amountRadius": str(radius),
                "withinBudget": radius <= budget,
            }
        )
    require(
        not intervals[0]["withinBudget"] and intervals[1]["withinBudget"],
        "Numerical repair alone cannot certify the first data-uncertainty budget",
    )
    return {
        "epsilon": str(epsilon),
        "exactStoredInputCenter": list(map(str, center)),
        "requestedAbsoluteBudget": str(budget),
        "intervals": intervals,
    }


def main():
    result = {
        "scope": "Design fixtures only; no production lesson/browser/build validation",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "rounding": rounding_fixture(),
        "cancellation": cancellation_fixture(),
        "sensitivity": sensitivity_fixture(),
        "backward": backward_fixture(),
        "summation": sums_fixture(),
        "refinement": refinement_fixture(),
        "leastSquares": least_squares_fixture(),
        "recurrence": recurrence_fixture(),
        "capstone": capstone_fixture(),
        "tolerances": {
            "numpyDefaultNearZero": bool(np.isclose(1e-9, 2e-9)),
            "mathDefaultNearZero": math.isclose(1e-9, 2e-9),
            "explicitUnitTolerance": math.isclose(
                1e-9, 2e-9, rel_tol=1e-6, abs_tol=1e-12
            ),
        },
    }
    result["checks"] = checks
    result["completedAt"] = datetime.now(timezone.utc).isoformat()
    result["scriptSHA256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"checks": checks, "output": str(OUT), "completedAt": result["completedAt"]}
        )
    )


if __name__ == "__main__":
    main()
