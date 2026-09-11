"""Complement the author's grid with exact and high-precision changed inputs."""
import contextlib
import io
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/algebra-functions-independent"
packet = json.loads((DIRECTORY / "source-and-models.json").read_text(encoding="utf-8"))
namespaces = {}
for example in packet["examples"]:
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(example["code"], example["id"] + ".py", "exec"), namespace)
    assert captured.getvalue().strip() == example["expected"]
    namespaces[example["id"]] = namespace

# These exact rational coefficients differ from the bounded integer UI inputs.
solver = namespaces["equations"]["solve_linear"]
equations = 0
for a in [Fraction(-7, 3), Fraction(-1, 11), Fraction(0), Fraction(3, 17)]:
    for b, c in [(Fraction(13, 9), Fraction(-2, 7)), (Fraction(5, 6), Fraction(5, 6))]:
        answer = solver(a, b, c)
        if a:
            assert a * answer + b == c
        else:
            assert answer == ("all real numbers" if b == c else "no solution")
        equations += 1

for state in packet["compositions"]:
    x = Fraction(state["x"])
    # Compare the expanded difference, rather than repeat the nested functions.
    assert Fraction(state["squareAfterAffine"]) - Fraction(state["affineAfterSquare"]) == 2 * x * (x + 2)
    assert (state["squareAfterAffine"] == state["affineAfterSquare"]) == (x in (0, -2))
    assert state["recovered"] == x

for state in packet["probes"]:
    x = Fraction(state["x"])
    allowed = not (state["kind"] == "reciprocal" and x == 0) and not (state["kind"] == "square" and state["restricted"] and x < 0)
    assert state["allowed"] == allowed
    if not allowed:
        assert state["y"] is None and not state["preimages"]
    else:
        expected = 2 * x + 1 if state["kind"] == "affine" else x * x if state["kind"] == "square" else 1 / x
        assert math.isclose(state["y"], float(expected), rel_tol=2e-15, abs_tol=1e-15)
        for inverse in state["preimages"]:
            assert (2 * inverse + 1 if state["kind"] == "affine" else inverse**2 if state["kind"] == "square" else 1 / inverse) == state["y"]

maximum_growth_relative_error = 0.0
with localcontext() as context:
    context.prec = 70
    for state in packet["growth"]:
        rate, factor = Decimal(str(state["rate"])), Decimal(str(state["factor"]))
        multiplier = 1 + rate
        for row in state["rows"]:
            expected = 100 * multiplier ** row["t"]
            error = abs(Decimal(str(row["growth"])) - expected) / expected
            maximum_growth_relative_error = max(maximum_growth_relative_error, float(error))
            assert error < Decimal("4e-15")
            assert row["additive"] == 100 + 20 * row["t"]
        if rate == 0:
            assert state["crossing"] == (0 if factor == 1 else None)
        else:
            expected = factor.ln() / multiplier.ln()
            assert abs(Decimal(str(state["crossing"])) - expected) < Decimal("2e-14")
            assert state["future"] == (expected >= 0)
    for state in packet["logStates"]:
        base, exponent = Decimal(str(state["base"])), Decimal(str(state["exponent"]))
        expected = (exponent * base.ln()).exp()
        assert abs(Decimal(str(state["value"])) - expected) / expected < Decimal("5e-16")
        assert state["increasing"] == (base > 1)

# Verify the prose's general quadratic formula, including a negative leading
# coefficient, against polynomials assembled independently from their roots.
quadratics = 0
for a in [Fraction(-7, 3), Fraction(-1, 5), Fraction(2, 9)]:
    for r, s in [(Fraction(-3, 2), Fraction(5, 7)), (Fraction(2, 3), Fraction(2, 3)), (Fraction(-8), Fraction(-1, 4))]:
        b, c = -a * (r + s), a * r * s
        discriminant = b * b - 4 * a * c
        radius = math.sqrt(discriminant)
        actual = sorted([(-float(b) - radius) / (2 * float(a)), (-float(b) + radius) / (2 * float(a))])
        assert all(math.isclose(left, float(right), abs_tol=2e-14) for left, right in zip(actual, sorted([r, s])))
        quadratics += 1

# Exact near-power comparisons avoid inferring the integer decision from a
# rounded log. Initial sizes are non-powers of two, including a 217-bit case.
doubling = namespaces["doubling"]["first_doubling"]
threshold_cases = 0
for initial in [3, 7, 19]:
    for power in [1, 4, 31, 217]:
        for offset in [-1, 0, 1]:
            target = (initial << power) + offset
            answer = doubling(initial, target)
            # Inspect the helper return shape explicitly; it returns count/value.
            count, value = answer
            assert value >= target and value == initial << count
            assert count == 0 or initial << (count - 1) < target
            threshold_cases += 1

result = {
    "at": datetime.now(timezone.utc).isoformat(), "passed": True,
    "production": packet["production"], "actualPrograms": len(namespaces),
    "changedFractionEquations": equations, "compositionPolynomialIdentities": len(packet["compositions"]),
    "domainPreimageStates": len(packet["probes"]), "offPresetGrowthFactors": len(packet["growth"]),
    "growthRelativeError": maximum_growth_relative_error, "decimalLogStates": len(packet["logStates"]),
    "signedGeneralQuadraticChecks": quadratics, "exactNearPowerThresholds": threshold_cases,
}
(DIRECTORY / "native-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "production"}))
