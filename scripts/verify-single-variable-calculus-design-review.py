"""Complementary finite contract checks, not the final lesson verification suite."""

import contextlib
import io
import json
import math
from fractions import Fraction
from pathlib import Path

import mpmath as mp
from scipy.integrate import quad


data = json.loads(Path("scratch/single-variable-calculus-design-review/cases.json").read_text(encoding="utf-8"))


def close(actual, expected, tolerance=3e-12):
    assert math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


for case in data["limitCases"]:
    epsilon, delta = Fraction(case["epsilon"]), Fraction(case["delta"])
    state = case["state"]
    jump = 2 if state["kind"] == "jump" else 0
    # Monotonic error to the right of 2 gives an open-boundary supremum.
    expected = (2 + delta) ** 2 - 4 + jump <= epsilon
    assert state["guaranteed"] == expected
    witness = state["witness"]
    if expected:
        assert witness is None
    else:
        x = Fraction(witness["inputExact"])
        displacement = Fraction(witness["displacementExact"])
        error = Fraction(witness["errorExact"])
        assert x == 2 + displacement and 0 < displacement < delta
        assert error == x * x - 4 + jump and error >= epsilon

for state in data["rates"]:
    t, h = Fraction(str(state["baseTime"])), Fraction(str(state["increment"]))
    # Expanded polynomial, rather than reusing the model's factorization.
    position = lambda x: x ** 3 - 6 * x ** 2 + 9 * x
    exact = position(t + h) - position(t)
    derivative = 3 * t * t - 12 * t + 9
    close(state["exactChange"], exact)
    close(state["secant"], exact / h)
    close(state["remainder"], exact - derivative * h)

for state in data["accumulations"]:
    upper = state["upper"]
    points = [t for t in [1, 3] if t < upper]
    signed = quad(lambda t: 3 * t * t - 12 * t + 9, 0, upper, points=points)[0]
    distance = quad(lambda t: abs(3 * t * t - 12 * t + 9), 0, upper, points=points)[0]
    close(state["exactDisplacement"], signed)
    close(state["exactDistance"], distance)
    end = Fraction(str(upper))
    width = end / state["panelCount"]
    shift = {"left": 0, "midpoint": Fraction(1, 2), "right": 1}[state["method"]]
    terms = []
    for i in range(state["panelCount"]):
        t = (i + shift) * width
        terms.append((3 * t * t - 12 * t + 9) * width)
    close(state["signedEstimate"], sum(terms))
    close(state["distanceEstimate"], sum(map(abs, terms)))

for state in data["compositions"]:
    x, h = Fraction(str(state["input"])), Fraction(str(state["increment"]))
    f = lambda x: 9 * x * x + 6 * x + 1
    close(state["exactChange"], f(x + h) - f(x))
    close(state["derivative"], 18 * x + 6)
    close(state["remainder"], f(x + h) - f(x) - (18 * x + 6) * h)

mp.mp.dps = 75
for state in data["taylors"]:
    x = mp.mpf(str(state["input"]))
    f = mp.exp if state["kind"] == "exp" else lambda x: mp.log(1 + x)
    polynomial = mp.polyval(list(reversed(mp.taylor(f, 0, state["degree"]))), x)
    close(state["polynomial"], polynomial)
    close(state["exactValue"], f(x))
    assert abs(polynomial - f(x)) <= mp.mpf(str(state["remainderBound"])) * (1 + mp.mpf("1e-12"))

programs = []
for example in data["examples"]:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(example["code"], example["id"] + ".py", "exec"), {"__name__": "__main__"})
    actual = capture.getvalue().strip()
    assert actual == example["expected"].strip(), example["id"]
    programs.append(example["id"])

rectangle = next(e for e in data["examples"] if e["id"] == "signed-rectangles")
assert "148/9" in rectangle["expected"] and "92/9" in rectangle["expected"]
assert "coincid" in rectangle["question"]
print(json.dumps({
    "passed": True,
    "exactLimitDecisionsAndWitnesses": len(data["limitCases"]),
    "fractionSecants": len(data["rates"]),
    "quadratureAndFractionAccumulations": len(data["accumulations"]),
    "expandedCompositionChanges": len(data["compositions"]),
    "highPrecisionTaylorCases": len(data["taylors"]),
    "actualPrograms": programs,
    "signedRectangleRepairedOutput": rectangle["expected"],
    "limitations": ["No final body or lab reviewed here.", "Floating-point visual rendering of exact rational counterexamples still requires author browser checks."]
}))
