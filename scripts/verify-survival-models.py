"""Bounded independent Survival oracles against current JS and displayed helpers.

Run through verify-survival-models.mjs. Does not refit the unchanged 17 displayed
programs: their author executions are checked and explicitly reused by hash.
"""

import ast
import hashlib
import importlib.metadata
import itertools
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import cumulative_incidence_competing_risks
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = {}
practice = {}
largest_error = {}


def close(actual, expected, label, tolerance=3e-11):
    expected = float(expected)
    actual = float(actual)
    error = abs(actual - expected)
    largest_error[label.split(":")[0]] = max(
        largest_error.get(label.split(":")[0], 0), error
    )
    assert error <= tolerance * max(1, abs(expected)), (label, actual, expected, error)


def tick(name):
    counts[name] = counts.get(name, 0) + 1


def definitions(example, names, scope):
    """Execute exactly the requested displayed definitions, not their demo main."""
    source = data["examples"][example]["code"]
    parsed = ast.parse(source)
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), f"displayed:{example}", "exec"
        ),
        scope,
    )
    return scope


km_native = definitions(
    "kaplanMeier", ["km_table", "step_value", "restricted_mean"], {"Fraction": F}
)
cox_native = definitions("coxFit", ["cox_terms", "fit_cox"], {"np": np})


def redistributed_mass(times, events):
    """Redistribute each censor's empirical mass among strictly later records.

    All tied failures receive their mass before same-time censors depart. A final
    censored mass is retained at an unidentified tail, rather than assigned a time.
    """
    weights = [F(1, len(times)) for _ in times]
    masses = {}
    for at in sorted(set(times)):
        failures = [i for i, time in enumerate(times) if time == at and events[i]]
        censored = [i for i, time in enumerate(times) if time == at and not events[i]]
        masses[at] = sum((weights[i] for i in failures), F(0))
        later = [i for i, time in enumerate(times) if time > at]
        if later:
            shared = sum((weights[i] for i in censored), F(0)) / len(later)
            for i in later:
                weights[i] += shared
    return masses


for case in data["km"]:
    times, events = case["input"]["times"], case["input"]["events"]
    masses = redistributed_mass(times, events)
    rows = case["table"]["rows"]
    library = KaplanMeierFitter().fit(times, events)
    native_rows = km_native["km_table"](times, events)
    g = F(0)
    for index, row in enumerate(rows):
        at = row["time"]
        expected = 1 - sum(mass for time, mass in masses.items() if time <= at)
        close(row["survival"], expected, "KM:redistribution")
        close(row["survival"], library.predict(at), "KM:library")
        assert native_rows[index][4] == expected
        n, d = row["risk"], row["failures"]
        if n == d:
            assert row["greenwood"] == "Infinity" and row["interval"] is None
            g = None
        elif g is not None:
            g += F(d, n * (n - d))
            close(row["greenwood"], g, "Greenwood:exact")
        if row["interval"] is not None:
            for actual, bound in zip(
                row["interval"], library.confidence_interval_.loc[at]
            ):
                close(actual, bound, "Greenwood:log-log-library")
    expected_median = next(
        (
            at
            for at in sorted(masses)
            if 1 - sum(mass for time, mass in masses.items() if time <= at) <= F(1, 2)
        ),
        None,
    )
    assert case["table"]["median"] == expected_median
    tail = 1 - sum(masses.values())
    for point in case["evaluations"]:
        at = F(str(point["time"]))
        expected_area = (
            sum((mass * min(F(str(time)), at) for time, mass in masses.items()), F(0))
            + tail * at
        )
        expected_survival = 1 - sum(mass for time, mass in masses.items() if time <= at)
        close(point["area"], expected_area, "RMST:truncated-lifetime expectation")
        close(point["survival"], expected_survival, "KM:right-continuity")
        assert km_native["restricted_mean"](native_rows, at) == expected_area
        assert km_native["step_value"](native_rows, at) == expected_survival
    tick("KM/Greenwood/RMST exact+library+actual-native fixtures")
practice["B"] = {"survival": ["4/5", "3/5", "3/10"], "median": 4, "RMST": "33/10"}
assert km_native["restricted_mean"](
    km_native["km_table"]([1, 2, 2, 4, 5], [1, 1, 0, 1, 0]), 5
) == F(33, 10)


def decimal_cox(beta, times, events, features, ties):
    # Direct likelihood only. Derivatives below never use weighted moments.
    x = [Decimal(str(value)) for value in features]
    objective = Decimal(0)
    for at in sorted({time for time, event in zip(times, events) if event}):
        risk = [i for i, time in enumerate(times) if time >= at]
        failed = [i for i, time in enumerate(times) if time == at and events[i]]
        risk_weight = sum((beta * x[i]).exp() for i in risk)
        removed_weight = sum((beta * x[i]).exp() for i in failed)
        objective += beta * sum(x[i] for i in failed)
        for step in range(len(failed)):
            fraction = Decimal(step) / len(failed) if ties == "efron" else Decimal(0)
            objective -= (risk_weight - fraction * removed_weight).ln()
    return objective


for case in data["cox"]:
    inp, actual = case["input"], case["actual"]
    with localcontext() as context:
        context.prec = 75
        beta, step = Decimal(str(inp["beta"])), Decimal("1e-18")
        args = (inp["times"], inp["events"], inp["features"], inp["ties"])
        center = decimal_cox(beta, *args)
        minus = decimal_cox(beta - step, *args)
        plus = decimal_cox(beta + step, *args)
        score = (plus - minus) / (2 * step)
        information = -(plus - 2 * center + minus) / step**2
    native = cox_native["cox_terms"](
        inp["beta"],
        np.array(inp["times"]),
        np.array(inp["events"]),
        np.array(inp["features"]),
        inp["ties"],
    )
    for key, reference, own in zip(
        ["logLikelihood", "score", "information"], [center, score, information], native
    ):
        close(actual[key], reference, "Cox:75-digit likelihood derivatives")
        close(own, reference, "Cox:actual displayed helper")
    tick("Cox objective/score/information derivative fixtures")

for case_index in [0, 12]:
    inp = data["cox"][case_index]["input"]
    time, event, feature = (
        np.array(inp["times"]),
        np.array(inp["events"]),
        np.array(inp["features"]),
    )
    beta, _, status = cox_native["fit_cox"](time, event, feature)
    library = CoxPHSurvivalAnalysis(ties="efron", tol=1e-12, n_iter=200).fit(
        feature[:, None], Surv.from_arrays(event, time)
    )
    direct = minimize_scalar(
        lambda value: -float(
            decimal_cox(
                Decimal(str(value)),
                time.tolist(),
                event.tolist(),
                feature.tolist(),
                "efron",
            )
        ),
        bounds=(-3, 3),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert status == "score tolerance reached"
    close(beta, library.coef_[0], "Cox:changed fit library", 2e-7)
    close(beta, direct.x, "Cox:changed direct optimizer", 2e-7)
    tick("Cox actual fit versus separate likelihood optimizer/library")
changed = cox_native["cox_terms"](
    math.log(2),
    np.array([2, 3, 1]),
    np.array([False, False, True]),
    np.array([0, 1, 2]),
)
close(math.exp(changed[0]), F(4, 7), "Practice D:contribution")
close(changed[1], F(4, 7), "Practice D:score")
close(changed[2], F(26, 49), "Practice D:information")
practice["D"] = {"contribution": "4/7", "score": "4/7", "information": "26/49"}

for case in data["logrank"]:
    inp, actual = case["input"], case["actual"]
    total, variance = F(0), F(0)
    for row in actual["rows"]:
        at = row["time"]
        risk = [i for i, time in enumerate(inp["times"]) if time >= at]
        outcomes = [
            sum(inp["groups"][i] for i in selected)
            for selected in itertools.combinations(risk, row["failures"])
        ]
        expected = F(sum(outcomes), len(outcomes))
        var = sum(((F(value) - expected) ** 2 for value in outcomes), F(0)) / len(
            outcomes
        )
        close(row["expected"], expected, "Logrank:enumerated labels")
        close(row["variance"], var, "Logrank:enumerated label variance")
        total += row["observed"] - expected
        variance += var
    close(actual["difference"], total, "Logrank:total")
    close(actual["variance"], variance, "Logrank:variance")
    if variance:
        groups = np.array(inp["groups"])
        times, events = np.array(inp["times"]), np.array(inp["events"])
        library = logrank_test(
            times[groups == 0],
            times[groups == 1],
            events[groups == 0],
            events[groups == 1],
        )
        close(actual["statistic"], library.test_statistic, "Logrank:library statistic")
    else:
        assert actual["statistic"] is None
    tick("Logrank exact conditional-label distributions")
practice["E"] = {
    "expected": "3/4",
    "U": "1/4",
    "variance": "45/112",
    "failureProbability": 0.4,
    "riskRatio": 0.625,
}
close(data["logrank"][-1]["actual"]["variance"], F(45, 112), "Practice E:variance")
close(
    data["logrank"][-1]["actual"]["difference"],
    F(1, 4),
    "Practice E:observed-minus-expected",
)
comparison_risk = 1 - math.sqrt(0.36)
close(comparison_risk, 0.4, "Practice E:PH probability")
close(comparison_risk / (1 - 0.36), F(5, 8), "Practice E:probability ratio")

for case in data["pairs"]:
    inp, actual = case["input"], case["actual"]
    if actual["comparable"]:
        library = concordance_index_censored(
            np.array(inp["events"]), np.array(inp["times"]), np.array(inp["scores"])
        )
        for own, expected in zip(
            [
                actual["value"],
                actual["concordant"],
                actual["discordant"],
                actual["tied"],
            ],
            library[:4],
        ):
            close(own, expected, "Concordance:library pairs")
    else:
        assert actual["value"] is None
    tick("Concordance observed/time/risk-tied fixtures")
assert data["pairs"][0]["actual"]["value"] == 0
practice["G"] = {"comparablePairs": 1, "C": 0}

for case in data["cif"]:
    inp, actual = case["input"], case["actual"]
    times, statuses = inp["times"], inp["statuses"]
    masses = redistributed_mass(times, [status > 0 for status in statuses])
    totals = [F(0), F(0)]
    for row in actual["rows"]:
        at = row["time"]
        failures = [
            status for time, status in zip(times, statuses) if time == at and status
        ]
        for cause in [1, 2]:
            if failures:
                totals[cause - 1] += masses[at] * F(
                    failures.count(cause), len(failures)
                )
        close(row["first"], totals[0], "CIF:redistributed cause mass")
        close(row["second"], totals[1], "CIF:redistributed cause mass")
        close(row["survival"], 1 - sum(totals), "CIF:remaining mass")
    if set(statuses) >= {1, 2}:
        grid, library = cumulative_incidence_competing_risks(
            np.array(statuses), np.array(times, dtype=float)
        )
        for index, at in enumerate(grid):
            row = next(row for row in actual["rows"] if row["time"] == at)
            close(row["first"], library[1, index], "CIF:library")
            close(row["second"], library[2, index], "CIF:library")
    tick("CIF changed tied/single/no-cause fixtures")
practice["I"] = {"F1": "3/4", "F2": "1/4", "survival": 0, "oneMinusKM": 1}
for key, expected in [("first", F(3, 4)), ("second", F(1, 4)), ("survival", 0)]:
    close(
        data["cif"][0]["actual"]["rows"][-1][key], expected, "Practice I:changed mass"
    )
close(data["cif"][0]["actual"]["naiveFirstRisk"], 1, "Practice I:net quantity")
for case in data["constants"]:
    inp, actual = case["input"], case["actual"]
    for cause, key in [(inp["firstRate"], "first"), (inp["secondRate"], "second")]:
        integral = quad(
            lambda t: cause * math.exp(-(inp["firstRate"] + inp["secondRate"]) * t),
            0,
            inp["horizon"],
        )[0]
        close(actual[key], integral, "CIF:continuous hazard integral")
    tick("Competing constant-hazard integrated densities")

for case in data["clocks"]:
    inp, actual = case["input"], case["actual"]
    shape = inp["shape"]
    effective_scale = inp["scale"] * (
        inp["multiplier"]
        if inp["mode"] == "time"
        else inp["multiplier"] ** (-1 / shape)
    )
    law = weibull_min(shape, scale=effective_scale)
    close(actual["median"], law.median(), "Clock:library median")
    conditional = -math.expm1(
        law.logsf(inp["age"] + inp["interval"]) - law.logsf(inp["age"])
    )
    close(actual["conditionalFailure"], conditional, "Clock:conditional survival ratio")
    for row in actual["curve"][::20]:
        close(row["survival"], law.sf(row["time"]), "Clock:distribution")
        if row["time"] == 0 and shape < 1:
            assert row["hazard"] is None
        else:
            close(
                row["hazard"],
                (
                    math.exp(law.logpdf(row["time"]) - law.logsf(row["time"]))
                    if row["time"] or shape == 1
                    else 0
                ),
                "Clock:density/survival",
            )
    tick("Weibull clock/time-hazard distribution fixtures")

for case in data["ph"]:
    inp, actual = case["input"], case["actual"]["selected"]
    time = inp["time"]
    if inp["mode"] == "switch":
        accumulated = quad(
            lambda t: 0.05 if t < 4 else 0.2, 0, time, points=[4] if time > 4 else None
        )[0]
        close(
            actual["comparisonSurvival"],
            math.exp(-accumulated),
            "PH:piecewise integrated hazard",
        )
    else:
        surviving = np.array([math.exp(-0.1 * time), math.exp(-0.4 * time)])
        posterior = surviving / surviving.sum()
        comparison_surviving = np.sqrt(surviving)
        comparison_posterior = comparison_surviving / comparison_surviving.sum()
        close(
            actual["referenceHazard"],
            posterior @ np.array([0.1, 0.4]),
            "PH:survivor composition",
        )
        close(
            actual["comparisonHazard"],
            comparison_posterior @ np.array([0.05, 0.2]),
            "PH:changed survivor composition",
        )
    tick("PH piecewise-integral/surviving-mixture fixtures")

for case in data["brier"]:
    inp, actual = case["input"], case["actual"]
    q, g = F(str(inp["prediction"])), F(str(inp["lateCensorProbability"]))
    # Exact joint probability of lifetime and observation state; no JS sums reused.
    full = F(2, 5) * q**2 + F(3, 5) * (1 - q) ** 2
    weighted = F(2, 5) * q**2 + F(3, 5) * g * (1 - q) ** 2 / g
    complete = (F(2, 5) * q**2 + F(3, 5) * g * (1 - q) ** 2) / (F(2, 5) + F(3, 5) * g)
    naive = (F(2, 5) + F(3, 5) * (1 - g)) * q**2 + F(3, 5) * g * (1 - q) ** 2
    for key, expected in [
        ("full", full),
        ("weighted", weighted),
        ("completeCases", complete),
        ("censoredAsFailure", naive),
    ]:
        close(actual[key], expected, "IPCW:exact known observation law")
    tick("Known-G exact expectation and alternative estimands")
practice["H"] = {"full": "1/4", "weighted": "1/4", "survivorWeight": 4}

for case in data["observation"]:
    inp, actual = case["input"], case["actual"]
    times, events = [2, 3, 4, 4, 6, 7, 8, 9], [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    if inp.get("toggled") is not None:
        events[inp["toggled"]] = not events[inp["toggled"]]
    expected_events = [
        event and time <= inp["cutoff"] and not inp.get("allCensored", False)
        for time, event in zip(times, events)
    ]
    assert actual["events"] == expected_events
    assert actual["times"] == [min(time, inp["cutoff"]) for time in times]
    tick("Observation cutoff/event-status fixtures")

# Changed native interval observations: integrate the density directly, then
# compare to the displayed stable log-likelihood helper without rerunning its fit.
interval_scope = definitions(
    "intervalLikelihood", ["negative_log_likelihood"], {"np": np}
)
for bounds in [[(0, 1), (1, 4), (2, math.inf)], [(1, 3), (2, 6), (7, math.inf)]]:
    interval_scope["bounds"] = bounds
    for rate in [0.07, 0.4]:
        expected = 0
        for left, right in bounds:
            probability = (
                math.exp(-rate * left)
                if math.isinf(right)
                else quad(lambda t: rate * math.exp(-rate * t), left, right)[0]
            )
            expected -= math.log(probability)
        close(
            interval_scope["negative_log_likelihood"](math.log(rate)),
            expected,
            "Native interval:integrated densities",
        )
        tick("Actual displayed interval helper changed observations")

practice["C"] = {
    "intervalProbability": -math.expm1(-0.6),
    "hourlyRate": float(F(1, 120)),
    "exposureMLE": "3/25",
    "noEvents": "boundary zero or positive-domain supremum",
}
exposure_scope = definitions(
    "exponential",
    ["negative_log_likelihood"],
    {"np": np, "failures": 3, "exposure": 25},
)
fit = minimize_scalar(
    exposure_scope["negative_log_likelihood"],
    bounds=(-5, 1),
    method="bounded",
    options={"xatol": 1e-12},
)
close(math.exp(fit.x), F(3, 25), "Practice C:actual changed exposure fit", 2e-8)
exposure_scope["failures"] = 0
assert (
    exposure_scope["negative_log_likelihood"](-12)
    < exposure_scope["negative_log_likelihood"](-8)
    < exposure_scope["negative_log_likelihood"](-4)
)
tick("Actual exponential helper changed exposure and no-event boundary")
practice["F"] = {"hazardChange": 3, "positiveSurvivalCrossing": 6}
close(0.1 * 3 + 0.3 * (6 - 3), 0.2 * 6, "Practice F:integrated equality")
practice["J"] = {"hazardRatio": 2**-1.5, "histories": [[2, 4, 0, 0], [4, 7, 1, 1]]}
assert [row for row in practice["J"]["histories"] if row[0] < 4 <= row[1]] == [
    [2, 4, 0, 0]
]
assert [row for row in practice["J"]["histories"] if row[0] < 7 <= row[1]] == [
    [4, 7, 1, 1]
]
practice["L"] = {
    "threshold": float(F(30, 200) / F(1, 2)),
    "tie": "declare an action rule",
    "causalBenefit": "stipulated, not identified by prognostic fit",
}

# K is itself a complete changed program, already actually executed by author.
# Bind its invocation, candidate protocol and expected output; do not refit it.
changed_tree = ast.parse(data["examples"]["changedReport"]["code"])
base_tree = ast.parse(data["examples"]["reliabilityReport"]["code"])
changed_def = next(
    node
    for node in changed_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "report"
)
base_def = next(
    node
    for node in base_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "report"
)
assert ast.dump(changed_def) == ast.dump(
    base_def
), "Changed report retains exact candidate/split/metric protocol."
invocation = changed_tree.body[-1].value
assert {kw.arg: ast.literal_eval(kw.value) for kw in invocation.keywords} == {
    "seed": 91,
    "shape": 2.0,
    "nonlinear": False,
}
output = data["reusedPrograms"]["outputs"]["changedReport"]
for phrase in [
    "selected: linear Cox",
    "test IBS model/KM: 0.134459 0.149032",
    "0.647823 0.659939",
    "0.000489",
]:
    assert phrase in output
practice["K"] = {
    "status": "reused exact actual author execution, not rerun",
    "record": data["reusedPrograms"]["record"],
    "protocolASTMatches": True,
    "stdout": output,
}
for letter in "BCDEFGHIJKL":
    assert letter in practice

result = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "scope": "Bounded numerical author verification; complementary to root's 23 invariants. No browser or independent final-review claim.",
    "sourceHashes": {
        data["modelSource"]: data["modelHash"],
        data["exampleSource"]: data["exampleHash"],
    },
    "scriptHashes": {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in [
            "scripts/verify-survival-models.mjs",
            "scripts/verify-survival-models.py",
        ]
    },
    "versions": {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "pandas", "lifelines", "scikit-survival"]
    },
    "counts": counts,
    "totalFiniteFixtures": sum(counts.values()),
    "invalidContracts": data["invalidCases"],
    "largestAbsoluteErrorsByCheck": largest_error,
    "changedPractice": practice,
    "programExecutionReuse": {
        key: data["reusedPrograms"][key]
        for key in [
            "record",
            "recordHash",
            "executedAt",
            "programs",
            "executedPrograms",
            "sha256",
        ]
    },
    "limits": [
        "Finite stated teaching contracts only, not a production survival API campaign.",
        "17 program stdout executions reused by exact module hash; only named native helper definitions executed on changed inputs here.",
        "Exact arithmetic and library agreement do not establish censoring assumptions in real data or causal identification.",
        "Browser evidence and final author source freeze belong to the lesson author.",
    ],
}
target = Path(sys.argv[1]).with_name("results.json")
target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(
    f"PASS: {sum(counts.values())} finite independent fixtures; {len(data['invalidCases'])} rejected input contracts; changed practice B–L; 17 exact executions reused."
)
