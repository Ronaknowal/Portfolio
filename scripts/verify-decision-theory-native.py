"""Independent exact finite oracles and actual displayed-program executions."""

import contextlib
import io
import json
import math
from datetime import datetime, timezone
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path

import mpmath as mp

root = Path(__file__).resolve().parents[1]
directory = root / "scratch/decision-theory-review"
fixtures = json.loads((directory / "model-fixtures.json").read_text(encoding="utf-8"))
examples = json.loads((directory / "actual-examples.json").read_text(encoding="utf-8"))
counts = {}


def fraction(value):
    return F(str(value))


def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=3e-11, abs_tol=2e-11), (
        actual,
        expected,
    )


def exact_allocation(probabilities, capacity):
    candidates = []
    for size in range(capacity + 1):
        for selected in combinations(range(len(probabilities)), size):
            loss = sum(
                F(10) if index in selected else 80 * p
                for index, p in enumerate(probabilities)
            )
            candidates.append((loss, selected))
    return min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))


for case in fixtures["binary"]:
    p = fraction(case["p"])
    exact = [
        (1 - p) * fraction(row[0]) + p * fraction(row[1]) for row in case["losses"]
    ]
    for actual, expected in zip(case["result"]["risks"], exact):
        close(actual, expected)
    close(case["result"]["minimum"], min(exact))
    assert case["result"]["actions"] == [
        index for index, value in enumerate(exact) if value == min(exact)
    ]
counts["binary_exact_risks"] = len(fixtures["binary"])

for case in fixtures["signals"]:
    prior = fraction(case["prior"])
    joints = [
        [(1 - prior) * F(4, 5), (1 - prior) * F(1, 5)],
        [prior * F(1, 5), prior * F(4, 5)],
    ]
    policies = list(product((0, 1), repeat=2))
    best = min(
        sum(
            joints[state][signal] * (policy[signal] != state)
            for state in (0, 1)
            for signal in (0, 1)
        )
        for policy in policies
    )
    close(case["result"]["minimum"], best)
    ordered_rules = [(0, 0), (1, 1), (0, 1), (1, 0)]
    exact_risks = [
        sum(
            joints[state][signal] * (policy[signal] != state)
            for state in (0, 1)
            for signal in (0, 1)
        )
        for policy in ordered_rules
    ]
    assert case["result"]["actions"] == [
        index for index, value in enumerate(exact_risks) if value == best
    ]
    for signal in (0, 1):
        mass = sum(row[signal] for row in joints)
        close(case["result"]["signals"][signal]["posterior"], joints[1][signal] / mass)
counts["signal_policy_enumerations"] = len(fixtures["signals"])

for case in fixtures["provisioning"]:
    q, under = fraction(case["quantity"]), fraction(case["under"])
    exact = sum(
        fraction(mass) * (under * max(fraction(y) - q, 0) + max(q - fraction(y), 0))
        for y, mass in zip(case["values"], case["masses"])
    )
    close(case["result"]["risk"], exact)
counts["provisioning_exact_states"] = len(fixtures["provisioning"])

for case in fixtures["allocations"]:
    exact, selected = exact_allocation(
        list(map(fraction, case["probabilities"])), case["capacity"]
    )
    close(case["result"]["risk"], exact)
    assert case["result"]["selected"] == list(selected)
counts["capacity_subset_oracles"] = len(fixtures["allocations"])

for case in fixtures["information"]:
    p, sensitivity, false_positive = map(
        fraction, (case["p"], case["sensitivity"], case["falsePositive"])
    )
    joint = [
        [(1 - p) * (1 - false_positive), p * (1 - sensitivity)],
        [(1 - p) * false_positive, p * sensitivity],
    ]
    policy_losses = []
    for policy in product((0, 1), repeat=2):
        policy_losses.append(
            sum(
                joint[signal][state] * ([0, 80][state] if policy[signal] == 0 else 10)
                for signal in (0, 1)
                for state in (0, 1)
            )
        )
    optimum = min(policy_losses)
    close(case["result"]["afterSignal"], optimum)
    for actual, masses in zip(case["result"]["branches"], joint):
        if sum(masses) == 0:
            assert actual["posterior"] is None and actual["actions"] == []
        else:
            close(actual["posterior"], masses[1] / sum(masses))
counts["information_four_policy_oracles"] = len(fixtures["information"])

for case in fixtures["tails"]:
    values, weights, alpha = (
        list(map(fraction, case["values"])),
        list(map(fraction, case["weights"])),
        fraction(case["alpha"]),
    )
    hinge = [
        threshold
        + sum(
            weight * max(value - threshold, 0) for value, weight in zip(values, weights)
        )
        / (1 - alpha)
        for threshold in values
    ]
    close(case["result"]["cvar"], min(hinge))
    close(sum(atom["usedMass"] for atom in case["result"]["tail"]), 1 - alpha)
    if alpha:
        cumulative, var = F(0), None
        for value, weight in sorted(zip(values, weights)):
            cumulative += weight
            if weight and cumulative >= alpha:
                var = value
                break
        assert case["result"]["valueAtRisk"] == float(var)
counts["tail_hinge_oracles"] = len(fixtures["tails"])

mp.mp.dps = 60
for case in fixtures["utility"]:
    low, high, p = (mp.mpf(str(case[key])) for key in ("low", "high", "p"))
    expected = ((1 - p) * mp.sqrt(low) + p * mp.sqrt(high)) ** 2
    close(case["result"]["certaintyEquivalent"], expected)
counts["high_precision_utility_states"] = len(fixtures["utility"])

probabilities = list(map(F, [".02", ".06", ".1", ".2", ".35", ".6"]))
for case in fixtures["contingent"]:
    branch_policies = []
    for state in (0, 1):
        beliefs = probabilities.copy()
        beliefs[case["index"]] = F(state)
        _, policy = exact_allocation(beliefs, case["capacity"])
        branch_policies.append(policy)
    exact = F(0)
    for world in product((0, 1), repeat=6):
        mass = math.prod(
            p if state else 1 - p for state, p in zip(world, probabilities)
        )
        policy = branch_policies[world[case["index"]]]
        loss = case["price"] + sum(
            10 if index in policy else state * 80 for index, state in enumerate(world)
        )
        exact += mass * loss
    close(case["result"]["total"], exact)
counts["contingent_64_world_oracles"] = len(fixtures["contingent"])

for case in fixtures["thresholds"]:
    predicted = [
        int(fraction(score) >= fraction(case["threshold"])) for score in case["scores"]
    ]
    assert predicted == case["result"]["predictions"]
    assert (
        sum(
            10 if prediction else 80 * label
            for prediction, label in zip(predicted, case["labels"])
        )
        == case["result"]["total"]
    )
counts["frozen_thresholds"] = len(fixtures["thresholds"])

namespaces = {}
for key, example in examples.items():
    namespace, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example["code"], key, "exec"), namespace)
    assert output.getvalue().rstrip() == example["expected"], key
    namespaces[key] = namespace
counts["actual_complete_program_stdout"] = len(examples)

# Changed calls execute the displayed helpers, not copies of their implementation.
changed = 0
for p in (F(0), F(1, 8), F(1, 5), F(1)):
    actual = namespaces["lossTable"]["action_risks"](p, [(0, 60), (12, 12)])
    assert actual == [60 * p, F(12)]
    changed += 1
for q in (F(3), F(7, 2), F(5), F(9)):
    assert (
        namespaces["summaries"]["provisioning"](
            q, [1, 3, 9], [F(1, 4), F(1, 2), F(1, 4)], 3, 1
        )
        == 5
    )
    changed += 1
for values, weights in [
    ([F(-3), F(5), F(5)], [F(1, 4), F(1, 4), F(1, 2)]),
    ([F(1), F(9)], [F(0), F(1)]),
]:
    for alpha in (F(0), F(1, 4), F(3, 4), F(99, 100)):
        actual, _ = namespaces["utilityTail"]["tail_mean"](values, weights, alpha)
        expected = min(
            t + sum(p * max(y - t, 0) for y, p in zip(values, weights)) / (1 - alpha)
            for t in values
        )
        assert actual == expected
        changed += 1
for capacity in (0, 1, 2, 3, 6):
    for index in range(6):
        actual, _ = namespaces["capstone"]["inspect"](
            index, price=F(12), capacity=capacity
        )
        case = next(
            case
            for case in fixtures["contingent"]
            if case["capacity"] == capacity
            and case["index"] == index
            and case["price"] == 12
        )
        close(case["result"]["total"], actual)
        changed += 1
counts["actual_changed_helper_calls"] = changed
result = dict(
    timestamp=datetime.now(timezone.utc).isoformat(),
    status="passed",
    counts=counts,
    note="Exact Fraction finite policies, independent hinge and world enumeration; 60-digit mpmath utility; actual displayed stdout and changed helper execution. Not independent author review or production integration.",
)
(directory / "native-results.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
