"""Evaluate proposed teaching fixtures; not a production lesson verification."""

from fractions import Fraction as F
from itertools import product, combinations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
checks = {}


def risk(losses, probabilities):
    return sum(loss * probability for loss, probability in zip(losses, probabilities))


def information(prior, sensitivity, false_positive):
    joint = [
        [(1 - prior) * (1 - false_positive), prior * (1 - sensitivity)],
        [(1 - prior) * false_positive, prior * sensitivity],
    ]
    before = min(80 * prior, F(10))
    after = sum(min(80 * row[1], 10 * sum(row)) for row in joint)
    perfect = 10 * prior
    return {
        "joint": joint,
        "posterior": [row[1] / sum(row) if sum(row) else None for row in joint],
        "before": before,
        "after": after,
        "evsi": before - after,
        "evpi": before - perfect,
    }


anchor = information(F(2, 25), F(4, 5), F(1, 10))
assert anchor["before"] == F(32, 5)
assert anchor["after"] == F(71, 25)
assert anchor["evsi"] == F(89, 25)
assert anchor["evpi"] == F(28, 5)
assert anchor["posterior"] == [F(4, 211), F(16, 39)]
checks["inspection_anchor"] = anchor
checks["changed_prevalence"] = information(F(1, 50), F(4, 5), F(1, 10))
checks["informative_but_no_action_change"] = information(F(1, 100), F(4, 5), F(1, 10))
assert checks["informative_but_no_action_change"]["evsi"] == 0
assert checks["changed_prevalence"]["evsi"] == F(7, 50)
voi_count = 0
for prior, sensitivity, false_positive in product(
    [F(0), F(1, 100), F(1, 5), F(1, 2), F(1)],
    [F(0), F(1, 5), F(4, 5), F(1)],
    [F(0), F(1, 4), F(1)],
):
    values = information(prior, sensitivity, false_positive)
    assert 0 <= values["evsi"] <= values["evpi"]
    voi_count += 1

# A rule maps the observed signal into an action; averaging is over repeated signals.
likelihood = [[F(4, 5), F(1, 5)], [F(1, 5), F(4, 5)]]  # state rows, signal columns
prior = [F(7, 10), F(3, 10)]
policies = {}
for rule in product(range(2), repeat=2):
    by_state = [
        sum(likelihood[state][signal] * (rule[signal] != state) for signal in range(2))
        for state in range(2)
    ]
    joint_loss = sum(
        prior[state] * likelihood[state][signal] * (rule[signal] != state)
        for state, signal in product(range(2), repeat=2)
    )
    assert joint_loss == risk(by_state, prior)
    policies[str(rule)] = {"risk_by_state": by_state, "bayes_risk": joint_loss}
assert policies["(0, 1)"]["bayes_risk"] == F(1, 5)
checks["procedure_risk"] = policies

# Probability estimate vs action: all four risks use the same true binary state law.
true_p = F(1, 10)
reported_p = F(1, 5)
checks["score_vs_action"] = {
    "true_probability": true_p,
    "misreported": reported_p,
    "honest_brier": true_p * (1 - true_p),
    "distorted_brier": true_p * (1 - reported_p) ** 2 + (1 - true_p) * reported_p**2,
    "best_operational_loss": 80 * true_p,
    "distorted_operational_loss": F(10),
    "prior_shift_new_probability": F(1, 5),
}
assert checks["score_vs_action"]["distorted_brier"] == F(1, 10)

# Finite posterior: squared, absolute, and asymmetric under/over provisioning loss.
values, masses = [0, 2, 10], [F(1, 5), F(1, 2), F(3, 10)]
mean = risk(values, masses)
absolute = {str(a): risk([abs(a - y) for y in values], masses) for a in values}
asymmetric = {
    str(a): risk([4 * max(y - a, 0) + max(a - y, 0) for y in values], masses)
    for a in values
}
assert (
    mean == 4
    and min(absolute, key=absolute.get) == "2"
    and min(asymmetric, key=asymmetric.get) == "10"
)
checks["loss_selects_summary"] = {
    "mean": mean,
    "absolute_losses": absolute,
    "asymmetric_losses": asymmetric,
}

# Finite minimax lower bound: prior (.6,.4) makes A and B both risk2.4; C has2.5.
losses = [[F(0), F(6)], [F(4), F(0)], [F(5, 2), F(5, 2)]]
mix = [F(2, 5), F(3, 5), F(0)]
mixed_risks = [sum(mix[a] * losses[a][state] for a in range(3)) for state in range(2)]
assert mixed_risks == [F(12, 5), F(12, 5)]
assert min(risk(row, [F(3, 5), F(2, 5)]) for row in losses) == F(12, 5)
checks["randomized_minimax"] = {
    "losses": losses,
    "mix": mix,
    "risks": mixed_risks,
    "deterministic_worst": [max(row) for row in losses],
}
checks["regret_changes_criterion"] = {
    "losses": [[0, 10], [6, 7]],
    "worst_losses": [10, 7],
    "worst_regrets": [3, 6],
}

# Probability interval: minimum worst-case loss and worst-case regret are distinct.
interval = [F(2, 25), F(9, 50)]
checks["ambiguity_interval"] = {
    "interval": interval,
    "ship_worst": 80 * interval[1],
    "hold_worst": F(10),
    "ship_worst_regret": 80 * interval[1] - 10,
    "hold_worst_regret": 10 - 80 * interval[0],
}


# Finite CVaR: exact upper probability-mass allocation, independently via hinge breakpoints.
def finite_cvar(values, masses, alpha):
    remaining = 1 - alpha
    total = F(0)
    for loss, mass in sorted(zip(values, masses), reverse=True):
        take = min(mass, remaining)
        total += loss * take
        remaining -= take
    assert remaining == 0
    return total / (1 - alpha)


tail_values, tail_masses = [0, 10, 100], [F(4, 5), F(3, 20), F(1, 20)]
tail_count = 0
for alpha in [F(0), F(1, 2), F(4, 5), F(9, 10), F(19, 20), F(99, 100)]:
    cvar = finite_cvar(tail_values, tail_masses, alpha)
    hinge = min(
        t + risk([max(loss - t, 0) for loss in tail_values], tail_masses) / (1 - alpha)
        for t in tail_values
    )
    assert cvar == hinge
    tail_count += 1
checks["tail_atoms"] = {
    "losses": tail_values,
    "masses": tail_masses,
    "alpha": F(9, 10),
    "var": 10,
    "cvar": finite_cvar(tail_values, tail_masses, F(9, 10)),
    "conditional_at_or_above_var": F(65, 2),
    "conditional_above_var": 100,
}
assert checks["tail_atoms"]["cvar"] == 55

# A declared utility example, not personal financial guidance.
checks["utility"] = {
    "outcomes": [50, 150],
    "probabilities": [F(1, 2), F(1, 2)],
    "expected_resource": 100,
    "sqrt_expected_utility": (math.sqrt(50) + math.sqrt(150)) / 2,
    "certainty_equivalent": ((math.sqrt(50) + math.sqrt(150)) / 2) ** 2,
    "sure_alternative": 95,
}
assert checks["utility"]["certainty_equivalent"] < 95 < 100

probabilities = [F(1, 50), F(3, 50), F(1, 10), F(1, 5), F(7, 20), F(3, 5)]
subsets = [set(c) for size in range(3) for c in combinations(range(6), size)]


def cohort_risk(p, held):
    return sum(
        F(10) if index in held else 80 * probability
        for index, probability in enumerate(p)
    )


def best_policy(p):
    held = min(subsets, key=lambda held: (cohort_risk(p, held), tuple(sorted(held))))
    return held, cohort_risk(p, held)


held, baseline = best_policy(probabilities)
assert held == {4, 5} and baseline == F(252, 5)
tests = []
for selected in range(6):
    conditional = []
    policies = []
    for state in [0, 1]:
        updated = probabilities[:]
        updated[selected] = F(state)
        held, value = best_policy(updated)
        conditional.append(value)
        policies.append(held)
    expected = (1 - probabilities[selected]) * conditional[0] + probabilities[
        selected
    ] * conditional[1]
    enumerated = F(0)
    for world in product([0, 1], repeat=6):
        mass = math.prod(p if s else 1 - p for p, s in zip(probabilities, world))
        world_held = policies[world[selected]]
        enumerated += mass * sum(
            10 if index in world_held else 80 * state
            for index, state in enumerate(world)
        )
    assert enumerated == expected
    tests.append(
        {
            "test_item": selected + 1,
            "conditional_losses": conditional,
            "held_after_signal": [sorted(i + 1 for i in held) for held in policies],
            "expected_before_test_cost": expected,
            "gross_value": baseline - expected,
            "with_cost_one": expected + 1,
        }
    )
checks["capacity_capstone"] = {
    "probabilities": probabilities,
    "baseline": baseline,
    "held_without_test": [5, 6],
    "tests": tests,
    "best_test": min(tests, key=lambda t: t["with_cost_one"]),
}
checks["batch_posterior_decision"] = {
    "beta_uniform_both_fail": F(1, 3),
    "plugin_both_fail": F(1, 4),
    "integrated_launch_loss": 10,
    "plugin_launch_loss": F(15, 2),
    "mitigation_loss": 9,
}
checks["causal_benefit"] = {
    "untreated_failure": [F(4, 5), F(3, 10)],
    "treated_failure": [F(3, 4), F(1, 20)],
    "avoided_loss": [F(5, 2), F(25, 2)],
    "action_cost": 6,
    "net_benefit": [F(-7, 2), F(13, 2)],
}
checks["calibrated_coarsening"] = {
    "group_probabilities": [F(1, 10), F(3, 10)],
    "reported_constant": F(1, 5),
    "fixed_ship_loss": 16,
    "group_informed_loss": 14,
}
checks["unequal_resource_counterexample"] = {
    "resources": [3, 2, 2],
    "benefits": [5, F(7, 2), F(7, 2)],
    "capacity": 4,
    "best_items": [2, 3],
    "best_benefit": 7,
}

validation_scores = [F(1, 20), F(1, 10), F(1, 5), F(1, 5), F(2, 5), F(4, 5)]
validation_labels = [0, 1, 0, 1, 0, 1]
test_scores = [F(3, 100), F(2, 25), F(3, 25), F(3, 10), F(3, 5), F(9, 10)]
test_labels = [0, 0, 1, 0, 1, 0]


def realized_cost(scores, labels, threshold):
    return sum(
        10 if score >= threshold else 80 * label for score, label in zip(scores, labels)
    )


thresholds = sorted(set(validation_scores)) + [F(2)]
chosen = min(
    thresholds,
    key=lambda t: (realized_cost(validation_scores, validation_labels, t), -t),
)
assert chosen == F(1, 10)
assert realized_cost(test_scores, test_labels, chosen) == 40
assert realized_cost(test_scores, test_labels, F(1, 2)) == 100
checks["validation_then_test"] = {
    "validation_scores": validation_scores,
    "validation_labels": validation_labels,
    "threshold": chosen,
    "validation_total_cost": 50,
    "test_scores": test_scores,
    "test_labels": test_labels,
    "test_total_cost": 40,
    "default_half_test_total_cost": 100,
    "endpoint_sentinel": 2,
}
assert min(test["with_cost_one"] for test in tests) == 41
assert min(test["expected_before_test_cost"] + 12 for test in tests) > baseline
checks["changed_capstone_test_price_twelve"] = {
    "best_action": "do not test",
    "loss": baseline,
    "best_tested_loss": 52,
}

changed_demands, changed_masses = [1, 3, 9], [F(1, 4), F(1, 2), F(1, 4)]
flat_quantile_losses = {
    str(q): risk(
        [3 * max(y - q, 0) + max(q - y, 0) for y in changed_demands], changed_masses
    )
    for q in range(1, 11)
}
assert all(flat_quantile_losses[str(q)] == 5 for q in range(3, 10))
assert flat_quantile_losses["1"] > 5 and flat_quantile_losses["10"] > 5
checks["changed_quantile_ties"] = flat_quantile_losses
for p in [F(3, 40), F(7, 10)]:
    alternatives = [8 * p, 2 * (1 - p), F(3, 5)]
    assert alternatives.count(min(alternatives)) == 2
assert F(1, 2) * (F(8) + F(20)) == 14
assert F(1, 5) * 80 == 16
assert 50 * (F(4, 5) - F(3, 4)) - 6 == F(-7, 2)
assert 50 * (F(3, 10) - F(1, 20)) - 6 == F(13, 2)
assert F(30, 3) == 10 and F(30, 4) == F(15, 2)
checks["changed_scalar_practice"] = {
    "threshold": F(12, 60),
    "release_at_point_fifteen": F(9),
    "release_at_point_twenty_five": F(15),
    "new_fallback_boundaries": [F(1, 10), F(3, 5)],
}

result = {
    "checkedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "design arithmetic only; no production lesson/models/browser implemented",
    "status": "passed",
    "counts": {
        "voi_joint_cases": voi_count,
        "procedure_rules": len(checks["procedure_risk"]),
        "cvar_atom_levels": tail_count,
        "capstone_signal_policies": 12,
        "capstone_exact_world_checks": 6 * 64,
    },
    "checks": checks,
    "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
destination = ROOT / "docs/teaching/evidence/decision-theory-design-checks.json"
destination.write_text(
    json.dumps(
        result,
        indent=2,
        default=lambda value: str(value) if isinstance(value, F) else value,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "path": str(destination),
            "time": result["checkedAt"],
            "best_capstone_test": checks["capacity_capstone"]["best_test"],
        },
        default=str,
    )
)
