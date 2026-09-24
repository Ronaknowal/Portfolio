"""Generate and execute the decision lesson's complete standalone programs."""

import contextlib
import io
import json
import textwrap
from pathlib import Path

import black

examples = {}


def add(key, title, question, source):
    code = black.format_str(
        textwrap.dedent(source).strip() + "\n", mode=black.Mode(line_length=80)
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, key, "exec"), {})
    examples[key] = dict(
        title=title,
        question=question,
        code=code.rstrip(),
        expected=output.getvalue().rstrip(),
        language="python",
    )


add(
    "lossTable",
    "Evaluate the whole loss table",
    "At p=0.08, what is each state's contribution? Does the best action change at p=0.20?",
    """
    from fractions import Fraction as F

    def action_risks(probability, losses):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be in [0, 1].")
        if not losses or any(len(row) != 2 for row in losses):
            raise ValueError("Each action needs a sound and a faulty loss.")
        return [(1 - probability) * row[0] + probability * row[1] for row in losses]

    losses = [(F(0), F(80)), (F(10), F(10))]
    for p in (F(8, 100), F(20, 100), F(1, 8)):
        risks = action_risks(p, losses)
        best = [index for index, risk in enumerate(risks) if risk == min(risks)]
        contributions = [[(1 - p) * sound, p * faulty] for sound, faulty in losses]
        print(f"p={p}: contributions={[[str(x) for x in row] for row in contributions]}")
        print(f"  risks={list(map(str, risks))}; best action indices={best}")
    changed = action_risks(F(15, 100), [(F(0), F(60)), (F(12), F(12))])
    assert changed == [F(9), F(12)]
    print("changed costs at p=.15:", list(map(str, changed)))
""",
)

add(
    "procedureRisk",
    "Average a complete rule in two different orders",
    "A rule is chosen before the signal. With prior high=0.3, which rule minimizes its overall error?",
    """
    from fractions import Fraction as F

    signal_given_state = [[F(4, 5), F(1, 5)], [F(1, 5), F(4, 5)]]
    rules = {"always low": (0, 0), "always high": (1, 1),
             "follow signal": (0, 1), "invert signal": (1, 0)}

    def inspect(prior_high):
        prior = [1 - prior_high, prior_high]
        results = {}
        for name, actions in rules.items():
            state_risks = [sum(weight * (actions[x] != state)
                               for x, weight in enumerate(row))
                           for state, row in enumerate(signal_given_state)]
            bayes_risk = sum(prior[state] * risk for state, risk in enumerate(state_risks))
            results[name] = bayes_risk
            print(name, "state risks", list(map(str, state_risks)), "Bayes", bayes_risk)
        by_signal = F(0)
        for x in range(2):
            joint = [prior[state] * signal_given_state[state][x] for state in range(2)]
            mass = sum(joint)
            posterior = joint[1] / mass if mass else None
            # Predicting low incurs joint high mass; predicting high incurs joint low mass.
            by_signal += min(joint)
            print(f"signal {x}: mass={mass}, posterior high={posterior}")
        assert by_signal == min(results.values())
        print("best Bayes risk:", by_signal)

    inspect(F(3, 10))
    print("Changed prior:")
    inspect(F(1, 10))
""",
)

add(
    "summaries",
    "Choose a quantity for the actual shortage and surplus costs",
    "Why does a mean of 4 not solve the underage/overage decision? Can a changed distribution have an entire interval of optimal quantities?",
    """
    from fractions import Fraction as F

    def provisioning(quantity, values, masses, under, over):
        if len(values) != len(masses) or sum(masses) != 1 or any(p < 0 for p in masses):
            raise ValueError("Use aligned values and normalized nonnegative masses.")
        if under <= 0 or over <= 0:
            raise ValueError("Both marginal costs must be positive.")
        return sum(p * (under * max(y - quantity, 0) + over * max(quantity - y, 0))
                   for y, p in zip(values, masses))

    values, masses = [0, 2, 10], [F(1, 5), F(1, 2), F(3, 10)]
    print("mean:", sum(y * p for y, p in zip(values, masses)))
    for q in (0, 2, 4, 10):
        print(f"q={q}, asymmetric cost={provisioning(q, values, masses, 4, 1)}")
    changed_values, changed_masses = [1, 3, 9], [F(1, 4), F(1, 2), F(1, 4)]
    costs = [provisioning(F(q), changed_values, changed_masses, 3, 1) for q in range(1, 11)]
    print("changed integer minimizers:", [q for q, cost in zip(range(1, 11), costs) if cost == min(costs)])
    print("changed minimum:", min(costs))
    # Between adjacent demand values the risk is affine: endpoint equality proves a flat interval.
    assert provisioning(F(7, 2), changed_values, changed_masses, 3, 1) == 5
""",
)

add(
    "predictiveBatch",
    "Integrate the batch consequence before choosing",
    "A shared uncertain fault rate makes two future faults dependent after averaging. Which action changes if you plug in the mean rate?",
    """
    from fractions import Fraction as F

    # Theta is uniform on [0,1]. Conditional on theta the two faults are independent.
    # Integral_0^1 theta^k dtheta = 1/(k+1).
    def uniform_moment(power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Use a nonnegative integer power.")
        return F(1, power + 1)

    mean_rate = uniform_moment(1)
    both_integrated = uniform_moment(2)
    both_plugged_in = mean_rate ** 2
    mitigate_cost = F(9)
    print("both faulty, integrated:", both_integrated)
    print("both faulty, plug-in:", both_plugged_in)
    print("release expected loss, integrated:", 30 * both_integrated)
    print("release expected loss, plug-in:", 30 * both_plugged_in)
    print("mitigation cost:", mitigate_cost)
    assert 30 * both_plugged_in < mitigate_cost < 30 * both_integrated
""",
)

add(
    "capacity",
    "Check the capacity allocation against every feasible subset",
    "With two slots, which items save the most expected loss? Will that ordering survive unequal resource requirements?",
    """
    from fractions import Fraction as F
    from itertools import combinations

    def feasible_subsets(count, capacity):
        for size in range(capacity + 1):
            yield from combinations(range(count), size)

    def best_allocation(probabilities, capacity, handling=F(10), damage=F(80)):
        if not 0 <= capacity <= len(probabilities):
            raise ValueError("Capacity must fit the item count.")
        candidates = [(sum(handling if i in selected else damage * p
                           for i, p in enumerate(probabilities)), selected)
                      for selected in feasible_subsets(len(probabilities), capacity)]
        # Minimal loss, then fewer slots, then lexicographically earlier item indices.
        return min(candidates, key=lambda row: (row[0], len(row[1]), row[1]))

    probabilities = list(map(F, [".02", ".06", ".10", ".20", ".35", ".60"]))
    for capacity in (0, 2, 6):
        loss, selected = best_allocation(probabilities, capacity)
        print(f"capacity={capacity}: items={tuple(i + 1 for i in selected)}, loss={float(loss):.1f}")
    sizes, benefits = [3, 2, 2], [F(5), F(7, 2), F(7, 2)]
    candidates = [(sum(benefits[i] for i in chosen), chosen)
                  for chosen in feasible_subsets(3, 3) if sum(sizes[i] for i in chosen) <= 4]
    value, selected = max(candidates)
    print("unequal-resource optimum:", tuple(i + 1 for i in selected), "saving", value)
""",
)

add(
    "validation",
    "Freeze a threshold before opening the test labels",
    "Can two rules have the same test accuracy and very different realized costs? Which data select the threshold?",
    """
    from fractions import Fraction as F

    def evaluate(scores, labels, threshold):
        if len(scores) != len(labels) or any(y not in (0, 1) for y in labels):
            raise ValueError("Use aligned scores and binary labels.")
        actions = [int(score >= threshold) for score in scores]  # 1 means quarantine.
        total = sum(10 if action else 80 * y for action, y in zip(actions, labels))
        correct = sum(action == y for action, y in zip(actions, labels))
        return total, correct, actions

    validation_scores = list(map(F, [".05", ".10", ".20", ".20", ".40", ".80"]))
    validation_labels = [0, 1, 0, 1, 0, 1]
    # Zero includes all items; two excludes every score in [0,1]. Repeated scores stay together.
    thresholds = sorted(set([F(0), F(2)] + validation_scores))
    validation_results = [(evaluate(validation_scores, validation_labels, t)[0], t) for t in thresholds]
    frozen = min(validation_results, key=lambda result: (result[0], -result[1]))[1]
    print("validation (threshold, total):", [(str(t), cost) for cost, t in validation_results])
    print("frozen threshold:", frozen)
    test_scores = list(map(F, [".03", ".08", ".12", ".30", ".60", ".90"]))
    test_labels = [0, 0, 1, 0, 1, 0]
    for threshold in (frozen, F(1, 2)):
        total, correct, actions = evaluate(test_scores, test_labels, threshold)
        print(f"threshold={threshold}, test total={total}, correct={correct}/6, actions={actions}")
""",
)

add(
    "information",
    "Fold the test tree using joint masses",
    "Does this test's gross value exceed its price? What happens to a zero-probability branch?",
    """
    from fractions import Fraction as F

    def test_value(p, sensitivity=F(4, 5), false_positive=F(1, 10), price=F(2)):
        if any(not 0 <= value <= 1 for value in (p, sensitivity, false_positive)) or price < 0:
            raise ValueError("Use probabilities in [0,1] and a nonnegative price.")
        joints = [[(1 - p) * (1 - false_positive), p * (1 - sensitivity)],
                  [(1 - p) * false_positive, p * sensitivity]]
        after = F(0)
        for signal, joint in enumerate(joints):
            mass = sum(joint)
            risks = [80 * joint[1], 10 * mass]  # Joint-weighted, not conditional risks.
            after += min(risks)
            posterior = joint[1] / mass if mass else None
            actions = [i for i, risk in enumerate(risks) if risk == min(risks)] if mass else []
            print(f"  signal={signal}, mass={mass}, posterior={posterior}, actions={actions}")
        baseline, perfect = min(80 * p, F(10)), 10 * p
        value = baseline - after
        assert 0 <= value <= baseline - perfect
        return baseline, after, value, after + price

    for prior in (F(8, 100), F(2, 100), F(1, 100)):
        print("prior:", prior)
        baseline, after, value, total = test_value(prior)
        print(f"  baseline={float(baseline):.2f}, after={float(after):.2f}, value={float(value):.2f}, with price={float(total):.2f}")
    print("Impossible positive signal:")
    test_value(F(0), false_positive=F(0))
""",
)

add(
    "forecastAndCausal",
    "Separate calibrated scores, action losses and intervention benefit",
    "How can a calibrated score discard useful decision information, and why is high untreated risk insufficient to choose an intervention?",
    """
    from fractions import Fraction as F

    p, q = F(1, 10), F(1, 5)
    brier = lambda report: p * (1 - report) ** 2 + (1 - p) * report ** 2
    print("Brier, truthful / changed:", brier(p), brier(q))
    groups = [F(1, 10), F(3, 10)]
    assert sum(groups) / 2 == F(1, 5)  # A constant .2 score is calibrated in this population.
    coarse_loss = min(80 * F(1, 5), F(20))
    full_loss = sum(min(80 * risk, F(20)) for risk in groups) / 2
    print("coarse / full-information action loss:", coarse_loss, full_loss)
    untreated = [F(4, 5), F(3, 10)]
    treated = [F(3, 4), F(1, 20)]
    # These are stipulated intervention probabilities, not observed treatment groups.
    benefit = [50 * (before - after) - 6 for before, after in zip(untreated, treated)]
    print("net intervention benefits:", list(map(str, benefit)))
    likelihood_ratio = F(4)
    for prior in (F(1, 10), F(1, 50)):
        posterior = likelihood_ratio * prior / (1 - prior + likelihood_ratio * prior)
        print(f"fixed likelihood ratio, prior={prior}, posterior={posterior}")
""",
)

add(
    "criteria",
    "Compare worst loss, regret and a certified mixture",
    "Can the safest rule under worst loss differ from the smallest-regret rule? Can a randomized rule improve a deterministic minimax choice?",
    """
    from fractions import Fraction as F

    losses = [[F(0), F(10)], [F(6), F(7)]]
    state_best = [min(row[state] for row in losses) for state in range(2)]
    regrets = [[loss - state_best[state] for state, loss in enumerate(row)] for row in losses]
    print("worst losses:", [max(row) for row in losses])
    print("worst regrets:", [max(row) for row in regrets])
    rules = [[F(0), F(6)], [F(4), F(0)], [F(5, 2), F(5, 2)]]
    weight_a, prior = F(2, 5), [F(3, 5), F(2, 5)]
    mixed = [weight_a * rules[0][s] + (1 - weight_a) * rules[1][s] for s in range(2)]
    lower_bounds = [sum(p * risk for p, risk in zip(prior, row)) for row in rules]
    print("mixture state risks:", list(map(str, mixed)))
    print("state-weighted pure-rule risks:", list(map(str, lower_bounds)))
    assert max(mixed) == min(lower_bounds) == F(12, 5)
    print("certified minimax value:", F(12, 5))
""",
)

add(
    "utilityTail",
    "Split probability atoms when averaging the worst tail",
    "At alpha=.9, which half of the loss-10 atom belongs to the worst tenth? Does either naive conditional mean give the answer?",
    """
    from fractions import Fraction as F
    from math import sqrt

    def tail_mean(values, masses, alpha):
        if not 0 <= alpha < 1 or sum(masses) != 1 or any(p < 0 for p in masses):
            raise ValueError("Use normalized masses and 0 <= alpha < 1.")
        remaining, numerator, taken = 1 - alpha, F(0), []
        for loss, mass in sorted(zip(values, masses), reverse=True):
            use = min(mass, remaining)
            numerator += use * loss
            taken.append((loss, use))
            remaining -= use
        return numerator / (1 - alpha), taken

    def hinge(values, masses, alpha, threshold):
        return threshold + sum(p * max(loss - threshold, 0) for loss, p in zip(values, masses)) / (1 - alpha)

    values, masses = [F(0), F(10), F(100)], [F(4, 5), F(3, 20), F(1, 20)]
    for alpha in (F(9, 10), F(4, 5), F(19, 20), F(0)):
        cvar, taken = tail_mean(values, masses, alpha)
        assert cvar == min(hinge(values, masses, alpha, threshold) for threshold in values)
        print(f"alpha={alpha}: CVaR={cvar}, included={[(str(z), str(p)) for z, p in taken]}")
    print("conditional mean at or above 10:", F(13, 2) / F(1, 5))
    print("conditional mean above 10:", F(100))
    expected_utility = (sqrt(50) + sqrt(150)) / 2
    print(f"certainty equivalent: {expected_utility ** 2:.6f}")
    print("sure 95 preferred:", sqrt(95) > expected_utility)
""",
)

add(
    "capstone",
    "Choose which item to test before allocating two slots",
    "Will the highest-risk item be the most valuable one to inspect perfectly? Verify the contingent policy in all 64 possible worlds.",
    """
    from fractions import Fraction as F
    from itertools import combinations, product

    probabilities = list(map(F, [".02", ".06", ".10", ".20", ".35", ".60"]))

    def allocate(beliefs, capacity=2):
        candidates = []
        for size in range(capacity + 1):
            for selected in combinations(range(len(beliefs)), size):
                risk = sum(F(10) if i in selected else 80 * p for i, p in enumerate(beliefs))
                candidates.append((risk, selected))
        return min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))

    def inspect(index, price=F(1), capacity=2):
        branches = []
        for state in (0, 1):
            beliefs = probabilities.copy()
            beliefs[index] = F(state)
            risk, selected = allocate(beliefs, capacity)
            mass = probabilities[index] if state else 1 - probabilities[index]
            branches.append((mass, risk, selected))
        conditional_total = sum(mass * risk for mass, risk, _ in branches) + price
        world_total = F(0)
        for world in product((0, 1), repeat=len(probabilities)):
            mass = F(1)
            for state, p in zip(world, probabilities):
                mass *= p if state else 1 - p  # Declared independent Bernoulli faults.
            selected = branches[world[index]][2]
            loss = sum(10 if i in selected else 80 * state for i, state in enumerate(world))
            world_total += mass * (loss + price)
        assert world_total == conditional_total
        return conditional_total, branches

    baseline, baseline_items = allocate(probabilities)
    print("no test:", float(baseline), tuple(i + 1 for i in baseline_items))
    results = []
    for index in range(len(probabilities)):
        total, branches = inspect(index)
        results.append((total, index))
        print(f"test item {index + 1}: total={float(total):.2f}")
        for state, (mass, risk, selected) in enumerate(branches):
            print(f"  state={state}, mass={mass}, risk={float(risk):.1f}, quarantine={tuple(i + 1 for i in selected)}")
    best_total, best_index = min(results)
    print("co-optimal tests:", [index + 1 for total, index in results if total == best_total])
    print("selected by lower-index tie-break:", best_index + 1, "total", float(best_total))
    changed = min(inspect(i, price=F(12))[0] for i in range(len(probabilities)))
    print("price 12: best tested loss", float(changed), "; choose no test:", baseline < changed)
    for capacity in (1, 3):
        no_test = allocate(probabilities, capacity)[0]
        changed_candidates = [(inspect(i, capacity=capacity)[0], i) for i in range(len(probabilities))]
        tested, index = min(changed_candidates)
        print(f"changed capacity {capacity}, all tested totals:", [float(total) for total, _ in changed_candidates])
        print(f"changed capacity {capacity}: no-test={float(no_test):.2f}, best test={index + 1}, tested={float(tested):.2f}")
        print("  co-optimal tests:", [item + 1 for total, item in changed_candidates if total == tested])
        for state, (mass, risk, selected) in enumerate(inspect(index, capacity=capacity)[1]):
            print(f"  selected-test state={state}, mass={mass}, risk={float(risk):.1f}, quarantine={tuple(i + 1 for i in selected)}")
""",
)

root = Path(__file__).resolve().parents[1]
target = root / "src/learn/data/decision-theory-examples.js"
target.write_text(
    "// Complete programs formatted and executed by generate-decision-theory-examples.py.\nexport const decisionTheoryExamples = "
    + json.dumps(examples, ensure_ascii=False, indent=2)
    + ";\n",
    encoding="utf-8",
)
print(f"Executed and saved {len(examples)} complete programs.")
