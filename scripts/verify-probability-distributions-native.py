"""Independent finite enumeration and SciPy checks for probability lesson states."""
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import scipy
from scipy.integrate import quad
from scipy.stats import binom, expon, hypergeom, norm, poisson

destination = Path(sys.argv[1])
cases = json.loads((destination / "model-cases.json").read_text(encoding="utf-8"))


def close(actual, expected, *, absolute=2e-13, relative=3e-12):
    if expected is None:
        assert actual is None, (actual, expected)
    else:
        assert math.isclose(actual, float(expected), abs_tol=absolute, rel_tol=relative), (actual, expected)


def fraction(value):
    return Fraction(str(value))


for case in cases["events"]:
    first, second = set(case["first"]), set(case["second"])
    state = case["state"]
    both = len(first & second)
    close(state["eventProbability"], Fraction(len(first), 6))
    close(state["conditionProbability"], Fraction(len(second), 6))
    close(state["intersectionProbability"], Fraction(both, 6))
    close(state["unionProbability"], Fraction(len(first | second), 6))
    close(state["conditionalProbability"], Fraction(both, len(second)) if second else None)
    assert state["independent"] == (both * 6 == len(first) * len(second))
    assert state["disjoint"] == (both == 0)

for case in cases["bayes"]:
    prior, sensitivity, false_positive = map(fraction, [case["prior"], case["sensitivity"], case["falsePositive"]])
    state = case["state"]
    cells = [prior * sensitivity, prior * (1-sensitivity),
             (1-prior) * false_positive, (1-prior) * (1-false_positive)]
    for index, mass in enumerate(cells):
        close(state["cells"][index]["mass"], mass)
        close(state["expectedCounts"][index], mass * 100000, absolute=1e-9)
    positive, negative = cells[0] + cells[2], cells[1] + cells[3]
    close(state["positivePosterior"], cells[0]/positive if positive else None)
    close(state["negativePosterior"], cells[1]/negative if negative else None)
    close(state["positiveMass"], positive)
    close(state["negativeMass"], negative)


def enumerate_pairs(p, copy):
    single = {True: p, False: 1-p}
    pairs = Counter()
    for value, mass in single.items():
        pairs[value, value] += copy * mass
    for first, second in product(single, repeat=2):
        pairs[first, second] += (1-copy) * single[first] * single[second]
    return pairs


for case in cases["pairs"]:
    prior, sensitivity, false_positive, copy = map(fraction, [case["prior"], case["sensitivity"], case["falsePositive"], case["copy"]])
    hypotheses = [enumerate_pairs(p, copy) for p in [sensitivity, false_positive]]
    fresh = [enumerate_pairs(p, Fraction(0)) for p in [sensitivity, false_positive]]
    for index, pair in enumerate([(True, True), (True, False), (False, True), (False, False)]):
        state = case["state"]["patterns"][index]
        weights = [prior * hypotheses[0][pair], (1-prior) * hypotheses[1][pair]]
        fresh_weights = [prior * fresh[0][pair], (1-prior) * fresh[1][pair]]
        close(state["givenHypothesis"], hypotheses[0][pair])
        close(state["givenAlternative"], hypotheses[1][pair])
        close(state["evidenceMass"], sum(weights))
        close(state["posterior"], weights[0]/sum(weights) if sum(weights) else None)
        close(state["independentPosterior"], fresh_weights[0]/sum(fresh_weights) if sum(fresh_weights) else None)


@lru_cache(None)
def enumerate_urn(marked, draws, replacement):
    choices = product(range(6), repeat=draws) if replacement else combinations(range(6), draws)
    counts = Counter(sum(value < marked for value in selected) for selected in choices)
    total = sum(counts.values())
    masses = [Fraction(counts[k], total) for k in range(draws+1)]
    mean = sum(k*p for k, p in enumerate(masses))
    variance = sum((k-mean)**2*p for k, p in enumerate(masses))
    return masses, mean, variance


for case in cases["urns"]:
    masses, mean, variance = enumerate_urn(case["marked"], case["draws"], case["replacement"])
    state = case["state"]
    reference = binom(case["draws"], case["marked"]/6) if case["replacement"] else hypergeom(6, case["marked"], case["draws"])
    for index, mass in enumerate(masses):
        close(state["masses"][index]["mass"], mass)
        close(state["masses"][index]["mass"], reference.pmf(index))
        close(state["masses"][index]["cumulative"], sum(masses[:index+1]))
    close(state["mean"], mean)
    close(state["variance"], variance)
    close(state["cumulativeMass"], sum(masses[:case["threshold"]+1]))
    close(state["selectedMass"], masses[case["threshold"]] if case["threshold"] < len(masses) else 0)

for case in cases["delays"]:
    width, atom, left, right = map(fraction, [case["width"], case["atom"], case["left"], case["right"]])
    scale = 1 if case["unit"] == "seconds" else 1000
    state = case["state"]
    length = width * scale
    continuous = (1-atom)*(right-left)
    included = atom if left == 0 else 0
    close(state["density"], (1-atom)/length)
    close(state["continuousMass"], continuous)
    close(state["includedAtom"], included)
    close(state["intervalMass"], continuous+included)
    close(state["cdfLeftLimit"], 0 if left == 0 else atom+(1-atom)*left)
    close(state["cdfRight"], atom+(1-atom)*right)
    # Integrate moments independently in physical display units.
    mean = quad(lambda x: x * float((1-atom)/length), 0, float(length))[0]
    second = quad(lambda x: x*x * float((1-atom)/length), 0, float(length))[0]
    close(state["mean"], mean, absolute=1e-9)
    close(state["variance"], second-mean*mean, absolute=1e-8)
    for point in state["cdfSamples"]:
        close(point["value"], float(atom) + (1-float(atom))*point["x"]/float(length))

for case in cases["arrivals"]:
    rate, window, quantile = case["rate"], case["window"], case["quantile"]
    state = case["state"]
    reference = poisson(rate*window)
    for entry in state["masses"]:
        close(entry["mass"], reference.pmf(entry["count"]), absolute=1e-300)
    close(state["tailBeyond24"], reference.sf(24), absolute=1e-300)
    close(state["noArrivals"], reference.pmf(0))
    close(state["atLeastOne"], reference.sf(0))
    close(state["quantileWait"], expon.ppf(quantile, scale=1/rate))
    close(state["meanWait"], expon.mean(scale=1/rate))
    cumulative = 0
    for event in state["arrivals"]:
        wait = expon.ppf(event["uniform"], scale=1/rate)
        cumulative += wait
        close(event["wait"], wait)
        close(event["time"], cumulative)
        assert event["inWindow"] == (cumulative <= window)
    assert state["realizationTruncated"] == (cumulative <= window)

# Changed exercises and independent continuous calculations.
dice = list(product(range(1, 7), repeat=2))
sum_seven = {pair for pair in dice if sum(pair) == 7}
even_first = {pair for pair in dice if pair[0] % 2 == 0}
any_six = {pair for pair in dice if 6 in pair}
assert Fraction(len(sum_seven & even_first), len(even_first)) == Fraction(1, 6)
assert Fraction(len(sum_seven & any_six), len(any_six)) == Fraction(2, 11)
assert Fraction(36, 36+48) == Fraction(3, 7)
assert Fraction(4, 4+912) == Fraction(1, 229)
assert Fraction(8, 8+18) == Fraction(4, 13)
assert Fraction(64, 64+36) == Fraction(16, 25)
close(expon.ppf(.9, scale=1/3)*60, 46.051701859880914)
close(norm.cdf(.1, scale=.2)-norm.cdf(-.1, scale=.2), .38292492254802624)
close(norm.sf(8), .5*math.erfc(8/math.sqrt(2)), absolute=1e-28)
for time in [.2, 1, 2, math.log(3)/2]:
    densities = [expon.pdf(time, scale=1/rate) for rate in [3, 1]]
    posterior = densities[0]/sum(densities)
    if time == math.log(3)/2:
        close(posterior, .5)
    for width in [.1, .001, .00001]:
        masses = [quad(lambda x: expon.pdf(x, scale=1/rate), time, time+width)[0] for rate in [3, 1]]
        stable_masses = [math.exp(-rate*time)*(-math.expm1(-rate*width)) for rate in [3, 1]]
        close(masses[0]/sum(masses), stable_masses[0]/sum(stable_masses))

result = {"status": "passed", "python": sys.version.split()[0], "scipy": scipy.__version__,
          "cases": {key: len(value) for key, value in cases.items()},
          "urn_enumerations": enumerate_urn.cache_info().currsize, "changed_practice_groups": 8}
(destination / "independent-oracles.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result))
