export const measureTheoryExamples = {
  "fairDie": {
    "title": "A complete finite probability calculation",
    "question": "Which part is the outcome space, which part assigns weights, and which part asks an event question?",
    "code": "outcomes = range(1, 7)\nprobability = {outcome: 1 / 6 for outcome in outcomes}\n\np_even = sum(probability[x] for x in {2, 4, 6})\np_at_most_four = sum(probability[x] for x in {1, 2, 3, 4})\nexpected_roll = sum(x * probability[x] for x in outcomes)\n\nprint(p_even)\nprint(round(p_at_most_four, 3))\nprint(expected_roll)",
    "expected": "0.5\n0.667\n3.5"
  },
  "information": {
    "title": "Generate an information algebra from its cells",
    "question": "Will the event {1,2,3} appear among unions of the three adjacent pairs?",
    "code": "from itertools import combinations\n\ncells = [{1, 2}, {3, 4}, {5, 6}]\nevents = set()\nfor count in range(len(cells) + 1):\n    for selected in combinations(cells, count):\n        events.add(frozenset().union(*selected))\n\nomega = frozenset(range(1, 7))\nassert all(omega - event in events for event in events)\nassert all(a | b in events for a in events for b in events)\nprint(\"observable events:\", len(events))\nfor event in [{1, 2}, {1, 2, 3}, {1, 2, 5, 6}]:\n    print(sorted(event), \"observable:\", frozenset(event) in events)",
    "expected": "observable events: 8\n[1, 2] observable: True\n[1, 2, 3] observable: False\n[1, 2, 5, 6] observable: True"
  },
  "pushforward": {
    "title": "Obtain the output law by adding preimage masses",
    "question": "Three die faces produce output zero. How much mass does that output receive?",
    "code": "from collections import defaultdict\nfrom fractions import Fraction\n\nweights = {face: Fraction(1, 6) for face in range(1, 7)}\nlaw = defaultdict(Fraction)\nfor face, probability in weights.items():\n    law[max(face - 3, 0)] += probability\n\nfor value, mass in sorted(law.items()):\n    print(\"Z =\", value, \"mass =\", mass)\npreimage = [face for face in weights if max(face - 3, 0) <= 1]\nprint(\"preimage:\", preimage)\nprint(\"event probability:\", sum(weights[face] for face in preimage))\nprint(\"mean from law:\", sum(value * mass for value, mass in law.items()))",
    "expected": "Z = 0 mass = 1/2\nZ = 1 mass = 1/6\nZ = 2 mass = 1/6\nZ = 3 mass = 1/6\npreimage: [1, 2, 3, 4]\nevent probability: 2/3\nmean from law: 1"
  },
  "prefixes": {
    "title": "Count finite prefixes without pretending to simulate infinity",
    "question": "Why does total probability remain one while the probability of one specified prefix shrinks?",
    "code": "from fractions import Fraction\n\nfor n in [1, 2, 4, 8]:\n    sequences = 2**n\n    prefix_mass = Fraction(1, sequences)\n    cantor_length = Fraction(2, 3) ** n\n    print(\n        n,\n        \"prefixes:\",\n        sequences,\n        \"each mass:\",\n        prefix_mass,\n        \"total mass:\",\n        sequences * prefix_mass,\n        \"Cantor-cover length:\",\n        cantor_length,\n    )",
    "expected": "1 prefixes: 2 each mass: 1/2 total mass: 1 Cantor-cover length: 2/3\n2 prefixes: 4 each mass: 1/4 total mass: 1 Cantor-cover length: 4/9\n4 prefixes: 16 each mass: 1/16 total mass: 1 Cantor-cover length: 16/81\n8 prefixes: 256 each mass: 1/256 total mass: 1 Cantor-cover length: 256/6561"
  },
  "mixture": {
    "title": "Integrate a law with an atom and a continuous part",
    "question": "Does a closed interval of zero width always have probability zero?",
    "code": "from fractions import Fraction as F\n\nw = F(1, 4)\n\n\ndef interval_probability(low, high):\n    if low > high:\n        raise ValueError(\"Endpoints are reversed.\")\n    length = max(F(0), min(F(1), high) - max(F(0), low))\n    return w * (low <= 0 <= high) + (1 - w) * length\n\n\nfor low, high in [(F(0), F(0)), (F(0), F(1, 2)), (F(1, 4), F(3, 4))]:\n    print(f\"[{low}, {high}]:\", interval_probability(low, high))\nprint(\"E[X]:\", (1 - w) / 2)\nprint(\"E[X squared]:\", (1 - w) / 3)",
    "expected": "[0, 0]: 1/4\n[0, 1/2]: 5/8\n[1/4, 3/4]: 3/8\nE[X]: 3/8\nE[X squared]: 1/4"
  },
  "simple": {
    "title": "Compute each simple-function contribution",
    "question": "Will refining the value bands move the lower integral upward or downward?",
    "code": "from math import sqrt\n\n\ndef lower_integral(level):\n    count = 2**level\n    return sum(\n        (k / count) * (sqrt((k + 1) / count) - sqrt(k / count))\n        for k in range(count)\n    )\n\n\nprevious = 0.0\nfor level in [1, 2, 3, 4, 5, 6]:\n    value = lower_integral(level)\n    bound = 1 / 2**level\n    assert previous <= value <= 1 / 3 <= value + bound\n    print(\n        level,\n        f\"lower={value:.6f}\",\n        f\"error={1 / 3 - value:.6f}\",\n        f\"bound={bound:.6f}\",\n    )\n    previous = value",
    "expected": "1 lower=0.146447 error=0.186887 bound=0.500000\n2 lower=0.231717 error=0.101616 bound=0.250000\n3 lower=0.279370 error=0.053964 bound=0.125000\n4 lower=0.305169 error=0.028165 bound=0.062500\n5 lower=0.318816 error=0.014517 bound=0.031250\n6 lower=0.325917 error=0.007417 bound=0.015625"
  },
  "limits": {
    "title": "Check exact integrals and a fixed-point observation",
    "question": "For the spike, which changes at x=1/4 when n passes 4: the value, the integral, or both?",
    "code": "from fractions import Fraction as F\n\nx = F(1, 4)\nfor n in [1, 2, 4, 8, 16]:\n    spike_value = n if 0 < x < F(1, n) else 0\n    spike_integral = F(n) * F(1, n)\n    power_integral = F(1, n + 1)\n    truncation_integral = 2 - F(1, n)\n    print(\n        n,\n        \"spike at 1/4:\",\n        spike_value,\n        \"integral:\",\n        spike_integral,\n        \"power integral:\",\n        power_integral,\n        \"truncation integral:\",\n        truncation_integral,\n    )",
    "expected": "1 spike at 1/4: 1 integral: 1 power integral: 1/2 truncation integral: 1\n2 spike at 1/4: 2 integral: 1 power integral: 1/3 truncation integral: 3/2\n4 spike at 1/4: 0 integral: 1 power integral: 1/5 truncation integral: 7/4\n8 spike at 1/4: 0 integral: 1 power integral: 1/9 truncation integral: 15/8\n16 spike at 1/4: 0 integral: 1 power integral: 1/17 truncation integral: 31/16"
  },
  "joint": {
    "title": "Preserve a probability cell through a coordinate change",
    "question": "The area grows by six. What must happen to density at the corresponding point?",
    "code": "from fractions import Fraction as F\n\nx0, x1, y0, y1 = F(1, 2), F(3, 4), F(1, 4), F(1, 2)\na, b = F(2), F(3)\nmass = (x1**2 - x0**2) * (y1**2 - y0**2)\narea = (x1 - x0) * (y1 - y0)\ndensity_at_center = 4 * ((x0 + x1) / 2) * ((y0 + y1) / 2)\nu0, u1, v0, v1 = a * x0, a * x1, b * y0, b * y1\ntransformed_mass = (u1**2 - u0**2) * (v1**2 - v0**2) / (a**2 * b**2)\nassert mass == transformed_mass\nprint(\"cell mass:\", mass)\nprint(\"area:\", area, \"->\", a * b * area)\nprint(\n    \"center density:\", density_at_center, \"->\", density_at_center / (a * b)\n)\nprint(\"transformed cell:\", (u0, u1), (v0, v1))\nprint(\"marginal x interval:\", x1**2 - x0**2)",
    "expected": "cell mass: 15/256\narea: 1/16 -> 3/8\ncenter density: 15/16 -> 5/32\ntransformed cell: (Fraction(1, 1), Fraction(3, 2)) (Fraction(3, 4), Fraction(3, 2))\nmarginal x interval: 5/16"
  },
  "signedSums": {
    "title": "Contrast finite sums with two limits of windows",
    "question": "A 4×4 square and a 4×5 rectangle have different entries. Must their totals agree?",
    "code": "def entry(i, j):\n    return int(j == i) - int(j == i + 1)\n\n\nfor n in [2, 4, 8, 16]:\n    for columns in [n, n + 1]:\n        matrix = [[entry(i, j) for j in range(columns)] for i in range(n)]\n        by_rows = sum(sum(row) for row in matrix)\n        by_columns = sum(\n            sum(matrix[i][j] for i in range(n)) for j in range(columns)\n        )\n        assert by_rows == by_columns\n        absolute_sum = sum(abs(value) for row in matrix for value in row)\n        print(\n            f\"{n}x{columns}: total={by_rows}, absolute sum={absolute_sum}\"\n        )",
    "expected": "2x2: total=1, absolute sum=3\n2x3: total=0, absolute sum=4\n4x4: total=1, absolute sum=7\n4x5: total=0, absolute sum=8\n8x8: total=1, absolute sum=15\n8x9: total=0, absolute sum=16\n16x16: total=1, absolute sum=31\n16x17: total=0, absolute sum=32"
  },
  "conditioning": {
    "title": "Compute a conditional mean and verify its defining identity",
    "question": "Should one prediction be assigned to each outcome or to each observable cell?",
    "code": "from fractions import Fraction as F\nfrom itertools import combinations\n\nloss = [F(value) for value in [0, 2, 4, 4, 8, 12]]\nweights = [F(1, 6)] * 6\ncells = [[0, 1], [2, 3], [4, 5]]\nprediction = [F(0)] * 6\nfor cell in cells:\n    mass = sum(weights[i] for i in cell)\n    mean = sum(weights[i] * loss[i] for i in cell) / mass\n    for i in cell:\n        prediction[i] = mean\n\nfor count in range(len(cells) + 1):\n    for selected in combinations(cells, count):\n        event = [i for cell in selected for i in cell]\n        assert sum(weights[i] * prediction[i] for i in event) == sum(\n            weights[i] * loss[i] for i in event\n        )\nmean = sum(p * y for p, y in zip(weights, loss))\nrisk = sum(p * (y - z) ** 2 for p, y, z in zip(weights, loss, prediction))\nvariance = sum(p * (y - mean) ** 2 for p, y in zip(weights, loss))\nexplained = sum(p * (z - mean) ** 2 for p, z in zip(weights, prediction))\nprint(\"prediction:\", [str(z) for z in prediction])\nprint(\"mean:\", mean, \"risk:\", risk)\nprint(\"variance:\", variance, \"=\", explained, \"+\", risk)",
    "expected": "prediction: ['1', '1', '4', '4', '10', '10']\nmean: 5 risk: 5/3\nvariance: 47/3 = 14 + 5/3"
  },
  "projection": {
    "title": "Verify the extra risk of a different permitted predictor",
    "question": "What extra error comes from replacing the pair means [1,4,10] with [0,5,9]?",
    "code": "from fractions import Fraction as F\n\ny = [F(value) for value in [0, 2, 4, 4, 8, 12]]\nz = [F(value) for value in [1, 1, 4, 4, 10, 10]]\ncandidate = [F(value) for value in [0, 0, 5, 5, 9, 9]]\n\n\ndef average(values):\n    return sum(values) / 6\n\n\ncross = average([(a - b) * (b - c) for a, b, c in zip(y, z, candidate)])\noptimal_risk = average([(a - b) ** 2 for a, b in zip(y, z)])\nextra_risk = average([(b - c) ** 2 for b, c in zip(z, candidate)])\ncandidate_risk = average([(a - c) ** 2 for a, c in zip(y, candidate)])\nassert candidate_risk == optimal_risk + extra_risk\nprint(\"cross term:\", cross)\nprint(\"candidate risk:\", candidate_risk)\nprint(\"optimal + extra:\", optimal_risk, \"+\", extra_risk)",
    "expected": "cross term: 0\ncandidate risk: 8/3\noptimal + extra: 5/3 + 1"
  },
  "reweighting": {
    "title": "Compute a density ratio and reject missing support",
    "question": "Why can the target mean change even though the loss values stay fixed?",
    "code": "from fractions import Fraction as F\n\nloss = [0, 2, 4, 4, 8, 12]\ntarget = [F(0), F(0), F(1, 6), F(1, 6), F(1, 3), F(1, 3)]\n\n\ndef density_ratio(source, destination):\n    if any(p == 0 and q > 0 for p, q in zip(source, destination)):\n        raise ValueError(\n            \"Target gives positive mass to a source-null event.\"\n        )\n    return [q / p if p else F(0) for p, q in zip(source, destination)]\n\n\nsource = [F(1, 6)] * 6\nratio = density_ratio(source, target)\nprint(\"weights:\", [str(w) for w in ratio])\nprint(\"source mean:\", sum(p * y for p, y in zip(source, loss)))\nprint(\"target mean:\", sum(q * y for q, y in zip(target, loss)))\nprint(\n    \"weighted mean:\", sum(p * w * y for p, w, y in zip(source, ratio, loss))\n)\ntry:\n    density_ratio([F(1, 5)] * 5 + [F(0)], target)\nexcept ValueError as error:\n    print(error)",
    "expected": "weights: ['0', '0', '1', '1', '2', '2']\nsource mean: 5\ntarget mean: 8\nweighted mean: 8\nTarget gives positive mass to a source-null event."
  }
};
