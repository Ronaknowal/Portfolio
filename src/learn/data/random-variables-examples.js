// Complete programs; outputs captured by the topic builder.
export const randomVariableExamples = {
  "mapping": {
    "title": "Collect the probability of every preimage",
    "question": "If the two head probabilities differ, are the three possible head counts equally likely?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom itertools import product\n\n\ndef pushforward(outcomes, value_of):\n    result = {}\n    for outcome, mass in outcomes:\n        value = value_of(outcome)\n        result[value] = result.get(value, Q()) + mass\n    return result\n\n\np, q = Q(1, 2), Q(1, 4)\noutcomes = [\n    (bits, (p if bits[0] else 1 - p) * (q if bits[1] else 1 - q))\n    for bits in product([0, 1], repeat=2)\n]\nlaw = pushforward(outcomes, sum)\nprint(\"head-count law:\", {x: str(mass) for x, mass in sorted(law.items())})\nprint(\"total:\", sum(law.values()))\nprint(\"P(X <= 1):\", sum(mass for x, mass in law.items() if x <= 1))\nprint(\"expected heads:\", sum(x * mass for x, mass in law.items()))\n",
    "expected": "head-count law: {0: '3/8', 1: '1/2', 2: '1/8'}\ntotal: 1\nP(X <= 1): 7/8\nexpected heads: 3/4",
    "interpretation": "The two outcomes with one head contribute to the same mass. The distribution is induced by a fixed function; it is not a second random experiment."
  },
  "moments": {
    "title": "Compare a prediction with the whole distribution",
    "question": "Does squaring the mean produce the expected squared value? Which constant minimizes squared loss?",
    "language": "python",
    "code": "from fractions import Fraction as Q\n\n\ndef expectation(values, masses):\n    if (\n        len(values) != len(masses)\n        or not values\n        or any(p < 0 for p in masses)\n        or sum(masses) != 1\n    ):\n        raise ValueError(\n            \"Use matching finite values and normalized nonnegative masses.\"\n        )\n    return sum((x * p for x, p in zip(values, masses)), Q())\n\n\nvalues, masses = [-2, 0, 3], [Q(1, 4), Q(1, 2), Q(1, 4)]\nmu = expectation(values, masses)\nsecond = expectation([x * x for x in values], masses)\nvariance = expectation([(x - mu) ** 2 for x in values], masses)\nprint(\"mean / second moment / squared mean:\", mu, second, mu * mu)\nprint(\"variance:\", variance)\nfor c in [Q(0), mu, Q(1)]:\n    loss = expectation([(x - c) ** 2 for x in values], masses)\n    assert loss == variance + (c - mu) ** 2\n    print(\"prediction / loss:\", c, loss)\nprint(\"after Y=1000X+7: mean / variance:\", 1000 * mu + 7, 1000**2 * variance)\n",
    "expected": "mean / second moment / squared mean: 1/4 13/4 1/16\nvariance: 51/16\nprediction / loss: 0 13/4\nprediction / loss: 1/4 51/16\nprediction / loss: 1 15/4\nafter Y=1000X+7: mean / variance: 257 3187500",
    "interpretation": "The minimum loss is the population variance, reached at the mean. A change from seconds to milliseconds multiplies variance by one million, while an offset changes only the mean."
  },
  "indicators": {
    "title": "Count successes even when trials depend on each other",
    "question": "When two of four objects are marked and two are drawn without replacement, what changes relative to replacement?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom itertools import permutations, product\n\n\ndef describe(draws):\n    pairs = [(int(a < 2), int(b < 2)) for a, b in draws]\n    weight = Q(1, len(pairs))\n    mean = sum((a + b) * weight for a, b in pairs)\n    variance = sum((a + b - mean) ** 2 * weight for a, b in pairs)\n    covariance = sum(a * b * weight for a, b in pairs) - Q(1, 4)\n    return mean, variance, covariance\n\n\nprint(\"replacement:\", describe(list(product(range(4), repeat=2))))\nprint(\"without replacement:\", describe(list(permutations(range(4), 2))))\n",
    "expected": "replacement: (Fraction(1, 1), Fraction(1, 2), Fraction(0, 1))\nwithout replacement: (Fraction(1, 1), Fraction(1, 3), Fraction(-1, 12))",
    "interpretation": "Both count means are one because each indicator still has mean one half. The negative covariance without replacement reduces the count variance; linearity alone never required independence."
  },
  "joint": {
    "title": "Keep the joint law, not just its marginals",
    "question": "Can matching, opposite and independent pairings share the same individual distributions?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom itertools import product\n\n\ndef moments(rows):\n    mx = sum(x * p for x, y, p in rows)\n    my = sum(y * p for x, y, p in rows)\n    vx = sum((x - mx) ** 2 * p for x, y, p in rows)\n    vy = sum((y - my) ** 2 * p for x, y, p in rows)\n    covariance = sum((x - mx) * (y - my) * p for x, y, p in rows)\n    return mx, my, vx, vy, covariance\n\n\nlaws = {\n    \"matching\": [(x, x, Q(1, 3)) for x in [-1, 0, 1]],\n    \"opposite\": [(x, -x, Q(1, 3)) for x in [-1, 0, 1]],\n    \"independent\": [\n        (x, y, Q(1, 9)) for x, y in product([-1, 0, 1], repeat=2)\n    ],\n    \"nonlinear\": [(x, x * x, Q(1, 3)) for x in [-1, 0, 1]],\n}\nfor name, rows in laws.items():\n    print(\n        name, \"means, variances, covariance:\", tuple(map(str, moments(rows)))\n    )\nprint(\"nonlinear P(X=0,Y=1):\", Q(0))\nprint(\"product of its marginals:\", Q(1, 3) * Q(2, 3))\n",
    "expected": "matching means, variances, covariance: ('0', '0', '2/3', '2/3', '2/3')\nopposite means, variances, covariance: ('0', '0', '2/3', '2/3', '-2/3')\nindependent means, variances, covariance: ('0', '0', '2/3', '2/3', '0')\nnonlinear means, variances, covariance: ('0', '2/3', '2/3', '2/9', '0')\nnonlinear P(X=0,Y=1): 0\nproduct of its marginals: 2/9",
    "interpretation": "The first three laws have the same means and variances but different covariance. The last has zero covariance even though Y is determined by X; a joint cell with mass zero versus marginal product 2/9 is a direct independence counterexample."
  },
  "noise": {
    "title": "Trace common error through an average and a difference",
    "question": "Which part of the noise survives averaging two readings, and which part cancels in their difference?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom itertools import product\n\n\ndef noise_law(common=Q(2), local=Q(1)):\n    return [\n        (10 + common * s + local * e, 20 + common * s + local * f)\n        for s, e, f in product([-1, 1], repeat=3)\n    ]\n\n\ndef mean(values):\n    return sum(values, Q()) / len(values)\n\n\ndef variance(values):\n    mu = mean(values)\n    return mean([(x - mu) ** 2 for x in values])\n\n\nrows = noise_law()\na, b = [row[0] for row in rows], [row[1] for row in rows]\ncovariance = mean([(x - mean(a)) * (y - mean(b)) for x, y in rows])\nprint(\"means:\", mean(a), mean(b))\nprint(\"variances / covariance:\", variance(a), variance(b), covariance)\nprint(\n    \"average mean / variance:\",\n    mean([(x + y) / 2 for x, y in rows]),\n    variance([(x + y) / 2 for x, y in rows]),\n)\nprint(\n    \"difference mean / variance:\",\n    mean([y - x for x, y in rows]),\n    variance([y - x for x, y in rows]),\n)\nfor common in [Q(0), Q(3)]:\n    changed = noise_law(common, Q(1))\n    print(\n        \"changed common / difference variance:\",\n        common,\n        variance([y - x for x, y in changed]),\n    )\n",
    "expected": "means: 10 20\nvariances / covariance: 5 5 4\naverage mean / variance: 15 9/2\ndifference mean / variance: 10 2\nchanged common / difference variance: 0 2\nchanged common / difference variance: 3 2",
    "interpretation": "These are exact synthetic populations in mV, not measured device performance. The average targets a different baseline from either original reading. Differencing eliminates the common term algebraically, but it does not eliminate independent local error."
  },
  "matrix": {
    "title": "Propagate a covariance matrix and reject an impossible one",
    "question": "What covariance does the pair (average, difference) have? Why can [[1,2],[2,1]] not be a covariance matrix?",
    "language": "python",
    "code": "from fractions import Fraction as Q\n\n\ndef transpose(matrix):\n    return list(map(list, zip(*matrix)))\n\n\ndef multiply(a, b):\n    if not a or not b or len(a[0]) != len(b):\n        raise ValueError(\"Incompatible nonempty matrices.\")\n    return [\n        [\n            sum((a[i][k] * b[k][j] for k in range(len(b))), Q())\n            for j in range(len(b[0]))\n        ]\n        for i in range(len(a))\n    ]\n\n\nsigma = [[Q(5), Q(4)], [Q(4), Q(5)]]\ntransform = [[Q(1, 2), Q(1, 2)], [Q(-1), Q(1)]]\nchanged = multiply(multiply(transform, sigma), transpose(transform))\nprint(\n    \"covariance of (average,difference):\",\n    [[str(x) for x in row] for row in changed],\n)\nbad = [[Q(1), Q(2)], [Q(2), Q(1)]]\ndirection = [[Q(1), Q(-1)]]\nprint(\n    \"impossible variance:\",\n    multiply(multiply(direction, bad), transpose(direction))[0][0],\n)\n",
    "expected": "covariance of (average,difference): [['9/2', '0'], ['0', '2']]\nimpossible variance: -2",
    "interpretation": "A covariance matrix represents variances of every linear combination. A negative value for one such variance refutes the proposed matrix. The transformed off-diagonal zero is a lack of linear covariance, not a general proof of independence."
  },
  "conditioning": {
    "title": "Separate within-group uncertainty from changing group means",
    "question": "How can both groups have covariance −1 while their pooled covariance is positive?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom itertools import product\n\n\ndef moments(rows):\n    total = sum(p for x, y, p in rows)\n    if total == 0:\n        return None\n    mx = sum(x * p for x, y, p in rows) / total\n    my = sum(y * p for x, y, p in rows) / total\n    vy = sum((y - my) ** 2 * p for x, y, p in rows) / total\n    cov = sum((x - mx) * (y - my) * p for x, y, p in rows) / total\n    return mx, my, vy, cov\n\n\ndef decomposition(positive_probability):\n    rows = [\n        (\n            g,\n            g + u,\n            g - u,\n            (positive_probability if g == 2 else 1 - positive_probability)\n            / 2,\n        )\n        for g, u in product([-2, 2], [-1, 1])\n    ]\n    overall = moments([(x, y, p) for g, x, y, p in rows])\n    groups = [\n        (\n            sum(p for h, x, y, p in rows if h == g),\n            moments([(x, y, p) for h, x, y, p in rows if h == g]),\n        )\n        for g in [-2, 2]\n    ]\n    within_v = sum(weight * m[2] for weight, m in groups if m is not None)\n    between_v = sum(\n        weight * (m[1] - overall[1]) ** 2\n        for weight, m in groups\n        if m is not None\n    )\n    within_c = sum(weight * m[3] for weight, m in groups if m is not None)\n    between_c = sum(\n        weight * (m[0] - overall[0]) * (m[1] - overall[1])\n        for weight, m in groups\n        if m is not None\n    )\n    assert overall[2] == within_v + between_v\n    assert overall[3] == within_c + between_c\n    return overall, (within_v, between_v), (within_c, between_c), groups\n\n\nfor p in [Q(1, 2), Q(1, 4), Q(0)]:\n    overall, variances, covariances, groups = decomposition(p)\n    print(\"positive group probability:\", p)\n    print(\n        \"Y variance = within + between:\",\n        overall[2],\n        tuple(map(str, variances)),\n    )\n    print(\n        \"covariance = within + between:\",\n        overall[3],\n        tuple(map(str, covariances)),\n    )\n    print(\"undefined groups:\", sum(m is None for weight, m in groups))\n",
    "expected": "positive group probability: 1/2\nY variance = within + between: 5 ('1', '4')\ncovariance = within + between: 3 ('-1', '4')\nundefined groups: 0\npositive group probability: 1/4\nY variance = within + between: 4 ('1', '3')\ncovariance = within + between: 2 ('-1', '3')\nundefined groups: 0\npositive group probability: 0\nY variance = within + between: 1 ('1', '0')\ncovariance = within + between: -1 ('-1', '0')\nundefined groups: 1",
    "interpretation": "A group with zero probability has no conditional law identified by division. The positive pooled covariance is a between-group effect in this specified population; these calculations alone do not identify a causal effect."
  },
  "prediction": {
    "title": "Distinguish the best line from the best informed prediction",
    "question": "If X is symmetric and Y=X², can a straight-line predictor use X effectively?",
    "language": "python",
    "code": "from fractions import Fraction as Q\n\nrows = [(Q(x), Q(x * x), Q(1, 3)) for x in [-1, 0, 1]]\nmx = sum(x * p for x, y, p in rows)\nmy = sum(y * p for x, y, p in rows)\nvariance_x = sum((x - mx) ** 2 * p for x, y, p in rows)\ncovariance = sum((x - mx) * (y - my) * p for x, y, p in rows)\nslope = covariance / variance_x\nintercept = my - slope * mx\nlinear_risk = sum((y - (intercept + slope * x)) ** 2 * p for x, y, p in rows)\nconditional_risk = sum((y - x * x) ** 2 * p for x, y, p in rows)\nprint(\"best affine intercept / slope:\", intercept, slope)\nprint(\"affine risk:\", linear_risk)\nprint(\"conditional-mean risk:\", conditional_risk)\n",
    "expected": "best affine intercept / slope: 2/3 0\naffine risk: 2/9\nconditional-mean risk: 0",
    "interpretation": "The best affine predictor is constant because covariance is zero. The full conditional mean X² predicts Y perfectly in this model. A restriction to straight lines is an extra modeling decision."
  },
  "continuous": {
    "title": "Fold a continuous interval without losing a branch",
    "question": "For U uniform on [−1,1], which U values make 0.25≤U²≤0.81? What is their total probability?",
    "language": "python",
    "code": "from math import sqrt\nfrom fractions import Fraction as Q\n\n\ndef squared_uniform_interval(a, b):\n    if not 0 <= a <= b <= 1:\n        raise ValueError(\"Use 0 <= a <= b <= 1.\")\n    return [(-sqrt(b), -sqrt(a)), (sqrt(a), sqrt(b))], sqrt(b) - sqrt(a)\n\n\nintervals, mass = squared_uniform_interval(0.25, 0.81)\nprint(\"two preimage intervals:\", intervals)\nprint(\"interval probability:\", round(mass, 12))\n\n\n# Direct LOTUS: E[U^(2k)] = (1/2)*integral(-1,1) u^(2k) du.\ndef moment_y(k):\n    if not isinstance(k, int) or k < 0:\n        raise ValueError(\"Use a nonnegative integer moment.\")\n    return Q(1, 2 * k + 1)\n\n\nprint(\"mean / variance:\", moment_y(1), moment_y(2) - moment_y(1) ** 2)\nprint(\"zero-width probability:\", squared_uniform_interval(0.25, 0.25)[1])\n",
    "expected": "two preimage intervals: [(-0.9, -0.5), (0.5, 0.9)]\ninterval probability: 0.4\nmean / variance: 1/3 4/45\nzero-width probability: 0.0",
    "interpretation": "The two input intervals each have length 0.4 and uniform density 1/2, giving total probability 0.4. The transformed density diverges near zero but its point mass remains zero."
  },
  "sample_mean": {
    "title": "Compare independent readings with copied readings",
    "question": "Does writing one observed bit eight times reduce uncertainty about its underlying mean?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom math import comb\n\n\ndef mean_law(n, p):\n    if not isinstance(n, int) or not 1 <= n <= 30 or not 0 <= p <= 1:\n        raise ValueError(\"Use 1 <= integer n <= 30 and a probability p.\")\n    return [\n        (Q(k, n), Q(comb(n, k)) * p**k * (1 - p) ** (n - k))\n        for k in range(n + 1)\n    ]\n\n\nfor n in [1, 4, 8]:\n    p = Q(1, 4)\n    rows = mean_law(n, p)\n    assert sum(mass for value, mass in rows) == 1\n    variance = sum((value - p) ** 2 * mass for value, mass in rows)\n    print(\n        \"n / independent variance / copied variance:\",\n        n,\n        variance,\n        p * (1 - p),\n    )\nprint(\"n=4 exact law:\", [(str(x), str(p)) for x, p in mean_law(4, Q(1, 4))])\n",
    "expected": "n / independent variance / copied variance: 1 3/16 3/16\nn / independent variance / copied variance: 4 3/64 3/16\nn / independent variance / copied variance: 8 3/128 3/16\nn=4 exact law: [('0', '81/256'), ('1/4', '27/64'), ('1/2', '27/128'), ('3/4', '3/64'), ('1', '1/256')]",
    "interpretation": "The distribution of the independent average narrows as n grows. Copies retain the original bit distribution. These are exact distributions over repeated datasets, not a claim about the appearance of one finite dataset."
  },
  "sample_covariance": {
    "title": "Check what the n−1 correction actually corrects",
    "question": "Why is the average squared distance to the sample mean too small on average under iid sampling?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom itertools import product\n\n\ndef sample_covariance(xs, ys, ddof=1):\n    n = len(xs)\n    if n != len(ys) or n <= ddof or ddof not in (0, 1):\n        raise ValueError(\"Matching paired samples need n > ddof, ddof 0 or1.\")\n    mx, my = sum(xs, Q()) / n, sum(ys, Q()) / n\n    return sum(((x - mx) * (y - my) for x, y in zip(xs, ys)), Q()) / (\n        n - ddof\n    )\n\n\nxs = list(map(Q, [1, 2, 4]))\nprint(\n    \"one dataset: divide by n / n-1:\",\n    sample_covariance(xs, xs, 0),\n    sample_covariance(xs, xs, 1),\n)\nindependent = list(product([Q(0), Q(1)], repeat=3))\nfor ddof in [0, 1]:\n    expected = sum(\n        sample_covariance(row, row, ddof) for row in independent\n    ) / len(independent)\n    print(\"iid expected sample variance, ddof\", ddof, \":\", expected)\ncopied = [(Q(0),) * 3, (Q(1),) * 3]\nprint(\n    \"copied expected corrected variance:\",\n    sum(sample_covariance(row, row) for row in copied) / 2,\n)\n",
    "expected": "one dataset: divide by n / n-1: 14/9 7/3\niid expected sample variance, ddof 0 : 1/6\niid expected sample variance, ddof 1 : 1/4\ncopied expected corrected variance: 0",
    "interpretation": "Dividing by n−1 is unbiased for the stated iid model with finite second moments. Every copied dataset has sample variance zero, even though the marginal population variance is 1/4; the denominator cannot repair dependence."
  },
  "stable": {
    "title": "Avoid subtracting two nearly equal large moments",
    "question": "Can shifting every reading by a trillion change its true variance?",
    "language": "python",
    "code": "from fractions import Fraction as Q\n\n\ndef welford(values):\n    n = 0\n    mean = 0.0\n    m2 = 0.0\n    for value in values:\n        n += 1\n        delta = value - mean\n        mean += delta / n\n        m2 += delta * (value - mean)\n    if not n:\n        raise ValueError(\"At least one value is required.\")\n    return mean, m2 / n\n\n\nvalues = [10**12 - 1, 10**12, 10**12 + 1]\nexact_mean = sum(map(Q, values)) / 3\nexact_variance = sum((Q(x) - exact_mean) ** 2 for x in values) / 3\nmean, variance = welford(values)\nprint(\"mean:\", mean)\nprint(\"exact variance:\", exact_variance)\nprint(\"Welford variance:\", round(variance, 12))\nprint(\"population normalization n:\", len(values))\n",
    "expected": "mean: 1000000000000.0\nexact variance: 2/3\nWelford variance: 0.666666666667\npopulation normalization n: 3",
    "interpretation": "Centering avoids the particular cancellation in E[X²]−E[X]². This finite calculation does not make floating-point arithmetic exact or establish safe behavior for arbitrarily large inputs. Population normalization is explicit; use n−1 only for the corresponding sample-estimator question."
  },
  "tails": {
    "title": "Check the integrals before naming a moment",
    "question": "Can a positive random variable have a finite mean but an infinite second moment?",
    "language": "python",
    "code": "from fractions import Fraction as Q\n\n\n# Pareto density alpha*x^(-alpha-1), x>=1.\n# Integral alpha*x^(k-alpha-1) converges exactly when k<alpha.\ndef pareto_moment(alpha, k):\n    if alpha <= 0 or k < 0:\n        raise ValueError(\"Use alpha>0 and k>=0.\")\n    return alpha / (alpha - k) if k < alpha else None\n\n\nfor alpha in [Q(1), Q(3, 2), Q(3)]:\n    mean, second = pareto_moment(alpha, 1), pareto_moment(alpha, 2)\n    variance = second - mean * mean if second is not None else None\n    print(\"alpha / mean / second / variance:\", alpha, mean, second, variance)\n",
    "expected": "alpha / mean / second / variance: 1 None None None\nalpha / mean / second / variance: 3/2 3 None None\nalpha / mean / second / variance: 3 3/2 3 3/4",
    "interpretation": "None marks a divergent nonnegative moment, not a missing numerical integration result. At alpha=3/2 the mean is 3 but the second moment diverges. A finite sample can still return finite numbers, which do not establish finite population moments."
  },
  "dice": {
    "title": "Solve a changed joint-variable problem",
    "question": "For two independent fair dice, let S be their sum and D their difference. Does zero covariance make S and D independent?",
    "language": "python",
    "code": "from fractions import Fraction as Q\nfrom itertools import product\n\nrows = [(a + b, a - b, Q(1, 36)) for a, b in product(range(1, 7), repeat=2)]\nms = sum(s * p for s, d, p in rows)\nmd = sum(d * p for s, d, p in rows)\nvs = sum((s - ms) ** 2 * p for s, d, p in rows)\nvd = sum((d - md) ** 2 * p for s, d, p in rows)\ncov = sum((s - ms) * (d - md) * p for s, d, p in rows)\njoint = sum(p for s, d, p in rows if s == 2 and d == 0)\nproduct_mass = sum(p for s, d, p in rows if s == 2) * sum(\n    p for s, d, p in rows if d == 0\n)\nprint(\"means:\", ms, md)\nprint(\"variances / covariance:\", vs, vd, cov)\nprint(\"P(S=2,D=0) versus product:\", joint, product_mass)\n",
    "expected": "means: 7 0\nvariances / covariance: 35/6 35/6 0\nP(S=2,D=0) versus product: 1/36 1/216",
    "interpretation": "The zero covariance follows from equal die variances and cancellation. The joint cell differs from the product of marginals, so these dice-derived variables are dependent. This is an independently solvable transfer of the pairing principle."
  }
};
