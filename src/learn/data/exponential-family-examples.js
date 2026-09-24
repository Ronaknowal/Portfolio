export const exponentialFamilyExamples = {
  "bernoulli": {
    "title": "The original six-of-eight fit",
    "question": "Why does the fitted expected count equal six, and why does this program require both successes and failures?",
    "code": "import math\n\nobservations = [1, 0, 1, 1, 0, 1, 1, 1]\nsuccesses, n = sum(observations), len(observations)\np_mle = successes / n\neta = math.log(p_mle / (1 - p_mle))\nlog_partition = math.log1p(math.exp(eta))\n\nprint(successes, n, p_mle)\nprint(round(eta, 3), round(log_partition, 3))\nprint(round(n / (1 + math.exp(-eta)), 3))",
    "expected": "6 8 0.75\n1.099 1.386\n6.0"
  },
  "conditional": {
    "title": "Enumerate a summary group",
    "question": "Does the conditional distribution change when the common probability changes?",
    "code": "from fractions import Fraction\nfrom itertools import product\n\ndatasets = list(product((0, 1), repeat=4))\nfiber = [data for data in datasets if sum(data) == 2]\nfor p in (Fraction(1, 5), Fraction(4, 5)):\n    weights = [p ** sum(data) * (1 - p) ** (4 - sum(data)) for data in fiber]\n    total = sum(weights)\n    conditional = {weight / total for weight in weights}\n    print('p =', p, '| group size =', len(fiber), '| conditional masses =',\n          ', '.join(str(value) for value in sorted(conditional)))\nprint('1001 belongs:', (1, 0, 0, 1) in fiber)",
    "expected": "p = 1/5 | group size = 6 | conditional masses = 1/6\np = 4/5 | group size = 6 | conditional masses = 1/6\n1001 belongs: True"
  },
  "poissonRatio": {
    "title": "Same information, different raw likelihood",
    "question": "Can two Poisson datasets have the same total but unequal probabilities?",
    "code": "import math\n\nfirst, second = [0, 3, 1], [2, 1, 1]\n\ndef log_likelihood(data, rate):\n    return sum(value * math.log(rate) - rate - math.lgamma(value + 1)\n               for value in data)\n\nprint('n and totals:', len(first), sum(first), sum(second))\nfor rate in (0.5, 2.0, 7.0):\n    log_ratio = log_likelihood(first, rate) - log_likelihood(second, rate)\n    print(f'rate {rate:.1f}: likelihood ratio {math.exp(log_ratio):.6f}')",
    "expected": "n and totals: 3 4 4\nrate 0.5: likelihood ratio 0.333333\nrate 2.0: likelihood ratio 0.333333\nrate 7.0: likelihood ratio 0.333333"
  },
  "normalizer": {
    "title": "Compute normalization, moments and a tilt",
    "question": "Can direct finite sums reproduce the first two derivatives and the shifted log-partition identity?",
    "code": "import math\n\noutcomes, base = [-1, 0, 1], [1, 2, 1]\n\ndef family(eta):\n    logs = [math.log(weight) + eta * value\n            for value, weight in zip(outcomes, base)]\n    shift = max(logs)\n    weights = [math.exp(value - shift) for value in logs]\n    total = sum(weights)\n    probabilities = [weight / total for weight in weights]\n    mean = sum(p * value for p, value in zip(probabilities, outcomes))\n    variance = sum(p * (value - mean) ** 2\n                   for p, value in zip(probabilities, outcomes))\n    return shift + math.log(total), probabilities, mean, variance\n\neta, tilt = 0.7, 0.2\npartition, probabilities, mean, variance = family(eta)\nprint('probabilities:', [round(value, 6) for value in probabilities])\nprint('A, mean, variance:', [round(value, 6) for value in (partition, mean, variance)])\nstep = 1e-4\nderivative = (family(eta + step)[0] - family(eta - step)[0]) / (2 * step)\ncurvature = (family(eta + step)[0] - 2 * partition + family(eta - step)[0]) / step ** 2\nprint('finite differences:', round(derivative, 6), round(curvature, 6))\ndirect = math.log(sum(p * math.exp(tilt * value)\n                      for p, value in zip(probabilities, outcomes)))\nshifted = family(eta + tilt)[0] - partition\nprint('log MGF:', round(direct, 6), round(shifted, 6))\nprint('A at eta=1000:', round(family(1000)[0], 3))",
    "expected": "probabilities: [0.110099, 0.443426, 0.446475]\nA, mean, variance: [1.506372, 0.336376, 0.443426]\nfinite differences: 0.336376 0.443426\nlog MGF: 0.075936 0.075936\nA at eta=1000: 1000.0"
  },
  "moments": {
    "title": "Fit a three-outcome family",
    "question": "Why do positive category counts admit a finite fit while a zero count does not?",
    "code": "import math\n\ndef fit(counts):\n    if len(counts) != 3 or any(type(value) is not int or value < 0 for value in counts):\n        raise ValueError('Provide three nonnegative integer counts.')\n    total = sum(counts)\n    if total == 0:\n        return 'no observations', None\n    if min(counts) == 0:\n        return 'boundary: no finite natural-parameter MLE', None\n    negative, zero, positive = counts\n    eta1 = 0.5 * math.log(positive / negative)\n    eta2 = 0.5 * (math.log(positive) + math.log(negative) - 2 * math.log(zero))\n    return 'finite', (eta1, eta2)\n\nfor counts in ([2, 5, 3], [2, 0, 3], [0, 0, 0]):\n    status, parameters = fit(counts)\n    print(counts, status)\n    if parameters is not None:\n        eta1, eta2 = parameters\n        logs = [-eta1 + eta2, 0.0, eta1 + eta2]\n        shift = max(logs)\n        weights = [math.exp(value - shift) for value in logs]\n        probabilities = [value / sum(weights) for value in weights]\n        print('eta:', [round(value, 6) for value in parameters])\n        print('model masses:', [round(value, 6) for value in probabilities])",
    "expected": "[2, 5, 3] finite\neta: [0.202733, -0.713558]\nmodel masses: [0.2, 0.5, 0.3]\n[2, 0, 3] boundary: no finite natural-parameter MLE\n[0, 0, 0] no observations"
  },
  "gaussianMerge": {
    "title": "Merge Gaussian summaries without raw records",
    "question": "Where does the between-group spread go when two summaries are combined?",
    "code": "def summarize(values):\n    count, mean, squared_deviations = 0, 0.0, 0.0\n    for value in values:\n        count += 1\n        delta = value - mean\n        mean += delta / count\n        squared_deviations += delta * (value - mean)\n    return count, mean, squared_deviations\n\ndef merge(left, right):\n    n_left, mean_left, m2_left = left\n    n_right, mean_right, m2_right = right\n    if n_left == 0:\n        return right\n    if n_right == 0:\n        return left\n    count = n_left + n_right\n    difference = mean_right - mean_left\n    mean = mean_left + difference * n_right / count\n    m2 = m2_left + m2_right + difference ** 2 * n_left * n_right / count\n    return count, mean, m2\n\nleft, right = [1, 2, 3], [5, 6]\ncombined = merge(summarize(left), summarize(right))\nprint('left:', summarize(left), '| right:', summarize(right))\nprint('merged:', combined)\ncount, mean, m2 = combined\nprint('Gaussian MLE mean and variance:', round(mean, 6), round(m2 / count, 6))\nprint('unbiased sample variance:', round(m2 / (count - 1), 6))\nprint('same as one pass:', all(abs(a - b) < 1e-12\n                               for a, b in zip(combined, summarize(left + right))))",
    "expected": "left: (3, 2.0, 2.0) | right: (2, 5.5, 0.5)\nmerged: (5, 3.4, 17.2)\nGaussian MLE mean and variance: 3.4 3.44\nunbiased sample variance: 4.3\nsame as one pass: True"
  },
  "features": {
    "title": "Differentiate a fixed-feature Bernoulli model",
    "question": "What information does a feature-weighted count retain that one total loses?",
    "code": "import math\n\ndesign = [(1, -1), (1, 0), (1, 1)]\noutcomes = [0, 1, 1]\ncoefficients = [-0.2, 0.7]\n\ndef softplus(value):\n    return max(value, 0) + math.log1p(math.exp(-abs(value)))\n\ndef sigmoid(value):\n    if value >= 0:\n        return 1 / (1 + math.exp(-value))\n    exponential = math.exp(value)\n    return exponential / (1 + exponential)\n\nscores = [sum(feature * coefficient for feature, coefficient in zip(row, coefficients))\n          for row in design]\nprobabilities = [sigmoid(score) for score in scores]\nloss = sum(softplus(score) - outcome * score\n           for score, outcome in zip(scores, outcomes))\ngradient = [sum(row[column] * (probability - outcome)\n                for row, probability, outcome in zip(design, probabilities, outcomes))\n            for column in range(2)]\nhessian = [[sum(row[first] * row[second] * p * (1 - p)\n                for row, p in zip(design, probabilities))\n            for second in range(2)] for first in range(2)]\nstatistic = [sum(row[column] * value for row, value in zip(design, outcomes))\n             for column in range(2)]\nprint('X-transpose y:', statistic)\nprint('loss:', round(loss, 6))\nprint('gradient:', [round(value, 6) for value in gradient])\nprint('Hessian:', [[round(value, 6) for value in row] for row in hessian])",
    "expected": "X-transpose y: [2, 1]\nloss: 1.61337\ngradient: [-0.638324, -0.666591]\nHessian: [[0.688021, 0.029503], [0.029503, 0.440504]]"
  },
  "exposure": {
    "title": "Update a shared rate with unequal exposures",
    "question": "Why does total time, rather than the number of rows, update the Gamma rate parameter?",
    "code": "counts, hours = [3, 1, 8], [1.0, 0.5, 2.0]\nprior_shape, prior_rate = 2.0, 1.0\n\ndef update(shape, rate, counts, exposures):\n    if len(counts) != len(exposures):\n        raise ValueError('Each count needs one exposure.')\n    if any(type(value) is not int or value < 0 for value in counts):\n        raise ValueError('Counts must be nonnegative integers.')\n    if any(value <= 0 for value in exposures):\n        raise ValueError('This example requires positive exposures.')\n    return shape + sum(counts), rate + sum(exposures)\n\nshape, rate = update(prior_shape, prior_rate, counts, hours)\nprint('events, hours:', sum(counts), sum(hours))\nprint('posterior shape, rate:', shape, rate)\nprint('mean events per hour:', round(shape / rate, 6))\nfirst_shape, first_rate = update(prior_shape, prior_rate, counts[:2], hours[:2])\nsequential = update(first_shape, first_rate, counts[2:], hours[2:])\nprint('sequential equals combined:', sequential == (shape, rate))",
    "expected": "events, hours: 12 3.5\nposterior shape, rate: 14.0 4.5\nmean events per hour: 3.111111\nsequential equals combined: True"
  },
  "coordinates": {
    "title": "Verify the prior-density Jacobian",
    "question": "Which exponents describe a Beta prior after changing from p to log odds?",
    "code": "import math\n\nalpha, beta = 2, 3\nnormalizer = math.factorial(alpha + beta - 1) / (\n    math.factorial(alpha - 1) * math.factorial(beta - 1))\n\ndef softplus(value):\n    return max(value, 0) + math.log1p(math.exp(-abs(value)))\n\nfor probability in (0.2, 0.5, 0.8):\n    eta = math.log(probability / (1 - probability))\n    density_p = normalizer * probability ** (alpha - 1) * (1 - probability) ** (beta - 1)\n    density_eta = density_p * probability * (1 - probability)\n    canonical = normalizer * math.exp(alpha * eta - (alpha + beta) * softplus(eta))\n    print(f'p={probability:.1f}: density_p={density_p:.6f}, '\n          f'density_eta={density_eta:.6f}, equal={abs(density_eta - canonical) < 1e-12}')",
    "expected": "p=0.2: density_p=1.536000, density_eta=0.245760, equal=True\np=0.5: density_p=1.500000, density_eta=0.375000, equal=True\np=0.8: density_p=0.384000, density_eta=0.061440, equal=True"
  }
};
