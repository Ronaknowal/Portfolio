"""Execute complete learner programs before writing their displayed outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
original = json.loads((ROOT / 'docs/teaching/evidence/exponential-family-original-content.json').read_text(encoding='utf-8'))
examples = {}

def add(key, title, question, code):
    examples[key] = {'title': title, 'question': question, 'code': code.strip()}

add('bernoulli', 'The original six-of-eight fit', 'Why does the fitted expected count equal six, and why does this program require both successes and failures?', original['originalPrograms'][0]['code'])

add('conditional', 'Enumerate a summary group', 'Does the conditional distribution change when the common probability changes?', r'''
from fractions import Fraction
from itertools import product

datasets = list(product((0, 1), repeat=4))
fiber = [data for data in datasets if sum(data) == 2]
for p in (Fraction(1, 5), Fraction(4, 5)):
    weights = [p ** sum(data) * (1 - p) ** (4 - sum(data)) for data in fiber]
    total = sum(weights)
    conditional = {weight / total for weight in weights}
    print('p =', p, '| group size =', len(fiber), '| conditional masses =',
          ', '.join(str(value) for value in sorted(conditional)))
print('1001 belongs:', (1, 0, 0, 1) in fiber)
''')

add('poissonRatio', 'Same information, different raw likelihood', 'Can two Poisson datasets have the same total but unequal probabilities?', r'''
import math

first, second = [0, 3, 1], [2, 1, 1]

def log_likelihood(data, rate):
    return sum(value * math.log(rate) - rate - math.lgamma(value + 1)
               for value in data)

print('n and totals:', len(first), sum(first), sum(second))
for rate in (0.5, 2.0, 7.0):
    log_ratio = log_likelihood(first, rate) - log_likelihood(second, rate)
    print(f'rate {rate:.1f}: likelihood ratio {math.exp(log_ratio):.6f}')
''')

add('normalizer', 'Compute normalization, moments and a tilt', 'Can direct finite sums reproduce the first two derivatives and the shifted log-partition identity?', r'''
import math

outcomes, base = [-1, 0, 1], [1, 2, 1]

def family(eta):
    logs = [math.log(weight) + eta * value
            for value, weight in zip(outcomes, base)]
    shift = max(logs)
    weights = [math.exp(value - shift) for value in logs]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    mean = sum(p * value for p, value in zip(probabilities, outcomes))
    variance = sum(p * (value - mean) ** 2
                   for p, value in zip(probabilities, outcomes))
    return shift + math.log(total), probabilities, mean, variance

eta, tilt = 0.7, 0.2
partition, probabilities, mean, variance = family(eta)
print('probabilities:', [round(value, 6) for value in probabilities])
print('A, mean, variance:', [round(value, 6) for value in (partition, mean, variance)])
step = 1e-4
derivative = (family(eta + step)[0] - family(eta - step)[0]) / (2 * step)
curvature = (family(eta + step)[0] - 2 * partition + family(eta - step)[0]) / step ** 2
print('finite differences:', round(derivative, 6), round(curvature, 6))
direct = math.log(sum(p * math.exp(tilt * value)
                      for p, value in zip(probabilities, outcomes)))
shifted = family(eta + tilt)[0] - partition
print('log MGF:', round(direct, 6), round(shifted, 6))
print('A at eta=1000:', round(family(1000)[0], 3))
''')

add('moments', 'Fit a three-outcome family', 'Why do positive category counts admit a finite fit while a zero count does not?', r'''
import math

def fit(counts):
    if len(counts) != 3 or any(type(value) is not int or value < 0 for value in counts):
        raise ValueError('Provide three nonnegative integer counts.')
    total = sum(counts)
    if total == 0:
        return 'no observations', None
    if min(counts) == 0:
        return 'boundary: no finite natural-parameter MLE', None
    negative, zero, positive = counts
    eta1 = 0.5 * math.log(positive / negative)
    eta2 = 0.5 * (math.log(positive) + math.log(negative) - 2 * math.log(zero))
    return 'finite', (eta1, eta2)

for counts in ([2, 5, 3], [2, 0, 3], [0, 0, 0]):
    status, parameters = fit(counts)
    print(counts, status)
    if parameters is not None:
        eta1, eta2 = parameters
        logs = [-eta1 + eta2, 0.0, eta1 + eta2]
        shift = max(logs)
        weights = [math.exp(value - shift) for value in logs]
        probabilities = [value / sum(weights) for value in weights]
        print('eta:', [round(value, 6) for value in parameters])
        print('model masses:', [round(value, 6) for value in probabilities])
''')

add('gaussianMerge', 'Merge Gaussian summaries without raw records', 'Where does the between-group spread go when two summaries are combined?', r'''
def summarize(values):
    count, mean, squared_deviations = 0, 0.0, 0.0
    for value in values:
        count += 1
        delta = value - mean
        mean += delta / count
        squared_deviations += delta * (value - mean)
    return count, mean, squared_deviations

def merge(left, right):
    n_left, mean_left, m2_left = left
    n_right, mean_right, m2_right = right
    if n_left == 0:
        return right
    if n_right == 0:
        return left
    count = n_left + n_right
    difference = mean_right - mean_left
    mean = mean_left + difference * n_right / count
    m2 = m2_left + m2_right + difference ** 2 * n_left * n_right / count
    return count, mean, m2

left, right = [1, 2, 3], [5, 6]
combined = merge(summarize(left), summarize(right))
print('left:', summarize(left), '| right:', summarize(right))
print('merged:', combined)
count, mean, m2 = combined
print('Gaussian MLE mean and variance:', round(mean, 6), round(m2 / count, 6))
print('unbiased sample variance:', round(m2 / (count - 1), 6))
print('same as one pass:', all(abs(a - b) < 1e-12
                               for a, b in zip(combined, summarize(left + right))))
''')

add('features', 'Differentiate a fixed-feature Bernoulli model', 'What information does a feature-weighted count retain that one total loses?', r'''
import math

design = [(1, -1), (1, 0), (1, 1)]
outcomes = [0, 1, 1]
coefficients = [-0.2, 0.7]

def softplus(value):
    return max(value, 0) + math.log1p(math.exp(-abs(value)))

def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)

scores = [sum(feature * coefficient for feature, coefficient in zip(row, coefficients))
          for row in design]
probabilities = [sigmoid(score) for score in scores]
loss = sum(softplus(score) - outcome * score
           for score, outcome in zip(scores, outcomes))
gradient = [sum(row[column] * (probability - outcome)
                for row, probability, outcome in zip(design, probabilities, outcomes))
            for column in range(2)]
hessian = [[sum(row[first] * row[second] * p * (1 - p)
                for row, p in zip(design, probabilities))
            for second in range(2)] for first in range(2)]
statistic = [sum(row[column] * value for row, value in zip(design, outcomes))
             for column in range(2)]
print('X-transpose y:', statistic)
print('loss:', round(loss, 6))
print('gradient:', [round(value, 6) for value in gradient])
print('Hessian:', [[round(value, 6) for value in row] for row in hessian])
''')

add('exposure', 'Update a shared rate with unequal exposures', 'Why does total time, rather than the number of rows, update the Gamma rate parameter?', r'''
counts, hours = [3, 1, 8], [1.0, 0.5, 2.0]
prior_shape, prior_rate = 2.0, 1.0

def update(shape, rate, counts, exposures):
    if len(counts) != len(exposures):
        raise ValueError('Each count needs one exposure.')
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError('Counts must be nonnegative integers.')
    if any(value <= 0 for value in exposures):
        raise ValueError('This example requires positive exposures.')
    return shape + sum(counts), rate + sum(exposures)

shape, rate = update(prior_shape, prior_rate, counts, hours)
print('events, hours:', sum(counts), sum(hours))
print('posterior shape, rate:', shape, rate)
print('mean events per hour:', round(shape / rate, 6))
first_shape, first_rate = update(prior_shape, prior_rate, counts[:2], hours[:2])
sequential = update(first_shape, first_rate, counts[2:], hours[2:])
print('sequential equals combined:', sequential == (shape, rate))
''')

add('coordinates', 'Verify the prior-density Jacobian', 'Which exponents describe a Beta prior after changing from p to log odds?', r'''
import math

alpha, beta = 2, 3
normalizer = math.factorial(alpha + beta - 1) / (
    math.factorial(alpha - 1) * math.factorial(beta - 1))

def softplus(value):
    return max(value, 0) + math.log1p(math.exp(-abs(value)))

for probability in (0.2, 0.5, 0.8):
    eta = math.log(probability / (1 - probability))
    density_p = normalizer * probability ** (alpha - 1) * (1 - probability) ** (beta - 1)
    density_eta = density_p * probability * (1 - probability)
    canonical = normalizer * math.exp(alpha * eta - (alpha + beta) * softplus(eta))
    print(f'p={probability:.1f}: density_p={density_p:.6f}, '
          f'density_eta={density_eta:.6f}, equal={abs(density_eta - canonical) < 1e-12}')
''')

destination = ROOT / 'scratch/exponential-family-examples'
destination.mkdir(parents=True, exist_ok=True)
executions = []
for key, example in examples.items():
    path = destination / f'{key}.py'
    path.write_text(example['code'] + '\n', encoding='utf-8')
    result = subprocess.run([sys.executable, '-X', 'utf8', '-I', str(path)], capture_output=True, text=True, encoding='utf-8', check=True)
    example['expected'] = result.stdout.rstrip()
    executions.append({'key': key, 'sha256': hashlib.sha256(example['code'].encode()).hexdigest(), 'stdout': example['expected']})
assert examples['bernoulli']['expected'] == original['originalPrograms'][0]['expected']
(ROOT / 'src/learn/data/exponential-family-examples.js').write_text('export const exponentialFamilyExamples = ' + json.dumps(examples, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
(destination / 'executed.json').write_text(json.dumps(executions, indent=2), encoding='utf-8')
print(json.dumps(executions, indent=2))
