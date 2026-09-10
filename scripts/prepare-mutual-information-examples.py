"""Build complete lesson programs and capture their actually executed outputs."""
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/mutual-information-native-verification'
DIRECTORY.mkdir(parents=True, exist_ok=True)

INFORMATION = dedent('''\
import math


def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def information(joint):
    px = [sum(row) for row in joint]
    py = [sum(row[j] for row in joint) for j in range(len(joint[0]))]
    mi = sum(
        p * (math.log2(p) - math.log2(px[i]) - math.log2(py[j]))
        for i, row in enumerate(joint)
        for j, p in enumerate(row)
        if p > 0
    )
    remaining = sum(
        px[i] * entropy([p / px[i] for p in row])
        for i, row in enumerate(joint) if px[i] > 0
    )
    assert abs(mi - (entropy(py) - remaining)) < 1e-10
    return mi, remaining


''')

examples = {}


def add(name, title, question, code):
    code = dedent(code).strip() + '\n'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, timeout=30)
    if result.stderr:
        raise RuntimeError(result.stderr)
    examples[name] = {'title': title, 'question': question, 'code': code, 'expected': result.stdout.strip()}
    (DIRECTORY / f'{name}.py').write_text(code, encoding='utf-8')
    print(name, result.stdout.strip(), sep='\n')


add('original', 'The original binary channel, with its output preserved',
    'With balanced labels and a 20% flip probability, how much uncertainty remains after observing X?',
    (DIRECTORY / 'original-example.py').read_text())

add('nonlinear', 'Zero covariance can coexist with a perfectly useful feature',
    'Let X be equally likely −1, 0, 1 and Y=X². Does zero correlation make X useless?',
    INFORMATION + dedent('''\
values = [-1, 0, 1]
joint = [[0, 1 / 3], [1 / 3, 0], [0, 1 / 3]]
mi, remaining = information(joint)
mean_x = sum(values) / 3
mean_y = sum(x * x for x in values) / 3
covariance = sum((x - mean_x) * (x * x - mean_y) for x in values) / 3
print('covariance:', f'{covariance:.6f}')
print('MI bits:', f'{mi:.6f}')
print('remaining label entropy:', f'{remaining:.6f}')
print('sign-only MI:', f'{information([[1 / 3, 1 / 3], [0, 1 / 3]])[0]:.6f}')
# This sign code is 0 for X<=0 and 1 for X>0. It confuses -1 with 0.
'''))

add('conditional', 'Reveal the information hidden in a pair',
    'A is fair, B has probability 0.25 of being 1, and Y=A xor B. Calculate individual, joint and conditional information.',
    INFORMATION + dedent('''\
rows = [(a, b, a ^ b, .5 * (.25 if b else .75)) for a in range(2) for b in range(2)]
tables = {'A': [[0., 0.] for _ in range(2)], 'B': [[0., 0.] for _ in range(2)], 'pair': [[0., 0.] for _ in range(4)]}
for a, b, y, p in rows:
    tables['A'][a][y] += p
    tables['B'][b][y] += p
    tables['pair'][2 * a + b][y] += p
for name, table in tables.items():
    print(name, 'MI:', f'{information(table)[0]:.6f}')
conditional = 0.
for b in range(2):
    table = [[0., 0.], [0., 0.]]
    for a in range(2):
        table[a][a ^ b] = .5
    conditional += (.25 if b else .75) * information(table)[0]
print('I(A;Y|B):', f'{conditional:.6f}')
print('chain rule:', f'{information(tables["B"])[0] + conditional:.6f}')
# Contrast: when X=Y=C for a fair common bit C, unconditional MI is 1,
# but within each C slice X and Y are constants, so conditional MI is 0.
common_joint = [[0.5, 0.0], [0.0, 0.5]]
constant_slices = [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]
common_conditional = sum(0.5 * information(table)[0] for table in constant_slices)
print('common-bit unconditional / conditional:',
      f'{information(common_joint)[0]:.6f} {common_conditional:.6f}')
'''))

add('processing', 'Calculate the loss in a noisy processing chain',
    'Y flips X with probability 0.1 and Z independently flips X with probability 0.2. How much of the original label information is lost?',
    INFORMATION + dedent('''\
rows = [(x, y, z, .5 * (.9 if y == x else .1) * (.8 if z == x else .2))
        for x in range(2) for y in range(2) for z in range(2)]


def pair_table(first, second):
    table = [[0., 0.], [0., 0.]]
    for row in rows:
        table[row[first]][row[second]] += row[3]
    return table


ixy = information(pair_table(0, 1))[0]
izy = information(pair_table(2, 1))[0]
conditional = 0.
for z in range(2):
    mass = sum(p for x, y, zz, p in rows if zz == z)
    table = [[0., 0.], [0., 0.]]
    for x, y, zz, p in rows:
        if zz == z:
            table[x][y] += p / mass
    conditional += mass * information(table)[0]
print('I(X;Y):', f'{ixy:.6f}')
print('I(Z;Y):', f'{izy:.6f}')
print('I(X;Y|Z):', f'{conditional:.6f}')
print('DPI gap identity:', abs(ixy - izy - conditional) < 1e-12)
'''))

ENCODER = INFORMATION + dedent('''\
PX = [.25, .25, .25, .25]
LABEL = [[.9, .1], [.9, .1], [.1, .9], [.1, .9]]


def state(encoder, beta):
    pxz = [[PX[x] * value for value in row] for x, row in enumerate(encoder)]
    pzy = [[sum(PX[x] * encoder[x][z] * LABEL[x][y] for x in range(4))
            for y in range(2)] for z in range(len(encoder[0]))]
    pz = [sum(row) for row in pzy]
    decoder = [[p / pz[z] for p in row] for z, row in enumerate(pzy)]
    rate, relevance = information(pxz)[0], information(pzy)[0]
    return rate, relevance, rate - beta * relevance, pz, decoder, pzy


''')

add('representations', 'Keep signal, nuisance, both, or neither',
    'X contains a fair signal S and an independent fair nuisance N. The label flips S with probability 0.1. Which representation wins at beta = 3?',
    ENCODER + dedent('''\
encoders = {
    'constant': [[1.] for _ in range(4)],
    'signal': [[1., 0.], [1., 0.], [0., 1.], [0., 1.]],
    'nuisance': [[1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    'both': [[float(x == z) for z in range(4)] for x in range(4)],
    'noisy signal': [[.8, .2], [.8, .2], [.2, .8], [.2, .8]],
}
for name, encoder in encoders.items():
    rate, relevance, objective, *_ = state(encoder, 3)
    print(name + ':', ' '.join(f'{value:.6f}' for value in [rate, relevance, objective]))
# Columns are input information, label information, J=rate-3*relevance.
# This finite list is a comparison of choices, not all possible encoders.
'''))

add('iteration', 'Run a complete finite information bottleneck',
    'At beta = 3, compare an informative initialization with an exactly symmetric one. Does a stationary objective certify the best solution?',
    ENCODER + dedent('''\
def update(encoder, beta):
    _, _, _, pz, decoder, _ = state(encoder, beta)
    result = []
    for label in LABEL:
        distances = [sum(p * (math.log(p) - math.log(q)) for p, q in zip(label, representative))
                     for representative in decoder]
        log_weights = [math.log(mass) - beta * distance for mass, distance in zip(pz, distances)]
        maximum = max(log_weights)
        weights = [math.exp(value - maximum) for value in log_weights]
        result.append([value / sum(weights) for value in weights])
    return result


for name, encoder in [
    ('informative', [[.8, .2], [.6, .4], [.3, .7], [.2, .8]]),
    ('symmetric', [[.5, .5] for _ in range(4)]),
]:
    previous = state(encoder, 3)[2]
    for step in range(1, 41):
        encoder = update(encoder, 3)
        current = state(encoder, 3)
        assert current[2] <= previous + 1e-12
        previous = current[2]
        if step in [1, 4, 40]:
            print(name, step, ' '.join(f'{value:.6f}' for value in current[:3]))
    print('P(Z=1|X):', [round(row[1], 6) for row in encoder])
# Natural-log KL in the exponential; reported information is in bits.
# We checked finite descent, not the global optimum of every IB problem.
'''))

add('bounds', 'Measure how loose the variational quantities are',
    'Fix the noisy-signal encoder. Change only the reference marginal and predictive decoder. Should actual mutual information change?',
    ENCODER + dedent('''\
encoder = [[.8, .2], [.8, .2], [.2, .8], [.2, .8]]
rate, relevance, objective, pz, truth, pzy = state(encoder, 3)


def kl(p, q):
    return sum(a * (math.log2(a) - math.log2(b)) for a, b in zip(p, q) if a > 0)


for name, reference, decoder in [
    ('matched', [.5, .5], [[.74, .26], [.26, .74]]),
    ('mismatched', [.8, .2], [[.6, .4], [.4, .6]]),
]:
    upper = sum(PX[x] * kl(row, reference) for x, row in enumerate(encoder))
    ce = -sum(pzy[z][y] * math.log2(decoder[z][y]) for z in range(2) for y in range(2))
    lower = 1 - ce
    rate_gap = kl(pz, reference)
    prediction_gap = sum(pz[z] * kl(truth[z], decoder[z]) for z in range(2))
    assert abs(upper - rate - rate_gap) < 1e-12
    assert abs(relevance - lower - prediction_gap) < 1e-12
    upper_j = upper + 3 * ce - 3
    assert abs(upper_j - objective - rate_gap - 3 * prediction_gap) < 1e-12
    print(name, 'rate upper / relevance lower:', f'{upper:.6f}', f'{lower:.6f}')
    print('rate / predictive gaps:', f'{rate_gap:.6f}', f'{prediction_gap:.6f}')
    print('true J / upper J:', f'{objective:.6f}', f'{upper_j:.6f}')
'''))

add('continuous', 'Keep coordinate units separate from information units',
    'Let X be standard Normal and Z=X+sigma E with independent standard-Normal E. What happens as measurement noise shrinks?',
    dedent('''\
import math

for sigma in [.25, .5, 1., 2.]:
    variance_z = 1 + sigma * sigma
    covariance = 1.
    determinant = variance_z - covariance * covariance
    nats = .5 * (math.log(variance_z) - math.log(determinant))
    bits = nats / math.log(2)
    # Z'=10Z scales its variance by100 and its covariance with X by10.
    scaled_determinant = 100 * variance_z - (10 * covariance) ** 2
    scaled = .5 * (math.log(100 * variance_z) - math.log(scaled_determinant))
    assert abs(scaled - nats) < 1e-12
    print(sigma, 'MI bits:', f'{bits:.6f}', 'rescaling unchanged:', True)
print('sigma=0:', 'singular deterministic joint; MI is infinite')
# The infinite case is a measure-theoretic conclusion, not a finite estimate.
'''))

add('estimation', 'A finite count table can report dependence under independence',
    'Generate independent categorical pairs. Compare the empirical MI with its known population value 0, then inspect unique identifiers.',
    INFORMATION + dedent('''\
import random

for categories, count in [(2, 40), (8, 40), (8, 4000)]:
    randomizer = random.Random(204)
    counts = [[0 for _ in range(categories)] for _ in range(categories)]
    for _ in range(count):
        counts[randomizer.randrange(categories)][randomizer.randrange(categories)] += 1
    joint = [[value / count for value in row] for row in counts]
    print(categories, count, 'plug-in MI:', f'{information(joint)[0]:.6f}', 'true MI:', 0)

# Each observed ID is unique; empirical labels are then deterministic given ID.
labels = [0, 1, 1, 0, 1, 0, 0, 1]
id_joint = [[1 / len(labels) if y == label else 0 for y in range(2)] for label in labels]
print('empirical ID-label MI:', f'{information(id_joint)[0]:.6f}')
print('empirical label entropy:', f'{entropy([.5, .5]):.6f}')
# This describes the empirical table, not predictive information for a new ID.
# These seeded runs illustrate sampling effects; they are not a confidence bound.
'''))

output = 'export const mutualInformationExamples = ' + json.dumps(examples, ensure_ascii=False, indent=2) + ';\n'
(ROOT / 'src/learn/data/mutual-information-examples.js').write_text(output, encoding='utf-8')
(DIRECTORY / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {len(examples)} executed programs.')
