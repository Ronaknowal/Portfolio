"""Execute complete, topic-owned programs and save their actual displayed stdout."""
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = json.loads((ROOT / 'docs/teaching/evidence/random-matrix-original-content.json').read_text(encoding='utf-8'))
EXAMPLES = {}


def add(key, title, question, code, explanation):
    EXAMPLES[key] = {'title': title, 'question': question, 'code': textwrap.dedent(code).strip(), 'explanation': explanation}


add('original', 'Calculate the original noise edges',
    'For 1,000 observations and 250 unit-variance features, what is the asymptotic reference band?',
    ARCHIVE['program'], 'These are limiting reference edges under the stated iid model, not a finite-sample acceptance interval.')

add('gram', 'Connect counted rows, singular values and centering',
    'Can one possible sample from independent sign-valued coordinates have principal scales 1.5 and 0.5?', r'''
import numpy as np

data = np.array([[1, 1]] * 3 + [[-1, -1]] * 3
                + [[1, -1], [-1, 1]], dtype=float)
n, p = data.shape
second_moment = data.T @ data / n
singular = np.linalg.svd(data, compute_uv=False)
print("column means:", data.mean(axis=0).tolist())
print("raw second moment:", second_moment.tolist())
print("eigenvalues:", np.linalg.eigvalsh(second_moment).tolist())
print("squared singular values / n:", np.round(singular**2 / n, 6).tolist())
centered = data - data.mean(axis=0)
sample_covariance = centered.T @ centered / (n - 1)
print("centered n-1 eigenvalues:",
      np.round(np.linalg.eigvalsh(sample_covariance), 6).tolist())

wide = np.random.default_rng(17).normal(size=(5, 8))
print("wide raw rank:", np.linalg.matrix_rank(wide))
print("wide centered rank:", np.linalg.matrix_rank(wide - wide.mean(axis=0)))
''', 'The first fixture happens to have zero column means; the change of denominator still multiplies its eigenvalues by 8/7. Rank is a numerical diagnostic here; the row-space argument establishes the exact upper bound.')

add('moments', 'Check the finite Gaussian second moment',
    'Why can E[S]=I coexist with a dispersed empirical spectrum when p grows with n?', r'''
import numpy as np

def exact_gaussian_second_moment(n, p):
    diagonal = p * (1 + 2 / n)
    off_diagonal = p * (p - 1) / n
    return (diagonal + off_diagonal) / p

rng = np.random.default_rng(41)
for n, p in [(40, 4), (40, 20), (80, 40)]:
    observed = []
    for _ in range(600):
        data = rng.normal(size=(n, p))
        covariance = data.T @ data / n
        observed.append(np.sum(covariance * covariance) / p)
    print(n, p, "exact expectation:",
          round(exact_gaussian_second_moment(n, p), 4),
          "Monte Carlo mean:", round(float(np.mean(observed)), 4))
''', 'The formula is exact for independent standard Gaussian entries with a known zero mean and denominator n. The seeded Monte Carlo mean is an approximation, not the derivation or a proof.')

add('mass', 'Integrate the continuous mass and keep the atom',
    'At gamma=4, where is the missing three quarters of the eigenvalue probability?', r'''
import math
from scipy.integrate import quad

def mp_moments(gamma, variance=1):
    lower = variance * (1 - math.sqrt(gamma))**2
    upper = variance * (1 + math.sqrt(gamma))**2
    def density(value):
        return math.sqrt((upper - value) * (value - lower)) / (
            2 * math.pi * gamma * variance * value)
    continuous = [quad(lambda x: x**k * density(x), lower, upper)[0]
                  for k in range(3)]
    return max(0, 1 - 1 / gamma), continuous

for gamma in [0.25, 1, 4]:
    atom, moments = mp_moments(gamma)
    print("gamma:", gamma, "zero atom:", atom,
          "continuous mass, mean, second moment:",
          [round(value, 6) for value in moments])
''', 'Quadrature integrates the improper but finite integral at gamma=1; it does not evaluate an infinite endpoint height as a probability. The zero atom contributes to total mass but not positive moments.')

add('bound', 'Calculate a finite Gaussian bound',
    'How wide is a valid 95% bound compared with the asymptotic MP support?', r'''
import math

def gaussian_covariance_bound(n, p, delta):
    if not isinstance(n, int) or not isinstance(p, int) or not 1 <= p <= n:
        raise ValueError("Use integer dimensions with 1 <= p <= n")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("Use a failure probability strictly between zero and one")
    margin = math.sqrt(2 * (math.log(2) - math.log(delta)))
    lower = max(0, math.sqrt(n) - math.sqrt(p) - margin)
    upper = math.sqrt(n) + math.sqrt(p) + margin
    return lower**2 / n, upper**2 / n

for n, p in [(1000, 250), (25, 20)]:
    result = gaussian_covariance_bound(n, p, 0.05)
    print(n, p, "finite covariance bounds:",
          [round(value, 6) for value in result])
''', 'The event is simultaneous for all singular values and therefore all covariance eigenvalues. A negative lower singular-value bound becomes zero before squaring.')

add('calibration', 'Run a fully specified Monte Carlo comparison',
    'What does a finite upper-tail rank say when the null preserves a duplicated feature?', r'''
import numpy as np

def largest_value(rng, duplicate):
    data = rng.normal(size=(40, 12))
    if duplicate:
        data[:, -1] = data[:, 0]
    return float(np.linalg.eigvalsh(data.T @ data / 40)[-1])

observed = largest_value(np.random.default_rng(811), duplicate=True)
for duplicate in [False, True]:
    null_rng = np.random.default_rng(400)
    null = [largest_value(null_rng, duplicate) for _ in range(199)]
    exceedances = sum(value >= observed for value in null)
    print("duplicated-coordinate null:", duplicate,
          "observed:", round(observed, 6),
          "at least observed:", exceedances,
          "upper-tail rank:", round((1 + exceedances) / 200, 4))
''', 'The null fixes dimensions, variance and generation rules. The add-one rank is never zero. Its test interpretation requires an exchangeable observed statistic under that fully specified null; choosing models after seeing results changes the problem.')

add('spike', 'Compare population value, sample value and direction',
    'Does an estimated eigenvalue near 2.5 mean the population eigenvalue is 2.5?', r'''
import numpy as np

def limits(gamma, population):
    if population <= 1 + np.sqrt(gamma):
        return (1 + np.sqrt(gamma))**2, 0.0
    strength = population - 1
    value = population * (1 + gamma / strength)
    overlap = (1 - gamma / strength**2) / (1 + gamma / strength)
    return value, overlap

n, p = 160, 40
noise = np.random.default_rng(121).normal(size=(n, p))
for population in [1.2, 1.5, 2.0, 4.0]:
    data = noise.copy()
    data[:, 0] *= np.sqrt(population)
    values, vectors = np.linalg.eigh(data.T @ data / n)
    theory = limits(p / n, population)
    print("population:", population,
          "limits:", [round(float(x), 4) for x in theory],
          "sample:", round(float(values[-1]), 4),
          "squared overlap:", round(float(vectors[0, -1]**2), 4))
''', 'The same Gaussian noise is held fixed across strengths. The asymptotic limits and one finite draw need not coincide, especially near the threshold; overlap ignores the arbitrary eigenvector sign.')

add('ridge', 'Follow a perturbation through inverse gains',
    'Why does a small input error grow along a nearly zero covariance direction?', r'''
import numpy as np

covariance = np.diag([0.01, 2.0])
perturbation = np.array([0.02, 0.02])
for penalty in [0, 0.1, 1.0]:
    regularized = covariance + penalty * np.eye(2)
    response = np.linalg.solve(regularized, perturbation)
    print("penalty:", penalty,
          "gains:", np.round(1 / np.diag(regularized), 6).tolist(),
          "response:", np.round(response, 6).tolist(),
          "condition:", round(float(np.linalg.cond(regularized)), 6))
''', 'This computes a linear-system response, not a measured prediction improvement. Regularization changes the target solution as it reduces amplification.')

add('wigner', 'Generate the correct symmetric Gaussian ensemble',
    'Why must the GOE diagonal have twice the off-diagonal variance in this convention?', r'''
import numpy as np

rng = np.random.default_rng(63)
for size in [12, 48]:
    independent = rng.normal(size=(size, size))
    matrix = (independent + independent.T) / np.sqrt(2 * size)
    values = np.linalg.eigvalsh(matrix)
    print("size:", size, "extremes:", np.round(values[[0, -1]], 6).tolist(),
          "trace-square / size:",
          round(float(np.sum(matrix * matrix) / size), 6),
          "exact expected second moment:", round(1 + 1 / size, 6))
    signs = rng.choice([-1.0, 1.0], size=(size, size))
    upper = np.triu(signs, 1)
    bounded = (upper + upper.T) / np.sqrt(size)
    print("zero-diagonal sign second moment:",
          round(float(np.sum(bounded * bounded) / size), 6),
          "exact:", round(1 - 1 / size, 6))
''', 'Both ensembles have off-diagonal variance 1/size and a semicircle bulk limit. Their finite diagonal conventions and finite second moments differ.')

add('gap', 'Check the exact two-level gap',
    'How does an off-diagonal coupling change an eigenvalue crossing?', r'''
import math
import numpy as np

def two_levels(offset, difference, coupling):
    radius = math.hypot(difference, coupling)
    return np.array([offset - radius, offset + radius])

for difference, coupling in [(0, 0), (0, 0.5), (1, 0.5), (-1, 0.5)]:
    matrix = np.array([[2 + difference, coupling],
                       [coupling, 2 - difference]])
    analytic = two_levels(2, difference, coupling)
    np.testing.assert_allclose(analytic, np.linalg.eigvalsh(matrix))
    print("difference, coupling:", difference, coupling,
          "levels:", np.round(analytic, 6).tolist(),
          "gap:", round(float(analytic[1] - analytic[0]), 6))
for gap in [0.1, 1, 3]:
    print("unscaled two-by-two GOE P(gap <= g):",
          gap, round(-math.expm1(-gap**2 / 8), 6))
''', 'The probability calculation uses independent N(0,1) difference and coupling coordinates of an unscaled two-by-two GOE. It is not the exact gap law of every larger matrix.')

add('linear', 'Distinguish average preservation from uniform preservation',
    'Does variance 1/d make one random linear layer an isometry?', r'''
import numpy as np

size = 48
rng = np.random.default_rng(91)
matrix = rng.normal(size=(size, size)) / np.sqrt(size)
singular = np.linalg.svd(matrix, compute_uv=False)
fixed = np.zeros(size)
fixed[0] = 1
print("fixed direction squared output:",
      round(float(np.linalg.norm(matrix @ fixed)**2), 6))
print("minimum and maximum squared stretch:",
      np.round(singular[[-1, 0]]**2, 6).tolist())
orthogonal, _ = np.linalg.qr(rng.normal(size=(size, size)))
print("orthogonal singular range:",
      np.round(np.linalg.svd(orthogonal, compute_uv=False)[[0, -1]], 6).tolist())
''', 'The fixed-input expectation equals one over new draws. The extrema choose a direction after observing the matrix. This is a linear algebra comparison, not a deep-network training benchmark.')


def main():
    out = ROOT / 'scratch/random-matrix-native/programs'
    out.mkdir(parents=True, exist_ok=True)
    for key, example in EXAMPLES.items():
        path = out / (key + '.py')
        path.write_text(example['code'] + '\n', encoding='utf-8')
        run = subprocess.run([sys.executable, '-X', 'utf8', '-I', str(path)], check=True, capture_output=True, text=True, encoding='utf-8')
        if run.stderr:
            raise RuntimeError(run.stderr)
        example['expected'] = run.stdout.rstrip()
    assert EXAMPLES['original']['code'] == ARCHIVE['program'].strip()
    assert EXAMPLES['original']['expected'] == ARCHIVE['output'].strip()
    destination = ROOT / 'src/learn/data/random-matrix-examples.js'
    destination.write_text('// Actual Python stdout is regenerated by prepare-random-matrix-examples.py.\nexport const randomMatrixExamples = ' + json.dumps(EXAMPLES, indent=2, ensure_ascii=False) + ';\n', encoding='utf-8')
    print(json.dumps({'programsExecuted': len(EXAMPLES), 'originalPreserved': True, 'keys': list(EXAMPLES)}, indent=2))


if __name__ == '__main__':
    main()
