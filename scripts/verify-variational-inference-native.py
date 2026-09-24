"""Independent probability/numerical oracles for the scoped VI lesson."""
import itertools
import json
import math
import pathlib
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm

directory = pathlib.Path(sys.argv[1])
fixtures = json.loads((directory / 'model-fixtures.json').read_text())

def close(actual, expected, tolerance=1e-9):
    assert abs(actual - expected) <= tolerance, (actual, expected)

mixture_checks = []
for fixture in fixtures['mixtureFixtures']:
    m, s, d = [fixture[key] for key in ['mean', 'sd', 'separation']]
    def integrand(noise):
        value = m + s * noise
        log_p = logsumexp([norm.logpdf(value, -d), norm.logpdf(value, d)]) - math.log(2)
        return norm.pdf(noise) * (norm.logpdf(value, m, s) - log_p)
    independent, error = quad(integrand, -np.inf, np.inf, epsabs=1e-10, limit=200)
    close(fixture['kl'], independent, 2e-9)
    close(fixture['right'], norm.sf(0, loc=m, scale=s), 2e-10)
    mixture_checks.append({'mean': m, 'sd': s, 'separation': d, 'kl': independent, 'quad_error': error})

gradient_checks = []
for fixture in fixtures['gradientFixtures']:
    m, s = fixture['mean'], fixture['sd']
    def contributions(noise):
        value = m + s * noise
        slope = (1.5 - value) / 0.49
        log_ratio = norm.logpdf(value, 1.5, 0.7) - norm.logpdf(value, m, s)
        return [slope, 1 + s*noise*slope, log_ratio*noise/s, log_ratio*(noise**2 - 1)]
    integrals = [quad(lambda e: norm.pdf(e)*contributions(e)[index], -np.inf, np.inf,
                      epsabs=1e-10)[0] for index in range(4)]
    expected = [fixture['exactMean'], fixture['exactLogSd']] * 2
    for integral, reference in zip(integrals, expected):
        close(integral, reference, 1e-9)
    for row in fixture['rows']:
        reference = contributions(row['noise'])
        for name, value in zip(['pathMean', 'pathLogSd', 'scoreMean', 'scoreLogSd'], reference):
            close(row[name], value, 1e-9)
    gradient_checks.append({'mean': m, 'sd': s, 'integrated_gradients': integrals})

# Matrix operations provide a distinct Gaussian objective oracle, plus numerical coordinate optimization.
for rho in np.linspace(-0.9, 0.9, 19):
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    precision = np.linalg.inv(covariance)
    diagonal = np.diag(1 / np.diag(precision))
    gap = 0.5*(np.trace(precision @ diagonal)-2 + np.linalg.slogdet(covariance)[1]-np.linalg.slogdet(diagonal)[1])
    close(gap, -0.5*math.log(1-rho*rho))
    mean, target = np.array([-2.0, 2.0]), np.array([1.0, -1.0])
    for step in range(6):
        index, other = step % 2, 1-step % 2
        def objective(value):
            candidate = mean.copy()
            candidate[index] = value
            difference = candidate-target
            return gap + float(difference @ precision @ difference)/2
        numeric = minimize_scalar(objective, bracket=(-5, 5))
        analytic = target[index]+rho*(mean[other]-target[other])
        close(numeric.x, analytic, 3e-6)
        mean[index] = analytic

# All finite minibatches, including changed batch sizes: data contributions only are replicated.
data = np.array([1.0, 2.0, 4.0, 5.0])
for batch_size in range(1, 5):
    estimates = [4/batch_size * sum(data[list(batch)]-1)-1
                 for batch in itertools.combinations(range(4), batch_size)]
    close(np.mean(estimates), 7)

# Changed exercises are independently calculated rather than copied from solution literals.
posterior = np.array([0.2, 0.6, 0.2])
def restricted_objective(a):
    q = np.array([a, (1-a)/2, (1-a)/2])
    return float(np.sum(q*np.log(q/posterior)))
best = minimize_scalar(restricted_objective, bounds=(1e-8, 1-1e-8), method='bounded', options={'xatol': 1e-14})
close(best.x, 0.2240092377397959, 1e-7)
close(best.fun, 0.11336992436646452, 1e-10)
close(math.log(0.5)-best.fun, -0.8065171049264098, 1e-10)
close(0.5*math.log(0.5/0.4)+0.5*math.log(0.5/0.6), 0.020410997260127586)
close(1/(1+1/4), 0.8)

# At p=q with normalized target, raw score contributions vanish; chosen pathwise variances do not.
mean_path_variance = quad(lambda e: norm.pdf(e) * (-e/0.7)**2, -np.inf, np.inf)[0]
scale_path_variance = quad(lambda e: norm.pdf(e) * (1-e*e)**2, -np.inf, np.inf)[0]
close(mean_path_variance, 1/0.49)
close(scale_path_variance, 2)

evidence = {'mixture_adaptive_quadrature': mixture_checks,
            'gradient_independent_integration': gradient_checks,
            'matched_target_pathwise_variances': [mean_path_variance, scale_path_variance],
            'changed_finite_exercise': {'qA': best.x, 'KL': best.fun},
            'matrix_coordinate_configurations': 19, 'all_minibatch_sizes': [1, 2, 3, 4]}
(directory / 'independent-evidence.json').write_text(json.dumps(evidence, indent=2))
print('Independent SciPy integration, matrix/coordinate optimization, all minibatches, changed exercises and estimator-variance checks passed.')
