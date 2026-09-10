"""Finite independent checks: enumerate events; integrate polynomials; audit costs."""
import contextlib
import io
import itertools
import json
import math
import random
import sys
from fractions import Fraction as F

programs = json.load(open(sys.argv[1], encoding="utf-8"))
namespaces = {}
for name, example in programs.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name + ".py", "exec"), namespace)
    namespaces[name] = namespace

log_likelihood = namespaces["bernoulli"]["log_likelihood"]
event_checks = 0
for n in range(9):
    for numerator in range(11):
        p = F(numerator, 10)
        by_count = [F(0)] * (n + 1)
        for data in itertools.product((0, 1), repeat=n):
            direct = math.prod(p if value else 1 - p for value in data)
            observed = log_likelihood(data, float(p))
            assert math.isclose(math.exp(observed), float(direct), abs_tol=2e-14)
            by_count[sum(data)] += direct
            event_checks += 1
        assert sum(by_count) == 1
        if n:
            expectation = sum(F(k, n) * mass for k, mass in enumerate(by_count))
            variance = sum((F(k, n)-p)**2 * mass for k, mass in enumerate(by_count))
            assert expectation == p
            assert variance == p*(1-p)/n
for bad_data, bad_p in [([2], .5), ([1], -.1), ([1], 1.1), ([1], float('nan'))]:
    try:
        log_likelihood(bad_data, bad_p)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid data accepted")

# Integrate the expanded integer-shape beta polynomial exactly.
def beta_integral(a, b, power=0, endpoint=F(1)):
    coefficient = math.factorial(a+b-1) // (math.factorial(a-1)*math.factorial(b-1))
    return coefficient * sum((-1)**j * math.comb(b-1, j) * endpoint**(a+j+power) / (a+j+power) for j in range(b))

posterior_checks = 0
for s, f, alpha, beta in itertools.product(range(7), range(7), range(1, 6), range(1, 6)):
    a, b, mode, prediction = namespaces["beta"]["posterior"](s, f, alpha, beta)
    assert beta_integral(a, b) == 1
    assert beta_integral(a, b, power=1) == prediction
    if isinstance(mode, F):
        peak = mode**(a-1) * (1-mode)**(b-1)
        for numerator in range(21):
            x = F(numerator, 20)
            assert x**(a-1) * (1-x)**(b-1) <= peak
    else:
        assert a == b == 1
    posterior_checks += 1

cdf = namespaces["coordinate"]["cdf"]
mass = beta_integral(5, 3, endpoint=F(3, 4)) - beta_integral(5, 3, endpoint=F(1, 2))
assert mass == F(8681, 16384) == cdf(F(3, 4))-cdf(F(1, 2))
def simpson(function, low, high, panels=10000):
    step = (high-low)/panels
    return step/3*(function(low)+function(high)+sum((4 if i % 2 else 2)*function(low+i*step) for i in range(1, panels)))
def eta_density(eta):
    p = 1/(1+math.exp(-eta))
    return 105*p**5*(1-p)**3
assert math.isclose(simpson(eta_density, 0, math.log(3)), float(mass), abs_tol=1e-12)

rng = random.Random(54116)
cost_checks = 0
for _ in range(500):
    values = [F(rng.randrange(-20, 21)) for _ in range(rng.randrange(1, 13))]
    n = len(values)
    mean = sum(values)/n
    sorted_values = sorted(values)
    median = (sorted_values[(n-1)//2]+sorted_values[n//2])/2
    best_sse = sum((value-mean)**2 for value in values)
    best_absolute = sum(abs(value-median) for value in values)
    sigma2, tau2, scale = F(4), F(3), F(2)
    gaussian_mode = n*mean/(n+sigma2/tau2)
    threshold = sigma2/(n*scale)
    laplace_mode = max(mean-threshold, F(0))-max(-mean-threshold, F(0))
    def gaussian_cost(center):
        return sum((value-center)**2 for value in values)/(2*sigma2)+center**2/(2*tau2)
    def laplace_cost(center):
        return sum((value-center)**2 for value in values)/(2*sigma2)+abs(center)/scale
    for center in [F(k, 2) for k in range(-44, 45)]:
        assert sum((value-center)**2 for value in values) == best_sse+n*(center-mean)**2
        assert sum(abs(value-center) for value in values) >= best_absolute
        assert gaussian_cost(center) >= gaussian_cost(gaussian_mode)
        assert laplace_cost(center) >= laplace_cost(laplace_mode)
        cost_checks += 1

# Nonexistence/certified-root checks independently inspect objective values.
failures = namespaces["failures"]
assert failures["uniform_likelihood"](.69) == 0
assert failures["uniform_likelihood"](.7) > failures["uniform_likelihood"](.8)
root = (failures["low"]+failures["high"])/2
assert abs(2/(1+math.exp(root))-root) < 1e-14
def regularized_log(w):
    return -2*math.log1p(math.exp(-w))-w*w/2
assert all(regularized_log(root) >= regularized_log(k/100) - 1e-13 for k in range(-500, 501))
assert all((1/(1+math.exp(-w)))**2 < 1 for w in [0, 2, 10])
population = [F(-1), F(2), F(3)]
population_mean = sum(population)/3
population_variance = sum((x-population_mean)**2 for x in population)/3
variance_checks = 0
for n in range(2, 7):
    total_sse = F(0)
    for observations in itertools.product(population, repeat=n):
        mean = sum(observations)/n
        total_sse += sum((x-mean)**2 for x in observations)
        variance_checks += 1
    assert total_sse/3**n == (n-1)*population_variance
print(f"Independent checks: {event_checks} enumerated event masses; {posterior_checks} exact beta integrals/modes; {cost_checks} exact location/penalty comparisons; {variance_checks} sample-variance expectation cases; Jacobian interval mass and optimizer/support checks passed.")
