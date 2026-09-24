export const maximumLikelihoodExamples = {
  bernoulli: {
    title: 'Score a fixed sequence without multiplying tiny factors',
    code: `from math import comb, exp, log

def log_likelihood(data, p):
    if not 0 <= p <= 1 or any(x not in (0, 1) for x in data):
        raise ValueError("binary data and probability required")
    successes = sum(data)
    failures = len(data) - successes
    def term(count, probability):
        if count == 0:
            return 0.0
        return float("-inf") if probability == 0 else count * log(probability)
    return term(successes, p) + term(failures, 1 - p)

data = [1, 1, 1, 0]
for p in [0.5, 0.75, 1.0]:
    score = log_likelihood(data, p)
    print(f"p={p:.2f}: sequence={exp(score):.6f}, log={score:.6f}")
print("count probability:", f"{comb(4, 3) * exp(log_likelihood(data, .75)):.6f}")
print("all successes at p=1:", log_likelihood([1, 1], 1))
print("no data:", log_likelihood([], .2))
print("direct long product:", .5 ** 2000)
print("long log score:", f"{log_likelihood([1] * 1000 + [0] * 1000, .5):.3f}")`,
    expected: `p=0.50: sequence=0.062500, log=-2.772589
p=0.75: sequence=0.105469, log=-2.249341
p=1.00: sequence=0.000000, log=-inf
count probability: 0.421875
all successes at p=1: 0.0
no data: 0.0
direct long product: 0.0
long log score: -1386.294`,
  },
  location: {
    title: 'Fit a center, then change one observation',
    code: `from statistics import mean, median

def describe(values):
    center = mean(values)
    sse = sum((x - center) ** 2 for x in values)
    print(f"mean={center:.3f}, median={median(values):.3f}, SSE={sse:.3f}")
    print(f"variance MLE={sse / len(values):.3f}")
    if len(values) > 1:
        print(f"unbiased variance={sse / (len(values) - 1):.3f}")

data = [2.0, 3.0, 4.0, 7.0]
describe(data)
for center in [3, 3.5, 4]:
    print(f"center {center}: absolute cost {sum(abs(x-center) for x in data):.1f}")
describe(data + [27.0])`,
    expected: `mean=4.000, median=3.500, SSE=14.000
variance MLE=3.500
unbiased variance=4.667
center 3: absolute cost 6.0
center 3.5: absolute cost 6.0
center 4: absolute cost 6.0
mean=8.600, median=4.000, SSE=437.200
variance MLE=87.440
unbiased variance=109.300`,
  },
  beta: {
    title: 'Keep posterior mode, mean and prediction separate',
    code: `from fractions import Fraction

def posterior(successes, failures, alpha, beta):
    if min(successes, failures) < 0 or min(alpha, beta) < 1:
        raise ValueError("nonnegative counts and shapes at least one required")
    a, b = alpha + successes, beta + failures
    if a == b == 1:
        mode = "every p in [0,1]"
    elif a == 1:
        mode = Fraction(0)
    elif b == 1:
        mode = Fraction(1)
    else:
        mode = Fraction(a - 1, a + b - 2)
    return a, b, mode, Fraction(a, a + b)

for inputs in [(3, 1, 2, 2), (4, 0, 1, 1), (0, 0, 1, 1)]:
    a, b, mode, prediction = posterior(*inputs)
    print(f"Beta({a},{b}): mode={mode}; mean/predictive={prediction}")`,
    expected: `Beta(5,3): mode=2/3; mean/predictive=5/8
Beta(5,1): mode=1; mean/predictive=5/6
Beta(1,1): mode=every p in [0,1]; mean/predictive=1/2`,
  },
  shrinkage: {
    title: 'Combine measurement precision with prior precision',
    code: `from fractions import Fraction as F

data = [2, 3, 4, 7]
sigma2, prior_mean, tau2 = F(4), F(0), F(1)
n = len(data)
xbar = F(sum(data), n)
precision = F(n) / sigma2 + 1 / tau2
posterior_mean = (n * xbar / sigma2 + prior_mean / tau2) / precision
print("posterior mean/MAP:", posterior_mean)
print("posterior variance:", 1 / precision)
print("data weight:", (F(n) / sigma2) / precision)

# Prior mean zero, but a different prior variance for regularization.
tau2 = F(4)
l2_sum_coefficient = sigma2 / tau2
l2_mean_coefficient = sigma2 / (n * tau2)
gaussian_map = n * xbar / (n + l2_sum_coefficient)
laplace_scale = F(1)
threshold = sigma2 / (n * laplace_scale)
laplace_map = max(xbar - threshold, F(0)) - max(-xbar - threshold, F(0))
print("SSE L2 coefficient:", l2_sum_coefficient)
print("MSE L2 coefficient:", l2_mean_coefficient)
print("Gaussian-prior MAP:", gaussian_map)
print("Laplace-prior MAP:", laplace_map)`,
    expected: `posterior mean/MAP: 2
posterior variance: 1/2
data weight: 1/2
SSE L2 coefficient: 1
MSE L2 coefficient: 1/4
Gaussian-prior MAP: 16/5
Laplace-prior MAP: 3`,
  },
  sampling: {
    title: 'Enumerate possible estimates before a new study',
    code: `from fractions import Fraction as F
from math import comb

n, p = 4, F(1, 2)
rows = [(F(k, n), comb(n, k) * p**k * (1-p)**(n-k)) for k in range(n+1)]
for estimate, mass in rows:
    print(f"estimate={estimate}: probability={mass}")
expectation = sum(estimate * mass for estimate, mass in rows)
variance = sum((estimate - expectation)**2 * mass for estimate, mass in rows)
print("total:", sum(mass for _, mass in rows))
print("expectation:", expectation)
print("variance:", variance)`,
    expected: `estimate=0: probability=1/16
estimate=1/4: probability=1/4
estimate=1/2: probability=3/8
estimate=3/4: probability=1/4
estimate=1: probability=1/16
total: 1
expectation: 1/2
variance: 1/16`,
  },
  coordinate: {
    title: 'Transform a density, not just its horizontal labels',
    code: `from math import log
from fractions import Fraction as F

a, b = 5, 3
p_mode = F(a-1, a+b-2)
p_at_eta_mode = F(a, a+b)
print("p-density mode:", p_mode)
print("p at log-odds-density mode:", p_at_eta_mode)
print("log-odds of p-mode:", f"{log(float(p_mode / (1-p_mode))):.6f}")
print("log-odds mode:", f"{log(float(p_at_eta_mode / (1-p_at_eta_mode))):.6f}")
# Beta(5,3) CDF: integrate 105*(p^4 - 2*p^5 + p^6).
def cdf(p):
    return 21*p**5 - 35*p**6 + 15*p**7
print("mass for 1/2 <= p <= 3/4:", cdf(F(3,4)) - cdf(F(1,2)))
print("corresponding eta interval:", f"[0, {log(3):.6f}]")`,
    expected: `p-density mode: 2/3
p at log-odds-density mode: 5/8
log-odds of p-mode: 0.693147
log-odds mode: 0.510826
mass for 1/2 <= p <= 3/4: 8681/16384
corresponding eta interval: [0, 1.098612]`,
  },
  failures: {
    title: 'Check support, ridges and a supremum that is never attained',
    code: `from math import exp, log, pi

data = [.2, .7, .4]
def uniform_likelihood(theta):
    return theta**(-len(data)) if theta > 0 and all(0 <= x <= theta for x in data) else 0.0
for theta in [.6, .7, 1.0]:
    print(f"uniform theta={theta}: L={uniform_likelihood(theta):.6f}")
for a, b in [(0,4), (1,3), (2,2)]:
    print(f"offsets ({a},{b}): SSE={sum((x-a-b)**2 for x in [2,3,4,7])}")
for variance in [1, .1, .01]:
    print(f"identical normal, variance={variance}: logL={-2*log(2*pi*variance):.6f}")
# x=[-1,1], y=[0,1], no intercept: L(w)=sigmoid(w)^2.
for w in [0, 2, 10]:
    print(f"separated w={w}: L={(1/(1+exp(-w)))**2:.6f}")
# A N(0,1) prior gives strictly decreasing derivative 2/(1+exp(w))-w.
low, high = 0.0, 2.0
for _ in range(80):
    middle = (low + high) / 2
    if 2/(1+exp(middle)) - middle > 0:
        low = middle
    else:
        high = middle
print(f"regularized finite MAP={(low+high)/2:.6f}")`,
    expected: `uniform theta=0.6: L=0.000000
uniform theta=0.7: L=2.915452
uniform theta=1.0: L=1.000000
offsets (0,4): SSE=14
offsets (1,3): SSE=14
offsets (2,2): SSE=14
identical normal, variance=1: logL=-3.675754
identical normal, variance=0.1: logL=0.929416
identical normal, variance=0.01: logL=5.534586
separated w=0: L=0.250000
separated w=2: L=0.775803
separated w=10: L=0.999909
regularized finite MAP=0.674832`,
  },
};
