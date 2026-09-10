// Complete standard-library Python programs; each runs independently.
export const probabilityDistributionExamples = {
  eventRules: {
    title: "Restrict the sample space before dividing",
    code: String.raw`from fractions import Fraction

outcomes = set(range(1, 7))  # one fair six-sided die
event = {2, 4, 6}
condition = {4, 5, 6}

def probability(values):
    assert values <= outcomes
    return Fraction(len(values), len(outcomes))

def conditional(selected, given):
    return None if not given else Fraction(len(selected & given), len(given))

intersection = probability(event & condition)
union = probability(event | condition)
print("P(A), P(B), P(A and B):", probability(event), probability(condition), intersection)
print("P(A or B):", union)
print("P(A|B), P(B|A):", conditional(event, condition), conditional(condition, event))
print("independent:", intersection == probability(event) * probability(condition))
print("disjoint:", not event & condition)
print("empty conditioning event:", conditional(event, set()))

first, second = {1, 2}, {3, 4}
print("disjoint positive events independent:",
      probability(first & second) == probability(first) * probability(second))
assert union == probability(event) + probability(condition) - intersection
`,
    expected: "P(A), P(B), P(A and B): 1/2 1/2 1/3\nP(A or B): 2/3\nP(A|B), P(B|A): 2/3 2/3\nindependent: False\ndisjoint: False\nempty conditioning event: None\ndisjoint positive events independent: False",
  },
  moments: {
    title: "Keep the original weighted mean, then challenge its summary",
    code: String.raw`from fractions import Fraction

def moments(outcomes, masses):
    assert len(outcomes) == len(masses) and sum(masses) == 1
    assert all(p >= 0 for p in masses)
    mean = sum(x*p for x, p in zip(outcomes, masses))
    variance = sum((x-mean)**2*p for x, p in zip(outcomes, masses))
    return mean, variance

outcomes = [0, 1, 2]
masses = [Fraction(1, 5), Fraction(1, 2), Fraction(3, 10)]
mean, variance = moments(outcomes, masses)
print("original mean, variance:", float(mean), float(variance))
second_moment = sum(x*x*p for x, p in zip(outcomes, masses))
print("E[X^2], E[X]^2:", float(second_moment), float(mean**2))
changed_mean, changed_variance = moments([2*x+3 for x in outcomes], masses)
print("Y=2X+3 mean, variance:", float(changed_mean), float(changed_variance))
assert changed_mean == 2*mean+3 and changed_variance == 4*variance

for label, values, weights in [
    ("A", [-1, 1], [Fraction(1, 2)]*2),
    ("B", [-2, 0, 2], [Fraction(1, 8), Fraction(3, 4), Fraction(1, 8)]),
]:
    mean, variance = moments(values, weights)
    tail = sum(p for x, p in zip(values, weights) if abs(x) >= Fraction(3, 2))
    print(label, "mean:", mean, "variance:", variance, "P(|X|>=1.5):", tail)
`,
    expected: "original mean, variance: 1.1 0.49\nE[X^2], E[X]^2: 1.7 1.21\nY=2X+3 mean, variance: 5.2 1.96\nA mean: 0 variance: 1 P(|X|>=1.5): 0\nB mean: 0 variance: 1 P(|X|>=1.5): 1/4",
  },
  baseRates: {
    title: "Recover Bayes' rule from all four population cells",
    code: String.raw`from fractions import Fraction

sensitivity = Fraction(95, 100)
specificity = Fraction(90, 100)
false_positive = 1 - specificity

for prior in [Fraction(1, 100), Fraction(1, 5)]:
    cells = [
        prior*sensitivity, prior*(1-sensitivity),
        (1-prior)*false_positive, (1-prior)*specificity,
    ]
    assert sum(cells) == 1
    positive_mass = cells[0] + cells[2]
    negative_mass = cells[1] + cells[3]
    positive_posterior = cells[0] / positive_mass
    negative_posterior = cells[1] / negative_mass
    print("prior:", prior)
    print("expected counts in 100000:", [int(p*100000) for p in cells])
    print("positive posterior:", positive_posterior, f"{float(positive_posterior):.6f}")
    print("negative posterior:", negative_posterior, f"{float(negative_posterior):.6f}")
    assert positive_posterior*positive_mass + negative_posterior*negative_mass == prior

# If both explanations assign zero probability to a positive result,
# observing one is outside this model, not evidence for a posterior of zero.
zero_evidence = Fraction(0)
print("posterior for impossible evidence:",
      None if zero_evidence == 0 else Fraction(0)/zero_evidence)
`,
    expected: "prior: 1/100\nexpected counts in 100000: [950, 50, 9900, 89100]\npositive posterior: 19/217 0.087558\nnegative posterior: 1/1783 0.000561\nprior: 1/5\nexpected counts in 100000: [19000, 1000, 8000, 72000]\npositive posterior: 19/27 0.703704\nnegative posterior: 1/73 0.013699\nposterior for impossible evidence: None",
  },
  pairedEvidence: {
    title: "Enumerate fresh evidence and an exact copy",
    code: String.raw`from fractions import Fraction
from itertools import product

def conditional_pairs(positive_probability, copy_share):
    masses = {pair: Fraction(0) for pair in product([0, 1], repeat=2)}
    single = {1: positive_probability, 0: 1-positive_probability}
    for value, p in single.items():
        masses[value, value] += copy_share*p
    for first, second in masses:
        masses[first, second] += (1-copy_share)*single[first]*single[second]
    assert sum(masses.values()) == 1
    assert sum(p for (first, _), p in masses.items() if first) == positive_probability
    assert sum(p for (_, second), p in masses.items() if second) == positive_probability
    return masses

prior = Fraction(1, 100)
sensitivity = Fraction(19, 20)
false_positive = Fraction(1, 10)
one_positive = prior*sensitivity + (1-prior)*false_positive
for copy_share in [Fraction(0), Fraction(1, 2), Fraction(1)]:
    given_h = conditional_pairs(sensitivity, copy_share)
    given_not_h = conditional_pairs(false_positive, copy_share)
    joint = {pair: prior*given_h[pair] + (1-prior)*given_not_h[pair]
             for pair in given_h}
    posterior = prior*given_h[1, 1] / joint[1, 1]
    print("copy share:", copy_share)
    print("P(++|H), P(++|not H):", given_h[1, 1], given_not_h[1, 1])
    print("P(H|++):", f"{float(posterior):.6f}")
    print("unconditionally independent:", joint[1, 1] == one_positive**2)

# Conditional independence is the zero-copy branch, not independence
# after mixing cases from two different hypothesis populations.
`,
    expected: "copy share: 0\nP(++|H), P(++|not H): 361/400 1/100\nP(H|++): 0.476882\nunconditionally independent: False\ncopy share: 1/2\nP(++|H), P(++|not H): 741/800 11/200\nP(H|++): 0.145380\nunconditionally independent: False\ncopy share: 1\nP(++|H), P(++|not H): 19/20 1/10\nP(H|++): 0.087558\nunconditionally independent: False",
  },
  urnCounts: {
    title: "Derive counts from labeled sequences and subsets",
    code: String.raw`from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import comb

population = range(6)  # objects 0 and 1 are marked
marked = 2
draws = 3

for replacement in [True, False]:
    selections = list(product(population, repeat=draws) if replacement
                      else combinations(population, draws))
    counts = Counter(sum(item < marked for item in selection) for selection in selections)
    masses = [Fraction(counts[k], len(selections)) for k in range(draws+1)]
    p = Fraction(marked, len(population))
    if replacement:
        formula = [comb(draws, k)*p**k*(1-p)**(draws-k) for k in range(draws+1)]
    else:
        formula = [Fraction(comb(marked, k)*comb(6-marked, draws-k), comb(6, draws))
                   if k <= marked else Fraction(0) for k in range(draws+1)]
    assert masses == formula
    mean = sum(k*mass for k, mass in enumerate(masses))
    variance = sum((k-mean)**2*mass for k, mass in enumerate(masses))
    print("replacement:", replacement, "equally likely selections:", len(selections))
    print("P(X=0..3):", [str(mass) for mass in masses])
    print("mean:", mean, "variance:", variance, "P(X<=1):", sum(masses[:2]))

# Drawing the whole population without replacement fixes the count.
all_objects = next(combinations(population, 6))
print("six draws without replacement:", sum(item < marked for item in all_objects))
`,
    expected: "replacement: True equally likely selections: 216\nP(X=0..3): ['8/27', '4/9', '2/9', '1/27']\nmean: 1 variance: 2/3 P(X<=1): 20/27\nreplacement: False equally likely selections: 20\nP(X=0..3): ['1/5', '3/5', '1/5', '0']\nmean: 1 variance: 2/5 P(X<=1): 4/5\nsix draws without replacement: 2",
  },
  mixedDelay: {
    title: "Calculate a point mass, interval area and changed units",
    code: String.raw`from fractions import Fraction

width = Fraction(1, 5)  # seconds
atom = Fraction(3, 10)  # immediate completion at exactly zero

def cdf(x):
    if x < 0:
        return Fraction(0)
    return atom + (1-atom)*min(x/width, 1)

def interval_probability(left, right):
    assert 0 <= left <= right <= width
    cdf_left_limit = 0 if left == 0 else cdf(left)
    return cdf(right) - cdf_left_limit

for left, right in [(Fraction(0), Fraction(0)),
                    (Fraction(1, 20), Fraction(3, 20)),
                    (Fraction(0), Fraction(1, 10))]:
    print("closed interval:", str(left), str(right),
          "probability:", interval_probability(left, right))
density_seconds = (1-atom)/width
density_milliseconds = density_seconds/1000
print("density per second:", float(density_seconds))
print("density per millisecond:", float(density_milliseconds))
print("same positive interval area:",
      density_seconds*Fraction(1, 10), density_milliseconds*100)
mean = (1-atom)*width/2
variance = (1-atom)*width**2/3 - mean**2
print("mean seconds:", float(mean), "variance seconds^2:", f"{float(variance):.8f}")
assert cdf(width) == 1 and interval_probability(0, 0) == atom
`,
    expected: "closed interval: 0 0 probability: 3/10\nclosed interval: 1/20 3/20 probability: 7/20\nclosed interval: 0 1/10 probability: 13/20\ndensity per second: 3.5\ndensity per millisecond: 0.0035\nsame positive interval area: 7/20 7/20\nmean seconds: 0.07 variance seconds^2: 0.00443333",
  },
  countsAndWaits: {
    title: "Connect no arrivals, exponential waits and quantiles",
    code: String.raw`from math import exp, expm1, factorial, isclose, log1p

rate = 2.0  # events per minute, assumed homogeneous Poisson process
window = 1.5  # minutes
mean_count = rate*window
pmf = [exp(-mean_count)*mean_count**k/factorial(k) for k in range(9)]
print("mean count:", mean_count)
print("P(N=0):", f"{pmf[0]:.6f}")
print("P(T>window):", f"{exp(-rate*window):.6f}")
print("P(N>=1):", f"{-expm1(-mean_count):.6f}")
print("count mass beyond 8:", f"{1-sum(pmf):.6f}")
for quantile in [0.1, 0.5, 0.9]:
    wait = -log1p(-quantile)/rate
    assert isclose(-expm1(-rate*wait), quantile)
    print("wait quantile:", quantile, f"{wait:.6f}", "minutes")

uniforms = [0.18, 0.72, 0.41, 0.86, 0.09, 0.55, 0.28, 0.64, 0.33, 0.91, 0.12, 0.47]
time = 0.0
inside = []
for index, uniform in enumerate(uniforms, start=1):
    time += -log1p(-uniform)/rate
    if time <= window:
        inside.append((index, round(time, 6)))
print("fixed realization's arrivals inside window:", inside)
print("last supplied event is beyond window:", time > window)
print("memoryless survival for 2 more minutes:",
      f"{exp(-rate*3)/exp(-rate*1):.6f}", f"{exp(-rate*2):.6f}")

p = 0.25  # independent Bernoulli trials; T includes the successful trial
print("geometric P(T=3):", f"{(1-p)**2*p:.6f}", "mean trials:", 1/p)
`,
    expected: "mean count: 3.0\nP(N=0): 0.049787\nP(T>window): 0.049787\nP(N>=1): 0.950213\ncount mass beyond 8: 0.003803\nwait quantile: 0.1 0.052680 minutes\nwait quantile: 0.5 0.346574 minutes\nwait quantile: 0.9 1.151293 minutes\nfixed realization's arrivals inside window: [(1, 0.099225), (2, 0.735708), (3, 0.999525)]\nlast supplied event is beyond window: True\nmemoryless survival for 2 more minutes: 0.018316 0.018316\ngeometric P(T=3): 0.140625 mean trials: 4.0",
  },
  normalAreas: {
    title: "Standardize normal error and retain a small tail",
    code: String.raw`from math import erfc, exp, isclose, pi, sqrt
from statistics import NormalDist

mean = 0.0
sigma = 0.2  # volts; this is a stated synthetic sensor-error model
distribution = NormalDist(mean, sigma)
left, right = -0.1, 0.1
probability = distribution.cdf(right) - distribution.cdf(left)
standard = NormalDist()
standardized = standard.cdf((right-mean)/sigma) - standard.cdf((left-mean)/sigma)
assert isclose(probability, standardized)
density_at_mean = 1/(sigma*sqrt(2*pi))
print("density at mean, per volt:", f"{density_at_mean:.6f}")
print("P(-0.1<=error<=0.1):", f"{probability:.6f}")
z = (0.4-mean)/sigma
print("P(error>0.4):", f"{0.5*erfc(z/sqrt(2)):.6f}")
print("P(Z>8), direct survival:", f"{0.5*erfc(8/sqrt(2)):.9e}")

# Under Y=1000X, density per millivolt divides by 1000,
# while the probability of the corresponding interval is unchanged.
millivolts = NormalDist(1000*mean, 1000*sigma)
print("millivolt interval probability:", f"{millivolts.cdf(100)-millivolts.cdf(-100):.6f}")
print("density per millivolt:", f"{density_at_mean/1000:.9f}")
`,
    expected: "density at mean, per volt: 1.994711\nP(-0.1<=error<=0.1): 0.382925\nP(error>0.4): 0.022750\nP(Z>8), direct survival: 6.220960574e-16\nmillivolt interval probability: 0.382925\ndensity per millivolt: 0.001994711",
  },
  continuousEvidence: {
    title: "Use a continuous likelihood through shrinking intervals",
    code: String.raw`from math import exp, expm1, isclose

prior_fast = 0.5
rate_fast, rate_slow = 3.0, 1.0  # per minute; two synthetic operating modes

def posterior_from_densities(time, fast_rate, slow_rate):
    weighted_fast = prior_fast*fast_rate*exp(-fast_rate*time)
    weighted_slow = (1-prior_fast)*slow_rate*exp(-slow_rate*time)
    return weighted_fast/(weighted_fast+weighted_slow)

for time in [0.2, 1.0, 2.0]:
    posterior = posterior_from_densities(time, rate_fast, rate_slow)
    print("observed time:", time, "minutes; P(fast|time):", f"{posterior:.6f}")
    changed_units = posterior_from_densities(time*60, rate_fast/60, rate_slow/60)
    assert isclose(posterior, changed_units)

time = 0.2
limit = posterior_from_densities(time, rate_fast, rate_slow)
for width in [0.1, 0.001, 0.00001]:
    fast_mass = exp(-rate_fast*time)*(-expm1(-rate_fast*width))
    slow_mass = exp(-rate_slow*time)*(-expm1(-rate_slow*width))
    posterior = prior_fast*fast_mass/(prior_fast*fast_mass+(1-prior_fast)*slow_mass)
    print("interval width:", width, "posterior:", f"{posterior:.6f}")
print("density-limit posterior:", f"{limit:.6f}")
print("seconds and minutes posteriors agree:", True)
`,
    expected: "observed time: 0.2 minutes; P(fast|time): 0.667880\nobserved time: 1.0 minutes; P(fast|time): 0.288765\nobserved time: 2.0 minutes; P(fast|time): 0.052085\ninterval width: 0.1 posterior: 0.646101\ninterval width: 0.001 posterior: 0.667658\ninterval width: 1e-05 posterior: 0.667878\ndensity-limit posterior: 0.667880\nseconds and minutes posteriors agree: True",
  },
};
