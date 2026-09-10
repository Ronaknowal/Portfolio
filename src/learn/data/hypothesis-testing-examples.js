// Complete programs executed with Python 3.12, NumPy 2.3.5 and SciPy 1.18.1.
export const hypothesisExamples = {
  paired: {
    title: "Calculate the paired estimate, t interval and test from raw timings",
    code: String.raw`import numpy as np
from scipy import stats

old = np.array([102, 110, 98, 105, 100], dtype=float)
new = np.array([100, 106, 99, 102, 98], dtype=float)

def paired_report(old, new, confidence=0.95):
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    if old.ndim != 1 or old.shape != new.shape or old.size < 2:
        raise ValueError("Use equal-length one-dimensional arrays with at least two pairs.")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("All timings must be finite; handle missing pairs explicitly.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one.")
    try:
        with np.errstate(over="raise", invalid="raise"):
            d = old - new
            mean, sd = d.mean(), d.std(ddof=1)
    except FloatingPointError as error:
        raise ValueError("The timings exceed this floating-point calculation's range.") from error
    if sd == 0:
        raise ValueError("Zero sample spread: the ordinary t pivot is undefined.")
    se, df = sd / np.sqrt(d.size), d.size - 1
    critical = stats.t.ppf((1 + confidence) / 2, df)
    interval = (mean - critical * se, mean + critical * se)
    test = stats.ttest_rel(old, new, alternative="two-sided", nan_policy="raise")
    np.testing.assert_allclose(interval, test.confidence_interval(confidence))
    return d, mean, sd, se, df, critical, interval, test

d, mean, sd, se, df, critical, interval, test = paired_report(old, new)
print("savings ms:", d.tolist())
print(f"mean={mean:.6f} sd={sd:.6f} se={se:.6f} df={df}")
print(f"critical={critical:.6f} CI95=[{interval[0]:.6f}, {interval[1]:.6f}]")
print(f"t={test.statistic:.6f} two-sided p={test.pvalue:.6f}")`,
    expected: String.raw`savings ms: [2.0, 4.0, -1.0, 3.0, 2.0]
mean=2.000000 sd=1.870829 se=0.836660 df=4
critical=2.776445 CI95=[-0.322941, 4.322941]
t=2.390457 two-sided p=0.075130`,
  },
  coverage: {
    title: "Repeat whole normal experiments with known and estimated spread",
    code: String.raw`import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)
truth, sigma, n, experiments = 100.0, 10.0, 25, 4000
samples = rng.normal(truth, sigma, size=(experiments, n))
means = samples.mean(axis=1)
estimated_sd = samples.std(axis=1, ddof=1)
for label, critical, standard_error in [
    ("known sigma, z", stats.norm.ppf(0.975), sigma / np.sqrt(n)),
    ("estimated sigma, t", stats.t.ppf(0.975, n - 1), estimated_sd / np.sqrt(n)),
]:
    low = means - critical * standard_error
    high = means + critical * standard_error
    count = np.count_nonzero((low <= truth) & (truth <= high))
    print(f"{label}: {count}/{experiments} cover = {count / experiments:.3%}")
    print(f"first interval: [{low[0]:.3f}, {high[0]:.3f}] ms")
print("These finite simulation counts estimate coverage; neither must equal 95%.")`,
    expected: String.raw`known sigma, z: 3813/4000 cover = 95.325%
first interval: [95.481, 103.321] ms
estimated sigma, t: 3809/4000 cover = 95.225%
first interval: [96.524, 102.279] ms
These finite simulation counts estimate coverage; neither must equal 95%.`,
  },
  tails: {
    title: "Change the scientific question and keep the tail and bound consistent",
    code: String.raw`import numpy as np
from scipy import stats

d = np.array([2, 4, -1, 3, 2], dtype=float)
for alternative in ["two-sided", "greater", "less"]:
    result = stats.ttest_1samp(d, popmean=0.0, alternative=alternative)
    interval = result.confidence_interval(0.95)
    print(f"{alternative}: p={result.pvalue:.6f}, CI95=[{interval.low:.6f}, {interval.high:.6f}]")

# Shift the entire paired dataset for a transparent teaching counterfactual.
shifted = d + 1
useful = stats.ttest_1samp(shifted, popmean=1.0, alternative="greater")
lower95 = shifted.mean() - stats.t.ppf(0.95, 4) * stats.sem(shifted)
print(f"shifted: p for saving >1 ms = {useful.pvalue:.6f}")
print(f"shifted: one-sided 95% lower bound = {lower95:.6f} ms")
print("Choose this direction and threshold before inspecting the result.")`,
    expected: String.raw`two-sided: p=0.075130, CI95=[-0.322941, 4.322941]
greater: p=0.037565, CI95=[0.216369, inf]
less: p=0.962435, CI95=[-inf, 3.783631]
shifted: p for saving >1 ms = 0.037565
shifted: one-sided 95% lower bound = 1.216369 ms
Choose this direction and threshold before inspecting the result.`,
  },
  equivalence: {
    title: "Test a declared negligible-effect interval instead of accepting a point null",
    code: String.raw`import numpy as np
from scipy import stats

original = np.array([2, 4, -1, 3, 2], dtype=float)
tolerance = 0.5  # ms; a hypothetical criterion declared before observing data
for name, d in [("centered, original spread", original - 2),
                ("centered, quarter spread", (original - 2) / 4)]:
    lower_test = stats.ttest_1samp(d, -tolerance, alternative="greater")
    upper_test = stats.ttest_1samp(d, tolerance, alternative="less")
    ordinary = stats.ttest_1samp(d, 0)
    ci90 = ordinary.confidence_interval(0.90)
    ci95 = ordinary.confidence_interval(0.95)
    equivalent = max(lower_test.pvalue, upper_test.pvalue) < 0.05
    by_interval = (-tolerance < ci90.low) and (ci90.high < tolerance)
    assert equivalent == by_interval
    print(name)
    print(f"point-null p={ordinary.pvalue:.6f}, TOST p={max(lower_test.pvalue, upper_test.pvalue):.6f}")
    print(f"CI90=[{ci90.low:.6f}, {ci90.high:.6f}], CI95=[{ci95.low:.6f}, {ci95.high:.6f}]")
    print(f"equivalent within +/-{tolerance} ms: {equivalent}")`,
    expected: String.raw`centered, original spread
point-null p=1.000000, TOST p=0.291142
CI90=[-1.783631, 1.783631], CI95=[-2.322941, 2.322941]
equivalent within +/-0.5 ms: False
centered, quarter spread
point-null p=1.000000, TOST p=0.037565
CI90=[-0.445908, 0.445908], CI95=[-0.580735, 0.580735]
equivalent within +/-0.5 ms: True`,
  },
  prediction: {
    title: "Compare uncertainty about a mean with uncertainty about a new observation",
    code: String.raw`import numpy as np
from scipy import stats

mean, sigma, confidence = 100.0, 10.0, 0.95
critical = stats.norm.ppf((1 + confidence) / 2)
for n in [5, 25, 100]:
    mean_margin = critical * sigma / np.sqrt(n)
    prediction_margin = critical * sigma * np.sqrt(1 + 1 / n)
    print(f"n={n}: mean CI=[{mean-mean_margin:.3f}, {mean+mean_margin:.3f}] ms")
    print(f"       future-observation PI=[{mean-prediction_margin:.3f}, {mean+prediction_margin:.3f}] ms")
print(f"prediction half-width limit: {critical * sigma:.3f} ms")`,
    expected: String.raw`n=5: mean CI=[91.235, 108.765] ms
       future-observation PI=[78.530, 121.470] ms
n=25: mean CI=[96.080, 103.920] ms
       future-observation PI=[80.012, 119.988] ms
n=100: mean CI=[98.040, 101.960] ms
       future-observation PI=[80.303, 119.697] ms
prediction half-width limit: 19.600 ms`,
  },
  power: {
    title: "Plan an independent normal-mean test before collecting the data",
    code: String.raw`from math import ceil, sqrt
from scipy import stats

alpha, desired_power, sigma = 0.05, 0.80, 4.0
critical = stats.norm.ppf(1 - alpha)
for effect in [2.0, 0.5]:
    n = ceil(((critical + stats.norm.ppf(desired_power)) * sigma / effect) ** 2)
    se = sigma / sqrt(n)
    power = stats.norm.sf(critical - effect / se)
    previous_power = stats.norm.sf(critical - effect * sqrt(n - 1) / sigma)
    assert power >= desired_power and previous_power < desired_power
    print(f"true saving={effect:.1f} ms: planned n={n}, power={power:.6f}")
    print(f"one fewer observation: power={previous_power:.6f}")
print("Exact for independent Normal data with known sigma and a predeclared upper-tail test.")`,
    expected: String.raw`true saving=2.0 ms: planned n=25, power=0.803765
one fewer observation: power=0.789485
true saving=0.5 ms: planned n=396, power=0.800278
one fewer observation: power=0.799398
Exact for independent Normal data with known sigma and a predeclared upper-tail test.`,
  },
  multiplicity: {
    title: "Calculate family risk and execute the Holm step-down rule",
    code: String.raw`import numpy as np

alpha, number_of_tests = 0.05, 20
print(f"20 independent true nulls, raw alpha=.05: FWER={1-(1-alpha)**number_of_tests:.6f}")
print(f"Bonferroni per-test threshold={alpha/number_of_tests:.6f}")

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not p.size or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Use a nonempty vector of valid p-values.")
    order = np.argsort(p, kind="stable")
    scaled = (p.size - np.arange(p.size)) * p[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted

p = np.array([0.04, 0.001, 0.8, 0.01, 0.2])
adjusted = holm_adjust(p)
print("original order p:", p.tolist())
print("Holm adjusted:", np.round(adjusted, 6).tolist())
print("reject at .05:", (adjusted < alpha).tolist())
print("Holm controls family-wise error with valid p-values, without requiring independent tests.")`,
    expected: String.raw`20 independent true nulls, raw alpha=.05: FWER=0.641514
Bonferroni per-test threshold=0.002500
original order p: [0.04, 0.001, 0.8, 0.01, 0.2]
Holm adjusted: [0.12, 0.005, 0.8, 0.04, 0.4]
reject at .05: [False, True, False, True, False]
Holm controls family-wise error with valid p-values, without requiring independent tests.`,
  },
  peeking: {
    title: "Enumerate every fair-coin path instead of treating repeated looks as independent",
    code: String.raw`from itertools import product
from math import comb
from fractions import Fraction

def two_sided_p(n, heads):
    distance = abs(2 * heads - n)
    extreme = sum(comb(n, k) for k in range(n + 1) if abs(2 * k - n) >= distance)
    return Fraction(extreme, 2 ** n)

maximum, alpha = 12, Fraction(1, 20)
fixed_count = peek_count = 0
first_hits = [0] * maximum
for sequence in product([0, 1], repeat=maximum):
    heads, first = 0, None
    for n, outcome in enumerate(sequence, 1):
        heads += outcome
        if first is None and two_sided_p(n, heads) < alpha:
            first = n
    fixed_count += two_sided_p(maximum, heads) < alpha
    if first is not None:
        peek_count += 1
        first_hits[first - 1] += 1
total = 2 ** maximum
print(f"fixed last look: {fixed_count}/{total} = {fixed_count/total:.6f}")
print(f"stop at any first rejection: {peek_count}/{total} = {peek_count/total:.6f}")
print("first-hit path counts by look:", first_hits)
print("All 4096 paths are equally likely under the independent fair-coin null.")`,
    expected: String.raw`fixed last look: 158/4096 = 0.038574
stop at any first rejection: 290/4096 = 0.070801
first-hit path counts by look: [0, 0, 0, 0, 0, 128, 0, 0, 96, 0, 0, 66]
All 4096 paths are equally likely under the independent fair-coin null.`,
  },
  clusters: {
    title: "Calculate why repeated measurements do not create independent units",
    code: String.raw`from math import sqrt

# X[j,r] = mu + b[j] + e[j,r]; all b and e mutually independent.
# Var(b)=tau2, Var(e)=sigma2, with equal cluster sizes.
m, r, tau2, sigma2 = 5, 2000, 20.0, 80.0
n = m * r
icc = tau2 / (tau2 + sigma2)
actual_variance = tau2 / m + sigma2 / n
naive_variance = (tau2 + sigma2) / n
design_effect = 1 + (r - 1) * icc
assert abs(actual_variance / naive_variance - design_effect) < 1e-10
print(f"observations={n}, independent clusters={m}, ICC={icc:.3f}")
print(f"naive SE={sqrt(naive_variance):.6f}, actual SE={sqrt(actual_variance):.6f}")
print(f"variance ratio={design_effect:.6f}, effective n={n/design_effect:.6f}")
print("Effective n here only matches this mean's variance; it is not a universal t degrees-of-freedom rule.")`,
    expected: String.raw`observations=10000, independent clusters=5, ICC=0.200
naive SE=0.100000, actual SE=2.001999
variance ratio=400.800000, effective n=24.950100
Effective n here only matches this mean's variance; it is not a universal t degrees-of-freedom rule.`,
  },
  welch: {
    title: "Use independent-group standard errors when no meaningful pairing exists",
    code: String.raw`import numpy as np
from scipy import stats

group_a = np.array([10, 12, 9, 11, 14], dtype=float)
group_b = np.array([8, 13, 7, 10, 9, 15, 6], dtype=float)
n_a, n_b = len(group_a), len(group_b)
v_a, v_b = group_a.var(ddof=1) / n_a, group_b.var(ddof=1) / n_b
se = np.sqrt(v_a + v_b)
df = (v_a + v_b) ** 2 / (v_a ** 2 / (n_a - 1) + v_b ** 2 / (n_b - 1))
result = stats.ttest_ind(group_a, group_b, equal_var=False)
ci = result.confidence_interval(0.95)
np.testing.assert_allclose(df, result.df)
print(f"mean difference={group_a.mean()-group_b.mean():.6f}, SE={se:.6f}, df={df:.6f}")
print(f"Welch t={result.statistic:.6f}, p={result.pvalue:.6f}")
print(f"CI95=[{ci.low:.6f}, {ci.high:.6f}]")
print("The Welch t reference is approximate in general, even for independent normal groups.")`,
    expected: String.raw`mean difference=1.485714, SE=1.500068, df=9.793479
Welch t=0.990431, p=0.345802
CI95=[-1.866224, 4.837653]
The Welch t reference is approximate in general, even for independent normal groups.`,
  },
  wilson: {
    title: "Invert a binomial score inequality and expose the zero-success Wald failure",
    code: String.raw`from math import sqrt
from scipy import stats

def wilson(k, n, confidence=0.95):
    if type(n) is not int or type(k) is not int or n < 1 or not 0 <= k <= n:
        raise ValueError("Use integer n>=1 and integer 0<=k<=n.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one.")
    z = stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - margin, center + margin

for k, n in [(0, 10), (3, 10), (10, 10)]:
    low, high = wilson(k, n)
    oracle = stats.binomtest(k, n).proportion_ci(method="wilson")
    assert abs(low - oracle.low) < 1e-12 and abs(high - oracle.high) < 1e-12
    print(f"{k}/{n}: Wilson95=[{low:.6f}, {high:.6f}]")
print("The plug-in Wald interval at 0/10 is [0,0]; that is not justified certainty.")
print("Wilson coverage is approximate; discreteness prevents uniform exact 95% coverage.")`,
    expected: String.raw`0/10: Wilson95=[0.000000, 0.277533]
3/10: Wilson95=[0.107791, 0.603222]
10/10: Wilson95=[0.722467, 1.000000]
The plug-in Wald interval at 0/10 is [0,0]; that is not justified certainty.
Wilson coverage is approximate; discreteness prevents uniform exact 95% coverage.`,
  },
  signFlip: {
    title: "Enumerate all 32 sign assignments under an explicit symmetry null",
    code: String.raw`from itertools import product
import numpy as np
from scipy import stats

d = np.array([2, 4, -1, 3, 2], dtype=int)
observed_sum = int(d.sum())
sums = np.array([sum(s * int(x) for s, x in zip(signs, d))
                 for signs in product([-1, 1], repeat=d.size)])
extreme = int(np.count_nonzero(np.abs(sums) >= abs(observed_sum)))
result = stats.permutation_test((d,), np.mean, permutation_type="samples",
                                alternative="two-sided", n_resamples=np.inf)
assert abs(extreme / len(sums) - result.pvalue) < 1e-12
print(f"observed mean={observed_sum/len(d):.3f} ms")
print(f"as-or-more-extreme assignments={extreme}/{len(sums)}, p={extreme/len(sums):.6f}")
print("Assumption: independent sign-symmetric differences around zero, or justified random swaps under a sharp null.")
print("A zero mean alone does not make every sign assignment equally likely.")`,
    expected: String.raw`observed mean=2.000 ms
as-or-more-extreme assignments=4/32, p=0.125000
Assumption: independent sign-symmetric differences around zero, or justified random swaps under a sharp null.
A zero mean alone does not make every sign assignment equally likely.`,
  },
  bootstrap: {
    title: "Resample independent paired differences, preserving the chosen statistic",
    code: String.raw`from itertools import product
import numpy as np

d = np.array([2, 4, -1, 3, 2], dtype=float)
# Enumerate the empirical bootstrap exactly for this tiny example: n**n resamples.
means = np.array([d[list(indices)].mean() for indices in product(range(len(d)), repeat=len(d))])
interval = np.quantile(means, [0.025, 0.975], method="linear")
print(f"empirical resamples={len(means)}")
print(f"bootstrap mean={means.mean():.6f}, bootstrap SD={means.std(ddof=0):.6f}")
print(f"percentile endpoints=[{interval[0]:.6f}, {interval[1]:.6f}]")
print("Exact enumeration removes Monte Carlo error, not the small-sample approximation of coverage.")
print("This empirical distribution cannot invent rare workloads missing from the original five pairs.")`,
    expected: String.raw`empirical resamples=3125
bootstrap mean=2.000000, bootstrap SD=0.748331
percentile endpoints=[0.400000, 3.200000]
Exact enumeration removes Monte Carlo error, not the small-sample approximation of coverage.
This empirical distribution cannot invent rare workloads missing from the original five pairs.`,
  },
  finiteCoverage: {
    title: "Sum binomial probabilities to evaluate actual interval coverage",
    code: String.raw`from math import sqrt
from scipy import stats

n, confidence = 10, 0.95
z = stats.norm.ppf((1 + confidence) / 2)
intervals = {"wald": [], "wilson": [], "exact": []}
for k in range(n + 1):
    estimate = k / n
    margin = z * sqrt(estimate * (1 - estimate) / n)
    intervals["wald"].append((estimate - margin, estimate + margin))
    for method in ["wilson", "exact"]:
        interval = stats.binomtest(k, n).proportion_ci(confidence, method=method)
        intervals[method].append((interval.low, interval.high))
for true_p in [0.04, 0.2, 0.5]:
    print(f"true p={true_p:.2f}, n={n}")
    for method, limits in intervals.items():
        coverage = sum(stats.binom.pmf(k, n, true_p)
                       for k, (low, high) in enumerate(limits) if low <= true_p <= high)
        print(f"  {method}: coverage={coverage:.6f}")
print("This sums every possible dataset count; no estimated Monte Carlo frequency is involved.")`,
    expected: String.raw`true p=0.04, n=10
  wald: coverage=0.334725
  wilson: coverage=0.941846
  exact: coverage=0.993786
true p=0.20, n=10
  wald: coverage=0.886256
  wilson: coverage=0.967207
  exact: coverage=0.993631
true p=0.50, n=10
  wald: coverage=0.890625
  wilson: coverage=0.978516
  exact: coverage=0.978516
This sums every possible dataset count; no estimated Monte Carlo frequency is involved.`,
  },
};
