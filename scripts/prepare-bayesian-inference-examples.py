"""Capture complete teaching programs; independent verification lives separately."""
from pathlib import Path
import ast
import json
import subprocess
import sys
import textwrap
import black

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "scratch/bayesian-inference-review/native"
DESTINATION.mkdir(parents=True, exist_ok=True)
programs = {}

def add(key, title, code, question):
    original = textwrap.dedent(code).strip()
    formatted = black.format_str(original, mode=black.Mode(line_length=88)).strip()
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(formatted)), key
    programs[key] = {"title": title, "code": formatted, "question": question}

beta_helpers = """\
from math import comb, factorial
from fractions import Fraction

def beta_integral(a, b):
    # Positive integer shapes only.
    return Fraction(factorial(a-1) * factorial(b-1), factorial(a+b-1))

def beta_tail(x, a, b, upper=False):
    n = a+b-1
    indices = range(a) if upper else range(a, n+1)
    return sum(comb(n, k) * x**k * (1-x)**(n-k) for k in indices)

def beta_quantile(q, a, b):
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low+high)/2
        below = beta_tail(mid, a, b) < q if q <= .5 else beta_tail(mid, a, b, True) > 1-q
        if below:
            low = mid
        else:
            high = mid
    return (low+high)/2
"""
add("betaUpdate", "Normalize the original visitor model", beta_helpers + """
a, b = 2+8, 2+2
mean = Fraction(a, a+b)
evidence_sequence = beta_integral(a, b)/beta_integral(2, 2)
evidence_count = comb(10, 8)*evidence_sequence
print("posterior:", (a, b), "mean:", mean, "mode:", Fraction(a-1, a+b-2))
print("95% credible interval:", tuple(round(beta_quantile(q, a, b), 6) for q in (.025, .975)))
print("P(theta > .70):", round(beta_tail(.7, a, b, True), 6))
print("sequence evidence:", evidence_sequence, "count evidence:", evidence_count)
print("no-success next-trial probability:", Fraction(1, 12))
""", "Predict why the evidence for a count differs from the evidence for one ordered sequence while the posterior stays the same.")

add("batchPrediction", "Integrate once for an entire future batch", beta_helpers + """
a, b, m = 10, 4, 10
p = Fraction(a, a+b)
integrated = [comb(m,k)*beta_integral(a+k,b+m-k)/beta_integral(a,b) for k in range(m+1)]
plugin = [comb(m,k)*p**k*(1-p)**(m-k) for k in range(m+1)]
def moments(masses):
    mean = sum(Fraction(k)*mass for k,mass in enumerate(masses))
    variance = sum((k-mean)**2*mass for k,mass in enumerate(masses))
    return mean, variance
assert sum(integrated) == sum(plugin) == 1
print("integrated mean, variance:", moments(integrated))
print("plug-in mean, variance:", moments(plugin))
print("P(at least 9), integrated / plug-in:", round(float(sum(integrated[9:])),6), round(float(sum(plugin[9:])),6))
print("correlation of two future outcomes:", Fraction(1,a+b+1))
print("one-future-trial agreement:", beta_integral(a+1,b)/beta_integral(a,b) == p)
""", "Will two models with the same expected number of successes give the same probability of nine or ten successes?")

add("sequentialSensitivity", "Compare fresh updates, reused rows and prior assumptions", """
from fractions import Fraction

def summary(a, b):
    return Fraction(a,a+b), Fraction(a*b,(a+b)**2*(a+b+1))

for prior in [(1,1),(2,2),(20,20),(3,7)]:
    posterior = (prior[0]+8,prior[1]+2)
    print("prior:",prior,"posterior:",posterior,"mean:",summary(*posterior)[0])
sequential = (2+8+3,2+2+2)
combined = (2+11,2+4)
assert sequential == combined == (13,6)
print("fresh sequential and combined:", sequential)
print("reusing the original rows:", (2+8+8,2+2+2))
before, after = summary(100,1)[1], summary(100,2)[1]
print("variance before/after a surprising failure:", round(float(before),8), round(float(after),8))
assert after > before
# A finite candidate model cannot recover prior-excluded support.
candidates, prior = [.2,.5,.8], [0.5,0.5,0.0]
weighted = [w*p**8*(1-p)**2 for p,w in zip(candidates,prior)]
normalizer = sum(weighted)
print("excluded .8 posterior mass:", weighted[-1]/normalizer)
""", "Can one surprising observation widen this parameter posterior even though information improves on average?")

add("gammaExposure", "Update an event rate using counted time", """
from fractions import Fraction
from math import comb

shape, rate = Fraction(2), Fraction(1)  # rate parameter has units of hours
counts, hours = [3,6], [Fraction(1,2),Fraction(2)]
a, b = shape+sum(counts), rate+sum(hours)
print("Gamma posterior shape / rate-hours:", a,b)
print("MLE events/hour:", Fraction(sum(counts),sum(hours)))
print("posterior mean events/hour:", a/b)
print("average of interval rates:", sum(Fraction(c,t) for c,t in zip(counts,hours))/2)
future_hours = Fraction(1)
p_zero = (b/(b+future_hours))**a
mean = a*future_hours/b
variance = mean*(1+future_hours/b)
print("next-hour expected count, variance:", mean,variance)
print("next-hour P(zero):", round(float(p_zero),6))
minutes_rate = b*60
assert a/minutes_rate == (a/b)/60
assert (minutes_rate/(minutes_rate+60))**a == p_zero
print("per-minute posterior mean:", a/minutes_rate)
print("same event probability under minutes:", True)
""", "Three events in half an hour and six in two hours do not supply the same rate evidence per observation row.")

add("dirichletCategories", "Keep categorical uncertainty on the simplex", """
from fractions import Fraction
from math import factorial

alpha, counts = [1,1,1],[6,3,1]
posterior = [a+c for a,c in zip(alpha,counts)]
total = sum(posterior)
means = [Fraction(a,total) for a in posterior]
variance = Fraction(posterior[0]*(total-posterior[0]),total**2*(total+1))
covariance = Fraction(-posterior[0]*posterior[1],total**2*(total+1))
assert sum(means) == 1
print("posterior shapes:",posterior)
print("next-category probabilities:",means)
print("variance first probability / covariance first-second:",variance,covariance)
unseen = [1+6,1+4,1+0]
print("declared but unseen third category:",Fraction(unseen[2],sum(unseen)))
def multivariate_beta(shapes):
    product = 1
    for a in shapes:
        product *= factorial(a-1)
    return Fraction(product,factorial(sum(shapes)-1))
sequence_evidence = multivariate_beta(posterior)/multivariate_beta(alpha)
count_multiplicity = factorial(10)//(factorial(6)*factorial(3)*factorial(1))
print("sequence / count evidence:",sequence_evidence,count_multiplicity*sequence_evidence)
""", "Why do the three posterior probabilities sum to one, and why does uncertainty in two categories have negative covariance?")

add("normalPrecision", "Separate uncertainty in a mean from one noisy observation", """
from fractions import Fraction
from statistics import NormalDist

values = [2,3,4,7]
prior_mean, prior_variance, noise_variance = Fraction(0),Fraction(1),Fraction(4)
n, sample_mean = len(values),Fraction(sum(values),len(values))
prior_precision, data_precision = 1/prior_variance,n/noise_variance
variance = 1/(prior_precision+data_precision)
mean = variance*(prior_mean*prior_precision+sample_mean*data_precision)
predictive_variance = variance+noise_variance
print("prior / data precision:",prior_precision,data_precision)
print("posterior mean / variance:",mean,variance)
print("one-measurement predictive variance:",predictive_variance)
for label,v in [("parameter",variance),("prediction",predictive_variance)]:
    distribution = NormalDist(float(mean),float(v)**.5)
    print(label,"95% interval:",tuple(round(distribution.inv_cdf(q),6) for q in (.025,.975)))
factor = 1000
new_variance = 1/(1/(prior_variance*factor**2)+n/(noise_variance*factor**2))
new_mean = new_variance*(prior_mean*factor/(prior_variance*factor**2)+sum(values)*factor/(noise_variance*factor**2))
assert new_mean == factor*mean and new_variance == factor**2*variance
print("unit-converted mean / variance:",new_mean,new_variance)
""", "Will the interval for the next reading become as narrow as the interval for its shared mean?")

add("unknownVariance", "Complete a joint normal model when noise variance is unknown", """
# Python 3.12; install SciPy for Student-t quantiles: python -m pip install scipy
from math import sqrt
from scipy.stats import t

values = [2.,3.,4.,7.]
m0,kappa0,alpha0,beta0 = 0.,1.,2.,2.
# sigma^2 ~ Inverse-Gamma(alpha0,beta0), density proportional to
# v^(-alpha0-1) exp(-beta0/v); mu | v ~ Normal(m0,v/kappa0).
n = len(values)
average = sum(values)/n
sse = sum((x-average)**2 for x in values)
kappa = kappa0+n
mean = (kappa0*m0+n*average)/kappa
alpha = alpha0+n/2
beta = beta0+sse/2+kappa0*n*(average-m0)**2/(2*kappa)
df = 2*alpha
parameter_scale = sqrt(beta/(alpha*kappa))
prediction_scale = sqrt(beta*(kappa+1)/(alpha*kappa))
print("posterior m, kappa, alpha, beta:",mean,kappa,alpha,beta)
print("posterior expected variance:",round(beta/(alpha-1),6))
print("predictive Student-t df / scale:",df,round(prediction_scale,6))
print("predictive variance:",round(df/(df-2)*prediction_scale**2,6))
print("mean 95% interval:",tuple(round(float(x),6) for x in t.ppf([.025,.975],df,loc=mean,scale=parameter_scale)))
print("prediction 95% interval:",tuple(round(float(x),6) for x in t.ppf([.025,.975],df,loc=mean,scale=prediction_scale)))
""", "Unknown measurement variance changes the posterior family and the predictive tails; it is not the known-variance formula with a guessed plug-in value.")

add("predictivePatterns", "Ask a model about a pattern its fitted totals discard", beta_helpers + """
from itertools import product

patterns = [[1]*8+[0]*2,[1,1,0,1,1,1,0,1,1,1]]
def runs(sequence):
    return 1+sum(a!=b for a,b in zip(sequence,sequence[1:]))
sequences = list(product([0,1],repeat=10))
for mode,a,b in [("prior",2,2),("posterior",10,4)]:
    masses = {}
    for sequence in sequences:
        s = sum(sequence)
        mass = beta_integral(a+s,b+10-s)/beta_integral(a,b)
        r = runs(sequence)
        masses[r] = masses.get(r,Fraction(0))+mass
    assert sum(masses.values()) == 1
    for pattern in patterns:
        tail = sum(mass for r,mass in masses.items() if r<=runs(pattern))
        print(mode,"observed runs:",runs(pattern),"lower-tail probability:",round(float(tail),6))
same_count = [sequence for sequence in sequences if sum(sequence)==8]
print("same-count sequences:",len(same_count))
for pattern in patterns:
    tail = Fraction(sum(runs(sequence)<=runs(pattern) for sequence in same_count),len(same_count))
    print("conditional on eight successes, runs <=",runs(pattern),":",tail)
""", "Both orderings yield Beta(10,4), but do they look equally typical under the assumed common-rate sequence model?")

add("modelEvidence", "Compare complete models using the same observed event", beta_helpers + """
s,f = 8,2
null_sequence = Fraction(1,2)**(s+f)
prior_null,prior_alternative = Fraction(1,2),Fraction(1,2)
for a,b in [(1,1),(2,2),(20,20)]:
    alternative_sequence = beta_integral(a+s,b+f)/beta_integral(a,b)
    bayes_factor = alternative_sequence/null_sequence
    posterior_alternative = alternative_sequence*prior_alternative/(alternative_sequence*prior_alternative+null_sequence*prior_null)
    count_factor = comb(s+f,s)
    assert (alternative_sequence*count_factor)/(null_sequence*count_factor) == bayes_factor
    print("alternative prior:",(a,b),"Bayes factor:",round(float(bayes_factor),6),"posterior model probability:",round(float(posterior_alternative),6))
print("null model assigns point probability through a separate model indicator")
""", "Does a prior over rates disappear from the model evidence just because it has little effect on one point estimate?")

add("decisionLoss", "An inference result does not choose the loss function", beta_helpers + """
a,b = 10,4
q = beta_tail(Fraction(7,10),a,b,True)
mean = Fraction(a,a+b)
wrong_launch_cost,missed_launch_cost = 19,1
risk_launch = wrong_launch_cost*(1-q)
risk_wait = missed_launch_cost*q
print("posterior P(rate > .70):",round(float(q),6))
print("threshold-loss launch / wait risk:",round(float(risk_launch),6),round(float(risk_wait),6))
print("threshold-loss action:","launch" if risk_launch<risk_wait else "wait")
expected_net = 100*mean-70
print("different utility: expected net per trial:",round(float(expected_net),6))
print("linear-utility action:","launch" if expected_net>0 else "wait")
""", "Why can two fully specified decisions disagree while using exactly the same posterior?")

add("nonconjugateGrid", "Refine a small nonconjugate posterior calculation", """
from math import log,log1p,exp

def grid_inference(cells):
    width = 1/cells
    grid = [(index+.5)*width for index in range(cells)]
    log_kernel = []
    for theta in grid:
        reported_probability = .1+.8*theta
        # Beta(2,2) prior; eight reported successes and two reported failures.
        log_kernel.append(log(6)+log(theta)+log1p(-theta)+8*log(reported_probability)+2*log1p(-reported_probability))
    peak = max(log_kernel)
    scaled = [exp(value-peak) for value in log_kernel]
    total = sum(scaled)
    weights = [value/total for value in scaled]
    evidence = exp(peak)*total*width  # retain cell width for the integral
    mean = sum(theta*weight for theta,weight in zip(grid,weights))
    threshold = sum(weight for theta,weight in zip(grid,weights) if theta>.7)
    return mean,threshold,evidence

for cells in [100,1000,10000]:
    print("cells:",cells,"mean, P(theta>.7), sequence evidence:",tuple(round(value,8) for value in grid_inference(cells)))
print("reported success chance is .1+.8*theta, so the posterior is not generally Beta")
""", "Where must the grid-cell width remain even though it cancels from equal-width normalized weights?")

captured = {}
for key, item in programs.items():
    filename = DESTINATION / f"{key}.py"
    filename.write_text(item["code"] + "\n", encoding="utf-8")
    run = subprocess.run([sys.executable, str(filename)], check=True, capture_output=True, text=True, cwd=ROOT)
    captured[key] = {**item, "expected": run.stdout.rstrip()}

output = "// Complete independently executed programs; outputs captured by scripts/prepare-bayesian-inference-examples.py.\n"
output += "export const bayesianInferenceExamples = " + json.dumps(captured, ensure_ascii=False, indent=2) + ";\n"
(ROOT / "src/learn/data/bayesian-inference-examples.js").write_text(output, encoding="utf-8")
(DESTINATION / "captured.json").write_text(json.dumps(captured, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: item["expected"] for key,item in captured.items()}, indent=2))
