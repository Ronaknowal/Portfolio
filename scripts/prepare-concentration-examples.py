"""Execute complete Python teaching programs; preserve the three original blocks."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'scratch/concentration-native'
OUT.mkdir(parents=True,exist_ok=True)
examples = {}

def add(name,title,code,expected=None):
    code = textwrap.dedent(code).strip()
    if expected is None:
        original_tree = ast.parse(code)
        code = ast.unparse(original_tree)
        assert ast.dump(ast.parse(code)) == ast.dump(original_tree)
    file = OUT/f'{name}.py'
    file.write_text(code+'\n',encoding='utf-8')
    actual = subprocess.check_output([sys.executable,'-X','utf8','-I',str(file)],text=True,encoding='utf-8').strip()
    if expected is not None:
        assert actual == expected.strip(), (name,actual,expected)
    examples[name] = {'title':title,'code':code,'expected':actual}

old = json.loads((ROOT/'docs/teaching/evidence/concentration-original-content.json').read_text(encoding='utf-8'))
for (name,title), original in zip([
    ('hoeffdingBudget','From a failure tolerance to 185 observations'),
    ('varianceRadii','Compare radii with a justified variance bound'),
    ('rareCount','Calculate a relative-count guarantee')
],old['original_programs']):
    add(name,title,original['code'],original['expected'])

add('exactBinomial','Compare an exact finite tail with several guarantees',r'''
from fractions import Fraction
from math import comb, exp, log, log1p

def binomial_tail(n, p, threshold):
    """Exact inclusive upper tail for rational p; no normal approximation."""
    if n < 0 or not 0 <= p <= 1:
        raise ValueError("nonnegative n and probability in [0,1] required")
    return sum((Fraction(comb(n, k)) * p**k * (1-p)**(n-k)
                for k in range(max(0,threshold),n+1)),Fraction(0))

n, p, threshold = 1000, Fraction(1,50), 30
mean = n*p
t = threshold-mean
r = float(t/mean)
q = Fraction(threshold,n)
kl = float(q)*log(float(q/p)) + float(1-q)*log(float((1-q)/(1-p)))
print("exact P(S >= 30):",f"{float(binomial_tail(n,p,threshold)):.8f}")
print("Hoeffding:",f"{exp(-2*float(t*t)/n):.8f}")
variance = float(n*p*(1-p))
print("Bernstein, c=1:",f"{exp(-float(t*t)/(2*(variance+float(t)/3))):.8f}")
print("mean-only Chernoff:",f"{exp(-float(mean)*((1+r)*log1p(r)-r)):.8f}")
print("binomial KL:",f"{exp(-n*kl):.8f}")
print("small rational tail:",binomial_tail(4,Fraction(1,2),3))
print("impossible threshold:",binomial_tail(4,Fraction(1,2),5))
''')

add('exponentialWitness','Choose an exponential witness without changing the event',r'''
from fractions import Fraction
from math import comb, exp, log, log1p, expm1

n, p, threshold = 20, Fraction(1,4), 10
law = [Fraction(comb(n,k))*p**k*(1-p)**(n-k) for k in range(n+1)]
exact_tail = sum(law[threshold:])

def witness(lam):
    # Each event indicator is at most exp(lam*(k-threshold)) for lam>=0.
    enumerated = sum(float(mass)*exp(lam*(k-threshold)) for k,mass in enumerate(law))
    factorized = exp(n*log1p(float(p)*expm1(lam))-lam*threshold)
    return enumerated,factorized

for lam in [0,.5,log(3),2]:
    enumerated,formula = witness(lam)
    assert abs(enumerated-formula) < 1e-12
    assert float(exact_tail) <= formula+1e-14
    print(f"lambda={lam:.6f}: raw upper bound={formula:.8f}")
print("actual event probability:",f"{float(exact_tail):.8f}")
print("optimal lambda:",f"{log(3):.8f}")
''')

add('unequalRanges','A bounded sum need not contain identically distributed variables',r'''
from fractions import Fraction
from itertools import product
from math import exp

# Independent fair choices on three different intervals.
supports = [(0,1),(0,2),(-1,1)]
mean = sum(Fraction(a+b,2) for a,b in supports)
threshold = Fraction(2)
outcomes = [sum(values) for values in product(*supports)]
actual = Fraction(sum(abs(value-mean)>=threshold for value in outcomes),len(outcomes))
width_squares = sum((b-a)**2 for a,b in supports)
bound = min(1,2*exp(-2*float(threshold**2)/width_squares))
print("mean of sum:",mean)
print("sum of squared widths:",width_squares)
print("exact two-sided tail:",actual)
print("Hoeffding upper bound:",f"{bound:.8f}")

# Fixed weighted means: scale each interval by its nonnegative weight.
weights = [Fraction(1,5),Fraction(3,10),Fraction(1,2)]
weighted_width_squares = sum(w*w for w in weights)
epsilon = Fraction(3,5)
print("weighted [0,1] bound:",f"{min(1,2*exp(-2*float(epsilon**2/weighted_width_squares))):.8f}")
''')

add('bernsteinInversion','Invert Bernstein and catch a false variance substitution',r'''
from fractions import Fraction
from math import sqrt,log,exp

def bernstein_radii(n,variance_upper,cap,delta):
    if n<1 or variance_upper<0 or cap<=0 or not 0<delta<1:
        raise ValueError("valid count, variance bound, centered cap and delta required")
    level = log(2/delta)
    linear = cap*level/(3*n)
    exact = linear+sqrt(2*variance_upper*level/n+linear**2)
    relaxed = sqrt(2*variance_upper*level/n)+2*linear
    return exact,relaxed

n, variance, cap, delta = 100,.09,1,.05
exact,relaxed = bernstein_radii(n,variance,cap,delta)
resubstituted = 2*exp(-n*exact**2/(2*(variance+cap*exact/3)))
print("quadratic radius:",f"{exact:.9f}")
print("convenient relaxed radius:",f"{relaxed:.9f}")
print("tail bound at quadratic radius:",f"{resubstituted:.9f}")

# A sample of 100 zeros has empirical variance 0, but p=.025 is possible.
n,p = 100,Fraction(1,40)
probability_all_zero = (1-p)**n
false_radius,_ = bernstein_radii(n,0,1,.05)
print("all-zero sample probability:",f"{float(probability_all_zero):.8f}")
print("plug-in radius:",f"{false_radius:.8f}","actual error:",float(p))
print("counterexample event violates this radius:",float(p)>false_radius)
print("this event alone exceeds the claimed .05 budget:",probability_all_zero>Fraction(1,20))
''')

add('samplingLaws','Exact laws for independent, copied and subset samples',r'''
from fractions import Fraction
from math import comb,exp

def choose(n,k):
    return comb(n,k) if 0<=k<=n else 0

def compare(n,epsilon):
    population,marked = 20,5
    p = Fraction(marked,population)
    independent = [Fraction(comb(n,k))*p**k*(1-p)**(n-k) for k in range(n+1)]
    copied = [1-p if k==0 else p if k==n else Fraction(0) for k in range(n+1)]
    subset = [Fraction(choose(marked,k)*choose(population-marked,n-k),choose(population,n)) for k in range(n+1)]
    result = {}
    for name,law in [('independent',independent),('copied',copied),('subset',subset)]:
        assert sum(law)==1
        assert sum(Fraction(k,n)*mass for k,mass in enumerate(law))==p
        result[name] = sum(mass for k,mass in enumerate(law) if abs(Fraction(k,n)-p)>=epsilon)
    return result

for n in [10,20]:
    result = compare(n,Fraction(1,5))
    print("n =",n)
    for name,probability in result.items():
        print(" ",name,f"{float(probability):.8f}")
    print(" Hoeffding expression:",f"{min(1,2*exp(-2*n*.2**2)):.8f}")
print("At n=20 the subset is the entire labelled population.")
''')

add('simultaneousBudget','Allocate error to a fixed family and to a sequence of times',r'''
from fractions import Fraction
from math import comb,log,sqrt,ceil

def exact_miss(n,budget):
    # p=.5; squared integer difference keeps the inclusive event transparent.
    boundary = 2*n*log(2/budget)
    return sum((Fraction(comb(n,k),2**n) for k in range(n+1)
                if (2*k-n)**2>=boundary),Fraction(0))

n,checks,delta = 100,100,.05
for label,budget in [('per-check only',delta),('allocated',delta/checks)]:
    miss = exact_miss(n,budget)
    family = 1-(1-miss)**checks  # independently generated checks for this calculation
    print(label,"radius",f"{sqrt(log(2/budget)/(2*n)):.8f}",
          "actual family failure",f"{float(family):.8f}")

# A conservative countable union budget; failure events need not be independent.
total_delta = Fraction(1,20)
horizon = 100
spent = sum((total_delta/Fraction(t*(t+1)) for t in range(1,horizon+1)),Fraction(0))
remaining = total_delta/Fraction(horizon+1)
assert spent+remaining==total_delta
print("first 100 time budgets:",spent,"remaining:",remaining)
for t in [100,1000]:
    budget = float(total_delta/Fraction(t*(t+1)))
    radius = sqrt(log(2/budget)/(2*t))
    print("anytime union radius at",t,f"{radius:.8f}")
print("changed 400 sample radius:",f"{sqrt(log(2/.05)/(2*400)):.8f}")
print("changed epsilon .05 / delta .01 count:",ceil(log(2/.01)/(2*.05**2)))
''')

target=ROOT/'src/learn/data/concentration-inequalities-examples.js'
target.write_text('export const concentrationInequalityExamples = '+json.dumps(examples,indent=2,ensure_ascii=False)+';\n',encoding='utf-8')
(OUT/'captured.json').write_text(json.dumps(examples,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({name:value['expected'] for name,value in examples.items()},indent=2,ensure_ascii=False))

