"""Independent finite-law, scipy, Fraction and scalar-root concentration checks."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
from io import StringIO
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, hypergeom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/concentration-verification'
OUT.mkdir(parents=True, exist_ok=True)

def node(source):
    return json.loads(subprocess.check_output(['node', '--input-type=module', '-e', source], cwd=ROOT, text=True, encoding='utf-8'))

def close(actual, expected, tolerance=2e-11):
    if expected == -math.inf:
        assert actual is None, (actual, expected)
    else:
        assert actual is not None and abs(actual-expected) <= tolerance * max(1, abs(expected)), (actual, expected)

examples = node("import {concentrationInequalityExamples as e} from './src/learn/data/concentration-inequalities-examples.js'; console.log(JSON.stringify(e));")
original = json.loads((ROOT/'docs/teaching/evidence/concentration-original-content.json').read_text(encoding='utf-8'))
for name, before in zip(['hoeffdingBudget', 'varianceRadii', 'rareCount'], original['original_programs']):
    assert examples[name]['code'] == before['code']
    assert examples[name]['expected'] == before['expected']
native = {}
for name, example in examples.items():
    file = OUT/f'{name}.py'
    file.write_text(example['code'], encoding='utf-8')
    actual = subprocess.check_output([sys.executable, '-X', 'utf8', '-I', str(file)], encoding='utf-8').strip()
    assert actual == example['expected'], (name, actual)
    namespace = {}
    with redirect_stdout(StringIO()):
        exec(compile(example['code'], str(file), 'exec'), namespace)
    native[name] = namespace

data = node(r"""
import * as m from './src/learn/data/concentration-inequalities-models.js';
const tails=[], budgets=[], samples=[], families=[], witnesses=[];
for(const n of [1,2,8,20,50,100,200]) for(const p of [0,1,2,10,25,50,90,99,100]) for(let k=0;k<=n+1;k++){
 const t=m.countTail(n,p,k); tails.push({n,p,k,exact:t.exactLog,complement:t.exactComplementLog,bounds:t.bounds});
}
for(const n of [10,50,100,185,400,1000]) for(const delta of [.005,.01,.05,.1]) for(const width of [1,10]) for(const variance of [0,.01,.09,.25]){
 const r=m.precisionBudget(n,delta,width,variance,.1); delete r.curve; budgets.push(r);
}
for(let n=1;n<=20;n++) for(let e=1;e<=80;e++) samples.push(m.samplingComparison(n,e));
for(const n of [20,50,100,400]) for(const delta of [.01,.05,.1]) for(const k of [1,2,5,25,50,100]) for(const relation of ['independent','identical']){
 const r=m.familyBudget(n,k,delta,relation); delete r.curve; families.push(r);
}
for(let index=0;index<=100;index++){ const r=m.exponentialWitness(index/50); delete r.curve; witnesses.push(r); }
const invalid=[];
for(const action of [()=>m.binomialLaw(0,50),()=>m.binomialLaw(10,-1),()=>m.countTail(10,50,12),()=>m.exponentialWitness(-1),()=>m.exponentialWitness(Infinity),()=>m.precisionBudget(0),()=>m.precisionBudget(100,0),()=>m.precisionBudget(100,.05,1,.26),()=>m.samplingComparison(21),()=>m.samplingComparison(20,0),()=>m.familyBudget(100,101),()=>m.familyBudget(100,20,.05,'unknown')]){
 try{action();invalid.push(false)}catch(error){invalid.push(error instanceof RangeError)}
}
const small=m.countTail(2000,1,1800),near=m.countTail(200,90,1),extreme=m.countTail(2000,99,1);
console.log(JSON.stringify({tails,budgets,samples,families,witnesses,invalid,small,near,extreme,
 format:{zero:m.formatLogProbability(-Infinity),one:m.formatLogProbability(0),tiny:m.formatLogProbability(small.exactLog),near:m.formatLogProbability(near.exactLog,near.exactComplementLog),extreme:m.formatLogProbability(extreme.exactLog,extreme.exactComplementLog)}}));
""")
assert all(data['invalid'])

maximum_log_error = 0
scipy_underflow_fallbacks = 0
for case in data['tails']:
    n, p, k = case['n'], case['p']/100, case['k']
    # Integer polynomial summation is independent of both JS log-normalization and
    # scipy's extreme-tail approximations; retain scipy as a central-tail comparison.
    denominator=100**n
    numerator=sum(math.comb(n,s)*case['p']**s*(100-case['p'])**(n-s) for s in range(k,n+1))
    exact=Fraction(numerator,denominator)
    expected=(math.log(numerator)-math.log(denominator) if exact<=Fraction(1,2) else math.log1p(-float(1-exact))) if exact else -math.inf
    complement=(math.log(denominator-numerator)-math.log(denominator) if exact>=Fraction(1,2) else math.log1p(-float(exact))) if exact<1 else -math.inf
    close(case['exact'],expected)
    close(case['complement'],complement)
    scipy_log=float(binom.logsf(k-1,n,p))
    if expected != -math.inf and expected > -500:
        close(scipy_log,expected)
    elif scipy_log != expected:
        scipy_underflow_fallbacks += 1
    if math.isfinite(expected) and case['exact'] is not None:
        maximum_log_error = max(maximum_log_error, abs(case['exact']-expected))
    for label, bound in case['bounds'].items():
        if expected != -math.inf:
            assert bound is not None and bound+1e-9 >= expected, (case, label)
    if n <= 20:
        rational = Fraction(case['p'],100)
        exact = sum((Fraction(math.comb(n,s))*rational**s*(1-rational)**(n-s) for s in range(k,n+1)),Fraction(0))
        actual = native['exactBinomial']['binomial_tail'](n,rational,k)
        assert actual == exact

# Finite rational tail avoids scipy double underflow for the deliberately extreme upper tail.
small = data['small']
small_exact = native['exactBinomial']['binomial_tail'](2000,Fraction(1,100),1800)
small_log = math.log(small_exact.numerator)-math.log(small_exact.denominator)
close(small['exactLog'],small_log)
assert data['format']['tiny'].startswith('≈ 10^(')
assert data['format']['near'].startswith('1 − ') and data['format']['extreme'].startswith('1 − 10^(')
assert data['format']['zero']=='0' and data['format']['one']=='1'
close(data['near']['exactComplementLog'],200*math.log(.1))
close(data['extreme']['exactComplementLog'],2000*math.log(.01))

for row in data['budgets']:
    n,delta,width,variance = row['count'],row['delta'],row['width'],row['variance']
    logarithm=math.log(2/delta)
    close(row['hoeffding'],width*math.sqrt(logarithm/(2*n)))
    if variance:
        # Solve the tail equation as a scalar root, without using the displayed quadratic formula.
        target=lambda radius: n*radius*radius/(2*(variance+width*radius/3))-logarithm
        root=brentq(target,0,100*width,xtol=1e-14)
        close(row['bernsteinExact'],root)
    assert row['bernsteinExact'] <= row['bernsteinRelaxed']+1e-12
    needed=math.ceil(width*width*logarithm/(2*.1**2))
    assert row['needed']==needed
    assert 2*math.exp(-2*needed*.1**2/width**2) <= delta*(1+1e-12)
    assert 2*math.exp(-2*(needed-1)*.1**2/width**2) > delta

for row in data['samples']:
    n,e = row['count'],row['epsilonPercent']
    event=[k for k in range(n+1) if abs(100*k-25*n)>=e*n]
    assert row['eventCounts']==event
    laws=[binom.pmf(range(n+1),n,.25),np.array([.75]+[0]*(n-1)+[.25]),hypergeom.pmf(range(n+1),20,5,n)]
    for shown,law in zip(row['laws'],laws):
        exact=float(sum(law[k] for k in event))
        close(shown['logFailure'],math.log(exact) if exact else -math.inf)
        mean=float(np.dot(np.arange(n+1)/n,law))
        variance=float(np.dot((np.arange(n+1)/n-mean)**2,law))
        close(shown['meanVariance'],variance)
        assert abs(sum(math.exp(v) if v is not None else 0 for v in shown['logMass'])-1)<1e-12

for row in data['witnesses']:
    lam=row['lambda']
    law=[Fraction(math.comb(20,s))*Fraction(1,4)**s*Fraction(3,4)**(20-s) for s in range(21)]
    exact=sum(float(mass)*math.exp(lam*(s-10)) for s,mass in enumerate(law))
    close(row['logBound'],math.log(exact))
    close(sum(item['witness'] for item in row['contributions']),exact)
    assert all(item['witness']+1e-14>=item['tail'] for item in row['contributions'])
    assert row['logHoeffdingEnvelope']+1e-12>=row['logBound']>=row['exactLog']-1e-12
    close(row['optimalLambda'],math.log(3))

for row in data['families']:
    n,k,delta=row['count'],row['checks'],row['delta']
    for label,budget in [('naive',delta),('allocated',delta/k)]:
        cutoff=2*n*math.log(2/budget)
        law=[Fraction(math.comb(n,s),2**n) for s in range(n+1)]
        miss=sum((mass for s,mass in enumerate(law) if (2*s-n)**2>=cutoff),Fraction(0))
        actual=miss if row['dependence']=='identical' else 1-(1-miss)**k
        close(row[label+'FamilyLog'],math.log(float(actual)) if actual else -math.inf)
        assert float(actual) <= (min(1,k*delta) if label=='naive' else delta)+1e-12

# Independent proof stress checks: non-Bernoulli bounded laws and asymmetric weights.
rng=np.random.default_rng(180719)
lemma_checks=0
for _ in range(100):
    values=np.sort(rng.uniform(-4,7,5)); probabilities=rng.dirichlet(np.ones(5))
    centered=values-float(values@probabilities); width=values[-1]-values[0]
    variance=float((centered**2)@probabilities); cap=max(abs(centered))
    for lam in [-2,-.4,.01,.4,2]:
        mgf=math.log(float(np.exp(lam*centered)@probabilities))
        assert mgf <= lam*lam*width*width/8+1e-12
        if abs(lam)<3/cap:
            assert mgf <= lam*lam*variance/(2*(1-abs(lam)*cap/3))+1e-12
        lemma_checks+=1

practice={
    'radius400':math.sqrt(math.log(40)/800),
    'samples1060':math.ceil(math.log(200)/(2*.05**2)),
    'coin8':str(Fraction(37,256)),
    'unequal':2*math.exp(-32/14),
    'weighted':2*math.exp(-32/10.25),
    'bernstein200':brentq(lambda r:200*r*r/(2*(.04+r/3))-math.log(40),0,1),
    'relative':math.ceil(3*math.log(40)/(.25*.01)),
    'additive':math.ceil(math.log(40)/(2*.005**2)),
    'subsetVariance':.1875/10*10/19,
    'familyRadius':math.sqrt(math.log(1000)/1000),
    'timeBudget':str(sum((Fraction(1,20*t*(t+1)) for t in range(1,101)),Fraction(0))+Fraction(1,2020)),
    'invalidVarianceEvent':float(Fraction(39,40)**100),
}
assert practice['timeBudget']=='1/20'
assert practice['invalidVarianceEvent']>.05 and 2*math.log(40)/300<.025
body=(ROOT/'src/learn/data/topics/concentration-inequalities-hoeffding-bernstein-chernoff.jsx').read_text(encoding='utf-8')
for key in ['radius400','unequal','weighted','bernstein200','subsetVariance','familyRadius']:
    assert f"{practice[key]:.8f}" in body or f"{practice[key]:.8f}".lstrip('0') in body,(key,practice[key])
for number in ['1060','4427','73778','37/256']:
    assert number in body
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'nativePrograms':list(examples),'threeOriginalProgramsExactlyPreserved':True,'tailCases':len(data['tails']),'maximumAbsoluteLogTailError':maximum_log_error,'budgetRootCases':len(data['budgets']),'samplingCases':len(data['samples']),'witnessCases':len(data['witnesses']),'familyCases':len(data['families']),'independentBoundedMgfCases':lemma_checks,'invalidInputs':len(data['invalid']),'extremeFormats':data['format'],'changedPractice':practice,'sourceSha256':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in ['src/learn/data/concentration-inequalities-models.js','src/learn/data/concentration-inequalities-examples.js','src/learn/data/topics/concentration-inequalities-hoeffding-bernstein-chernoff.jsx']}}
(OUT/'native-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

