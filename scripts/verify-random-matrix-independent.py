"""Bounded complementary review: MP endpoint layers, finite Schur geometry,
exact Gaussian moments and changed displayed helper inputs.
"""
import contextlib
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import warnings

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/random-matrix-independent-review'
OUT.mkdir(parents=True, exist_ok=True)

def node(code):
    return json.loads(subprocess.check_output(['node','--input-type=module','-e',code], cwd=ROOT, text=True, encoding='utf-8'))

def close(a,b,atol=2e-10,rtol=2e-9):
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)

examples = node("import{randomMatrixExamples as e}from'./src/learn/data/random-matrix-examples.js';console.log(JSON.stringify(e))")
functions = {}
for name, example in examples.items():
    namespace={}
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'],name+'.py','exec'),namespace)
    assert output.getvalue().strip()==example['expected'].strip(),name
    functions[name]=namespace

# Transform x=a+(b-a)sin²(theta), but integrate independently rather than using
# the browser's analytic antiderivative. Resolve the near-zero boundary layer.
def integrated_mp(gamma,variance,lower=None,upper=None,power=0):
    a=variance*(1-math.sqrt(gamma))**2
    b=variance*(1+math.sqrt(gamma))**2
    width=b-a
    lo=max(a, lower if lower is not None else a)
    hi=min(b, upper if upper is not None else b)
    if hi<=lo:return 0.0
    angle=lambda value: math.asin(math.sqrt(max(0,min(1,(value-a)/width))))
    start,end=angle(lo),angle(hi)
    layer=abs(1-math.sqrt(gamma))/(1+math.sqrt(gamma))
    points=sorted({start,end} | {v for v in [layer/100,layer/10,layer,layer*10,layer*100,0.01,0.1,0.5,1] if start<v<end})
    def integrand(t):
        s,c=math.sin(t),math.cos(t)
        x=a+width*s*s
        if gamma==1:
            base=4*c*c/math.pi
        else:
            base=width*width*s*s*c*c/(math.pi*gamma*variance*x)
        return base*x**power
    return sum(quad(integrand,l,r,epsabs=2e-13,epsrel=2e-13)[0] for l,r in zip(points,points[1:]))

cases=node("""
import * as m from './src/learn/data/random-matrix-models.js';
const spectra=[];
for(const [rows,columns]of [[7,3],[7,12],[11,5],[19,4]])for(const law of ['gaussian','sign'])for(const centered of [false,true])spectra.push(m.randomMatrixSpectrum({rows,columns,law,centered,seed:137}));
const mp=[];
for(const gamma of [.09,.36,.9999,.99999,.999999,1,1.000001,1.00001,1.0001,2.25,9])for(const variance of [.04,4,25]){
const law=m.marchenkoPastur(gamma,variance);for(const [left,right]of [[0,1],[.02,.4],[.4,.91]]){const lo=law.lower+left*(law.upper-law.lower),hi=law.lower+right*(law.upper-law.lower);mp.push({gamma,variance,lo,hi,mass:m.mpIntervalMass(gamma,lo,hi,variance)});}}
const schur=[];for(const seed of [23,71])for(const spike of [1.2,1.5,3,8])schur.push(m.randomMatrixSpectrum({rows:20,columns:5,seed,spike}));
const limits=[];for(const gamma of [.16,.36,.81])for(const population of [1.1,1+Math.sqrt(gamma),2+Math.sqrt(gamma),5])limits.push({gamma,population,...m.spikeLimits(gamma,population)});
const bounds=[];for(const n of [2,37,400])for(const p of [1,n])for(const delta of [.9,.05,1e-200])bounds.push({n,p,delta,...m.gaussianSpectrumBound(n,p,delta)});
const levels=[];for(const m0 of [-3,0,3])for(const d of [-2,0,1])for(const c of [-1.5,0,.7])levels.push({m:m0,d,c,...m.twoLevelSpectrum(m0,d,c)});
console.log(JSON.stringify({spectra,mp,schur,limits,bounds,levels}));
""")

for case in cases['spectra']:
    data=np.array(case['data']);active=data-data.mean(axis=0) if case['centered'] else data
    gram=active.T@active/case['denominator']
    close(gram,case['covariance'])
    singular=np.linalg.svd(active,compute_uv=False)
    expected=np.sort(np.r_[singular**2/case['denominator'],np.zeros(max(0,case['columns']-len(singular)))])
    close(case['values'],expected)
    close(sum(case['values']),np.sum(active*active)/case['denominator'])
    if case['centered']: close(active.sum(axis=0),0)

mp_errors=[]
for case in cases['mp']:
    oracle=integrated_mp(case['gamma'],case['variance'],case['lo'],case['hi'])
    close(case['mass'],oracle,atol=3e-12,rtol=1e-10)
    mp_errors.append(abs(case['mass']-oracle))

mass_helper_results=[]
for gamma in [.9999,.99999,.999999,1,1.000001,1.00001,1.0001,2.25]:
    with warnings.catch_warnings(record=True) as caught:
        atom,moments=functions['mass']['mp_moments'](gamma,4)
    target=[min(1,1/gamma),4,16*(1+gamma)]
    errors=np.abs(np.array(moments)-target)
    mass_helper_results.append({'gamma':gamma,'atom':atom,'moments':moments,'expected':target,'errors':errors.tolist(),'warnings':[str(w.message) for w in caught]})

# Gaussian quadrature integrates all degree-four entry polynomials exactly.
# This independently includes the sample-mean subtraction and unbiased scale.
nodes,weights=hermgauss(3);nodes=nodes*math.sqrt(2);weights=weights/math.sqrt(math.pi)
moment_rows=[]
for n,p in [(2,2),(3,2)]:
    raw=0.0;centered=0.0;average=np.zeros((p,p))
    for indices in itertools.product(range(3),repeat=n*p):
        probability=np.prod([weights[i] for i in indices])
        X=np.array([nodes[i] for i in indices]).reshape(n,p)
        S=X.T@X/n
        H=X-X.mean(axis=0)
        C=H.T@H/(n-1)
        average+=probability*S
        raw+=probability*np.trace(S@S)/p
        centered+=probability*np.trace(C@C)/p
    close(average,np.eye(p))
    close(raw,functions['moments']['exact_gaussian_second_moment'](n,p))
    close(centered,1+(p+1)/(n-1))
    moment_rows.append({'n':n,'p':p,'raw':raw,'centered':centered,'quadratureStates':3**(n*p)})

for case in cases['schur']:
    S=np.array(case['covariance']);a=S[0,0];b=S[1:,0];C=S[1:,1:]
    edge=np.linalg.eigvalsh(C)[-1]
    equation=lambda r:r-a-b@np.linalg.solve(r*np.eye(len(C))-C,b)
    root=brentq(equation,edge+1e-10,np.trace(S)+1,xtol=1e-13)
    w=np.linalg.solve(root*np.eye(len(C))-C,b)
    overlap=1/(1+w@w)
    close(root,case['largest']);close(overlap,case['alignment'])

limit_residuals=[]
for case in cases['limits']:
    gamma,population=case['gamma'],case['population']
    native=functions['spike']['limits'](gamma,population)
    close(native,[case['sample'],case['alignment']])
    if population>1+math.sqrt(gamma):
        a=(1-math.sqrt(gamma))**2;b=(1+math.sqrt(gamma))**2;r=case['sample']
        density=lambda x:math.sqrt(max(0,(b-x)*(x-a)))/(2*math.pi*gamma*x)
        first=quad(lambda x:x/(r-x)*density(x),a,b,epsabs=1e-12)[0]
        second=quad(lambda x:x/(r-x)**2*density(x),a,b,epsabs=1e-12)[0]
        residual=r-population*(1+gamma*first)
        close(residual,0,atol=2e-10)
        close(case['alignment'],1/(1+population*gamma*second))
        limit_residuals.append(residual)

for case in cases['bounds']:
    n,p,delta=case['n'],case['p'],case['delta']
    margin=math.sqrt(2*math.log(2/delta))
    target=[max(0,math.sqrt(n)-math.sqrt(p)-margin)**2/n,(math.sqrt(n)+math.sqrt(p)+margin)**2/n]
    close([case['lower'],case['upper']],target)
    close(functions['bound']['gaussian_covariance_bound'](n,p,delta),target)
for case in cases['levels']:
    matrix=np.array([[case['m']+case['d'],case['c']],[case['c'],case['m']-case['d']]])
    values=np.linalg.eigvalsh(matrix)
    close([case['lower'],case['upper']],values)
    close(functions['gap']['two_levels'](case['m'],case['d'],case['c']),values)
    close(case['gap'],np.ptp(values))
gap_integrals=[]
for gap in [.02,.2,.7,2,5]:
    # Radial integration of the independent d,c Gaussian joint density.
    radial=quad(lambda r:r*math.exp(-r*r/2),0,gap/2,epsabs=1e-14)[0]
    close(radial,-math.expm1(-gap*gap/8),atol=2e-14)
    gap_integrals.append(radial)

rank_rows=[]
for null_count in [2,3,4]:
    scores=[]
    for sample in itertools.product(range(3),repeat=null_count+1):
        scores.append(Fraction(1+sum(value>=sample[0] for value in sample[1:]),null_count+1))
    for numerator in range(1,null_count+2):
        alpha=Fraction(numerator,null_count+1)
        probability=Fraction(sum(score<=alpha for score in scores),len(scores))
        assert probability<=alpha
        rank_rows.append({'nullReplicates':null_count,'alpha':str(alpha),'exactRejectionProbability':str(probability)})
rank_result={'cases':351,'levels':rank_rows}
(OUT/'tied-rank-enumeration.json').write_text(json.dumps(rank_result,indent=2),encoding='utf-8')

archive=json.loads((ROOT/'docs/teaching/evidence/random-matrix-original-content.json').read_text(encoding='utf-8'))
assert examples['original']['code']==archive['program']
assert examples['original']['expected']==archive['output']
paths=['src/learn/data/topics/random-matrix-theory.jsx','src/learn/data/random-matrix-models.js','src/learn/data/random-matrix-examples.js','src/learn/components/lesson-labs/RandomMatrixLabs.jsx','src/learn/components/lesson-labs/random-matrix-labs.css','src/learn/data/curriculum/blueprints/random-matrix-theory.js']
result={'at':datetime.now(timezone.utc).isoformat(),'passedExceptTrackedMassHelper':True,'massHelperPassed':all(max(row['errors'])<2e-8 for row in mass_helper_results),
        'counts':{'displayedPrograms':len(examples),'changedCovariances':len(cases['spectra']),'independentMPIntervals':len(cases['mp']),'gaussianMomentQuadratureStates':sum(row['quadratureStates'] for row in moment_rows),'finiteSchurRootsAndOverlaps':len(cases['schur']),'spikeBranches':len(cases['limits']),'outlierIntegralIdentities':len(limit_residuals),'changedBounds':len(cases['bounds']),'changedTwoLevelMatrices':len(cases['levels']),'radialGapIntegrals':len(gap_integrals)},
        'maxMPIntervalError':max(mp_errors),'massHelper':mass_helper_results,'gaussianMoments':moment_rows,'tiedRankEnumeration':rank_result,
        'production':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in paths]}
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['massHelper','production']},indent=2))
if not result['massHelperPassed']:
    raise SystemExit('The tracked displayed MP mass-helper finding is still unresolved.')
