"""Independent finite moments, exact sums and high-precision cancellation checks."""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/ito-sde-verification'
fixtures = json.loads((OUT/'model-fixtures.json').read_text(encoding='utf-8'))
comparisons = 0
max_relative = 0.0
max_absolute = 0.0
max_relative_above_threshold = 0.0


def close(actual, expected, rtol=2e-9, atol=1e-28):
    global comparisons, max_relative, max_absolute, max_relative_above_threshold
    expected = float(expected)
    error = abs(actual-expected)
    max_absolute=max(max_absolute,error)
    assert error <= max(atol, abs(expected)*rtol), (actual, expected, error)
    if expected != 0:
        max_relative = max(max_relative, error/abs(expected))
        if abs(expected)>1e-9:
            max_relative_above_threshold=max(max_relative_above_threshold,error/abs(expected))
    comparisons += 1


for case in fixtures['errors']:
    p, out = case['input'], case['output']
    with localcontext() as ctx:
        ctx.prec = 100
        mu, sigma, t, x0 = [Decimal(str(p[key])) for key in ['mu','sigma','horizon','initial']]
        n = p['steps']
        h = t/n
        s = sigma*sigma*h
        correction = s*s/2 if p['method'] == 'milstein' else Decimal(0)
        mean_exact = x0*(mu*t).exp()
        mean_numeric = x0*(1+mu*h)**n
        exact_second = x0*x0*((2*mu+sigma*sigma)*t).exp()
        numeric_second = x0*x0*((1+mu*h)**2+s+correction)**n
        cross = x0*x0*(mu*t).exp()*(1+mu*h+s+correction)**n
        mse = exact_second+numeric_second-2*cross
        if sigma == 0:
            mse = (mean_numeric-mean_exact)**2
        assert mse >= 0
        close(out['mse'], mse, atol=1e-60)
        close(out['rms'], mse.sqrt(), atol=1e-50)
        close(out['meanBias'], mean_numeric-mean_exact, atol=1e-50)
        close(out['secondBias'], numeric_second-exact_second, atol=1e-45)
        close(out['exactSecond'], exact_second)
        close(out['numericalSecond'], numeric_second)

for case in fixtures['ou']:
    out = case['output']
    with localcontext() as ctx:
        ctx.prec = 90
        theta, h = Decimal(str(case['theta'])), Decimal(str(case['step']))
        if theta == 0:
            covariance, variance, residual = h, h, Decimal(0)
        else:
            covariance = (1-(-theta*h).exp())/theta
            variance = (1-(-2*theta*h).exp())/(2*theta)
            residual = variance-covariance*covariance/h
        close(out['covariance'], covariance)
        close(out['variance'], variance)
        close(out['residualVariance'], residual, rtol=2e-12, atol=1e-30)
        close(out['coefficient'], covariance/h)
        close(out['variance'], out['covariance']**2/float(h)+out['residualVariance'])

for case in fixtures['ouLaws']:
    p,o=case['input'],case['output']
    with localcontext() as ctx:
        ctx.prec=90
        theta,eta,v0,t,x0,target=map(lambda key:Decimal(str(p[key])),['theta','eta','initialVariance','time','initial','target'])
        factor=(-theta*t).exp()
        v=v0*factor*factor+(eta*eta*t if theta==0 else eta*eta*(1-(-2*theta*t).exp())/(2*theta))
        close(o['mean'],target+(x0-target)*factor,atol=1e-14)
        close(o['variance'],v,atol=1e-14)
        close(o['varianceRate'],eta*eta-2*theta*v,atol=1e-14)
        assert o['atom'] == (v == 0)

for case in fixtures['integrals']:
    values = list(map(Fraction,case['increments']))
    final = sum(values)
    square = sum(value*value for value in values)
    out = case['output']
    close(out['left'], (final*final-square)/2, atol=0, rtol=0)
    close(out['right'], (final*final+square)/2, atol=0, rtol=0)
    close(out['symmetric'], final*final/2, atol=0, rtol=0)
    close(out['quadratic'], square, atol=0, rtol=0)

for case in fixtures['grouped']:
    assert len(case['active'])*case['group'] == len(case['increments'])
    for index,value in enumerate(case['active']):
        close(value, math.fsum(case['increments'][index*case['group']:(index+1)*case['group']]),
              atol=3e-15)
    close(math.fsum(case['active']),math.fsum(case['increments']),atol=5e-15)

for case in fixtures['growth']:
    p,o = case['input'],case['output']
    mu,sigma,t,x0 = (p[key] for key in ['mu','sigma','horizon','initial'])
    close(o['mean'],x0*math.exp(mu*t))
    close(o['median'],x0*math.exp((mu-sigma*sigma/2)*t))
    close(o['variance'],x0*x0*math.exp(2*mu*t)*math.expm1(sigma*sigma*t))
    assert o['atom'] == (sigma*t == 0)
    if not o['atom']:
        for key,target in [('lower',.05),('upper',.95)]:
            z=(math.log(o[key]/x0)-(mu-sigma*sigma/2)*t)/(sigma*math.sqrt(t))
            close(ndtr(z),target,atol=2e-14)

for case in fixtures['paths']:
    p,o = case['input'],case['output']
    h=p['horizon']/len(p['increments'])
    for n,row in enumerate(o['rows']):
        dw=np.array(p['increments'][:n])
        em=p['initial']*np.prod(1+p['mu']*h+p['sigma']*dw)
        mil=p['initial']*np.prod(1+p['mu']*h+p['sigma']*dw+
                               p['sigma']**2/2*(dw*dw-h))
        exact=p['initial']*math.exp((p['mu']-p['sigma']**2/2)*n*h+p['sigma']*math.fsum(dw))
        close(row['euler'],em,atol=5e-14)
        close(row['milstein'],mil,atol=5e-14)
        close(row['exact'],exact,atol=5e-14)

# Compare OU recurrence against unrolled weighted sums, not the recurrence itself.
for case in fixtures['ouPaths']:
    theta=case['theta']; out=case['output']; h=out['step']
    rows=out['rows']; factor=math.exp(-theta*h)
    for n,row in enumerate(rows):
        weighted=math.fsum(factor**(n-j)*rows[j]['weightedNoise'] for j in range(1,n+1))
        exact=0.3+(-0.7-0.3)*factor**n+0.4*weighted
        emfactor=1-theta*h
        euler=0.3+(-0.7-0.3)*emfactor**n+0.4*math.fsum(emfactor**(n-j)*rows[j]['deltaW'] for j in range(1,n+1))
        close(row['exact'],exact,atol=3e-14)
        close(row['euler'],euler,atol=3e-14)
        if theta == 0:
            close(row['weightedNoise'],row['deltaW'],atol=0)

# Finite Gaussian quadrature directly tests an adapted two-interval isometry.
nodes,weights=np.polynomial.hermite.hermgauss(5)
nodes*=math.sqrt(2)
weights/=math.sqrt(math.pi)
isometries=0
for h1,h2,c in [(0.2,0.7,-0.3),(.125,.875,.7),(1.5,.25,1.2)]:
    mean=second=energy=0.0
    for i,z1 in enumerate(nodes):
        for j,z2 in enumerate(nodes):
            w=weights[i]*weights[j]
            d1,d2=math.sqrt(h1)*z1,math.sqrt(h2)*z2
            integral=c*d1+(c+d1)*d2
            mean+=w*integral
            second+=w*integral*integral
            energy+=w*(c*c*h1+(c+d1)**2*h2)
    close(mean,0,atol=2e-14)
    close(second,energy,atol=2e-14)
    isometries+=1

for case in fixtures['sampled']:
    output=case['output']
    # These are reproducible statistical smoke checks with wide declared limits,
    # not fixed sample outputs or proof that a stochastic error bar always covers.
    assert abs(output['bias']-output['analytic']['meanBias']) < 6*output['biasStandardError']+1e-10
    assert abs(output['mse']-output['analytic']['mse']) < 6*output['mseStandardError']+1e-10

record={'verifiedAt':datetime.now(timezone.utc).isoformat(),'passed':True,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'cases':{key:len(value) for key,value in fixtures.items() if isinstance(value,list)},
        'finiteAdaptedIsometries':isometries,'rejected':fixtures['rejected'],
        'comparisons':comparisons,'maximumRelativeDiscrepancy':max_relative,
        'maximumAbsoluteDiscrepancy':max_absolute,'maximumRelativeForReferencesAbove1eMinus9':max_relative_above_threshold,
        'sourceSha256':hashlib.sha256((ROOT/'src/learn/data/ito-sde-models.js').read_bytes()).hexdigest(),
        'limits':'Independent finite identities, positive variance decomposition, unrolled OU paths and 100-digit moments. Sample coverage is a bounded smoke test, not a guarantee.'}
(OUT/'model-results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
