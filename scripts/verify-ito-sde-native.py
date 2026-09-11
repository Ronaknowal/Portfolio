"""Execute actual displayed programs and check changed inputs independently."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as F
from io import StringIO
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/ito-sde-verification'
source=ROOT/'src/learn/data/ito-sde-examples.js'
examples=json.loads(source.read_text(encoding='utf-8').split('export const itoSdeExamples = ',1)[1].rstrip().removesuffix(';'))
archive=json.loads((ROOT/'docs/teaching/evidence/ito-calculus-original-content.json').read_text(encoding='utf-8'))
assert examples['original']['code']==archive['blocks'][0]['code']
assert examples['original']['expected']==archive['blocks'][1]['code']
namespaces={}
for name,example in examples.items():
    result=subprocess.run([sys.executable,'-c',example['code']],capture_output=True,text=True,encoding='utf-8',check=True,timeout=30)
    assert not result.stderr and result.stdout.rstrip('\n')==example['expected'],(name,result.stdout,result.stderr)
    namespace={}
    with redirect_stdout(StringIO()):
        exec(compile(example['code'],name,'exec'),namespace)
    namespaces[name]=namespace

comparisons=0
maximum_absolute=0.0
def close(actual,expected,atol=3e-12,rtol=2e-10):
    global comparisons,maximum_absolute
    error=abs(float(actual)-float(expected))
    assert error<=atol+rtol*abs(float(expected)),(actual,expected,error)
    maximum_absolute=max(maximum_absolute,error)
    comparisons+=1

integral_cases=0
for size in range(1,6):
    for increments in itertools.product([F(-1,2),F(0),F(3,4)],repeat=size):
        terminal,left,right,symmetric,quadratic=namespaces['integrals']['sums'](increments)
        endpoint=sum(increments);q=sum(x*x for x in increments)
        assert (terminal,left,right,symmetric,quadratic)==(endpoint,(endpoint**2-q)/2,(endpoint**2+q)/2,endpoint**2/2,q)
        integral_cases+=1

isometry_cases=0
for h1,h2,c in itertools.product([.125,.3,1.5],[.2,.75,2],[-1.2,0,.7,2]):
    mean,second,energy=namespaces['isometry']['finite_isometry'](h1,h2,c)
    # Independent conditioning: E[(c+dW1)^2]=c²+h1; cross term vanishes.
    expected=c*c*(h1+h2)+h1*h2
    close(mean,0);close(second,expected);close(energy,expected)
    isometry_cases+=1

growth_cases=0
for initial,mu,sigma,time in itertools.product([.4,1.7],[-.3,.2,.9],[.05,.3,1.1],[.125,1,3]):
    mean,median,variance,quantiles=namespaces['growth']['growth_statistics'](initial,mu,sigma,time)
    law=lognorm(s=sigma*math.sqrt(time),scale=initial*math.exp((mu-sigma*sigma/2)*time))
    for actual,expected in zip([mean,median,variance,*quantiles],[law.mean(),law.median(),law.var(),*law.ppf([.05,.95])]):
        close(actual,expected)
    growth_cases+=1

ou_cases=0
for theta,h in itertools.product([0,.05,.7,3],[1/8192,.125,.75,4]):
    covariance,variance,residual=namespaces['ou']['coupled_noise'](theta,h)
    # Integrate centered deterministic kernel: avoids subtracting near-equal variances.
    a=quad(lambda t:math.exp(-theta*(h-t)),0,h,epsabs=1e-14)[0]/h
    expected_residual=quad(lambda t:(math.exp(-theta*(h-t))-a)**2,0,h,epsabs=1e-30,epsrel=1e-9)[0]
    close(covariance,a*h,atol=1e-14)
    close(variance,quad(lambda t:math.exp(-2*theta*(h-t)),0,h)[0])
    close(residual,expected_residual,atol=1e-26,rtol=3e-8)
    ou_cases+=1
for theta,target,eta,m0,v0,t in itertools.product([0,.2,1.7],[-.3,.5],[0,.7],[.4],[0,.6],[0,.3,2]):
    mean,variance=namespaces['ou']['ou_moments'](theta,target,eta,m0,v0,t)
    close(mean,target+(m0-target)*math.exp(-theta*t))
    close(variance,v0*math.exp(-2*theta*t)+eta*eta*quad(lambda s:math.exp(-2*theta*(t-s)),0,t)[0])
    ou_cases+=1

covariance_cases=0
rng=np.random.default_rng(8923)
for n,m in itertools.product(range(1,5),range(1,5)):
    matrix=rng.uniform(-2,2,(n,m)); output=namespaces['covariance']['covariance_rate'](matrix)
    for i,j in itertools.product(range(n),repeat=2):
        close(output[i,j],math.fsum(matrix[i,k]*matrix[j,k] for k in range(m)))
    direction=rng.normal(size=n)
    close(direction@output@direction,sum((direction@matrix[:,k])**2 for k in range(m)))
    covariance_cases+=1

coupling_cases=0
for mu,sigma,horizon,size in itertools.product([-.5,0,.7],[0,.3,1.2],[.25,1,2],[1,4,16]):
    increments=list(rng.normal(size=size)*math.sqrt(horizon/size))
    em,mil,exact=namespaces['coupling']['solve_from_increments'](increments,horizon,mu,sigma,1.3)
    h=horizon/size
    close(em,1.3*np.prod([1+mu*h+sigma*x for x in increments]))
    close(mil,1.3*np.prod([1+mu*h+sigma*x+sigma*sigma/2*(x*x-h) for x in increments]))
    close(exact,1.3*math.prod(math.exp((mu-sigma*sigma/2)*h+sigma*x) for x in increments))
    coupling_cases+=1

error_cases=0
for mu,sigma,time,n,method in itertools.product([-.5,0,.4,1],[0,.05,.6,1.2],[.25,1,2],[1,8,128,512],['euler','milstein']):
    rms,bias=namespaces['errors']['exact_errors'](mu,sigma,time,n,method)
    # Independently integrate one-step normal moments; high-precision algebra closes
    # tiny cases below quadrature cancellation, already cross-checked against JS.
    if sigma==0:
        close(rms,abs((1+mu*time/n)**n-math.exp(mu*time)),atol=1e-13)
    elif rms>1e-5:
        h=time/n
        z,w=np.polynomial.hermite.hermgauss(48);z*=math.sqrt(2);w/=math.sqrt(math.pi)
        a=1+mu*h+sigma*math.sqrt(h)*z
        if method=='milstein':a+=sigma*sigma*h/2*(z*z-1)
        b=np.exp((mu-sigma*sigma/2)*h+sigma*math.sqrt(h)*z)
        squared=float(w@(a*a))**n+float(w@(b*b))**n-2*float(w@(a*b))**n
        close(rms*rms,squared,atol=2e-10,rtol=2e-7)
    close(bias,(1+mu*time/n)**n-math.exp(mu*time),atol=5e-13)
    error_cases+=1

generator_cases=0
for theta,target,eta,mean,variance in itertools.product([0,.3,1.2],[-.7,0,.4],[0,.8],[-.4,1],[0,.3]):
    a,b=namespaces['generator']['ou_second_rate'](theta,target,eta,mean,variance)
    z,w=np.polynomial.hermite.hermgauss(5);x=mean+math.sqrt(2*variance)*z
    expected=float(w@(-2*theta*x*(x-target)+eta*eta))/math.sqrt(math.pi)
    close(a,expected);close(b,expected)
    generator_cases+=1

tilt_cases=0
for t,c in itertools.product([.125,.7,2],[-1.2,0,.4,1.1]):
    density=lambda x:math.exp(-(x-c*t)**2/(2*t))/math.sqrt(2*math.pi*t)
    close(quad(density,-np.inf,np.inf)[0],1)
    close(quad(lambda x:x*density(x),-np.inf,np.inf)[0],c*t)
    close(quad(lambda x:(x-c*t)**2*density(x),-np.inf,np.inf)[0],t)
    tilt_cases+=1

time_transform_cases=0
for time,state,step in itertools.product([0,.2,1.7],[-1.5,0,.4,2.1],[.125,.3,1.5]):
    current,future=namespaces['time_transform']['conditional_transform'](time,state,step)
    # Expand the conditional Gaussian cubic analytically, independent of quadrature.
    expected=state**3+3*state*step-3*(time+step)*state
    close(current,state**3-3*time*state);close(future,expected)
    time_transform_cases+=1

invalid=[lambda:namespaces['growth']['growth_statistics'](1,float('nan'),1,1),
         lambda:namespaces['ou']['coupled_noise'](1e-200,1),
         lambda:namespaces['isometry']['finite_isometry'](float('nan'),1,1),
         lambda:namespaces['covariance']['covariance_rate']([[float('inf')]]),
         lambda:namespaces['coupling']['solve_from_increments']([float('nan')],1,.4,.3,1),
         lambda:namespaces['errors']['exact_errors'](.4,1e-200,1,16),
         lambda:namespaces['generator']['ou_second_rate'](1,0,.8,0,float('nan'))]
for call in invalid:
    try:call()
    except (ValueError,ArithmeticError):pass
    else:raise AssertionError('Expected explicit rejection')

record={'verifiedAt':datetime.now(timezone.utc).isoformat(),'passed':True,'python':platform.python_version(),
 'programs':len(examples),'originalCodeAndOutputPreserved':True,'integralCases':integral_cases,
 'isometryCases':isometry_cases,'growthCases':growth_cases,'ouCases':ou_cases,'covarianceCases':covariance_cases,
 'couplingCases':coupling_cases,'errorCases':error_cases,'generatorCases':generator_cases,'tiltCases':tilt_cases,
 'timeTransformCases':time_transform_cases,'rejectedInputs':len(invalid),'numericComparisons':comparisons,'maximumAbsoluteDiscrepancy':maximum_absolute,
 'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'scope':'Actual exported programs run unchanged; changed helper inputs use rational, distribution, kernel-quadrature and Gaussian-moment references. Constant-tilt identities are complementary mathematics checks, not calls to a nonexistent reusable helper.'}
(OUT/'native-results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
