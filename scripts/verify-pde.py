"""Compare actual analytic models and saved native examples to independent finite oracles."""
from collections import Counter
import contextlib
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import mpmath as mp
import numpy as np
from scipy.integrate import quad
import sympy as sp

directory=Path('scratch/pde-verification')
data=json.loads((directory/'fixtures.json').read_text(encoding='utf-8'))
checks=Counter()
max_error=0.0

def close(actual,expected,name,tolerance=3e-11):
    global max_error
    actual=np.asarray(actual,dtype=float)
    expected=np.asarray(expected,dtype=float)
    error=float(np.max(np.abs(actual-expected))) if actual.size else 0
    scaled=error/max(1,float(np.max(np.abs(expected))) if expected.size else 0)
    assert np.all(np.isfinite(actual)) and scaled<tolerance,(name,actual,expected,scaled)
    max_error=max(max_error,scaled)
    checks[name]+=1

environments={}
for key,example in data['examples'].items():
    result=subprocess.run([sys.executable,'-c',example['code']],capture_output=True,text=True,encoding='utf-8',check=True)
    assert result.stdout.rstrip()==example['expected'].rstrip(),(key,result.stdout,example['expected'])
    namespace={'__name__':'verification_import'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'],f'displayed-{key}','exec'),namespace)
    environments[key]=namespace
checks['actual_complete_programs']=len(environments)

for row in data['volume']:
    t,a,b=row['time'],row['left'],row['right']
    integral=quad(lambda x:3-x,a,b)[0]
    sources=quad(lambda x:t-2-9*x,a,b)[0]
    close(row['accumulation'],integral,'control_volume_integral')
    close(row['sourceIntegral'],sources,'control_volume_source')
    close(row['balance'],integral,'control_volume_total')
    with mp.workdps(80):
        ma,mb,mt=map(mp.mpf,[a,b,t])
        exact=(3*(mb-ma)-(mb*mb-ma*ma)/2)
        close(row['accumulation']/float(exact),1,'control_volume_relative_thin_slice')
        close(row['balance']/float(exact),1,'control_volume_relative_balance')

for row in data['transport']:
    t,x,c=row['time'],row['x'],row['curvature']
    origin= max(0,t-x)
    expected=1+x-t if t<=x else 1-origin+c*origin**2
    close(row['value'],expected,'transport_trace')
    integral=quad(lambda z:environments['transport']['trace'](z,t,c)[2],0,1,points=[t] if 0<t<1 else [])[0]
    close(row['mass'],integral,'transport_integrated_mass')
    close(row['massDerivative'],row['incoming']-row['outgoing'],'transport_boundary_balance')

for row in data['modes']:
    # The independently executed native helper also computes endpoint derivatives.
    n,boundary=row['index'],row['boundary']
    values=[environments['modes']['mode'](boundary,n,p['x'])[0] for p in row['profile']]
    close([p['value'] for p in row['profile']],values,'mode_native_correspondence')
    lhs=quad(lambda x:environments['modes']['mode'](boundary,n,x)[1]**2,0,1)[0]
    rhs=row['eigenvalue']*quad(lambda x:environments['modes']['mode'](boundary,n,x)[0]**2,0,1)[0]
    close(lhs,rhs,'mode_energy_identity')

for row in data['heat']:
    t=row['theta']
    for x in [0,.125,.5,.875,1]:
        position=round(x*128)
        for boundary in ['dirichlet','neumann']:
            if t==0:
                expected=math.sin(math.pi*x)**2
            else:
                # Independent method of images: convolve reflected initial data.
                sign=-1 if boundary=='dirichlet' else 1
                def integrand(y):
                    kernels=math.fsum((math.exp(-(x-y+2*k)**2/(4*t))+sign*math.exp(-(x+y+2*k)**2/(4*t)))
                                      /math.sqrt(4*math.pi*t) for k in range(-6,7))
                    return kernels*math.sin(math.pi*y)**2
                expected=quad(integrand,0,1,epsabs=2e-13)[0]
            close(row['profile'][position][boundary],expected,'heat_images_oracle')
    for boundary in ['dirichlet','neumann']:
        f=lambda x:environments['heat']['heat'](x,t,boundary)
        close(row[boundary]['mean'],quad(f,0,1,epsabs=1e-12)[0],'heat_mean_quadrature')
        close(row[boundary]['squaredNorm'],quad(lambda x:f(x)**2,0,1,epsabs=1e-12)[0],'heat_norm_quadrature')
    if t:
        with mp.workdps(80):
            expected=mp.log(mp.mpf(36)/(5*mp.pi*63**2))-64**2*mp.pi**2*t
            close(row['logTailBound'],expected,'heat_tail_log_high_precision')
            for i in [1,17,63,96,127]:
                x=row['profile'][i]['x']
                ref=mp.fsum(-8/(mp.pi*n*(n*n-4))*mp.exp(-n*n*mp.pi**2*t)*mp.sin(n*mp.pi*x) for n in range(1,200,2))
                close(row['profile'][i]['dirichlet'],ref,'heat_arithmetic_vs_long_sum')

for row in data['kernel']:
    with mp.workdps(80):
        x,t,alpha=map(mp.mpf,[row['x'],row['time'],row['alpha']])
        expected=mp.exp(-x*x/(4*alpha*t))/mp.sqrt(4*mp.pi*alpha*t)
        close(row['value'],expected,'kernel_high_precision')
        if float(expected)>1e-300:
            close(row['value']/float(expected),1,'kernel_relative_arithmetic')
for row in data['periodic']:
    with mp.workdps(80):
        z,phase=mp.mpf(row['depth']),mp.mpf(row['phase'])
        expected=mp.re(mp.exp(1j*phase-(1+1j)*z))
        close(row['value'],expected,'periodic_complex_solution')
        close(row['envelope'],mp.exp(-z),'periodic_envelope')

for row in data['wave']:
    x,t,v=row['x'],row['time'],row['velocityFactor']
    lo,hi=x-t,x+t
    f=lambda z:(1-z*z)**3 if abs(z)<1 else 0
    velocity=v*quad(f,lo,hi,points=[p for p in [-1,1] if lo<p<hi],epsabs=1e-12)[0]/2
    expected=(f(lo)+f(hi))/2+velocity
    close(row['velocityContribution'],velocity,'wave_velocity_quadrature')
    close(row['total'],expected,'wave_formula_changed')
    close(environments['wave']['wave'](x,t,v)[2],expected,'actual_native_wave_changed')
for row in data['standing']:
    t=row['time']
    ut=lambda x:-math.pi*math.sin(math.pi*x)*math.sin(math.pi*t)+.5*math.pi*math.sin(2*math.pi*x)*math.cos(2*math.pi*t)
    ux=lambda x:math.pi*math.cos(math.pi*x)*math.cos(math.pi*t)+.5*math.pi*math.cos(2*math.pi*x)*math.sin(2*math.pi*t)
    close(row['kinetic'],quad(lambda x:ut(x)**2/2,0,1)[0],'wave_kinetic_integral')
    close(row['strain'],quad(lambda x:ux(x)**2/2,0,1)[0],'wave_strain_integral')
    close(row['total'],5*math.pi**2/16,'wave_energy_conservation')

for row in data['poisson']:
    a,b,left,right=row['a'],row['b'],row['left'],row['right']
    compatible=row['boundary']=='dirichlet' or Fraction(left+right)==Fraction(a)+Fraction(b,2)
    assert row['compatible']==compatible
    checks['poisson_compatibility']+=1
    if not compatible:
        assert row['profile']==[]
        continue
    # Fit a cubic using its PDE coefficients and actual boundary/mean constraints.
    matrix=[[0,0,-2,0],[0,0,0,-6]]
    target=[a,b]
    if row['boundary']=='dirichlet':
        matrix += [[1,0,0,0],[1,1,1,1]]
        target += [left,right]
    else:
        matrix += [[0,1,0,0],[1,.5,1/3,.25]]
        target += [left,row['mean']]
    coeff=np.linalg.solve(np.asarray(matrix,float),target)
    expected=np.polynomial.polynomial.polyval([p['x'] for p in row['profile']],coeff)
    close([p['value'] for p in row['profile']],expected,'poisson_independent_constraints')
    if row['boundary']=='neumann':
        native=environments['poisson']['neumann'](a,b,left,right,row['mean'])
        close([float(v) for v in native],coeff,'actual_native_neumann_changed')
for row in data['harmonic']:
    n=row['frequency']
    # Stable exponential representation is separate from the browser sinh ratio.
    values=[]
    for cell in row['cells']:
        x,y=cell['x'],cell['y']
        ratio=math.exp(n*math.pi*(y-1))*(-math.expm1(-2*n*math.pi*y))/(-math.expm1(-2*n*math.pi))
        values.append(math.sin(n*math.pi*x)*ratio)
    close([p['value'] for p in row['cells']],values,'harmonic_exponential_representation')
for row in data['weak']:
    a=row['location']
    for degree in range(1,6):
        derivative=lambda x:degree*x**(degree-1)-(degree+1)*x**degree
        pairing=quad(lambda x:row['leftSlope']*derivative(x),0,a)[0]+quad(lambda x:row['rightSlope']*derivative(x),a,1)[0]
        close(pairing,a**degree*(1-a),'weak_source_probe')
    close(row['energyIntegral'],row['leftSlope']**2*a+row['rightSlope']**2*(1-a),'weak_source_energy')

for row in data['burgers']:
    l,r,t=row['left'],row['right'],row['time']
    speed,balance,entropy=environments['burgers']['jump'](l,r)
    close(row['speed'],speed,'burgers_jump_speed')
    close(row['entropyProduction'],entropy,'burgers_entropy_fraction')
    assert balance==0
    if l<r and t>0 and not row['showExpansion']:
        for p in row['profile']:
            close(p['value'],environments['burgers']['rarefaction'](p['x'],t,l,r),'burgers_rarefaction')
for row in data['inverse']:
    with mp.workdps(80):
        ref=mp.exp(-row['index']**2*mp.pi**2*row['time'])
        close(row['attenuation']/float(ref),1,'inverse_heat_relative_amplitude')
        close(row['gain']*float(ref),1,'inverse_heat_gain')
for row in data['rod']:
    t,L,A=row['time'],row['length'],row['amplitude']
    close([p['value'] for p in row['profile']],[environments['rod']['rod'](p['x'],t,L,A) for p in row['profile']],'actual_native_rod_profile_changed')
    integral=quad(lambda x:environments['rod']['rod'](x,t,L,A),0,L)[0]
    close(row['mean'],integral/L,'rod_mean_integral')
    close(row['accumulation'],row['totalSource']-2*row['eachOutflow'],'rod_physical_balance')
    close(row['settlingTime'],environments['rod']['settle'](L,A,row['tolerance']),'actual_native_settling_changed')

# Changed written-practice answers, independently worked from equations and integrals.
x,t=sp.symbols('x t',real=True)
assert sp.expand(sp.diff(x*x+t,t)-sp.diff((2+x)*sp.diff(x*x+t,x),x))==-3-4*x
assert 3*4==2*6 and -2*6==-12
assert sp.Rational(3,10)<2*sp.Rational(4,10)
assert 2+sp.Rational(4,10)-sp.Rational(3,10)/2==sp.Rational(9,4)
assert 2-(sp.Rational(9,10)-2*sp.Rational(4,10))==sp.Rational(19,10)
assert sp.diff(sp.sin(sp.pi*x/4),x).subs(x,2)==0
assert sp.simplify(sp.integrate((3+2*sp.exp(-9*sp.pi**2*t)*sp.cos(3*sp.pi*x))**2,(x,0,1)))==9+2*sp.exp(-18*sp.pi**2*t)
assert sp.diff(sp.sin(sp.pi*x)**2,x,2).subs(x,0)==2*sp.pi**2
assert sp.integrate(sp.Integer(6),(x,-sp.Rational(1,2),sp.Rational(1,2)))/4==sp.Rational(3,2)
assert sp.integrate(1+x,(x,0,1))==sp.Rational(3,2)
a=sp.Rational(1,4)
assert (1-a)*sp.integrate(1-2*x,(x,0,a))-a*sp.integrate(1-2*x,(x,a,1))==sp.Rational(3,16)
assert environments['burgers']['jump'](3,-1)[2]==Fraction(-16,3)
assert environments['burgers']['jump'](-1,3)[2]==Fraction(16,3)
close(environments['rod']['settle'](2,.4,.025),16*math.log(16)/math.pi**2,'changed_practice_settling')
checks['changed_written_practice_identities']=12

checks['rejected_model_inputs']=data['invalidInputs']
paths=['src/learn/data/pde-models.js','src/learn/data/pde-examples.js']
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'checks':dict(checks),'maxScaledError':max_error,
        'numericSourceHashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        'python':sys.version,'numpy':np.__version__,'sympy':sp.__version__,
        'limits':'Finite independent numerical/model and actual-program checks. General PDE claims require the separate visible proofs; browser/integration evidence is separate.'}
(directory/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
