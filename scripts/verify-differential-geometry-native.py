"""Complementary checks call the exact helper functions shown to the learner."""
import contextlib
import hashlib
import io
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm, logm
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/differential-geometry-verification'
OUT.mkdir(exist_ok=True,parents=True)
source=ROOT/'src/learn/data/differential-geometry-examples.js'
examples=json.loads(source.read_text(encoding='utf8').split('export const differentialGeometryExamples = ',1)[1].rstrip().removesuffix(';'))
archive=json.loads((ROOT/'docs/teaching/evidence/differential-geometry-original-content.json').read_text())
assert examples['original']['code']==archive['blocks'][0]['text']
assert examples['original']['expected']==archive['blocks'][1]['text']
ns={}
for key,example in examples.items():
    result=subprocess.run([sys.executable,'-I','-c',example['code']],capture_output=True,text=True,encoding='utf8',check=True,timeout=30)
    assert result.stdout.rstrip()==example['expected'],(key,result.stdout)
    assert not result.stderr
    namespace={}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'],key,'exec'),namespace)
    ns[key]=namespace

count=0
maximum=0.
def close(actual,expected,atol=1e-10,rtol=1e-10):
    global count,maximum
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    err=np.abs(a-b)
    assert np.all(err<=atol+rtol*np.abs(b)),(a,b,err)
    maximum=max(maximum,float(err.max()))
    count+=a.size

for angle in range(361):
    alpha,beta=ns['charts']['labels'](angle)
    for value in [alpha,beta]:
        if value is not None:
            close([math.cos(math.radians(value)),math.sin(math.radians(value))],
                  [math.cos(math.radians(angle)),math.sin(math.radians(angle))])
for shear,cost in itertools.product([-2.,-.3,0.,.7,2.5],[.4,.8,1.3,2.1]):
    G,a,gradient,world=ns['metric']['metric_gradient'](shear,cost)
    # Solve the steepest direction by constrained optimization on the unit ellipse.
    objective=lambda angle: -(2*math.cos(angle)-math.sin(angle)/cost)
    optimum=minimize_scalar(objective,bounds=(-math.pi,0),method='bounded')
    close(-optimum.fun,math.sqrt(4+1/cost**2),atol=1e-9)
    close(world,[2,-1/cost**2])
    for v in [[1,2],[-.7,.3],[0.,1.]]:
        close(gradient@G@v,a@v)
for theta,phi,R in itertools.product([.2,.9,1.7,2.8],[-2.3,0.,1.4],[.6,1.7,3.]):
    G,area=ns['area']['patch'](theta,phi,R)
    # Independent metric obtained from finite coordinate differences.
    F=lambda t,p: R*np.array([math.sin(t)*math.cos(p),math.sin(t)*math.sin(p),math.cos(t)])
    h=1e-5
    J=np.column_stack([(F(theta+h,phi)-F(theta-h,phi))/(2*h),
                       (F(theta,phi+h)-F(theta,phi-h))/(2*h)])
    close(G,J.T@J,atol=2e-9)
    close(area,R**2*math.sin(theta))
for R,angle,T in itertools.product([.7,2.3],[.01,.7,2.8,math.pi],[.4,1.7]):
    short,long,chord,energy=ns['paths']['route_values'](R,angle,T)
    close(short,quad(lambda _:R*angle/T,0,T)[0])
    close(energy,quad(lambda _: .5*(R*angle/T)**2,0,T)[0])
    close(chord,math.dist([R,0],[R*math.cos(angle),R*math.sin(angle)]))
    close(short+long,2*math.pi*R)

rng=np.random.default_rng(4297)
map_cases=0
for _ in range(24):
    x=rng.normal(size=3);x/=np.linalg.norm(x)
    v=rng.normal(size=3);v-=x*(x@v);v/=np.linalg.norm(v)
    for length in [1e-8,.03,.7,2.1]:
        tangent=v*length
        endpoint=ns['maps']['sphere_exp'](x,tangent)
        close(endpoint,expm(np.outer(tangent,x)-np.outer(x,tangent))@x)
        close(ns['maps']['sphere_log'](x,endpoint),tangent)
        map_cases+=1
for time,height in itertools.product([-1.8,-.25,0.,.65,1.7],[.35,1.3,2.4]):
    radial,angular=ns['connection']['polar_terms'](time,height)
    h=2e-4
    r=lambda t:math.hypot(t,height)
    theta=lambda t:math.atan2(height,t)
    def richardson_second(function):
        def central(step):
            return (function(time+step)-2*function(time)+function(time-step))/step**2
        return (4*central(h/2)-central(h))/3
    close(radial[0],richardson_second(r),atol=2e-7)
    close(angular[0],richardson_second(theta),atol=2e-7)
    close(sum(radial),0);close(sum(angular),0)

transport_cases=0
for _ in range(20):
    x=rng.normal(size=3);x/=np.linalg.norm(x)
    v=rng.normal(size=3);v-=x*(x@v)
    direction=rng.normal(size=3);direction-=x*(x@direction);direction/=np.linalg.norm(direction)
    angle=rng.uniform(.05,2.5)
    y=expm(angle*(np.outer(direction,x)-np.outer(x,direction)))@x
    def derivative(_,state):
        point,speed,arrow=np.split(state,3)
        return np.r_[speed,-(speed@speed)*point,-(arrow@speed)*point]
    reference=solve_ivp(derivative,[0,1],np.r_[x,angle*direction,v],rtol=1e-12,atol=1e-13)
    result=ns['transport']['transport'](x,y,v)
    close(result,reference.y[6:,-1])
    close(ns['transport']['transport'](y,x,result),v)
    close(result@y,0);close(result@result,v@v)
    transport_cases+=1

u=ns['curvature']['u']
for a,expected in [(1+u*u,-2/(1+u*u)),(sp.exp(2*u),-4),(sp.Integer(3),0),(sp.sqrt(u),1/(4*u*u))]:
    _,K=ns['curvature']['curvature_from_metric'](a)
    assert sp.simplify(K-expected)==0
loss=ns['means']['angular_loss']
for values in [[-.2,.1,.5],[.1,.4,.9],[.7,.8,.8]]:
    reference=minimize_scalar(lambda m:loss(m,values),bounds=(-1,2),method='bounded')
    close(reference.x,np.mean(values),atol=1e-7)
for p,q in itertools.product([.001,.08,.25,.6,.95],[.004,.3,.75,.99]):
    distance=abs(ns['fisher']['fisher_coordinate'](p)-ns['fisher']['fisher_coordinate'](q))
    integrated=quad(lambda value:1/math.sqrt(value*(1-value)),min(p,q),max(p,q),epsabs=1e-11)[0]
    close(distance,integrated)

spd_cases=0
for size in [2,3,4]:
    for _ in range(8):
        raw=rng.normal(size=(size,size));A=raw@raw.T+np.eye(size)*.7
        raw=rng.normal(size=(size,size));B=raw@raw.T+np.eye(size)*.7
        root=expm(.5*logm(A));inverse=expm(-.5*logm(A))
        relative=inverse@B@inverse
        for fraction in [.15,.5,.85]:
            expected=root@expm(fraction*logm(relative))@root
            close(ns['spd']['affine_path'](A,B,fraction),expected)
            spd_cases+=1
        distance=ns['spd']['affine_distance'](A,B)
        close(distance,np.linalg.norm(logm(relative)))
        change=np.eye(size)+.12*rng.normal(size=(size,size))
        close(ns['spd']['affine_distance'](change@A@change.T,change@B@change.T),distance)

theta=ns['laplacian']['theta'];phi=ns['laplacian']['phi'];R=ns['laplacian']['R']
for function,expected in [(sp.sin(theta)*sp.cos(phi),-2*sp.sin(theta)*sp.cos(phi)/R**2),
                          ((3*sp.cos(theta)**2-1)/2,-3*(3*sp.cos(theta)**2-1)/R**2)]:
    assert sp.simplify(ns['laplacian']['laplacian'](function)-expected)==0

optimization_cases=0
statuses={}
gradient_checks=0
for size in [2,3,5]:
    for _ in range(10):
        Q,_=np.linalg.qr(rng.normal(size=(size,size)))
        eig=np.linspace(.8,4.7,size)
        A=Q@np.diag(eig)@Q.T
        start=Q@np.ones(size)
        x,status,history=ns['optimize']['minimize_rayleigh'](A,start,tolerance=1e-6)
        statuses[status]=statuses.get(status,0)+1
        close(x@A@x,eig[0],atol=1e-8)
        close(x@x,1)
        assert all(b[0]<=a[0]+1e-12 for a,b in zip(history,history[1:]))
        assert np.linalg.norm(2*(A@x-(x@A@x)*x))<2e-6
        # Directional central differences use an independent circle rotation.
        y=start/np.linalg.norm(start);v=rng.normal(size=size);v-=y*(y@v);v/=np.linalg.norm(v)
        exact=2*(A@y-(y@A@y)*y)@v
        errors=[]
        for h in [.01,.005,.0025]:
            plus=math.cos(h)*y+math.sin(h)*v
            minus=math.cos(h)*y-math.sin(h)*v
            approximate=(plus@A@plus-minus@A@minus)/(2*h)
            errors.append(abs(approximate-exact));gradient_checks+=1
        assert errors[-1]<=errors[0]/12+1e-12
        optimization_cases+=1
changed=ns['optimize']['changed']
for start,tol,wanted in [([0,0,1],1e-8,9),([1,2,3],1e-6,2)]:
    x,status,_=ns['optimize']['minimize_rayleigh'](changed,start,tolerance=tol)
    assert status=='stationary';close(x@changed@x,wanted,atol=1e-8)

# Check the actual eleven changed questions, not only nearby random fixtures.
practice_answers=[]
assert ns['charts']['labels'](315)==(-45,315)
close(np.dot([math.sqrt(.5),-math.sqrt(.5)],[math.sqrt(.5),math.sqrt(.5)]),0)
practice_answers.append('315-degree chart and tangent')
x=np.array([0.,3.,0.]);a=np.array([2.,5.,-1.]);v=np.array([1.,0.,2.])
close(a-(a@x)/(x@x)*x,[2,0,-1]);close(x@v,0)
practice_answers.append('radius-three tangent projection')
G,a,g,world=ns['metric']['metric_gradient'](2.,2.)
close(G,[[1,2],[2,8]]);close(a,[2,3]);close(g,[2.5,-.25]);close(world,[2,-.25])
practice_answers.append('changed shear and metric')
band=quad(lambda theta:math.sin(theta)/2,math.pi/6,math.pi/3)[0]
close(band,(math.sqrt(3)-1)/4);close(2*math.pi*2*math.sin(math.pi/6),2*math.pi)
practice_answers.append('latitude and band area')
endpoint=ns['maps']['sphere_exp']([1,0,0],[0,3*math.pi/2,0])
close(endpoint,[0,-1,0]);close(ns['maps']['sphere_log']([1,0,0],endpoint),[0,-math.pi/2,0])
practice_answers.append('long Exp versus shortest Log')
radial,angular=ns['connection']['polar_terms'](0.,2.)
close(radial,[.5,-.5]);close(angular,[0,0])
practice_answers.append('changed polar cancellation')
N=np.array([0.,0,1]);A=np.array([1.,0,0]);B=np.array([.5,math.sqrt(3)/2,0.])
for route,sign in [([N,A,B,N],1),([N,B,A,N],-1)]:
    arrow=np.array([1.,0,0])
    for first,second in zip(route,route[1:]):arrow=ns['transport']['transport'](first,second,arrow)
    close(math.atan2(arrow[1],arrow[0]),sign*math.pi/3)
practice_answers.append('radius-two 60-degree loop and reverse')
_,K=ns['curvature']['curvature_from_metric'](1+u*u)
assert K.subs(u,0)==-2 and 2*K.subs(u,0)==-4
practice_answers.append('new warped curvature')
close(ns['spd']['affine_path'](np.diag([1.,9.]),np.diag([4.,1.]),.5),np.diag([2.,3.]))
practice_answers.append('different covariance midpoint')
close(ns['fisher']['fisher_coordinate'](.25),math.pi/3)
practice_answers.append('finite Fisher boundary')
practice_answers.append('changed maximum and mixed optimizer starts checked above')

invalid=[
    lambda:ns['charts']['labels'](-1),
    lambda:ns['metric']['metric_gradient'](1,0),
    lambda:ns['maps']['sphere_exp']([1,0,0],[1,0,0]),
    lambda:ns['maps']['sphere_exp']([2,0,0],[0,1,0]),
    lambda:ns['maps']['sphere_log']([1,0,0],[-1,0,0]),
    lambda:ns['transport']['transport']([1,0,0],[-1,0,0],[0,1,0]),
    lambda:ns['spd']['affine_path'](np.eye(2),np.diag([-1,1]),.5),
    lambda:ns['spd']['matrix_power_spd']([[1,2],[0,1]],.5),
    lambda:ns['fisher']['fisher_coordinate'](0),
    lambda:ns['fisher']['fisher_coordinate'](1),
    lambda:ns['optimize']['minimize_rayleigh'](np.eye(2),[0,0]),
    lambda:ns['optimize']['minimize_rayleigh'](np.eye(2),[1,0],tolerance=float('nan')),
]
for call in invalid:
    try:call()
    except ValueError:pass
    else:raise AssertionError('Invalid input accepted')
result=dict(checkedAt=datetime.now(timezone.utc).isoformat(),passed=True,programs=len(examples),
            preservedOriginal=True,comparisons=count,maximumAbsoluteDiscrepancy=maximum,
            sphereMapCases=map_cases,transportODECases=transport_cases,noncommutingSPDPaths=spd_cases,
            optimizationCases=optimization_cases,optimizationStatuses=statuses,gradientStepChecks=gradient_checks,
            symbolicCurvatureVariants=4,changedLaplacianEigenfunctions=2,changedPractice=len(practice_answers),practiceAnswers=practice_answers,
            rejected=len(invalid),python=sys.version,examplesSha256=hashlib.sha256(source.read_bytes()).hexdigest())
(OUT/'native-results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
