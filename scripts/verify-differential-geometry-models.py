"""Independent linear algebra, symbolic geometry, ODE and quadrature references."""
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/differential-geometry-verification'
subprocess.run(['node','scripts/verify-differential-geometry-models.mjs'],cwd=ROOT,check=True)
cases=json.loads((OUT/'model-fixtures.json').read_text())
comparisons=0
maximum=0.
def close(actual,expected,atol=3e-11,rtol=3e-11):
    global comparisons,maximum
    actual,expected=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    error=np.abs(actual-expected)
    assert np.all(error<=atol+rtol*np.abs(expected)),(actual,expected,error)
    maximum=max(maximum,float(np.max(error)))
    comparisons+=actual.size

for row in cases['charts']:
    for key in ['alpha','beta']:
        if row[key] is not None: close([math.cos(row[key]),math.sin(row[key])],row['point'])
    close(np.dot(row['point'],row['tangent']),0)
    if row['alpha'] is not None: assert -math.pi<row['alpha']<math.pi
    if row['beta'] is not None: assert 0<row['beta']<2*math.pi
for row in cases['metrics']:
    S=np.array([[1,row['shear']],[0,1.]])
    B=np.diag([1,row['verticalCost']**2])
    G=S.T@B@S
    differential=S.T@np.array([2.,-1])
    grad=np.linalg.solve(G,differential)
    close(row['metric'],G);close(row['gradient'],grad);close(row['covector'],differential)
    close(S@grad,row['worldGradient'])
    close(np.einsum('ni,ij,nj->n',row['coordinateEllipse'],G,row['coordinateEllipse']),1)
    close(np.asarray(row['coordinateEllipse'])@S.T,row['worldEllipse'])
    v=np.asarray(row['coordinateDirection'])
    close(v@G@v,1);close(grad@G@v,row['directionalDerivative'])
    assert abs(row['directionalDerivative'])<=row['gradientNorm']+1e-14

for row in cases['patches']:
    J=np.array([row['thetaBasis'],row['phiBasis']]).T
    close(J.T@J,row['metric']);close(J.T@row['point'],[0,0])
    close(np.linalg.norm(row['point']),row['radius'])
    close(np.linalg.norm(np.cross(J[:,0],J[:,1])),row['areaWeight'])

mp.mp.dps=80
for row in cases['maps']:
    x,v=np.array(row['point']),np.array(row['tangent'])
    generator=np.outer(v,x)-np.outer(x,v)
    expected=expm(generator)@x
    close(row['exponential'],expected)
    close(np.linalg.norm(row['exponential']),1)
    close(row['retracted'],(x+v)/np.linalg.norm(x+v))
    if row['logarithm'] is not None:
        close(row['logarithm'],v,atol=2e-10)
    if row['length'] <= 1e-4:
        m=mp.matrix([[mp.mpf(float(value)) for value in line] for line in generator])
        high=mp.expm(m)*mp.matrix([mp.mpf(float(value)) for value in x])
        close(row['exponential'],[float(value) for value in high],atol=5e-16,rtol=5e-16)

for row in cases['steps']:
    x=np.array(row['point']); ambient=np.array([1.,2,0])
    # Nullspace basis is formed independently from the circle parameterization.
    basis=np.array([-x[1],x[0],0.])
    grad=basis*np.dot(basis,ambient)
    close(row['tangentGradient'],grad)
    close(row['candidateNorm'],math.sqrt(1+row['learningRate']**2*np.dot(grad,grad)))
    close(row['retractionAngle'],math.atan(row['learningRate']*np.linalg.norm(grad)))

t,b=sp.symbols('t b',real=True,positive=True)
radial=sp.sqrt(t*t+b*b)
radial_second=sp.lambdify((t,b),sp.diff(radial,t,2),'numpy')
angular_second=sp.lambdify((t,b),sp.diff(sp.atan(b/t),t,2),'numpy')
for row in cases['polar']:
    close(row['radialAcceleration'],radial_second(row['time'],row['height']))
    if row['time']!=0: close(row['angularAcceleration'],angular_second(row['time'],row['height']))
    close(row['covariantAcceleration'],[0,0]);close(row['cartesianVelocity'],[1,0])
    close(row['radialVelocity']**2+row['r']**2*row['angularVelocity']**2,1)

transport_ode_cases=0
for row in cases['transport']:
    vertices=np.asarray(row['vertices'])/row['radius']
    initial=np.asarray(row['initial'])
    current=initial.copy()
    for index in range(3):
        start,end=vertices[index:index+2]
        axis=np.cross(start,end); sine=np.linalg.norm(axis); axis/=sine
        angle=math.atan2(sine,np.dot(start,end))
        velocity=angle*np.cross(axis,start)
        # Solve both the geodesic and the transport equation; do not use the
        # author's endpoint formula to construct the expected transported arrow.
        def evolution(_, state):
            position,speed,vector=np.split(state,3)
            return np.r_[speed,-np.dot(speed,speed)*position,-np.dot(vector,speed)*position]
        result=solve_ivp(evolution,[0,1],np.r_[start,velocity,current],rtol=2e-12,atol=2e-13,dense_output=True)
        assert result.success
        if index==min(2,int(row['progress'])):
            fraction=1 if row['progress']==3 else row['progress']-index
            state=result.sol(fraction)
            close(row['point'],state[:3]*row['radius'],atol=3e-10)
            close(row['currentVector'],state[6:],atol=3e-10)
        current=result.y[6:,-1]
    close(current,row['finalVector'],atol=3e-10)
    close(row['currentNorm'],1);close(row['tangencyResidual'],0)
    close(row['finalTurn'],row['expectedTurn'])
    close(row['enclosedArea']*row['curvature'],abs(row['expectedTurn']))
    transport_ode_cases+=1

u,v,R=sp.symbols('u v R',positive=True,real=True)
symbolic=[]
for kind,a in [('plane',u),('cylinder',R),('sphere',R*sp.sin(u/R)),('hyperbolic',R*sp.exp(u/R))]:
    metric=sp.diag(1,a*a);inverse=metric.inv();coordinates=[u,v]
    gamma=[[[sp.simplify(sum(inverse[k,l]*(sp.diff(metric[l,j],coordinates[i])+sp.diff(metric[l,i],coordinates[j])-sp.diff(metric[i,j],coordinates[l]))/2 for l in range(2))) for j in range(2)] for i in range(2)] for k in range(2)]
    # R^u_{vuv}, using the declared R(X,Y)Z ordering.
    curvature=sp.simplify((sp.diff(gamma[0][1][1],u)-sp.diff(gamma[0][0][1],v)+sum(gamma[l][1][1]*gamma[0][0][l]-gamma[l][0][1]*gamma[0][1][l] for l in range(2)))/(a*a))
    expected={'plane':0,'cylinder':0,'sphere':1/R**2,'hyperbolic':-1/R**2}[kind]
    assert sp.simplify(curvature-expected)==0
    symbolic.append({'kind':kind,'curvature':str(curvature)})
for case in cases['curvature']:
    for row in case['rows']:
        K=row['curvature']; distance=case['distance']
        solution=solve_ivp(lambda _,y:[y[1],-K*y[0]],[0,distance],[0.,1.],rtol=1e-12,atol=1e-13,dense_output=True)
        coefficient=solution.y[0,-1]
        close(row['circumference'],2*math.pi*coefficient)
        area=quad(lambda s:2*math.pi*solution.sol(s)[0],0,distance,epsabs=1e-11)[0]
        close(row['area'],area)
        close(row['riemannUVVU']/row['a']**2,K)

for row in cases['covariance']:
    A=np.diag(row['first']);B=np.diag(row['second'])
    root=np.diag(np.sqrt(row['first']));invroot=np.linalg.inv(root)
    eig,Q=np.linalg.eigh(invroot@B@invroot)
    path=root@(Q@np.diag(eig**row['fraction'])@Q.T)@root
    close(np.diag(path),row['affine'])
    close(np.linalg.norm(np.log(eig)),row['distance'])

summary=dict(checkedAt=datetime.now(timezone.utc).isoformat(),passed=True,
    comparisons=comparisons,maximumAbsoluteDiscrepancy=maximum,
    fixtureCounts={k:len(v) for k,v in cases.items()},transportODECases=transport_ode_cases,
    symbolicCurvature=symbolic,invalidInputs=21,
    versions={'numpy':np.__version__,'scipy':scipy.__version__,'sympy':sp.__version__,'mpmath':mp.__version__},
    modelSha256=hashlib.sha256((ROOT/'src/learn/data/differential-geometry-models.js').read_bytes()).hexdigest())
(OUT/'model-results.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
