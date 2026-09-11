from pathlib import Path
from fractions import Fraction as F
import contextlib
import datetime
import hashlib
import io
import json
import math
import sys
import numpy as np
import scipy
from scipy.integrate import quad

root = Path(__file__).resolve().parents[1]
folder = root / 'scratch/functional-analysis-verification'
payload = json.loads((folder/'inputs.json').read_text(encoding='utf-8'))
cases = payload['cases']
counts = {}
maximum_error = 0.

def near(actual, expected, atol=2e-10, rtol=2e-9):
    global maximum_error
    a,b = np.asarray(actual,float), np.asarray(expected,float)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    maximum_error = max(maximum_error,float(np.max(np.abs(a-b))) if a.size else 0.)
    assert np.allclose(a,b,atol=atol,rtol=rtol), (a,b)

namespaces = {}
for example in payload['examples']:
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'],example['id']+'.py','exec'),namespace)
    assert output.getvalue().rstrip('\n') == example['expected']
    namespaces[example['id']] = namespace
counts['actualDisplayedPrograms'] = len(payload['examples'])
original = json.loads((root/'docs/teaching/evidence/functional-analysis-rkhs-original-content.json').read_text(encoding='utf-8'))['programs'][0]
retained = next(e for e in payload['examples'] if e['id']=='original')
assert retained['code'] == original['code']
assert retained['expected'] == original['expectedOutput'].replace('\r\n','\n')
counts['exactOriginalProgramAndOutput'] = 1

for state in cases['spikes']:
    w,h = state['epsilon'], state['height']
    f = lambda x: h*max(1-abs(x-.5)/w,0)
    near(state['squaredSize'],quad(lambda x:f(x)**2,0,1,points=[.5-w,.5,.5+w])[0])
    slopes = [-h/w,h/w]
    near(state['slopeEnergy'],sum(value**2*w for value in slopes))
counts['spikeIntegrals'] = len(cases['spikes'])

for state in cases['integrals']:
    q = F(str(state['query']))
    slopes = [F(str(x)) for x in state['slopes']]
    exact = sum(slopes[i]*max(F(0),min(q,F(i+1,4))-F(i,4)) for i in range(4))
    energy = sum(x*x/4 for x in slopes)
    near(state['evaluation'],float(exact))
    near(state['squaredNorm'],float(energy))
    near(state['bound'],math.sqrt(float(q*energy)))
counts['exactSlopeEvaluations'] = len(cases['integrals'])

for state in cases['validity']:
    matrix = np.array(state['gram'])
    if state['kind']=='polynomial':
        xs = np.array([-1,0,1]); near(matrix,(1+np.outer(xs,xs))**2)
    elif state['kind']=='bigrams':
        words = ['ABA','BAB','ABAB']
        phi = np.array([[sum(word[j:j+2]==pattern for j in range(len(word)-1)) for pattern in ['AB','BA']] for word in words])
        near(matrix,phi@phi.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert bool(eigenvalues.min() >= -1e-12)==state['valid']
    c=np.array(state['coefficients']);near(state['quadratic'],c@matrix@c)
counts['gramConstructions'] = len(cases['validity'])

for state in cases['representers']:
    a=state['amplitude']
    nodes=np.array([0,.25,.5,1]);values=np.array([0,.5+a,1,0])
    near(state['values'],np.interp([0,.125,.25,.375,.5,.75,1],nodes,values))
    near(state['totalEnergy'],np.sum(np.diff(values)**2/np.diff(nodes)))
    near(state['extraResidual'],a if state['extraObservation'] else 0)
counts['changedWiggles'] = len(cases['representers'])

for state in cases['ridge']:
    xs,ys=np.array(state['xs']),np.array(state['ys'])
    k=np.exp(-state['gamma']*(xs[:,None]-xs[None,:])**2)
    alpha=np.linalg.solve(k+len(xs)*state['lambda']*np.eye(len(xs)),ys)
    query=np.array([-.25,0,.125,.5,.875,1,1.25])
    near(state['alpha'],alpha)
    near(state['values'],np.exp(-state['gamma']*(query[:,None]-xs[None,:])**2)@alpha)
    near(state['squaredNorm'],alpha@k@alpha,atol=2e-8)
    near(state['trainMse'],np.mean((k@alpha-ys)**2))
    assert state['squaredNorm']>=0
counts['independentRidgeSystems'] = len(cases['ridge'])

for state in cases['means']:
    x=np.array(state['support']); p,q=np.array(state['p']),np.array(state['q'])
    kernel=lambda a,b: math.exp(-state['gamma']*(a-b)**2)
    # Explicit independent product enumeration, not a matrix call to the model.
    pp=sum(p[i]*p[j]*kernel(a,b) for i,a in enumerate(x) for j,b in enumerate(x))
    qq=sum(q[i]*q[j]*kernel(a,b) for i,a in enumerate(x) for j,b in enumerate(x))
    pq=sum(p[i]*q[j]*kernel(a,b) for i,a in enumerate(x) for j,b in enumerate(x))
    near(state['squaredMmd'],pp+qq-2*pq)
    near(state['values'],[sum((p-q)[i]*kernel(a,t) for i,a in enumerate(x)) for t in [-3,-2,-.3,0,.4,1,3]])
    if state['sameLaw']: assert state['squaredMmd']==0 and all(v==0 for v in state['values'])
counts['meanProductEnumerations'] = len(cases['means'])

for state in cases['quadrature']:
    x,w=state['node'],state['weight']
    exact=quad(lambda t:(1-t-w*(t<x))**2,0,1,points=[x])[0]
    near(state['squaredError'],exact)
    assert state['linearError'] <= state['error']+1e-14
counts['residualIntegrals'] = len(cases['quadrature'])

interpolate=namespaces['interpolation']['anchored_interpolate']
native_ridge=namespaces['ridge']['fit_ridge']
native_quad=namespaces['quadrature']['quadrature_rule']
rng=np.random.default_rng(9901)
for n in range(1,8):
    for trial in range(8):
        nodes=np.sort(rng.choice(np.linspace(.05,1,20),n,replace=False))
        values=rng.normal(size=n)
        alpha,energy,gramenergy=interpolate(nodes,values)
        reconstructed=np.minimum.outer(nodes,nodes)@alpha
        near(reconstructed,values)
        derivative=lambda t: sum(alpha[i] for i in range(n) if t<nodes[i])
        integral=quad(lambda t:derivative(t)**2,0,1,points=nodes.tolist())[0]
        near(energy,integral);near(gramenergy,integral)
counts['actualNativeChangedInterpolants']=56
for nodes, values in [([1e-310],[1.]),([.5],[1e-200]),([.5,1.],[1e308,-1e308])]:
    with __import__('warnings').catch_warnings(record=True) as caught:
        try:
            interpolate(nodes,values)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported interpolation arithmetic must reject')
        assert not caught
for nodes,values in [([1e-308],[1.]),([.5],[0.]),([.5],[1e-150])]:
    alpha,energy,kernel_energy=interpolate(nodes,values)
    assert np.all(np.isfinite(alpha)) and np.isfinite(energy) and np.isfinite(kernel_energy)
    expected=float(values[0])**2/nodes[0]
    assert math.isclose(energy,expected,rel_tol=2e-14,abs_tol=0)
counts['actualNativeArithmeticBoundaries']=6
for n in [1,2,3,6,12]:
    for gamma in [.05,.3,3,32]:
        xs=rng.choice([-1.,0,.2,.7,1.],n)
        ys=rng.normal(size=n)
        k,alpha=native_ridge(xs,ys,gamma,.025)
        # Independent kernel eigensystem solution; tests duplicate/null directions.
        values,vectors=np.linalg.eigh(k)
        expected=vectors@((vectors.T@ys)/(values+n*.025))
        near(alpha,expected)
counts['actualNativeChangedRidge']=20
for n in range(1,8):
    for trial in range(5):
        nodes=rng.choice(np.linspace(0,1,9),n,replace=True)
        weights,energy=native_quad(nodes)
        # The representer's normal equations must hold, including duplicate anchors.
        near(np.minimum.outer(nodes,nodes)@weights,nodes-nodes**2/2)
        numerical=quad(lambda t:(1-t-sum(w for x,w in zip(nodes,weights) if t<x))**2,0,1,points=np.unique(nodes).tolist())[0]
        near(energy,numerical)
counts['actualNativeChangedQuadrature']=35

# Changed independent exercise arithmetic, including the exact constant-kernel solve.
k=np.ones((2,2));alpha=np.linalg.solve(k+.2*np.eye(2),[0,2])
near(alpha,[-50/11,60/11]);near(alpha@k@alpha,100/121)
assert F(1,3)-F(1,4)*F(7,8)**2==F(109,768)
near(math.sqrt(109/768),.376732,atol=6e-7)
assert sum(F(s)**2*width for s,width in [(4,F(1,4)),(-1,F(1,2))])==F(9,2)
counts['independentChangedPracticeChecks']=4
counts['explicitModelDomainRejections']=payload['invalidRejections']
result={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed',
        'counts':counts,'maximumAbsoluteDifference':maximum_error,
        'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__},
        'limits':'Finite oracles and actual native helpers; no browser or infinite-theorem claim.'}
(folder/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
