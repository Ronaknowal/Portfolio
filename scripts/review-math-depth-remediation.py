"""Integration-owner complementary numerical checks for math depth additions."""
import hashlib
import importlib.util
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scratch/math-depth-remediation/extra-dependencies'))
import numpy as np
from scipy.optimize import brentq


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derivative():
    m = load('public/learn-assets/matrix-calculus/derivative-library-bridge.py')
    rng = np.random.default_rng(3)
    for n, d, q in [(2, 3, 1), (3, 2, 4)]:
        arrays = [rng.normal(size=shape) for shape in [(n,d), (d,q), (q,), (n,q)]]
        result = m.affine_loss_pullback(*arrays)
        for j in range(3):
            numerical = np.empty_like(arrays[j])
            for index in np.ndindex(arrays[j].shape):
                plus, minus = [a.copy() for a in arrays], [a.copy() for a in arrays]
                plus[j][index] += 1e-6; minus[j][index] -= 1e-6
                numerical[index] = (m.affine_loss_pullback(*plus)[0] - m.affine_loss_pullback(*minus)[0]) / 2e-6
            np.testing.assert_allclose(result[j+1], numerical, atol=3e-8)
    return 'all affine coordinates by finite differences on two non-square changed shapes'


def simplex():
    m = load('public/learn-assets/constrained-optimization/simplex-solver-bridge.py')
    rng = np.random.default_rng(43)
    for n in [2, 5, 13]:
        center, curvature = rng.normal(size=n), rng.uniform(.5, 2, size=n)
        budget = 1.7
        threshold = brentq(lambda value: np.maximum(center-value/curvature, 0).sum()-budget,
                           np.min(curvature*center)-budget*curvature.max(), np.max(curvature*center))
        exact = np.maximum(center-threshold/curvature, 0)
        answer, status = m.solve_projected(center, curvature, budget)
        assert status['converged']
        np.testing.assert_allclose(answer, exact, atol=1e-9)
        np.testing.assert_allclose(m.project_simplex(center+1024, budget), m.project_simplex(center, budget), atol=3e-13)
    return 'diagonal quadratic KKT scalar-root oracle; projection translation and active sets'


def sinkhorn():
    m = load('public/learn-assets/optimal-transport/sinkhorn-library-bridge.py')
    for epsilon in [.1, .7, 2.]:
        answer, status = m.sinkhorn_log([.5,.5], [.5,.5], [[0,1],[1,0]], epsilon)
        diagonal = .5 / (1+math.exp(-1/epsilon))
        np.testing.assert_allclose(answer, [[diagonal,.5-diagonal],[.5-diagonal,diagonal]], atol=1e-11)
        assert status['converged']
    a, b = np.array([.2,.3,.5]), np.array([.4,.6])
    cost = np.array([[0.,2.],[1.,.5],[3.,0.]])
    base, status = m.sinkhorn_log(a,b,cost,.7)
    shifted, other = m.sinkhorn_log(a,b,cost+np.array([2.,-3.,1.])[:,None]+np.array([-.2,.8]),.7)
    assert status['converged'] and other['converged']
    np.testing.assert_allclose(base,shifted,atol=3e-10)
    return 'analytic 2x2 entropic solution; nonconstant row/column cost-shift invariance'


def causal():
    m = load('public/learn-assets/causal-inference/causal-estimation-bridge.py')
    sizes, successes = [17,31,23,43], [3,12,8,27]
    exact = sum(Fraction(sizes[2*z]+sizes[2*z+1],sum(sizes)) * (Fraction(successes[2*z+1],sizes[2*z+1])-Fraction(successes[2*z],sizes[2*z])) for z in range(2))
    manual, fitted, raw = m.estimate(sizes,successes)
    np.testing.assert_allclose(manual,float(exact),atol=1e-14)
    np.testing.assert_allclose(fitted,float(exact),atol=1e-10)
    # Swapping treatment labels negates the target, without changing stratum mass.
    swapped = [1,0,3,2]
    changed, _, _ = m.estimate([sizes[i] for i in swapped],[successes[i] for i in swapped])
    np.testing.assert_allclose(changed,-manual,atol=1e-14)
    return 'exact rational adjusted contrast on changed unequal cells; treatment-label reversal'


def mmd():
    m = load('public/learn-assets/divergences/mmd-library-bridge.py')
    x, y = np.array([[0.,1.],[.25,2.],[1.,-.5]]), np.array([[2.,1.],[-1.,.5],[1.5,2.],[0.,0.]])
    def kernel(a,b): return math.exp(-sum(float(v-w)**2 for v,w in zip(a,b))/(2*.7**2))
    for unbiased in [False,True]:
        xx = math.fsum(kernel(a,b) for i,a in enumerate(x) for j,b in enumerate(x) if not unbiased or i!=j)
        yy = math.fsum(kernel(a,b) for i,a in enumerate(y) for j,b in enumerate(y) if not unbiased or i!=j)
        xy = math.fsum(kernel(a,b) for a in x for b in y)
        exact = xx/(len(x)*(len(x)-int(unbiased)))+yy/(len(y)*(len(y)-int(unbiased)))-2*xy/(len(x)*len(y))
        for block in [1,2,7]:
            np.testing.assert_allclose(m.mmd2(x,y,.7,block,unbiased),exact,atol=5e-15)
            np.testing.assert_allclose(m.mmd2(x+2**24,y+2**24,.7,block,unbiased),exact,atol=5e-15)
    return 'scalar pair oracle, unequal sample sizes, both estimators, block partition and large translation'


def information():
    m = load('public/learn-assets/mutual-information/mutual-information-library-bridge.py')
    counts = np.array([[0,3,1],[2,0,4],[0,0,0]])
    rows, cols, total = counts.sum(axis=1), counts.sum(axis=0), int(counts.sum())
    exact = math.fsum(float(Fraction(int(counts[i,j]),total)) * math.log(float(Fraction(int(counts[i,j])*total,int(rows[i])*int(cols[j]))))
                     for i,j in zip(*np.nonzero(counts)))
    np.testing.assert_allclose(m.count_mi(counts),exact,atol=2e-15)
    np.testing.assert_allclose(m.count_mi(counts.T),exact,atol=2e-15)
    np.testing.assert_allclose(m.count_mi(np.diag([3,3,3])),math.log(3),atol=2e-15)
    return 'rational cell-probability scalar oracle, empty margins, axis swap and perfect three-class MI'


def graph():
    m = load('public/learn-assets/graph-fundamentals/laplacian-library-bridge.py')
    matrix = m.adjacency(5,[(0,1,2.),(1,2,3.),(2,3,4.),(1,1,7.),(3,4,0.)])
    # Constant current on a weighted chain: voltage increments proportional to 1/conductance.
    resistance = np.array([1/2,1/3,1/4])
    expected = np.r_[0.,np.cumsum(resistance)/resistance.sum()*6,9.]
    actual = m.harmonic_extension(matrix,[0,3,4],[0.,6.,9.])
    np.testing.assert_allclose(actual,expected,atol=2e-14)
    ordinary, normalized = m.operators(matrix)
    vector = np.array([2.,-1.,3.,.5,-4.])
    energy = math.fsum(w*(vector[i]-vector[j])**2 for i,j,w in [(0,1,2.),(1,2,3.),(2,3,4.)])
    np.testing.assert_allclose(vector@ordinary@vector,energy,atol=3e-14)
    assert np.all(normalized.toarray()[4]==0)
    return 'series-resistance harmonic oracle, edge-energy identity, loop cancellation and zero-edge isolate'


def queue():
    m = load('public/learn-assets/queueing/fcfs-simpy-bridge.py')
    arrivals, services = [0.,0.,.5,2.,2.,6.],[0.,1.,2.,0.,1.,0.]
    expected = np.array([[0,0,0],[0,0,1],[.5,1,3],[2,3,3],[2,3,4],[6,6,6]],float)
    for trace in [m.fcfs(arrivals,services),m.simulate(arrivals,services)]:
        np.testing.assert_array_equal(trace,expected)
        # Integrate independently over sorted arrival/departure events, including censoring.
        horizon=3.5
        boundaries=sorted(set([0.,horizon]+[float(t) for t in trace[:,[0,2]].ravel() if 0<t<horizon]))
        area=0.
        for left,right in zip(boundaries,boundaries[1:]):
            midpoint=(left+right)/2
            area+=(right-left)*sum(a<=midpoint<d for a,_,d in trace)
        np.testing.assert_allclose(m.occupancy_area(trace,horizon),area,atol=1e-15)
    return 'hand-scheduled tied and zero-service jobs; independent interval sweep for censored occupancy'


def sde():
    m = load('public/learn-assets/ito-calculus/sde-library-bridge.py')
    times=np.linspace(0,1,6); dw=np.array([[.1],[-.2],[.3],[.05],[-.1]])
    drift=lambda x,t:.2*x
    diffusion=lambda x,t:np.diag(.4*x)
    path=m.euler_maruyama(drift,diffusion,[1.3],times,dw)
    multipliers=1+.2*.2+.4*dw[:,0]
    expected=1.3*np.r_[1.,np.cumprod(multipliers)]
    np.testing.assert_allclose(path[:,0],expected,atol=1e-14)
    milstein=m.scalar_milstein(lambda x,t:.2*x,lambda x,t:.4*x,lambda x,t:.4,1.3,.2,dw[:,0])
    corrected=multipliers+.5*.4**2*(dw[:,0]**2-.2)
    np.testing.assert_allclose(milstein,1.3*np.r_[1.,np.cumprod(corrected)],atol=1e-14)
    return 'closed multiplicative Euler/Milstein path factors on specified nonrandom increments'


def finite_element():
    m = load('public/learn-assets/numerical-pdes/finite-element-banded-bridge.py')
    nodes=np.array([0.,.1,.4,.8,1.]); conductivity=np.array([1.,2.,4.,.5])
    diagonal,off,rhs=m.assemble(nodes,conductivity,np.zeros(4),boundary=(-1.,3.))
    factor=m.factor_ldl(diagonal,off)
    computed=m.solve_ldl(factor,rhs)
    resistance=np.diff(nodes)/conductivity
    exact=-1+4*np.cumsum(resistance)/resistance.sum()
    np.testing.assert_allclose(computed,exact[:-1],atol=2e-14)
    np.testing.assert_allclose(m.solve_ldl(factor,2*rhs),2*computed,atol=2e-14)
    # The load at an interior node must enter that single nodal basis function.
    _,_,point_rhs=m.assemble(nodes,conductivity,np.zeros(4),point_load=(.4,3.))
    np.testing.assert_array_equal(point_rhs,[0.,3.,0.])
    return 'piecewise-conductivity constant-flux analytic solution, factor reuse, exactly nodal point load'


def main():
    path = ROOT / 'docs/teaching/evidence/math-depth-remediation-independent.json'
    report = {'status':'running','reviewer':'root (not the author)','checks':{},'sourceHashes':{}}
    path.write_text(json.dumps(report,indent=2)+'\n')
    try:
        for name,function in [('derivative',derivative),('simplex',simplex),('sinkhorn',sinkhorn),('causal',causal),('mmd',mmd),('information',information),('graph',graph),('queue',queue),('sde',sde),('finite_element',finite_element)]:
            report['checks'][name]=function()
        report['status']='passed'
    except Exception:
        report['status']='failed'
        raise
    finally:
        for p in ['scripts/review-math-depth-remediation.py',
                  'public/learn-assets/matrix-calculus/derivative-library-bridge.py',
                  'public/learn-assets/constrained-optimization/simplex-solver-bridge.py',
                  'public/learn-assets/optimal-transport/sinkhorn-library-bridge.py',
                  'public/learn-assets/causal-inference/causal-estimation-bridge.py',
                  'public/learn-assets/divergences/mmd-library-bridge.py',
                  'public/learn-assets/mutual-information/mutual-information-library-bridge.py',
                  'public/learn-assets/graph-fundamentals/laplacian-library-bridge.py',
                  'public/learn-assets/queueing/fcfs-simpy-bridge.py',
                  'public/learn-assets/ito-calculus/sde-library-bridge.py',
                  'public/learn-assets/numerical-pdes/finite-element-banded-bridge.py']:
            report['sourceHashes'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'groups':len(report['checks'])}))


if __name__=='__main__': main()
