"""Complete finite constructions and bounded simulations for PAC/VC learning.

Run with Python 3.12+, NumPy and SciPy. Constructed worlds have known risks;
the simulation illustrates a theorem's setting and is not its proof.
"""
from pathlib import Path
from itertools import product
from fractions import Fraction
from math import comb, exp, log, log2, sqrt, ceil, pi
import json
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent


def interval_patterns(n):
    """All patterns on n distinct ordered points, including all negative."""
    patterns = {(0,)*n}
    for left in range(n):
        for right in range(left, n):
            patterns.add(tuple(int(left <= i <= right) for i in range(n)))
    return sorted(patterns)


def threshold_patterns(n):
    return [tuple(int(i >= boundary) for i in range(n)) for boundary in range(n+1)]


def interval_witness(points, labels):
    x, y = np.asarray(points, float), np.asarray(labels, int)
    if x.ndim != 1 or y.shape != x.shape or not len(x) or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Use matching nonempty finite points and binary labels.')
    positive = x[y == 1]
    if not len(positive):
        return {'feasible':True, 'interval':None, 'predicted':[0]*len(x)}
    left, right = float(positive.min()), float(positive.max())
    predicted = ((x >= left) & (x <= right)).astype(int)
    return {'feasible':bool(np.array_equal(predicted,y)), 'interval':[left,right], 'predicted':predicted.tolist()}


def halfplane_patterns(points):
    """Finite LP feasibility checks, with numerical residual verification.

    A finite strictly separable labeling can be rescaled to signed margin >=1.
    Numerical success/failure is evidence for fixtures, not a proof about every
    point configuration. The manuscript supplies the geometric VC proof.
    """
    x = np.asarray(points,float)
    augmented = np.c_[x,np.ones(len(x))]
    found = []
    rejected = []
    for labeling in product([0,1], repeat=len(x)):
        sign = 2*np.array(labeling)-1
        result = linprog(np.zeros(3), A_ub=-sign[:,None]*augmented,
                         b_ub=-np.ones(len(x)), bounds=[(None,None)]*3, method='highs')
        if result.success:
            margin = sign*(augmented@result.x)
            assert margin.min() >= 1-1e-8
            found.append({'labels':list(labeling),'w':result.x[:2].tolist(),'b':float(result.x[2]),'minimum_signed_margin':float(margin.min())})
        elif result.status == 2:
            rejected.append(list(labeling))
        else:
            raise RuntimeError(result.message)
    return {'points':x.tolist(),'realized':found,'infeasible':rejected,'count':len(found)}


def finite_radius(k,n,delta):
    if k < 1 or n < 1 or not 0 < delta < 1:
        raise ValueError('Use positive k/n and delta in (0,1).')
    return sqrt(log(2*k/delta)/(2*n))


def vc_radius(d,n,delta):
    """Conservative explicit theorem in Mehta lectures12/13, section4."""
    if not 1 <= d <= n or not 0 < delta < 1:
        raise ValueError('This displayed form requires 1 <= d <= n, 0 < delta < 1.')
    return sqrt(32*(d*log(exp(1)*n/d)+log(8/delta))/n)


def realizable_vc_sample_bound(d,epsilon,delta):
    """Classical sufficient bound, Blumer et al.1989 Theorem2.1; log base2."""
    if d < 1 or not 0 < epsilon < 1 or not 0 < delta < 1:
        raise ValueError('Use positive d, epsilon/delta in (0,1).')
    return ceil(max(4/epsilon*log2(2/delta),8*d/epsilon*log2(13/epsilon)))


def interval_risk(interval,target=(.3,.7)):
    a,b = target
    if interval is None:
        return b-a
    left,right = interval
    # Inputs are in [0,1], and the distribution is uniform on [0,1].
    intersection = max(0.,min(b,right)-max(a,left))
    return (b-a)+(right-left)-2*intersection


def fit_realizable_interval(points,target=(.3,.7)):
    x = np.asarray(points,float)
    a,b = target
    y = ((x>=a)&(x<=b)).astype(int)
    fitted = interval_witness(x,y)
    assert fitted['feasible']
    return {**fitted,'points':x.tolist(),'labels':y.tolist(),'empirical_error':0.,'true_error':interval_risk(fitted['interval'],target)}


def repeated_intervals(seed=41,repetitions=1000,epsilon=.1):
    rng = np.random.default_rng(seed)
    rows = []
    for n in [10,20,50,100,200]:
        errors = np.array([fit_realizable_interval(rng.random(n))['true_error'] for _ in range(repetitions)])
        rows.append({'n':n,'repetitions':repetitions,'epsilon':epsilon,'failure_count':int(np.sum(errors>epsilon)),
                     'failure_fraction':float(np.mean(errors>epsilon)), 'mean_true_error':float(errors.mean()),
                     'minimum_true_error':float(errors.min()),'maximum_true_error':float(errors.max()),
                     'risk_quantiles':np.quantile(errors,[.05,.5,.95]).tolist(),
                     'distribution_specific_failure_bound':min(1.,2*(1-epsilon/2)**n),
                     'uniform_vc_radius_raw_delta005':vc_radius(2,n,.05),
                     'sample_true_errors':errors[:20].tolist()})
    return {'seed':seed,'target':[.3,.7],'distribution':'Uniform[0,1]','rows':rows}


def sine_witness(labels):
    """Exact binary-fraction sign construction; finite precision is explicit."""
    n = len(labels)
    r = sum((Fraction(1-int(y),2**(i+1)) for i,y in enumerate(labels)),Fraction(1,2**(n+2)))
    points = [2**i for i in range(n)]
    fractions = [(r*x)%1 for x in points]
    predictions = [int(0 < value < Fraction(1,2)) for value in fractions]
    assert predictions == list(labels)
    return {'labels':list(labels),'points':points,'r_fraction':str(r),'theta_approx':2*pi*float(r),
            'fractional_cycles':[str(v) for v in fractions],'predicted':predictions}


def finite_world(n,epsilon=.25,target=(0,0,1,1)):
    """Exact occupancy DP; no sampling noise in probabilities in this world.

    All 16 binary functions are candidates. Lexicographically first consistent
    rule predicts zero at unobserved points. Samples drawn iid from four points.
    """
    probabilities = [Fraction(0) for _ in range(16)]; probabilities[0] = Fraction(1)
    for _ in range(n):
        updated = [Fraction(0) for _ in range(16)]
        for mask,probability in enumerate(probabilities):
            for point in range(4):
                updated[mask | (1<<point)] += probability/4
        probabilities = updated
    states = []
    failure = Fraction(0)
    for mask,probability in enumerate(probabilities):
        prediction = tuple(target[i] if mask & (1<<i) else 0 for i in range(4))
        risk = Fraction(sum(a!=b for a,b in zip(prediction,target)),4)
        if risk > epsilon:
            failure += probability
        if probability:
            states.append({'seen':[i for i in range(4) if mask&(1<<i)],'probability':str(probability),
                           'selected_hypothesis':list(prediction),'true_risk':float(risk)})
    assert sum(probabilities) == 1
    return {'n':n,'epsilon':epsilon,'target':list(target),'failure_probability':float(failure),
            'failure_fraction_exact':str(failure),'bound_raw':16*exp(-n*epsilon),
            'bound_clipped':min(1.,16*exp(-n*epsilon)),'states':states}


def examples():
    growth = [{'n':n,'thresholds':len(threshold_patterns(n)), 'intervals':len(interval_patterns(n)),
               'all_binary':2**n,'sauer_d2':sum(comb(n,i) for i in range(min(2,n)+1))} for n in range(1,11)]
    geometry = {name:halfplane_patterns(points) for name,points in {
        'triangle':[(0,0),(1,0),(0,1)],'square':[(0,0),(1,0),(1,1),(0,1)],
        'collinear':[(0,0),(1,0),(2,0)],'interior':[(0,0),(2,0),(0,2),(.5,.5)]}.items()}
    assert geometry['triangle']['count']==8 and geometry['square']['count']==14 and geometry['collinear']['count']==6
    base = fit_realizable_interval([.1,.2,.35,.55,.65,.9])
    negatives = fit_realizable_interval([.01,.1,.2,.35,.55,.65,.9,.99])
    assert base['interval'] == negatives['interval']
    sine = [sine_witness(label) for label in product([0,1],repeat=4)]
    fixtures = {'growth':growth,'geometry':geometry,
                'intervals':{'base':base,'closer_edges':fit_realizable_interval([.1,.2,.31,.35,.55,.65,.69,.9]),
                             'negative_only_null':negatives,'no_positives':fit_realizable_interval([.1,.2,.8,.9]),
                             'inconsistent':interval_witness([.2,.5,.8],[1,0,1])},
                'finite_family':{'k25_n500_delta005':finite_radius(25,500,.05),
                                 'single_n500_delta005':finite_radius(1,500,.05),
                                 'k25_n2000_delta005':finite_radius(25,2000,.05),
                                 'realizable_k32_eps005_delta001':ceil((log(32)+log(100))/.05)},
                'bounds':[{'d':d,'n':n,'delta':.05,'uniform_radius_raw':vc_radius(d,n,.05)} for d in [1,2,10] for n in [100,1000,10000,100000]],
                'realizable_sample_bounds':[{'d':d,'epsilon':e,'delta':.05,'sufficient_n':realizable_vc_sample_bound(d,e,.05)} for d in [1,2,10] for e in [.2,.1,.05]],
                'sauer_n100_d5':sum(comb(100,i) for i in range(6)),
                'sine_four_labels':sine,'finite_world':[finite_world(n) for n in [1,2,4,8,16,24]],
                'finite_world_zero_target_null':finite_world(4,target=(0,0,0,0)),
                'simulation':repeated_intervals(),
                'practice':{'k12_n800_delta002':finite_radius(12,800,.02),
                            'interval_patterns_n4':interval_patterns(4),
                            'sauer_n6_d2':sum(comb(6,i) for i in range(3)),
                            'uniform_bound_n2000_d2':vc_radius(2,2000,.05),
                            'changed_target_interval':fit_realizable_interval([.05,.3,.4,.7,.95],(.25,.75)),
                            'changed_target_positive':fit_realizable_interval([.05,.26,.3,.4,.7,.95],(.25,.75)),
                            'changed_target_negative_null':fit_realizable_interval([.05,.3,.4,.7,.95,.99],(.25,.75))}}
    return fixtures


if __name__ == '__main__':
    results = examples()
    (HERE/'checked-results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print('growth',results['growth'])
    print('geometry',{k:v['count'] for k,v in results['geometry'].items()})
    print('finite family',results['finite_family'])
    print('finite-world failures',[(v['n'],v['failure_fraction_exact']) for v in results['finite_world']])
    print('interval simulation',results['simulation']['rows'])
