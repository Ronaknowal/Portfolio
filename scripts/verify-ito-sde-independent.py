import contextlib
import datetime
from fractions import Fraction as F
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

root = Path.cwd()
directory = root/'scratch/ito-sde-independent-review'
data = json.loads((directory/'inputs.json').read_text())
namespaces = {}
for name, example in data['examples'].items():
    space = {}
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        exec(compile(example['code'], '<actual-ito-example-'+name+'>', 'exec'), space)
    assert captured.getvalue().strip() == example['expected'].strip()
    namespaces[name] = space
preserved = json.loads((root/'docs/teaching/evidence/ito-calculus-original-content.json').read_text())
assert data['examples']['original']['code'] == preserved['blocks'][0]['code']
assert data['examples']['original']['expected'] == preserved['blocks'][1]['code']

errors = []
def near(actual, expected, relative=5e-10, absolute=2e-13):
    assert math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute), (actual, expected)
    errors.append(abs(actual-expected))

# Integrate the actual weighted kernels and the orthogonal residual, rather than
# subtract two closed-form almost-equal variances.
for case in data['ou']:
    theta, h = case['theta'], case['step']
    covariance = quad(lambda s: math.exp(-theta*(h-s)), 0, h, epsabs=1e-15)[0]
    variance = quad(lambda s: math.exp(-2*theta*(h-s)), 0, h, epsabs=1e-15)[0]
    residual = quad(lambda u: h*(math.exp(-theta*h*(1-u))-covariance/h)**2, 0, 1, epsabs=1e-28)[0]
    js = case['result']
    near(js['covariance'], covariance)
    near(js['variance'], variance)
    near(js['residualVariance'], residual, relative=2e-9, absolute=1e-26)
    native = namespaces['ou']['coupled_noise'](theta,h)
    near(native[0], covariance)
    near(native[1], variance)
    near(native[2], residual, relative=2e-9, absolute=1e-26)

# Integrate two independent Gaussian increments through the actual native solver.
# This never uses the closed-form product/cross-moment equations under review.
points, weights = np.polynomial.hermite.hermgauss(24)
points *= math.sqrt(2)
weights /= math.sqrt(math.pi)
solve = namespaces['coupling']['solve_from_increments']
for case in data['moments']:
    h = case['horizon']/2
    bias = mse = second_bias = numerical_mean = 0.0
    for i, first in enumerate(points):
        for j, second in enumerate(points):
            em, milstein, exact = solve([float(first*math.sqrt(h)),float(second*math.sqrt(h))], case['horizon'], case['mu'],case['sigma'],case['initial'])
            approximate = em if case['method']=='euler' else milstein
            probability = weights[i]*weights[j]
            bias += probability*(approximate-exact)
            mse += probability*(approximate-exact)**2
            second_bias += probability*(approximate**2-exact**2)
            numerical_mean += probability*approximate
    js = case['result']
    for key, expected in [('meanBias',bias),('mse',mse),('secondBias',second_bias),('numericalMean',numerical_mean)]:
        near(js[key],expected)

# Exact rationals track different deterministic partitions of the SAME increments.
for case in data['sums']:
    increments = list(map(F,case['increments']))
    endpoint = sum(increments)
    quadratic = sum(item*item for item in increments)
    expected = {'terminal': endpoint, 'quadratic': quadratic, 'left': (endpoint**2-quadratic)/2, 'right':(endpoint**2+quadratic)/2,'symmetric':endpoint**2/2}
    for key,value in expected.items():
        assert case['result'][key] == float(value)
    actual = namespaces['integrals']['sums'](increments)
    assert actual == (endpoint,expected['left'],expected['right'],expected['symmetric'],quadratic)

# A Markov semigroup composition and a polynomial backward equation both use
# the actual native OU moment helper, with changed target, random initial law,
# and the deterministic-noise boundary.
ou = namespaces['ou']['ou_moments']
for theta in [.23,1.7]:
    for eta in [0,.7]:
        for start_variance in [0,.4]:
            target, initial, first, second = -.6,1.1,.3,.8
            m,v=ou(theta,target,eta,initial,start_variance,first)
            via=ou(theta,target,eta,m,v,second)
            direct=ou(theta,target,eta,initial,start_variance,first+second)
            near(via[0],direct[0]);near(via[1],direct[1])
            rate=namespaces['generator']['ou_second_rate'](theta,target,eta,m,v)
            near(rate[0],rate[1])

# Conditional cubic martingale identity and a changed exponential tilt checked
# by adaptive integration of a normalized density on the whole real line.
for t,x,h in [(0,-.4,.15),(.3,1.1,.7),(2,-1.7,.05)]:
    current,future=namespaces['time_transform']['conditional_transform'](t,x,h)
    near(current,future)
for T,c in [(.4,-.7),(1.3,.9),(.7,0.)]:
    density=lambda w: math.exp(-(w-c*T)**2/(2*T))/math.sqrt(2*math.pi*T)
    total=quad(density,-np.inf,np.inf)[0]
    mean=quad(lambda w:w*density(w),-np.inf,np.inf)[0]
    variance=quad(lambda w:(w-c*T)**2*density(w),-np.inf,np.inf)[0]
    near(total,1);near(mean,c*T);near(variance,T)

production = ['src/learn/data/topics/ito-calculus-stochastic-differential-equations.jsx','src/learn/data/ito-sde-models.js','src/learn/data/ito-sde-examples.js','src/learn/components/lesson-labs/ItoSdeLabs.jsx','src/learn/components/lesson-labs/ito-sde-labs.css','src/learn/data/curriculum/blueprints/it-calculus-stochastic-differential-equations.js']
result = {'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','counts':{'actualDisplayedPrograms':len(namespaces),'exactOriginalProgramAndOutput':1,'independentKernelIntegrals':len(data['ou']),'twoIncrementGaussianQuadratures':len(data['moments']),'actualSolverCalls':len(data['moments'])*24**2,'exactNestedPartitionTraces':len(data['sums']),'ouSemigroupAndMomentCases':8,'conditionalCubicCases':3,'changedGaussianTilts':3},'maximumAbsoluteDifference':max(errors),'production':[{'path':p,'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()} for p in production], 'limits':'Complementary finite numerical checks and full mathematical source read; not a rerun of author browser or an infinite-dimensional theorem proof.'}
(directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='production'},indent=2))
