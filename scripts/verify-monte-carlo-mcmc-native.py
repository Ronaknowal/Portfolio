"""Independent rational, linear-algebra, quadrature and diagnostic oracles."""
import contextlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
from fractions import Fraction as F
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, norm, rankdata

folder = Path(sys.argv[1])
programs = json.loads((folder / 'programs.json').read_text())
namespaces = {}
for name, example in programs.items():
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(example['code'], scope)
    namespaces[name] = scope

# Polynomial integration with exact fractions: h(U)=U², paired group
# g(U)=U²-U+1/2. This does not depend on the random fixture.
integrate = lambda coefficients: sum((v/F(i+1) for i, v in enumerate(coefficients)), F(0))
assert integrate([0, 0, 1]) == F(1, 3)
assert integrate([0, 0, 0, 0, 1])-F(1, 9) == F(4, 45)
assert integrate([F(1,4), -1, 2, -2, 1])-F(1,9) == F(1,180)
assert namespaces['importance']['truth'] == F(17,10)
assert namespaces['importance']['weighted_variance'] == F(109,50)
assert namespaces['importance']['one_draw_ratio_mean'] != F(17,10)
mean = quad(lambda x: x*beta.pdf(x,10,4),0,1)[0]
tail = quad(lambda x: beta.pdf(x,10,4),.7,1)[0]
assert abs(mean-5/7) < 1e-12
assert abs(tail-beta.sf(.7,10,4)) < 1e-12
upper = F(3,10)
changed_tail = 36*upper**7 - 63*upper**8 + 28*upper**9
assert changed_tail == F(2145447,500000000)
assert abs(float(changed_tail)-quad(lambda x:beta.pdf(x,3,7),.7,1)[0]) < 1e-13
assert abs(float(changed_tail)-beta.sf(.7,3,7)) < 1e-13
assert abs(statistics.mean(namespaces['metropolis']['samples'])-mean) < .01
assert abs(statistics.mean(x>.7 for x in namespaces['metropolis']['samples'])-tail) < .02

class PortableRandom:
    def __init__(self, seed): self.state = seed
    def uniform(self):
        self.state ^= (self.state << 13) & 0xffffffff
        self.state ^= self.state >> 17
        self.state ^= (self.state << 5) & 0xffffffff
        return (self.state + .5)/4294967296
    def normal(self): return math.sqrt(-2*math.log(self.uniform()))*math.cos(2*math.pi*self.uniform())

fixtures = json.loads((folder / 'browser-fixtures.json').read_text())
rng = PortableRandom(7)
state = .5
for row in fixtures['mh']['rows']:
    proposal = state + .12*rng.normal()
    u = rng.uniform()
    alpha = min(1, beta.pdf(proposal,10,4)/beta.pdf(state,10,4))
    if u < alpha: state = proposal
    assert abs(state-row['state']) < 1e-12

hmc = namespaces['hamiltonian']
for epsilon in [.01, .1, .4, 1.5]:
    matrix = np.array([[1-epsilon**2/2, epsilon],[-epsilon*(1-epsilon**2/4),1-epsilon**2/2]])
    assert abs(np.linalg.det(matrix)-1) < 1e-12
    actual = hmc['leapfrog'](.8,-.3,epsilon)
    assert np.allclose(actual, matrix@np.array([.8,-.3]),rtol=0,atol=1e-12)
    assert np.allclose(hmc['leapfrog'](*actual,-epsilon),[.8,-.3],rtol=0,atol=1e-12)

transformed = namespaces['transformedHmc']
for eta in np.linspace(-15,15,101):
    difference = (transformed['potential'](eta+1e-4)-transformed['potential'](eta-1e-4))/2e-4
    assert abs(difference-transformed['gradient'](eta)) < 1e-7
    theta = transformed['sigmoid'](eta)
    # Compare the actual transformed density to scipy's original density times
    # dtheta/deta, not to the author's algebraic expansion.
    assert math.isclose(2860*math.exp(-transformed['potential'](eta)),beta.pdf(theta,10,4)*theta*(1-theta),rel_tol=2e-9)
assert abs(quad(lambda eta:2860*math.exp(-transformed['potential'](eta)),-100,100)[0]-1) < 1e-10
assert abs(statistics.mean(transformed['samples'])-mean) < .01
assert abs(statistics.mean(x>.7 for x in transformed['samples'])-tail) < .02

# Each NUTS result starts independently from the KNOWN stationary normal law.
# Check one-transition invariance empirically across step sizes/depth caps.
# This is a falsification check, not a finite proof of the general algorithm.
nuts = namespaces['nuts']
invariance = []
for epsilon, depth in [(.1,1),(.25,6),(.7,5),(1.2,4)]:
    rng = random.Random(927 + depth)
    outputs = [nuts['nuts_step'](rng.gauss(0,1),rng,epsilon,depth)[0] for _ in range(12000)]
    m, v = statistics.mean(outputs), statistics.variance(outputs)
    assert abs(m) < .04, (epsilon,m)
    assert abs(v-1) < .065, (epsilon,v)
    invariance.append((epsilon,depth,m,v))

diagnostics = namespaces['diagnostics']
for seed in range(20):
    rng = np.random.default_rng(seed)
    chains = rng.integers(-4,5,size=(4,40)).tolist()  # intentional ties
    actual = np.asarray(diagnostics['rank_normalize'](chains))
    flat = np.array(chains).ravel()
    oracle = norm.ppf((rankdata(flat,method='average')-3/8)/(flat.size+1/4)).reshape(4,40)
    assert np.allclose(actual,oracle,rtol=0,atol=1e-12)
    split = np.array([half for c in chains for half in (c[:20],c[-20:])],float)
    def oracle_rhat(values):
        scores = norm.ppf((rankdata(values.ravel(),method='average')-3/8)/(values.size+1/4)).reshape(values.shape)
        n = scores.shape[1]
        w = scores.var(axis=1,ddof=1).mean()
        b = n*scores.mean(axis=1).var(ddof=1)
        return np.sqrt(((n-1)*w/n+b/n)/w)
    expected = max(oracle_rhat(split),oracle_rhat(abs(split-np.median(split))))
    assert abs(diagnostics['rank_folded_rhat'](chains)-expected) < 1e-12
shifted = [[x+i*3 for x in c] for i,c in enumerate(diagnostics['chains'])]
assert diagnostics['rank_folded_rhat'](shifted) > 1.5
assert math.isnan(diagnostics['rank_folded_rhat']([[1]*20]*4))
# Known batch means: per-chain mean variance 1.25/4; two independent chains
# give sqrt(2*(1.25/4))/2. Each batch is intentionally constant, not iid draws.
c = [v for v in [0,1,2,3] for _ in range(10)]
expected_mcse = math.sqrt(2*(statistics.variance([0,1,2,3])/4))/2
assert abs(diagnostics['batch_mcse']([c,c],lambda x:x,10)-expected_mcse)<1e-12
(folder/'independent-evidence.json').write_text(json.dumps({'beta_mean':mean,'beta_tail_above_point7':tail,'changed_beta_3_7_tail':float(changed_tail),'changed_beta_3_7_tail_fraction':str(changed_tail),'one_step_stationarity_checks':invariance},indent=2))
print('Rational variance/importance, SciPy Beta quadrature, portable MH trace, leapfrog matrix/reversal, transformed-Beta Jacobian/gradient/normalization, 48,000 independent NUTS stationarity starts, tied/folded rank-Rhat and batch-MCSE checks passed.')
