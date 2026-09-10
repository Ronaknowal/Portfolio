"""Independent numerical and exact checks for the exponential-family lesson."""
import datetime
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, logsumexp, softmax
from scipy.stats import beta as beta_distribution, norm, poisson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/exponential-family-verification'
OUT.mkdir(parents=True, exist_ok=True)
node_source = r'''
import * as model from './src/learn/data/exponential-family-models.js';
import {exponentialFamilyExamples as examples} from './src/learn/data/exponential-family-examples.js';
const families=[];
for(const first of [-100,-5,-2,-.5,0,.3,2,5,100]) for(const second of [-5,-1,0,.4,5]) for(const base of [[1,1,1],[1,2,1],[.5,3,2]]) families.push({first,second,base,result:model.finiteExponentialFamily(first,second,base)});
const fits=[];for(let a=0;a<=8;a++)for(let b=0;b<=8;b++)for(let c=0;c<=8;c++)fits.push({counts:[a,b,c],result:model.finiteMomentFit([a,b,c])});
const comparisons=[];for(let bits=0;bits<256;bits++)for(const grouped of [false,true]){const data=Array.from({length:8},(_,index)=>(bits>>index)&1);comparisons.push({data,grouped,result:model.summaryComparison(data,.8,.3,grouped)});}
const priors=[];for(let alpha=1;alpha<=8;alpha++)for(let beta=1;beta<=8;beta++){const state=model.betaCoordinateModel(alpha,beta);priors.push({...state,heights:[.1,.5,.9].map(p=>({p,density:state.probabilityDensity(p),etaDensity:state.etaDensity(Math.log(p/(1-p)))}))});}
const summaries=[];for(let offset of [0,1e6,1e10])for(let count=2;count<=31;count++){const data=Array.from({length:count},(_,index)=>offset+((index*17)%13)-6);const middle=Math.floor(count/2);summaries.push({data,result:model.gaussianSummary(data),merged:model.mergeGaussianSummaries(model.gaussianSummary(data.slice(0,middle)),model.gaussianSummary(data.slice(middle)))});}
let invalid=0;for(const callback of [()=>model.finiteMomentFit([-1,2,3]),()=>model.finiteMomentFit([.5,2,3]),()=>model.finiteExponentialFamily(Infinity),()=>model.finiteExponentialFamily(101),()=>model.finiteExponentialFamily(0,0,[0,1,2]),()=>model.betaCoordinateModel(0,1),()=>model.betaCoordinateModel(2.5,3),()=>model.summaryComparison([0]),()=>model.summaryComparison(Array(8).fill(0),0),()=>model.gaussianSummary([]),()=>model.mergeGaussianSummaries({count:0},{count:1})]){try{callback();throw Error('accepted invalid input')}catch(error){if(!(error instanceof RangeError))throw error;invalid++;}}
console.log(JSON.stringify({examples,families,fits,comparisons,priors,summaries,invalid,tiny:model.formatFamilyNumber(1e-12)}));
'''
payload = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', node_source], cwd=ROOT, text=True, encoding='utf-8'))
assert payload['tiny'] != '0'
maximum_error = {'finite': 0.0, 'gaussian': 0.0, 'prior': 0.0, 'merge': 0.0}

programs = []
namespaces = {}
for key, example in payload['examples'].items():
    path = OUT / f'program-{key}.py'
    path.write_text(example['code'] + '\n', encoding='utf-8')
    actual = subprocess.check_output([sys.executable, '-X', 'utf8', '-I', str(path)], text=True, encoding='utf-8').rstrip()
    assert actual == example['expected'], (key, actual, example['expected'])
    programs.append({'key': key, 'stdout': actual})
    # Load the actual learner functions for changed-input acceptance checks.
    namespace = {}
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'], str(path), 'exec'), namespace)
    namespaces[key] = namespace
original = json.loads((ROOT / 'docs/teaching/evidence/exponential-family-original-content.json').read_text(encoding='utf-8'))
assert payload['examples']['bernoulli']['code'] == original['originalPrograms'][0]['code']
assert payload['examples']['bernoulli']['expected'] == original['originalPrograms'][0]['expected']

outcomes = np.array([-1., 0., 1.])
statistics = np.column_stack((outcomes, outcomes ** 2))
for item in payload['families']:
    parameters = np.array([item['first'], item['second']])
    logs = statistics @ parameters + np.log(item['base'])
    partition = logsumexp(logs)
    probabilities = softmax(logs)
    mean = probabilities @ statistics
    centered = statistics - mean
    covariance = centered.T @ np.diag(probabilities) @ centered
    result = item['result']
    np.testing.assert_allclose(result['probabilities'], probabilities, rtol=8e-14, atol=0)
    np.testing.assert_allclose(result['mean'], mean, atol=2e-14)
    np.testing.assert_allclose(result['covariance'], covariance, rtol=8e-13, atol=2e-16)
    np.testing.assert_allclose(np.diag(result['covariance']), np.diag(covariance), rtol=8e-13, atol=1e-100)
    maximum_error['finite'] = max(maximum_error['finite'], abs(partition - result['logPartition']))
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-14
    # Check that scalar tilt CGF agrees with direct finite expectation.
    for tilt in [-.2,.15]:
        direct = logsumexp(logs - partition + tilt * outcomes)
        shifted = logsumexp(logs + tilt * outcomes) - partition
        assert abs(direct - shifted) < 6e-14

for item in payload['fits']:
    counts = np.array(item['counts'])
    result = item['result']
    if counts.sum() == 0:
        assert result['status'] == 'empty' and result['mean'] is None
    elif counts.min() == 0:
        assert result['status'] == 'boundary' and result['parameters'] is None
    else:
        logits = statistics @ np.array(result['parameters'])
        probabilities = np.exp(logits - logsumexp(logits))
        np.testing.assert_allclose(probabilities, counts / counts.sum(), atol=5e-16)
        np.testing.assert_allclose(probabilities @ statistics, result['mean'], atol=5e-16)
        # Cross-entropy/KL certifies the global maximizer, not just score zero.
        candidate = np.array([.2,.3,.5])
        assert np.dot(counts, np.log(probabilities)) >= np.dot(counts, np.log(candidate)) - 1e-13

reference = [1,1,1,1,0,0,1,1]
for item in payload['comparisons']:
    probabilities = [Fraction(4,5)] * 4 + [Fraction(3,10) if item['grouped'] else Fraction(4,5)] * 4
    def likelihood(data):
        return math.prod(p if value else 1-p for value,p in zip(data,probabilities))
    ratio = float(likelihood(item['data']) / likelihood(reference))
    assert math.isclose(item['result']['ratio'], ratio, rel_tol=4e-15)
    if sum(item['data']) == 6 and not item['grouped']:
        assert item['result']['ratio'] == 1

for item in payload['priors']:
    alpha,beta = item['alpha'],item['beta']
    oracle = beta_distribution(alpha,beta)
    expected = oracle.cdf(.75)-oracle.cdf(.25)
    assert abs(item['intervalMass']-expected) < 2e-14
    integrated_eta = quad(lambda eta: oracle.pdf(expit(eta))*expit(eta)*expit(-eta), -math.log(3), math.log(3), epsabs=1e-12)[0]
    assert abs(integrated_eta-expected) < 1e-12
    for point,density in item['probabilityCurve']:
        maximum_error['prior'] = max(maximum_error['prior'],abs(density-oracle.pdf(point)))
        assert math.isclose(density,oracle.pdf(point),rel_tol=2e-13,abs_tol=1e-14)
    for eta,density in item['etaCurve']:
        expected_density=oracle.pdf(expit(eta))*expit(eta)*expit(-eta)
        assert math.isclose(density,expected_density,rel_tol=5e-13,abs_tol=1e-16)

for item in payload['summaries']:
    values=np.asarray(item['data'], dtype=np.longdouble)
    expected_mean=float(np.mean(values))
    expected_m2=float(np.sum((values-values.mean())**2))
    for summary in [item['result'], item['merged']]:
        assert abs(summary['mean']-expected_mean) < 5e-6
        error=abs(summary['squaredDeviations']-expected_m2)
        maximum_error['merge']=max(maximum_error['merge'],error)
        assert error < 8e-5, (summary,error)

gaussian_cases=0
for mean in [-4,0,.7,3]:
    for variance in [.1,.5,1,4,10]:
        first,second=mean/variance,-1/(2*variance)
        partition=-first*first/(4*second)+.5*math.log(-math.pi/second)
        for observation in [-5,-1,0,2,7]:
            canonical=first*observation+second*observation**2-partition
            direct=norm.logpdf(observation,loc=mean,scale=math.sqrt(variance))
            maximum_error['gaussian']=max(maximum_error['gaussian'],abs(canonical-direct))
            assert abs(canonical-direct)<2e-13
            gaussian_cases+=1

# Check actual learner-function changes and all numerical hand tasks.
for rate in [.3,1,4,20]:
    actual=namespaces['poissonRatio']['log_likelihood']([0,3,1],rate)
    np.testing.assert_allclose(actual,poisson.logpmf([0,3,1],rate).sum(),atol=2e-14)
for eta in [-5,-1,0,1,5]:
    actual=namespaces['normalizer']['family'](eta)
    # h=(1,2,1) yields the centered sum of two Bernoulli(sigmoid eta).
    probability=expit(eta)
    assert abs(actual[2]-(2*probability-1))<1e-15
    assert abs(actual[3]-2*probability*(1-probability))<1e-15
for counts in [[1,8,1],[4,2,4]]:
    status,parameters=namespaces['moments']['fit'](counts)
    expected=(0,math.log(counts[0]/counts[1]))
    np.testing.assert_allclose(parameters,expected,atol=1e-14)
for values in [[0,2,4,6],[1,1,1],[1e8+1,1e8+2,1e8+3]]:
    count,mean,m2=namespaces['gaussianMerge']['summarize'](values)
    assert math.isclose(m2,float(np.var(values))*len(values),abs_tol=1e-12)
merged=namespaces['gaussianMerge']['merge']((2,1,2),(2,5,2))
assert merged==(4,3,20)
assert namespaces['exposure']['update'](3,2,[2,4],[.5,1.5])==(9,4)
assert namespaces['exposure']['update'](3,2,[2,4],[.5,2.5])==(9,5)
assert Fraction(38,15)-Fraction(22,15)**2==Fraction(86,225)
assert Fraction(9,240)==Fraction(9,4)/60
assert beta_distribution(4,2).mean()==4/6
feature=namespaces['features']
matrix=np.asarray(feature['design']); outcomes_y=np.asarray(feature['outcomes']); coefficients=np.asarray(feature['coefficients'])
scores=matrix@coefficients; probabilities=expit(scores)
np.testing.assert_allclose(feature['gradient'],matrix.T@(probabilities-outcomes_y),atol=1e-15)
np.testing.assert_allclose(feature['hessian'],matrix.T@np.diag(probabilities*(1-probabilities))@matrix,atol=1e-15)
np.testing.assert_allclose(matrix.T@np.array([1,1,0]),[2,-1])
assert Fraction(3,10)/Fraction(7,10)/(Fraction(4,5)/Fraction(1,5))==Fraction(3,28)
for p in [Fraction(1,5),Fraction(4,5)]:
    for successes in range(5):
        fiber=[values for values in itertools.product([0,1],repeat=4) if sum(values)==successes]
        probabilities=[p**successes*(1-p)**(4-successes)]*len(fiber)
        assert all(weight/sum(probabilities)==Fraction(1,len(fiber)) for weight in probabilities)

result={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'programs':programs,'preservedOriginalPrograms':1,'finiteFamilies':len(payload['families']),'momentCountCases':len(payload['fits']),'exactLikelihoodComparisons':len(payload['comparisons']),'betaPriorCases':len(payload['priors']),'gaussianDensityCases':gaussian_cases,'gaussianSummaryCases':len(payload['summaries']),'invalidContracts':payload['invalid'],'maximumErrors':maximum_error,'changedPractice':'All numerical hand-task results and model-change counterexamples checked; capstone and constrained-family argument inspected logically.'}
(OUT/'native-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
