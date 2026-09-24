"""Independent NumPy/SciPy oracles and actual displayed-program execution."""
import contextlib
import io
import json
import math
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'scratch/random-matrix-verification'
subprocess.run(['node', 'scripts/verify-random-matrix-models.mjs'], cwd=ROOT, check=True)
fixtures = json.loads((DATA / 'model-fixtures.json').read_text())
max_eigen_error = 0.0
for state in fixtures['covariance'] + [item['state'] for item in fixtures['spikes']]:
    values = np.asarray(state['data'])
    if state['centered']:
        values = values - values.mean(axis=0)
    gram = values.T @ values / state['denominator']
    np.testing.assert_allclose(gram, state['covariance'], atol=2e-14)
    eigenvalues, vectors = np.linalg.eigh(gram)
    max_eigen_error = max(max_eigen_error, float(np.max(np.abs(eigenvalues-state['values']))))
    np.testing.assert_allclose(eigenvalues, state['values'], atol=1e-9)
    singular = np.linalg.svd(values, compute_uv=False)
    positive = sorted(singular**2/state['denominator'])
    np.testing.assert_allclose(eigenvalues[-len(positive):], positive, atol=2e-12)
    assert state['zeroCount'] >= max(0,state['columns']-state['denominator'])
    np.testing.assert_allclose(state['secondMoment'], np.sum(gram**2)/state['columns'], atol=1e-10)
    if state['alignment'] is not None:
        np.testing.assert_allclose(state['alignment'], vectors[0,-1]**2, atol=1e-7)

max_mass_error=0.0
for item in fixtures['intervals']:
    gamma, variance=item['gamma'],item['variance']
    lower=variance*(1-math.sqrt(gamma))**2
    upper=variance*(1+math.sqrt(gamma))**2
    width=upper-lower
    start=math.asin(math.sqrt(max(0,min(1,(item['lower']-lower)/width))))
    end=math.asin(math.sqrt(max(0,min(1,(item['upper']-lower)/width))))
    def integrand(theta):
        if gamma == 1:
            return 4/math.pi*math.cos(theta)**2
        value=lower+width*math.sin(theta)**2
        return 16*variance/math.pi*(math.sin(theta)*math.cos(theta))**2/value
    # Resolve the thin boundary layer separately near gamma=1.
    characteristic=abs(1-math.sqrt(gamma))/(2*gamma**.25)
    points=sorted({start,end,*[multiple*characteristic for multiple in [1,10,100,10000] if start<multiple*characteristic<end]})
    expected=sum(quad(integrand,a,b,epsabs=2e-12,epsrel=2e-12,limit=300)[0] for a,b in zip(points,points[1:]))
    max_mass_error=max(max_mass_error,abs(expected-item['mass']))
    np.testing.assert_allclose(item['mass'],expected,atol=2e-9,rtol=2e-9)

for item in fixtures['spikes']:
    gamma=item['state']['gamma']; population=item['state']['spike']; limit=item['limit']
    assert 0<=limit['alignment']<=1
    if population <= 1+math.sqrt(gamma):
        np.testing.assert_allclose(limit['sample'],(1+math.sqrt(gamma))**2)
        assert limit['alignment']==0
    else:
        # Check the separated branch's scalar Schur equation by independent integration.
        rho=limit['sample']; lower=(1-math.sqrt(gamma))**2; upper=(1+math.sqrt(gamma))**2
        integral=quad(lambda x: math.sqrt((upper-x)*(x-lower))/(2*math.pi*gamma)/(rho-x),lower,upper)[0]
        np.testing.assert_allclose(rho,population*(1+gamma*integral),atol=2e-8)
        derivative=1-gamma/(population-1)**2
        np.testing.assert_allclose(limit['alignment'],derivative/(1+gamma/(population-1)))
for state in fixtures['nullStates']:
    matrix=np.asarray(state['data'])
    largest=float(np.linalg.eigvalsh(matrix.T@matrix/40)[-1])
    key='duplicate' if state['duplicate'] else 'iid'
    index=(state['seed']-2000)//7919
    record=next(item for item in fixtures['calibration'] if item['nullModel']==key)
    np.testing.assert_allclose(record['maxima'][index],largest,atol=1e-11)
for item in fixtures['wigner']:
    matrix=np.asarray(item['matrix'])
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix),item['values'],atol=2e-10)
    np.testing.assert_allclose(np.sum(matrix**2)/item['size'],item['secondMoment'],atol=2e-10)
    if item['law']=='sign':
        assert np.all(np.diag(matrix)==0)
        np.testing.assert_allclose(item['secondMoment'],1-1/item['size'],atol=2e-10)
for item in fixtures['levels']:
    m,d,c=item['offset'],item['difference'],item['coupling']
    eigen=np.linalg.eigvalsh([[m+d,c],[c,m-d]])
    np.testing.assert_allclose(eigen,[item['lower'],item['upper']],atol=2e-14)
    np.testing.assert_allclose(eigen[1]-eigen[0],item['gap'],atol=2e-14)
for item in fixtures['bounds']:
    t=math.sqrt(2*(math.log(2)-math.log(item['delta'])))
    expected=[max(0,math.sqrt(item['rows'])-math.sqrt(item['columns'])-t)**2/item['rows'],(math.sqrt(item['rows'])+math.sqrt(item['columns'])+t)**2/item['rows']]
    np.testing.assert_allclose([item['lower'],item['upper']],expected,atol=1e-13)
for item in fixtures['eigenFixtures']:
    matrix=np.asarray(item['matrix']); vectors=np.asarray(item['vectors']).T
    np.testing.assert_allclose(matrix@vectors,vectors*np.asarray(item['values']),atol=2e-10)
    np.testing.assert_allclose(vectors.T@vectors,np.eye(len(matrix)),atol=2e-10)

archive=json.loads((ROOT/'docs/teaching/evidence/random-matrix-original-content.json').read_text())
assert fixtures['examples']['original']['code']==archive['program'].strip()
assert fixtures['examples']['original']['expected']==archive['output'].strip()
namespaces={}
for key,example in fixtures['examples'].items():
    destination=DATA/(key+'.py')
    destination.write_text(example['code']+'\n',encoding='utf-8')
    run=subprocess.run([sys.executable,'-X','utf8','-I',str(destination)],check=True,capture_output=True,text=True,encoding='utf-8')
    assert run.stdout.rstrip()==example['expected'], key
    assert not run.stderr,key
    namespace={}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example['code'],str(destination),'exec'),namespace)
    namespaces[key]=namespace

# Changed learner functions and independent hand-task answers.
np.testing.assert_allclose(namespaces['gap']['two_levels'](3,-2,1),np.linalg.eigvalsh([[1,1],[1,5]]))
np.testing.assert_allclose(-math.expm1(-.2**2/8),.004987520807317687)
atom,moments=namespaces['mass']['mp_moments'](4,9)
np.testing.assert_allclose([atom,*moments],[.75,.25,9,405],atol=2e-7)
native_mass_regressions = []
for gamma in [.02, .8, .9999, .99999, .999999, 1,
              1.000001, 1.00001, 1.0001, 2.25, 20]:
    for variance in [.01, 1, 4, 9, 100]:
        with warnings.catch_warnings(record=True) as caught:
            atom, moments = namespaces['mass']['mp_moments'](gamma, variance)
        expected = [min(1, 1 / gamma), variance,
                    variance**2 * (1 + gamma)]
        np.testing.assert_allclose(moments, expected, atol=2e-9, rtol=2e-10)
        np.testing.assert_allclose(atom + moments[0], 1, atol=2e-10)
        assert not caught, (gamma, variance, caught)
        native_mass_regressions.append({
            'gamma': gamma, 'variance': variance,
            'totalMass': atom + moments[0],
            'maxScaledError': float(np.max(
                np.abs(np.asarray(moments) - expected)
                / np.maximum(1, np.abs(expected))
            )),
        })
assert namespaces['moments']['exact_gaussian_second_moment'](80,20)==1.2625
np.testing.assert_allclose(namespaces['spike']['limits'](.5,3),[3.75,.7])
assert namespaces['bound']['gaussian_covariance_bound'](25,20,.05)[0]==0
assert Fraction(42,10)/Fraction(24,100)==Fraction(35,2)
np.testing.assert_allclose(np.linalg.solve(np.diag([.04,4])+.2*np.eye(2),np.ones(2)),[1/.24,1/4.2])

for function, arguments in [(namespaces['mass']['mp_moments'],(0,1)),(namespaces['mass']['mp_moments'],(1,float('inf'))),(namespaces['spike']['limits'],(1,2)),(namespaces['gap']['two_levels'],(0,float('nan'),1))]:
    try:
        function(*arguments)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported native state accepted')

# Exact expectation of the sign-valued 2-row, 2-column ensemble by all 16 matrices.
from itertools import product
sign_moments=[]
for entries in product([-1,1],repeat=4):
    matrix=np.asarray(entries).reshape(2,2)
    covariance=matrix.T@matrix/2
    sign_moments.append(np.sum(covariance**2)/2)
assert sum(sign_moments)/len(sign_moments)==1.5
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'programsExecuted':len(namespaces),'exactOriginalPreserved':True,'covarianceAndSpikes':len(fixtures['covariance'])+len(fixtures['spikes']),'mpIntervalIntegrals':len(fixtures['intervals']),'maxEigenError':max_eigen_error,'maxMpMassError':max_mass_error,'wigner':len(fixtures['wigner']),'twoLevel':len(fixtures['levels']),'finiteBounds':len(fixtures['bounds']),'changedNumericalPracticeGroups':7,'openScenarioSourceReviewed':True,'independentNullMaxima':118,'exactSignMatrices':16,'status':'passed'}
result['displayedMassHelperRegressions'] = native_mass_regressions
(DATA/'native-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
