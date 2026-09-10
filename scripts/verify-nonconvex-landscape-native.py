import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize_scalar

folder = Path(sys.argv[1])
cases = json.loads((folder / 'model-cases.json').read_text())
examples = json.loads((folder / 'examples.json').read_text(encoding='utf8'))
signs = json.loads((folder / 'signs.json').read_text())
modules = {}
for key, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(example['code'], namespace)
    modules[key] = namespace

def close(actual, expected, **kwargs):
    np.testing.assert_allclose(actual, expected, atol=kwargs.get('atol', 2e-10), rtol=kwargs.get('rtol', 2e-9))

well_frames = 0
for state in cases['wells']:
    tilt = state['tilt']
    roots = modules['wells']['stationary_points'](tilt)
    close([point['x'] for point in state['critical']], roots)
    values = [modules['wells']['well'](root, tilt) for root in roots]
    close([point['value'] for point in state['critical']], values)
    close(state['best'], min(values))
    # Separate bounded minimizations of the two wells, using a library algorithm.
    if state['initial'] == 0 and state['rate'] == 0:
        minima = [minimize_scalar(lambda x: modules['wells']['well'](x, tilt), bounds=interval, method='bounded').fun for interval in [(-1.5,-0.58),(0.58,1.5)]]
        close(min(minima), state['best'], atol=1e-9)
    x = state['initial']
    polynomial = np.polynomial.Polynomial([tilt, -1, 0, 1])
    for frame in state['frames']:
        close(frame['x'], x)
        close(frame['gradient'], polynomial(x))
        close(frame['value'], modules['wells']['well'](x, tilt))
        x -= state['rate']*polynomial(x)
        well_frames += 1

polynomial_terms = {
    'bowl': {(2,0):1,(0,2):1}, 'cap': {(2,0):-1,(0,2):-1},
    'saddle': {(2,0):1,(0,2):-1}, 'flatMinimum': {(4,0):1,(0,4):1},
    'flatSaddle': {(4,0):1,(0,4):-1}, 'valley': {(2,0):1},
}
for state in cases['stationary']:
    angle = math.radians(state['degrees'])
    point = state['radius']*np.array([math.cos(angle),math.sin(angle)])
    terms = polynomial_terms[state['kind']]
    actual = sum(coefficient*point[0]**i*point[1]**j for (i,j),coefficient in terms.items())
    diagonal = [2*terms.get((2,0),0),2*terms.get((0,2),0)]
    close(state['point'], point)
    close(state['actual'], actual)
    close(sorted(state['eigenvalues']), np.linalg.eigvalsh(np.diag(diagonal)))
    close(state['quadratic'], point @ np.diag(diagonal) @ point / 2)

noise_frames = 0
for state in cases['noise']:
    rate, amplitude = state['rate'], state['amplitude']
    multipliers = np.array([1-2*rate,1+2*rate])
    initial = np.array([0.6,state['initialY']])
    vector = np.array(state['vector'])
    # Closed-form weighted convolution, separate from the runtime gradient recurrence.
    for frame in state['frames']:
        k = frame['step']
        expected = multipliers**k*initial
        for i in range(k):
            expected -= rate*amplitude*signs[i]*(multipliers**(k-1-i))*vector
        close(frame['point'], expected)
        assert max(abs(np.array(frame['point']))) <= 4
        close(frame['value'], expected @ np.diag([1,-1]) @ expected)
        noise_frames += 1
    if state['stopped']:
        k=state['stopped']['attemptedStep']
        assert k == len(state['frames'])
        point=np.array(state['frames'][-1]['point'])
        proposed=multipliers*point-rate*amplitude*signs[k-1]*vector
        close(state['stopped']['point'], proposed)
        assert max(abs(proposed)) > 4

for state in cases['symmetry']:
    point = np.array([state['a'],state['b']])
    hessian = modules['factors']['hessian'](point)
    close(state['eigenvalues'], np.linalg.eigvalsh(hessian))
    close(np.linalg.norm(state['unit']), 1)
    perturbed = point+state['displacement']*np.array(state['unit'])
    close(state['loss'], modules['factors']['loss'](perturbed))
    close(state['quadratic'], state['displacement']**2*np.array(state['unit'])@hessian@state['unit']/2)
    if state['direction']=='tangent':
        close(state['loss'], state['displacement']**4/(2*np.sum(point**2)**2))
    close(state['coefficient'],1)

for state in cases['paths']:
    t=state['t']
    close(state['straightLoss'],t*t*(1-t)**2/8)
    close(state['curvedLoss'],0)
    close(state['oppositeLoss'],8*t*t*(1-t)**2)
for state in cases['interpolation']:
    predicted=modules['generalization']['predict'](state['parameter'],np.array([state['input']]))[0]
    close(state['value'],predicted)

rng=np.random.default_rng(3941)
derivative_cases=0
for _ in range(160):
    point=rng.uniform(-2,2,2)
    step=1e-5
    gradient=np.array([(modules['factors']['loss'](point+step*axis)-modules['factors']['loss'](point-step*axis))/(2*step) for axis in np.eye(2)])
    close(gradient,modules['factors']['gradient'](point),atol=3e-9)
    hessian=np.column_stack([(modules['factors']['gradient'](point+step*axis)-modules['factors']['gradient'](point-step*axis))/(2*step) for axis in np.eye(2)])
    close(hessian,modules['factors']['hessian'](point),atol=3e-9)
    derivative_cases+=1

relu_cases=0
for width in [1,2,5,9]:
    for _ in range(20):
        incoming=rng.normal(size=width)
        bias=rng.normal(size=width)
        outgoing=rng.normal(size=width)
        inputs=rng.uniform(-5,5,23)
        reference=np.array([sum(outgoing[j]*max(0,incoming[j]*x+bias[j]) for j in range(width)) for x in inputs])
        permutation=rng.permutation(width)
        scales=np.exp(rng.uniform(-2,2,width))
        network=modules['relu']['network']
        close(network(inputs,incoming,bias,outgoing),reference)
        close(network(inputs,incoming[permutation],bias[permutation],outgoing[permutation]),reference)
        close(network(inputs,incoming*scales,bias*scales,outgoing/scales),reference)
        relu_cases+=1

ensemble_sequences=0
for rate in [0.03,0.1,0.22]:
    for amplitude in [0.05,0.3]:
        for vector in [[1,0],[0,1],[0.6,0.8]]:
            values=[]
            for sequence in itertools.product([-1,1],repeat=7):
                end=modules['noise']['noisy_saddle']([0.6,0],rate,amplitude,vector,sequence)[-1]
                values.append(end[1])
                ensemble_sequences+=1
            expected=(rate*amplitude*vector[1])**2*sum((1+2*rate)**(2*j) for j in range(7))
            close(np.mean(values),0)
            close(np.mean(np.array(values)**2),expected)

diagnostic_cases=0
for preactivation in [-3,-0.1,0.2,0.5,1,1.5]:
    for rate in [0.01,0.1,0.25]:
        rows=modules['dead']['train_unit']([0,preactivation],rate,10)
        for step,prediction,loss,norm,active in rows:
            expected=max(0,preactivation) if preactivation<=0 else 1+(preactivation-1)*(1-2*rate)**step
            close(prediction,expected)
            close(loss,(expected-1)**2/2)
            assert active == (preactivation>0)
        diagnostic_cases+=1

# Changed practice: same Hessian/opposite quartic directions, alternate path, exact two-step signs.
assert (0.2**2+0.1**4)>0 and (0**2-0.1**4)<0
maximum=minimize_scalar(lambda t:-0.5*((1+2*t)*(1-2*t/3)-1)**2,bounds=(0,1),method='bounded')
close(maximum.x,0.5,atol=1e-7);close(-maximum.fun,1/18)
close(modules['noise']['noisy_saddle']([0,0],0.1,0.2,[0,1],[1,-1])[-1],[0,-0.004])
close(modules['noise']['noisy_saddle']([0,0],0.1,0.2,[1,0],[1,-1])[-1],[0.004,0])
assert 1.25*max(0,8*1-3)==6.25
close(4+1/4,4.25)
print(json.dumps({'numpy':np.__version__,'wellFrames':well_frames,'noiseFrames':noise_frames,'independentFiniteDifferenceCases':derivative_cases,'arbitraryReLUNetworks':relu_cases,'exactChangedNoiseSequences':ensemble_sequences,'diagnosticRuns':diagnostic_cases,'practiceChecks':7,'status':'passed'}))
