"""Exact small SSM computations, NumPy/SciPy; all fixtures are original.

Run this file to write mechanism-results.json. It implements transparent operators,
not optimized S4/Mamba kernels or learned replicas of large language models.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.linalg import expm


def held_discretize(A, B, interval):
    """Block exponential is valid even when A is singular."""
    n, channels = B.shape
    block = np.zeros((n + channels, n + channels), dtype=np.result_type(A, B, float))
    block[:n, :n] = A
    block[:n, n:] = B
    lifted = expm(interval * block)
    return lifted[:n, :n], lifted[:n, n:]


def bilinear_discretize(A, B, interval):
    identity = np.eye(len(A))
    left = identity - interval * A / 2
    return np.linalg.solve(left, identity + interval * A / 2), np.linalg.solve(left, interval * B)


def recurrent(A, B, C, D, inputs, initial=None):
    state = np.zeros(len(A)) if initial is None else np.array(initial, dtype=float)
    outputs, states = [], []
    for value in inputs:
        state = A @ state + B[:, 0] * value
        outputs.append(float(C @ state + D * value))
        states.append(state.tolist())
    return np.array(outputs), states


def kernel(A, B, C, length):
    state = B[:, 0].copy()
    taps = []
    for _ in range(length):
        taps.append(float(C @ state))
        state = A @ state
    return np.array(taps)


def fft_convolve(inputs, taps):
    size = len(inputs) + len(taps) - 1
    return np.fft.irfft(np.fft.rfft(inputs, size) * np.fft.rfft(taps, size), size)[:len(inputs)]


def selective_scalar(inputs, gates):
    state = 0.
    states = []
    for value, gate in zip(inputs, gates):
        state = (1 - gate) * state + gate * value
        states.append(state)
    return states


def ssd_recurrent(decay, write, read, values, initial=None):
    state = np.zeros((write.shape[1], values.shape[1])) if initial is None else initial.copy()
    outputs, states = [], []
    for a, b, c, value in zip(decay, write, read, values):
        state = a * state + np.outer(b, value)
        outputs.append(c @ state)
        states.append(state.copy())
    return np.array(outputs), np.array(states)


def decay_matrix(decay):
    length = len(decay)
    result = np.eye(length)
    for i in range(length):
        for j in range(i):
            result[i, j] = np.prod(decay[j + 1:i + 1])
    return result


def ssd_chunked(decay, write, read, values, chunk_size):
    """Four-step state passing; small matmuls and a serial chunk-boundary scan."""
    final_states, chunk_products, locals_, boundaries = [], [], [], []
    for start in range(0, len(decay), chunk_size):
        stop = min(start + chunk_size, len(decay))
        a, b, c, v = decay[start:stop], write[start:stop], read[start:stop], values[start:stop]
        local = (decay_matrix(a) * (c @ b.T)) @ v
        end_weights = np.array([np.prod(a[j+1:]) for j in range(len(a))])
        final_states.append((b * end_weights[:, None]).T @ v)
        chunk_products.append(np.prod(a))
        locals_.append(local)
        boundaries.append((start, stop))
    carry = np.zeros_like(final_states[0])
    outputs, traces = [], []
    for (start, stop), local, product, own_final in zip(boundaries, locals_, chunk_products, final_states):
        initial_part = np.cumprod(decay[start:stop])[:, None] * (read[start:stop] @ carry)
        outputs.extend(local + initial_part)
        traces.append({'start':start,'stop':stop,'local':local.tolist(),
                       'initial_contribution':initial_part.tolist(),'incoming_state':carry.tolist(),
                       'own_final_state':own_final.tolist(),'decay_product':float(product)})
        carry = product * carry + own_final
    return np.array(outputs), traces


def main():
    A = np.diag([-1., -2.]); B = np.ones((2, 1)); C = np.array([1., -.5])
    interval = math.log(2); Ad, Bd = held_discretize(A, B, interval)
    inputs = np.array([2., 0., 1., 0.]); D = .25
    yr, states = recurrent(Ad, Bd, C, D, inputs)
    taps = kernel(Ad, Bd, C, len(inputs))
    yd = np.convolve(inputs, taps)[:len(inputs)] + D * inputs
    yf = fft_convolve(inputs, taps) + D * inputs
    assert np.allclose(yr, yd) and np.allclose(yr, yf)
    initial = np.array([1., -2.])
    yi, _ = recurrent(Ad, Bd, C, D, inputs, initial)
    response = np.array([C @ np.linalg.matrix_power(Ad, t+1) @ initial for t in range(len(inputs))])
    assert np.allclose(yi, yr + response)
    integratorA, integratorB = held_discretize(np.zeros((1,1)), np.ones((1,1)), .5)
    integrator, _ = recurrent(integratorA, integratorB, np.ones(1), 0, [2.,-1.], [3.])
    feedthrough, _ = recurrent(Ad, Bd, np.zeros(2), 1., inputs, initial)
    assert np.allclose(feedthrough, inputs)
    bilinearA, bilinearB = bilinear_discretize(A, B, interval)
    # A rotation plus decay, represented in two real coordinates.
    oscillatorA = np.array([[-.2, -2.],[2.,-.2]])
    oscillatorD, _ = held_discretize(oscillatorA,np.zeros((2,1)),.25)
    _, oscillator_states = recurrent(oscillatorD,np.zeros((2,1)),np.array([1.,0.]),0.,np.zeros(12),[1.,0.])
    # LegS normal-plus-rank-one identity, computed directly rather than guessed.
    N=4; index=np.arange(N); root=np.sqrt(2*index+1)
    hippo=-np.tril(root[:,None]*root[None,:],k=-1)-np.diag(index+1)
    p=np.sqrt(index+.5); normal=hippo+np.outer(p,p)
    assert np.allclose(normal+normal.T,-np.eye(N))
    normal_eigenvalues=np.linalg.eigvals(normal)
    # Uniform [0,t] projection of f(s)=s onto 1 and sqrt(3)(2s/t-1).
    t=2.; coefficients=[t/2,t*math.sqrt(3)/6]
    locations=np.linspace(0,t,5)
    reconstructed=coefficients[0]+coefficients[1]*math.sqrt(3)*(2*locations/t-1)
    assert np.allclose(reconstructed,locations)
    # A finite fixed delay is exactly LTI, via a shift register.
    delayA=np.zeros((3,3));delayA[1,0]=1;delayA[2,1]=1
    delayed,_=recurrent(delayA,np.array([[1.],[0.],[0.]]),np.array([0.,0.,1.]),0.,[2,5,-1,7,0])
    assert np.allclose(delayed,[0,0,2,5,-1])
    values=[4.,9.,-7.,6.]; gate=[.99,.01,.01,.99]
    chosen=selective_scalar(values,gate); fixed=selective_scalar(values,[.5]*4)
    zero=selective_scalar([0.]*4,gate)
    z=.7;delta=np.logaddexp(0,z);g=1/(1+np.exp(-z))
    assert np.allclose([np.exp(-delta),-np.expm1(-delta)],[1-g,g])
    # Complete SSD fixture: state N2 by value P2, four positions.
    decay=np.array([.5,.5,.25,.8]);write=np.array([[1,0],[0,1],[1,1],[1,-1.]],float)
    read=np.array([[1,0],[1,1],[0,1],[1,2.]],float)
    value=np.array([[2,1],[3,-1],[1,2],[-2,1.]],float)
    ssd, ssd_states=ssd_recurrent(decay,write,read,value)
    L=decay_matrix(decay);M=L*(read@write.T);dense=M@value
    chunked,trace=ssd_chunked(decay,write,read,value,2)
    assert np.allclose(ssd,dense) and np.allclose(ssd,chunked)
    assert all(np.allclose(ssd,ssd_chunked(decay,write,read,value,q)[0]) for q in [1,3,4,8])
    edit=value.copy();edit[-1]=[8.,-3.]
    future,_=ssd_recurrent(decay,write,read,edit)
    assert np.allclose(ssd[:-1],future[:-1])
    no_write=np.zeros_like(write);null,_=ssd_recurrent(decay,no_write,read,value)
    assert np.allclose(null,0)
    reset_decay=decay.copy();reset_decay[2]=0
    reset,_=ssd_recurrent(reset_decay,write,read,value)
    # A second-endpoint term and its Mamba-2 limiting case.
    previous_state=1.;previous_input=2.;current_input=6.;a=.5;step=1.
    two_endpoint=a*previous_state + .5*step*a*previous_input + .5*step*current_input
    endpoint_one=a*previous_state + step*current_input
    # Rotation parity is an isolated exact mechanism, not a trained Mamba-3.
    bits=[1,0,1,1];rotation_state=np.array([1.,0.]);parity=[]
    for bit in bits:
        angle=np.pi*bit
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rotation_state=R@rotation_state
        parity.append(float((1-rotation_state[0])/2))
    assert np.allclose(parity,np.cumsum(bits)%2)
    output={'versions':{'numpy':np.__version__},'lti':{'A':A.tolist(),'B':B.tolist(),'C':C.tolist(),'D':D,'interval':interval,
        'Ad':Ad.tolist(),'Bd':Bd.tolist(),'input':inputs.tolist(),'states':states,'kernel':taps.tolist(),
        'recurrent':yr.tolist(),'direct':yd.tolist(),'fft':yf.tolist(),'nonzero_initial':initial.tolist(),
        'initial_response':response.tolist(),'with_initial':yi.tolist(),'feedthrough_only':feedthrough.tolist(),
        'bilinear_Ad':bilinearA.tolist(),'bilinear_Bd':bilinearB.tolist()},
        'singular_integrator':{'initial':3.,'interval':.5,'inputs':[2,-1],'outputs':integrator.tolist()},
        'oscillation':{'A':oscillatorA.tolist(),'interval':.25,'initial':[1,0],'states':oscillator_states},
        'hippo':{'A':hippo.tolist(),'B':root.tolist(),'p':p.tolist(),'normal':normal.tolist(),
                 'normal_eigenvalues':[[z.real,z.imag] for z in normal_eigenvalues],
                 'projection_t':t,'coefficients':coefficients,'locations':locations.tolist(),'reconstruction':reconstructed.tolist()},
        'selection':{'input':values,'gates':gate,'selected_states':chosen,'fixed_states':fixed,'zero_input_states':zero,
                     'fixed_delay_output':delayed.tolist(),'z':z,'delta_softplus':float(delta),'sigmoid_gate':float(g)},
        'ssd':{'decay':decay.tolist(),'write':write.tolist(),'read':read.tolist(),'values':value.tolist(),'L':L.tolist(),
               'M':M.tolist(),'states':ssd_states.tolist(),'recurrent':ssd.tolist(),'matrix':dense.tolist(),
               'chunked':chunked.tolist(),'chunk_trace':trace,'future_edit':future.tolist(),'zero_write':null.tolist(),
               'reset_before_index2':reset.tolist()},
        'mamba3_mechanisms':{'two_endpoint':two_endpoint,'lambda1':endpoint_one,'parity_bits':bits,'rotation_parity':parity},
        'cache_bytes':{'ssm_layers12_width64_state16_float32':12*64*16*4,
                       'mha_layers12_tokens4096_width64_float16':2*12*4096*64*2}}
    Path(__file__).with_name('mechanism-results.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'lti_output':yr.tolist(),'kernel':taps.tolist(),'singular_integrator':integrator.tolist(),
        'selection':chosen,'fixed_filter':fixed,'ssd':ssd.tolist(),'endpoints':[two_endpoint,endpoint_one],
        'parity':parity,'cache_bytes':output['cache_bytes']},indent=2))


if __name__=='__main__':
    main()
