"""Exact tiny operators for RWKV and related linear memories. NumPy only."""
from pathlib import Path
import json
import numpy as np

def positive_matrix(query, key, value):
    weights = np.tril(query @ key.T)
    denominator = weights.sum(1)
    if np.any(denominator <= 0):
        raise ValueError("The normalized operator needs positive total weight.")
    return weights @ value / denominator[:, None]

def positive_recurrent(query, key, value, chunk_size=1):
    state = np.zeros((key.shape[1], value.shape[1]))
    normalizer = np.zeros(key.shape[1])
    result = []
    for start in range(0, len(query), chunk_size):
        q = query[start:start+chunk_size]
        k = key[start:start+chunk_size]
        v = value[start:start+chunk_size]
        local = np.tril(q @ k.T)
        numerator = q @ state + local @ v
        denominator = q @ normalizer + local.sum(1)
        if np.any(denominator <= 0):
            raise ValueError("The normalized operator needs positive total weight.")
        result.append(numerator / denominator[:, None])
        state += k.T @ v
        normalizer += k.sum(0)
    return np.concatenate(result), state, normalizer

def rwkv_direct(key, value, log_decay, current_bonus):
    outputs = []
    for t in range(len(key)):
        log_weights = key[:t+1].copy()
        log_weights[:t] += np.arange(t-1, -1, -1)[:, None] * log_decay
        log_weights[t] += current_bonus
        weights = np.exp(log_weights - log_weights.max(0))
        outputs.append((weights * value[:t+1]).sum(0) / weights.sum(0))
    return np.array(outputs)

def rwkv_stable(key, value, log_decay, current_bonus):
    numerator = np.zeros(key.shape[1])
    denominator = np.zeros(key.shape[1])
    log_scale = np.full(key.shape[1], -np.inf)
    outputs = []; history = []
    for k, v in zip(key, value):
        output_scale = np.maximum(log_scale, current_bonus+k)
        old_weight = np.exp(log_scale-output_scale)
        new_weight = np.exp(current_bonus+k-output_scale)
        outputs.append((old_weight*numerator+new_weight*v) /
                       (old_weight*denominator+new_weight))
        next_scale = np.maximum(log_scale+log_decay, k)
        old_weight = np.exp(log_scale+log_decay-next_scale)
        new_weight = np.exp(k-next_scale)
        numerator = old_weight*numerator+new_weight*v
        denominator = old_weight*denominator+new_weight
        log_scale = next_scale
        history.append([numerator.tolist(),denominator.tolist(),log_scale.tolist()])
    return np.array(outputs), history

def additive_and_delta(keys, values, rate):
    additive = np.zeros((values.shape[1], keys.shape[1]))
    delta = additive.copy(); history = []
    for key, value in zip(keys, values):
        additive += np.outer(value, key)
        residual = value-delta@key
        delta += rate*np.outer(residual,key)
        history.append({"additive":additive.tolist(),"delta":delta.tolist(),
                        "read_additive":(additive@key).tolist(),"read_delta":(delta@key).tolist()})
    return additive, delta, history

def gated_memory(query, key, value, gates):
    state = np.zeros((key.shape[1], value.shape[1])); ys = []
    for q,k,v,g in zip(query,key,value,gates):
        state = g[:,None]*state+np.outer(k,v)
        ys.append(q@state)
    return np.array(ys)

def rwkv7_step(state, decay, removal_key, rate, replacement_key, value):
    transition = np.diag(decay)-np.outer(removal_key,rate*removal_key)
    return state@transition+np.outer(value,replacement_key), transition

def main():
    q=np.array([[1.,1.],[2.,1.],[1.,2.],[3.,1.]])
    k=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.]])
    v=np.array([[2.],[8.],[-1.],[5.]])
    dense=positive_matrix(q,k,v)
    variants={str(c):float(np.max(abs(positive_recurrent(q,k,v,c)[0]-dense))) for c in [1,2,3,4,8]}
    changed=v.copy();changed[-1]=11
    fresh_v=np.array([[3.],[-2.],[7.],[1.]])
    kernel={"q":q.tolist(),"k":k.tolist(),"v":v.tolist(),"output":dense.tolist(),
            "final_state":positive_recurrent(q,k,v)[1].tolist(),"final_normalizer":positive_recurrent(q,k,v)[2].tolist(),
            "chunk_errors":variants,"last_edit_output":positive_matrix(q,k,changed).tolist(),
            "constant_value_output":positive_matrix(q,k,np.full_like(v,4)).tolist(),
            "fresh_v":fresh_v.tolist(),"fresh_output":positive_matrix(q,k,fresh_v).tolist()}
    raw_q=np.array([[1.],[2.]])
    raw_k=np.array([[0.],[1.]])
    soft_weights=np.exp(raw_q@raw_k.T-np.max(raw_q@raw_k.T,axis=1,keepdims=True))
    soft=(np.tril(soft_weights)@np.array([[0.],[10.]]))/np.tril(soft_weights).sum(1)[:,None]
    elu=positive_matrix(np.maximum(raw_q,0)+1,np.maximum(raw_k,0)+1,np.array([[0.],[10.]]))
    kernel["softmax_vs_elu"]={"softmax":soft.tolist(),"elu_plus_one":elu.tolist()}
    try:positive_matrix(np.zeros_like(q),k,v)
    except ValueError as error:kernel["zero_query"]=str(error)
    log_key=np.log(np.array([[1.],[2.],[1.],[4.]]));values=np.array([[2.],[8.],[-1.],[5.]])
    log_decay=np.array([np.log(.5)]);bonus=np.array([np.log(2.)])
    stable,history=rwkv_stable(log_key,values,log_decay,bonus)
    direct=rwkv_direct(log_key,values,log_decay,bonus)
    bonus_off,_=rwkv_stable(log_key,values,log_decay,np.zeros(1))
    fresh,_=rwkv_stable(log_key,fresh_v,log_decay,bonus)
    shifted,_=rwkv_stable(log_key+1000,values,log_decay,bonus)
    with np.errstate(over='ignore'): overflow=bool(np.isinf(np.exp(log_key+1000)).any())
    weighted={"key":log_key.tolist(),"values":values.tolist(),"output":stable.tolist(),"history":history,
              "direct_error":float(abs(stable-direct).max()),"bonus_off":bonus_off.tolist(),
              "bonus_does_not_change_state":history==rwkv_stable(log_key,values,log_decay,np.zeros(1))[1],
              "large_shift_output":shifted.tolist(),"shift_error":float(abs(stable-shifted).max()),
              "raw_exponential_overflows":overflow,"fresh_values":fresh_v.tolist(),"fresh_output":fresh.tolist(),
              "constant_value":rwkv_stable(log_key,np.full_like(values,7),log_decay,bonus)[0].tolist()}
    keys=np.array([[1.,0.],[0.,1.],[1.,0.]])
    vals=np.array([[2.],[7.],[5.]])
    add,delta,history=additive_and_delta(keys,vals,1.)
    collision=np.array([[1.,0.],[.6,.8],[1.,0.]])
    collided=additive_and_delta(collision,vals,1.)
    fresh_keys=np.array([[0.,1.],[1.,0.],[0.,1.]])
    fresh_vals=np.array([[4.],[-3.],[9.]])
    fresh_memory=additive_and_delta(fresh_keys,fresh_vals,.5)
    delta_record={"keys":keys.tolist(),"values":vals.tolist(),"history":history,
                  "additive_final":add.tolist(),"delta_final":delta.tolist(),
                  "collision_keys":collision.tolist(),"collision_history":collided[2],
                  "fresh_keys":fresh_keys.tolist(),"fresh_values":fresh_vals.tolist(),"fresh_rate":.5,
                  "fresh_history":fresh_memory[2],"zero_rate":additive_and_delta(keys,vals,0)[1].tolist()}
    state=np.array([[2.,7.],[-1.,3.]])
    next_state,transition=rwkv7_step(state,np.array([.8,.9]),np.array([1.,0.]),
                                    np.array([.6,.2]),np.array([1.,0.]),np.array([5.,2.]))
    mixed_state,mixed_transition=rwkv7_step(state,np.array([.8,.9]),np.array([.6,.8]),
                                    np.array([.6,.2]),np.array([1.,0.]),np.array([5.,2.]))
    version7={"initial":state.tolist(),"transition":transition.tolist(),"next":next_state.tolist(),
              "mixed_transition":mixed_transition.tolist(),"mixed_next":mixed_state.tolist(),
              "eigenvalues":np.linalg.eigvals(mixed_transition).tolist()}
    gates=np.full((4,2),.5)
    varying=gates.copy();varying[2]=[.1,.9]
    decay={"constant":gated_memory(q,k,v,gates).tolist(),
           "varying":gated_memory(q,k,v,varying).tolist(),
           "all_zero_values":gated_memory(q,k,np.zeros_like(v),varying).tolist()}
    report={"positive_kernel":kernel,"rwkv4":weighted,"delta":delta_record,"rwkv7":version7,"gated_memory":decay,
            "cache_bytes":{"rwkv4_12_layers_512_width_fp32":12*5*512*4,
                           "matrix_12_layers_8_heads_64_square_fp32":12*8*64*64*4,
                           "mha_12_layers_4096_tokens_8_kvheads_64_width_fp16":12*4096*2*8*64*2,
                           "gqa_same_but_2_kvheads":12*4096*2*2*64*2}}
    assert max(variants.values())<1e-12 and weighted["direct_error"]<1e-12
    assert weighted["shift_error"]<1e-10 and weighted["bonus_does_not_change_state"]
    path=Path(__file__).with_name('mechanism-results.json')
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()

