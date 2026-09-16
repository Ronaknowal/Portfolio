"""Small exact memory, recurrence and latent-attention calculations.

Python 3.12+ and NumPy. Run beside the lesson; writes checked-results.json.
These are transparent operators, not trained replicas of named large models.
"""
from pathlib import Path
import json
import math
import numpy as np


def softmax(scores):
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def attention(query, keys, values, allowed=None, bias=None):
    scores = query @ keys.T / math.sqrt(keys.shape[1])
    if bias is not None:
        scores = scores + bias
    if allowed is not None:
        if not np.all(np.any(allowed, axis=-1)):
            raise ValueError('Every query needs at least one legal key')
        scores = np.where(allowed, scores, -np.inf)
    weights = softmax(scores)
    return weights @ values, weights


def segmented_attention(keys, values, segment_length, memory_length, distance_penalty=0):
    outputs, traces = [], []
    for start in range(0, len(keys), segment_length):
        end = min(start + segment_length, len(keys))
        visible = np.arange(max(0, start-memory_length), end)
        queries_at = np.arange(start, end)
        legal = visible[None, :] <= queries_at[:, None]
        relative_distance = queries_at[:, None] - visible[None, :]
        output, weights = attention(
            np.ones((end-start, 1)), keys[visible], values[visible], legal,
            -distance_penalty * relative_distance)
        outputs.extend(output[:, 0].tolist())
        traces.append({'queries': queries_at.tolist(), 'keys': visible.tolist(),
                       'legal': legal.tolist(), 'weights': weights.tolist(),
                       'outputs': output[:, 0].tolist()})
    return {'outputs': outputs, 'segments': traces}


def gated_recurrence(inputs, input_gates, recurrence_gates, base=.8, exponent=8):
    state = 0.
    history = []
    for value, input_gate, recurrence_gate in zip(inputs, input_gates, recurrence_gates):
        decay = base ** (exponent*recurrence_gate)
        injection = math.sqrt(max(0, 1-decay*decay))*input_gate*value
        state = decay*state + injection
        history.append({'decay': decay, 'injection': injection, 'state': state})
    return history


def main():
    keys = np.array([[math.log(9)], [0], [0], [0], [0]], float)
    values = np.array([[6], [1], [8], [2], [0]], float)
    segment_cases = {
        str(memory): segmented_attention(keys, values, 2, memory)
        for memory in [0, 2, 4]
    }
    full = segmented_attention(keys, values, 5, 0)
    assert np.allclose(full['outputs'], segment_cases['4']['outputs'])
    changed_future = values.copy()
    changed_future[4] = 9
    future_case = segmented_attention(keys, changed_future, 2, 4)
    assert np.allclose(future_case['outputs'][:4], full['outputs'][:4])
    weights_output, weights = attention(np.ones((1,1)), np.log([[2],[1],[3]]),
                                       np.array([[2],[4],[8]]), np.array([[True,True,False]]))
    inputs = [1,0,0,0,0,0]
    ordinary = gated_recurrence(inputs, [1]*6, [1/8]*6)
    hold = gated_recurrence(inputs, [1]*6, [1/8,0,0,0,0,0])
    near_hold = gated_recurrence(inputs, [1]*6, [1/8,.001,.001,.001,.001,.001])
    zero = gated_recurrence([0]*6,[1]*6,[1/8]*6)
    recurrence = {'ordinary':ordinary, 'hold_limit':hold, 'near_hold':near_hold,
                  'zero_input':zero, 'same_input_gate_only':gated_recurrence(inputs,[1,0,0,0,0,0],[1/8]*6)}
    position_keys = np.array([[-1.],[0.],[1.]])
    data_values = np.array([[2.],[6.],[10.]])
    latent_query = np.array([[-math.log(2)],[math.log(2)]])
    latent, latent_weights = attention(latent_query,position_keys,data_values)
    perm=[2,0,1]
    permuted,_=attention(latent_query,position_keys[perm],data_values[perm])
    assert np.allclose(latent,permuted)
    reattached,_=attention(latent_query,position_keys,data_values[perm])
    uniform,_=attention(np.zeros((1,1)),position_keys,data_values)
    collision,_=attention(np.zeros((1,1)),position_keys,np.array([[0.],[6.],[12.]]))
    assert np.allclose(uniform,collision)
    n, width, depth, memory, window, latent_count = 4096, 64, 8, 128, 128, 32
    result = {'masked_attention':{'weights':weights.tolist(),'output':weights_output.tolist()},
              'segments':segment_cases,'full_attention':full,
              'future_edit_earlier_outputs':future_case['outputs'][:4],
              'relative_distances':{'query_global_5_keys_1_5':[4,0],
                                    'shifted_query_105_keys_101_105':[4,0]},
              'recurrence':recurrence,
              'latents':{'query':latent_query.tolist(),'keys':position_keys.tolist(),
                  'values':data_values.tolist(),'weights':latent_weights.tolist(),
                  'output':latent.tolist(),'paired_permutation':permuted.tolist(),
                  'reattached_position_output':reattached.tolist(),
                  'uniform_mean':float(uniform[0,0]),'uniform_collision_mean':float(collision[0,0])},
              'cost':{'tokens':n,'width':width,'layers':depth,'memory':memory,'window':window,
                  'latents':latent_count,'dense_score_cells_per_layer':n*n,
                  'chunk_score_cells_L256_M128':(n//256)*256*(256+memory),
                  'local_score_slots_per_layer':n*window,
                  'one_latent_read_plus_8_latent_layers':n*latent_count+8*latent_count**2,
                  'full_mha_kv_bytes_one_layer_fp16':2*n*width*2,
                  'bounded_mha_kv_bytes_one_layer_fp16':2*window*width*2},
              'practice':{'masked_two_keys':float((3*7+1*3)/4),
                  'decay_three_steps':.75**3*2,'latent_reverse_values':reattached.tolist(),
                  'kv_bytes_layers12_heads8_head_dim32_tokens2048_fp16':2*12*2048*8*32*2}}
    Path(__file__).with_name('checked-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'segment_last_outputs':{k:v['outputs'][-1] for k,v in segment_cases.items()},
                      'recurrence_last':{k:v[-1]['state'] for k,v in recurrence.items()},
                      'latent_outputs':latent.tolist(),'cost':result['cost']},indent=2))


if __name__ == '__main__':
    main()
