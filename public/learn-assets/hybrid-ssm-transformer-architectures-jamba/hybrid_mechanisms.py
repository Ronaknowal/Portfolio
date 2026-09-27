"""Independent, exact small examples for the hybrid architecture manuscript."""
from pathlib import Path
import json, math
import numpy as np

HERE=Path(__file__).resolve().parent

def memory_read(values,keys,query,decay=.5,beta=math.log(9)):
    values=np.asarray(values,dtype=float);keys=np.asarray(keys)
    scores=np.array([beta if key==query else 0 for key in keys])
    weights=np.exp(scores-scores.max());weights/=weights.sum()
    numerator=normalizer=0.;states=[]
    for value in values:
        numerator=decay*numerator+value;normalizer=decay*normalizer+1
        states.append([numerator,normalizer,numerator/normalizer])
    return {'states':states,'weights':weights.tolist(),'attention':float(weights@values),
            'summary':states[-1][-1]}

def cache_bytes(length,batch=1,layers=32,attention_layers=4,width=4096,
                kv_heads=8,head_width=128,state_size=16,expand=2,
                conv_width=4,kv_bytes=2,state_bytes=4,conv_bytes=2,window=None):
    retained=length if window is None else min(length,window)
    kv=batch*attention_layers*2*kv_heads*head_width*retained*kv_bytes
    recurrent=batch*(layers-attention_layers)*width*expand*state_size*state_bytes
    convolution=batch*(layers-attention_layers)*width*expand*conv_width*conv_bytes
    return {'kv':kv,'recurrent':recurrent,'convolution':convolution,
            'total':kv+recurrent+convolution,'total_GiB':(kv+recurrent+convolution)/2**30}

def route(logits,expert_values,k=2,renormalize=False):
    logits=np.asarray(logits);p=np.exp(logits-logits.max());p/=p.sum()
    indices=sorted(range(len(p)),key=lambda i:(-p[i],i))[:k]
    weights=p[indices].copy()
    if renormalize:weights/=weights.sum()
    return {'probabilities':p.tolist(),'selected':indices,'weights':weights.tolist(),
            'selected_mass':float(p[indices].sum()),
            'output':(weights@np.asarray(expert_values)[indices]).tolist(),
            'boundary_tie':bool(k<len(p) and p[indices[-1]]==sorted(p,reverse=True)[k])}

def run():
    keys=['A','B','C']
    expert_values=[[2,-1],[0,3],[-2,0],[1,1]]
    logits=[math.log(4),math.log(2),0,0]
    record={'reads':{
        'worked':memory_read([2,7,4],keys,'A'),
        'collision':memory_read([6,5,4],keys,'A'),
        'uniform_query':memory_read([2,7,4],keys,'absent'),
        'constant_null':memory_read([4,4,4],keys,'A'),
        'fresh':memory_read([3,8,1,5],['A','B','A','C'],'A',.75,math.log(4)),
        'fresh_changed_key':memory_read([3,8,1,5],['A','A','A','C'],'A',.75,math.log(4))},
        'memory':[], 'routing':{
        'worked':route(logits,expert_values),
        'renormalized':route(logits,expert_values,renormalize=True),
        'changed_selected':route([math.log(4),math.log(2),math.log(8),0],expert_values),
        'fresh':route([0,math.log(3),math.log(6),math.log(2)],[[1,2],[3,0],[-1,4],[2,-2]]),
        'fresh_changed_unselected':route([math.log(2),math.log(3),math.log(6),math.log(2)],[[1,2],[3,0],[-1,4],[2,-2]]),
        'zero_outputs_null':route(logits,[[0,0]]*4),
        'ties':route([0,0,0,0],expert_values)}}
    for length in [0,1024,4096,16384,65536,262144]:
        record['memory'].append({'length':length,'hybrid':cache_bytes(length),
            'attention':cache_bytes(length,attention_layers=32),
            'ssm':cache_bytes(length,attention_layers=0),
            'window_hybrid':cache_bytes(length,window=4096)})
    record['fresh_memory']=cache_bytes(2048,batch=3,layers=12,attention_layers=3,width=512,
        kv_heads=2,head_width=64,state_size=8,expand=2,conv_width=4)
    record['fresh_memory_double']=cache_bytes(4096,batch=3,layers=12,attention_layers=3,width=512,
        kv_heads=2,head_width=64,state_size=8,expand=2,conv_width=4)
    one=3*4096*14336
    record['ffn']={'one_expert':one,'total':(16+16*16)*one,'active':(16+16*2)*one,
                   'baseline_dense':32*one,'full_bf16_weight_bytes_using_rounded_52B':104_000_000_000}
    record['attention_arithmetic']=[{'length':length,
        'projection_flops':4*length*4096*(4096+1024),
        'ideal_causal_pair_flops':2*4096*length*(length+1),
        'one_new_token_pair_flops':4*4096*(length+1)} for length in [1024,32768,262144]]
    (HERE/'mechanism-results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':run()
