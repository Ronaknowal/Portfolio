"""Bounded independent author calculations; does not refit any model."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
saved = json.loads((HERE / 'calculated-inputs.json').read_text())


def linear_parameters(inputs, outputs):
    # Independent framework tensor count for the algebraic head budget.
    with torch.device('meta'):
        layer = nn.Linear(inputs, outputs)
    return sum(p.numel() for p in layer.parameters())


def check_head(c, h, w, k, d=4096):
    direct = linear_parameters(c*h*w, k)
    gap = linear_parameters(c, k)
    deep = sum([linear_parameters(c*h*w,d),linear_parameters(d,d),
                linear_parameters(d,k)])
    assert direct == (c*h*w+1)*k and gap == (c+1)*k
    return {'shape':[c,h,w],'classes':k,'direct':direct,'gap':gap,'two_hidden':deep}


def cam_by_loops(features, weights, bias):
    c,h,w = features.shape
    cells = [[sum(float(weights[k])*float(features[k,i,j]) for k in range(c))
              for j in range(w)] for i in range(h)]
    score = sum(sum(row) for row in cells)/(h*w)+bias
    return cells, score


def gate_cells(a,b):
    hidden = max(np.mean(a)-np.mean(b),0)
    return [1/(1+math.exp(-hidden)),1/(1+math.exp(hidden))]


out = {'head_cases':[check_head(512,7,7,1000),check_head(128,7,7,7),
                     check_head(1,1,1,1),check_head(16,1,1,3),
                     check_head(16,3,5,3)]}
assert out['head_cases'][0]['two_hidden'] == 123642856
assert out['head_cases'][1]['direct'] == 43911
assert out['head_cases'][1]['gap'] == 903
assert out['head_cases'][2]['direct'] == out['head_cases'][2]['gap'] == 2
maps = np.array(saved['exact']['cam']['features'])
new_weights = np.array([-1.,2.])
new_map, before = cam_by_loops(maps,new_weights,-.5)
edited = maps.copy()
edited[0,0,0] = 5
edited_map, after = cam_by_loops(edited,new_weights,-.5)
assert before == 0 and after == -1
out['changed_cam_practice'] = {'map':new_map,'score':before,
                               'edited_map':edited_map,'edited_score':after}
actual_errors = []
for row in saved['fits']:
    if row['seed'] != 1:
        continue
    for observation in row['observations']:
        f = np.array(observation['feature_maps'])
        for k in range(10):
            _,value = cam_by_loops(f,row['head_weight'][k],row['head_bias'][k])
            actual_errors.append(abs(value-observation['logits'][k]))
out['independent_python_cam_vs_saved_float32_max_error'] = max(actual_errors)
assert max(actual_errors) < 1e-5
a = np.array([[0.,4.],[2.,2.]])
b = np.array([[1.,0.],[2.,1.]])
changed = b.copy()
changed[0,1] = 4
out['gate_cells'] = {'base':gate_cells(a,b),'changed':gate_cells(a,changed),
                     'permuted':gate_cells(a,b[::-1,::-1]),
                     'equal_shift':gate_cells(a+1,b+1)}
assert out['gate_cells']['base'] == out['gate_cells']['permuted']
assert out['gate_cells']['base'] == out['gate_cells']['equal_shift']
out['scaling'] = [{'d':d,'w':w,'r':r,'parameters':d*w*w,'macs':d*w*w*r*r}
                  for d,w,r in [(2,1,1),(1,math.sqrt(2),1),(1,1,math.sqrt(2)),
                                (1,1,1),(1,1,2),(1.5,math.sqrt(2/(1.5*1.25**2)),1.25)]]
costs = {r['kind']:r['cost'] for r in saved['fits'] if r['seed']==1}
out['eligibility'] = []
for parameters,macs in [(5000,60000),(4650,76192),(2502,41920),(2501,41920),(2502,41919)]:
    eligible = [k for k,v in costs.items() if v['parameters']<=parameters and v['macs']<=macs]
    out['eligibility'].append({'parameter_budget':parameters,'mac_budget':macs,'eligible':eligible})
assert out['eligibility'][0]['eligible'] == ['parallel','inverted_gated']
assert out['eligibility'][2]['eligible'] == ['parallel']
assert out['eligibility'][3]['eligible'] == out['eligibility'][4]['eligible'] == []
out['practice_counts'] = {'two3_middle64':2*9*32*64,'one5_same32':25*32*32,
                          'projection24to48_weights':24*48,'projection_macs':24*48*8*8}
(HERE/'author-check-results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
