"""Bounded author evidence; no browser or production implementation checks."""
import json, hashlib
import numpy as np
import torch
from torch.nn import functional as F
from stroke_models import HERE, data_split, load_fit, StrokeModel
from inspect_stroke import run_trace

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
data,split=data_split();saved=json.loads((HERE/'stroke-results.json').read_text())
report={'reload_validation_max_difference':{},'fixtures':{}}
for fit in saved['fits']:
    model=load_fit(fit['key'])
    with torch.no_grad():actual=model(data['validation'][0]).numpy()
    report['reload_validation_max_difference'][fit['key']]=float(np.max(np.abs(actual-np.array(fit['validation_logits']))))
rows=np.loadtxt(HERE/'pendigits.tra',delimiter=',')
for name,index in [('worked',0),('fresh',31)]:
    row_id=split['ids_1_based']['validation'][index];coordinates=rows[row_id-1,:16].reshape(8,2)
    report['fixtures'][name]={'source_id':row_id,'label':int(rows[row_id-1,-1]),
        'traces':{mode:run_trace(coordinates,mode=mode) for mode in
                  ['carry','recurrent-reset','kv-reset','convolution-reset','position-reset']}}
    edited=coordinates.copy();edited[3]=[50,50]
    report['fixtures'][name]['edited_point_4_50_50']=run_trace(edited)
    report['fixtures'][name]['no_boundary_history_null']=run_trace(coordinates,break_at=0,mode='recurrent-reset')
    report['fixtures'][name]['no_suffix_null']=run_trace(coordinates,break_at=8,mode='kv-reset')
report['fixtures']['constant_coordinates']=run_trace(np.full((8,2),50))
model=load_fit('MAM-37').double();x=data['validation'][0][:2].double().requires_grad_(True)
full=model(x,True);loss=full.square().mean();full_grad=torch.autograd.grad(loss,[x,*model.parameters()])
stream,_=model.stream(x);loss=stream.square().mean();step_grad=torch.autograd.grad(loss,[x,*model.parameters()])
report['double_precision_full_stream']={'output':float((full-stream).abs().max().detach()),
    'input_gradient':float((full_grad[0]-step_grad[0]).abs().max()),
    'parameter_gradient':max(float((a-b).abs().max()) for a,b in zip(full_grad[1:],step_grad[1:]))}
with torch.no_grad():
    x=data['validation'][0][:10].double();changed=x.clone();changed[:,4:]=-.75
    report['causal_prefix_future_edit']=float((model(x,True)[:,:4]-model(changed,True)[:,:4]).abs().max())
    report['causal_prefix_short_input']=float((model(x,True)[:,:4]-model(x[:,:4],True)).abs().max())
report['program_sha256']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for name in ['stroke_models.py','inspect_stroke.py','hybrid_mechanisms.py']}
(HERE/'author-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'reload':report['reload_validation_max_difference'],'double':report['double_precision_full_stream'],
    'future':report['causal_prefix_future_edit'],'prefix':report['causal_prefix_short_input'],
    'fixtures':{key:{'row':f['source_id'],'label':f['label'],'outputs':{mode:{'prediction':t['branch_prediction'],
    'probability_of_label':t['branch_probabilities'][f['label']], 'difference':t['maximum_logit_difference']}
    for mode,t in f['traces'].items()}} for key,f in report['fixtures'].items() if 'source_id' in f}},indent=2))
