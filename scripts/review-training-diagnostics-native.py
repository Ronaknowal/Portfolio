"""Independent stateful normalization, true SGD restore, and new checkpoint-boundary probes."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
ID='neural-training-diagnostics-reproducible-experiments'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
rng=np.random.default_rng(52813)
result={'modeChains':[],'sgd':[],'checkpointBoundaries':[]}

for case in range(40):
    linear=case%8==0
    layer=nn.Linear(1,1) if linear else nn.BatchNorm1d(1)
    mean=float(rng.uniform(-5,5));variance=0. if case%7==0 else float(rng.uniform(0,25))
    with torch.no_grad():
        if linear:layer.weight.fill_(2);layer.bias.fill_(1)
        else:layer.running_mean.fill_(mean);layer.running_var.fill_(variance)
    chain=[]
    for step in range(5):
        values=rng.uniform(-10,10,2).tolist()
        if case%6==0:values[1]=values[0]
        if case%9==0:values=[1.,1.+1e-6]
        training=(case+step)%3!=0
        grad_enabled=(case+step)%2==0
        before_mean=mean if linear else layer.running_mean.item()
        before_var=variance if linear else layer.running_var.item()
        layer.train(training)
        with torch.set_grad_enabled(grad_enabled):output=layer(torch.tensor(values).reshape(2,1))
        chain.append({'input':{'values':values,'mean':before_mean,'variance':before_var,'training':training,'gradEnabled':grad_enabled,'module':'linear' if linear else 'batchnorm'},'output':output.detach().flatten().tolist(),'mean':None if linear else layer.running_mean.item(),'variance':None if linear else layer.running_var.item(),'graph':output.requires_grad})
    result['modeChains'].append(chain)

def optimizer_walk(w,opt,target,count,start,previous_velocity=0.):
    trace=[]
    velocity=previous_velocity
    for i in range(count):
        before=w.item()
        opt.zero_grad(set_to_none=True)
        loss=(w-target).square()/2
        loss.backward()
        gradient=w.grad.item()
        old_velocity=velocity
        opt.step()
        velocity=opt.state[w]['momentum_buffer'].item() if opt.param_groups[0]['momentum'] else gradient
        trace.append({'index':start+i,'weight':before,'velocity':old_velocity,'gradient':gradient,'loss':loss.item(),'nextVelocity':velocity,'nextWeight':w.item()})
    return trace,velocity

for case in range(80):
    target=float(rng.uniform(-5,5));initial=float(rng.uniform(-5,5))
    rate=0. if case%9==0 else float(rng.uniform(.01,.5))
    momentum=0. if case%7==0 else float(rng.uniform(.01,.95))
    save=1+case%5;steps=1+(case//5)%5
    if case%13==0:initial=target
    w=nn.Parameter(torch.tensor(initial))
    opt=torch.optim.SGD([w],lr=rate,momentum=momentum)
    prefix,velocity=optimizer_walk(w,opt,target,save,1)
    checkpoint={'weight':w.item(),'velocity':velocity}
    stream=io.BytesIO();torch.save(copy.deepcopy(opt.state_dict()),stream);stream.seek(0)
    saved_optimizer=torch.load(stream,weights_only=True)
    full,_=optimizer_walk(w,opt,target,steps,save+1,velocity)
    restored_w=nn.Parameter(torch.tensor(checkpoint['weight']))
    restored=torch.optim.SGD([restored_w],lr=rate,momentum=momentum)
    restored.load_state_dict(saved_optimizer)
    restored_trace,_=optimizer_walk(restored_w,restored,target,steps,save+1,velocity)
    assert restored_trace==full
    reset_w=nn.Parameter(torch.tensor(checkpoint['weight']))
    reset_opt=torch.optim.SGD([reset_w],lr=rate,momentum=momentum)
    reset,_=optimizer_walk(reset_w,reset_opt,target,steps,save+1)
    result['sgd'].append({'input':{'target':target,'weight':initial,'rate':rate,'momentum':momentum,'save':save,'steps':steps},'prefix':prefix,'checkpoint':checkpoint,'full':full,'reset':reset,'finalFull':w.item(),'finalReset':reset_w.item()})

sys.path.insert(0,str(PACKET))
import checkpoint_replay as replay
features,labels,split,_,_=replay.load_inputs()
features=features[split['train']];labels=labels[split['train']]
for boundary in [2,6,8]:
    state=replay.fresh_training_state()
    order=torch.randperm(len(features),generator=state[3])
    order,cursor,_=replay.train_until(state,features,labels,order,0,0,boundary)
    checkpoint=replay.snapshot(state,order,cursor,boundary)
    saved_model=copy.deepcopy(checkpoint['model'])
    stream=io.BytesIO();torch.save(checkpoint,stream);stream.seek(0);checkpoint=torch.load(stream,weights_only=True)
    _,_,reference=replay.train_until(state,features,labels,order,cursor,boundary,13)
    assert all(torch.equal(saved_model[k],checkpoint['model'][k]) for k in saved_model)
    comparisons=[]
    for omission in ['none','optimizer_buffers','torch_rng','active_order_cursor','scheduler']:
        fresh,new_order,new_cursor=replay.restore(checkpoint,omission)
        _,_,trace=replay.train_until(fresh,features,labels,new_order,new_cursor,boundary,13)
        exact=all(torch.equal(p,q) for p,q in zip(state[0].parameters(),fresh[0].parameters()))
        first=None
        for a,b in zip(reference,trace):
            for field,title in [('training_positions','batch positions'),('pre_update_loss','pre-update loss'),('learning_rate_used','learning rate used')]:
                if a[field]!=b[field]:first={'update':a['update'],'quantity':title};break
            if first:break
        # At boundary 6 the order is exhausted and the scheduler phase is a full
        # period. Regenerating the next order / restarting that phase is a null.
        expected_null=omission=='none' or (boundary==6 and omission in ['scheduler','active_order_cursor'])
        if expected_null:assert exact and trace==reference and first is None,(boundary,omission)
        else:assert not exact and first is not None,(boundary,omission)
        comparisons.append({'omission':omission,'trace':trace,'first':first,'finalExact':exact})
    result['checkpointBoundaries'].append({'boundary':boundary,'cursor':cursor,'reference':reference,'comparisons':comparisons})

result['environment']={'torch':torch.__version__,'numpy':np.__version__,'threads':1,'dtype':'float64','device':'CPU'}
(OUT/'independent-native-fixtures.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print(json.dumps({'modeChains':40,'nativeModeTransitions':200,'trueSGDRestores':80,'newWineCheckpointBoundaries':[2,6,8],'limits':'Short new checkpoint probes only; no repeated eight-fit campaign or test evaluation.'}))
