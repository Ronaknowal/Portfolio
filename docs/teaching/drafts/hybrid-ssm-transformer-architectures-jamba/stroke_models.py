"""Small causal mixer comparison on UCI PenDigits. No Jamba weights or CUDA kernels.

Run this file with --train to reproduce all predeclared fits. With no arguments,
print the saved report without training. Requires numpy and torch.
"""
from pathlib import Path
import argparse, copy, hashlib, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
WIDTH=16
STATE=4

def data_split():
    arrays=[np.loadtxt(HERE/name,delimiter=',',dtype=np.int64)
            for name in ['pendigits.tra','pendigits.tes']]
    groups={}
    for file_index,rows in enumerate(arrays):
        for row_index,row in enumerate(rows):
            groups.setdefault(tuple(row[:-1]),[]).append((file_index,row_index,int(row[-1])))
    conflict=[members for members in groups.values() if len({m[2] for m in members})>1]
    cross=[members for members in groups.values() if len({m[0] for m in members})>1]
    eligible=[[],[]]
    for members in groups.values():
        if len({m[2] for m in members})>1 or len({m[0] for m in members})>1:
            continue
        file_index,row_index,_=members[0]
        eligible[file_index].append(row_index)
    eligible=[np.array(sorted(ids)) for ids in eligible]
    generator=np.random.default_rng(181)
    fit=[];validation=[]
    for label in range(10):
        ids=eligible[0][arrays[0][eligible[0],-1]==label].copy()
        generator.shuffle(ids)
        fit.extend(ids[:100]);validation.extend(ids[100:130])
    ids={'fit':np.array(fit),'validation':np.array(validation),'assessment':eligible[1]}
    selected={}
    for role,indices in ids.items():
        rows=arrays[1 if role=='assessment' else 0][indices]
        selected[role]=(torch.tensor(rows[:,:16].reshape(-1,8,2)/50-1,dtype=torch.float32),
                        torch.tensor(rows[:,-1],dtype=torch.long))
    record={'source_rows':[len(a) for a in arrays], 'unique_feature_groups':len(groups),
            'repeated_feature_groups':[m for m in groups.values() if len(m)>1],
            'conflicting_groups':conflict,'cross_official_split_groups':cross,
            'eligible_counts':[len(e) for e in eligible],
            'ids_1_based':{k:(v+1).tolist() for k,v in ids.items()},
            'class_counts':{k:torch.bincount(v[1],minlength=10).tolist() for k,v in selected.items()}}
    return selected,record

class RMS(nn.Module):
    def __init__(self,width):
        super().__init__();self.weight=nn.Parameter(torch.ones(width))
    def forward(self,x):
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight

class SelectiveMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(WIDTH,2*WIDTH)
        self.convolution=nn.Conv1d(WIDTH,WIDTH,3,groups=WIDTH)
        self.delta=nn.Linear(WIDTH,WIDTH)
        self.write=nn.Linear(WIDTH,STATE,bias=False)
        self.read=nn.Linear(WIDTH,STATE,bias=False)
        self.write_norm=RMS(STATE);self.read_norm=RMS(STATE)
        self.log_rates=nn.Parameter(torch.arange(1,STATE+1).float().log().repeat(WIDTH,1))
        self.skip=nn.Parameter(torch.ones(WIDTH))
        self.output=nn.Linear(WIDTH,WIDTH)
    def step(self,x,cache=None):
        projected,gate=self.input(x).chunk(2,-1)
        if cache is None:
            state=x.new_zeros(x.shape[0],WIDTH,STATE)
            history=x.new_zeros(x.shape[0],WIDTH,2)
        else: state,history=cache
        window=torch.cat([history,projected.unsqueeze(-1)],-1)
        u=F.silu((window*self.convolution.weight[:,0,:]).sum(-1)+self.convolution.bias)
        delta=F.softplus(self.delta(u))
        write=self.write_norm(self.write(u));read=self.read_norm(self.read(u))
        decay=torch.exp(-delta.unsqueeze(-1)*self.log_rates.exp())
        state=decay*state+delta.unsqueeze(-1)*write.unsqueeze(1)*u.unsqueeze(-1)
        value=(state*read.unsqueeze(1)).sum(-1)+self.skip*u
        return self.output(value*F.silu(gate)),(state,window[:,:,1:])
    def forward(self,x):
        cache=None;outputs=[]
        for token in x.unbind(1):
            output,cache=self.step(token,cache);outputs.append(output)
        return torch.stack(outputs,1)

class AttentionMixer(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(WIDTH,3*WIDTH);self.output=nn.Linear(WIDTH,WIDTH)
    def forward(self,x):
        q,k,v=self.qkv(x).chunk(3,-1)
        scores=q@k.transpose(-1,-2)/WIDTH**0.5
        mask=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device).triu(1)
        scores=scores.masked_fill(mask,float('-inf'))
        return self.output(scores.softmax(-1)@v)
    def step(self,x,cache=None):
        q,k,v=self.qkv(x).chunk(3,-1)
        keys=k[:,None,:] if cache is None else torch.cat([cache[0],k[:,None,:]],1)
        values=v[:,None,:] if cache is None else torch.cat([cache[1],v[:,None,:]],1)
        scores=(q[:,None,:]@keys.transpose(-1,-2))/WIDTH**0.5
        output=(scores.softmax(-1)@values).squeeze(1)
        return self.output(output),(keys,values)

class Layer(nn.Module):
    def __init__(self,kind):
        super().__init__();self.norm1=RMS(WIDTH);self.norm2=RMS(WIDTH)
        self.mixer=SelectiveMixer() if kind=='M' else AttentionMixer()
        self.gate_up=nn.Linear(WIDTH,2*WIDTH*2);self.down=nn.Linear(WIDTH*2,WIDTH)
    def channel(self,x):
        gate,up=self.gate_up(self.norm2(x)).chunk(2,-1)
        return self.down(F.silu(gate)*up)
    def forward(self,x):
        x=x+self.mixer(self.norm1(x));return x+self.channel(x)
    def step(self,x,cache):
        mixed,cache=self.mixer.step(self.norm1(x),cache)
        x=x+mixed;return x+self.channel(x),cache

class StrokeModel(nn.Module):
    def __init__(self,pattern):
        super().__init__();self.pattern=pattern
        if pattern=='linear': self.classifier=nn.Linear(16,10);return
        self.embed=nn.Linear(4,WIDTH)
        self.layers=nn.ModuleList([Layer(kind) for kind in pattern])
        self.norm=RMS(WIDTH);self.classifier=nn.Linear(WIDTH,10)
    def positioned(self,x,offset=0):
        position=torch.arange(offset,offset+x.shape[1],device=x.device,dtype=x.dtype)/7
        tags=torch.stack([position,position.square()],-1).expand(x.shape[0],-1,-1)
        return self.embed(torch.cat([x,tags],-1))
    def forward(self,x,all_positions=False):
        if self.pattern=='linear': return self.classifier(x.flatten(1))
        h=self.positioned(x)
        for layer in self.layers: h=layer(h)
        logits=self.classifier(self.norm(h))
        return logits if all_positions else logits[:,-1]
    def stream(self,x,cache=None,offset=0):
        if self.pattern=='linear': raise ValueError('Flat baseline has no prefix model.')
        caches=[None]*len(self.layers) if cache is None else cache
        outputs=[]
        h=self.positioned(x,offset)
        for token in h.unbind(1):
            for index,layer in enumerate(self.layers):
                token,caches[index]=layer.step(token,caches[index])
            outputs.append(self.classifier(self.norm(token)))
        return torch.stack(outputs,1),caches

def metrics(logits,labels):
    predicted=logits.argmax(-1)
    confusion=torch.bincount(labels*10+predicted,minlength=100).reshape(10,10)
    return {'n':len(labels),'errors':int((predicted!=labels).sum()),
            'cross_entropy':float(F.cross_entropy(logits,labels)),
            'confusion':confusion.tolist()}

def train():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data,record=data_split();report={'data':record,'torch':torch.__version__,'fits':[]};weights={}
    for pattern,seed in [('linear',37),('MMM',37),('AAA',37),('MAM',37),('AMM',37),('MAM',73)]:
        torch.manual_seed(seed);model=StrokeModel(pattern)
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        batch_generator=torch.Generator().manual_seed(491)
        best_loss=float('inf');best_state=None;history=[]
        for epoch in range(1,81):
            model.train();order=torch.randperm(len(data['fit'][0]),generator=batch_generator)
            for indices in order.split(100):
                loss=F.cross_entropy(model(data['fit'][0][indices]),data['fit'][1][indices])
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
            model.eval()
            with torch.no_grad(): val=metrics(model(data['validation'][0]),data['validation'][1])
            history.append({'epoch':epoch,'validation':val['cross_entropy'],'errors':val['errors']})
            if val['cross_entropy']<best_loss:
                best_loss=val['cross_entropy'];best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
        model.load_state_dict(best_state);model.eval()
        with torch.no_grad():
            entry={'key':f'{pattern}-{seed}','pattern':pattern,'seed':seed,'selected_epoch':best_epoch,
                   'parameters':sum(p.numel() for p in model.parameters()),'history':history,
                   'metrics':{role:metrics(model(x),y) for role,(x,y) in data.items()},
                   'validation_logits':model(data['validation'][0]).tolist()}
            if pattern!='linear':
                full=model(data['validation'][0],True);stream,_=model.stream(data['validation'][0])
                left,cache=model.stream(data['validation'][0][:,:3])
                right,_=model.stream(data['validation'][0][:,3:],cache,offset=3)
                reset,_=model.stream(data['validation'][0][:,3:],offset=3)
                entry['checks']={'stream_max_difference':float((full-stream).abs().max()),
                    'carried_chunk_max_difference':float((full-torch.cat([left,right],1)).abs().max()),
                    'reset_chunk_max_difference':float((full[:,3:]-reset).abs().max())}
                entry['validation_prefix_logits']=full.tolist()
        for name,array in best_state.items():weights[f"{entry['key']}::{name}"]=array.numpy()
        report['fits'].append(entry)
        print(entry['key'],best_epoch,{k:v['errors'] for k,v in entry['metrics'].items()},flush=True)
    np.savez_compressed(HERE/'stroke-fits.npz',**weights)
    (HERE/'stroke-results.json').write_text(json.dumps(report,indent=2)+'\n')

def load_fit(key):
    pattern=key.rsplit('-',1)[0];model=StrokeModel(pattern)
    with np.load(HERE/'stroke-fits.npz') as saved:
        state={name:torch.tensor(saved[f'{key}::{name}']) for name in model.state_dict()}
    model.load_state_dict(state);return model.eval()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true');args=parser.parse_args()
    if args.train:train()
    else:
        report=json.loads((HERE/'stroke-results.json').read_text())
        for fit in report['fits']:
            print(fit['key'],fit['selected_epoch'],{k:v['errors'] for k,v in fit['metrics'].items()})
