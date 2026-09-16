"""Learn small RWKV-4-style and positive-kernel classifiers from real trajectories.

Offline educational training, not a released checkpoint reproduction.
Python 3.12+, NumPy, PyTorch and scikit-learn.
"""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def prepare(folder):
    data=np.loadtxt(folder/'movement_libras.data',delimiter=',')
    labels=data[:,-1].astype(int)-1
    first={};groups={}
    for row_id,row in enumerate(data[:,:90]):
        key=tuple(row);first.setdefault(key,row_id)
        groups.setdefault(first[key],[]).append(row_id)
        assert labels[row_id]==labels[first[key]]
    unique=np.array(sorted(first.values()))
    rng=np.random.default_rng(73);roles=[[],[],[]]
    for label in range(15):
        order=rng.permutation(unique[labels[unique]==label])
        cut=2*len(order)//3
        for role,indices in zip(roles,[order[:cut],order[cut:-4],order[-4:]]):
            role.extend(indices)
    x=torch.from_numpy((2*data[:,:90].reshape(-1,45,2)-1).astype(np.float32))
    return x,torch.from_numpy(labels),[np.array(r) for r in roles],groups

def shift(x,previous):
    if previous is None:
        previous=torch.zeros_like(x[:,0])
    return torch.cat([previous[:,None],x[:,:-1]],dim=1)

class MemoryMixer(nn.Module):
    def __init__(self,kind,width):
        super().__init__()
        self.kind=kind;self.width=width
        self.mix=nn.Parameter(torch.zeros(3,width))
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.output=nn.Linear(width,width,bias=False)
        if kind=='rwkv4':
            self.raw_decay=nn.Parameter(torch.linspace(-3.,0.,width))
            self.bonus=nn.Parameter(torch.zeros(width))

    def forward(self,x,saved=None):
        previous=None if saved is None else saved[0]
        old=shift(x,previous);mix=self.mix.sigmoid()
        q=self.query(mix[0]*x+(1-mix[0])*old)
        k=self.key(mix[1]*x+(1-mix[1])*old)
        v=self.value(mix[2]*x+(1-mix[2])*old)
        outputs=[]
        if self.kind=='rwkv4':
            if saved is None:
                numerator=torch.zeros_like(x[:,0])
                denominator=torch.zeros_like(numerator)
                log_scale=torch.full_like(numerator,-torch.inf)
            else:
                numerator,denominator,log_scale=saved[1:]
            log_decay=-self.raw_decay.exp()
            for t in range(x.shape[1]):
                output_scale=torch.maximum(log_scale,self.bonus+k[:,t])
                old_weight=(log_scale-output_scale).exp()
                new_weight=(self.bonus+k[:,t]-output_scale).exp()
                average=(old_weight*numerator+new_weight*v[:,t])/(old_weight*denominator+new_weight)
                outputs.append(q[:,t].sigmoid()*average)
                next_scale=torch.maximum(log_scale+log_decay,k[:,t])
                old_weight=(log_scale+log_decay-next_scale).exp()
                new_weight=(k[:,t]-next_scale).exp()
                numerator=old_weight*numerator+new_weight*v[:,t]
                denominator=old_weight*denominator+new_weight
                log_scale=next_scale
            state=(x[:,-1],numerator,denominator,log_scale)
        else:
            q=F.elu(q)+1;k=F.elu(k)+1
            if saved is None:
                memory=x.new_zeros((len(x),self.width,self.width))
                normalizer=torch.zeros_like(x[:,0])
            else:
                memory,normalizer=saved[1:]
            for t in range(x.shape[1]):
                memory=memory+k[:,t,:,None]*v[:,t,None,:]
                normalizer=normalizer+k[:,t]
                numerator=torch.einsum('bd,bdv->bv',q[:,t],memory)
                denominator=(q[:,t]*normalizer).sum(-1,keepdim=True)
                outputs.append(numerator/denominator)
            state=(x[:,-1],memory,normalizer)
        return self.output(torch.stack(outputs,dim=1)),state

class MemoryBlock(nn.Module):
    def __init__(self,kind,width):
        super().__init__()
        self.time_norm=nn.LayerNorm(width)
        self.mixer=MemoryMixer(kind,width)
        self.channel_norm=nn.LayerNorm(width)
        self.channel_mix=nn.Parameter(torch.zeros(2,width))
        self.channel_key=nn.Linear(width,2*width,bias=False)
        self.channel_value=nn.Linear(2*width,width,bias=False)
        self.channel_gate=nn.Linear(width,width,bias=False)

    def forward(self,x,saved=None):
        mixed,time_state=self.mixer(self.time_norm(x),None if saved is None else saved[0])
        x=x+mixed
        normalized=self.channel_norm(x)
        old=shift(normalized,None if saved is None else saved[1])
        mix=self.channel_mix.sigmoid()
        key=self.channel_key(mix[0]*normalized+(1-mix[0])*old)
        gate=self.channel_gate(mix[1]*normalized+(1-mix[1])*old).sigmoid()
        x=x+gate*self.channel_value(F.relu(key).square())
        return x,(time_state,normalized[:,-1])

class TrajectoryMemoryClassifier(nn.Module):
    def __init__(self,kind,width=16):
        super().__init__()
        self.input=nn.Linear(2,width)
        self.blocks=nn.ModuleList([MemoryBlock(kind,width) for _ in range(2)])
        self.classifier=nn.Linear(width,15)

    def features(self,x,saved=None):
        x=self.input(x);states=[]
        for i,block in enumerate(self.blocks):
            x,state=block(x,None if saved is None else saved[i]);states.append(state)
        return x,states

    def forward(self,x):
        features,_=self.features(x)
        return self.classifier(features.mean(1))

def metrics(logits,labels):
    predicted=logits.argmax(1)
    return {'errors':int((predicted!=labels).sum()),'n':len(labels),
            'cross_entropy':float(F.cross_entropy(logits,labels)),
            'predictions':predicted.tolist(),
            'confusion':confusion_matrix(labels,predicted,labels=np.arange(15)).tolist()}

def main():
    folder=Path(__file__).parent
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    x,labels,roles,groups=prepare(folder);fit,val,test=roles
    report={'versions':{'numpy':np.__version__,'torch':torch.__version__},
            'roles':{n:(r+1).tolist() for n,r in zip(['fit','validation','test'],roles)},
            'duplicate_groups':[[i+1 for i in g] for g in groups.values() if len(g)>1],
            'epochs':100,'learning_rate':.003,'seeds':[17,41],'width':16,'blocks':2,'models':[]}
    arrays={}
    flat=x.numpy().reshape(360,90)
    baseline=LogisticRegression(C=1,max_iter=1000).fit(flat[fit],labels[fit])
    report['baseline']={n:{'errors':int((baseline.predict(flat[r])!=labels[r].numpy()).sum()),'n':len(r)}
                        for n,r in zip(['fit','validation','test'],roles)}
    arrays['baseline_coef']=baseline.coef_;arrays['baseline_intercept']=baseline.intercept_
    for kind in ['rwkv4','positive_kernel']:
        for seed in [17,41]:
            torch.manual_seed(seed);model=TrajectoryMemoryClassifier(kind)
            optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            best=float('inf');history=[]
            for epoch in range(1,101):
                optimizer.zero_grad()
                loss=F.cross_entropy(model(x[fit]),labels[fit]);loss.backward();optimizer.step()
                with torch.no_grad():value=float(F.cross_entropy(model(x[val]),labels[val]))
                assert np.isfinite(value) and torch.isfinite(loss)
                history.append([epoch,float(loss.detach()),value])
                if value<best:
                    best=value;selected=copy.deepcopy(model.state_dict());selected_epoch=epoch
            model.load_state_dict(selected);model.eval()
            with torch.no_grad():
                logits=model(x);features,_=model.features(x[:5])
                pieces=[];state=None
                for start,stop in [(0,1),(1,13),(13,29),(29,45)]:
                    part,state=model.features(x[:5,start:stop],state);pieces.append(part)
                chunk_error=float((features-torch.cat(pieces,dim=1)).abs().max())
                assert chunk_error<2e-5
            prefix=f'{kind}_seed{seed}::'
            for name,tensor in selected.items():arrays[prefix+name]=tensor.numpy()
            arrays[prefix+'logits']=logits.numpy()
            result={'kind':kind,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
                    'selected_epoch':selected_epoch,'history':history,'chunk_feature_max_error':chunk_error,
                    'metrics':{n:metrics(logits[r],labels[r]) for n,r in zip(['fit','validation','test'],roles)}}
            report['models'].append(result)
            print(kind,seed,selected_epoch,[result['metrics'][n]['errors'] for n in ['fit','validation','test']],flush=True)
    np.savez_compressed(folder/'trajectory-memory-fits.npz',**arrays)
    (folder/'trajectory-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()

