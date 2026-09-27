"""Small S4D-style and selective sequence classifiers on local Libras trajectories.

Python3.12+, NumPy, PyTorch, scikit-learn. CPU author experiment, not research
checkpoint replication. Run beside movement_libras.data. Hyperparameters are fixed.
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
    source=np.loadtxt(folder/'movement_libras.data',delimiter=',')
    labels=source[:,-1].astype(int)-1
    first={};groups={}
    for i,row in enumerate(source[:,:90]):
        key=tuple(row)
        first.setdefault(key,i)
        groups.setdefault(first[key],[]).append(i)
        assert labels[i]==labels[first[key]]
    unique=np.array(sorted(first.values()))
    rng=np.random.default_rng(73);fit=[];validation=[];test=[]
    for label in range(15):
        order=rng.permutation(unique[labels[unique]==label]);count=2*len(order)//3
        fit.extend(order[:count]);validation.extend(order[count:-4]);test.extend(order[-4:])
    coords=(2*source[:,:90].reshape(-1,45,2)-1).astype(np.float32)
    return torch.from_numpy(coords),torch.from_numpy(labels),[np.array(r) for r in [fit,validation,test]],groups


class DiagonalMixer(nn.Module):
    """Four complex modes plus their implicit conjugates per channel."""
    def __init__(self,width,state_size=4):
        super().__init__()
        self.log_decay=nn.Parameter(torch.full((width,state_size),np.log(.5)))
        self.frequency=nn.Parameter(torch.arange(state_size).float()[None,:].repeat(width,1)*np.pi)
        self.log_step=nn.Parameter(torch.linspace(np.log(.02),np.log(.2),width))
        self.read_real=nn.Parameter(torch.randn(width,state_size)*.15)
        self.read_imag=nn.Parameter(torch.randn(width,state_size)*.15)
        self.skip=nn.Parameter(torch.ones(width))

    def parameters_discrete(self):
        A=torch.complex(-self.log_decay.exp(),self.frequency)
        step=self.log_step.exp()[:,None]
        Ad=torch.exp(step*A)
        Bd=torch.expm1(step*A)/A  # B=1; negative real A is nonsingular.
        C=torch.complex(self.read_real,self.read_imag)
        return Ad,Bd,C

    def forward(self,u,recurrent=False):
        Ad,Bd,C=self.parameters_discrete()
        if recurrent:
            state=torch.zeros((len(u),)+Ad.shape,dtype=Ad.dtype,device=u.device)
            result=[]
            for t in range(u.shape[1]):
                state=Ad*state+Bd*u[:,t,:,None]
                result.append(2*(state*C).sum(-1).real+self.skip*u[:,t])
            return torch.stack(result,dim=1)
        length=u.shape[1]
        powers=Ad[:,:,None]**torch.arange(length,device=u.device)
        taps=2*(C[:,:,None]*Bd[:,:,None]*powers).sum(1).real
        size=2*length
        transformed=torch.fft.rfft(u.transpose(1,2),n=size)*torch.fft.rfft(taps,n=size)[None,:,:]
        y=torch.fft.irfft(transformed,n=size)[:,:,:length].transpose(1,2)
        return y+self.skip*u


class SelectiveMixer(nn.Module):
    """Reference-style exp(Delta*A), Delta*B injection, serial scan."""
    def __init__(self,width,state_size=8):
        super().__init__()
        self.width=width;self.state_size=state_size
        self.log_decay=nn.Parameter(torch.arange(1,state_size+1).float().log()[None,:].repeat(width,1))
        self.coefficients=nn.Linear(width,2*state_size+width)
        with torch.no_grad():
            self.coefficients.bias[2*state_size:]=-2.
        self.skip=nn.Parameter(torch.ones(width))

    def forward(self,u,recurrent=False):
        n=self.state_size
        projected=self.coefficients(u)
        B,C,raw_delta=torch.split(projected,[n,n,self.width],dim=-1)
        delta=F.softplus(raw_delta)
        A=-self.log_decay.exp()
        state=torch.zeros((len(u),self.width,n),device=u.device,dtype=u.dtype)
        ys=[]
        for t in range(u.shape[1]):
            decay=torch.exp(delta[:,t,:,None]*A)
            injection=delta[:,t,:,None]*B[:,t,None,:]*u[:,t,:,None]
            state=decay*state+injection
            ys.append((state*C[:,t,None,:]).sum(-1)+self.skip*u[:,t])
        return torch.stack(ys,dim=1)


class TrajectoryClassifier(nn.Module):
    def __init__(self,kind,width=16):
        super().__init__()
        self.input=nn.Linear(2,width)
        self.norms=nn.ModuleList([nn.LayerNorm(width),nn.LayerNorm(width)])
        mixer=DiagonalMixer if kind=='diagonal' else SelectiveMixer
        self.mixers=nn.ModuleList([mixer(width),mixer(width)])
        self.outputs=nn.ModuleList([nn.Linear(width,width),nn.Linear(width,width)])
        self.classifier=nn.Linear(width,15)

    def forward(self,x,recurrent=False):
        z=self.input(x)
        for norm,mixer,output in zip(self.norms,self.mixers,self.outputs):
            z=z+output(F.gelu(mixer(norm(z),recurrent)))
        return self.classifier(z.mean(1))


def metric(logits,labels):
    predictions=logits.argmax(1)
    return {'errors':int((predictions!=labels).sum()),'n':len(labels),
            'cross_entropy':float(F.cross_entropy(logits,labels)),
            'predictions':predictions.tolist(),
            'confusion':confusion_matrix(labels,predictions,labels=np.arange(15)).tolist()}


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    folder=Path(__file__).parent
    x,labels,roles,groups=prepare(folder);fit,val,test=roles
    report={'versions':{'numpy':np.__version__,'torch':torch.__version__},
            'roles':{name:(r+1).tolist() for name,r in zip(['fit','validation','test'],roles)},
            'duplicate_groups':[[i+1 for i in g] for g in groups.values() if len(g)>1],
            'seeds':[17,41],'epochs':100,'learning_rate':.003,'width':16,'blocks':2,'models':[]}
    arrays={}
    ordered=x.numpy().reshape(360,90)
    baseline=LogisticRegression(C=1,max_iter=1000).fit(ordered[fit],labels[fit])
    report['ordered_baseline']={name:{'errors':int((baseline.predict(ordered[r])!=labels[r].numpy()).sum()),'n':len(r)}
         for name,r in zip(['fit','validation','test'],roles)}
    arrays['baseline_coef']=baseline.coef_;arrays['baseline_intercept']=baseline.intercept_
    for kind in ['diagonal','selective']:
        for seed in [17,41]:
            torch.manual_seed(seed);model=TrajectoryClassifier(kind)
            optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            best=float('inf');best_state=None;history=[]
            for epoch in range(1,101):
                optimizer.zero_grad();loss=F.cross_entropy(model(x[fit]),labels[fit]);loss.backward();optimizer.step()
                with torch.no_grad():validation_loss=float(F.cross_entropy(model(x[val]),labels[val]))
                assert np.isfinite(validation_loss) and torch.isfinite(loss)
                history.append([epoch,float(loss.detach()),validation_loss])
                if validation_loss<best:
                    best=validation_loss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            model.load_state_dict(best_state);model.eval()
            with torch.no_grad():
                logits=model(x)
                parity=float((logits-model(x,recurrent=True)).abs().max())
                assert parity<2e-4
            prefix=f'{kind}_seed{seed}::'
            for name,value in best_state.items():arrays[prefix+name]=value.numpy()
            arrays[prefix+'logits']=logits.numpy()
            result={'kind':kind,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
                    'selected_epoch':best_epoch,'history':history,
                    'metrics':{name:metric(logits[r],labels[r]) for name,r in zip(['fit','validation','test'],roles)}}
            if kind=='diagonal':
                result['fft_recurrence_max_logit_difference']=parity
            else:
                result['evaluation_note']='Serial reference only; no independent fused scan was executed.'
            report['models'].append(result)
            print(kind,seed,'epoch',best_epoch,'errors',[result['metrics'][n]['errors'] for n in ['fit','validation','test']],flush=True)
    np.savez_compressed(folder/'trajectory-state-fits.npz',**arrays)
    (folder/'trajectory-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
