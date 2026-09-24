"""A small trainable latent-attention classifier with mean and ordered-coordinate baselines.

Python 3.12+, NumPy, PyTorch, scikit-learn. All input data is local.
Run beside movement_libras.data; writes trajectory-results.json and small-fits.npz.
"""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, queries, inputs, valid=None):
        score = self.query(queries) @ self.key(inputs).transpose(-1,-2)
        score = score / queries.shape[-1]**.5
        if valid is not None:
            score = score.masked_fill(~valid[:,None,:], -torch.inf)
        weight = torch.softmax(score,dim=-1)
        return self.output(weight @ self.value(inputs)), weight


class LatentClassifier(nn.Module):
    def __init__(self, latent_count, width=24):
        super().__init__()
        self.input_projection = nn.Linear(3,width)
        self.latent = nn.Parameter(torch.randn(latent_count,width)*.05)
        self.cross = Attention(width)
        self.self_attention = Attention(width)
        self.cross_norm = nn.LayerNorm(width)
        self.self_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.final_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width,15)

    def forward(self, frames, valid):
        inputs = self.input_projection(frames)
        latent = self.latent.unsqueeze(0).expand(len(frames),-1,-1)
        # The second pass asks the input a new question using updated latents.
        for _ in range(2):
            update, weights = self.cross(self.cross_norm(latent),inputs,valid)
            latent = latent + update
            update, _ = self.self_attention(self.self_norm(latent),self.self_norm(latent))
            latent = latent + update
            latent = latent + self.feedforward(self.final_norm(latent))
        logits = self.classifier(latent.mean(dim=1))
        return logits, weights


def prepare(folder):
    table=np.loadtxt(folder/'movement_libras.data',delimiter=',')
    labels=table[:,-1].astype(int)-1
    raw=table[:,:90]
    first={}
    duplicate_groups={}
    for i,row in enumerate(raw):
        key=tuple(row)
        if key not in first:
            first[key]=i
        original=first[key]
        duplicate_groups.setdefault(original,[]).append(i)
        if labels[i]!=labels[original]:
            raise ValueError('Duplicate trajectory has conflicting labels')
    retained=np.array(sorted(first.values()))
    rng=np.random.default_rng(73)
    fit=[];validation=[];test=[]
    for label in range(15):
        order=rng.permutation(retained[labels[retained]==label])
        count=2*len(order)//3
        fit.extend(order[:count]);validation.extend(order[count:-4]);test.extend(order[-4:])
    fit=np.array(fit);validation=np.array(validation);test=np.array(test)
    coordinates=(2*raw.reshape(-1,45,2)-1).astype(np.float32)
    positions=np.broadcast_to(np.linspace(-1,1,45,dtype=np.float32)[None,:,None],(360,45,1))
    frames=torch.from_numpy(np.concatenate([coordinates,positions],axis=2).copy())
    valid=torch.ones((360,45),dtype=torch.bool)
    data={'labels':labels,'retained':retained,'duplicate_groups':duplicate_groups}
    return data,frames,valid,coordinates.mean(1),coordinates.reshape(360,90),fit,validation,test


def metrics(logits,labels):
    prediction=logits.argmax(1)
    return {'errors':int((prediction!=labels).sum()),'n':len(labels),
            'accuracy':float((prediction==labels).float().mean()),
            'cross_entropy':float(nn.functional.cross_entropy(logits,labels)),
            'confusion':confusion_matrix(labels.numpy(),prediction.numpy(),labels=np.arange(15)).tolist(),
            'predictions':prediction.tolist()}


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    folder=Path(__file__).parent
    data,frames,valid,pooled,ordered,fit,val,test=prepare(folder)
    labels=torch.from_numpy(data['labels']).long()
    result={'versions':{'numpy':np.__version__,'torch':torch.__version__},
        'data_roles':{'fit_source_rows':(fit+1).tolist(),'validation_source_rows':(val+1).tolist(),
                      'test_source_rows':(test+1).tolist()},
        'retained_source_rows':(data['retained']+1).tolist(),
        'duplicate_groups':[[i+1 for i in ids] for ids in data['duplicate_groups'].values() if len(ids)>1],
        'coordinate_transform':'2*x-1; appended ordinal position -1 to1',
        'seeds':[11,29],'epochs':80,'learning_rate':.005,'width':24,'repeat_reads':2,
        'baselines':{},'models':[]}
    arrays={}
    for name,inputs in [('mean',pooled),('ordered',ordered)]:
        baseline=LogisticRegression(C=1,max_iter=1000).fit(inputs[fit],labels[fit])
        result['baselines'][name]={role:{
            'errors':int(np.sum(baseline.predict(inputs[idx])!=data['labels'][idx])),'n':len(idx)}
            for role,idx in [('fit',fit),('validation',val),('test',test)]}
        arrays[name+'_coef']=baseline.coef_;arrays[name+'_intercept']=baseline.intercept_
    for latent_count in [1,4]:
        for seed in [11,29]:
            torch.manual_seed(seed)
            model=LatentClassifier(latent_count)
            optimizer=torch.optim.Adam(model.parameters(),lr=.005)
            best_loss=float('inf');best_state=None;best_epoch=None
            history=[]
            for epoch in range(1,81):
                model.train();optimizer.zero_grad()
                logits,_=model(frames[fit],valid[fit])
                loss=nn.functional.cross_entropy(logits,labels[fit])
                loss.backward();optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits,_=model(frames[val],valid[val])
                    validation_loss=float(nn.functional.cross_entropy(logits,labels[val]))
                history.append([epoch,float(loss.detach()),validation_loss])
                if validation_loss<best_loss:
                    best_loss=validation_loss
                    best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            model.load_state_dict(best_state);model.eval()
            with torch.no_grad():
                logits,weights=model(frames,valid)
                summary={role:metrics(logits[idx],labels[idx]) for role,idx in [('fit',fit),('validation',val),('test',test)]}
                paired_logits,_=model(frames.flip(1),valid.flip(1))
                paired_error=float((paired_logits-logits).abs().max())
                padded=torch.cat([frames,torch.full((360,5,3),1000.)],dim=1)
                padded_valid=torch.cat([valid,torch.zeros((360,5),dtype=torch.bool)],dim=1)
                padded_logits,_=model(padded,padded_valid)
                padding_error=float((padded_logits-logits).abs().max())
            assert paired_error<1e-4 and padding_error<1e-4
            key=f'latents{latent_count}_seed{seed}'
            for name,value in best_state.items():
                arrays[key+'::'+name]=value.numpy()
            arrays[key+'::logits']=logits.numpy()
            arrays[key+'::read_weights_first_validation']=weights[val[0]].numpy()
            result['models'].append({'latents':latent_count,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
                'best_epoch':best_epoch,'history':history,'metrics':summary,
                'paired_permutation_max_logit_difference':paired_error,
                'padding_addition_max_logit_difference':padding_error})
    np.savez_compressed(folder/'small-fits.npz',**arrays)
    (folder/'trajectory-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'baselines':result['baselines'],'models':[
        {k:m[k] for k in ['latents','seed','parameters','best_epoch']} | {
            role+'_errors':m['metrics'][role]['errors'] for role in ['fit','validation','test']}
        for m in result['models']]},indent=2))


if __name__=='__main__':
    main()
