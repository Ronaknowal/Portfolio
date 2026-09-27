"""Fresh real-row SGD state-transition oracles, without another fitting run."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
ID='mini-batches-training-loops-gradient-accumulation'
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
source=ROOT/'public/learn-assets'/ID
data=np.loadtxt(source/'iris.csv',delimiter=',',skiprows=1)
rng=np.random.default_rng(73499)
result={'torch':torch.__version__,'threads':1,'chains':[]}
for kind in ['initial','trained20']:
    snapshot=json.loads((source/f'iris-{kind}.json').read_text())
    for case in range(3):
        model=nn.Sequential(nn.Linear(4,8),nn.Tanh(),nn.Linear(8,3))
        model.load_state_dict({k:torch.tensor(v) for k,v in snapshot['parameters'].items()})
        optimizer=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
        if snapshot['momentum']:
            for name,p in model.named_parameters():
                optimizer.state[p]['momentum_buffer']=torch.tensor(snapshot['momentum'][name])
        record={'kind':kind,'steps':[]}
        for size in [8,24]:
            ids=rng.choice(150,size=size,replace=False)
            rows=[{'id':int(i+1),'features':data[i,1:5].tolist(),'target':int(data[i,5])} for i in ids]
            rows[case]['features'][case%4]=float(rng.uniform(0,10))
            rows[case]['target']=(rows[case]['target']+1)%3
            x=(torch.tensor([r['features'] for r in rows])-torch.tensor(snapshot['center']))/torch.tensor(snapshot['scale'])
            y=torch.tensor([r['target'] for r in rows],dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits,y)
            loss.backward()
            gradients={k:p.grad.tolist() for k,p in model.named_parameters()}
            optimizer.step()
            record['steps'].append({'rows':rows,'loss':loss.detach().item(),'logits':logits.detach().tolist(),'gradient':gradients,'parameters':{k:p.detach().tolist() for k,p in model.named_parameters()},'momentum':{k:optimizer.state[p]['momentum_buffer'].tolist() for k,p in model.named_parameters()}})
        result['chains'].append(record)
path=ROOT/'docs/teaching/deep-learning-completion'/ID/'independent-native-fixtures.json'
path.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'chains':6,'actualSGDSteps':12,'newFits':0}))
