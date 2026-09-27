"""Independent PyG full-network references, without author-model imports."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

torch.set_num_threads(1)
root=Path('public/learn-assets/message-passing-graph-convolutions-gcn-gat-graphsage')
e=json.loads(Path('src/learn/data/message-passing-examples.json').read_text())
n=len(e['graph']['nodes'])
adj=np.zeros((n,n))
for edge in e['graph']['edges']:
    u,v=edge[:2]
    adj[u,v]=adj[v,u]=1
degree=adj.sum(1)
clustering=[]
for i in range(n):
    neighbors=np.flatnonzero(adj[i])
    clustering.append(float(adj[np.ix_(neighbors,neighbors)].sum()/(len(neighbors)*(len(neighbors)-1))) if len(neighbors)>1 else 0.)
base=np.column_stack((np.ones(n),degree/33,clustering))

def layer(kind,state,prefix):
    weight=torch.tensor(state[prefix+'linear.weight'],dtype=torch.float64)
    out,ins=weight.shape
    if kind=='gcn': result=GCNConv(ins,out,cached=False).double()
    elif kind=='sage': result=SAGEConv(ins,out,normalize=False).double()
    elif kind=='gat': result=GATConv(ins,out,heads=1,concat=True,negative_slope=.2,dropout=0).double()
    else: result=nn.Linear(ins,out).double()
    with torch.no_grad():
        if kind in ('gcn','gat'):result.lin.weight.copy_(weight)
        elif kind=='sage':
            result.lin_l.weight.copy_(weight)
            result.lin_r.weight.copy_(torch.tensor(state[prefix+'self_linear.weight']))
        else:result.weight.copy_(weight)
        bias=result.lin_l.bias if kind=='sage' else result.bias
        bias.copy_(torch.tensor(state[prefix+'bias']))
        if kind=='gat':
            result.att_src.copy_(torch.tensor(state[prefix+'sender_score']).reshape(1,1,-1))
            result.att_dst.copy_(torch.tensor(state[prefix+'receiver_score']).reshape(1,1,-1))
    return result

cases=[]
for kind in ['mlp','gcn','sage','gat']:
    model=json.loads((root/f'{kind}-11.json').read_text())
    first,last=layer(kind,model['state'],'first.'),layer(kind,model['state'],'last.')
    for intervention in ['original','features','directed','isolated','no_edges']:
        a=adj.copy();x=base.copy()
        if intervention=='features':x[7]=[.6,.9,-.3];x[31]=[1.1,.12,.8]
        if intervention=='directed':a=np.tril(a)
        if intervention=='isolated':a[4,:]=0;a[:,4]=0
        if intervention=='no_edges':a[:]=0
        receivers,senders=np.nonzero(a)
        edges=torch.tensor(np.array([senders,receivers]),dtype=torch.long)
        inputs=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        h=first(inputs) if kind=='mlp' else first(inputs,edges)
        logits=last(h.relu()) if kind=='mlp' else last(h.relu(),edges)
        probabilities=logits.softmax(-1)
        gradient=torch.autograd.grad(probabilities[7,1],inputs)[0]
        cases.append(dict(kind=kind,intervention=intervention,adjacency=a.tolist(),features=x.tolist(),
                          first=h.detach().tolist(),hidden=h.relu().detach().tolist(),
                          logits=logits.detach().tolist(),probabilities=probabilities.detach().tolist(),gradient=gradient.tolist()))
out=Path('docs/teaching/deep-learning-completion/message-passing-graph-convolutions-gcn-gat-graphsage')
(out/'independent-fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')
print('20 fresh complete networks and input derivatives computed by native PyG/Linear.')
