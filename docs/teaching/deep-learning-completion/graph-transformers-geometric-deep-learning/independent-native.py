"""Independent review oracles; no training, author model imports, or downloads."""
from pathlib import Path
import json, itertools
import numpy as np
import scipy.special
from scipy.sparse.csgraph import shortest_path
import torch
from torch.nn import functional as F
from torch_geometric.nn import GPSConv, GCNConv
import torch_geometric

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
ID='graph-transformers-geometric-deep-learning'
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
rng=np.random.default_rng(27631)
def tensor(x): return torch.tensor(x,dtype=torch.float64)
def values(x): return x.detach().tolist() if isinstance(x,torch.Tensor) else np.asarray(x).tolist()
def forward(model,x,raw):
    a=tensor(raw);n=len(a);s={k:tensor(v) for k,v in model['state'].items()};kind=model['kind']
    transition=a/a.sum(1).clamp_min(1)[:,None]
    if kind=='walk':x=torch.cat([x,torch.stack([torch.linalg.matrix_power(transition,k).diag() for k in range(1,5)],1)],1)
    closed=a+torch.eye(n);inv=closed.sum(1).rsqrt();g=inv[:,None]*closed*inv[None,:]
    if kind=='gcn':
        z=F.relu(g@F.linear(x,s['embed.weight'])+s['embed.bias'])
        return g@F.linear(z,s['head.weight'])+s['head.bias'],None
    z=F.linear(x,s['embed.weight'],s['embed.bias']);weights=[]
    d=shortest_path(np.ascontiguousarray(raw),directed=False,unweighted=True)
    d=np.where(np.isfinite(d),d,n).astype(int)
    bias=s['bias.weight'][torch.tensor(d)].permute(2,0,1) if kind=='distance' else torch.zeros(2,n,n)
    for b in range(2):
        p=f'blocks.{b}.'
        norm=F.layer_norm(z,(16,),s[p+'norm1.weight'],s[p+'norm1.bias'],1e-5)
        q,k,v=F.linear(norm,s[p+'qkv.weight'],s[p+'qkv.bias']).chunk(3,dim=-1)
        q,k,v=[t.reshape(n,2,8).transpose(0,1) for t in (q,k,v)]
        mixed=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0).transpose(0,1).reshape(n,16)
        weights.append((q@k.transpose(-1,-2)/np.sqrt(8)+bias).softmax(-1))
        z=z+F.linear(mixed,s[p+'out.weight'],s[p+'out.bias'])
        norm=F.layer_norm(z,(16,),s[p+'norm2.weight'],s[p+'norm2.bias'],1e-5)
        z=z+F.linear(F.gelu(F.linear(norm,s[p+'ff.0.weight'],s[p+'ff.0.bias']),approximate='none'),s[p+'ff.2.weight'],s[p+'ff.2.bias'])
    return F.linear(z,s['head.weight'],s['head.bias']),weights

fitted=[]
for kind in ['gcn','set','distance','walk']:
    model=json.loads((ROOT/'public/learn-assets'/ID/f'{kind}-11.json').read_text())
    for case in range(3):
        n=34
        raw=(rng.random((n,n)) < [.035,.13,.31][case]).astype(float)
        raw=np.triu(raw,1);raw=raw+raw.T
        raw[case,:]=0;raw[:,case]=0
        x0=rng.uniform(-1.7,1.9,(n,3));x=tensor(x0).requires_grad_()
        y,w=forward(model,x,raw);probe=torch.linspace(-.7,.8,n*2).reshape(n,2)
        grad=torch.autograd.grad((y*probe).sum(),x)[0]
        perm=rng.permutation(n);py,_=forward(model,tensor(x0[perm]),raw[perm][:,perm])
        torch.testing.assert_close(py,y[perm],atol=3e-12,rtol=3e-12)
        fitted.append({'kind':kind,'case':case,'features':x0.tolist(),'adjacency':raw.tolist(),'logits':values(y),'probabilities':values(y.softmax(-1)),'weights':None if w is None else [values(t) for t in w],'gradient':values(grad),'probe':values(probe),'permutation':perm.tolist(),'permutedLogits':values(py)})

# New actual PyG state and directed unequal-degree local graphs, packed 4/1/2.
torch.manual_seed(761)
package=GPSConv(4,GCNConv(4,4),heads=2,dropout=0.,norm=None,act='relu',attn_kwargs={'dropout':0.}).double().eval()
state={}
state['local.linear.weight']=values(package.conv.lin.weight)
state['local.bias']=values(package.conv.bias)
state.update({'global_attention.'+k:values(v) for k,v in package.attn.state_dict().items()})
state.update({'feedforward.'+k:values(v) for k,v in package.mlp.state_dict().items()})
gps=[]
for empty in [False,True]:
    edges=torch.empty((2,0),dtype=torch.long) if empty else torch.tensor([[0,0,1,2,3,5],[1,2,2,3,0,6]],dtype=torch.long)
    batch=torch.tensor([0,0,0,0,1,2,2]);x=tensor(rng.uniform(-2,2,(7,4))).requires_grad_()
    y=package(x,edges,batch=batch);probe=torch.linspace(-1,.9,28).reshape(7,4)
    grad=torch.autograd.grad((y*probe).sum(),x)[0]
    xx=x.detach().clone();xx[5:]+=1.4
    changed=package(xx,edges,batch=batch)
    torch.testing.assert_close(changed[:5],y[:5],rtol=0,atol=0)
    gps.append({'empty':empty,'state':state,'features':values(x),'edges':edges.tolist(),'batch':batch.tolist(),'output':values(y),'gradient':values(grad),'probe':values(probe),'changed':values(changed)})

eigen=[]
edges=list(itertools.combinations(range(4),2))
for bits in range(64):
    a=np.zeros((4,4))
    for j,(u,v) in enumerate(edges):
        if bits&(1<<j):a[u,v]=a[v,u]=1
    l=np.diag(a.sum(1))-a;w,u=np.linalg.eigh(l);groups=[];used=set()
    for i in range(4):
        if i in used:continue
        ix=np.flatnonzero(np.abs(w-w[i])<1e-9);used.update(ix)
        groups.append({'indices':ix.tolist(),'projector':(u[:,ix]@u[:,ix].T).tolist()})
    eigen.append({'a':a.tolist(),'values':w.tolist(),'groups':groups})

geometry=[]
for i in range(12):
    n=2+i%5;x=tensor(rng.uniform(-3,3,(n,3))).requires_grad_()
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if i%2:q[:,0]*=-1
    t=rng.uniform(-2,2,3);r=x[:,None,:]-x[None,:,:]
    energy=.25*r.square().sum();force=-torch.autograd.grad(energy,x)[0]
    step=[0,.03,.21][i%3]
    updated=x+step*(r/(1+r.square().sum(-1,keepdim=True))).sum(1)
    torque=torch.linalg.cross(x,force).sum(0)
    torch.testing.assert_close(torque,torch.zeros(3),atol=1e-12,rtol=0)
    torch.testing.assert_close((x*force).sum(),-2*energy,atol=1e-12,rtol=1e-12)
    geometry.append({'points':values(x),'q':q.tolist(),'t':t.tolist(),'step':step,'updated':values(updated),'energy':float(energy.detach()),'forces':values(force)})

xs=sorted(set(np.linspace(-12,12,121).tolist()+[sign*np.sqrt(3)*(1+delta) for sign in [-1,1] for delta in [-1e-10,0,1e-10]]))
special=[{'x':x,'gelu':float(x*scipy.special.ndtr(x)),'erf':float(scipy.special.erf(x))} for x in xs]
result={'versions':{'torch':torch.__version__,'pyg':torch_geometric.__version__,'numpy':np.__version__},'fitted':fitted,'gps':gps,'eigen':eigen,'geometry':geometry,'special':special,'threads':1,'noTraining':True}
(OUT/'independent-native-fixtures.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print(json.dumps({'passed':True,'fitted':len(fitted),'gps':len(gps),'spectra':len(eigen),'geometry':len(geometry),'special':len(special),'threads':1,'noTraining':True}))
