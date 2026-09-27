"""Execute all declared graph fits, geometry arithmetic and actual PyG GPS composition."""
from pathlib import Path
import contextlib, io, json, platform, re, runpy, shutil, sys
import numpy as np
import torch
import torch_geometric
ROOT=Path(__file__).resolve().parents[1]
ID='graph-transformers-geometric-deep-learning'
PACKET=ROOT/'docs/teaching/drafts'/ID
PREVIOUS=ROOT/'docs/teaching/drafts/message-passing-graph-convolutions-gcn-gat-graphsage'
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
PUBLIC=ROOT/'public/learn-assets'/ID
DOWNLOAD=ROOT/'public/learn-code'/ID
for folder in (OUT,PUBLIC,DOWNLOAD):folder.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
def save(path,value):path.write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8')
source=(PACKET/'graph-transformer-study.py').read_text(encoding='utf-8')
displayed=re.search(r'~~~python\n(.*?)~~~',(PACKET/'lesson.md').read_text(encoding='utf-8'),re.S).group(1)
assert source.strip()==displayed.strip()
scope={'__file__':str(PACKET/'graph-transformer-study.py'),'__name__':'geometric_graph_native','OUT':OUT}
exec(compile(source.replace("(HERE / 'calculated-inputs.json').write_text","(OUT / 'reproduced-study.json').write_text"),str(PACKET/'graph-transformer-study.py'),'exec'),scope)
original_structure=scope['structural_inputs']
def typed_structure(adjacency):
    if adjacency.dtype==torch.float32:return original_structure(adjacency)
    n=len(adjacency);degree=adjacency.sum(-1);transition=adjacency/degree[:,None].clamp_min(1);power=torch.eye(n,dtype=adjacency.dtype);returns=[]
    for _ in range(4):power=power@transition;returns.append(power.diagonal())
    distance=original_structure(adjacency.float())[1]
    closed=adjacency+torch.eye(n,dtype=adjacency.dtype);inv=closed.sum(-1).rsqrt()
    return torch.stack(returns,-1),distance,inv[:,None]*closed*inv[None,:]
scope['structural_inputs']=typed_structure
print('Executing twelve complete declared fits with one CPU thread and no downloads.',flush=True)
if '--reuse-study' not in sys.argv:
    with contextlib.redirect_stdout(io.StringIO()):scope['run']()
original=json.loads((PACKET/'calculated-inputs.json').read_text())
reproduced=json.loads((OUT/'reproduced-study.json').read_text())
assert original==reproduced,'Historical results differ: investigate without replacing them'
geometry_source=(PACKET/'geometry-calculations.py').read_text(encoding='utf-8')
geometry={'__file__':str(PACKET/'geometry-calculations.py'),'__name__':'geometry_native','OUT':OUT}
geometry_source=geometry_source.replace("(Path(__file__).resolve().parent / 'geometry-results.json')","(OUT / 'reproduced-geometry.json')")
exec(compile(geometry_source,str(PACKET/'geometry-calculations.py'),'exec'),geometry)
with contextlib.redirect_stdout(io.StringIO()):geometry['run']()
expected_geometry=json.loads((PACKET/'geometry-results.json').read_text())
assert json.loads((OUT/'reproduced-geometry.json').read_text())==expected_geometry
sys.path.insert(0,str(PREVIOUS))
bridge=runpy.run_path(str(PACKET/'gps_library_bridge.py'))
with contextlib.redirect_stdout(io.StringIO()) as stdout:bridge['main']()
library=json.loads(stdout.getvalue())
graph=json.loads((PACKET/'karate-club.json').read_text());n=len(graph['nodes'])
a=torch.zeros(n,n)
for u,v,_ in graph['edges']:a[u,v]=a[v,u]=1
x=torch.tensor(original['features'])
fixtures=[]
for record in original['records']:
    if record['seed']!=11:continue
    model=scope['Model'](record['kind'],n)
    model.load_state_dict({k:torch.tensor(v) for k,v in record['state_dict'].items()});model.eval()
    save(PUBLIC/f"{record['kind']}-11.json",{'kind':record['kind'],'seed':11,'state':record['state_dict']})
    for change in ('baseline','edge_edit','recomputed_features','no_edges','feature_edit','permutation'):
        adj=a.clone();features=x.clone();order=list(range(n))
        if change in ('edge_edit','recomputed_features'):adj[0,1]=adj[1,0]=0
        if change=='recomputed_features':
            degree=adj.sum(-1)
            for i in range(n):
                nbr=torch.where(adj[i]>0)[0];k=len(nbr)
                features[i,1]=degree[i]/33
                features[i,2]=adj[nbr][:,nbr].sum()/(k*(k-1)) if k>1 else 0
        if change=='no_edges':adj.zero_()
        if change=='feature_edit':features[2,1]=.8
        if change=='permutation':order=order[::-1];adj=adj[order][:,order];features=features[order]
        with torch.no_grad():logits,weights=model(features,scope['structural_inputs'](adj))
        with torch.no_grad():
            model.double();double_logits,double_weights=model(features.double(),scope['structural_inputs'](adj.double()));model.float()
        fixtures.append({'doubleLogits':double_logits.tolist(),'doubleProbabilities':double_logits.softmax(-1).tolist(),'doubleAttention':None if double_weights is None else double_weights.tolist(),'kind':record['kind'],'change':change,'features':features.tolist(),'adjacency':adj.tolist(),'order':order,'logits':logits.tolist(),'probabilities':logits.softmax(-1).tolist(),'attention':None if weights is None else weights.tolist()})
        if change=='baseline':torch.testing.assert_close(logits.softmax(-1),torch.tensor(record['probabilities']),atol=0,rtol=0)
# Actual GPS bridge plus a third isolate; preserve all parameters for editable browser inference.
torch.manual_seed(629)
manual=bridge['LocalGlobalBlock']().double().eval();package,pairs=bridge['matched_package'](manual);package.eval()
edge=torch.tensor([[0,1,1,2,3,4],[1,0,2,1,4,3]])
batch=torch.tensor([0,0,0,1,1,2])
raw=torch.tensor([[1,0,.2],[-1,.3,1],[.4,-.7,2],[.3,1,-.5],[2,-.4,.7],[.8,-.5,1.2]],dtype=torch.float64)
features=torch.cat([raw,bridge['return_feature'](6,edge)],-1)
gps={'state':{k:v.tolist() for k,v in manual.state_dict().items()},'features':features.tolist(),'edges':edge.tolist(),'batch':batch.tolist(),'cases':[]}
for change in ('baseline','edit_b','missing_batch','edit_b_missing_batch','permutation'):
    xx=features.clone();ee=edge.clone();bb=batch.clone();order=list(range(6))
    if 'edit_b' in change:xx[3,0]+=4
    if 'missing_batch' in change:bb.zero_()
    if change=='permutation':order=[2,0,1,4,3,5];inverse=torch.argsort(torch.tensor(order));xx=xx[order];ee=inverse[ee]
    with torch.no_grad():ours=manual(xx,ee,bb);theirs=package(xx,ee,batch=bb)
    torch.testing.assert_close(ours,theirs,atol=1e-10,rtol=1e-10)
    gps['cases'].append({'change':change,'features':xx.tolist(),'edges':ee.tolist(),'batch':bb.tolist(),'order':order,'output':theirs.tolist()})
base=np.array(gps['cases'][0]['output']);edited=np.array(gps['cases'][1]['output'])
assert np.allclose(base[[0,1,2,5]],edited[[0,1,2,5]],atol=1e-12)
assert np.max(np.abs(np.array(gps['cases'][2]['output'])[5]-np.array(gps['cases'][3]['output'])[5]))>1e-4
save(OUT/'library-checks.json',{'passed':True,'pyg':torch_geometric.__version__,'canonical':library,'thirdIsolateAndMissingBatch':True})
probes=[];rng=np.random.default_rng(702)
for count in (2,3,5):
    points=rng.normal(size=(count,3));q,_=np.linalg.qr(rng.normal(size=(3,3)));t=rng.normal(size=3)
    updated=geometry['geometric_update'](points)
    y=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    energy=sum(.5*((y[i]-y[j])**2).sum() for i in range(count) for j in range(i))
    force=-torch.autograd.grad(energy,y)[0]
    probes.append({'points':points.tolist(),'q':q.tolist(),'t':t.tolist(),'updated':updated.tolist(),'transformedUpdated':geometry['geometric_update'](points@q.T+t).tolist(),'energy':float(energy.detach()),'forces':force.tolist()})
save(OUT/'native-fixtures.json',{'fitted':fixtures,'gps':gps,'geometry':probes})
previous=json.loads((ROOT/'src/learn/data/message-passing-examples.json').read_text())
assert graph==previous['graph']
summary=[{k:v for k,v in r.items() if k not in ('state_dict','last_attention','probabilities','edge_removed_probabilities')} for r in original['records']]
save(ROOT/'src/learn/data/geometric-graph-examples.json',{'graph':graph,'layout':previous['layout'],'roles':original['roles'],'features':original['features'],'geometry':expected_geometry,'summary':summary,'gps':{k:v for k,v in gps.items() if k!='cases'}})
save(PUBLIC/'recorded-probabilities.json',[{k:v for k,v in r.items() if k in ('kind','seed','probabilities','edge_removed_probabilities')} for r in original['records']])
for name in ('graph-transformer-study.py','geometry-calculations.py','gps_library_bridge.py','karate-club.json','calculated-inputs.json','geometry-results.json','data-provenance.md','NETWORKX-LICENSE.txt'):shutil.copyfile(PACKET/name,DOWNLOAD/name)
shutil.copyfile(PREVIOUS/'graph_library_bridge.py',DOWNLOAD/'graph_library_bridge.py')
save(OUT/'native-checks.json',{'passed':True,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pyg':torch_geometric.__version__,'threads':1,'checks':['Complete displayed program identical; all twelve full fits exactly reproduce all historical results','Complete geometry program rerun; original arithmetic exactly reproduced','Actual PyG GPS forward/input and parameter gradients/SGD step, separate-batch isolation and within-graph permutation passed','Third isolate, cross-graph feature edit, missing batch and permutation: five full native GPS fixtures','24 full real fitted-model intervention oracles; four baseline probability arrays exactly replay originals','Three new QR orthogonal/translation geometry probes and native autograd energy-derived forces'], 'limits':['No pretrained network, molecular dataset or GPU performance claimed','Twelve weak observed graph outcomes preserved without model selection']})
print('Complete graph fits, geometry and actual GPS bridge passed.',flush=True)
