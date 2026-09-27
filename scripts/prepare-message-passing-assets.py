"""Execute the complete tiny graph study and actual sparse/PyG bridge; retain native oracles."""
from pathlib import Path
import contextlib,io,json,platform,re,runpy,shutil,sys
import numpy as np
import torch
import torch_geometric
ROOT=Path(__file__).resolve().parents[1]
ID='message-passing-graph-convolutions-gcn-gat-graphsage'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
PUBLIC=ROOT/'public/learn-assets'/ID
DOWNLOAD=ROOT/'public/learn-code'/ID
for folder in (OUT,PUBLIC,DOWNLOAD):folder.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
def save(path,value):path.write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8')
source=(PACKET/'message-passing-study.py').read_text(encoding='utf-8')
displayed=re.search(r'~~~python\n(.*?)~~~',(PACKET/'lesson.md').read_text(encoding='utf-8'),re.S).group(1)
assert source.strip()==displayed.strip()
namespace={'__file__':str(PACKET/'message-passing-study.py'),'__name__':'message_passing_native','OUT':OUT}
exec(compile(source.replace('(HERE / "calculated-inputs.json").write_text','(OUT / "reproduced-study.json").write_text'),str(PACKET/'message-passing-study.py'),'exec'),namespace)
print('Executing all twelve declared tiny graph fits; one CPU thread; no downloads.',flush=True)
with contextlib.redirect_stdout(io.StringIO()):namespace['main']()
original=json.loads((PACKET/'calculated-inputs.json').read_text())
reproduced=json.loads((OUT/'reproduced-study.json').read_text())
assert reproduced==original,'Full study changed; investigate instead of replacing history'
bridge=runpy.run_path(str(PACKET/'graph_library_bridge.py'))
routes=[]
for scratch in (False,True):
    sys.argv=['graph_library_bridge.py']+(['--scratch-only'] if scratch else [])
    with contextlib.redirect_stdout(io.StringIO()) as stdout:bridge['main']()
    routes.append(json.loads(stdout.getvalue()))
save(OUT/'library-checks.json',{'passed':True,'actualPyg':torch_geometric.__version__,'result':routes[0],'scratch':routes[1]})
graph=json.loads((PACKET/'karate-club.json').read_text())
n=len(graph['nodes']);a=torch.zeros(n,n)
for left,right,_ in graph['edges']:a[left,right]=a[right,left]=1
x=torch.tensor([[1,node['degree']/33,node['clustering']] for node in graph['nodes']])
fixtures=[]
for record in original['measurements']:
    if record['seed']!=11:continue
    model=namespace['NodeClassifier'](record['model']);model.load_state_dict({k:torch.tensor(v) for k,v in record['state_dict'].items()});model.eval()
    save(PUBLIC/f"{record['model']}-11.json",{'model':record['model'],'seed':11,'state':record['state_dict']})
    for change in ('baseline','no_edges','feature_edit','edge_edit','permutation'):
        features=x.clone();adjacency=a.clone();order=list(range(n))
        if change=='no_edges':adjacency.zero_()
        if change=='feature_edit':features[2,1]=.8
        if change=='edge_edit':adjacency[0,2]=adjacency[2,0]=0
        if change=='permutation':order=list(reversed(order));features=features[order];adjacency=adjacency[order][:,order]
        with torch.no_grad():
            logits,hidden,weights=model(features,namespace['graph_operators'](adjacency))
        fixtures.append({'model':record['model'],'change':change,'order':order,'features':features.tolist(),'adjacency':adjacency.tolist(),'logits':logits.tolist(),'hidden':hidden.tolist(),'probabilities':logits.softmax(-1).tolist(),'attention':None if weights is None else weights.tolist()})
        if change=='baseline':torch.testing.assert_close(logits.softmax(-1),torch.tensor(record['probabilities']),atol=0,rtol=0)
# Fresh small undirected, directed and edgeless per-layer cases; native package evidence is separate.
torch.manual_seed(491)
layers=[]
for count in (2,4,6):
    for kind in ('gcn','sage','gat'):
        features=torch.randn(count,3,dtype=torch.float64)
        adjacency=(torch.rand(count,count)>.65).double();adjacency.fill_diagonal_(0)
        if count==2:adjacency.zero_()
        layer=namespace['GraphLayer'](3,2,kind).double()
        output,weights=layer(features,namespace['graph_operators'](adjacency))
        layers.append({'kind':kind,'features':features.tolist(),'adjacency':adjacency.tolist(),'state':{k:v.tolist() for k,v in layer.state_dict().items()},'output':output.detach().tolist(),'attention':None if weights is None else weights.detach().tolist()})
save(OUT/'native-fixtures.json',{'fitted':fixtures,'layers':layers})
# Deterministic source-independent topology layout, no label or prediction controls positions.
import networkx as nx
g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from((u,v) for u,v,_ in graph['edges'])
positions=nx.spring_layout(g,seed=271,iterations=120,weight=None)
def spread_layout(positions):
    # Layout is presentation only. Preserve readable 12px node IDs without overlaps.
    coordinates=np.array([positions[i] for i in range(len(positions))],dtype=float)
    low=coordinates.min(0);span=coordinates.max(0)-low
    pixels=(coordinates-low)/span*np.array([480.,360.])+np.array([40.,40.])
    for _ in range(120):
        for i in range(len(pixels)):
            for j in range(i):
                delta=pixels[i]-pixels[j];distance=float(np.linalg.norm(delta))
                if distance<32:
                    direction=delta/max(distance,1e-12)
                    shift=.51*(32-distance)*direction
                    pixels[i]+=shift;pixels[j]-=shift
        pixels[:,0]=pixels[:,0].clip(25,535);pixels[:,1]=pixels[:,1].clip(25,425)
    return [[float((x-40)/240-1),float((y-30)/190-1)] for x,y in pixels]
layout=spread_layout(positions)
summary=[{k:v for k,v in r.items() if k not in ('state_dict','hidden','attention','probabilities')} for r in original['measurements']]
save(ROOT/'src/learn/data/message-passing-examples.json',{'graph':graph,'layout':layout,'roles':original['roles'],'exact':original['exact'],'baseline':original['label_propagation_correct'],'summary':summary})
save(PUBLIC/'recorded-probabilities.json',[{'model':r['model'],'seed':r['seed'],'probabilities':r['probabilities']} for r in original['measurements']])
for name in ('message-passing-study.py','graph_library_bridge.py','karate-club.json','calculated-inputs.json','data-provenance.md','NETWORKX-LICENSE.txt'):shutil.copyfile(PACKET/name,DOWNLOAD/name)
save(OUT/'native-checks.json',{'passed':True,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pyg':torch_geometric.__version__,'threads':1,'checks':['Complete displayed and companion program identical; all twelve fits rerun, every historical output exactly reproduced','Actual installed PyG GCN/SAGE/GAT: all12 graph/operator including unequal directed receiving degrees output/input-gradient/parameter-gradient/SGD cases passed','20 full trained-model baseline/removed/feature/edge/permutation outputs retained; all4 seed11 baseline probabilities exactly replay historical values','9 fresh native graph-layer fixtures span directed and empty-edge cases'],'limits':['No GPU performance or new benchmark claim','Edits hold original graph-derived features fixed unless explicitly entered; no retraining in browser']})
print('Complete graph study, native fixtures and actual PyG checks passed.',flush=True)
