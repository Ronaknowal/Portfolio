"""Execute complete prepared study once, retain its outcomes, add native mechanism checks."""
from pathlib import Path
import contextlib, csv, hashlib, io, json, math, platform, re, runpy, shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
ID='interleaved-cross-attention-architectures'
PACKET=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
PUBLIC=ROOT/'public/learn-assets'/ID
DOWNLOAD=ROOT/'public/learn-code'/ID
for folder in (OUT,PUBLIC,DOWNLOAD):folder.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
def save(path,value):path.write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8')
source=(PACKET/'cross-attention-study.py').read_text(encoding='utf-8')
displayed=re.search(r'~~~python\n(.*?)~~~',(PACKET/'lesson.md').read_text(encoding='utf-8'),re.S).group(1)
assert source.strip()==displayed.strip()
observed=[]
def retain(model,record,images,questions,targets,indices,source_ids):
    if model.mode=='flat':return
    # First assessment image of each digit is deterministic, independent of correctness.
    selected=[]
    for digit in range(10):
        index=next(int(i) for i in indices['assessment'][::2] if int(targets[i])==digit)
        selected.extend((index,index+1))
    with torch.no_grad():logits,weights=model(images[selected],questions[selected],capture=True)
    observed.append({'mode':record['mode'],'seed':record['seed'],'examples':[{'source_id':source_ids[i//2],'question':int(questions[i]),'target':int(targets[i]),'prediction':int(logits[j].argmax()),'head_weights':weights[j,:,0].tolist(),'probabilities':logits[j].softmax(-1).tolist()} for j,i in enumerate(selected)]})
    if model.mode=='cross' and record['seed']==11:save(OUT/'reproduced-cross-11-state.json',{k:v.tolist() for k,v in model.state_dict().items()})
namespace={'__file__':str(PACKET/'cross-attention-study.py'),'__name__':'cross_attention_native','OUT':OUT,'retain':retain}
instrumented=source.replace('(HERE / "calculated-inputs.json").write_text','(OUT / "reproduced-study.json").write_text').replace('            measured.append(record)','            retain(model,record,images,questions,targets,indices,source_ids)\n            measured.append(record)')
exec(compile(instrumented,str(PACKET/'cross-attention-study.py'),'exec'),namespace)
print('Executing the nine declared tiny fits once; one CPU thread; no downloads.',flush=True)
with contextlib.redirect_stdout(io.StringIO()):namespace['main']()
original=json.loads((PACKET/'calculated-inputs.json').read_text())
reproduced=json.loads((OUT/'reproduced-study.json').read_text())
assert reproduced==original,'Study changed: retain and investigate rather than replace historical outcomes'
cache=runpy.run_path(str(PACKET/'cross_attention_cache.py'))
with contextlib.redirect_stdout(io.StringIO()) as capture:cache['main']()
cache_output=capture.getvalue()
fixtures=[]
rng=np.random.default_rng(42719)
for case in range(18):
    t=2+case%3;s=2+case%5
    q=rng.normal(size=(t,2));k=rng.normal(size=(s,2));v=rng.normal(size=(s,2));allowed=rng.random((t,s))>.35;allowed[:,0]=True
    output,weights=namespace['attention'](q,k,v,allowed)
    fixtures.append({'query':q.tolist(),'key':k.tolist(),'value':v.tolist(),'allowed':allowed.tolist(),'output':output.tolist(),'weights':weights.tolist()})
torch.manual_seed(714)
layer=torch.nn.MultiheadAttention(8,2,batch_first=True,dropout=0).double().eval()
memory=torch.randn(1,5,8,dtype=torch.float64);query=torch.randn(1,3,8,dtype=torch.float64)
allowed=torch.arange(5)[None,:]<torch.tensor([2,4,5])[:,None]
with torch.no_grad():
    projection=cache['project_memory'](memory,layer,('memory-0','model-0'))
    output=cache['read_memory'](query,layer,projection,allowed,('memory-0','model-0'))
    paired=cache['project_memory'](memory.flip(1),layer,('reverse','model-0'))
    correctly_permuted=cache['read_memory'](query,layer,paired,allowed.flip(1),('reverse','model-0'))
    torch.testing.assert_close(output,correctly_permuted,atol=1e-12,rtol=1e-12)
    wrong=cache['read_memory'](query,layer,paired,allowed,('reverse','model-0'));assert not torch.allclose(output,wrong)
cache_fixture={'memory':memory[0].tolist(),'query':query[0].tolist(),'allowed':allowed.tolist(),'state':{k:v.tolist() for k,v in layer.state_dict().items()},'output':output[0].tolist(),'keys':projection['keys'][0].tolist(),'values':projection['values'][0].tolist()}
gates=[]
for x,f,target,alpha,rate in [(1,2,3,0,.1),(-1,3,1,0,.2),(2,-1,0,.3,.1),(.5,0,-2,0,.2),(-2,1,4,-.6,.8)]:
    a=torch.tensor(alpha,dtype=torch.float64,requires_grad=True);value=torch.tensor(f,dtype=torch.float64,requires_grad=True)
    y=x+a.tanh()*value;loss=.5*(y-target)**2;ga,gf=torch.autograd.grad(loss,(a,value))
    gates.append({'x':x,'value':f,'target':target,'alpha':alpha,'rate':rate,'output':float(y),'loss':float(loss),'alphaGradient':float(ga),'valueGradient':float(gf),'nextAlpha':float(a-rate*ga)})
frozen=torch.nn.Linear(2,1,bias=False).double();frozen.weight.requires_grad_(False)
adapter=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True);frozen(adapter).square().sum().backward();assert adapter.grad is not None
with torch.no_grad(): detached=frozen(adapter)
assert not detached.requires_grad
rows=list(csv.DictReader((PACKET/'digits-400.csv').open()))
ids={e['source_id'] for r in original['measurements'] for e in r.get('examples',[])}|{e['source_id'] for r in observed for e in r['examples']}
images=[{'source_id':int(row['source_id']),'digit':int(row['digit']),'pixels':[int(row[f'pixel_{j}']) for j in range(64)]} for row in rows if int(row['source_id']) in ids]
save(ROOT/'src/learn/data/cross-attention-examples.json',{'exact':original['exact'],'cache':cache_fixture,'summary':[{k:v for k,v in r.items() if k!='examples'} for r in original['measurements']],'perturbations':original['perturbations']})
save(PUBLIC/'observed-reads.json',{'images':images,'original':original['measurements'],'additional':observed,'selection':'Original four retained sources plus first assessment source per digit, independent of model correctness; outputs are recorded fits, not editable neural inference.'})
save(OUT/'native-fixtures.json',{'reads':fixtures,'cache':cache_fixture,'gates':gates})
for file in ('cross-attention-study.py','cross_attention_cache.py','digits-400.csv','calculated-inputs.json','data-provenance.md'):shutil.copyfile(PACKET/file,DOWNLOAD/file)
save(OUT/'native-checks.json',{'passed':True,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':1,'checks':['Complete displayed source equals companion; all nine tiny fits executed because historical weights were not retained','Reproduced study equals every historical JSON value exactly; no unfavorable outcome replaced','Actual MHA full/streamed cache bridge, forbidden-memory null, stale-identity rejection and fresh paired-mask permutation executed','18 new independent NumPy rectangular cases; five autograd gate cases; frozen-parameter input derivative versus no_grad verified'],'libraryOutput':cache_output,'newObservationSelection':'First assessment image of each digit, both question forms, every cross/gated seed; no correctness selection','limitations':['No pretrained visual-language model, GPU speed measurement or downloaded checkpoint','Recorded-map UI will not imply arbitrary image edits without inference weights']})
print('Native study reproduction and mechanisms passed.',flush=True)
