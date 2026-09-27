"""Replay retained scientific inputs and export a bounded browser port oracle."""
from pathlib import Path
import csv, hashlib, importlib.util, json, platform, shutil, subprocess, sys, tempfile
import numpy as np
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
ROOT=Path(__file__).resolve().parents[1]
ID='titans-multi-memory-architecture'
SOURCE=ROOT/'docs/teaching/drafts'/ID
OUT=ROOT/'docs/teaching/deep-learning-completion'/ID
PUBLIC=ROOT/'public/learn-assets'/ID
PUBLIC.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path.insert(0,str(SOURCE))
import neural_memory as neural
import memory_mechanisms as mechanism
import rental_memory_study as study

def tree(value):
    if isinstance(value,torch.Tensor):return value.detach().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tree(v) for v in value]
    if isinstance(value,np.floating):return float(value)
    return value

checks=[]
def check(name,condition=True):
    assert condition,name
    checks.append({'name':name,'passed':True})

# Run the complete supplied author program without overwriting its preserved outputs.
with tempfile.TemporaryDirectory(prefix='titans-memory-') as folder:
    for name in ['neural_memory.py','memory_mechanisms.py','rental_memory_study.py','check_author_packet.py','mechanism-results.json','rental-results.json','bike-sharing-daily.csv']:
        shutil.copyfile(SOURCE/name,Path(folder)/name)
    replay=subprocess.run([sys.executable,'-B',str(Path(folder)/'check_author_packet.py')],capture_output=True,text=True)
    check('Complete prepared arithmetic/data/state/causality checker executes on copied immutable inputs',replay.returncode==0)
    if replay.returncode:print(replay.stderr)
    author_evidence=json.loads((Path(folder)/'author-checks.json').read_text())

result=json.loads((SOURCE/'rental-results.json').read_text())
days,counts=study.load_observations();mean,scale=result['mean'],result['scale']
keys=study.make_keys(days,counts,mean,scale);targets=torch.tensor((counts-mean)/scale,dtype=torch.float64)
fixtures={'linear':[],'gated':[],'nonlinear':[],'outer':[],'rental':{},'counterfactual':[]}
rng=np.random.default_rng(927)
for i in range(24):
    weight=rng.uniform(-2,2,(2,2));momentum=rng.uniform(-.4,.4,(2,2));key=rng.uniform(-2,2,2);value=rng.uniform(-5,5,2)
    rate=[0,.1,.5][i%3];retention=[0,.5,.8][i%3];decay=[0,.1,1][(i//3)%3]
    new_weight,new_momentum,gradient=mechanism.linear_write(weight,momentum,key,value,rate,retention,decay)
    fixtures['linear'].append(tree(dict(weight=weight,momentum=momentum,key=key,value=value,settings=dict(rate=rate,retention=retention,decay=decay),nextWeight=new_weight,nextMomentum=new_momentum,gradient=gradient)))
for n in [4,6,8]:
    tokens=rng.uniform(-2,2,(n,2))
    for prefix in [False,True]:
        for rate in [0,.125,.5]:
            trace,state=mechanism.gated_sequence(tokens,prefix=prefix,rate=rate)
            fixtures['gated'].append(tree(dict(tokens=tokens,prefix=prefix,rate=rate,records=trace,state=state)))
for dims in [(2,3),(3,4),(9,8)]:
    for seed in [3,7,19]:
        params=neural.initialize_memory(seed,input_size=dims[0],hidden_size=dims[1]);momentum=tuple(torch.tensor(rng.uniform(-.1,.1,p.shape),dtype=torch.float64) for p in params)
        key=torch.tensor(rng.uniform(-2,2,dims[0]),dtype=torch.float64);target=torch.tensor(rng.uniform(-2,2),dtype=torch.float64)
        new,state,loss,gradient=neural.write_memory(params,momentum,key,target,.13,.4,.07)
        fixtures['nonlinear'].append(tree(dict(parameters=params,momentum=momentum,key=key,target=target,settings=dict(rate=.13,retention=.4,decay=.07),prediction=neural.read_memory(params,key),nextParameters=new,nextMomentum=state,loss=loss,gradient=gradient)))
outer_initial=neural.initialize_memory(3,input_size=2,hidden_size=3)
for rate in [0,.1,.25,.5]:
    rate_tensor=torch.tensor(rate,dtype=torch.float64,requires_grad=True)
    p=neural.copy_parameters(outer_initial);key=torch.tensor([1.,-.5],dtype=torch.float64);target=torch.tensor(.8,dtype=torch.float64);query=torch.tensor([-.25,.75],dtype=torch.float64)
    new,_,_,_=neural.write_memory(p,tuple(torch.zeros_like(v) for v in p),key,target,rate_tensor,0,0,differentiable=True)
    prediction=neural.read_memory(new,query);loss=.5*(prediction+.3)**2;derivative=torch.autograd.grad(loss,rate_tensor)[0]
    detached,_,_,_=neural.write_memory(p,tuple(torch.zeros_like(v) for v in p),key,target,rate_tensor,0,0,differentiable=False)
    detached_loss=.5*(neural.read_memory(detached,query)+.3)**2
    detached_derivative=torch.autograd.grad(detached_loss,rate_tensor,allow_unused=True)[0]
    check(f'Outer graph mode at rate {rate}: identical forward and disconnected detached derivative',torch.allclose(prediction,neural.read_memory(detached,query),atol=0,rtol=0) and detached_derivative is None)
    fixtures['outer'].append(tree(dict(rate=rate,prediction=prediction,loss=loss,derivative=derivative)))
for seed,row in result['seeds'].items():
    params=tuple(torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in row['initial_parameters'])
    trace,final,state=study.replay(params,keys,targets)
    expected=np.array([[r[k] for k in ['prediction_z','loss_before_write','gradient_norm','update_norm']] for r in row['trace']])
    actual=np.array([[r[k] for k in ['prediction_z','loss_before_write','gradient_norm','update_norm']] for r in trace])
    check(f'Seed {seed}: full 366-day native replay matches all four retained trace quantities',np.max(np.abs(actual-expected))<1e-10)
    fixtures['rental'][seed]=tree(dict(trace=trace,final=final,momentum=state))
    (PUBLIC/f'rental-seed-{seed}.json').write_text(json.dumps(row,separators=(',',':'))+'\n',encoding='utf8')
for day,change in [(380,1000),(548,-1000),(720,2000)]:
    changed=counts.copy();changed[day]+=change;changed_keys=study.make_keys(days,changed,mean,scale);changed_targets=torch.tensor((changed-mean)/scale,dtype=torch.float64)
    params=tuple(torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in result['seeds']['3']['initial_parameters'])
    trace,_,_=study.replay(params,changed_keys,changed_targets)
    fixtures['counterfactual'].append(dict(day=day,change=change,trace=trace))
    check(f'Arrival edit at {day} preserves that day and every prior forecast',all(abs(trace[j]['prediction_z']-fixtures['rental']['3']['trace'][j]['prediction_z'])<1e-12 for j in range(day-365+1)))

allowlist=['neural_memory.py','memory_mechanisms.py','rental_memory_study.py','check_author_packet.py','mechanism-results.json','rental-results.json','bike-sharing-daily.csv','source-description.txt','experiment-protocol.md','data-provenance.md']
for name in allowlist:shutil.copyfile(SOURCE/name,PUBLIC/name)
display={'dates':[str(d) for d in days],'counts':counts.tolist(),'mean':mean,'scale':scale,'windows':result['windows'],'seeds':{s:{'fit_loss':r['fit_loss'],'windows':r['windows']} for s,r in result['seeds'].items()},'outerInitial':tree(outer_initial)}
(ROOT/'src/learn/data/titans-memory-study.js').write_text('// Lossless selected display data; full per-seed traces load when the real-data lab opens.\nexport const titansStudy='+json.dumps(display,separators=(',',':'))+';\n',encoding='utf8')
(OUT/'native-fixtures.json').write_text(json.dumps(fixtures,separators=(',',':'))+'\n',encoding='utf8')
source_files=[str(p.relative_to(ROOT)).replace('\\','/') for p in SOURCE.iterdir() if p.is_file()]
receipt={'passed':True,'checks':checks,'preparedChecker':author_evidence,'environment':dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=1,device='cpu',dtype='float64'),'sourceHashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},'limitations':['Replayed all saved initial weights; no new 1000-step fits or hyperparameter selection.','The gated block is an explicit small MAG specialization, not a published Titans checkpoint.','Native execution does not establish browser rendering or controls.']}
(OUT/'native-checks.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'checks':len(checks),'gatedCases':len(fixtures['gated']),'rentalDays':1098,'downloadFiles':len(allowlist),'selectedModelFiles':3}))
