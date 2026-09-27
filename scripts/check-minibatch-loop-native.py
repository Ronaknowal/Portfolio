"""Execute complete supplied CPU loop routes and export independent autograd fixtures."""
from pathlib import Path
import contextlib,copy,hashlib,io,json,platform,runpy,shutil,sys
import numpy as np
import torch
from torch.nn import functional as F
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
ROOT=Path(__file__).resolve().parents[1];ID='mini-batches-training-loops-gradient-accumulation';SOURCE=ROOT/'docs/teaching/drafts'/ID;OUT=ROOT/'docs/teaching/deep-learning-completion'/ID;PUBLIC=ROOT/'public/learn-assets'/ID;PUBLIC.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True;sys.path.insert(0,str(SOURCE))
import train_iris as study
import author_checks as original
saved=json.loads((SOURCE/'author-results.json').read_text());checks=[]
def check(name,condition=True):
    assert condition,name
    checks.append({'name':name,'passed':True})
def tree(v):
    if isinstance(v,torch.Tensor):return v.detach().tolist()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {k:tree(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [tree(x) for x in v]
    return v
output=io.StringIO()
with contextlib.redirect_stdout(output):runpy.run_path(str(SOURCE/'trace_update.py'),run_name='__main__')
check('Complete supplied scalar PyTorch program reproduces retained printed output',output.getvalue().replace('\r\n','\n')==(SOURCE/'trace-output.txt').read_text())
models={}
for physical in [32,12,7]:
    model,optimizer,history,calls=study.train(physical);models[physical]=(model,optimizer,history,calls)
    check(f'Complete real Iris train({physical}) reproduces every epoch metric',np.allclose(np.array(history),np.array(saved['iris']['history']),atol=1e-12,rtol=1e-12))
check('Three physical partitions keep80 optimizer updates, with80/220/380 backward calls',[models[p][3] for p in [32,12,7]]==[80,220,380])
for physical in [12,7]:
    a,oa,_,_=models[32];b,ob,_,_=models[physical]
    check(f'Physical{physical}: all final weights and momentum match full32',all(torch.allclose(x,y,atol=1e-12,rtol=0) and torch.allclose(oa.state[x]['momentum_buffer'],ob.state[y]['momentum_buffer'],atol=1e-12,rtol=0) for x,y in zip(a.parameters(),b.parameters())))
fixtures={'scalar':[],'weighted':[],'normalization':[],'iris':[]};rng=np.random.default_rng(927)
for n in [1,3,8]:
 for mu in [0,.9]:
  for rate in [0,.1,.2]:
   for policy in ['correct','clear_each','step_each']:
    x=rng.uniform(-5,5,n);y=rng.uniform(-5,5,n);groups=[list(range(i,min(n,i+2))) for i in range(0,n,2)];w=torch.nn.Parameter(torch.tensor(.3));optimizer=torch.optim.SGD([w],lr=rate,momentum=mu);optimizer.zero_grad(set_to_none=True);events=[]
    for gi,group in enumerate(groups):
     if policy=='clear_each':optimizer.zero_grad(set_to_none=True)
     loss=.5*(w*torch.tensor(x[group])-torch.tensor(y[group])).square().sum()/n;loss.backward();events.append({'weight':float(w.detach()),'gradient':float(w.grad),'group':gi})
     if policy=='step_each':optimizer.step();optimizer.zero_grad(set_to_none=True)
    if policy!='step_each':optimizer.step()
    fixtures['scalar'].append(dict(rows=[dict(x=float(a),y=float(b)) for a,b in zip(x,y)],groups=groups,settings=dict(initial=.3,rate=rate,momentum=mu,policy=policy),weight=float(w.detach()),buffer=float(optimizer.state[w]['momentum_buffer']) if mu else None,backwardEvents=events))
for n in [1,4,8]:
 for mode in ['mass','chunks','eligible','slots']:
  rows=[dict(x=float(rng.uniform(-5,5)),y=float(rng.uniform(-5,5)),weight=float(rng.choice([0,.5,2,4])),included=bool(rng.integers(0,2)),group=int(i%3)) for i in range(n)]
  w=torch.tensor(0.,requires_grad=True);terms=[r['weight']*int(r['included'])*.5*(w*r['x']-r['y'])**2 for r in rows];mass=sum(r['weight']*r['included'] for r in rows);reference=torch.autograd.grad(sum(terms)/mass,w,retain_graph=True)[0].item() if mass else None
  if mode=='chunks':
   active=[(sum(terms[i] for i,r in enumerate(rows) if r['group']==g),sum(r['weight']*r['included'] for r in rows if r['group']==g)) for g in sorted(set(r['group'] for r in rows))];active=[(s,d) for s,d in active if d>0];loss=sum(s/d for s,d in active)/len(active) if active else None
  else:
   denom={'mass':mass,'eligible':sum(r['included'] for r in rows),'slots':n}[mode];loss=sum(terms)/denom if denom else None
  compared=torch.autograd.grad(loss,w)[0].item() if loss is not None else None;fixtures['weighted'].append(dict(rows=rows,mode=mode,reference=reference,compared=compared))
for n in [4,6,8]:
 for mode in ['local','frozen','none']:
  for theta in [0,1,-1.5]:
   x=torch.tensor(rng.uniform(-12,12,n));y=torch.tensor(rng.uniform(-3,3,n));groups=[list(range(0,n,2)),list(range(1,n,2))];zfull=x.clone();zlocal=x.clone();bn=torch.nn.BatchNorm1d(1,affine=False,momentum=.1);localbn=torch.nn.BatchNorm1d(1,affine=False,momentum=.1)
   if mode=='local':
    zfull=bn(x[:,None]).flatten()
    for g in groups:zlocal[g]=localbn(x[g,None]).flatten()
   elif mode=='frozen':zfull=zlocal=(x-5)/np.sqrt(10+1e-5)
   scale=torch.tensor(float(theta),requires_grad=True);floss=.5*(scale*zfull-y).square().mean();lloss=.5*(scale*zlocal-y).square().mean();fg=torch.autograd.grad(floss,scale,retain_graph=True)[0];lg=torch.autograd.grad(lloss,scale)[0]
   fixtures['normalization'].append(tree(dict(rows=[dict(x=float(x[i]),y=float(y[i]),group=i%2) for i in range(n)],mode=mode,theta=theta,fullZ=zfull,localZ=zlocal,fullLoss=floss,localLoss=lloss,fullGradient=fg,localGradient=lg,fullRunning=float(bn.running_mean[0]),localRunning=float(localbn.running_mean[0]))))
for kind,model,optimizer in [('initial',study.initial_model,None),('trained20',models[32][0],models[32][1])]:
 snapshot={'parameters':tree(dict(model.named_parameters())),'momentum':tree({name:optimizer.state[p]['momentum_buffer'] for name,p in model.named_parameters()}) if optimizer else None,'center':tree(study.center),'scale':tree(study.scale),'kind':kind}
 (PUBLIC/f'iris-{kind}.json').write_text(json.dumps(snapshot,separators=(',',':'))+'\n',encoding='utf8')
 for size in [8,24,32]:
  ids=study.epoch_orders[0][:size];rows=[dict(id=int(i+1),features=study.records[i,1:5].tolist(),target=int(study.targets[i])) for i in ids]
  for changed in [False,True]:
   current=copy.deepcopy(rows)
   if changed:current[0]['features'][2]+=.5;current[0]['target']=(current[0]['target']+1)%3
   features=(torch.tensor([r['features'] for r in current])-study.center)/study.scale;targets=torch.tensor([r['target'] for r in current],dtype=torch.long);pred=model(features);loss=F.cross_entropy(pred,targets);gradients=torch.autograd.grad(loss,tuple(model.parameters()));gradient=dict(zip(dict(model.named_parameters()),gradients))
   individual=[]
   for i in range(size):
    one=F.cross_entropy(model(features[i:i+1]),targets[i:i+1]);individual.append(tree(dict(zip(dict(model.named_parameters()),torch.autograd.grad(one,tuple(model.parameters()))))))
   fixtures['iris'].append(tree(dict(kind=kind,rows=current,loss=loss,gradient=gradient,individual=individual,logits=pred,probabilities=pred.softmax(-1))))

# Re-execute the shape/reduction distinctions against the installed normal library.
logits=torch.tensor([[0.,0.],[1.,0.],[0.,2.],[3.,-1.]],requires_grad=True);targets=torch.tensor([0,1,1,-100]);weights=torch.tensor([1.,3.]);loss=F.cross_entropy(logits,targets,weight=weights,ignore_index=-100);check('Weighted class-index CE retains denominator7',abs(loss.item()-saved['cross_entropy']['class_index_weighted_loss'])<1e-12)
prob=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.]]);check('Weighted probability-target CE uses target-position count4',torch.allclose(F.cross_entropy(logits,prob,weight=weights),F.cross_entropy(logits,prob,weight=weights,reduction='sum')/4))
check('All-ignored sum is zero without taking an undefined mean',F.cross_entropy(logits,torch.full((4,),-100),reduction='sum').item()==0)
for name,args in [('contrast',([0,2,10,12],[0,0,1,1],[[0,1],[2,3]])),('equal',([0,2,0,2],[0,1,0,1],[[0,1],[2,3]])),('constant',([1,1,1,1],[0,0,1,1],[[0,1],[2,3]]))]:original.normalization_case(*args);check(f'Prepared {name} normalization agrees with native BatchNorm')
allowlist=['trace_update.py','train_iris.py','iris.csv','data-provenance.md','author-results.json']
for name in allowlist:shutil.copyfile(SOURCE/name,PUBLIC/name)
display={'iris':saved['iris'],'size7':saved['iris_microbatch_7_transfer'],'records':[dict(id=int(i+1),features=study.records[i,1:5].tolist(),target=int(study.targets[i])) for i in study.epoch_orders[0][:32]],'trainingRowIds':(study.training_rows+1).tolist(),'validationRowIds':(study.validation_rows+1).tolist()}
(ROOT/'src/learn/data/minibatch-loop-study.js').write_text('// Lossless experiment history and the actual first training group.\nexport const minibatchStudy='+json.dumps(display,separators=(',',':'))+';\n',encoding='utf8')
(OUT/'native-fixtures.json').write_text(json.dumps(fixtures,separators=(',',':'))+'\n',encoding='utf8')
sources=[str(p.relative_to(ROOT)).replace('\\','/') for p in SOURCE.iterdir() if p.is_file()];receipt={'passed':True,'checks':checks,'sourceHashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},'environment':dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=1,dtype='float64',device='cpu'),'cases':{k:len(v) for k,v in fixtures.items()},'limitations':['The successful verification executes all three declared 20-epoch CPU fits with unchanged settings; an earlier fixture-export attempt failed on an integer tensor after fitting, then was corrected without tuning or selecting runs.','AMP and multi-process DDP are operation contracts supported by primary docs and scalar arithmetic, not hardware executions.']};(OUT/'native-checks.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps({'passed':True,'checks':len(checks),'cases':receipt['cases']}))
