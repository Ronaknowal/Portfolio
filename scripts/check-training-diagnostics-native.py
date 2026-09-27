"""Execute the unchanged complete supplied programs in isolation; export faithful runtime evidence."""
import contextlib,hashlib,io,json,math,platform,re,shutil,statistics,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];ID='neural-training-diagnostics-reproducible-experiments';PACKET=ROOT/'docs/teaching/drafts'/ID;OUT=ROOT/'docs/teaching/deep-learning-completion'/ID;PUBLIC=ROOT/'public/learn-assets'/ID
torch.set_num_threads(1);torch.set_default_dtype(torch.float64);PUBLIC.mkdir(exist_ok=True)
checks=[];comparisons=0;max_error=0.
def compare(a,b,path=''):
 global comparisons,max_error
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'/{i}')
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  comparisons+=1;error=abs(a-b);max_error=max(max_error,error);assert error<=2e-12*(1+abs(b)),(path,a,b,error)
 else:assert a==b,(path,a,b)
def check(name):checks.append({'name':name,'passed':True})
with tempfile.TemporaryDirectory(prefix='diagnostics-native-') as temp:
 target=Path(temp)
 for name in ['wine_diagnostics.py','checkpoint_replay.py','calculations.py','wine.csv','split.json','experiment-protocol.md']:shutil.copyfile(PACKET/name,target/name)
 logs={}
 for program,result in [('wine_diagnostics.py','wine-results.json'),('checkpoint_replay.py','checkpoint-results.json'),('calculations.py','calculation-results.json')]:
  execution=subprocess.run([sys.executable,'-B',str(target/program)],cwd=target,capture_output=True,text=True,check=True);logs[program]=execution.stdout
  actual=json.loads((target/result).read_text());expected=json.loads((PACKET/result).read_text())
  if result=='wine-results.json':actual.pop('environment');expected.pop('environment')
  compare(actual,expected,result);check('Complete '+program+' executed; every retained numeric/result field matches')
 manuscript=(PACKET/'lesson.md').read_text(encoding='utf-8');program=re.findall(r'```python\n(.*?)\n```',manuscript,re.S)[0];executed=subprocess.run([sys.executable,'-B','-c',program],capture_output=True,text=True,check=True);assert executed.stdout==re.findall(r'```text\n(.*?)\n```',manuscript,re.S)[0]+'\n';check('Exact finite-difference program extracted from manuscript executes and matches displayed output')
fixtures={'scalar':[],'mode':[],'restart':[],'pairs':[]}
for n in [1,2,4]:
 for weight in [-3.,0.,.5,3.]:
  for rate in [0.,.1,.5]:
   rows=[{'id':i+1,'x':float((i*3)%7-3),'y':float(i-1)} for i in range(n)];w=torch.tensor(weight,requires_grad=True);x=torch.tensor([r['x'] for r in rows]);y=torch.tensor([r['y'] for r in rows]);loss=((w*x-y).square()/2).mean();loss.backward();fixtures['scalar'].append({'rows':rows,'weight':weight,'rate':rate,'gradient':w.grad.item(),'loss':loss.item(),'correct':weight-rate*w.grad.item()})
for values,mean,variance in [([1.,3.],0.,1.),([-2.,6.],1.,4.),([0.,1.],.5,.5),([3.,3.],-2.,0.),([-10.,10.],5.,25.)]:
 for module in ['batchnorm','linear']:
  for training in [True,False]:
   for grad in [True,False]:
    layer=nn.BatchNorm1d(1) if module=='batchnorm' else nn.Linear(1,1)
    with torch.no_grad():
     if module=='batchnorm':layer.running_mean.fill_(mean);layer.running_var.fill_(variance)
     else:layer.weight.fill_(2);layer.bias.fill_(1)
    layer.train(training)
    with torch.set_grad_enabled(grad):out=layer(torch.tensor(values).reshape(2,1))
    fixtures['mode'].append({'input':{'values':values,'mean':mean,'variance':variance,'module':module,'training':training,'gradEnabled':grad},'output':out.detach().flatten().tolist(),'mean':layer.running_mean.item() if module=='batchnorm' else None,'variance':layer.running_var.item() if module=='batchnorm' else None,'graph':out.requires_grad})
for target,weight in [(3.,1.),(4.,2.),(-5.,5.)]:
 for rate in [0.,.1,.5]:
  for momentum in [0.,.5,.95]:
   for save,steps in [(1,1),(3,4),(5,5)]:
    def walk(w,v,count,start):
     rows=[]
     for i in range(count):
      t=torch.tensor(w,requires_grad=True);loss=(t-target).square()/2;loss.backward();g=t.grad.item();nv=momentum*v+g;nw=w-rate*nv;rows.append({'index':start+i,'weight':w,'velocity':v,'gradient':g,'loss':loss.item(),'nextVelocity':nv,'nextWeight':nw});w,v=nw,nv
     return rows,w,v
    prefix,w,v=walk(weight,0.,save,1);full,fw,_=walk(w,v,steps,save+1);reset,rw,_=walk(w,0.,steps,save+1);fixtures['restart'].append({'input':{'target':target,'weight':weight,'rate':rate,'momentum':momentum,'save':save,'steps':steps},'prefix':prefix,'checkpoint':{'weight':w,'velocity':v},'full':full,'reset':reset,'finalFull':fw,'finalReset':rw})
for n in range(1,7):
 pairs=[{'id':str(i),'a':.2+i*.09,'b':.2+i*.09+(-1)**i*.03*(i+1)} for i in range(n)];diff=[p['b']-p['a'] for p in pairs];fixtures['pairs'].append({'pairs':pairs,'mean':statistics.mean(diff),'sd':statistics.stdev(diff) if n>1 else None})
check('Fresh native scalar gradients, all independent BatchNorm/linear mode switches, restart trajectories and paired summaries exported')
wine=json.loads((PACKET/'wine-results.json').read_text());replay=json.loads((PACKET/'checkpoint-results.json').read_text());split=json.loads((PACKET/'split.json').read_text());rows=np.loadtxt(PACKET/'wine.csv',delimiter=',',skiprows=1)
from sklearn.datasets import load_wine
source=load_wine();assert np.array_equal(rows[:,2:],source.data) and np.array_equal(rows[:,1],source.target);assert sorted(split['train']+split['validation']+split['test'])==list(range(178));check('178 measured source rows match sklearn and split is disjoint/exhaustive; test remains unevaluated')
def projection(run):return [[r['update'],r['train_supplied']['loss'],r['train_supplied']['accuracy'],r['train_original']['accuracy'],r['validation']['loss'],r['validation']['accuracy'],r['train_original']['loss']] for r in run['trace']]
paired={'columns':['update','training_loss','training_accuracy','training_original_accuracy','validation_loss','validation_accuracy','training_original_loss'],'seeds':{}}
for seed in [3,7,19]:
 selected=[r for r in wine['runs'] if r['seed']==seed and r['treatment'] in ['clean','shuffled_labels']];value={r['treatment']:{'trace':projection(r),'final_predictions':r['final_predictions']} for r in selected};(PUBLIC/f'wine-seed-{seed}.json').write_text(json.dumps(value,separators=(',',':'))+'\n');paired['seeds'][str(seed)]={r['treatment']:projection(r) for r in selected}
(PUBLIC/'wine-pairs.json').write_text(json.dumps(paired,separators=(',',':'))+'\n')
for name in ['calculations.py','calculation-results.json','checkpoint_replay.py','checkpoint-results.json','wine_diagnostics.py','wine-results.json','wine.csv','split.json','experiment-protocol.md','data-provenance.md']:shutil.copyfile(PACKET/name,PUBLIC/name)
display={'activation':wine['activation_probe'],'split':split,'tiny':[{'treatment':r['treatment'],'trace':r['trace']} for r in wine['runs'] if r['treatment'] in ['tiny','omitted_update']],'checkpoint':replay,'rawRows':[{'id':int(r[0]),'target':int(r[1]),'features':r[2:].tolist()} for r in rows if int(r[0]) in split['train']+split['validation']]}
(ROOT/'src/learn/data/training-diagnostics-study.js').write_text('// Lossless display metadata; full and paired trajectories load on intersection.\nexport const diagnosticsStudy='+json.dumps(display,separators=(',',':'))+';\n',encoding='utf-8')
(OUT/'native-fixtures.json').write_text(json.dumps(fixtures,separators=(',',':'))+'\n');(OUT/'native-output.txt').write_text('\n'.join(program+'\n'+output for program,output in logs.items()),encoding='utf-8');result={'passed':True,'checks':checks,'retainedNumericComparisons':comparisons,'maxRetainedError':max_error,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':1,'dtype':'float64','device':'CPU'},'fixtures':{k:len(v) for k,v in fixtures.items()},'limitations':['Only the unchanged eight declared Wine treatments and the declared12-update checkpoint continuations were fitted; no new hyperparameter selection or test score.','Exact continuation is established for this CPU environment, not across hardware, workers or releases.']};(OUT/'native-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
