"""Verify current programs and replay every retained digit model, without refits."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.special import expit
import scipy
import sklearn
from sklearn.neural_network import BernoulliRBM

ROOT = Path(__file__).resolve().parents[1]
ID = 'boltzmann-machines-restricted-boltzmann-machines-rbm'
DRAFT = ROOT / 'docs/teaching/drafts' / ID
OUT = ROOT / 'docs/teaching/deep-learning-completion' / ID
OUT.mkdir(parents=True, exist_ok=True)
checks = []
def check(name, condition, **extra):
    assert condition, name
    checks.append(dict(name=name, passed=True, **extra))
def module(name):
    spec = importlib.util.spec_from_file_location(name, DRAFT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

scratch = module('rbm-study')
bridge = module('bernoulli_rbm_bridge')
manuscript = (DRAFT/'lesson.md').read_text(encoding='utf8')
blocks = re.findall(r'(?:```|~~~)python\n(.*?)\n(?:```|~~~)', manuscript, re.S)
for name, block in zip(['rbm-study.py','bernoulli_rbm_bridge.py'], blocks):
    text = (DRAFT/name).read_text(encoding='utf8')
    check('Displayed/downloadable source equality: '+name, text.strip()==block.strip())
    ast.parse(text)
bridge.main()
check('Complete library fit/transform/gibbs/pseudo-likelihood program executed', True)
exact = scratch.exact_checks()
check('Scratch exact checks execute: direct joint enumeration, finite differences, stationary transition, energy offset and completion', exact['finite_difference_max_error'] < 1e-9)
retained = json.loads((DRAFT/'calculated-inputs.json').read_text())
check('Original digit CSV identity', hashlib.sha256((DRAFT/'digits-400.csv').read_bytes()).hexdigest() == retained['protocol']['csv_sha256'])
records = list(csv.DictReader((DRAFT/'digits-400.csv').open()))
images = {int(r['source_id']):np.array([int(r[f'pixel_{i}'])>=8 for i in range(64)],dtype=float) for r in records}
seen, unique, duplicates = {}, [], []
for r in records:
    source = int(r['source_id']); signature=tuple(images[source])
    if signature in seen: duplicates.append(dict(source_id=source,retained_id=seen[signature]))
    else: seen[signature]=source;unique.append(r)
check('Binary duplicate audit before split', duplicates==retained['protocol']['dropped_duplicates'])
rng=np.random.default_rng(91); roles={k:[] for k in ['fit','development','assessment']}
for digit in range(10):
    subset=[r for r in unique if int(r['digit'])==digit]
    order=rng.permutation(len(subset));ids=[int(subset[i]['source_id']) for i in order]
    roles['fit']+=ids[:-16];roles['development']+=ids[-16:-8];roles['assessment']+=ids[-8:]
check('Reconstructed declared 238/80/80 source roles',roles==retained['protocol']['roles'])
observed=np.array([i%8<4 for i in range(64)])
native_cases=[]
for fit in retained['fits']:
    w,a,b=np.array(fit['weights']),np.array(fit['visible_bias']),np.array(fit['hidden_bias'])
    name=f"{fit['method']}-{fit['seed']}"
    for role,ids in roles.items():
        v=np.array([images[i] for i in ids]);actual=scratch.metrics(v,w,a,b)
        error=float(np.max(abs(np.array(actual['per_image_nll'])-fit['metrics'][role]['per_image_nll'])))
        check(f'{name} {role}: every retained per-image NLL and reconstruction mean replay',error<1e-10 and abs(actual['mean_reconstruction_mse']-fit['metrics'][role]['mean_reconstruction_mse'])<1e-12,maxError=error)
    assessment=np.array([images[i] for i in roles['assessment']])
    complete=np.array([scratch.conditional_missing(v,observed,w,a,b) for v in assessment])
    check(f'{name} all 80 recorded completions and aggregate metrics',np.max(abs(complete-fit['assessment_completion']))<1e-10 and abs(np.mean((complete[:,~observed]-assessment[:,~observed])**2)-fit['completion_mse'])<1e-12 and int(np.sum((complete[:,~observed]>=.5)==assessment[:,~observed]))==fit['completion_correct'])
    for kind in ['half','none','all','checkerboard','edited','placeholder']:
        v=assessment[0].copy(); mask=observed.copy()
        if kind=='none':mask[:]=False
        elif kind=='all':mask[:]=True
        elif kind=='checkerboard':mask=np.array([(i//8+i%8)%2==0 for i in range(64)])
        elif kind=='edited':v[18]=1-v[18]
        elif kind=='placeholder':v[28]=1-v[28]
        probability=scratch.conditional_missing(v,mask,w,a,b)
        native_cases.append(dict(model=name,kind=kind,visible=v.tolist(),observed=mask.tolist(),completion=probability.tolist(),hidden=expit(v@w+b).tolist(),metrics=scratch.metrics(v[None],w,a,b)))
    h,p,pv,_=scratch.hidden_distribution(w,a,b)
    draws=np.random.default_rng(fit['seed']+3000)
    indices=draws.choice(len(h),16,p=p)
    samples=(draws.random((16,64))<pv[indices]).astype(int)
    check(f'{name} all16 original generated samples preserve order and hidden states',samples.tolist()==fit['exact_samples'] and h[indices].tolist()==fit['hidden_sample_states'])

states=scratch.bits(2)
library=BernoulliRBM(n_components=1,learning_rate=.05,batch_size=5,n_iter=20,random_state=19).fit(np.repeat(states,[1,2,2,5],axis=0))
library_rows=[]
for delta in [0., np.log(2.)]:
    model=dict(w=library.components_.T.tolist(),a=library.intercept_visible_.tolist(),b=(library.intercept_hidden_+delta).tolist())
    logz=scratch.hidden_distribution(np.array(model['w']),np.array(model['a']),np.array(model['b']))[3]
    library_rows.append(dict(model=model,hidden=expit(states@np.array(model['w'])+model['b']).ravel().tolist(),probabilities=np.exp(-scratch.free_energy(states,np.array(model['w']),np.array(model['a']),np.array(model['b']))-logz).tolist()))
tiny=[]
for w,a,b in [(np.array([[.4],[-1.2]]),np.array([.2,-.7]),np.array([.9])),(np.zeros((2,1)),np.zeros(2),np.zeros(1)),(np.array([[6.],[-6.]]),np.array([-6.,6.]),np.array([6.]))]:
    counts=np.array([1,3,2,4.]); pos=scratch.positive(np.repeat(states,counts.astype(int),axis=0),w,b);neg=scratch.exact_negative(w,a,b)
    tiny.append(dict(model=dict(w=w.tolist(),a=a.tolist(),b=b.tolist()),counts=counts.tolist(),probabilities=np.exp(-scratch.free_energy(states,w,a,b)-neg[3]).tolist(),gradient=np.concatenate([(pos[0]-neg[0]).ravel(),pos[1]-neg[1],pos[2]-neg[2]]).tolist()))
(OUT/'native-fixtures.json').write_text(json.dumps(dict(exact=exact,cases=native_cases,library=library_rows,tiny=tiny),separators=(',',':'))+'\n')
(OUT/'native-checks.json').write_text(json.dumps(dict(topicId=ID,passed=True,checks=checks,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,maximumThreads=2),limits=['Original nine 300-epoch fits reused without refitting; every final retained model and all recorded role metrics replayed.','No new image classification, GPU timing, large-state enumeration or browser acceptance claimed.']),indent=2)+'\n')
print(f'{len(checks)} RBM native/program/provenance checks passed; 54 model/mask cases retained.')
