"""Actual programs plus complementary exact/optimization/library oracles."""
import contextlib
import hashlib
import io
import itertools
import json
import math
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy.optimize
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'scratch/ensemble-methods'
data = json.loads((WORK/'verification-input.json').read_text(encoding="utf-8"))
counts = {}
scopes = {}
for example in data['examples']:
    output = io.StringIO()
    scope = {}
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id']+'.py', 'exec'), scope)
    assert output.getvalue().rstrip() == example['output'], example['id']
    scopes[example['id']] = scope
counts['actualPrograms'] = len(scopes)
original = json.loads((ROOT/'docs/teaching/evidence/ensemble-original-review.json').read_text(encoding="utf-8"))['programs'][0]
preserved = next(item for item in data['examples'] if item['id']=='full-numpy-comparison')
assert preserved['code'] == original['code']
assert preserved['output'] == original['stdout'].replace('\r\n','\n').rstrip()
counts['originalProgramAndOutputConserved'] = 1

def close(actual, expected, tolerance=2e-11):
    assert math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), (actual, expected)

for item in data['votes']:
    p = list(map(lambda x:F(str(x)), item['p']))
    w = list(map(lambda x:F(str(x)), item['weights']))
    total = sum(w)
    mean = sum(a*b for a,b in zip(p,w))/total
    hard = sum(weight for probability,weight in zip(p,w) if probability>F(1,2))/total
    close(item['result']['probability'], mean)
    close(item['result']['ballotMass'], hard)
    assert item['result']['hardClass']==int(hard>F(1,2))
    assert item['result']['softClass']==int(mean>F(1,2))
counts['exactVotingCases'] = len(data['votes'])

for item in data['blends']:
    state = item['result']
    w = F(str(item['weight']))
    errors = [w*F(row['a'])+(1-w)*F(row['b']) for row in state['rows']]
    loss = sum(value*value for value in errors)/4
    close(state['mse'],loss)
    close(state['averageMemberLoss']-state['disagreement'],loss)
    if state['optimum'] is not None:
        a=np.array([row['a'] for row in state['rows']]);b=np.array([row['b'] for row in state['rows']])
        result=scipy.optimize.minimize_scalar(lambda value:np.mean((value*a+(1-value)*b)**2),bounds=(0,1),method='bounded')
        candidates=[(float(np.mean((value*a+(1-value)*b)**2)),value) for value in [0,1,result.x]]
        optimum=min(candidates)[1]
        close(state['optimum'],optimum,1e-6)
counts['exactResidualStates'] = len(data['blends'])

ys = list(map(F,[1,2,2,6,7,8]))
for item in data['bags']:
    draws=item['draws'];state=item['result']
    unique=sorted(set(draws))
    candidates=[]
    def append(threshold):
        left=[ys[i] for i in draws if threshold is None or i<=threshold]
        right=[ys[i] for i in draws if threshold is not None and i>threshold]
        first=sum(left)/len(left)
        second=sum(right)/len(right) if right else first
        score=sum((value-first)**2 for value in left)+sum((value-second)**2 for value in right)
        candidates.append((score,threshold,first,second))
    append(None)
    for low,high in zip(unique,unique[1:]):append(F(low+high,2))
    best=min(enumerate(candidates),key=lambda item:(item[1][0],item[0]))[1]
    close(state['sse'],best[0]);close(state['leftMean'],best[2]);close(state['rightMean'],best[3])
    assert state['threshold'] == (float(best[1]) if best[1] is not None else None)
    for i,prediction in enumerate(state['predictions']):close(prediction,best[2] if best[1] is None or i<=best[1] else best[3])
counts['exactBootstrapMultisets'] = len(data['bags'])
for draws in [[0,0,2,3,3,5],[1,2,2,4,4,5],[0,1,1,3,4,4]]:
    tree=DecisionTreeRegressor(max_depth=1,random_state=7).fit(np.array(draws).reshape(-1,1),np.array([float(ys[i]) for i in draws]))
    assert tree.tree_.threshold[0] in [2,2.5,3]
counts['nativePresetFits'] = 3

# Exact normalized weight recurrence; no exponentials or production minimizer reused.
def exact_trace(x,y,rounds):
    unique=sorted(set(x));cuts=[F(unique[0])-1]+[F(a+b,2) for a,b in zip(unique,unique[1:])]+[F(unique[-1])+1]
    weights=[F(1,len(x))]*len(x);trace=[]
    for _ in range(rounds):
        candidates=[]
        for cut in cuts:
            for sign in [-1,1]:
                predictions=[sign if value<=cut else -sign for value in x]
                error=sum(weight for weight,actual,pred in zip(weights,y,predictions) if actual!=pred)
                candidates.append((error,cut,sign,predictions))
        error,cut,sign,pred=min(enumerate(candidates),key=lambda item:(item[1][0],item[0]))[1]
        if error==0:return trace,'perfect'
        if error>=F(1,2):return trace,'no-edge'
        before=weights
        weights=[w/(2*error) if predicted!=actual else w/(2*(1-error)) for w,predicted,actual in zip(weights,pred,y)]
        trace.append((error,before,weights,pred))
    return trace,'round-limit'

native=scopes['signed-adaboost']['train_adaboost']
cases=0
for length in range(1,8):
    for labels in itertools.product([-1,1],repeat=length):
        x=list(range(length));reference,status=exact_trace(x,labels,6);actual=native(x,labels,6)
        assert actual['status']==status
        assert len(actual['history'])==len(reference)
        for row,(error,before,after,predictions) in zip(actual['history'],reference):
            close(row['error'],error)
            np.testing.assert_allclose(row['after'],list(map(float,after)),rtol=2e-11,atol=2e-11)
        cases+=1
for invalid in [[1.2,-1],[0,1],[float('nan'),1]]:
    try:native([0,1],invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid labels accepted')
counts['changedNativeBoostingLaws'] = cases
for item in data['boosting']:
    trace=item['result'];reference,status=exact_trace(trace['x'],trace['y'],12)
    accepted=[frame for frame in trace['frames'] if frame['status']=='accepted']
    assert len(accepted)==len(reference)
    product=1
    for frame,(error,before,after,predictions) in zip(accepted,reference):
        close(frame['stump']['error'],error)
        np.testing.assert_allclose(frame['after'],list(map(float,after)),rtol=2e-11,atol=2e-11)
        assert frame['stump']['predictions']==predictions
        product*=2*math.sqrt(float(error*(1-error)))
        close(frame['loss'],product);close(frame['bound'],product)
        assert frame['trainingError']<=product+1e-12
counts['visualBoostingTraces'] = len(data['boosting'])

for state in data['oof']:
    rows=[]
    for stage in state['stages']:
        indices=stage['train']
        if state['mode']=='honest':assert not set(indices)&set(stage['held'])
        design=np.c_[np.ones(len(indices)),indices]
        coefficient=np.linalg.lstsq(design,np.array([float(ys[i]) for i in indices]),rcond=None)[0]
        for index in stage['held']:
            nearest=min(indices,key=lambda i:(abs(i-index),i))
            rows.append((index,[float(ys[nearest]),float(coefficient@[1,index])]))
    matrix=np.array([row for _,row in sorted(rows)])
    for fold in range(state['completed']):
        for index in state['stages'][fold]['held']:np.testing.assert_allclose(state['matrix'][index],matrix[index],rtol=1e-12,atol=1e-12)
    if state['completed']==3:
        result=scipy.optimize.minimize_scalar(lambda w:np.mean((w*matrix[:,0]+(1-w)*matrix[:,1]-list(map(float,ys)))**2),bounds=(0,1),method='bounded')
        candidates=[(np.mean((w*matrix[:,0]+(1-w)*matrix[:,1]-list(map(float,ys)))**2),w) for w in [0,1,result.x]]
        close(state['weightNearest'],min(candidates)[1],1e-6)
        full=np.linalg.lstsq(np.c_[np.ones(6),np.arange(6)],list(map(float,ys)),rcond=None)[0]
        close(state['linear'],full@[1,state['query']])
        close(state['nearest'],ys[min(range(6),key=lambda i:(abs(i-state['query']),i))])
counts['oofOwnershipStates'] = len(data['oof'])

artifact=json.loads((ROOT/'src/learn/data/ensemble-prediction-map.json').read_text(encoding="utf-8"))
run=scopes['held-out-comparison']['run']
grid=np.array([[a,b] for b in artifact['coordinates'] for a in artifact['coordinates']])
for name,record in artifact['models'].items():
    actual=run['fitted'][name].predict_proba(grid)[:,1]
    np.testing.assert_allclose(record['probabilities'],actual,rtol=0,atol=1e-13)
    assert np.isfinite(actual).all() and np.min(actual)>=0 and np.max(actual)<=1
counts['nativeGridProbabilities'] = len(grid)*len(artifact['models'])
for key in ['held-out-comparison','changed-protocol']:
    run=scopes[key]['run'];train,valid,test=map(set,[run['train'],run['valid'],run['test']])
    assert not(train&valid or train&test or valid&test)
    assert len(train|valid|test)==360
    selected=min(run['report'],key=lambda name:(run['report'][name]['validationLogLoss'],name))
    assert selected==run['selected']
    pv=run['fitted'][selected].predict_proba(run['x'][run['valid']])[:,1];yv=run['y'][run['valid']]
    cost=lambda threshold:int(np.sum((pv>=threshold)&(yv==0))+4*np.sum((pv<threshold)&(yv==1)))
    assert cost(run['threshold'])==min(map(cost,np.unique(np.r_[0,pv,1])))
    for name,model in run['fitted'].items():
        p=model.predict_proba(run['x'][run['test']])[:,1];y=run['y'][run['test']]
        close(run['report'][name]['testBrier'],sum((F(float(a))-int(b))**2 for a,b in zip(p,y))/len(y))
        clipped=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
        close(run['report'][name]['testLogLoss'],np.mean(-np.where(y==1,np.log(clipped),np.log1p(-clipped))))
counts['completeProtocolAudits'] = 2
counts['invalidModelCases'] = data['invalid']
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'counts':counts,'status':'passed','limits':'Bounded exact fixtures and actual sklearn 1.9.1 fits; no population performance, learner-study or arbitrary-range guarantee.'}
(WORK/'native-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
