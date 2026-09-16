"""Bounded author arithmetic and real-data experiment; not a production lab."""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction import DictVectorizer

HERE = Path(__file__).parent

def infer(emissions, transitions):
    length, labels = emissions.shape
    forward = np.empty_like(emissions)
    backward = np.zeros_like(emissions)
    forward[0] = emissions[0]
    for t in range(1, length):
        forward[t] = emissions[t] + np.logaddexp.reduce(forward[t-1, :, None] + transitions, axis=0)
    log_partition = np.logaddexp.reduce(forward[-1])
    for t in range(length-2, -1, -1):
        backward[t] = np.logaddexp.reduce(transitions + emissions[t+1] + backward[t+1], axis=1)
    nodes = np.exp(forward + backward - log_partition)
    edges = np.array([np.exp(forward[t-1, :, None] + transitions + emissions[t] + backward[t] - log_partition) for t in range(1,length)])
    best = emissions[0].copy()
    parents = []
    for t in range(1,length):
        candidates = best[:,None] + transitions
        parents.append(candidates.argmax(axis=0))
        best = candidates.max(axis=0) + emissions[t]
    path = [int(best.argmax())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    return float(log_partition), nodes, edges, path[::-1], forward, backward

def features(words):
    return [{'bias':1., 'word':w.lower(), 'suffix':w.lower()[-2:], 'capital':float(w.istitle()),
             'previous':words[i-1].lower() if i else '<START>',
             'next':words[i+1].lower() if i+1<len(words) else '<END>'} for i,w in enumerate(words)]

def run_real():
    rows=json.loads((HERE/'ewt-sequences.json').read_text(encoding='utf-8'))
    tag=lambda t: 0 if t in ('NOUN','PROPN') else 1 if t in ('VERB','AUX') else 2
    vectorizer=DictVectorizer(sparse=False)
    vectorizer.fit([f for r in rows if r['split']=='train' for f in features(r['tokens'])])
    groups={s:[(vectorizer.transform(features(r['tokens'])),np.array([tag(t) for t in r['upos']]),r) for r in rows if r['split']==s] for s in ['train','dev','test']}
    feature_count=len(vectorizer.feature_names_)
    results={}
    for structured in [False,True]:
        def objective(theta):
            weights=theta[:feature_count*3].reshape(feature_count,3)
            transitions=theta[feature_count*3:].reshape(3,3) if structured else np.zeros((3,3))
            loss=0.; grad_w=np.zeros_like(weights); grad_a=np.zeros((3,3))
            for x,y,_ in groups['train']:
                emissions=x@weights
                logz,nodes,edges,_,_,_=infer(emissions,transitions)
                loss += logz-emissions[np.arange(len(y)),y].sum()-transitions[y[:-1],y[1:]].sum()
                residual=nodes.copy(); residual[np.arange(len(y)),y]-=1
                grad_w += x.T@residual
                if structured:
                    grad_a += edges.sum(axis=0)
                    np.add.at(grad_a,(y[:-1],y[1:]),-1.)
            gradient=np.r_[grad_w.ravel(),grad_a.ravel()] if structured else grad_w.ravel()
            return loss/len(groups['train'])+.05*np.dot(theta,theta), gradient/len(groups['train'])+.1*theta
        initial=np.zeros(feature_count*3+(9 if structured else 0))
        fit=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-11,'gtol':1e-7})
        weights=fit.x[:feature_count*3].reshape(feature_count,3)
        transitions=fit.x[feature_count*3:].reshape(3,3) if structured else np.zeros((3,3))
        report={'success':bool(fit.success),'iterations':int(fit.nit),'objective':float(fit.fun),'feature_count':feature_count,'transitions':transitions.tolist()}
        for split in ['dev','test']:
            correct=total=whole=0; outputs=[]; confusion=np.zeros((3,3),dtype=int)
            for x,y,row in groups[split]:
                logz,nodes,_,path,_,_=infer(x@weights,transitions)
                correct+=int((y==path).sum()); total+=len(y); whole+=int(np.array_equal(y,path))
                np.add.at(confusion,(y,np.array(path)),1)
                outputs.append({'id':row['id'],'tokens':row['tokens'],'gold':y.tolist(),'prediction':path,'marginals':nodes.tolist()})
            report[split]={'token_accuracy':correct/total,'correct':correct,'total':total,'sequence_exact':whole/len(groups[split]),'confusion_true_rows':confusion.tolist(),'outputs':outputs}
        results['chain' if structured else 'independent']=report
    return results

def run_tiny():
    unary=np.log([[3.,1.],[1.,2.]])
    pair=np.log([[1.,4.],[1.,1.]])
    out=infer(unary,pair)
    paths=list(itertools.product(range(2),repeat=2))
    masses=[float(np.exp(unary[0,p[0]]+pair[p]+unary[1,p[1]])) for p in paths]
    result={'paths':paths,'masses':masses,'logz':out[0],'marginals':out[1].tolist(),'best':out[3],'forward':np.exp(out[4]).tolist(),'backward':np.exp(out[5]).tolist()}
    result['independent']= {'logz':infer(unary,np.zeros((2,2)))[0],'best':infer(unary,np.zeros((2,2)))[3]}
    result['changed_unary'] = {'best':infer(np.log([[3.,1.],[6.,2.]]),pair)[3], 'marginals':infer(np.log([[3.,1.],[6.,2.]]),pair)[1].tolist()}
    result['sequence_vs_marginal']={'mass':[[.35,.34],[.01,.30]],'sequence':[0,0],'marginal':[0,1]}
    # Finite difference of logZ with respect to the 0->1 score is its edge marginal.
    epsilon=1e-6; plus=pair.copy(); minus=pair.copy(); plus[0,1]+=epsilon; minus[0,1]-=epsilon
    derivative=(infer(unary,plus)[0]-infer(unary,minus)[0])/(2*epsilon)
    result['logz_derivative']={'finite_difference':derivative,'edge_marginal':float(out[2][0,0,1])}
    result['practice']={'masses':[8,4,6,12],'partition':30,'best':[1,1],'first_A':12/30,'second_A':14/30}
    assert abs(np.exp(out[0])-sum(masses))<1e-10
    assert np.allclose(out[1].sum(axis=1),1)
    assert abs(derivative-out[2][0,0,1])<1e-8
    return result

if __name__=='__main__':
    result={'tiny':run_tiny(),'real':run_real()}
    (HERE/'checked-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'tiny':result['tiny'],'real':{name:{key:value for key,value in info.items() if key not in ['dev','test']}|{split:{key:value for key,value in info[split].items() if key!='outputs'} for split in ['dev','test']} for name,info in result['real'].items()}},indent=2))
