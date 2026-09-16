"""Bounded AutoML evidence: fixed conditional candidates and protected feature groups."""
from pathlib import Path
import json
import platform
import warnings
import numpy as np
import scipy
from scipy.special import ndtr, softmax
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent

def candidates():
    result=[]
    for scale in [False,True]:
        for c in [.1,1.]:
            model=LogisticRegression(C=c,max_iter=1000,solver='lbfgs',tol=1e-8)
            if scale: model=make_pipeline(StandardScaler(),model)
            result.append({'id':f'logistic-{"standard" if scale else "raw"}-c{c:g}',
                           'family':'logistic','scale':scale,'c':c,'model':model})
    for depth in [2,5]:
        result.append({'id':f'tree-depth{depth}','family':'tree','depth':depth,
                       'model':DecisionTreeClassifier(max_depth=depth,random_state=74)})
    for neighbors in [3,9]:
        result.append({'id':f'neighbors-standard-k{neighbors}','family':'neighbors','neighbors':neighbors,
                       'model':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=neighbors))})
    for widths in [(8,),(16,),(8,8)]:
        result.append({'id':'mlp-tanh-'+'x'.join(map(str,widths)),'family':'mlp','widths':list(widths),
                       'parameter_count':sum((a+1)*b for a,b in zip((4,)+widths,widths+(1,))),
                       'model':make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=widths,
                           activation='tanh',solver='lbfgs',alpha=.01,max_iter=1000,
                           tol=1e-7,random_state=74))})
    return result

def data_roles():
    raw=np.loadtxt(ROOT/'banknote-data.csv',delimiter=',')
    x,y=raw[:,:4],raw[:,4].astype(int)
    unique,first,groups,counts=np.unique(x,axis=0,return_index=True,return_inverse=True,return_counts=True)
    for group in np.flatnonzero(counts>1):
        if len(np.unique(y[groups==group]))!=1: raise ValueError('Resolve conflicting labels within a feature group.')
    group_y=y[first]
    development_groups,rest=train_test_split(np.arange(len(unique)),train_size=900,
                                             stratify=group_y,random_state=71)
    inspection_groups,reserve_groups=train_test_split(rest,train_size=200,
                                             stratify=group_y[rest],random_state=72)
    rows=lambda group_ids:np.flatnonzero(np.isin(groups,group_ids))
    development,inspection,reserve=map(rows,[development_groups,inspection_groups,reserve_groups])
    folds=[]
    for fitting,validation in StratifiedKFold(3,shuffle=True,random_state=73).split(development_groups,group_y[development_groups]):
        folds.append((rows(development_groups[fitting]),rows(development_groups[validation])))
    return x,y,groups,counts,development,inspection,reserve,folds

def expected_improvement(best,mean,deviation):
    if deviation==0:return max(best-mean,0.)
    z=(best-mean)/deviation
    return (best-mean)*ndtr(z)+deviation*np.exp(-z*z/2)/np.sqrt(2*np.pi)

def constructed_fixtures():
    alpha=np.array([np.log(2),0.,0.]); operations=np.array([0.,2.,-2.])
    mixture=softmax(alpha)@operations
    gradient=(mixture-1)*softmax(alpha)*(operations-mixture)
    updated=alpha-.4*gradient
    curves=np.array([[.10,.10,.10],[.12,.09,.08],[.13,.08,.07],[.14,.07,.02],
                     [.20,.15,.09],[.25,.20,.10],[.30,.25,.20],[.35,.30,.25],[.40,.30,.20]])
    ids=list(range(9)); survivors=[ids.copy()]
    for stage in [0,1]:
        ids=sorted(ids,key=lambda i:(curves[i,stage],i))[:len(ids)//3]
        survivors.append(ids.copy())
    return {'conditional_counts':{'logistic':4,'tree':2,'neighbors':2,'mlp':3,'total':11},
            'ei':{label:expected_improvement(.4,mean,sd) for label,mean,sd in [('A',.35,.02),('B',.4,.2),('C',.5,0.)]},
            'mixture':{'logits':alpha.tolist(),'operation_outputs':operations.tolist(),'probabilities':softmax(alpha).tolist(),
                       'target':1,'output':float(mixture),'half_squared_loss':float((mixture-1)**2/2),
                       'gradient':gradient.tolist(),'step_size':.4,'updated_logits':updated.tolist(),
                       'updated_output':float(softmax(updated)@operations),
                       'translation_null_probabilities':softmax(alpha+5).tolist()},
            'discretization':{'outputs':[2,-2],'probabilities':[.5,.5],'target':0,'mixed_loss':0.,'selected_loss':2.},
            'bilevel_scalar':{'alpha':.2,'w':0.,'xi':.1,'w_prime':.02,'first_order_gradient':0.,
                              'one_step_gradient':-.098,'exact_inner_optimum':.2,'exact_outer_gradient':-.8},
            'halving':{'resource':[1,3,9],'curves':curves.tolist(),'survivors':survivors,
                        'selected':ids[0],'selected_final_loss':float(curves[ids[0],2]),
                        'full_budget_best':int(np.argmin(curves[:,2])),
                        'restart_cost':27,'resume_cost':21,'all_full_cost':81},
            'portfolio':{'old_task_losses':[[.1,.5],[.5,.1],[.25,.25]],
                         'new_task_losses':[.4,.4,.2]},
            'pareto':{'labels':['A','B','C','D','E'],'latency_ms':[2,4,8,5,3],
                       'accuracy':[.90,.94,.96,.93,.89],'nondominated':['A','B','C']},
            'additional_checks': {
                'practice_conditional_count': 3*2 + 4 + 2 + 2*2,
                'practice_cv_fits': (3*2 + 4 + 2 + 2*2)*4,
                'practice_halving_restart': 4*1 + 2*2 + 1*4,
                'practice_halving_continue': 4*1 + 2*(2-1) + 1*(4-2),
                'practice_neural_parameter_blocks': [(5+1)*6, (6+1)*3, (3+1)*1],
                'practice_ei': [expected_improvement(.3,.25,0), expected_improvement(.3,.3,.1)],
                'practice_mixture_output': .5*3 + .5*(-1),
                'practice_mixture_gradient': [1*.5*(3-1), 1*.5*(-1-1)],
                'practice_bilevel': {'direct':0, 'one_step':(.1-2)*.2, 'exact':.5-2},
                'stationary_one_step_gradient': (.2-1)*.1,
                'naswot_kernel': [[3,1],[1,3]],
                'naswot_determinant': 3*3-1*1,
                'naswot_log_determinant': float(np.log(8)),
                'identical_code_determinant': 3*3-3*3,
                'old_task_best_single_loss': .25,
                'old_task_portfolio_best_loss': .1,
                'new_task_portfolio_best_loss': .4,
                'new_task_alternative_loss': .2,
            }}

def calculate():
    x,y,groups,counts,development,inspection,reserve,folds=data_roles()
    configs=candidates(); records=[]
    for config in configs:
        oof=np.full(len(y),np.nan); fold_scores=[]; warning_messages=[]
        for fitting,validation in folds:
            model=clone(config['model'])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(x[fitting],y[fitting])
            warning_messages.extend([str(w.message) for w in caught])
            prediction=model.predict(x[validation]);oof[validation]=prediction
            fold_scores.append(float(accuracy_score(y[validation],prediction)))
        records.append({k:v for k,v in config.items() if k!='model'} | {
            'fold_accuracy':fold_scores,'mean_fold_accuracy':float(np.mean(fold_scores)),
            'pooled_oof_accuracy':float(accuracy_score(y[development],oof[development])),
            'oof_predictions':oof[development].astype(int).tolist(),'warnings':warning_messages})
    selected=max(range(len(records)),key=lambda i:records[i]['mean_fold_accuracy'])
    final=[]
    for name,index in [('selected',selected),('declared_logistic_baseline',3)]:
        model=clone(configs[index]['model'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(x[development],y[development])
        prediction=model.predict(x[inspection])
        final.append({'role':name,'id':configs[index]['id'],'accuracy':float(accuracy_score(y[inspection],prediction)),
                      'confusion':confusion_matrix(y[inspection],prediction,labels=[0,1]).tolist(),
                      'predictions':prediction.tolist(),'warnings':[str(w.message) for w in caught]})
    order=np.random.default_rng(75).permutation(len(configs))
    values=np.array([r['mean_fold_accuracy'] for r in records])
    return {'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
            'groups':{'source_row_group':groups.tolist(),'counts':counts.tolist(),'unique':len(counts),
                      'repeated_group_ids':np.flatnonzero(counts>1).tolist()},
            'roles':{'development_ids':development.tolist(),'inspection_ids':inspection.tolist(),'reserve_ids':reserve.tolist(),
                     'development_labels':y[development].tolist(),'inspection_labels':y[inspection].tolist(),
                     'folds':[{'fitting_ids':a.tolist(),'validation_ids':b.tolist()} for a,b in folds],
                     'row_counts':{name:len(ids) for name,ids in [('development',development),('inspection',inspection),('reserve',reserve)]},
                     'positive_counts':{name:int(y[ids].sum()) for name,ids in [('development',development),('inspection',inspection),('reserve',reserve)]}},
            'candidates':records,'selected_id':records[selected]['id'],'tie_rule':'first in declared candidate order',
            'search_order':order.tolist(),'search_best_so_far':np.maximum.accumulate(values[order]).tolist(),
            'inspection':final,'always_majority_inspection_accuracy':float(np.mean(y[inspection]==np.argmax(np.bincount(y[development])))),
            'constructed':constructed_fixtures(),'fits':len(configs)*3+2,'reserve_scored':False}

if __name__=='__main__':
    with threadpool_limits(limits=1):result=calculate()
    (ROOT/'calculated-inputs.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'roles':result['roles']['row_counts'],'positive_counts':result['roles']['positive_counts'],
                      'candidates':[{k:v for k,v in c.items() if k!='oof_predictions'} for c in result['candidates']],
                      'selected_id':result['selected_id'],'inspection':[{k:v for k,v in c.items() if k!='predictions'} for c in result['inspection']],
                      'search_order':result['search_order'],'best_so_far':result['search_best_so_far'],
                      'fits':result['fits'],'constructed':result['constructed']},indent=2))
