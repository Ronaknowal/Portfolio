"""Complete standalone tree programs, exported only after execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]
examples=[]
def add(identifier,title,question,source):
    code=textwrap.dedent(source).strip()+'\n'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,encoding='utf-8',timeout=120)
    if result.returncode or result.stderr: raise RuntimeError((identifier,result.stdout,result.stderr))
    examples.append(dict(id=identifier,title=title,question=question,code=code,expected=result.stdout.strip()))

add('split-ledger','Enumerate every legal root partition','Which threshold has the best gain, and does changing the minimum child size alter eligibility?',r'''
    from fractions import Fraction
    from math import log2
    X=[(1,1),(2,1),(1,3),(2,3),(4,1),(5,1),(4,3),(5,3)]
    y=[0,0,1,1,1,1,0,1]
    def gini(labels):
        p=Fraction(sum(labels),len(labels))
        return 2*p*(1-p)
    parent=gini(y)
    print('parent Gini:',parent)
    for feature in [0,1]:
        values=sorted(set(row[feature] for row in X))
        for low,high in zip(values,values[1:]):
            threshold=Fraction(low+high,2)
            left=[label for row,label in zip(X,y) if row[feature]<=threshold]
            right=[label for row,label in zip(X,y) if row[feature]>threshold]
            gain=parent-Fraction(len(left),8)*gini(left)-Fraction(len(right),8)*gini(right)
            print(f'feature={feature} threshold={threshold} counts={len(left)}/{len(right)} gain={gain} legal at min_leaf2={min(len(left),len(right))>=2}')
    p=.2
    print(f'80/20 Gini={2*p*(1-p):.6f}; entropy={-p*log2(p)-(1-p)*log2(1-p):.6f} bits')
''')

CORE=r'''
import math
import random
import numpy as np

class BinaryTree:
    """Numeric binary CART teaching implementation; no weights/missing/categorical support."""
    def __init__(self,max_depth=4,min_samples_leaf=1,max_features=None,seed=0,allow_zero=True):
        if max_depth is not None and (not isinstance(max_depth,int) or max_depth<0):
            raise ValueError('max_depth must be None or nonnegative integer')
        if not isinstance(min_samples_leaf,int) or min_samples_leaf<1:
            raise ValueError('minimum leaf size must be positive')
        self.max_depth=max_depth
        self.minimum=min_samples_leaf
        self.max_features=max_features
        self.seed=seed
        self.allow_zero=allow_zero

    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y)
        if X.ndim!=2 or not len(X) or X.shape[1]<1 or y.shape!=(len(X),):
            raise ValueError('nonempty matrix and one label per row required')
        if not np.isfinite(X).all() or not np.isin(y,[0,1]).all():
            raise ValueError('finite numeric inputs and binary labels required')
        self.features=X.shape[1]
        count=self.features if self.max_features is None else self.max_features
        if not isinstance(count,int) or not 1<=count<=self.features:
            raise ValueError('max_features must be an integer in 1..d or None')
        rng=random.Random(self.seed)
        def gini(positives,total):
            if not total: return 0.
            p=positives/total
            return 2*p*(1-p)
        def grow(indices,depth):
            positives=int(y[indices].sum()); total=len(indices)
            node={'probability':positives/total,'count':total}
            if positives in [0,total] or total<2*self.minimum or (self.max_depth is not None and depth>=self.max_depth):
                return node
            best=None
            for feature in sorted(rng.sample(range(self.features),count)):
                ordered=sorted(indices,key=lambda index:X[index,feature])
                left_positive=0
                for position in range(total-1):
                    left_positive+=int(y[ordered[position]])
                    left_size=position+1; right_size=total-left_size
                    low=X[ordered[position],feature]; high=X[ordered[position+1],feature]
                    if low==high or min(left_size,right_size)<self.minimum: continue
                    gain=gini(positives,total)-(left_size*gini(left_positive,left_size)+right_size*gini(positives-left_positive,right_size))/total
                    if best is None or gain>best[0]+1e-12:
                        midpoint=float(low/2+high/2)
                        threshold=midpoint if low<=midpoint<high else float(low)
                        best=(gain,feature,threshold,ordered[:left_size],ordered[left_size:])
            if best is None or (not self.allow_zero and best[0]<=1e-12): return node
            gain,feature,threshold,left,right=best
            return dict(node,feature=feature,threshold=threshold,gain=gain,
                left=grow(left,depth+1),right=grow(right,depth+1))
        self.root=grow(list(range(len(X))),0)
        return self

    def predict_proba(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.features or not np.isfinite(X).all():
            raise ValueError('prediction matrix must use the fitted finite feature schema')
        probabilities=[]
        for row in X:
            node=self.root
            while 'left' in node:
                node=node['left'] if row[node['feature']]<=node['threshold'] else node['right']
            probabilities.append([1-node['probability'],node['probability']])
        return np.asarray(probabilities).reshape(-1,2)

    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)  # lowest label wins a tie

class BinaryForest:
    def __init__(self,trees=40,max_depth=4,min_samples_leaf=2,max_features=1,seed=42):
        if not isinstance(trees,int) or trees<1: raise ValueError('positive tree count required')
        self.count=trees; self.depth=max_depth; self.minimum=min_samples_leaf
        self.features=max_features; self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y)
        BinaryTree(max_depth=0).fit(X,y)  # validate before sampling
        rng=random.Random(self.seed)
        self.trees=[]; self.draws=[]
        for _ in range(self.count):
            indices=[rng.randrange(len(X)) for _ in range(len(X))]
            tree=BinaryTree(self.depth,self.minimum,self.features,rng.randrange(2**31)).fit(X[indices],y[indices])
            self.draws.append(indices); self.trees.append(tree)
        return self
    def predict_proba(self,X):
        return np.mean([tree.predict_proba(X) for tree in self.trees],axis=0)
    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)
'''

add('scratch-tree-forest','Build a tree and forest with explicit contracts','Can every terminal node satisfy min_samples_leaf, including a tempting one-row split?',CORE+r'''
def moons(count=400,noise=.25,seed=0):
    if count%2: raise ValueError('this symmetric fixture requires an even count')
    rng=np.random.RandomState(seed); angle=np.linspace(0,np.pi,count//2)
    X=np.vstack([np.c_[np.cos(angle),np.sin(angle)],np.c_[1-np.cos(angle),.5-np.sin(angle)]])
    return X+rng.randn(count,2)*noise,np.repeat([0,1],count//2)
X,y=moons()
indices=np.random.RandomState(0).permutation(len(y)); test,train=indices[:100],indices[100:]
tree=BinaryTree(max_depth=4,min_samples_leaf=2,seed=0).fit(X[train],y[train])
forest=BinaryForest(trees=40,max_depth=4,min_samples_leaf=2,seed=42).fit(X[train],y[train])
def leaf_counts(node):
    return leaf_counts(node['left'])+leaf_counts(node['right']) if 'left' in node else [node['count']]
assert min(leaf_counts(tree.root))>=2
assert all(min(leaf_counts(member.root))>=2 for member in forest.trees)
print(f'tree root: feature={tree.root["feature"]}; midpoint={tree.root["threshold"]:.6f}; gain={tree.root["gain"]:.6f}')
print(f'held-out tree accuracy={np.mean(tree.predict(X[test])==y[test]):.6f}')
print(f'held-out 40-tree accuracy={np.mean(forest.predict(X[test])==y[test]):.6f}')
print('minimum terminal training multiplicity:',min(leaf_counts(tree.root)))
print('scope: numeric finite binary targets, unweighted Gini, seeded bootstrap, per-node feature sampling')
''')

add('xor-and-thresholds','Distinguish representability, greed and threshold conventions','Does zero gain at the root mean a tree cannot represent XOR?',CORE+r'''
X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([0,1,1,0])
for allow in [False,True]:
    tree=BinaryTree(max_depth=2,min_samples_leaf=1,allow_zero=allow).fit(X,y)
    print('allow zero gain:',allow,'predictions:',tree.predict(X).tolist())
# A strictly increasing nonlinear transform preserves training ordering, not midpoint inference.
raw=BinaryTree(max_depth=1).fit([[0],[2]],[0,1])
transformed=BinaryTree(max_depth=1).fit([[0],[4]],[0,1])
print('raw midpoint:',raw.root['threshold'],'squared midpoint:',transformed.root['threshold'])
print('query x=1.2 raw:',raw.predict([[1.2]]).tolist(),'after x squared:',transformed.predict([[1.2**2]]).tolist())
''')

add('regression-leaves','Fit constants inside learned regions','What changes if a new query lies far beyond every observed input?',r'''
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    X=np.arange(6.).reshape(-1,1); y=np.array([1.,1.5,2.,4.,4.5,5.])
    model=DecisionTreeRegressor(max_depth=1,random_state=0).fit(X,y)
    query=np.array([-10.,1.,2.5,3.,20.]).reshape(-1,1)
    print('split:',model.tree_.threshold[0])
    print('query:',query.ravel().tolist())
    print('predictions:',model.predict(query).tolist())
    print('leaf means:',float(y[:3].mean()),float(y[3:].mean()))
    logs=DecisionTreeRegressor(max_depth=1,random_state=0).fit(X,np.log(y))
    print(f'log-target then exp at20: {np.exp(logs.predict([[20.]])[0]):.6f}; still within observed target range')
    print('min/max targets:',float(y.min()),float(y.max()))
''')

add('pruning-path','Grow a tree and inspect its pruning path','Can the best pruned tree jump directly from five leaves to one?',r'''
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    X=np.array([[1,1],[2,1],[1,3],[2,3],[4,1],[5,1],[4,3],[5,3]],float)
    y=np.array([0,0,1,1,1,1,0,1])
    full=DecisionTreeClassifier(random_state=0).fit(X,y)
    path=full.cost_complexity_pruning_path(X,y)
    print('effective alphas:',np.round(path.ccp_alphas,6).tolist())
    print('leaf impurities:',np.round(path.impurities,6).tolist())
    for alpha in [0.,.04,.12,.2]:
        model=DecisionTreeClassifier(random_state=0,ccp_alpha=alpha).fit(X,y)
        leaf=model.tree_.children_left<0
        risk=np.sum(model.tree_.weighted_n_node_samples[leaf]*model.tree_.impurity[leaf])/len(y)
        print(f'alpha={alpha:.2f}; leaves={model.get_n_leaves()}; risk={risk:.6f}; objective={risk+alpha*model.get_n_leaves():.6f}')
    print('Path construction used training labels; selecting alpha still requires development evaluation.')
''')

add('held-out-forest','Compare locked candidates against a baseline','Which settings are selected by validation, and when are final test labels first used?',r'''
    import numpy as np
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score,log_loss
    X,y=make_moons(n_samples=500,noise=.25,random_state=42)
    train,other=train_test_split(np.arange(500),test_size=.4,stratify=y,random_state=42)
    valid,test=train_test_split(other,test_size=.5,stratify=y[other],random_state=43)
    candidates=[]
    for depth in [2,4]:
        model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=5,random_state=42).fit(X[train],y[train])
        candidates.append((f'tree depth{depth}',model))
    for leaf in [1,5]:
        model=RandomForestClassifier(n_estimators=100,min_samples_leaf=leaf,max_features='sqrt',oob_score=True,n_jobs=1,random_state=42).fit(X[train],y[train])
        candidates.append((f'forest min_leaf{leaf}',model))
    validation=[]
    for name,model in candidates:
        loss=log_loss(y[valid],model.predict_proba(X[valid]))
        validation.append((loss,name,model)); print(f'{name}: validation log loss={loss:.6f}')
    _,name,chosen=min(validation,key=lambda item:(item[0],item[1]))
    baseline=DummyClassifier(strategy='prior').fit(X[train],y[train])
    print('sizes:',len(train),len(valid),len(test),'selected:',name)
    for label,model in [('baseline',baseline),(name,chosen)]:
        print(f'{label}: test accuracy={accuracy_score(y[test],model.predict(X[test])):.6f}; log loss={log_loss(y[test],model.predict_proba(X[test])):.6f}')
    if hasattr(chosen,'oob_score_'): print(f'training OOB accuracy={chosen.oob_score_:.6f}; different metric/set from validation log loss')
    print('One synthetic dataset; no universal ranking. Test labels selected no model or threshold.')
''')

add('oob-accounting','Reconstruct out-of-bag predictions from sample identities','Does a row omitted by one tree necessarily have enough OOB votes across the forest?',r'''
    import numpy as np
    from sklearn.datasets import make_moons
    from sklearn.ensemble import RandomForestClassifier
    X,y=make_moons(n_samples=80,noise=.25,random_state=12)
    model=RandomForestClassifier(n_estimators=40,max_depth=3,oob_score=True,n_jobs=1,random_state=4).fit(X,y)
    sums=np.zeros((len(y),2)); counts=np.zeros(len(y),int)
    for tree,draws in zip(model.estimators_,model.estimators_samples_):
        omitted=np.setdiff1d(np.arange(len(y)),np.unique(draws))
        sums[omitted]+=tree.predict_proba(X[omitted]); counts[omitted]+=1
    covered=counts>0
    reconstructed=sums[covered]/counts[covered,None]
    assert np.allclose(reconstructed,model.oob_decision_function_[covered])
    assert np.allclose(np.mean([tree.predict_proba(X) for tree in model.estimators_],axis=0),model.predict_proba(X))
    print('OOB counts min/max:',int(counts.min()),int(counts.max()))
    print('first five OOB positive probabilities:',np.round(reconstructed[:5,1],6).tolist())
    print(f'omission probability n80={(1-1/80)**80:.6f}; limit={np.exp(-1):.6f}')
    print('OOB averages and forest mean probabilities match the fitted library model.')
''')

add('categories-missing-weights','Inspect encoding, missing routes and weighted leaves','Does assigning an integer to a category make its order meaningful?',r'''
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import OneHotEncoder
    categories=np.array(['red','green','blue','red','green','blue']).reshape(-1,1)
    y=np.array([0,1,0,0,1,0])
    encoding=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(categories)
    tree=DecisionTreeClassifier(max_depth=1,random_state=0).fit(encoding.transform(categories),y)
    queries=np.array(['red','green','blue','purple']).reshape(-1,1)
    print('encoded columns:',encoding.get_feature_names_out().tolist())
    print('category predictions:',tree.predict(encoding.transform(queries)).tolist())
    print('unseen purple representation:',encoding.transform(queries)[-1].tolist())
    missing=DecisionTreeClassifier(max_depth=1,random_state=0).fit([[0.],[1.],[6.],[np.nan]],[0,0,1,1])
    print('learned missing route probability:',missing.predict_proba([[np.nan]]).tolist())
    weighted=DecisionTreeClassifier(min_samples_leaf=4,random_state=0).fit([[0.],[0.],[0.],[0.]],[0,0,0,1],sample_weight=[1,1,1,6])
    print('weighted unsplit class probabilities:',np.round(weighted.predict_proba([[0.]]),6).tolist())
    print('Weighted mass is not the original empirical class frequency; unseen-category behavior needs a declared policy.')
''')

add('importance-report','Inspect model reliance on held-out data','Can a perfectly informative duplicate receive little individual permutation importance?',r'''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    rng=np.random.default_rng(23)
    signal=rng.integers(0,2,400)
    X=np.column_stack([signal,signal,rng.random(400)])
    y=signal.copy(); y[rng.choice(400,40,replace=False)]^=1
    train,test=np.arange(250),np.arange(250,400)
    model=RandomForestClassifier(n_estimators=80,min_samples_leaf=2,n_jobs=1,random_state=9).fit(X[train],y[train])
    result=permutation_importance(model,X[test],y[test],scoring='accuracy',n_repeats=8,random_state=11,n_jobs=1)
    baseline=model.score(X[test],y[test]); drops=[]
    for _ in range(8):
        changed=X[test].copy(); order=rng.permutation(len(test)); changed[:,:2]=changed[order,:2]
        drops.append(baseline-model.score(changed,y[test]))
    print(f'held-out baseline accuracy={baseline:.6f}')
    print('MDI [signal,copy,noise]:',np.round(model.feature_importances_,6).tolist())
    print('individual permutation mean:',np.round(result.importances_mean,6).tolist())
    print(f'joint signal/copy permutation mean={np.mean(drops):.6f}')
    print('Metric/model/split-specific reliance; joint shuffling preserves the duplicate relationship. No causal claim.')
''')

add('changed-capstone','A complete changed-data comparison','Can you select depth and leaf size without consulting final test accuracy?',r'''
    import numpy as np
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import log_loss,confusion_matrix
    X,y=make_moons(n_samples=500,noise=.35,random_state=17)
    train,other=train_test_split(np.arange(500),test_size=.4,stratify=y,random_state=42)
    valid,test=train_test_split(other,test_size=.5,stratify=y[other],random_state=43)
    candidates=[]
    for family in [RandomForestClassifier,ExtraTreesClassifier]:
        for leaf in [2,8]:
            model=family(n_estimators=80,min_samples_leaf=leaf,max_features='sqrt',random_state=3,n_jobs=1).fit(X[train],y[train])
            loss=log_loss(y[valid],model.predict_proba(X[valid]))
            name=f'{family.__name__} leaf{leaf}'
            print(f'validation {name}: {loss:.6f}'); candidates.append((loss,name,model))
    _,name,model=min(candidates,key=lambda row:(row[0],row[1]))
    baseline=DummyClassifier(strategy='prior').fit(X[train],y[train])
    print('frozen selection:',name)
    print(f'test baseline log loss={log_loss(y[test],baseline.predict_proba(X[test])):.6f}; model={log_loss(y[test],model.predict_proba(X[test])):.6f}')
    print('test confusion:',confusion_matrix(y[test],model.predict(X[test]),labels=[0,1]).tolist())
    print('models remain fit only on training; 100 synthetic test rows; no population or calibration guarantee')
''')

destination=ROOT/'src/learn/data/decision-tree-examples.js'
destination.write_text('// Executed standalone programs. Regenerate with scripts/generate-decision-tree-examples.py.\nexport const decisionTreeExamples = '+json.dumps(examples,ensure_ascii=False,indent=2)+';\n',encoding='utf-8')
record={'python':sys.version,'programs':[{'id':example['id'],'codeSha256':hashlib.sha256(example['code'].encode()).hexdigest(),'stdout':example['expected']} for example in examples]}
(ROOT/'scratch/decision-tree-native').mkdir(parents=True,exist_ok=True)
(ROOT/'scratch/decision-tree-native/example-runs.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(f'Exported {len(examples)} executed standalone tree programs.')
