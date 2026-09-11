"""Execute exact displayed definitions, avoiding their already-verified full demonstrations."""
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

out=Path('scratch/tree-knn-independent')
data=json.loads((out/'native-input.json').read_text())

def definitions(example):
    parsed=ast.parse(example['code'])
    selected=ast.Module(body=[node for node in parsed.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef))],type_ignores=[])
    namespace={}
    exec(compile(selected,f"actual displayed {example['id']}",'exec'),namespace)
    return namespace

tree=definitions(data['tree'])
Tree,Forest=tree['BinaryTree'],tree['BinaryForest']
x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y=np.array([0,1,1,0])
blocked=Tree(max_depth=2,allow_zero=False).fit(x,y)
assert 'left' not in blocked.root
solved=Tree(max_depth=None,allow_zero=True).fit(x,y)
assert np.array_equal(solved.predict(x),y)
forest=Forest(trees=3,max_depth=2,min_samples_leaf=1,max_features=1).fit(x,y)
old=forest.predict_proba(x).copy()
forest.fit(x,np.ones(4))
assert len(forest.trees)==3 and np.array_equal(forest.predict_proba(x),np.tile([0.,1.],(4,1)))
forest.fit(x,y)
assert np.array_equal(forest.predict_proba(x),old)
# Representable adjacent endpoints must remain separable under the stored threshold.
for value in [0.,1.,-1.,1e200]:
    after=np.nextafter(value,np.inf)
    fitted=Tree(max_depth=1).fit([[value],[after]],[0,1])
    assert np.array_equal(fitted.predict([[value],[after]]),[0,1])

neighbors=definitions(next(e for e in data['knn'] if e['id']=='scratch-estimator'))['LocalNeighbors']
kd=definitions(next(e for e in data['knn'] if e['id']=='kd-tree'))
queries=0
for scale in [1.,1e-200,1e150]:
    coordinates=scale*np.array([[0.,0.],[1.,0.],[0.,2.],[3.,3.]])
    labels=np.array(['A','B','C','D'])
    fitted=neighbors(k=1).fit(coordinates,labels)
    index=kd['build_tree'](coordinates)
    for query in coordinates:
        # Rescaling first yields the exact same rankings without under/overflow in the oracle.
        oracle=np.hypot.reduce(coordinates/scale-query/scale,axis=1)
        expected=int(np.argmin(oracle))
        assert fitted.predict([query])[0]==labels[expected]
        answer,_=kd['nearest'](coordinates,index,query)
        assert answer[1]==expected
        queries+=1
duplicate=neighbors(k=3,weights='distance').fit([[0.],[0.],[2.]],['A','B','B'])
assert np.array_equal(duplicate.predict_proba([[0.]]),[[.5,.5]])
reg=neighbors(k=2,weights='distance',task='regression').fit([[0.],[4.]],[10.,22.])
assert np.allclose(reg.predict([[1.]]),[13.])
result={'checkedAt':datetime.now(timezone.utc).isoformat(),'actualTreeCases':8,'actualNeighborKdQueries':queries,'duplicateAndRegressionCases':2,'scope':'Exact displayed imports/definitions with changed inputs; full example demos not repeated.'}
(out/'native-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
