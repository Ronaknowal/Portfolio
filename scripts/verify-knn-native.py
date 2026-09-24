"""Independent numeric/protocol checks of production KNN examples and model output."""
import contextlib
import hashlib
import io
import json
import pathlib
import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
payload = json.loads((ROOT/'scratch/knn-native/model-payload.json').read_text())
ids = np.array(['A1','A2','A3','B1','B2','B3','C1','C2'])
labels = np.array(list('AAABBBCC'))
points = np.array([[1,2],[1.5,1.8],[1.2,2.5],[5,5],[5.5,4.8],[4.8,5.2],[3,3],[3.2,2.8]])
for case in payload['neighbors']:
    delta=np.abs(points-case['query'])
    distance={'euclidean':np.sqrt((delta**2).sum(axis=1)),
              'manhattan':delta.sum(axis=1),'maximum':delta.max(axis=1)}[case['metric']]
    order=np.lexsort((ids,np.round(distance,12)))[:case['k']]
    chosen=distance[order]
    weight=np.ones(len(order))
    if case['weights']=='distance':
        weight=(chosen==0).astype(float) if np.any(chosen==0) else chosen.min()/chosen
    weight/=weight.sum()
    actual=case['result']
    assert [row['id'] for row in actual['neighbors']]==ids[order].tolist()
    probabilities=[weight[labels[order]==label].sum() for label in 'ABC']
    np.testing.assert_allclose([row['probability'] for row in actual['votes']],probabilities,atol=1e-12)
    np.testing.assert_allclose([row['distance'] for row in actual['neighbors']],chosen,atol=1e-12)
for case in payload['regression']:
    x=np.arange(6.)
    distances=np.abs(x-case['query'])
    order=np.argsort(distances,kind='stable')[:case['k']]
    weights=np.ones(len(order))
    if case['weights']=='distance':
        chosen=distances[order]
        weights=(chosen==0).astype(float) if np.any(chosen==0) else chosen.min()/chosen
    expected=np.average(x[order]**2,weights=weights)
    assert abs(expected-case['result']['prediction'])<1e-12
for case in payload['searches']:
    distance=np.linalg.norm(points-case['query'],axis=1)
    assert abs(case['result']['best']['distance']-distance.min())<1e-12
    for event in case['result']['events']:
        if event['kind']=='prune': assert event['lowerBound']>event['best']['distance']
unit_data=np.array([[1,100],[3,1000],[5,1100]],float)
scaler=StandardScaler().fit(unit_data)
for case in payload['units']:
    row=case['result']
    np.testing.assert_allclose(row['scales'],scaler.scale_)
    data=scaler.transform(unit_data) if case['scale'] else unit_data
    query=scaler.transform([[1.2,600]]) if case['scale'] else np.array([[1.2,600]])
    winner='RST'[np.linalg.norm(data-query,axis=1).argmin()]
    assert row['nearest']['id']==winner
for case in payload['volumes']:
    assert abs(case['result']['side']**case['dimension']-case['fraction'])<1e-14

examples_text=(ROOT/'src/learn/data/knn-examples.js').read_text(encoding='utf-8')
examples=json.loads(examples_text.removeprefix('export const knnExamples = ').removesuffix(';\n'))
actual_runs=json.loads((ROOT/'scratch/knn-native/example-runs.json').read_text())['runs']
for example in examples:
    run=next(row for row in actual_runs if row['id']==example['id'])
    assert hashlib.sha256(example['code'].encode()).hexdigest()==run['codeSha256']
    assert hashlib.sha256(example['expected'].encode()).hexdigest()==run['outputSha256']
    assert run['returncode']==0

scope={}
with contextlib.redirect_stdout(io.StringIO()):
    exec(next(row['code'] for row in examples if row['id']=='scratch-estimator'),scope)
estimator=scope['LocalNeighbors']
rng=np.random.default_rng(71)
library_cases=0
for n,d in [(7,1),(11,3),(17,5)]:
    X=rng.normal(size=(n,d)); y=rng.integers(0,3,size=n); target=rng.normal(size=n)
    queries=rng.normal(size=(4,d))
    for k in [1,3,n]:
        for weights in ['uniform','distance']:
            own=estimator(k,weights).fit(X,y)
            reference=KNeighborsClassifier(n_neighbors=k,weights=weights,algorithm='brute').fit(X,y)
            np.testing.assert_allclose(own.predict_proba(queries),reference.predict_proba(queries),atol=1e-12)
            own_reg=estimator(k,weights,'regression').fit(X,target)
            reference_reg=KNeighborsRegressor(n_neighbors=k,weights=weights,algorithm='brute').fit(X,target)
            np.testing.assert_allclose(own_reg.predict(queries),reference_reg.predict(queries),atol=1e-12)
            assert own.predict_proba(np.empty((0,d))).shape==(0,len(own.classes_))
            library_cases+=1
for invalid in [lambda: estimator(0),lambda: estimator(True),lambda: estimator().fit([[0],[1]],[0,1]),
                lambda: estimator(1).fit([[float('nan')]],[1]),lambda: estimator(1).fit([[0]],[float('nan')])]:
    try: invalid()
    except ValueError: pass
    else: raise AssertionError('invalid estimator contract accepted')

record={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'neighborCases':len(payload['neighbors']),'regressionCases':len(payload['regression']),
        'exactSearchQueries':len(payload['searches']),'changedLibraryCases':library_cases,
        'standalonePrograms':len(examples),'note':'Reused unchanged actual program executions; new independent NumPy/sklearn checks compare mechanisms and changed inputs.',
        'sourceHashes':{str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in ['src/learn/data/knn-models.js','src/learn/data/knn-examples.js']}}
(ROOT/'docs/teaching/evidence/knn-native-review.json').write_text(json.dumps(record,indent=2)+'\n')
print('Independent KNN distances, weights, scaling, KD bounds, regression and complete estimator checks passed.')
