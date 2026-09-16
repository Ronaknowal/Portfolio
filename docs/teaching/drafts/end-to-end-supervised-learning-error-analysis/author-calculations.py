"""Bounded author inputs: an offline, locked Wine comparison; not website code."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss, confusion_matrix

directory = Path(__file__).resolve().parent
data = np.genfromtxt(directory / 'wine.csv', delimiter=',', names=True)
features = ['alcohol', 'color_intensity']
X = np.column_stack([data[name] for name in features])
extended = np.column_stack([X, data['flavanoids']])
y = data['cultivar'].astype(int)
ids = np.arange(len(y))
development, test = train_test_split(ids, test_size=36, stratify=y, random_state=21)
train, valid = train_test_split(development, test_size=36, stratify=y[development], random_state=22)
models = {
    'majority': (make_pipeline(StandardScaler(), DummyClassifier(strategy='prior')), X),
    'linear_two': (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)), X),
    'forest_two': (RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=21, n_jobs=1), X),
    'linear_three': (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)), extended),
}
output = {'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
          'dataSha256': hashlib.sha256((directory / 'wine.csv').read_bytes()).hexdigest(),
          'splits': {name: (indices + 1).tolist() for name, indices in [('train',train),('validation',valid),('test',test)]},
          'candidates': {}, 'validationRows': []}
for name, (model, matrix) in models.items():
    model.fit(matrix[train], y[train])
    prediction = model.predict(matrix[valid])
    probability = model.predict_proba(matrix[valid])
    output['candidates'][name] = {
        'trainBalancedAccuracy': balanced_accuracy_score(y[train], model.predict(matrix[train])),
        'validationBalancedAccuracy': balanced_accuracy_score(y[valid], prediction),
        'validationAccuracy': accuracy_score(y[valid], prediction),
        'validationLogLoss': log_loss(y[valid], probability),
        'validationConfusion': confusion_matrix(y[valid], prediction).tolist(),
        'validationPrediction': prediction.tolist(),
        'validationProbability': probability.tolist(),
    }
    for position, index in enumerate(valid):
        output['validationRows'].append({'model':name, 'id':int(index+1), 'actual':int(y[index]),
            'prediction':int(prediction[position]), 'alcohol':float(X[index,0]),
            'colorIntensity':float(X[index,1]), 'flavanoids':float(extended[index,2]),
            'probabilityOfActual':float(probability[position,y[index]])})

eligible = ['linear_two', 'forest_two', 'linear_three']
winner = max(eligible, key=lambda name: output['candidates'][name]['validationBalancedAccuracy'])
model, matrix = models[winner]
# The teaching protocol evaluates the selected already-fitted model, with no refit.
probability = model.predict_proba(matrix[test])
prediction = model.predict(matrix[test])
output['selected'] = winner
output['test'] = {'accuracy':accuracy_score(y[test],prediction),
                  'balancedAccuracy':balanced_accuracy_score(y[test],prediction),
                  'logLoss':log_loss(y[test],probability),
                  'confusion':confusion_matrix(y[test],prediction).tolist(),
                  'predictions':prediction.tolist()}
output['slices'] = {}
for name in ['linear_two','forest_two','linear_three']:
    rows = [row for row in output['validationRows'] if row['model']==name]
    output['slices'][name] = {}
    for label, predicate in [('color<4',lambda row:row['colorIntensity']<4),
                             ('color>=4',lambda row:row['colorIntensity']>=4),
                             ('class1',lambda row:row['actual']==1)]:
        selected = [row for row in rows if predicate(row)]
        output['slices'][name][label] = {'n':len(selected),'errors':sum(row['prediction']!=row['actual'] for row in selected),
                                         'ids':[row['id'] for row in selected]}
output['paired'] = {name: {'fixed':sum((np.array(output['candidates']['linear_two']['validationPrediction'])!=y[valid]) & (np.array(output['candidates'][name]['validationPrediction'])==y[valid])),
                          'broken':sum((np.array(output['candidates']['linear_two']['validationPrediction'])==y[valid]) & (np.array(output['candidates'][name]['validationPrediction'])!=y[valid]))} for name in ['forest_two','linear_three']}
output['paired'] = {name:{key:int(value) for key,value in values.items()} for name,values in output['paired'].items()}
n, k, z = 36, 35, 1.96
proportion = k / n
center = (proportion + z*z/(2*n)) / (1+z*z/n)
radius = z*math.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n)) / (1+z*z/n)
output['wilsonIllustration'] = [center-radius, center+radius]
output['deferralFixture'] = [
    {'id':index+1, 'confidence':confidence, 'correct':correct}
    for index,(confidence,correct) in enumerate(zip(
        [.95,.9,.85,.8,.75,.7,.65,.6,.55,.5],
        [True,True,True,True,True,False,True,True,False,True]))
]
(directory / 'calculated-inputs.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key:output[key] for key in ['versions','selected','test','slices','paired']},indent=2))
print(json.dumps({name:{key:value for key,value in row.items() if 'Probability' not in key and 'Prediction' not in key} for name,row in output['candidates'].items()},indent=2))
