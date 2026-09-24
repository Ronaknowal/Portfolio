"""Measured development learning curves, not VC-dimension estimates or a test report."""
from pathlib import Path
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent


def run():
    data = np.genfromtxt(HERE/'banknote-subset.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    x = np.column_stack([data['variance'],data['skewness'],data['curtosis'],data['entropy']])
    y = data['class']
    training_order = np.random.default_rng(44).permutation(320)
    dev = np.arange(320,400)
    definitions = {
        'logistic_regression': lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500)),
        'rbf_svc': lambda:make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale')),
        'depth5_tree': lambda:DecisionTreeClassifier(max_depth=5,random_state=44)}
    rows = []
    for n in [20,40,80,160,320]:
        train = training_order[:n]
        assert len(np.unique(y[train])) == 2
        for name,make_model in definitions.items():
            model = make_model().fit(x[train],y[train])
            training_predictions = model.predict(x[train]); dev_predictions = model.predict(x[dev])
            rows.append({'model':name,'n':n,'train_correct':int(np.sum(training_predictions==y[train])),
                         'development_correct':int(np.sum(dev_predictions==y[dev])),
                         'development_n':len(dev),'train_source_rows':data['source_row'][train].tolist(),
                         'development_predictions':dev_predictions.tolist()})
    return {'seed':44,'features':['variance','skewness','curtosis','entropy'],
            'development_source_rows':data['source_row'][dev].tolist(),'development_labels':y[dev].tolist(),
            'rows':rows,'test_evaluated':False,
            'interpretation':'Fixed nested training subsets and one development set; no claim of independent error bars or inferred VC order.'}


if __name__ == '__main__':
    results = run()
    (HERE/'banknote-learning-curve-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for row in results['rows']:
        print(row['model'],row['n'],row['train_correct'],row['development_correct'])
