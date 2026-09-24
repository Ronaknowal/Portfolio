import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

x, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
xt, xe, yt, ye = train_test_split(x, y, test_size=.3, stratify=y, random_state=11)
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)
baseline = DummyClassifier(strategy='most_frequent').fit(xt, yt)
print(f'train={len(yt)}, test={len(ye)}; majority test accuracy={baseline.score(xe,ye):.3f}')
rbf = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                  {'svc__C': [.1, 1., 10.], 'svc__gamma': [.01, .1, 1.]}, cv=folds)
rbf.fit(xt, yt)  # Each fold fits its own scaler.
linear = GridSearchCV(make_pipeline(StandardScaler(), LinearSVC(dual=False, max_iter=10000)),
                     {'linearsvc__C': [.1, 1., 10.]}, cv=folds).fit(xt, yt)
for name, search in [('RBF', rbf), ('linear squared hinge', linear)]:
    print(name, search.best_params_)
    print(f'CV={search.best_score_:.3f} held-out accuracy={search.score(xe,ye):.3f}')
print('RBF support counts by class:', rbf.best_estimator_.named_steps['svc'].n_support_.tolist())
print('RBF decision shape:', rbf.decision_function(xe[:3]).shape)
print('Selection/CV and these held-out comparisons do not establish a universal winner.')
