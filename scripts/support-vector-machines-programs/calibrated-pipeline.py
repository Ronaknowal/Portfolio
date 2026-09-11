import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

x, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
xt, xe, yt, ye = train_test_split(x, y, test_size=.3, stratify=y, random_state=11)
pipeline = make_pipeline(StandardScaler(), SVC(C=1, gamma='scale'))
calibrated = CalibratedClassifierCV(pipeline, method='sigmoid', ensemble=False,
                                  cv=StratifiedKFold(3, shuffle=True, random_state=7))
calibrated.fit(xt, yt)
probabilities = calibrated.predict_proba(xe)
positive = int(np.flatnonzero(calibrated.classes_ == 1)[0])
p = probabilities[:, positive]
prior = np.full(len(ye), np.mean(yt))
print('classes:', calibrated.classes_.tolist(), 'probability shape:', probabilities.shape)
print('first three positive probabilities:', p[:3].round(4).tolist())
print(f'prior Brier={brier_score_loss(ye,prior):.4f}; calibrated Brier={brier_score_loss(ye,p):.4f}')
print(f'held-out log loss={log_loss(ye,probabilities):.4f}')
print('rows sum to one:', bool(np.allclose(probabilities.sum(axis=1), 1)))
print('A finite held-out proper score is an assessment, not proof of perfect calibration.')
