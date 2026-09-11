import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

x, y = make_classification(n_samples=160, n_features=10, n_informative=6,
                           n_redundant=2, n_classes=4, n_clusters_per_class=1, random_state=5)
xt, xe, yt, ye = train_test_split(x, y, test_size=.25, stratify=y, random_state=9)
scaler = StandardScaler().fit(xt)
xt, xe = scaler.transform(xt), scaler.transform(xe)
model = SVC(kernel='linear', decision_function_shape='ovo', tol=1e-8).fit(xt, yt)
predictions = model.predict(xe)
print('classes:', model.classes_.tolist(), 'binary pairs:', 4 * 3 // 2)
print('OVO score shape:', model.decision_function(xe[:3]).shape)
model.set_params(decision_function_shape='ovr')  # Presentation changes; no refitting.
print('OVR score shape:', model.decision_function(xe[:3]).shape)
print('labels unchanged:', bool(np.array_equal(predictions, model.predict(xe))))
train_gram, query_gram = xt @ xt.T, xe @ xt.T
precomputed = SVC(kernel='precomputed', tol=1e-8).fit(train_gram, yt)
print('train/query Gram shapes:', train_gram.shape, query_gram.shape)
print('precomputed labels agree:', bool(np.array_equal(predictions, precomputed.predict(query_gram))))
