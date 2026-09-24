import numpy as np
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

x, y = make_classification(n_samples=300, n_features=10, n_informative=6,
                           n_redundant=2, random_state=8)
xt, xe, yt, ye = train_test_split(x, y, test_size=.3, stratify=y, random_state=9)
scaler = StandardScaler().fit(xt)
xt, xe = scaler.transform(xt), scaler.transform(xe)
exact_gram = rbf_kernel(xe, xt, gamma=.1)
exact = SVC(C=1, gamma=.1, tol=1e-8).fit(xt, yt)
print(f'exact RBF held-out accuracy={exact.score(xe,ye):.3f}')
for name, features in [('random features', RBFSampler(gamma=.1, n_components=60, random_state=7)),
                       ('Nystrom', Nystroem(gamma=.1, n_components=60, random_state=7))]:
    zt = features.fit_transform(xt)  # Landmark selection sees training observations only.
    ze = features.transform(xe)
    # A linear-kernel SVC keeps the same hinge/intercept convention for this small comparison.
    model = SVC(kernel='linear', C=1, tol=1e-8).fit(zt, yt)
    error = np.sqrt(np.mean((ze @ zt.T - exact_gram) ** 2))
    print(f'{name}: shape={zt.shape}, Gram RMSE={error:.4f}, held-out accuracy={model.score(ze,ye):.3f}')
print('This is a finite approximation comparison, not a training-speed benchmark.')
