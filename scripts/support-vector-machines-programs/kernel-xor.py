import numpy as np
from sklearn.svm import SVC

x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
y = np.prod(x, axis=1)
phi = np.column_stack([x[:, 0] ** 2, np.sqrt(2) * x[:, 0] * x[:, 1], x[:, 1] ** 2])
assert np.allclose(phi @ phi.T, (x @ x.T) ** 2)
query = np.array([[.5, .5], [.5, -.5]])
for kernel in ['linear', 'poly', 'rbf']:
    fit = SVC(kernel=kernel, degree=2, coef0=0, gamma=.5 if kernel == 'rbf' else 1,
              C=1, tol=1e-10).fit(x, y)
    print(kernel, 'train accuracy:', f'{fit.score(x,y):.2f}',
          'query scores:', fit.decision_function(query).round(6).tolist())

bad = np.array([[1., 2.], [2., 1.]])
witness = np.array([1., -1.])
print('invalid Gram witness:', float(witness @ bad @ witness))
print('duplicate Gram eigenvalues:', np.linalg.eigvalsh(np.ones((2, 2))).tolist())
