import numpy as np
from sklearn.svm import SVR

x = np.array([[-1.], [0.], [1.]])
y = np.array([-2., 0., 2.])
epsilon, c = .5, .5
slope = min(2*c, max(2-epsilon, 0))
excess = np.maximum(0, np.abs(y - slope*x[:,0])-epsilon)
print(f'slope={slope:.1f}; excess={excess.tolist()}; objective={.5*slope*slope+c*excess.sum():.2f}')
raw = SVR(kernel='linear', C=c, epsilon=epsilon, tol=1e-10).fit(x, y)
scale = 10.
converted = SVR(kernel='linear', C=c/scale, epsilon=epsilon/scale, tol=1e-10).fit(x, y/scale)
print('original predictions:', raw.predict(x).round(6).tolist())
print('converted back:', (scale*converted.predict(x)).round(6).tolist())
print('signed dual coefficients:', raw.dual_coef_.round(6).tolist())
