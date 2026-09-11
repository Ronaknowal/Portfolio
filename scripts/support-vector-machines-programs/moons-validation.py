import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

x, y = make_moons(n_samples=200, noise=.2, random_state=42)
train_valid, test = train_test_split(np.arange(200), test_size=.2, stratify=y, random_state=17)
train, valid = train_test_split(train_valid, test_size=.25, stratify=y[train_valid], random_state=19)
scaler = StandardScaler().fit(x[train])
xt, xv = scaler.transform(x[train]), scaler.transform(x[valid])
fitted = []
print(f'rows: train={len(train)}, validation={len(valid)}, test={len(test)}')
for c in [.1, 1., 10.]:
    for gamma in [.1, 1., 10.]:
        model = SVC(C=c, gamma=gamma, tol=1e-8).fit(xt, y[train])
        training, validation = model.score(xt, y[train]), model.score(xv, y[valid])
        fitted.append((c, gamma, model, training, validation))
        print(f'C={c:4.1f} gamma={gamma:4.1f} train={training:.3f} valid={validation:.3f} SV={len(model.support_)}')
# First grid entry wins an exact validation tie: selection never uses the test labels.
best = max(range(len(fitted)), key=lambda index: fitted[index][4])
c, gamma, selected, _, validation = fitted[best]
print(f'selected C={c:g}, gamma={gamma:g}; one test score={selected.score(scaler.transform(x[test]),y[test]):.3f}')
