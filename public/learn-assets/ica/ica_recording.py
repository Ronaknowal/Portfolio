import numpy as np
from sklearn.decomposition import FastICA, PCA

digital = np.loadtxt("r01-first20s.csv", delimiter=",", skiprows=1)
microvolts = (digital + 32768) * (6553.6 / 65535) - 3276.8
reference = microvolts[:, 0]
X = microvolts[:, 1:]
train = slice(0, 12000)
development = slice(12000, 16000)
test = slice(16000, 20000)

def correlation_with_reference(values, target):
    joined = np.column_stack([values, target])
    return np.corrcoef(joined, rowvar=False)[-1, :-1]

pca = PCA(n_components=4, svd_solver="full").fit(X[train])
ica = FastICA(n_components=4, whiten="unit-variance",
              whiten_solver="svd", algorithm="parallel", fun="logcosh",
              random_state=7, max_iter=1000, tol=1e-5).fit(X[train])
representations = [("channel", X), ("PCA", pca.transform(X)),
                   ("ICA", ica.transform(X))]
for name, values in representations:
    dev = correlation_with_reference(values[development], reference[development])
    chosen = int(np.argmax(np.abs(dev)))
    result = correlation_with_reference(values[test], reference[test])[chosen]
    print(name, chosen + 1, f"{abs(dev[chosen]):.6f}", f"{abs(result):.6f}")
print("ICA iterations", ica.n_iter_)
