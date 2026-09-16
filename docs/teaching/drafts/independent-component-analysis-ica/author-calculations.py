"""Small authoring probes and exact data extraction; not a phase-two verifier.

Run from the repository root with the shared lesson-tools Python.
The downloaded, hash-checked EDF is only needed with --extract.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np
from sklearn.decomposition import FastICA, PCA

PACKET = Path(__file__).resolve().parent
SOURCE_HASH = "7549bbd378ea23851c20c0b7924f0a1f9fd909a3a3683c2334144a4c156dcb62"

if "--extract" in sys.argv:
    raw = Path("scratch/ica-content/r01.edf").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASH
    # EDF: 1792-byte header, 60 records of 5 seconds. Each record has
    # five 5000-sample signed-int16 signals then 500 annotation samples.
    records = np.frombuffer(raw[1792:], dtype="<i2").reshape(60, 25500)
    signals = records[:4, :25000].reshape(4, 5, 5000)
    signals = signals.transpose(0, 2, 1).reshape(20000, 5)
    np.savetxt(PACKET / "r01-first20s.csv", signals, fmt="%d", delimiter=",",
               header="direct_adc,abdomen1_adc,abdomen2_adc,abdomen3_adc,abdomen4_adc", comments="")

# Four equiprobable source states and their actual PCA whitening.
source = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
mixing = np.array([[2., 1.], [1., 2.]])
observed = source @ mixing.T
whitener = np.array([[1/(3*np.sqrt(2)), 1/(3*np.sqrt(2))],
                     [1/np.sqrt(2), -1/np.sqrt(2)]])
white = observed @ whitener.T
w = np.array([.8, .6])
projection = white @ w
raw_update = white.T @ projection**3 / 4 - (3*projection**2).mean()*w
next_w = -raw_update / np.linalg.norm(raw_update)
print("covariance", observed.T @ observed/4)
print("whitened covariance", np.round(white.T @ white/4, 12))
print("fixed-point raw", np.round(raw_update, 9), "sign-aligned", np.round(next_w, 9))
for name, k in [("binary", -2.), ("Laplace", 3.), ("Gaussian", 0.)]:
    values = [k*(np.cos(np.deg2rad(a))**4 + np.sin(np.deg2rad(a))**4) for a in [0, 30, 45, 90]]
    print(name, np.round(values, 9))

# Each offered contribution operation, including zero-source and compensated
# positive/negative scale nulls. These calculations support the written specs.
for source_values in [np.array([2., -3.]), np.array([2., 0.]), np.zeros(2)]:
    for keep in [(), (0,), (1,), (0, 1)]:
        retained = sum((mixing[:, j] * source_values[j] for j in keep), np.zeros(2))
        for j in [0, 1]:
            for scale in [2., -2.]:
                rescaled = source_values.copy()
                columns = mixing.copy()
                rescaled[j] *= scale
                columns[:, j] /= scale
                result = sum((columns[:, i] * rescaled[i] for i in keep), np.zeros(2))
                assert np.array_equal(result, retained)
print("contribution nulls and positive/negative scale compensation: exact")

# Real recording: fixed chronological train/development/test protocol.
digital = np.loadtxt(PACKET / "r01-first20s.csv", delimiter=",", skiprows=1)
physical = (digital + 32768) * (6553.6 / 65535) - 3276.8
reference, X = physical[:, 0], physical[:, 1:]
fit, development, test = slice(0, 12000), slice(12000, 16000), slice(16000, 20000)

def correlations(values, reference_values):
    joined = np.column_stack([values, reference_values])
    return np.corrcoef(joined, rowvar=False)[-1, :-1]

pca = PCA(n_components=4, svd_solver="full").fit(X[fit])
ica = FastICA(n_components=4, whiten="unit-variance", whiten_solver="svd",
              algorithm="parallel", fun="logcosh", random_state=7,
              max_iter=1000, tol=1e-5).fit(X[fit])
for name, values in [("channel", X), ("PCA", pca.transform(X)), ("ICA", ica.transform(X))]:
    dev_corr = correlations(values[development], reference[development])
    chosen = int(np.argmax(np.abs(dev_corr)))
    test_corr = correlations(values[test], reference[test])[chosen]
    print(name, "dev", np.round(dev_corr, 6), "selected", chosen+1,
          "test_abs", round(abs(test_corr), 6))
print("ICA iterations", ica.n_iter_)
print("train IC variance", np.round(ica.transform(X[fit]).var(axis=0), 6))
print("test reconstruction MSE", np.mean((ica.inverse_transform(ica.transform(X[test]))-X[test])**2))
print("csv sha256", hashlib.sha256((PACKET / "r01-first20s.csv").read_bytes()).hexdigest())
