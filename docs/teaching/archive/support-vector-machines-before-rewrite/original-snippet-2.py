# ---- Demo 1: linearly separable Gaussian blobs ----
np.random.seed(42)
X_pos = np.random.randn(20, 2) + np.array([2.0, 2.0])
X_neg = np.random.randn(20, 2) + np.array([-2.0, -2.0])
X_sep = np.vstack([X_pos, X_neg])
y_sep = np.hstack([np.ones(20), -np.ones(20)])

model_sep = SimpleSMO(C=1.0, kernel="rbf", gamma=0.5, max_passes=50)
model_sep.fit(X_sep, y_sep)
preds_sep = model_sep.predict(X_sep)
acc_sep = (preds_sep == y_sep).mean()
print(f"[Linearly separable]  n_support={model_sep.n_support}  accuracy={acc_sep:.3f}")
# [Linearly separable]  n_support=20  accuracy=1.000

# ---- Demo 2: XOR / non-separable data ----
np.random.seed(7)
X_xor = np.random.randn(40, 2)
y_xor = np.where((X_xor[:, 0] * X_xor[:, 1]) > 0, 1.0, -1.0)

model_xor = SimpleSMO(C=1.0, kernel="rbf", gamma=1.0, max_passes=50)
model_xor.fit(X_xor, y_xor)
preds_xor = model_xor.predict(X_xor)
acc_xor = (preds_xor == y_xor).mean()
print(f"[XOR / non-separable] n_support={model_xor.n_support}  accuracy={acc_xor:.3f}")
# [XOR / non-separable] n_support=32  accuracy=0.925