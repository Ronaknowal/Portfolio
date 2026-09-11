# Train one logistic classifier per label
weights_br = []
for j in range(Y_train.shape[1]):
    w = logistic_train(X_tr, Y_train[:, j], lr=0.05, n_iter=300)
    weights_br.append(w)

def br_predict(X, weights, threshold=0.5):
    preds = [( sigmoid(X @ w) >= threshold).astype(int) for w in weights]
    return np.column_stack(preds)

Y_pred_br = br_predict(X_te, weights_br)