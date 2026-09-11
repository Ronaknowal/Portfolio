L = Y_train.shape[1]
chain_weights = []

# Training: augment features with ground-truth previous labels
for j in range(L):
    X_aug = X_tr if j == 0 else np.column_stack([X_tr, Y_train[:, :j]])
    w = logistic_train(X_aug, Y_train[:, j], lr=0.05, n_iter=300)
    chain_weights.append(w)
    print(f"Label {j} trained. Weight shape: {w.shape}")
# Output:
# Label 0 trained. Weight shape: (21,)
# Label 1 trained. Weight shape: (22,)
# Label 2 trained. Weight shape: (23,)
# Label 3 trained. Weight shape: (24,)
# Label 4 trained. Weight shape: (25,)

# Inference: use predicted labels for subsequent positions
Y_pred_chain = np.zeros((len(X_test_n), L), dtype=int)
for j in range(L):
    X_aug = X_te if j == 0 else np.column_stack([X_te, Y_pred_chain[:, :j]])
    p = sigmoid(X_aug @ chain_weights[j])
    Y_pred_chain[:, j] = (p >= 0.5).astype(int)