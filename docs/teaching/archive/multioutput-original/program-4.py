def hamming_loss(Y_true, Y_pred):
    return np.mean(Y_true != Y_pred)

def subset_accuracy(Y_true, Y_pred):
    return np.mean(np.all(Y_true == Y_pred, axis=1))

def f1_micro(Y_true, Y_pred):
    tp = (Y_true * Y_pred).sum()
    fp = ((1 - Y_true) * Y_pred).sum()
    fn = (Y_true * (1 - Y_pred)).sum()
    prec = tp / (tp + fp + 1e-15)
    rec  = tp / (tp + fn + 1e-15)
    return 2 * prec * rec / (prec + rec + 1e-15)

def f1_macro(Y_true, Y_pred):
    f1s = []
    for j in range(Y_true.shape[1]):
        tp = (Y_true[:, j] * Y_pred[:, j]).sum()
        fp = ((1 - Y_true[:, j]) * Y_pred[:, j]).sum()
        fn = (Y_true[:, j] * (1 - Y_pred[:, j])).sum()
        prec = tp / (tp + fp + 1e-15)
        rec  = tp / (tp + fn + 1e-15)
        f1s.append(2 * prec * rec / (prec + rec + 1e-15))
    return np.mean(f1s)

def f1_samples(Y_true, Y_pred):
    scores = []
    for i in range(len(Y_true)):
        tp = (Y_true[i] * Y_pred[i]).sum()
        fp = ((1 - Y_true[i]) * Y_pred[i]).sum()
        fn = (Y_true[i] * (1 - Y_pred[i])).sum()
        denom = tp + fp + fn
        scores.append(2*tp / (2*tp + fp + fn) if denom > 0 else 1.0)
    return np.mean(scores)

print("=== Binary Relevance ===")
print(f"Hamming loss:      {hamming_loss(Y_test, Y_pred_br):.4f}")
# Output: Hamming loss:      0.1740
print(f"Subset accuracy:   {subset_accuracy(Y_test, Y_pred_br):.4f}")
# Output: Subset accuracy:   0.3950
print(f"F1 micro:          {f1_micro(Y_test, Y_pred_br):.4f}")
# Output: F1 micro:          0.7986
print(f"F1 macro:          {f1_macro(Y_test, Y_pred_br):.4f}")
# Output: F1 macro:          0.7454
print(f"F1 samples:        {f1_samples(Y_test, Y_pred_br):.4f}")
# Output: F1 samples:        0.8180

print()
print("=== Classifier Chain ===")
print(f"Hamming loss:      {hamming_loss(Y_test, Y_pred_chain):.4f}")
# Output: Hamming loss:      0.1790
print(f"Subset accuracy:   {subset_accuracy(Y_test, Y_pred_chain):.4f}")
# Output: Subset accuracy:   0.3950
print(f"F1 micro:          {f1_micro(Y_test, Y_pred_chain):.4f}")
# Output: F1 micro:          0.7954
print(f"F1 macro:          {f1_macro(Y_test, Y_pred_chain):.4f}")
# Output: F1 macro:          0.7370
print(f"F1 samples:        {f1_samples(Y_test, Y_pred_chain):.4f}")
# Output: F1 samples:        0.8140