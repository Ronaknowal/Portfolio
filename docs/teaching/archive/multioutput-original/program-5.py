import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, f1_score

np.random.seed(42)
X, Y = make_multilabel_classification(
    n_samples=1000, n_features=20, n_classes=5, n_labels=2,
    allow_unlabeled=False, random_state=42
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- OneVsRestClassifier (Binary Relevance) ---
ovr = OneVsRestClassifier(
    LogisticRegression(C=1.0, max_iter=300, random_state=42)
)
ovr.fit(X_train_s, Y_train)
Y_pred_ovr = ovr.predict(X_test_s)

print("OneVsRestClassifier (Binary Relevance):")
print(f"  Hamming loss:    {hamming_loss(Y_test, Y_pred_ovr):.4f}")
# Output:   Hamming loss:    0.1780
print(f"  Subset accuracy: {(np.all(Y_test==Y_pred_ovr, axis=1)).mean():.4f}")
# Output:   Subset accuracy: 0.4000
print(f"  F1 micro:        {f1_score(Y_test, Y_pred_ovr, average='micro', zero_division=0):.4f}")
# Output:   F1 micro:        0.7968
print(f"  F1 macro:        {f1_score(Y_test, Y_pred_ovr, average='macro', zero_division=0):.4f}")
# Output:   F1 macro:        0.7428
print(f"  F1 samples:      {f1_score(Y_test, Y_pred_ovr, average='samples', zero_division=0):.4f}")
# Output:   F1 samples:      0.8140

# --- ClassifierChain (random order = ensemble-style) ---
cc = ClassifierChain(
    LogisticRegression(C=1.0, max_iter=300, random_state=42),
    order='random', random_state=42
)
cc.fit(X_train_s, Y_train)
Y_pred_cc = cc.predict(X_test_s)

print()
print("ClassifierChain (random order):")
print(f"  Hamming loss:    {hamming_loss(Y_test, Y_pred_cc):.4f}")
# Output:   Hamming loss:    0.1660
print(f"  Subset accuracy: {(np.all(Y_test==Y_pred_cc, axis=1)).mean():.4f}")
# Output:   Subset accuracy: 0.4600
print(f"  F1 micro:        {f1_score(Y_test, Y_pred_cc, average='micro', zero_division=0):.4f}")
# Output:   F1 micro:        0.8096
print(f"  F1 macro:        {f1_score(Y_test, Y_pred_cc, average='macro', zero_division=0):.4f}")
# Output:   F1 macro:        0.7537
print(f"  F1 samples:      {f1_score(Y_test, Y_pred_cc, average='samples', zero_division=0):.4f}")
# Output:   F1 samples:      0.8310