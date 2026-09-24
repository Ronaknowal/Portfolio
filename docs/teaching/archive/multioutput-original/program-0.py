import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, Y = make_multilabel_classification(
    n_samples=1000, n_features=20, n_classes=5, n_labels=2,
    allow_unlabeled=False, random_state=42
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("X shape:", X.shape)
# Output: X shape: (1000, 20)
print("Y shape:", Y.shape)
# Output: Y shape: (1000, 5)
print("Label distribution:")
for i in range(Y.shape[1]):
    print(f"  Label {i}: {Y[:, i].sum()} positives ({100*Y[:, i].mean():.1f}%)")
# Output: Label distribution:
#   Label 0: 369 positives (36.9%)
#   Label 1: 635 positives (63.5%)
#   Label 2: 563 positives (56.3%)
#   Label 3: 478 positives (47.8%)
#   Label 4: 194 positives (19.4%)
print("Avg labels per sample:", Y.sum(axis=1).mean().round(2))
# Output: Avg labels per sample: 2.24

# Normalize features, add bias column
X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
X_train_n = (X_train - X_mean) / X_std
X_test_n  = (X_test  - X_mean) / X_std

def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])

X_tr = add_bias(X_train_n)
X_te = add_bias(X_test_n)