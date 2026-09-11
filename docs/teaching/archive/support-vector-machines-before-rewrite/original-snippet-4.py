# LinearSVC wraps liblinear — scales to millions of samples
lsvc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
lsvc.fit(X_sc, y)
print(f"LinearSVC train accuracy: {lsvc.score(X_sc, y):.3f}")
# LinearSVC train accuracy: 0.842

# For very large n (>100k), prefer SGDClassifier with hinge loss:
# from sklearn.linear_model import SGDClassifier
# sgd_svm = SGDClassifier(loss="hinge", alpha=0.001)  # alpha ~ 1/C
# Equivalent to SVM with linear kernel; trains in O(n) per epoch.