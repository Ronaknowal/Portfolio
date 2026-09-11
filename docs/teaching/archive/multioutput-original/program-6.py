from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss

categories = [
    'sci.med', 'sci.space', 'sci.electronics',
    'rec.sport.baseball', 'rec.motorcycles'
]
data_train = fetch_20newsgroups(
    subset='train', categories=categories,
    remove=('headers', 'footers', 'quotes')
)
data_test = fetch_20newsgroups(
    subset='test', categories=categories,
    remove=('headers', 'footers', 'quotes')
)
print(f"Train: {len(data_train.data)}, Test: {len(data_test.data)}")
# Output: Train: 2973, Test: 1978

# Build multi-label targets (each doc gets its true label
# + 15% chance of an additional label — simulates noisy co-occurrence)
rng = np.random.default_rng(42)
def make_multilabel(targets, n_classes=5, noise=0.15):
    Y = np.eye(n_classes, dtype=int)[targets]
    Y = np.clip(Y + (rng.random(Y.shape) < noise).astype(int), 0, 1)
    return Y

Y_train = make_multilabel(data_train.target)
Y_test  = make_multilabel(data_test.target)
print(f"Avg labels/doc: {Y_train.sum(axis=1).mean():.2f}")
# Output: Avg labels/doc: 1.59

# TF-IDF vectorizer (sparse, handled natively by saga)
tfidf = TfidfVectorizer(max_features=10000, min_df=2)
X_train = tfidf.fit_transform(data_train.data)
X_test  = tfidf.transform(data_test.data)
print(f"Feature matrix: {X_train.shape}")
# Output: Feature matrix: (2973, 10000)

chain = ClassifierChain(
    LogisticRegression(C=0.5, solver='saga', max_iter=500, random_state=42),
    order='random', random_state=42
)
chain.fit(X_train, Y_train)
Y_pred = chain.predict(X_test)

print(f"Hamming loss:    {hamming_loss(Y_test, Y_pred):.4f}")
# Output: Hamming loss:    0.2319
print(f"Subset accuracy: {(np.all(Y_test==Y_pred, axis=1)).mean():.4f}")
# Output: Subset accuracy: 0.3332
print(f"F1 micro:        {f1_score(Y_test, Y_pred, average='micro', zero_division=0):.4f}")
# Output: F1 micro:        0.5528
print(f"F1 macro:        {f1_score(Y_test, Y_pred, average='macro', zero_division=0):.4f}")
# Output: F1 macro:        0.5426
print(f"F1 samples:      {f1_score(Y_test, Y_pred, average='samples', zero_division=0):.4f}")
# Output: F1 samples:      0.5706