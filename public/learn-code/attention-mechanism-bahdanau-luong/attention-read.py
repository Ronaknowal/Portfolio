import numpy as np

def attention_read(query, keys, values, valid):
    if not np.any(valid):
        raise ValueError("At least one source position must be valid")
    scores = keys @ query
    scores = np.where(valid, scores, -np.inf)
    masses = np.exp(scores - np.max(scores))
    weights = masses / masses.sum()
    return weights, weights @ values

keys = np.array([[1., 0.], [0., 1.], [-1., 0.]])
values = np.array([[2., 0.], [0., 2.], [-1., 1.]])
weights, context = attention_read(
    np.array([1., 0.]), keys, values, np.array([True, True, True])
)
print(np.round(weights, 6), np.round(context, 6))
