import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances

data = np.loadtxt("digits-300.csv", delimiter=",", skiprows=1)
source_row, labels = data[:, 0].astype(int), data[:, 1].astype(int)
X = data[:, 2:] / 16.0

def neighbors(points, k):
    distances = pairwise_distances(points)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :k]

def retention(X, Y, k):
    return np.mean([len(set(a) & set(b)) / k
                    for a, b in zip(neighbors(X, k), neighbors(Y, k))])

layouts = {"PCA": PCA(n_components=2, svd_solver="full").fit_transform(X)}
for perplexity in (5, 30, 80):
    layouts[f"t-SNE p={perplexity}"] = TSNE(
        perplexity=perplexity, init="pca", learning_rate="auto",
        max_iter=1000, random_state=7, method="barnes_hut",
        angle=0.5, n_jobs=1).fit_transform(X)

for name, Y in layouts.items():
    print(name, f"R10={retention(X, Y, 10):.4f}",
          f"T10={trustworthiness(X, Y, n_neighbors=10):.4f}")
    np.savetxt(name.replace(" ", "_") + ".csv",
               np.column_stack((source_row, labels, Y)), delimiter=",")
