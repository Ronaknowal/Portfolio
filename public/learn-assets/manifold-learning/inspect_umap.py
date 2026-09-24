import numpy as np
import umap
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances

data = np.loadtxt("digits-300.csv", delimiter=",", skiprows=1)
X = data[:, 2:] / 16.0
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, spread=1.0,
                    n_components=2, metric="euclidean", init="spectral",
                    random_state=7, transform_seed=7, n_jobs=1)
Y = reducer.fit_transform(X)
distances = pairwise_distances(X)
np.fill_diagonal(distances, np.inf)
input_neighbors = np.argsort(distances, axis=1, kind="stable")[:, :10]
map_distances = pairwise_distances(Y)
np.fill_diagonal(map_distances, np.inf)
map_neighbors = np.argsort(map_distances, axis=1, kind="stable")[:, :10]
R10 = np.mean([len(set(a) & set(b)) / 10
               for a, b in zip(input_neighbors, map_neighbors)])
print(umap.__version__, Y.shape)
print(f"R10={R10:.4f} T10={trustworthiness(X, Y, n_neighbors=10):.4f}")
np.savetxt("umap-digits.csv", np.column_stack((data[:, :2], Y)), delimiter=",")
