import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = np.loadtxt("digits-300.csv", delimiter=",", skiprows=1)
X, labels = data[:, 2:] / 16.0, data[:, 1].astype(int)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, labels, test_size=0.25, stratify=labels, random_state=7)
representations = {"pixels": (X_train, X_valid)}
for name, reducer in {
    "PCA-10": PCA(n_components=10, svd_solver="full"),
    "UMAP-10": umap.UMAP(n_components=10, n_neighbors=15,
                         min_dist=0.1, random_state=7,
                         transform_seed=7, n_jobs=1)
}.items():
    train_coordinates = reducer.fit_transform(X_train)
    valid_coordinates = reducer.transform(X_valid)
    representations[name] = (train_coordinates, valid_coordinates)
for name, (train, valid) in representations.items():
    model = KNeighborsClassifier(n_neighbors=5).fit(train, y_train)
    print(name, f"accuracy={model.score(valid, y_valid):.4f}")
