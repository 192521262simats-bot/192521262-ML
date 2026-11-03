import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=0.7,
    random_state=42
)

model = GaussianMixture(
    n_components=3,
    random_state=42,
    n_init=5
)

model.fit(X)
labels = model.predict(X)

print("Number of Clusters:", model.n_components)
print("Cluster Counts:", np.bincount(labels))
print("Converged:", model.converged_)
print("Iterations:", model.n_iter_)
print("Silhouette Score:", round(silhouette_score(X, labels), 4))

print("First 10 Cluster Labels:", labels[:10])