import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from kmeans import k_means

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

start_time = time.time()
our_centroids, our_labels = k_means(X, k=4)
our_kmeans_time = time.time() - start_time

start_time = time.time()
sklearn_kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')
sklearn_labels = sklearn_kmeans.fit_predict(X)
sklearn_centroids = sklearn_kmeans.cluster_centers_
sklearn_kmeans_time = time.time() - start_time

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=our_labels, cmap='viridis', s=30)
plt.scatter(our_centroids[:, 0], our_centroids[:, 1], c='red', s=100, label='Centroids')
plt.title("Our K-Means")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', s=30)
plt.scatter(sklearn_centroids[:, 0], sklearn_centroids[:, 1], c='red', s=100, label='Centroids')
plt.title("Scikit-learn K-Means")
plt.legend()

plt.show()

inertia_our = np.sum([np.linalg.norm(X[our_labels == i] - centroid, axis=1).sum() for i, centroid in enumerate(our_centroids)])
inertia_sklearn = sklearn_kmeans.inertia_

print("Metrics Comparison")
print("-------------------")
print(f"Inertia:          Our K-Means = {inertia_our:.2f}, Sklearn = {inertia_sklearn:.2f}")
print(f"Runtime (seconds): Our K-Means = {our_kmeans_time:.4f}, Sklearn = {sklearn_kmeans_time:.4f}")