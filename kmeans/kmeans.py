import numpy as np


def k_means(X: np.array, k: int, max_iters: int =100, tol: float=1e-4):
    """
    Our K-Means implementation.

    Args:
        X (ndarray): Dataset of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid change.

    Returns (Tuple):
        centroids (ndarray): Final cluster centroids.
        labels (ndarray): Cluster assignments for each point.
    """
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        distances_to_centroids = np.linalg.norm(X[:, None] - centroids, axis=2)
        
        labels = np.argmin(distances_to_centroids, axis=1)

        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return centroids, labels
