import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.stats import mode

from kmeans.kmeans import k_means
from gmm.gmm import custom_train_gmm, generate_multivariate_data, custom_step_expectation

class ClusterComparison:
    def __init__(self, n_samples=300, n_features=3, n_clusters=3, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.X = generate_multivariate_data(n_samples, n_features, random_state)
        self.results = {}

    def calculate_inertia(self, X, centroids, labels):
        inertia = 0.0
        for i, centroid in enumerate(centroids):
            cluster_points = X[labels == i]
            inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia

    def run_kmeans(self):
        start_time = time.time()
        kmeans_centroids, kmeans_labels = k_means(self.X, self.n_clusters)
        kmeans_time = time.time() - start_time

        kmeans_inertia = self.calculate_inertia(self.X, kmeans_centroids, kmeans_labels)
        kmeans_silhouette = silhouette_score(self.X, kmeans_labels)

        self.results['K-Means'] = {
            'Execution Time (s)': kmeans_time,
            'Inertia': kmeans_inertia,
            'Silhouette Score': kmeans_silhouette
        }

        return kmeans_centroids, kmeans_labels

    def run_gmm(self):
        start_time = time.time()
        gmm_means, gmm_covariances, gmm_pi = custom_train_gmm(self.X, self.n_clusters)
        gmm_labels = np.argmax(custom_step_expectation(self.X, self.n_clusters, gmm_means, gmm_covariances), axis=0)
        gmm_time = time.time() - start_time

        gmm_silhouette = silhouette_score(self.X, gmm_labels)

        self.results['GMM'] = {
            'Execution Time (s)': gmm_time,
            'Silhouette Score': gmm_silhouette
        }

        return gmm_means, gmm_labels

    def plot_clusters(self, X, labels, centroids=None, title="Clusters"):
        plt.figure(figsize=(8, 6))
        for cluster_label in np.unique(labels):
            cluster_points = X[labels == cluster_label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_label}")
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label="Centroids")
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_comparison(self):
        algorithms = list(self.results.keys())
        execution_times = [self.results[alg]['Execution Time (s)'] for alg in algorithms]
        silhouette_scores = [self.results[alg]['Silhouette Score'] for alg in algorithms]

        plt.figure(figsize=(12, 5))

        # Execution Time
        plt.subplot(1, 2, 1)
        plt.bar(algorithms, execution_times, color=['blue', 'green'])
        plt.title("Execution Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Algorithm")

        plt.subplot(1, 2, 2)
        plt.bar(algorithms, silhouette_scores, color=['blue', 'green'])
        plt.title("Silhouette Score Comparison")
        plt.ylabel("Score")
        plt.xlabel("Algorithm")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    comparison = ClusterComparison()
    kmeans_centroids, kmeans_labels = comparison.run_kmeans()
    gmm_means, gmm_labels = comparison.run_gmm()

    for algorithm, metrics in comparison.results.items():
        print(f"\n{algorithm} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    comparison.plot_clusters(comparison.X, kmeans_labels, centroids=kmeans_centroids, title="K-Means Clustering")
    comparison.plot_clusters(comparison.X, gmm_labels, centroids=np.array(gmm_means), title="GMM Clustering")
    comparison.plot_comparison()
