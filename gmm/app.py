import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import time
import warnings
warnings.filterwarnings('ignore')

from gmm import generate_multivariate_data, custom_train_gmm

def compare_gmm_implementations(X, n_components=3, n_steps=50):
    """
    Compare custom GMM implementation with scikit-learn's GMM
    
    Args:
        X (np.ndarray): Input data
        n_components (int): Number of mixture components
        n_steps (int): Number of iterations for custom implementation
    """
    start_time = time.time()
    custom_means, custom_covariances, custom_pi = custom_train_gmm(X, n_components, n_steps)
    custom_time = time.time() - start_time
    
    start_time = time.time()
    sklearn_gmm = GaussianMixture(n_components=n_components, 
                                   covariance_type='full', 
                                   n_init=1, 
                                   max_iter=n_steps)
    sklearn_gmm.fit(X)
    sklearn_time = time.time() - start_time
    
    print("Custom GMM Implementation:")
    print("Means:")
    for i, mean in enumerate(custom_means):
        print(f"  Component {i+1}: {mean}")
    print("\nMixing Coefficients:", custom_pi)
    print(f"Execution Time: {custom_time:.4f} seconds")
    
    print("\nScikit-learn GMM Implementation:")
    print("Means:")
    for i, mean in enumerate(sklearn_gmm.means_):
        print(f"  Component {i+1}: {mean}")
    print("\nMixing Coefficients:", sklearn_gmm.weights_)
    print(f"Execution Time: {sklearn_time:.4f} seconds")
    
    custom_log_likelihood = np.sum(np.log(
        np.sum([
            multivariate_normal.pdf(X, mean=means, cov=cov) * pi 
            for means, cov, pi in zip(custom_means, custom_covariances, custom_pi)
        ], axis=0)
    ))
    
    sklearn_log_likelihood = sklearn_gmm.score(X) * X.shape[0]
    
    print("\nLog-Likelihood Comparison:")
    print(f"Custom Implementation: {custom_log_likelihood:.4f}")
    print(f"Scikit-learn Implementation: {sklearn_log_likelihood:.4f}")

if __name__ == '__main__':
    X = generate_multivariate_data(n_samples=300, dimensions=3)
    
    compare_gmm_implementations(X, n_components=3, n_steps=50)