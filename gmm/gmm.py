import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import time
import warnings
warnings.filterwarnings('ignore')

def generate_multivariate_data(n_samples=300, dimensions=3, random_state=42):
    """
    Generate multivariate data with multiple components
    
    Args:
        n_samples (int): Number of samples per component
        dimensions (int): Number of dimensions
        random_state (int): Random seed for reproducibility
    
    Returns:
        np.ndarray: Generated multivariate data
    """
    np.random.seed(random_state)
    
    means = [
        np.array([-5] * dimensions),
        np.array([5] * dimensions),
        np.zeros(dimensions)
    ]
    print('Original means.', means)
    
    
    covariances = [
        np.eye(dimensions) * 1.2,
        np.eye(dimensions) * 1.8,
        np.eye(dimensions) * 1.6
    ]
    
    components = [
        np.random.multivariate_normal(mean, cov, n_samples) 
        for mean, cov in zip(means, covariances)
    ]
    
    X = np.concatenate(components)
    np.random.shuffle(X)
    
    return X


def custom_random_init(X, n_components):
    dimensions = X.shape[1]
    
    means = X[np.random.choice(X.shape[0], n_components, replace=False)]
    
    covariances = [np.eye(dimensions) * np.var(X[:, i]) for i in range(dimensions)]
    
    pi = np.ones(n_components) / n_components
    
    return means, covariances, pi

def custom_step_expectation(X, n_components, means, covariances):
    weights = np.zeros((n_components, X.shape[0]))
    
    for j in range(n_components):
        weights[j, :] = multivariate_normal.pdf(X, mean=means[j], cov=covariances[j])
    
    return weights

def custom_step_maximization(X, weights, means, covariances, n_components, pi):
    dimensions = X.shape[1]
    
    r = []
    for j in range(n_components):
        r_j = (weights[j] * pi[j]) / (np.sum([weights[i] * pi[i] for i in range(n_components)], axis=0))
        r.append(r_j)
        
        means[j] = np.sum(r_j[:, np.newaxis] * X, axis=0) / np.sum(r_j)
        
        diff = X - means[j]
        covariances[j] = np.dot(r_j * diff.T, diff) / np.sum(r_j)
        
        pi[j] = np.mean(r_j)
    
    return covariances, means, pi

def custom_train_gmm(data, n_components=3, n_steps=50):
    """
    Train Custom Gaussian Mixture Model using EM algorithm
    
    Returns:
        tuple: Final means, covariances, and mixing coefficients
    """
    means, covariances, pi = custom_random_init(data, n_components)
    
    for _ in range(n_steps):
        weights = custom_step_expectation(data, n_components, means, covariances)
        
        covariances, means, pi = custom_step_maximization(data, weights, means, covariances, n_components, pi)
    
    return means, covariances, pi
