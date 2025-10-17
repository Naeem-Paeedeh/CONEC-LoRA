import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch import Tensor as T
import math
from torch.nn import functional as F


def GMM_EM(features, n_components, max_iter: int = 100, tol=1e-3):
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          init_params='kmeans',
                          max_iter=max_iter,
                          tol=tol,
                          verbose=0)
    gmm.fit(features)

    # [k, dim], [k, dim, dim], [k]
    return (gmm.means_, gmm.covariances_, gmm.weights_)


def random_sampling(num_samples, compression, n_components):
    (gmm_means, gmm_covariances, gmm_weights) = compression
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.means_ = gmm_means
    gmm.covariances_ = gmm_covariances
    gmm.weights_ = gmm_weights
    reconstructed_features = gmm.sample(num_samples)[0]

    return reconstructed_features
