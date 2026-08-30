import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch import Tensor as T
import math
from torch.nn import functional as F
import logging


def GMM_EM(features, n_components, max_iter: int = 100, tol=1e-3, reg_covar: float = 1e-6):
    # We retry by gradually increasing the reg_covar as the covariance matrices may be singular.
    last_error = None

    for reg in [reg_covar * (10 ** i) for i in range(9)]:
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type='full',
                              init_params='kmeans',
                              max_iter=max_iter,
                              tol=tol,
                              reg_covar=reg,
                              verbose=0)
        try:
            gmm.fit(features)
            # [k, dim], [k, dim, dim], [k]
            return (gmm.means_, gmm.covariances_, gmm.weights_)
        except (np.linalg.LinAlgError, ValueError) as error:
            last_error = error
            logging.warning(f'GMM_EM: the fit failed with reg_covar={reg:g} ({error}); retrying with a larger ridge.')

    raise RuntimeError('GMM_EM: the Gaussian mixture could not be fitted.') from last_error


def random_sampling(num_samples, compression, n_components):
    (gmm_means, gmm_covariances, gmm_weights) = compression
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.means_ = gmm_means
    gmm.covariances_ = gmm_covariances
    gmm.weights_ = gmm_weights
    reconstructed_features = gmm.sample(num_samples)[0]

    return reconstructed_features
