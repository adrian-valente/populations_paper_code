"""
Statistical testing and clustering analyses for mixed selectivity
Created on 24.03.2020
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sys
from low_rank_rnns import helpers

############################################################
## ePAIRS test
## implements test ePAIRS from Hirokawa et al., Nature, 2019


def estimate_cov(X, cov_type):
    if cov_type == 'diag':
        vars = np.var(X, axis=0)
        cov = np.diag(vars)
    elif cov_type == 'full':
        cov = X.transpose() @ X / X.shape[0]
    return cov


def neighbors_distance_distr(X, n_neighbors=1):
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='brute', metric='cosine')
    nn.fit(X)
    distr = np.empty(n)
    for i in range(n):
        xi = X[i]
        ind = nn.kneighbors(xi.reshape(1, -1), return_distance=False)  # returns indices of nearest neighbors
        cosines = []
        for j in range(n_neighbors):
            xj = X[ind[0, j + 1]]  # first nearest neighbor is the point itself!
            cosines.append(xi @ xj / (np.linalg.norm(xi) * np.linalg.norm(xj)))
        distr[i] = np.mean(cosines)
    return distr


def epairs_mc_task(cov, n, n_neighbors):
        d = cov.shape[0]
        Xb = np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
        return neighbors_distance_distr(Xb, n_neighbors)


def epairs(X, b, n_neighbors=3, metric='angles', cov_type='diag', align_cov=True, whiten=True, return_mc_distr=False,
           use_mc_distr=None, plot=True, xlim=(-.1, 1.), figsize=None, col='gray'):
    """
    Statistical test ePAIRS (tests the null hypothesis: data is as randomly distributed directionally
    than a gaussian with same covariance)
    :param X: numpy array of shape (samples x dimensions)
    :param b: int, number of Monte Carlo samples
    :param metric: 'cos' or 'angles'
    :param cov_type: 'diag' or 'full', type of covariance estimator, 'diag' is as in Hirokawa et al.
    :param align_cov: bool, if True, align data to principal axes of covariance
    :param return_mc_distr: bool
    :param use_mc_distr: None or a filename for an .npy file containing MC-simulated angles distribution or a numpy array
    containing this distribution
    :return: p-value (computed with KS test),
             c-value (effect size),
             if return mc_distr is True, numpy array of shape (b, n), with n the num of samples.
    """
    n = X.shape[0]
    d = X.shape[1]
    X = X - np.mean(X, axis=0)  # ensure that data is centered

    if align_cov:
        _, V = np.linalg.eigh(X.transpose() @ X)
        X = X @ V

    if whiten:
        X = X / np.std(X, axis=0)

    cov = estimate_cov(X, cov_type)
    angles_data = neighbors_distance_distr(X, n_neighbors)
    if metric == 'angles':
        angles_data = np.arccos(angles_data.clip(-1., 1.))

    if use_mc_distr is not None:
        if isinstance(use_mc_distr, str):
            angles_mc = np.load(use_mc_distr)
        else:
            angles_mc = use_mc_distr
        print(angles_mc.shape)
    else:
        # Running MC simulations in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            args = [(cov, n, n_neighbors)] * b
            angles_mc = pool.starmap(epairs_mc_task, args)
        angles_mc = np.array(angles_mc)
        print(angles_mc.shape)

        if metric == 'angles':
            angles_mc = np.arccos(angles_mc)

    mc_mean = np.median(angles_mc)
    data_mean = np.median(angles_data)
    sigma = np.std(angles_mc)
    # iqr = stats.iqr(angles_mc)
    clusteriness = (mc_mean - data_mean) / sigma
    print(f'clusteriness: {clusteriness}')
    print(f'data mean: {data_mean:.3f}, mc mean: {mc_mean:.3f}')
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(angles_mc.ravel(), fill=False, color='k', ax=ax)
        sns.histplot(angles_data, color=col, stat='density', ec=None, ax=ax)
        # ax.axvline(mc_mean, color='k')
        # ax.axvline(data_mean, color=col)
        ax.set_xlabel('angle')
        ax.set_xlim(*xlim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # lots of ways to compute the p-values
    _, p3 = stats.ks_2samp(angles_mc.ravel(), angles_data)
    print(f'KS 2 sample test: p={p3}')
    _, p4 = stats.ranksums(angles_mc.ravel(), angles_data)
    print(f'Wilcoxon rank-sum test: p={p4}')
    _, p5 = stats.kruskal(angles_mc.ravel(), angles_data)
    print(f'Kruskal-Wallis test: p={p5}')
    if return_mc_distr:
        return p4, clusteriness, angles_mc
    else:
        return p4, clusteriness


def normalize(vectors):
    return (vectors.T / np.sqrt(np.sum(vectors**2, axis=1))).T


def bingham(vectors):
    nun, p = vectors.shape
    vectors = normalize(vectors)
    T = vectors.T @ vectors / nun
    S = nun * p * (p + 2) / 2 * (np.trace(T @ T) - (1/p))

    p = stats.chi2.sf(S, (p - 1) * (p + 2) / 2)
    return S, p


def bingham_aligned(X):
    X = X - np.mean(X, axis=0)  # ensure that data is centered
    n = X.shape[0]

    cov = X.T @ X / n
    print(cov)

    # shuffle rows of X and estimate covariance on first half of data
    idx_est = np.random.choice(np.arange(n), size=n//2)
    Xest = X[idx_est]
    eigs, V = np.linalg.eigh(Xest.transpose() @ Xest)
    print(V.shape)
    X = (X @ V)
    X = X / np.std(X, axis=0)

    cov = X.T @ X / n
    print(cov)

    return bingham(X[np.setxor1d(np.arange(n), idx_est)])

