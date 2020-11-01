import numpy as np
from scipy import stats
from math import sqrt
import torch
import multiprocessing as mp
from itertools import repeat
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from low_rank_rnns.modules import SupportLowRankRNN
from low_rank_rnns.helpers import center_axes, gram_factorization


def gmm_fit(neurons_fs, n_components, algo='bayes', n_init=50, random_state=None, mean_precision_prior=None,
            weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None):
    """
    Fit a mixture of gaussians to a cloud of n points
    :param neurons_fs: list of numpy arrays of shape n or numpy array of shape n x d
    :param n_components: int
    :param algo: 'em' (expectation-maximization) or 'bayes'
    :param n_init: number of random seeds for the inference algorithm
    :param random_state: random seed for the rng to eliminate randomness
    :return: vector of population labels (of shape n), best fitted model
    """
    if isinstance(neurons_fs, list):
        X = np.vstack(neurons_fs).transpose()
    else:
        X = neurons_fs
    if algo == "em":
        model = GaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state)
    else:
        model = BayesianGaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state,
                                        init_params='random', mean_precision_prior=mean_precision_prior,
                                        weight_concentration_prior_type=weight_concentration_prior_type,
                                        weight_concentration_prior=weight_concentration_prior)
    model.fit(X)
    z = model.predict(X)
    return z, model


def make_vecs(net):
    """
    return a list of vectors (list of numpy arrays of shape n) composing a network
    """
    return [net.m[:, i].detach().cpu().numpy() for i in range(net.rank)] + \
           [net.n[:, i].detach().cpu().numpy() for i in range(net.rank)] + \
           [net.wi[i].detach().cpu().numpy() for i in range(net.input_size)] + \
           [net.wo[:, i].cpu().detach().numpy() for i in range(net.output_size)]


def to_support_net(net, z, new_size=None, take_means=False):
    """
    Generate a SupportLowRankRNN from a LowRankRNN, with populations defined by z
    :param net: LowRankRNN
    :param z: numpy array of size n whose entries are integer labels for populations
    :param new_size: int
    :param take_means: bool, whether to take into account means of components or set them to zero
    :return: the SupportLowRankRNN
    """
    X = np.vstack(make_vecs(net)).transpose()
    _, counts = np.unique(z, return_counts=True)
    n_components = counts.shape[0]
    weights = counts / net.hidden_size
    if take_means:
        means = np.vstack([X[z == i].mean(axis=0) for i in range(n_components)])
    else:
        means = np.zeros((n_components, X.shape[1]))
    covariances = [np.cov(X[z == i].transpose()) for i in range(n_components)]

    rank = net.rank
    basis_dim = 2 * rank + net.input_size + net.output_size
    m_init = torch.zeros(rank, n_components, basis_dim)
    n_init = torch.zeros(rank, n_components, basis_dim)
    wi_init = torch.zeros(net.input_size, n_components, basis_dim)
    wo_init = torch.zeros(net.output_size, n_components, basis_dim)

    if new_size is None:
        new_size = net.hidden_size
    m_means = torch.from_numpy(means[:, :rank]).t() / sqrt(new_size)
    n_means = torch.from_numpy(means[:, rank: 2*rank]).t() / sqrt(new_size)
    wi_means = torch.from_numpy(means[:, 2*rank: 2*rank + net.input_size]).t()

    for i in range(n_components):
        # Compute Gramian matrix of the basis we have to build
        G = covariances[i]
        # compute coefficients on the basis of iid normal vectors
        X_reduced = gram_factorization(G)
        for k in range(rank):
            m_init[k, i] = torch.from_numpy(X_reduced[k]) / sqrt(new_size)
            n_init[k, i] = torch.from_numpy(X_reduced[rank + k]) / sqrt(new_size)
        for k in range(net.input_size):
            wi_init[k, i] = torch.from_numpy(X_reduced[2 * rank + k])
        for k in range(net.output_size):
            wo_init[k, i] = torch.from_numpy(X_reduced[2 * rank + net.input_size + k]) / new_size

    net2 = SupportLowRankRNN(net.input_size, new_size, net.output_size, net.noise_std, net.alpha, rank, n_components,
                             weights, basis_dim, m_init, n_init, wi_init, wo_init, m_means, n_means, wi_means)
    return net2


def pop_scatter_linreg(vec1, vec2, pops, n_pops=None, colors=('blue', 'red', 'green', 'gray'),
                       linreg=True, figsize=(5, 5), ax=None):
    """
    scatter plot of (vec1, vec2) points separated in populations according to int labels in vector pops, with linear
    regressions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    center_axes(ax)

    # Computing axes limits
    xmax = max(abs(vec1.min()), vec1.max())
    xmin = -xmax
    ax.set_xlim(xmin - .1 * (xmax - xmin), xmax + .1 * (xmax - xmin))
    ymax = max(abs(vec2.min()), vec2.min())
    ymin = -ymax
    ax.set_ylim(ymin - .1 * (ymax - ymin), ymax + .1 * (ymax - ymin))
    xs = np.linspace(xmin - .1 * (xmax - xmin), xmax + .1 * (xmax - xmin), 100)

    if n_pops is None:
        n_pops = np.unique(pops).shape[0]
    for i in range(n_pops):
        ax.scatter(vec1[pops == i], vec2[pops == i], color=colors[i], s=.5)
        if linreg:
            slope, intercept, r_value, p_value, std_err = stats.linregress(vec1[pops == i], vec2[pops == i])
            print(f"pop {i}: slope={slope:.2f}, intercept={intercept:.2f}")
            ax.plot(xs, slope * xs + intercept, color=colors[i], zorder=-1)
    ax.set_xticks([])
    ax.set_yticks([])


### Spectral clustering and stability analysis

def generate_subsamples(neurons_fs, fraction=.8):
    indexes = np.random.choice(neurons_fs.shape[0], int(fraction * neurons_fs.shape[0]), replace=False)
    indexes = np.sort(indexes)
    return neurons_fs[indexes], indexes


def spectral_clustering(neurons_fs, n_clusters, metric='euclidean', n_neighbors=10):
    if metric == 'euclidean':
        model = SpectralClustering(n_clusters, affinity='nearest_neighbors')
        model.fit(neurons_fs)
    elif metric == 'cosine':
        model = SpectralClustering(n_clusters, affinity='precomputed')
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
        nn.fit(neurons_fs)
        knn_graph = nn.kneighbors_graph()
        knn_graph = 0.5 * (knn_graph + knn_graph.transpose())
        model.fit(knn_graph)
    return model


def clustering_stability_task(neurons_fs, algo, n_clusters, metric, n_neighbors, mean_precision_prior=1e5):
    sample, indexes = generate_subsamples(neurons_fs)
    if algo == 'spectral':
        model = spectral_clustering(sample, n_clusters, metric, n_neighbors)
        labels = model.labels_
    else:
        labels, _ = gmm_fit(sample, n_clusters, mean_precision_prior=mean_precision_prior,
                            weight_concentration_prior_type='dirichlet_distribution')
    return labels, indexes


def clustering_stability(neurons_fs, n_clusters, n_bootstrap, algo='gmm', metric='cosine', n_neighbors=10,
                         mean_precision_prior=1e5, normalize=None):
    """
    Compute clustering stability distribution for a particular number of clusters and algorithm
    :param neurons_fs: numpy array of shape Nxd (neurons embedded in some feature space)
    :param n_clusters: int
    :param n_bootstrap: int
    :param algo: 'spectral' or 'gmm'
    :param metric: 'euclidean' or 'cosine' for spectral clustering
    :param n_neighbors: int, for spectral clustering
    :param mean_precision_prior:
    :param normalize: None, 'normal' or 'uniform'
    :return: list of n_bootstrap x (n_bootstrap - 1) / 2 ARI values for bootstrapped clusterings (possibly normalized)
    """

    with mp.Pool(mp.cpu_count()) as pool:
        args = repeat((neurons_fs, algo, n_clusters, metric, n_neighbors, mean_precision_prior), n_bootstrap)
        res = pool.starmap(clustering_stability_task, args)
        labels_list, indexings = zip(*res)

    aris = []
    # Align bootstrap samples and compute pairwise Rand indexes
    for i in range(n_bootstrap):
        for j in range(i+1, n_bootstrap):
            # build aligned labellings
            indexes_i = indexings[i]
            indexes_j = indexings[j]
            labels_i = []
            labels_j = []
            l = 0
            for k in range(len(indexes_i)):
                if l > len(indexes_j):
                    break
                while l < len(indexes_j) and indexes_j[l] < indexes_i[k]:
                    l += 1
                if l < len(indexes_j) and indexes_j[l] == indexes_i[k]:
                    labels_i.append(labels_list[i][k])
                    labels_j.append(labels_list[j][l])
                    l += 1
            aris.append(adjusted_rand_score(labels_i, labels_j))

    if normalize is not None:
        if normalize == 'normal':
            X_base = np.random.randn(neurons_fs.shape[0], neurons_fs.shape[1])
        elif normalize == 'uniform':
            X_base = (np.random.rand(neurons_fs.shape[0], neurons_fs.shape[1]) - 0.5) * 2
        base_aris = clustering_stability(X_base, n_clusters, n_bootstrap, metric, normalize=None)
        base_mean, base_std = np.mean(base_aris), np.std(base_aris)
        aris = [(ari - base_mean) / base_std for ari in aris]
    return aris


def boxplot_clustering_stability(neurons_fs, clusters_nums, aris=None, algo='gmm', n_bootstrap=20, metric='cosine',
                                 n_neighbors=10, ax=None):
    """
    :param neurons_fs: numpy array of shape Nxd (neurons embedded in some feature space)
    :param clusters_nums: list of ints
    :param aris: if precomputed aris, list of lists with ari distributions
    :param algo: 'spectral' or 'gmm'
    :param n_bootstrap: int
    :param metric: 'euclidean' or 'cosine' for spectral clustering
    :param n_neighbors: int, for spectral clustering
    :param ax: matplotlib axes if already setup
    :return: axes
    """
    if aris is None:
        aris = [clustering_stability(neurons_fs, k, n_bootstrap, algo, metric, n_neighbors) for k in clusters_nums]
    aris = np.array(aris)
    if ax is None:
        fig, ax = plt.subplots()
    col_lines = 'indianred'
    bp = ax.boxplot(aris.T, patch_artist=True)
    for box in bp['boxes']:
        box.set(color='steelblue', facecolor='steelblue')
    for med in bp['medians']:
        med.set(color=col_lines)
    for l in bp['whiskers']:
        l.set(color=col_lines)
    for l in bp['caps']:
        l.set(color=col_lines)
    ax.set_xticks(list(range(1, aris.shape[0] + 1)))
    ax.set_xticklabels(clusters_nums)
    ax.set(xlabel='number of clusters', ylabel='stability', ylim=(-.1, 1.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax
