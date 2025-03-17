from sklearn.neighbors import NearestNeighbors
import numpy as np
from low_rank_rnns.helpers import radial_distribution_plot
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


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


def epairs(X, b, n_neighbors=1, metric='angles', cov_type='diag'):
    """
    Statistical test ePAIRS (tests the null hypothesis: data is as randomly distributed directionally
    than a gaussian with same covariance)
    :param X: numpy array of shape (samples x dimensions)
    :param b: int, number of Monte Carlo samples
    :param metric: 'cos' or 'angles'
    :return: p-value (computed with ranksum test)
    """
    n = X.shape[0]
    d = X.shape[1]
    X = X - np.mean(X, axis=0)  # ensure that data is centered
    if cov_type == 'diag':
        vars = np.var(X, axis=0)
        cov = np.diag(vars)
    elif cov_type == 'full':
        cov = X.transpose() @ X / n
    angles_data = neighbors_distance_distr(X, n_neighbors)
    if metric == 'angles':
        angles_data = np.arccos(angles_data)
    angles_bootstrap = np.empty((b, n))
    for i in range(b):
        Xb = np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
        angles_bootstrap[i] = neighbors_distance_distr(Xb, n_neighbors)
    if metric == 'angles':
        angles_bootstrap = np.arccos(angles_bootstrap)

    boot_mean = np.median(angles_bootstrap)
    data_mean = np.median(angles_data)
    clusteriness = (boot_mean - data_mean) / boot_mean
    print(f'clusteriness: {clusteriness}')
    print(f'data mean: {data_mean:.3f}, mc mean: {boot_mean:.3f}')
    sns.distplot(angles_bootstrap.ravel())
    sns.distplot(angles_data)
    plt.axvline(boot_mean, color='#1F77B4')
    plt.axvline(data_mean, color='#FF8D2A')
    plt.show()

    # lots of ways to compute the p-values
    _, p3 = stats.ks_2samp(angles_bootstrap.ravel(), angles_data)
    print(f'KS 2 sample test: p={p3}')
    _, p4 = stats.ranksums(angles_bootstrap.ravel(), angles_data)
    print(f'Wilcoxon rank-sum test: p={p4}')
    _, p5 = stats.kruskal(angles_bootstrap.ravel(), angles_data)
    print(f'Kruskal-Wallis test: p={p5}')
    return p3


def test_epairs1():
    metric = 'angles'
    n_neighbors = 3
    cov_type = 'full'

    print("Spheric gaussian, dim=2")
    X = np.random.randn(500, 2)
    p = epairs(X, 100, n_neighbors=n_neighbors, metric=metric, cov_type=cov_type)
    radial_distribution_plot(X[:, :2])
    plt.title(f"Spheric gaussian, dim=2, p={p:.3f}", fontsize='x-large')
    plt.savefig('../figures/epairs1.png')
    plt.show()
    print('--------')

    print('Elliptic gaussian, dim=2')
    X = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.array([[4, 0], [0, 1]]), size=500)
    p = epairs(X, 100, n_neighbors=n_neighbors, metric=metric, cov_type=cov_type)
    radial_distribution_plot(X[:, :2])
    plt.title(f"Elliptic gaussian, dim=2, p={p:.3f}", fontsize='x-large')
    plt.savefig('../figures/epairs2.png')
    plt.show()
    print('--------')

    print('Mixture of gaussians, dim=2')
    X1 = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.array([[20, 0], [0, 1]]), size=250)
    X2 = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.array([[1, 0], [0, 20]]), size=250)
    X = np.vstack([X1, X2])
    p = epairs(X, 100, n_neighbors=n_neighbors, metric=metric, cov_type=cov_type)
    radial_distribution_plot(X[:, :2])
    plt.title(f"Mixture of gaussians, dim=2, p={p:.3f}", fontsize='x-large')
    plt.savefig('../figures/epairs3.png')
    plt.show()
    print('--------')

    dim = 20

    print(f'Spheric gaussian, dim={dim}')
    X = np.random.randn(500, dim)
    p = epairs(X, 100, n_neighbors=n_neighbors, metric=metric, cov_type=cov_type)
    radial_distribution_plot(X[:, :2])
    plt.title(f"Spheric gaussian, dim={dim}, p={p:.3f}", fontsize='x-large')
    plt.savefig('../figures/epairs4.png')
    plt.show()
    print('--------')

    print(f'Elliptic gaussian, dim={dim}')
    cov = np.eye(dim)
    cov[1, 1] = 20
    X = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov, size=500)
    p = epairs(X, 100, n_neighbors=n_neighbors, metric=metric, cov_type=cov_type)
    radial_distribution_plot(X[:, :2])
    plt.title(f"Elliptic gaussian, dim={dim}, p={p:.3f}", fontsize='x-large')
    plt.savefig('../figures/epairs5.png')
    plt.show()
    print('--------')

    print(f'Mixture of gaussians, dim={dim}')
    X1 = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov, size=250)
    cov[1, 1] = 1; cov[0, 0] = 20
    X2 = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov, size=250)
    X = np.vstack([X1, X2])
    p = epairs(X, 100, n_neighbors=n_neighbors, metric=metric, cov_type=cov_type)
    radial_distribution_plot(X[:, :2])
    plt.title(f"Mixture of gaussians, dim={dim}, p={p:.3f}", fontsize='x-large')
    plt.savefig('../figures/epairs6.png')
    plt.show()
    print('--------')


def test_epairs2():
    metric = 'angles'
    for spread in (1, 5, 10, 15, 20):
        X1 = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.array([[spread, 0], [0, 1]]), size=250)
        X2 = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.array([[1, 0], [0, spread]]), size=250)
        X = np.vstack([X1, X2])
        p = epairs(X, 100, metric=metric)
        print(f"dim=2, spread={spread}, p={p:.3f}")
        print('--------')

    dim = 20
    for spread in (1, 5, 10, 15, 20):
        cov = np.eye(dim)
        cov[1, 1] = spread
        X1 = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov, size=250)
        cov[1, 1] = 1
        cov[0, 0] = spread
        X2 = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov, size=250)
        X = np.vstack([X1, X2])
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        p = epairs(X, 100, metric=metric)
        print(f"dim={dim}, spread={spread}, p={p:.3f}")


def test_epairs3():
    """
    Shows how the test with diagonal covariance estimation gives rise to false positives
    """
    r2 = .9
    X = np.random.multivariate_normal(mean=np.zeros(2), cov=np.array([[1, r2], [r2, 1]]), size=500)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    p = epairs(X, 100)
    print(f'Correlated X, diagonal covariance estimator, p={p:.3f}')

    p = epairs(X, 100, cov_type='full')
    print(f'Correlated X, full covariance estimator, p={p:.3f}')


def test_epairs4():
    """
    Applying the ePAIRS test with uniform distributions on hypercubes
    """
    n = 500
    d = 2
    X = np.random.rand(n, d)
    X = X - np.mean(X, axis=0)
    radial_distribution_plot(X[:, :2])
    plt.show()
    p = epairs(X, 100)
    print(f'Uniform cube in {d} dims, p={p:.3f}')

    d = 20
    X = np.random.rand(n, d)
    X = X - np.mean(X, axis=0)
    radial_distribution_plot(X[:, :2])
    plt.show()
    p = epairs(X, 100)
    print(f'Uniform cube in {d} dims, p={p:.3f}')


if __name__ == '__main__':
    test_epairs4()
