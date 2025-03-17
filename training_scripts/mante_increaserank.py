"""
Made for the supplementary figure where networks with higher rank are trained on Mante task, and resampling
with one or 2 populations is attempted. Both resamplings are done with extensive retraining of the connectivity
on the SupportLowRankRNNs. This is done because some retraining is necessary when resampling with more than one
population, so at least a similar amount of retraining should be done for the one population network. Note that
this retraining is done with constant resampling of the gaussian basis.
"""

from low_rank_rnns import mante, clustering
from low_rank_rnns.modules import *

hidden_size = 4096
noise_std = 5e-2
alpha = 0.2
lr_base = 1e-1

results_filename = "mante_higherrank_results.txt"
results_file = open(results_filename, "w")

for rank in (2, 3, 4):
    print(f"rank = {rank}")
    x_train, y_train, mask_train, x_val, y_val, mask_val, epochs = (
        mante.generate_mante_data(1000, fraction_validation_trials=0.2)
    )
    net = LowRankRNN(
        4, hidden_size, 1, noise_std, alpha, rank=rank, train_wi=True, train_wo=True
    )
    train(
        net,
        x_train,
        y_train,
        mask_train,
        lr=lr_base / sqrt(hidden_size),
        n_epochs=70,
        keep_best=True,
        cuda=True,
        clip_gradient=0.1,
    )
    loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
    print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
    results_file.write(str(acc) + "\n")

    # gaussian resampling, with retraining of the connectivity
    losses_normal = []
    accs_normal = []
    n_samples = 10
    net2 = clustering.to_support_net(net, np.zeros(hidden_size))
    train(
        net2,
        x_train,
        y_train,
        mask_train,
        100,
        lr=1e-4,
        resample=True,
        cuda=True,
        keep_best=True,
    )
    for _ in range(n_samples):
        net2.resample_basis()
        loss, acc = mante.test_mante(net2, x_val, y_val, mask_val)
        losses_normal.append(loss)
        accs_normal.append(acc)
    print(
        f"gaussian resampling: min={np.min(accs_normal):.2f}, med={np.median(accs_normal):.2f},"
        f"max={np.max(accs_normal):.2f}"
    )
    results_file.write(str(accs_normal) + "\n")

    # Attempt 2 populations resampling
    vecs = clustering.make_vecs(net)
    z, model = clustering.gmm_fit(vecs, 2, algo="bayes", n_init=50, random_state=2020)
    losses2 = []
    accs2 = []
    net3 = clustering.to_support_net(net, z)
    train(
        net3,
        x_train,
        y_train,
        mask_train,
        100,
        lr=1e-4,
        resample=True,
        cuda=True,
        keep_best=True,
    )
    for _ in range(n_samples):
        net3.resample_basis()
        loss, acc = mante.test_mante(net3, x_val, y_val, mask_val)
        losses2.append(loss)
        accs2.append(acc)
    print(
        f"2 populations resampling: min={np.min(accs2):.2f}, med={np.median(accs2):.2f},"
        f"max={np.max(accs2):.2f}"
    )
    results_file.write(str(accs2) + "\n")
