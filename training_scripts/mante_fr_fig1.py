"""
Train 100 full-rank DMS networks and make diverse analyses (epairs, truncations, resampling)
"""

import sys
sys.path.append('../')

from low_rank_rnns import mante, clustering
from low_rank_rnns.regressions import regression_mante
from low_rank_rnns import mixedselectivity as ms
from low_rank_rnns.modules import *

n_nets = 100
hidden_size = 512
input_size = 4
noise_std = 5e-2
alpha = 0.2

x_train, y_train, mask_train, x_val, y_val, mask_val = mante.generate_mante_data(1000)
acc_vals = []
loss_vals = []
preg_vals = []
creg_vals = []
ranks_trunc = list(range(1, 10))
trunc_accs = np.zeros((n_nets, len(ranks_trunc)))
trunc_losses = np.zeros((n_nets, len(ranks_trunc)))
p_vals = np.zeros((n_nets, len(ranks_trunc)))
c_vals = np.zeros((n_nets, len(ranks_trunc)))
n_samples = 10
accs_res = np.zeros((n_nets, n_samples, len(ranks_trunc)))
losses_res = np.zeros((n_nets, n_samples, len(ranks_trunc)))
cdistr = []

for i in range(n_nets):
    net = FullRankRNN(input_size, hidden_size, 1, noise_std, alpha, train_wi=True, rho=.1)
    w0 = net.wrec.detach().numpy().copy()
    train(net, x_train, y_train, mask_train, 60, lr=1e-4, clip_gradient=1,
          early_stop=0.05, keep_best=True, cuda=True)
    # net.load_state_dict(torch.load(f'../models/mante_many_fr/{i}.pt', map_location='cpu'))
    loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
    print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
    acc_vals.append(acc)
    loss_vals.append(loss)
    torch.save(net.state_dict(), f"../models/mante_many_fr4/{i}.pt")

    net = net.cpu()
    net._define_proxy_parameters()

    delta_w = net.wrec.detach().numpy() - w0
    u, s, v = np.linalg.svd(delta_w)

    # Truncating accuracies and losses
    for j, rank in enumerate(ranks_trunc):
        wrec = w0 + (u[:, :rank] * s[:rank]) @ v[:rank]
        net_trunc = FullRankRNN(input_size, hidden_size, 1, noise_std, alpha, wi_init=net.wi_full, wo_init=net.wo_full,
                                wrec_init=torch.from_numpy(wrec))
        loss, acc = mante.test_mante(net_trunc, x_val, y_val, mask_val)
        trunc_accs[i, j] = acc
        trunc_losses[i, j] = loss

    # Regression analysis
    reg_space, r2 = regression_mante(net)
    p_reg, c_reg, angles_mc = ms.epairs(reg_space, 500, plot=False, return_mc_distr=True)
    preg_vals.append(p_reg)
    creg_vals.append(c_reg)

    # Evaluating null distribution of c values on resampled regression spaces
    mean = np.mean(reg_space, axis=0)
    cov = np.cov(reg_space, rowvar=False)
    for _ in range(10):
        reg_resampled = np.random.multivariate_normal(mean, cov, size=hidden_size)
        _, c = ms.epairs(reg_resampled, 500, plot=False, use_mc_distr=angles_mc)
        cdistr.append(c)

    wi = net.wi.detach().numpy().squeeze()
    wo = net.wo.detach().numpy().squeeze()
    for j, rank in enumerate(ranks_trunc):
        m = u[:, :rank]
        n = v[:rank].T
        conn_space = np.concatenate([wi.reshape((-1, input_size)), wo.reshape((-1, 1)), m.reshape((-1, rank)), n.reshape((-1, rank))], axis=1)
        print(conn_space.shape)
        p, c = ms.epairs(conn_space, 500, plot=False)
        p_vals[i, j] = p
        c_vals[i, j] = c

    for j, rank in enumerate(ranks_trunc):
        m = u[:, :rank] * s[:rank] * np.sqrt(hidden_size)
        n = v[:rank].T * np.sqrt(hidden_size)
        net_trunc = LowRankRNN(input_size, hidden_size, 1, noise_std, alpha, rank=rank, wi_init=net.wi_full,
                               wo_init=net.wo_full * hidden_size, m_init=torch.from_numpy(m), n_init=torch.from_numpy(n))
        net_res = clustering.to_support_net(net_trunc, np.zeros(hidden_size))

        for k in range(n_samples):
            net_res.resample_basis()
            wrec_reb = w0 + (net_res.m @ net_res.n.t()).detach().numpy()
            net_reb = FullRankRNN(input_size, hidden_size, 1, noise_std, alpha, wi_init=net_res.wi_full, wo_init=net_res.wo_full,
                                  wrec_init=torch.from_numpy(wrec_reb))
            loss, acc = mante.test_mante(net_reb, x_val, y_val, mask_val)
            accs_res[i, k, j] = acc
            losses_res[i, k, j] = loss


np.savez('../data/mante_fr_results4.npz', p_vals, c_vals, preg_vals, creg_vals, acc_vals, loss_vals, trunc_accs, trunc_losses,
         accs_res, losses_res)
np.save('../data/mante_fr_c_boot_distr4.npy', np.array(cdistr))
