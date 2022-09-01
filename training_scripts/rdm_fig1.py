"""
Train 100 low-rank RDM networks and make diverse analyses (epairs, truncations, resampling)
"""

import sys
sys.path.append('../')

from low_rank_rnns import rdm, clustering
from low_rank_rnns.regressions import regression_rdm
from low_rank_rnns import mixedselectivity as ms
from low_rank_rnns.modules import *

n_nets = 100
hidden_size = 512
noise_std = 5e-2
alpha = 0.2

x_train, y_train, mask_train, x_val, y_val, mask_val = rdm.generate_rdm_data(1000)
p_vals = []
c_vals = []
preg_vals = []
creg_vals = []
acc_vals = []
loss_vals = []
accs_res = []
losses_res = []
cdistr_loadings = []

for i in range(n_nets):
    net = LowRankRNN(1, hidden_size, 1, noise_std, alpha, rank=1)
    train(net, x_train, y_train, mask_train, lr=5e-3, n_epochs=20, batch_size=32, keep_best=True, cuda=True)
    # net.load_state_dict(torch.load(f'../models/rdm_many/{i}.pt', map_location='cpu'))
    loss, acc = rdm.test_rdm(net, x_val, y_val, mask_val)
    print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
    acc_vals.append(acc)
    loss_vals.append(loss)
    torch.save(net.state_dict(), f"../models/rdm_many2/{i}.pt")

    net = net.cpu()
    net._define_proxy_parameters()
    m = net.m[:,0].detach().numpy()
    n = net.n[:,0].detach().numpy()
    wi = net.wi_full[0].detach().numpy()
    wo = net.wo_full[:,0].detach().numpy()

    conn_space = np.array([wi, n, m]).transpose()
    p, c = ms.epairs(conn_space, 500, plot=False)
    p_vals.append(p)
    c_vals.append(c)

    # Regression analysis
    reg_space = regression_rdm(net)
    p_reg, c_reg = ms.epairs(reg_space, 500, plot=False)
    preg_vals.append(p_reg)
    creg_vals.append(c_reg)

    # Resampling
    n_samples = 10
    accs_tmp = []
    losses_tmp = []
    for k in range(n_samples):
        net2 = clustering.to_support_net(net, np.zeros(hidden_size))
        loss, acc = rdm.test_rdm(net2, x_val, y_val, mask_val)
        accs_tmp.append(acc)
        losses_tmp.append(loss)
        vecs_res = np.array(clustering.make_vecs(net2)).T
        if k == 0:
            _, c, angles_mc = ms.epairs(vecs_res, 500, plot=False, return_mc_distr=True)
        else:
            _, c = ms.epairs(vecs_res, 500, plot=False, use_mc_distr=angles_mc)
        cdistr_loadings.append(c)
    accs_res.append(accs_tmp)
    losses_res.append(losses_tmp)

np.savez('../data/rdm_results3.npz', p_vals, c_vals, preg_vals, creg_vals, acc_vals, loss_vals,
         accs_res, losses_res)
np.save('../data/rdm_lr_c_boot_distr.npy', np.array(cdistr_loadings))

