"""
Train 100 low-rank networks on the working mem task and make diverse analyses (epairs, truncations, resampling)
"""

import sys
sys.path.append('../')

from low_rank_rnns import romo, clustering
from low_rank_rnns.regressions import regression_romo
from low_rank_rnns import mixedselectivity as ms
from low_rank_rnns.modules import *

n_nets = 100
hidden_size = 512
noise_std = 5e-3
alpha = 0.2
lr_base = 1e-2

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
    x_train, y_train, mask_train, x_val, y_val, mask_val = romo.generate_data(1000)
    net_fr = FullRankRNN(1, hidden_size, 1, noise_std, alpha)
    train(net_fr, x_train, y_train, mask_train, n_epochs=5, lr=lr_base / sqrt(hidden_size), batch_size=32, keep_best=True, cuda=True)
    x_val, y_val, mask_val = map_device([x_val, y_val, mask_val], net_fr)
    out = net_fr.forward(x_val)
    print("Final loss: {:.3f}".format(loss_mse(out, y_val, mask_val)))

    romo.delay_duration_max = 1000
    romo.setup()
    x_train, y_train, mask_train, x_val, y_val, mask_val = romo.generate_data(1000)
    wi_init = net_fr.wi_full.detach()
    wo_init = net_fr.wo_full.detach() * hidden_size
    print(wo_init.std())
    wrec = net_fr.wrec.detach().cpu().numpy()
    u, s, v = np.linalg.svd(wrec)
    m_init = torch.from_numpy(s[:2] * u[:, :2]).to(device=net_fr.wrec.device) * sqrt(hidden_size)
    n_init = torch.from_numpy(v[:2, :].transpose()).to(device=net_fr.wrec.device) * sqrt(hidden_size)
    print(m_init.std())
    print(n_init.std())
    net = LowRankRNN(1, hidden_size, 1, noise_std, alpha, rank=2, wi_init=wi_init, wo_init=wo_init, m_init=m_init, n_init=n_init)
    train(net, x_train, y_train, mask_train, n_epochs=100, lr=lr_base, batch_size=32, keep_best=True, cuda=True, clip_gradient=1, early_stop=0.01)
    x_val, y_val, mask_val = map_device([x_val, y_val, mask_val], net)
    out = net.forward(x_val)
    loss, acc = romo.test_romo(net, x_val, y_val, mask_val)
    print("Final loss: {:.3f}".format(loss))
    acc_vals.append(acc)
    loss_vals.append(loss)
    torch.save(net.state_dict(), f"../models/romo_many3/{i}.pt")

    net = net.cpu()
    net._define_proxy_parameters()
    m1 = net.m[:,0].detach().numpy()
    n1 = net.n[:,0].detach().numpy()
    m2 = net.m[:,1].detach().numpy()
    n2 = net.n[:,1].detach().numpy()
    wi = net.wi_full[0].detach().numpy()
    wo = net.wo_full[:,0].detach().numpy()
    vectors = [-wi, n1, n2, m1, m2, -wo]

    conn_space = np.array([wi, n1, n2, m1, m2]).transpose()
    p, c = ms.epairs(conn_space, 500, plot=False)
    p_vals.append(p)
    c_vals.append(c)

    reg_space = regression_romo(net)
    p_reg, c_reg = ms.epairs(reg_space, 500, plot=False)
    preg_vals.append(p_reg)
    creg_vals.append(c_reg)

    n_samples = 10
    accs_tmp = []
    losses_tmp = []
    for k in range(n_samples):
        net2 = clustering.to_support_net(net, np.zeros(hidden_size))
        loss, acc = romo.test_romo(net2, x_val.cpu(), y_val.cpu(), mask_val.cpu())
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

np.savez('../data/romo_results3.npz', p_vals, c_vals, preg_vals, creg_vals, acc_vals, loss_vals,
         accs_res, losses_res)
np.save('../data/romo_lr_c_boot_distr.npy', np.array(cdistr_loadings))