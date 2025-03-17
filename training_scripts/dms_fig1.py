"""
Train 100 low-rank DMS networks and make diverse analyses (epairs, truncations, resampling)
"""

from low_rank_rnns import dms, clustering
from low_rank_rnns.regressions import regression_dms
from low_rank_rnns import mixedselectivity as ms
from low_rank_rnns.modules import *

n_nets = 100
hidden_size = 512
noise_std = 5e-2
alpha = 0.2
lr_base = 1e-1
global_nepochs = 20

p_vals = []
c_vals = []
preg_vals = []
creg_vals = []
r2_vals = []
acc_vals = []
loss_vals = []
accs_res = []
losses_res = []
cdistr_loadings = []

for i in range(n_nets):
    # Priming with full rank network
    net_fr = FullRankRNN(2, hidden_size, 1, noise_std, alpha)
    dms.delay_duration_max = 2000
    dms.setup()
    x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
    train(
        net_fr,
        x_train,
        y_train,
        mask_train,
        lr=1e-5,
        n_epochs=50,
        early_stop=0.05,
        keep_best=True,
        cuda=True,
    )
    loss, acc = dms.test_dms(net_fr, x_val, y_val, mask_val)
    print("fr loss: {}\nfr accuracy: {}".format(loss, acc))
    wi_init = net_fr.wi.detach()
    wo_init = net_fr.wo.detach() * hidden_size
    u, s, v = np.linalg.svd(net_fr.wrec.detach().cpu().numpy())
    m_init = torch.from_numpy(s[:2] * u[:, :2]).to(device=net_fr.wrec.device) * sqrt(
        hidden_size
    )
    n_init = torch.from_numpy(v[:2, :].transpose()).to(
        device=net_fr.wrec.device
    ) * sqrt(hidden_size)

    # Moving to low-rank, some shaping involved
    net = LowRankRNN(
        2,
        hidden_size,
        1,
        noise_std,
        alpha,
        rank=2,
        wi_init=wi_init,
        wo_init=wo_init,
        m_init=m_init,
        n_init=n_init,
    )
    # net = LowRankRNN(2, hidden_size, 1, noise_std, alpha, rank=2)
    dms.delay_duration_max = 700
    dms.setup()
    x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
    train(
        net,
        x_train,
        y_train,
        mask_train,
        lr=1e-2,
        n_epochs=300,
        early_stop=0.05,
        keep_best=True,
        cuda=True,
        clip_gradient=0.01,
    )
    loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
    print("intermediate loss: {}\nintermediate accuracy: {}".format(loss, acc))

    ## Second training batch
    dms.delay_duration_max = 1000
    dms.setup()
    x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
    train(
        net,
        x_train,
        y_train,
        mask_train,
        lr=1e-3,
        n_epochs=50,
        early_stop=0.05,
        keep_best=True,
        cuda=True,
        clip_gradient=1,
    )
    loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
    print("intermediate 2 loss: {}\nintermediate 2 accuracy: {}".format(loss, acc))

    ## Third training batch
    dms.delay_duration_max = 4000
    dms.setup()
    x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
    train(
        net,
        x_train,
        y_train,
        mask_train,
        lr=1e-4,
        n_epochs=400,
        early_stop=0.05,
        keep_best=True,
        cuda=True,
    )
    # net.load_state_dict(torch.load(f'../models/dms_many/{i}.pt', map_location='cpu'))
    loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
    print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
    torch.save(net.state_dict(), f"../models/dms_many/{i}.pt")
    acc_vals.append(acc)
    loss_vals.append(loss)

    net = net.cpu()
    net._define_proxy_parameters()
    m1 = net.m[:, 0].detach().numpy()
    n1 = net.n[:, 0].detach().numpy()
    m2 = net.m[:, 1].detach().numpy()
    n2 = net.n[:, 1].detach().numpy()
    wi1 = net.wi_full[0].detach().numpy()
    wi2 = net.wi_full[1].detach().numpy()
    wo = net.wo_full[:, 0].detach().numpy()

    conn_space = np.array([wi1, wi2, n1, n2, m1, m2]).T
    p, c = ms.epairs(conn_space, 500, plot=False)
    p_vals.append(p)
    c_vals.append(c)

    reg_space, r2 = regression_dms(net)
    p_reg, c_reg = ms.epairs(reg_space, 500, plot=False)
    preg_vals.append(p_reg)
    creg_vals.append(c_reg)
    r2_vals.append(r2)

    n_samples = 10
    accs_tmp = []
    losses_tmp = []
    for k in range(n_samples):
        net2 = clustering.to_support_net(net, np.zeros(hidden_size))
        loss, acc = dms.test_dms(net2, x_val.cpu(), y_val.cpu(), mask_val.cpu())
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

np.savez(
    "../data/dms_results3.npz",
    p_vals,
    c_vals,
    preg_vals,
    creg_vals,
    acc_vals,
    loss_vals,
    accs_res,
    losses_res,
)
np.save("../data/dms_lr_c_boot_distr.npy", np.array(cdistr_loadings))
