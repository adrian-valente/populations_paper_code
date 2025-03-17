"""
Testing different ranks for DMS task
"""

from low_rank_rnns import dms
from low_rank_rnns.modules import *
import pickle

hidden_size = 1024
noise_std = 5e-2
alpha = 0.2
n_epochs = 100
losses = dict()
accs = dict()
n_samples = 10

dms.delay_duration_max = 700
dms.setup()
x_train0, y_train0, mask_train0, x_val0, y_val0, mask_val0 = dms.generate_dms_data(1000)

dms.delay_duration_max = 1000
dms.setup()
x_train1, y_train1, mask_train1, x_val1, y_val1, mask_val1 = dms.generate_dms_data(1000)


losses["full"] = []
accs["full"] = []
for rank in range(5, 0, -1):
    losses[rank] = []
    accs[rank] = []
for _ in range(n_samples):
    net_fr = FullRankRNN(2, hidden_size, 1, noise_std, alpha)
    train(
        net_fr,
        x_train0,
        y_train0,
        mask_train0,
        n_epochs=n_epochs // 4,
        lr=1e-4,
        keep_best=True,
        cuda=True,
    )
    train(
        net_fr,
        x_train1,
        y_train1,
        mask_train1,
        n_epochs * 2,
        lr=1e-5,
        keep_best=True,
        cuda=True,
    )
    loss, acc = dms.test_dms(net_fr, x_val1, y_val1, mask_val1)
    print(f"full rank, loss={loss:.3f}, acc={acc:.2f}")
    losses["full"].append(loss)
    accs["full"].append(acc)

    # Retrieve connectivity of full rank net
    wi_init = net_fr.wi.detach()
    wo_init = net_fr.wo.detach() * hidden_size
    u, s, v = np.linalg.svd(net_fr.wrec.detach().cpu().numpy())
    m_init = torch.from_numpy(s[:5] * u[:, :5]).to(device=net_fr.wrec.device) * sqrt(
        hidden_size
    )
    n_init = torch.from_numpy(v[:5, :].transpose()).to(
        device=net_fr.wrec.device
    ) * sqrt(hidden_size)

    for rank in range(5, 0, -1):
        net = LowRankRNN(
            2,
            hidden_size,
            1,
            noise_std,
            alpha,
            rank=rank,
            wi_init=wi_init,
            n_init=n_init[:, :rank],
            m_init=m_init[:, :rank],
            wo_init=wo_init,
        )
        train(
            net,
            x_train0,
            y_train0,
            mask_train0,
            lr=1e-2,
            n_epochs=n_epochs // 4,
            keep_best=True,
            cuda=True,
        )
        train(
            net,
            x_train1,
            y_train1,
            mask_train1,
            lr=1e-2,
            n_epochs=n_epochs,
            keep_best=True,
            cuda=True,
        )
        loss, acc = dms.test_dms(net, x_val1, y_val1, mask_val1)
        print(f"rank {rank}, loss={loss:.3f}, acc={acc:.2f}")
        losses[rank].append(loss)
        accs[rank].append(acc)

with open("../data/si_rank_loss_dms_loss2.pkl", "wb") as file:
    pickle.dump(losses, file)
with open("../data/si_rank_loss_dms_acc2.pkl", "wb") as file:
    pickle.dump(accs, file)
