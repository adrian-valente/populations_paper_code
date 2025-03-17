"""
This script trains a rank 2 net for the DMS task with shaping (first delay is limited to a maximum of 1s, then
increased)
"""

import low_rank_rnns.dms as dms
from low_rank_rnns.modules import *

hidden_size = 512
noise_std = 5e-2
alpha = 0.2

global_nepochs = 20

# Training full rank network
net_fr = FullRankRNN(2, hidden_size, 1, noise_std, alpha)
dms.delay_duration_max = 2000
dms.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
initial_wrec = net_fr.wrec.detach().cpu().numpy()
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
torch.save(net_fr.state_dict(), f"../models/dms_fullrank_{hidden_size}.pt")
# net_fr.load_state_dict(torch.load(f"../models/dms_fullrank_{hidden_size}.pt"))
loss, acc = dms.test_dms(net_fr, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
dms.confusion_matrix(net_fr)
wi_init = net_fr.wi.detach()
wo_init = net_fr.wo.detach() * hidden_size
u, s, v = np.linalg.svd(net_fr.wrec.detach().cpu().numpy())
m_init = torch.from_numpy(s[:2] * u[:, :2]).to(device=net_fr.wrec.device) * sqrt(
    hidden_size
)
n_init = torch.from_numpy(v[:2, :].transpose()).to(device=net_fr.wrec.device) * sqrt(
    hidden_size
)
print(m_init.std().item() * sqrt(hidden_size))
print(n_init.std().item() * sqrt(hidden_size))
print(wo_init.std().item() * hidden_size)
print(wi_init.std().item())

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


dms.delay_duration_max = 700
dms.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
train(
    net,
    x_train,
    y_train,
    mask_train,
    lr=1e-2,
    n_epochs=100,
    early_stop=0.05,
    keep_best=True,
    cuda=True,
    clip_gradient=0.01,
)
loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
dms.confusion_matrix(net)

## Second training batch
dms.delay_duration_max = 1000
dms.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
# train(net, x_train, y_train, mask_train, lr=global_lr, n_epochs=global_nepochs, keep_best=True, plot_learning_curve=True,
#      plot_gradient=True)
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
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
dms.confusion_matrix(net)

## Third training batch
dms.delay_duration_max = 4000
dms.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
# train(net, x_train, y_train, mask_train, lr=global_lr, n_epochs=global_nepochs, keep_best=True, plot_learning_curve=True,
#      plot_gradient=True)
train(
    net,
    x_train,
    y_train,
    mask_train,
    lr=1e-4,
    batch_size=128,
    n_epochs=40,
    early_stop=0.05,
    keep_best=True,
    cuda=True,
)
loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
dms.confusion_matrix(net)

torch.save(net.state_dict(), f"../models/dms_rank2_{hidden_size}.pt")
