"""
This script trains a rank 2 net for the DMS task with shaping (first delay is limited to a maximum of 1s, then
increased)
"""

import low_rank_rnns.dms as dms
from low_rank_rnns.modules import *

hidden_size = 4096
noise_std = 5e-2
alpha = 0.2

net = SupportLowRankRNN(
    2, hidden_size, 1, noise_std, alpha, rank=2, n_supports=2, gaussian_basis_dim=7
)
net.load_state_dict(torch.load("../models/dms_pretraining_2pop.pt"))

## Second training batch
dms.delay_duration_max = 1000
dms.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
train(
    net,
    x_train,
    y_train,
    mask_train,
    lr=1e-5,
    n_epochs=5,
    keep_best=True,
    cuda=True,
    clip_gradient=1,
    resample=True,
)
loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
dms.confusion_matrix(net)

## Third training batch
dms.delay_duration_max = 4000
dms.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
train(
    net,
    x_train,
    y_train,
    mask_train,
    lr=1e-5,
    n_epochs=5,
    keep_best=True,
    cuda=True,
    resample=True,
)
loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
dms.confusion_matrix(net)

torch.save(net.state_dict(), f"../models/dms_retrained_2pop.pt")
