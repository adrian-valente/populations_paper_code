"""
Training Mante net WITH scalings and with input training. Gives the 3-population solution
"""

from low_rank_rnns import mante
from low_rank_rnns.modules import *

hidden_size = 4096
noise_std = 5e-2
alpha = 0.2
lr_base = 1e-1
mante.SCALE_CTX = 1.0

x_train, y_train, mask_train, x_val, y_val, mask_val = mante.generate_mante_data(1000)
net = OptimizedLowRankRNN(
    4, hidden_size, 1, noise_std, alpha, rank=1, train_wi=True, train_wo=True
)
train(
    net,
    x_train,
    y_train,
    mask_train,
    lr=lr_base / sqrt(hidden_size),
    n_epochs=50,
    keep_best=True,
    cuda=True,
    clip_gradient=1,
)
loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
torch.save(net.state_dict(), "../models/mante_rank1_{}-3.pt".format(hidden_size))
