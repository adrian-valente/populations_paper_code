"""
IMPORTANT
Training Mante network for figure 3
"""

import sys
sys.path.append('../')

from low_rank_rnns import mante
from low_rank_rnns.modules import *

hidden_size = 4096
noise_std = 5e-2
alpha = 0.2
lr_base = 1e-1

mante.SCALE_CTX = .5  # easier with this
x_train, y_train, mask_train, x_val, y_val, mask_val = mante.generate_mante_data(1000)
net = LowRankRNN(4, hidden_size, 1, noise_std, alpha, rank=1, train_wi=True)
train(net, x_train, y_train, mask_train, lr=1e-2, n_epochs=100, keep_best=True, early_stop=0.01, clip_gradient=.1, cuda=True)
loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
torch.save(net.state_dict(), "../models/mante_rank1_{}.pt".format(hidden_size))