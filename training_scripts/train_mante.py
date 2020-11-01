"""
Train Mante network with context input throughout trial and without training input vector.
"""

import sys
sys.path.append('../')

from low_rank_rnns import mante
from low_rank_rnns.modules import *

hidden_size = 4096
noise_std = 5e-2
alpha = 0.2
lr_base = 1e-1

x_train, y_train, mask_train, x_val, y_val, mask_val, _ = mante.generate_mante_data(1000, fraction_validation_trials=.2)
wi_init = torch.randn((4, hidden_size))
wi_init[2:4] *= .5
net = LowRankRNN(4, hidden_size, 1, noise_std, alpha, rank=1, wi_init=wi_init)
net.si.requires_grad = False  # do not train the scale of inputs
train(net, x_train, y_train, mask_train, lr=1e-2, n_epochs=60, keep_best=True, cuda=True,
      clip_gradient=1)
loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
torch.save(net.state_dict(), "../models/mante_throughout-noscale-noimp-{}.pt".format(hidden_size))