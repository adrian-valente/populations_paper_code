"""
Modified on 11/01
"""

import sys
sys.path.append('../')

from low_rank_rnns import rdm
from low_rank_rnns.modules import *

hidden_size = 512
noise_std = 5e-2
alpha = 0.2

x_train, y_train, mask_train, x_val, y_val, mask_val = rdm.generate_rdm_data(1000)

net = LowRankRNN(1, hidden_size, 1, noise_std, alpha, rank=1)
train(net, x_train, y_train, mask_train, lr=5e-3, n_epochs=20, batch_size=32, keep_best=True, cuda=False)
loss, acc = rdm.test_rdm(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
# torch.save(net.state_dict(), f"../models/rdm_rank1_{hidden_size}-2.pt")
