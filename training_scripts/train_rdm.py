import sys
sys.path.append('../')

from low_rank_rnns import rdm
from low_rank_rnns.modules import *

hidden_size = 512
noise_std = 5e-2
alpha = 0.2

x_train, y_train, mask_train, x_val, y_val, mask_val = rdm.generate_rdm_data(1000)
net = LowRankRNN(1, hidden_size, 1, noise_std, alpha, rank=1)
train(net, x_train, y_train, mask_train, lr=1e-2, n_epochs=30, batch_size=128, keep_best=True, cuda=True)
loss, acc = rdm.test_rdm(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
torch.save(net.state_dict(), f"../models/rdm_rank1_{hidden_size}.pt")
