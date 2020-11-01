import sys
sys.path.append('../')

from low_rank_rnns import raposo
from low_rank_rnns.modules import *

hidden_size = 2048
noise_std = 5e-2
alpha = 0.2

x_train, y_train, mask_train, x_val, y_val, mask_val = raposo.generate_data(1000)
si_init = torch.tensor([.5, .5])
net = NoScalingOptimizedLowRankRNN(2, hidden_size, 1, noise_std, alpha, rank=1, train_si=False, si_init=si_init)
print(net.si.requires_grad)
train(net, x_train, y_train, mask_train, lr=1e-2, n_epochs=30, batch_size=128, keep_best=True, cuda=True)
print(net.si)
loss, acc = raposo.test(net, x_val, y_val, mask_val)
print("final loss: {}\nfinal accuracy: {}".format(loss, acc))
torch.save(net.state_dict(), f"../models/raposo_rank1_{hidden_size}-6.pt")
