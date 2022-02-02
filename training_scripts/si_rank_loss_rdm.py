"""
Testing different ranks for DM task
"""

import sys
sys.path.append('../')

from low_rank_rnns import rdm
from low_rank_rnns.modules import *
import pickle

hidden_size = 512
noise_std = 5e-2
alpha = 0.2
losses = dict()
accs = dict()
n_epochs = 40
n_samples = 10

x_train, y_train, mask_train, x_val, y_val, mask_val = rdm.generate_rdm_data(1000)

losses['full'] = []
accs['full'] = []
for _ in range(n_samples):
    net = FullRankRNN(1, hidden_size, 1, noise_std, alpha)
    train(net, x_train, y_train, mask_train, lr=1e-4, n_epochs=n_epochs * 2, keep_best=True, cuda=True)
    loss, acc = rdm.test_rdm(net, x_val, y_val, mask_val)
    print(f"full rank, loss={loss:.3f}, acc={acc:.2f}")
    losses['full'].append(loss)
    accs['full'].append(acc)

for rank in range(5, 0, -1):
    losses[rank] = []
    accs[rank] = []
    for _ in range(n_samples):
        net = LowRankRNN(1, hidden_size, 1, noise_std, alpha, rank=rank)
        train(net, x_train, y_train, mask_train, lr=5e-3, n_epochs=n_epochs, batch_size=32, keep_best=True, cuda=True)
        loss, acc = rdm.test_rdm(net, x_val, y_val, mask_val)
        print(f"rank {rank}, loss={loss:.3f}, acc={acc:.2f}")
        losses[rank].append(loss)
        accs[rank].append(acc)

with open('../data/si_rank_loss_rdm_loss2.pkl', 'wb') as file:
    pickle.dump(losses, file)
with open('../data/si_rank_loss_rdm_acc2.pkl', 'wb') as file:
    pickle.dump(accs, file)
