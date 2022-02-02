"""
Testing different ranks for WM task
"""

import sys
sys.path.append('../')

from low_rank_rnns import romo
from low_rank_rnns.modules import *
import pickle

hidden_size = 512
noise_std = 5e-2
alpha = 0.2
losses = dict()
accs = dict()
n_epochs = 300
n_samples = 10

romo.delay_duration_max = 2000
romo.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = romo.generate_data(1000)

losses['full'] = []
accs['full'] = []
for _ in range(n_samples):
    net = FullRankRNN(1, hidden_size, 1, noise_std, alpha)
    train(net, x_train, y_train, mask_train, lr=1e-4, n_epochs=n_epochs, keep_best=True, cuda=True)
    loss, acc = romo.test_romo(net, x_val, y_val, mask_val)
    print(f"full rank, loss={loss:.3f}, acc={acc:.2f}")
    losses['full'].append(loss)
    accs['full'].append(acc)

for rank in range(5, 0, -1):
    losses[rank] = []
    accs[rank] = []
    for _ in range(n_samples):
        net = LowRankRNN(1, hidden_size, 1, noise_std, alpha, rank=rank)
        train(net, x_train, y_train, mask_train, lr=1e-2, n_epochs=n_epochs, keep_best=True, cuda=True)
        loss, acc = romo.test_romo(net, x_val, y_val, mask_val)
        print(f"rank {rank}, loss={loss:.3f}, acc={acc:.2f}")
        losses[rank].append(loss)
        accs[rank].append(acc)

with open('../data/si_rank_loss_romo_loss2.pkl', 'wb') as file:
    pickle.dump(losses, file)
with open('../data/si_rank_loss_romo_acc2.pkl', 'wb') as file:
    pickle.dump(accs, file)
