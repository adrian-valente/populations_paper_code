import sys
sys.path.append('../')
from low_rank_rnns import mante
from low_rank_rnns.modules import *
import pickle

hidden_size = 1024
noise_std = 5e-2
alpha = 0.2
n_epochs = 200
losses = dict()
accs = dict()

x_train, y_train, mask_train, x_val, y_val, mask_val, epochs = mante.generate_mante_data(1000,
                                                                    fraction_validation_trials=.2)


net = FullRankRNN(4, hidden_size, 1, noise_std, alpha, train_wi=True)
train(net, x_train, y_train, mask_train, lr=1e-4, n_epochs=n_epochs, keep_best=True, cuda=True)
loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
print(f"full rank, loss={loss:.3f}, acc={acc:.2f}")
losses['full'] = loss
accs['full'] = acc

for rank in range(5, 0, -1):
    net = LowRankRNN(4, hidden_size, 1, noise_std, alpha, rank=rank, train_wi=True)
    train(net, x_train, y_train, mask_train, lr=1e-2, n_epochs=n_epochs, keep_best=True, cuda=True)
    loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
    print(f"rank {rank}, loss={loss:.3f}, acc={acc:.2f}")
    losses[rank] = loss
    accs[rank] = acc

with open('../data/si_rank_loss_mante_loss.pkl', 'wb') as file:
    pickle.dump(losses, file)
with open('../data/si_rank_loss_mante_acc.pkl', 'wb') as file:
    pickle.dump(accs, file)