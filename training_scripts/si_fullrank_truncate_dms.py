import sys
sys.path.append('../')
from low_rank_rnns.modules import *
from low_rank_rnns import dms
import pickle

size = 1024
alpha = 0.2
noise_std = 5e-2
max_rank = 10

x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000, fraction_validation_trials=.2)

losses = []
accs = []
losses_trunc = {i: [] for i in range(1, max_rank)}
accs_trunc = {i: [] for i in range(1, max_rank)}
for i in range(10):
    net = FullRankRNN(2, size, 1, noise_std, alpha)
    J_init = net.wrec.detach().numpy().copy()
    train(net, x_train, y_train, mask_train, 100, lr=1e-4, batch_size=128, clip_gradient=1., keep_best=True)
    loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
    losses.append(loss)
    accs.append(acc)
    DJ = net.wrec.detach().numpy() - J_init
    u, s, v = np.linalg.svd(DJ + J_init)

    # Truncating
    for rank in range(1, max_rank):
        DJ_new = (u[:, :rank] * s[:rank]) @ v[:rank]
        m = u[:, :rank] * np.sqrt(s[:rank]) * np.sqrt(size)
        n = (v[:rank].T * np.sqrt(s[:rank])) * np.sqrt(size)
        net_trunc = LowRankRNN(2, size, 1, noise_std, alpha, rank=rank, wi_init=net.wi, wo_init=net.wo * size,
                               m_init=torch.from_numpy(m), n_init=torch.from_numpy(n))
        loss, acc = dms.test_dms(net_trunc, x_val, y_val, mask_val)
        losses_trunc[rank].append(loss)
        accs_trunc[rank].append(acc)

results_filename = '../data/si_fr_dms_res.pkl'

with open(results_filename, 'wb') as f:
    pickle.dump([losses, accs, losses_trunc, accs_trunc], f)
