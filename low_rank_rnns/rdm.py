import numpy as np
import torch
from low_rank_rnns.modules import loss_mse
import matplotlib.pyplot as plt
from low_rank_rnns.helpers import map_device


# RDM constants
deltaT = 20.
fixation_duration = 100
stimulus_duration = 800
delay_duration = 300
decision_duration = 300
SCALE = 3.2
# decision targets
lo = -1
hi = 1

def setup():
    """
    Redefine global variables for the stimuli
    :return:
    """
    global fixation_duration_discrete, stimulus_duration_discrete, delay_duration_discrete, decision_duration_discrete
    global stimulus_end, response_begin
    global total_duration
    global index_response
    fixation_duration_discrete = int(fixation_duration / deltaT)
    stimulus_duration_discrete = int(stimulus_duration / deltaT)
    stimulus_end = fixation_duration_discrete + stimulus_duration_discrete
    delay_duration_discrete = int(delay_duration / deltaT)
    response_begin = stimulus_end + delay_duration_discrete
    decision_duration_discrete = int(decision_duration / deltaT)
    total_duration = fixation_duration_discrete + stimulus_duration_discrete + delay_duration_discrete + \
                     decision_duration_discrete
    index_response = fixation_duration_discrete + stimulus_duration_discrete + delay_duration_discrete


setup()


def generate_rdm_data(num_trials, coherences=None, std=3e-2, fraction_validation_trials=.2, fraction_catch_trials=0.):
    if coherences is None:
        coherences = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]

    inputs = std * torch.randn((num_trials, total_duration, 1), dtype=torch.float32)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

    for i in range(num_trials):
        if np.random.rand() > fraction_catch_trials:
            coh_current = coherences[int(np.random.rand() * len(coherences))]
            inputs[i, fixation_duration_discrete:stimulus_end, 0] += coh_current * SCALE / 100
            targets[i, response_begin:, 0] = hi if coh_current > 0 else lo
        mask[i, response_begin:, 0] = 1

    # Split
    split_at = num_trials - int(num_trials * fraction_validation_trials)
    inputs_train, inputs_val = inputs[:split_at], inputs[split_at:]
    targets_train, targets_val = targets[:split_at], targets[split_at:]
    mask_train, mask_val = mask[:split_at], mask[split_at:]

    return inputs_train, targets_train, mask_train, inputs_val, targets_val, mask_val


def accuracy_rdm(targets, output):
    good_trials = (targets != 0).any(dim=1).squeeze()
    target_decisions = torch.sign(targets[good_trials, index_response:, :].mean(dim=1).squeeze())
    decisions = torch.sign(output[good_trials, index_response:, :].mean(dim=1).squeeze())
    return (target_decisions == decisions).type(torch.float32).mean()


def test_rdm(net, x, y, mask):
    x, y, mask = map_device([x, y, mask], net)
    with torch.no_grad():
        output = net(x)
        loss = loss_mse(output, y, mask)
        acc = accuracy_rdm(y, output)
    return loss.item(), acc.item()


def psychometric_curve_rdm(net, color=None, figsize=None, ax=None, lw=3):
    cohs = [-4, -2, -1, -.5, -.2, 0., .2, .5, 1, 2, 4]
    probs = []
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    for coh in cohs:
        input, target, mask, _, _, _ = generate_rdm_data(50, coherences=[coh], fraction_validation_trials=0)
        with torch.no_grad():
            output = net(input)
            decisions = (torch.sign(output[:, index_response:, 0].mean(dim=1).squeeze()) + 1) // 2
            probs.append(decisions.mean().item())
    ax.plot(cohs, probs, c=color, lw=lw)
    ax.set_xlabel('coherence')
    ax.set_ylabel('choices to right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([0., .5, 1.])
    return ax


def psychometric_matrix(net, cmap='gray', figsize=(6, 3)):
    cohs = list(range(-9, 10, 2))
    probs = []
    for coh in cohs:
        input, target, mask, _, _, _ = generate_rdm_data(50, coherences=[coh], fraction_validation_trials=0)
        with torch.no_grad():
            output = net(input)
            decisions = (torch.sign(output[:, index_response:, 0].mean(dim=1).squeeze()) + 1) // 2
            probs.append(decisions.mean().item())
    mat = np.array(probs).reshape((1, -1))
    plt.matshow(mat, cmap=cmap, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])


def plot_outputs(net):
    cohs = [-16, -4, -1, 1, 4, 16]
    ax = plt.axes()
    for coh in cohs:
        input_c, target_c, mask_c, _, _, _ = generate_rdm_data(100, coherences=[coh], fraction_validation_trials=0.)
        output_c = net.forward(input_c)
        seq_len = input_c.shape[1] + 1
        output_c = output_c.squeeze().detach().numpy()
        output_mean = np.mean(output_c, axis=0)
        output_std = np.std(output_c, axis=0)
        ax.plot(output_mean, label=str(coh))
        ax.fill_between(np.arange(seq_len - 1), output_mean - output_std, output_mean + output_std, alpha=0.5)
    ax.legend()
    ax.set_xlabel("$t$")
    ax.set_ylabel("$z(t)$")
    plt.show()
