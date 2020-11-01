from math import floor
import random
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from low_rank_rnns.modules import loss_mse
from low_rank_rnns.helpers import map_device

deltaT = 20.
fixation_duration = 100
stimulus_duration = 800
delay_duration = 100
decision_duration = 500
SCALE = 1
# SCALE_CTX = 1
std_default = 5e-1
# decision targets
lo = -1
hi = 1


def setup():
    """
    Call this function whenever changing one of the global task variables (modifies other global variables)
    """
    global fixation_duration_discrete, stimulus_duration_discrete, delay_duration_discrete, \
           decision_duration_discrete, total_duration, stim_begin, stim_end, response_begin
    fixation_duration_discrete = floor(fixation_duration / deltaT)
    stim_begin = fixation_duration_discrete
    stimulus_duration_discrete = floor(stimulus_duration / deltaT)
    stim_end = stim_begin + stimulus_duration_discrete
    delay_duration_discrete = floor(delay_duration / deltaT)
    response_begin = stim_end + delay_duration_discrete
    decision_duration_discrete = floor(decision_duration / deltaT)
    total_duration = fixation_duration_discrete + stimulus_duration_discrete + delay_duration_discrete + \
                     decision_duration_discrete


setup()


def generate_data(num_trials, std=std_default, coherences=None, fraction_validation_trials=0.2, fraction_catch_trials=0.,
                  context=None):
    if coherences is None:
        coherences = [-8, -4, -2, -1, 1, 2, 4, 8]
    coherences_pos = [c for c in coherences if c >= 0]
    coherences_neg = [c for c in coherences if c < 0]

    inputs = std * torch.randn((num_trials, total_duration, 2), dtype=torch.float32)
    # inputs_context = torch.zeros((num_trials, total_duration, 2))
    # inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

    for i in range(num_trials):
        if random.random() > fraction_catch_trials:
            choice = random.choice([lo, hi])
            if len(coherences_pos) == 0:
                choice = lo
            elif len(coherences_neg) == 0:
                choice = hi
            if context is None:  # context represents the modality chosen between visual, auditory, or both
                context_ = random.randint(-1, 1)
            else:
                context_ = context
            if context_ in [1, 0]:  # set visual channel
                if choice > 0:
                    coh_current = coherences_pos[random.randint(0, len(coherences_pos) - 1)]
                else:
                    coh_current = coherences_neg[random.randint(0, len(coherences_neg) - 1)]
                inputs[i, stim_begin:stim_end, 0] += coh_current * SCALE
                # inputs[i, stim_begin:stim_end, 2] = 1. * SCALE_CTX
            if context_ in [-1, 0]:  # set auditory channel
                if choice > 0:
                    coh_current = coherences_pos[random.randint(0, len(coherences_pos) - 1)]
                else:
                    coh_current = coherences_neg[random.randint(0, len(coherences_neg) - 1)]
                inputs[i, stim_begin:stim_end, 1] += coh_current * SCALE
                # inputs[i, stim_begin:stim_end, 3] = 1. * SCALE_CTX
            targets[i, response_begin:, 0] = choice
        mask[i, response_begin:, 0] = 1

    # Split
    split_at = num_trials - int(num_trials * fraction_validation_trials)
    inputs_train, inputs_val = inputs[:split_at], inputs[split_at:]
    targets_train, targets_val = targets[:split_at], targets[split_at:]
    mask_train, mask_val = mask[:split_at], mask[split_at:]

    return inputs_train, targets_train, mask_train, inputs_val, targets_val, mask_val


def accuracy(targets, output):
    good_trials = (targets != 0).any(dim=1).squeeze()  # remove catch trials
    target_decisions = torch.sign(targets[good_trials, response_begin:, :].mean(dim=1).squeeze())
    decisions = torch.sign(output[good_trials, response_begin:, :].mean(dim=1).squeeze())
    return (target_decisions == decisions).type(torch.float32).mean()


def test(net, x, y, mask):
    x, y, mask = map_device([x, y, mask], net)
    with torch.no_grad():
        output = net(x)
        loss = loss_mse(output, y, mask)
        acc = accuracy(y, output)
    return loss.item(), acc.item()


def psychometric_curves(net):
    cohs = [-4, -2, -1, -.5, -.2, 0., .2, .5, 1, 2, 4]
    colors = ['blue', 'green', 'red']
    labels = ['visual', 'auditory', 'multimodal']
    for i, context in enumerate((1, -1, 0)):
        probs = []
        for coh in cohs:
            input, target, mask, _, _, _ = generate_data(50, coherences=[coh], context=context, fraction_validation_trials=0)
            with torch.no_grad():
                output = net(input)
                decisions = (torch.sign(output[:, response_begin:, 0].mean(dim=1).squeeze()) + 1) // 2
                probs.append(decisions.mean().item())
        plt.plot(cohs, probs, c=colors[i], label=labels[i])
    plt.xlabel('coherence')
    plt.ylabel('choices to right')
    plt.legend()


def psychometric_matrix(net, cmap='gray', figsize=None):
    n_trials = 10
    coherences = np.arange(-8, 9, 2)
    if figsize is None:
        figsize = matplotlib.rcParams['figure.figsize']
    fig, ax = plt.subplots(figsize=figsize)

    mat = np.zeros((len(coherences), len(coherences)))
    for i, coh1 in enumerate(coherences):
        for j, coh2 in enumerate(coherences):
            inputs = std_default * torch.randn((n_trials, total_duration, 2), dtype=torch.float32)
            # inputs_context = torch.zeros((n_trials, total_duration, 2))
            # inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
            inputs[:, stim_begin:stim_end, 0] += coh1 * SCALE
            inputs[:, stim_begin:stim_end, 1] += coh2 * SCALE
            # inputs[:, stim_begin:stim_end, 2] += SCALE_CTX
            # inputs[:, stim_begin:stim_end, 3] += SCALE_CTX
            output = net.forward(inputs)
            decisions = torch.sign(output[:, response_begin:, :].mean(dim=1).squeeze())
            mat[len(coherences) - j - 1, i] = decisions.mean().item()
    ax.matshow(mat, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
