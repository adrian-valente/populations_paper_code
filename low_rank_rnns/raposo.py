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
ctx_only_pre_duration = 350
stimulus_duration = 800
delay_duration = 100
decision_duration = 20
SCALE = 1e-1
SCALE_CTX = 1e-1
std_default = 1e-1
# decision targets
lo = -1
hi = 1


def setup():
    """
    Call this function whenever changing one of the global task variables (modifies other global variables)
    """
    global fixation_duration_discrete, stimulus_duration_discrete, ctx_only_pre_duration_discrete, \
        delay_duration_discrete, decision_duration_discrete, total_duration, stim_begin, stim_end, response_begin
    fixation_duration_discrete = floor(fixation_duration / deltaT)
    ctx_only_pre_duration_discrete = floor(ctx_only_pre_duration / deltaT)
    stimulus_duration_discrete = floor(stimulus_duration / deltaT)
    delay_duration_discrete = floor(delay_duration / deltaT)
    decision_duration_discrete = floor(decision_duration / deltaT)

    stim_begin = fixation_duration_discrete + ctx_only_pre_duration_discrete
    stim_end = stim_begin + stimulus_duration_discrete
    response_begin = stim_end + delay_duration_discrete
    total_duration = fixation_duration_discrete + stimulus_duration_discrete + delay_duration_discrete + \
                     ctx_only_pre_duration_discrete + decision_duration_discrete


setup()


def generate_data(num_trials, std=std_default, coherences=None, fraction_validation_trials=0.2, fraction_catch_trials=0.,
                  context=None):
    if coherences is None:
        coherences = [-4, -2, -1, 1, 2, 4]
    coherences_pos = [c for c in coherences if c >= 0]
    coherences_neg = [c for c in coherences if c < 0]

    inputs_sensory = std * torch.randn((num_trials, total_duration, 2), dtype=torch.float32)
    inputs_context = torch.zeros((num_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
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
                inputs[i, fixation_duration_discrete:stim_end, 2] = 1. * SCALE_CTX
            if context_ in [-1, 0]:  # set auditory channel
                if choice > 0:
                    coh_current = coherences_pos[random.randint(0, len(coherences_pos) - 1)]
                else:
                    coh_current = coherences_neg[random.randint(0, len(coherences_neg) - 1)]
                inputs[i, stim_begin:stim_end, 1] += coh_current * SCALE
                inputs[i, fixation_duration_discrete:stim_end, 3] = 1. * SCALE_CTX
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


def psychometric_matrices(net, cmap='gray', figsize=None, plot=True):
    n_trials = 10
    coherences = np.arange(-8, 9, 2)
    if plot:
        if figsize is None:
            figsize = matplotlib.rcParams['figure.figsize']
        fig, ax = plt.subplots(2, 1, figsize=figsize)

    mat = np.zeros((2, len(coherences), len(coherences)))
    for ctx in (0, 1):
        for i, coh1 in enumerate(coherences):
            for j, coh2 in enumerate(coherences):
                inputs_sensory = std_default * torch.randn((n_trials, total_duration, 2), dtype=torch.float32)
                inputs_context = torch.zeros((n_trials, total_duration, 2))
                inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
                inputs[:, stim_begin:stim_end, 0] += coh1 * SCALE
                inputs[:, stim_begin:stim_end, 1] += coh2 * SCALE
                inputs[:, stim_begin:stim_end, 2 + ctx] += SCALE_CTX
                output = net.forward(inputs)
                decisions = torch.sign(output[:, response_begin:, :].mean(dim=1).squeeze())
                mat[ctx, len(coherences) - j - 1, i] = decisions.mean().item()

        if plot:
            ax[ctx].matshow(mat[ctx], cmap=cmap, vmin=-1, vmax=1)
            ax[ctx].set_xticks([])
            ax[ctx].set_yticks([])
    if plot:
        return fig
    else:
        return mat


def generate_ordered_inputs(std=0.5):
    n_trials_spec = 10
    n_trials = n_trials_spec * 3 * 2
    i = 0
    coh_level = 1
    inputs_sensory = std * torch.randn((n_trials, total_duration, 2), dtype=torch.float32)
    inputs_context = torch.zeros((n_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    contexts = np.zeros(n_trials)
    cohs1 = np.zeros(n_trials)
    cohs2 = np.zeros(n_trials)
    choices = np.zeros(n_trials)
    for context in (1, -1, 0):
        for choice in (-1, 1):
            for _ in range(n_trials_spec):
                contexts[i] = context
                choices[i] = choice
                if context in [1, 0]:
                    inputs[i, stim_begin:stim_end, 0] += choice * coh_level * SCALE
                    inputs[i, stim_begin:stim_end, 2] = 1. * SCALE_CTX
                    cohs1[i] = choice * coh_level
                if context in [-1, 0]:
                    inputs[i, stim_begin:stim_end, 1] += choice * coh_level * SCALE
                    inputs[i, stim_begin:stim_end, 3] = 1. * SCALE_CTX
                    cohs2[i] = choice * coh_level
                i += 1
    return inputs, contexts, cohs1, cohs2, choices


def generate_ordered_inputs2(std=0.5):
    n_trials_spec = 10
    n_trials = n_trials_spec * 3 * 2 * 2
    i = 0
    coh_level = 1
    inputs_sensory = std * torch.randn((n_trials, total_duration, 2), dtype=torch.float32)
    inputs_context = torch.zeros((n_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    contexts = np.zeros(n_trials)
    cohs1 = np.zeros(n_trials)
    cohs2 = np.zeros(n_trials)
    choices = np.zeros(n_trials)
    for context in (1, -1, 0):
        for coh1 in (-1, 1):
            for coh2 in (-1, 1):
                for _ in range(n_trials_spec):
                    if context == 1:
                        choice = coh1
                    elif context == -1:
                        choice = coh2
                    else:
                        choice = (coh1 + coh2) / 2
                    contexts[i] = context
                    choices[i] = choice
                    if context in [1, 0]:
                        inputs[i, stim_begin:stim_end, 0] += coh1 * coh_level * SCALE
                        inputs[i, stim_begin:stim_end, 2] = 1. * SCALE_CTX
                        cohs1[i] = coh1 * coh_level
                    if context in [-1, 0]:
                        inputs[i, stim_begin:stim_end, 1] += coh2 * coh_level * SCALE
                        inputs[i, stim_begin:stim_end, 3] = 1. * SCALE_CTX
                        cohs2[i] = coh2 * coh_level
                    i += 1
    return inputs, contexts, cohs1, cohs2, choices