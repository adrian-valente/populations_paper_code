from math import floor
import torch
import random
from low_rank_rnns.helpers import *
from low_rank_rnns.modules import loss_mse

deltaT = 20.
fixation_duration = 100
stimulus1_duration = 100
delay_duration_min = 500
delay_duration_max = 2000
stimulus2_duration = 100
decision_duration = 100

fdiffs_global = [-24, -16, -8, 8, 16, 24]
fpairs_global = [(base, base + fdiff) for fdiff in fdiffs_global
                                      for base in range(max(10 - fdiff, 10), min(34, 34 - fdiff) + 1)]
fmax = max([max(*p) for p in fpairs_global])
fmin = min([min(*p) for p in fpairs_global])
fmiddle = (fmax + fmin) / 2.
fspan = fmax - fmin


def setup():
    """
    Call this function whenever changing one of the global task variables (modifies other global variables)
    """
    global fixation_duration_discrete, stimulus1_duration_discrete, stimulus1_duration_discrete, \
        stimulus2_duration_discrete, stimulus2_duration_discrete, decision_duration_discrete, \
        min_delay_duration_discrete, max_delay_duration_discrete, total_duration, stim1_end
    fixation_duration_discrete = floor(fixation_duration/deltaT)
    stimulus1_duration_discrete = floor(stimulus1_duration/deltaT)
    stim1_end = fixation_duration_discrete + stimulus1_duration_discrete
    stimulus2_duration_discrete = floor(stimulus2_duration/deltaT)
    decision_duration_discrete = floor(decision_duration/deltaT)
    min_delay_duration_discrete = floor(delay_duration_min/deltaT)
    max_delay_duration_discrete = floor(delay_duration_max/deltaT)
    total_duration = fixation_duration_discrete + stimulus1_duration_discrete + max_delay_duration_discrete + \
        stimulus2_duration_discrete + decision_duration_discrete


setup()


def generate_data(num_trials, std=1e-2, fpairs=None, fraction_validation_trials=.2, fraction_catch_trials=0.,
                  delay_discrete=None):
    if fpairs is None:
        fpairs = fpairs_global

    inputs = std * torch.randn((num_trials, total_duration, 1))
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

    for i in range(num_trials):
        if random.random() > fraction_catch_trials:
            # Sample frequencies
            fpair = random.choice(fpairs)
            f1 = fpair[0]
            f2 = fpair[1]
            # Sample delay duration
            if delay_discrete is None:
                delay_duration = random.randint(min_delay_duration_discrete, max_delay_duration_discrete)
            else:
                delay_duration = delay_discrete
            stim2_begin = stim1_end + delay_duration
            stim2_end = stim2_begin + stimulus2_duration_discrete

            decision_end = stim2_end + decision_duration_discrete
            inputs[i, fixation_duration_discrete:stim1_end] += (f1 - fmiddle) / fspan
            inputs[i, stim2_begin:stim2_end] += (f2 - fmiddle) / fspan
            targets[i, stim2_end:decision_end] = (f1 - f2) / fspan
            mask[i, stim2_end:decision_end] = 1

  
    
    # Split
    split_at = num_trials - int(num_trials * fraction_validation_trials)
    inputs_train, inputs_val = inputs[:split_at], inputs[split_at:]
    targets_train, targets_val = targets[:split_at], targets[split_at:]
    mask_train, mask_val = mask[:split_at], mask[split_at:]


    return inputs_train, targets_train, mask_train, inputs_val, targets_val, mask_val


def accuracy_romo(output, targets, mask):
    good_trials = (targets != 0).any(dim=1).squeeze()  # eliminates catch trials
    mask_bool = (mask[good_trials, :, 0] == 1)
    targets_filtered = torch.stack(targets[good_trials].squeeze()[mask_bool].chunk(good_trials.sum()))
    target_decisions = torch.sign(targets_filtered.mean(dim=1))
    decisions_filtered = torch.stack(output[good_trials].squeeze()[mask_bool].chunk(good_trials.sum()))
    decisions = torch.sign(decisions_filtered.mean(dim=1))
    return (target_decisions == decisions).type(torch.float32).mean()


def test_romo(net, x, y, mask):
    x, y, mask = map_device([x, y, mask], net)
    with torch.no_grad():
        output = net(x)
        loss = loss_mse(output, y, mask).item()
        acc = accuracy_romo(output, y, mask).item()
    return loss, acc


def psychometric_matrices(net, colorbar=False, cmap='gray', binarize=0, figsize=(6, 5), ylabel=False, ax=None):
    n_trials = 10
    delay_duration = 1000
    delay_duration_discrete = floor(delay_duration / deltaT)
    stim2_begin = stim1_end + delay_duration_discrete
    stim2_end = stim2_begin + stimulus2_duration_discrete
    f_vect = np.arange(10, 35, 2)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    mat = np.zeros((len(f_vect), len(f_vect)))
    for i, f1 in enumerate(f_vect):
        for j, f2 in enumerate(f_vect):
            inputs,_,_,_,_,_ = generate_data(n_trials, std=1e-2, fpairs=[(f1,f2)], fraction_validation_trials=.0, fraction_catch_trials=0.,
                  delay_discrete=delay_duration_discrete)
            output = net.forward(inputs)
            if binarize == 1:
                decisions = torch.sign(output[:, stim2_end:stim2_end + decision_duration_discrete, :].mean(dim=1).squeeze())
                mat[len(f_vect) - j - 1, i] = decisions.mean()
            else:
                decisions = output[0, stim2_end:stim2_end + decision_duration_discrete, 0].detach().numpy()
                mat[len(f_vect) - j - 1, i] = np.mean(decisions)
            
    mappable = ax.matshow(mat, cmap=cmap)
    ax.set_xlabel('$f_1$')
    if ylabel: ax.set_ylabel('$f_2$')
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(mappable, ax=ax)
    return ax


def psychometric_curve_romo(net, color=None, figsize=None, ax=None, lw=2):
    fdiffs = [-24, -16, -8, -4, 0, 4, 8, 16, 24]
    probs = []
    delay_duration = 1000
    delay_duration_discrete = floor(delay_duration / deltaT)
    stim2_begin = stim1_end + delay_duration_discrete
    stim2_end = stim2_begin + stimulus2_duration_discrete

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    for fdiff in fdiffs:
        fpairs = [(base, base + fdiff) for base in range(max(10 - fdiff, 10), min(34, 34 - fdiff) + 1)]
        input, target, mask, _, _, _ = generate_data(50, fpairs=fpairs, fraction_validation_trials=0,
                                                     delay_discrete=delay_duration_discrete)
        with torch.no_grad():
            output = net(input)
            decisions = (-torch.sign(output[:, stim2_end:stim2_end + decision_duration_discrete, 0].mean(dim=1).squeeze()) + 1) // 2
            probs.append(decisions.mean().item())
    ax.plot(fdiffs, probs, c=color, lw=2)
    ax.set_xlabel('frequency difference')
    ax.set_ylabel('choices to right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([0., .5, 1.])
    return ax
