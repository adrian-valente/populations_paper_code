from math import floor, sqrt
import numpy as np
import torch
from low_rank_rnns.modules import loss_mse
import low_rank_rnns.ranktwo as ranktwo
import matplotlib.pyplot as plt
from low_rank_rnns.helpers import map_device, remove_axes

# task constants
deltaT = 20.
tau = 100
alpha = deltaT / tau

fixation_duration = 100
stimulus1_duration_min = 500
stimulus1_duration_max = 500
delay_duration_min = 500
delay_duration_max = 3000
stimulus2_duration_min = 500
stimulus2_duration_max = 500
decision_duration = 1000

## Defining some global variables, whoses values can also be set in reset
fixation_duration_discrete = floor(fixation_duration / deltaT)
min_stimulus1_duration_discrete = floor(stimulus1_duration_min / deltaT)
max_stimulus1_duration_discrete = floor(stimulus1_duration_max / deltaT)
min_stimulus2_duration_discrete = floor(stimulus2_duration_min / deltaT)
max_stimulus2_duration_discrete = floor(stimulus2_duration_max / deltaT)
decision_duration_discrete = floor(decision_duration / deltaT)
min_delay_duration_discrete = floor(delay_duration_min / deltaT)
max_delay_duration_discrete = floor(delay_duration_max / deltaT)
total_duration = fixation_duration_discrete + max_stimulus1_duration_discrete + max_delay_duration_discrete + \
    max_stimulus2_duration_discrete + decision_duration_discrete


def setup():
    global fixation_duration_discrete, min_stimulus1_duration_discrete, max_stimulus1_duration_discrete, \
           min_stimulus2_duration_discrete, max_stimulus2_duration_discrete, decision_duration_discrete, \
           min_delay_duration_discrete, max_delay_duration_discrete, total_duration
    fixation_duration_discrete = floor(fixation_duration / deltaT)
    min_stimulus1_duration_discrete = floor(stimulus1_duration_min / deltaT)
    max_stimulus1_duration_discrete = floor(stimulus1_duration_max / deltaT)
    min_stimulus2_duration_discrete = floor(stimulus2_duration_min / deltaT)
    max_stimulus2_duration_discrete = floor(stimulus2_duration_max / deltaT)
    decision_duration_discrete = floor(decision_duration / deltaT)
    min_delay_duration_discrete = floor(delay_duration_min / deltaT)
    max_delay_duration_discrete = floor(delay_duration_max / deltaT)
    total_duration = fixation_duration_discrete + max_stimulus1_duration_discrete + max_delay_duration_discrete + \
        max_stimulus2_duration_discrete + decision_duration_discrete


def generate_dms_data(num_trials, type=None, fraction_validation_trials=.2, fraction_catch_trials=0., std=3e-2):
    x = std * torch.randn(num_trials, total_duration, 2)
    y = torch.zeros(num_trials, total_duration, 1)
    mask = torch.zeros(num_trials, total_duration, 1)

    types = ['A-A', 'A-B', 'B-A', 'B-B']
    for i in range(num_trials):
        if np.random.rand() > fraction_catch_trials:
            if type is None:
                cur_type = types[int(np.random.rand() * 4)]
            else:
                cur_type = type

            if cur_type == 'A-A':
                input1 = 1
                input2 = 1
                choice = 1
            elif cur_type == 'A-B':
                input1 = 1
                input2 = 0
                choice = -1
            elif cur_type == 'B-A':
                input1 = 0
                input2 = 1
                choice = -1
            elif cur_type == 'B-B':
                input1 = 0
                input2 = 0
                choice = 1

            # Sample durations
            delay_duration = np.random.uniform(delay_duration_min, delay_duration_max)
            delay_duration_discrete = floor(delay_duration / deltaT)
            stimulus1_duration = np.random.uniform(stimulus1_duration_min, stimulus1_duration_max)
            stimulus1_duration_discrete = floor(stimulus1_duration / deltaT)
            stimulus2_duration = np.random.uniform(stimulus2_duration_min, stimulus2_duration_max)
            stimulus2_duration_discrete = floor(stimulus2_duration / deltaT)
            decision_time_discrete = fixation_duration_discrete + stimulus1_duration_discrete + \
                                     delay_duration_discrete + stimulus2_duration_discrete
            stim1_begin = fixation_duration_discrete
            stim1_end = stim1_begin + stimulus1_duration_discrete
            stim2_begin = stim1_end + delay_duration_discrete
            stim2_end = stim2_begin + stimulus2_duration_discrete

            x[i, stim1_begin:stim1_end, 0] += input1
            x[i, stim1_begin:stim1_end, 1] += 1 - input1
            x[i, stim2_begin:stim2_end, 0] += input2
            x[i, stim2_begin:stim2_end, 1] += 1 - input2
            y[i, decision_time_discrete:decision_time_discrete + decision_duration_discrete] = choice
            mask[i, decision_time_discrete:decision_time_discrete + decision_duration_discrete] = 1

    # Split
    split_at = x.shape[0] - floor(x.shape[0] * fraction_validation_trials)
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]
    (mask_train, mask_val) = mask[:split_at], mask[split_at:]

    return x_train, y_train, mask_train, x_val, y_val, mask_val


def accuracy_dms(output, targets, mask):
    good_trials = (targets != 0).any(dim=1).squeeze()  # eliminates catch trials
    mask_bool = (mask[good_trials, :, 0] == 1)
    targets_filtered = torch.stack(targets[good_trials].squeeze()[mask_bool].chunk(good_trials.sum()))
    target_decisions = torch.sign(targets_filtered.mean(dim=1))
    decisions_filtered = torch.stack(output[good_trials].squeeze()[mask_bool].chunk(good_trials.sum()))
    decisions = torch.sign(decisions_filtered.mean(dim=1))
    return (target_decisions == decisions).type(torch.float32).mean()


def test_dms(net, x, y, mask):
    x, y, mask = map_device([x, y, mask], net)
    with torch.no_grad():
        output = net(x)
        loss = loss_mse(output, y, mask).item()
        acc = accuracy_dms(output, y, mask).item()
    return loss, acc


def plot_trial_epochs(net, input, epochs, rect=None, savepath=None, fp_load=None, sizes=1., axes=None):
    """
    Helper for supplementary figures plotting of detailed state-space trajectories
    """
    def get_input(input, time):
        if input[0, time, 0] == 0 and input[0, time, 1] == 0:
            return 0
        elif input[0, time, 0] == 1:
            return 1
        else:
            return 2

    m1 = net.m[:,0].detach().numpy()
    m2 = net.m[:,1].detach().numpy()
    output, trajectories = net.forward(input, return_dynamics=True)
    trajectories = trajectories.squeeze().detach().numpy()
    proj1 = trajectories @ m1 / net.hidden_size
    proj2 = trajectories @ m2 / net.hidden_size
    if rect is None:
        xmin, xmax, ymin, ymax = proj1.min(), proj1.max(), proj2.min(), proj2.max()
    else:
        xmin, xmax, ymin, ymax = rect
    for i in range(len(epochs) - 1):
        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes[i]
        if fp_load is not None:
            ranktwo.plot_field(net, m1, m2, xmin, xmax, ymin, ymax, input=input[0, epochs[i]], ax=ax, sizes=sizes,
                                      add_fixed_points=True, fp_load=fp_load[get_input(input, epochs[i])])
        else:
            ranktwo.plot_field(net, m1, m2, xmin, xmax, ymin, ymax, input=input[0, epochs[i]], ax=ax,
                                          sizes=sizes)
        ax.plot(proj1[epochs[i]:epochs[i+1]], proj2[epochs[i]:epochs[i+1]], c='r', lw=4)
        if i > 0:
            ax.plot(proj1[:epochs[i]], proj2[:epochs[i]], c='r', ls='--', lw=4)
        remove_axes(ax)
        if savepath is not None:
            plt.savefig(savepath+'_'+str(i)+'.pdf')


def plot_trajectories_steps_ranktwo(net, rect=None, scalings=True, savepath=None, fp_load=None, sizes=1.,
                                    ax=None):
    """
    Plotting of detailed state-space trajectories for supplementary figures
    """
    stim1_begin = fixation_duration_discrete
    stim1_end = fixation_duration_discrete + max_stimulus1_duration_discrete
    stim2_begin = stim1_end + max_delay_duration_discrete
    stim2_end = stim2_begin + max_stimulus2_duration_discrete
    decision_end = stim2_end + decision_duration_discrete
    epochs = [0, stim1_begin, stim1_end, stim2_begin, stim2_end, decision_end]
    input = torch.zeros(4, decision_end, 2)
    input[(0, 1), stim1_begin:stim1_end, 0] = 1
    input[(0, 2), stim2_begin:stim2_end, 0] = 1
    input[(2, 3), stim1_begin:stim1_end, 1] = 1
    input[(1, 3), stim2_begin:stim2_end, 1] = 1

    if savepath is not None:
        if ax is None:
            for i in range(4):
                plot_trial_epochs(net, input[i].unsqueeze(0), epochs, scalings, rect, savepath+str(i), fp_load, sizes)

        else:
            for i in range(4):
                plot_trial_epochs(net, input[i].unsqueeze(0), epochs, scalings, rect, savepath+str(i), fp_load, sizes,
                                  axes=ax[i])
    else:
        if ax is None:
            for i in range(4):
                plot_trial_epochs(net, input[i].unsqueeze(0), epochs, scalings, rect, None, fp_load, sizes)

        else:
            for i in range(4):
                plot_trial_epochs(net, input[i].unsqueeze(0), epochs, scalings, rect, None, fp_load, sizes,
                                  axes=ax[i])


def plot_trajectories_summary_ranktwo(net, rect=None, plot_wo=False, wo_scale=1., legend=False,
                                      figsize=(8, 8), alt_naming=False, timepoints=False, ax=None):
    """
    Make a plot of all 4 trajectories with flow field
    :param net:
    :param rect: 4-tuple, axis limits
    :param plot_wo: if True add vector wo
    :param wo_scale:
    :param scalings: set to True for networks with rescaled m and ns
    :param legend:
    :param figsize:
    :param alt_naming: set to True for SupportLowRankRNN_withMask
    :return:
    """
    # Prepare 4 inputs
    stim1_begin = fixation_duration_discrete
    stim1_end = fixation_duration_discrete + max_stimulus1_duration_discrete
    stim2_begin = stim1_end + max_delay_duration_discrete
    stim2_end = stim2_begin + max_stimulus2_duration_discrete
    decision_end = stim2_end + decision_duration_discrete
    input = torch.zeros(4, decision_end, 2)
    input[(0, 1), stim1_begin:stim1_end, 0] = 1
    input[(0, 2), stim2_begin:stim2_end, 0] = 1
    input[(2, 3), stim1_begin:stim1_end, 1] = 1
    input[(1, 3), stim2_begin:stim2_end, 1] = 1
    colors = ['orange', 'red', 'cyan', 'hotpink']
    labels = ['A-A', 'A-B', 'B-A', 'B-B']

    m1 = net.m[:, 0].detach().numpy()
    m2 = net.m[:, 1].detach().numpy()
    if alt_naming:
        m1 = net.m_rec[:, 0].detach().numpy()
        m2 = net.m_rec[:, 1].detach().numpy()
    output, trajectories = net.forward(input, return_dynamics=True)
    trajectories = trajectories.squeeze().detach().numpy()
    projections1 = trajectories @ m1 / net.hidden_size
    projections2 = trajectories @ m2 / net.hidden_size

    if rect is None:
        xmin, xmax, ymin, ymax = projections1.min(), projections1.max(), projections2.min(), projections2.max()
    else:
        xmin, xmax, ymin, ymax = rect
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    _, map = ranktwo.plot_field(net, m1, m2, xmin, xmax, ymin, ymax, ax=ax, res=100)
    for i in range(4):
        ax.plot(projections1[i], projections2[i], lw=3, c=colors[i], label=labels[i])
        if timepoints:
            ax.scatter(projections1[i, stim2_begin], projections2[i, stim2_begin], marker='^', s=100, color=colors[i])
            ax.scatter(projections1[i, -1], projections2[i, -1], marker='*', s=100, color=colors[i])

    # Plot the wo vector projection
    if plot_wo:
        wo = net.wo.squeeze().detach().numpy()
        readout1, readout2 = wo @ m1 / net.hidden_size, wo @ m2 / net.hidden_size
        print(readout1)
        print(readout2)
        ax.quiver(0, 0, readout1, readout2, color='yellow', scale=wo_scale)
        ax.text(-0.03, 0.03, "$\\vec w_o$", color='yellow', fontsize='xx-large')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if legend:
        ax.legend(loc='upper right', framealpha=1, frameon=True)
    return map


def psychometric_matrix(net, n_trials=10, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    stim1_begin = fixation_duration_discrete
    stim1_end = fixation_duration_discrete + max_stimulus1_duration_discrete
    stim2_begin = stim1_end + max_delay_duration_discrete
    stim2_end = stim2_begin + max_stimulus2_duration_discrete
    decision_end = stim2_end + decision_duration_discrete
    mean_outputs = np.zeros((2, 2))
    for inp1 in range(2):
        for inp2 in range(2):
            input = torch.zeros(n_trials, decision_end, 2)
            input[:, stim1_begin:stim1_end, inp1] = 1
            input[:, stim2_begin:stim2_end, inp2] = 1
            output = net.forward(input)
            output = output.squeeze().detach().numpy()
            mean_output = output[:, stim2_end:decision_end].mean()
            mean_outputs[inp1, inp2] = mean_output
    image = ax.matshow(mean_outputs, cmap='gray', vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    return image
