import sys
sys.path.append('../')
from low_rank_rnns import mante
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def compute_choice_sensory_regressors(net, rates=True):
    cohs = [-16, -8, -4, -2, 2, 4, 8, 16]
    n_inputs = 2 * len(cohs) ** 2
    inputs = torch.empty(n_inputs, mante.total_duration, 4)
    predictors_sens = np.zeros((n_inputs, 2), dtype=np.float)

    # Generate all inputs
    for ctx in (0, 1):
        for i, coh1 in enumerate(cohs):
            for j, coh2 in enumerate(cohs):
                x, y, mask, epochs = mante.generate_mante_data(1, std=0, fraction_validation_trials=0,
                                                               coh_color_spec=coh1, coh_motion_spec=coh2,
                                                               context_spec=ctx + 1)
                k = ctx * len(cohs) ** 2 + i * len(cohs) + j
                inputs[k] = x[0]
                predictors_sens[k, 0] = coh1
                predictors_sens[k, 1] = coh2

    output, trajectories = net.forward(inputs, return_dynamics=True)
    output = output.detach().squeeze().numpy()
    trajectories = trajectories.detach().numpy()
    if rates:
        trajectories = np.tanh(trajectories)

    # choice
    predictors_choice = np.sign(np.mean(output[:, mante.response_begin:], axis=1)).reshape((-1, 1))
    target = np.mean(trajectories[:, mante.response_begin:, :], axis=1)
    reg_choice = LinearRegression().fit(predictors_choice, target)
    target_hat = reg_choice.predict(predictors_choice)
    print(f'choice R2={r2_score(target, target_hat)}')

    # Regress the rest
    target = np.mean(trajectories[:, mante.stim_begin:mante.stim_end, :], axis=1)
    reg = LinearRegression().fit(predictors_sens, target)
    target_hat = reg.predict(predictors_sens)
    print(f'sensory R2={r2_score(target, target_hat)}')
    return np.hstack([reg.coef_, reg_choice.coef_])


def compute_ctx_regressors(net, epoch_start, epoch_end, rates=True):
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1, coh_color_spec=0, coh_motion_spec=0,
                                                                         context_spec=+1, std=0)
    inputs_array = 0*input_trial

    context = +1
    coh1 = 0
    coh2 = 0
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2,
                                                                         context_spec=context, std=0)
    input_trial = input_trial.numpy()
    inputs_array = np.append(inputs_array, input_trial, axis=0)
    
    context = 2
    coh1 = 0
    coh2 = 0
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2,
                                                                         context_spec=context, std=0)
    input_trial = input_trial.numpy()    
    inputs_array = np.append(inputs_array, input_trial, axis=0)
    
    context = 0
    coh1 = 0
    coh2 = 0
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1, coh_color_spec=coh1,
                                                                         coh_motion_spec=coh2,
                                                                         context_spec=context, std=0)
    
    input_trial = input_trial.numpy()
    inputs_array = np.append(inputs_array, input_trial, axis=0)

    context1_vector = np.array([1, 0, 0])
    context2_vector = np.array([0, 1, 0])

    inputs_array = inputs_array[1:, :, :]
    inputs_array = torch.from_numpy(inputs_array)

    output, trajectories = net.forward(inputs_array, return_dynamics=True)
    if rates:
        trajectories = torch.tanh(trajectories)
    stimulation_epoch = np.arange(epoch_start, epoch_end)
    variables_to_be_regressed = torch.mean(trajectories[:, stimulation_epoch, :], dim=1)
    variables_to_be_regressed1 = variables_to_be_regressed.detach().numpy()
    
    X = np.concatenate((context1_vector.reshape(len(context1_vector), 1),
                        context2_vector.reshape(len(context2_vector), 1)), axis=1).T
    X = X.T
    reg = LinearRegression().fit(X, variables_to_be_regressed1)
    r = reg.coef_
    return r
