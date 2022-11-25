"""
Functions to compute regressions on the Ctx task, mostly for figure 6.
"""

import sys
sys.path.append('../')
from low_rank_rnns import mante
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def compute_sensory_regressors(net, epoch_start, epoch_end):
    regs = []
    for context in (1, 2):
        coh1_vect = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
        coh2_vect = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
        inputs_array = torch.zeros((1, mante.total_duration, 4))
        inputs_array[0] = mante.generate_specified_input(0, 0, context, 1, 0.)

        inputs_array = []
        color_vector = np.array([])
        motion_vector = np.array([])
        for i in range(len(coh1_vect)):
            for j in range(len(coh2_vect)):
                coh1 = coh1_vect[i]
                coh2 = coh2_vect[j]
                input_trial = mante.generate_specified_input(coh1, coh2, context, 1, 0.)
                inputs_array.append(input_trial)
                color_vector = np.append(color_vector, coh1)
                motion_vector = np.append(motion_vector, coh2)

        inputs_array = torch.cat(inputs_array, dim=0)
        print(inputs_array.shape)
        output, trajectories = net.forward(inputs_array, return_dynamics=True)
        trajectories = torch.tanh(trajectories)
        stimulation_epoch = np.arange(epoch_start, epoch_end)
        variables_to_be_regressed = torch.mean(trajectories[:, stimulation_epoch, :], dim=1)
        variables_to_be_regressed = variables_to_be_regressed.detach().numpy()
        X = np.concatenate((color_vector.reshape(len(color_vector), 1), motion_vector.reshape(len(color_vector), 1)), axis=1)
        print(X.shape)
        reg = LinearRegression().fit(X, variables_to_be_regressed)
        regs.append(reg.coef_)
    return regs


def compute_sensory_regressors_mixed_ctx(net,epoch_start,epoch_end,data_type='mante_expe'):
    context = 1
    coh1_vect = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
    coh2_vect = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=context,std=0)

    inputs_array = 0*input_trial
    color_vector = np.array([])
    motion_vector = np.array([])
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)

            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            color_vector = np.append(color_vector,coh1)
            motion_vector = np.append(motion_vector,coh2)
    
    context = 2
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            color_vector = np.append(color_vector,coh1)
            motion_vector = np.append(motion_vector,coh2)
    
            
    inputs_array = inputs_array[1:,:,:]  
    inputs_array = torch.from_numpy(inputs_array)
    output,trajectories= net.forward(inputs_array,return_dynamics=True)
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    trajectories = torch.tanh(trajectories)
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed1 = variables_to_be_regressed.detach().numpy()
    
    X = np.concatenate((color_vector.reshape(len(color_vector),1),motion_vector.reshape(len(color_vector),1)),axis=1)
    print(X.shape)
    print(variables_to_be_regressed1.shape)
    reg1 = LinearRegression().fit(X,variables_to_be_regressed1)
    r = reg1.coef_    
    return r


def compute_sensory_context_regressors(net,epoch_start,epoch_end,data_type='mante_expe'):
    context = 1
    coh1_vect = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
    coh2_vect = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=context,std=0)

    inputs_array = 0*input_trial
    color_vector = np.array([])
    motion_vector = np.array([])
    context_vector = np.array([])
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)

            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            color_vector = np.append(color_vector,coh1)
            motion_vector = np.append(motion_vector,coh2)
            context_vector = np.append(context_vector,+1)
    
    context = 2
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            color_vector = np.append(color_vector,coh1)
            motion_vector = np.append(motion_vector,coh2)
            context_vector = np.append(context_vector,-1)
            
    inputs_array = inputs_array[1:,:,:]  
    inputs_array = torch.from_numpy(inputs_array)
    output,trajectories= net.forward(inputs_array,return_dynamics=True)
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    trajectories = torch.tanh(trajectories)
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed1 = variables_to_be_regressed.detach().numpy()
    
    X = np.concatenate((color_vector.reshape(len(color_vector),1),motion_vector.reshape(len(color_vector),1),context_vector.reshape(len(context_vector),1)),axis=1)
    print(X.shape)
    reg1 = LinearRegression().fit(X,variables_to_be_regressed1)
    r = reg1.coef_
    return r


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



def compute_choice_regressors(net,epoch_start,epoch_end,ctx1=1,ctx2=2,data_type='mante_expe',nb_regressors = 1):
    
    # NB: we here assume that the network performs perfectly (choice is not the actual choice of the net but the correct choice)
    context = ctx1
    coh1_vect = np.array([-16, 16])
    coh2_vect = np.array([-16, 16])
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=context,std=0)
    inputs_array = 0*input_trial
    choice_vector = np.array([])
    choice_vector1 = np.array([])
    choice_vector2 = np.array([])
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            if nb_regressors == 1:
                if context == +1:
                    choice_vector = np.append(choice_vector,np.sign(coh1))
                else:
                    choice_vector = np.append(choice_vector,np.sign(coh2))
            elif nb_regressors == 2:  
                if context == +1:
                    choice_vector1 = np.append(choice_vector1,np.sign(coh1))
                    choice_vector2 = np.append(choice_vector2,0)
                else:
                    choice_vector1 = np.append(choice_vector1,0)
                    choice_vector2 = np.append(choice_vector2,np.sign(coh2))

    context = ctx2
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            
            if nb_regressors == 1:
                if context == +1:
                    choice_vector = np.append(choice_vector,np.sign(coh1))
                else:
                    choice_vector = np.append(choice_vector,np.sign(coh2))
            elif nb_regressors == 2:  
                if context == +1:
                    choice_vector1 = np.append(choice_vector1,np.sign(coh1))
                    choice_vector2 = np.append(choice_vector2,0)
                else:
                    choice_vector1 = np.append(choice_vector1,0)
                    choice_vector2 = np.append(choice_vector2,np.sign(coh2))
            
    inputs_array = inputs_array[1:,:,:]
    inputs_array = torch.from_numpy(inputs_array)
    output, trajectories= net.forward(inputs_array,return_dynamics=True)
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    trajectories = torch.tanh(trajectories)
    
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed = variables_to_be_regressed.detach().numpy()
    
    if nb_regressors == 1:
        X = choice_vector.reshape(len(choice_vector),1)
    elif nb_regressors == 2:
        X = np.concatenate( (choice_vector1.reshape(len(choice_vector1),1),choice_vector2.reshape(len(choice_vector2),1)),axis=1)
        
    reg = LinearRegression().fit(X,variables_to_be_regressed)
    r = reg.coef_
    return r


def compute_choice_regressors_split_context(net,epoch_start,epoch_end,ctx1=1,ctx2=2,data_type='mante_expe',nb_regressors = 1):
    
    # NB: we here assume that the network performs perfectly (choice is not the actual choice of the net but the correct choice)
    context = 1
    coh1_vect = np.array([-16, 16])
    coh2_vect = np.array([-16, 16])
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=context,std=0)
    inputs_array = 0*input_trial
    choice_vector = np.array([])
    choice_vector1 = np.array([])
    choice_vector2 = np.array([])
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
                    
    inputs_array = inputs_array[1:,:,:]
    inputs_array = torch.from_numpy(inputs_array)
    output, trajectories = net.forward(inputs_array,return_dynamics=True)
    choice_vector = np.sign(output[:,-1,0].detach().numpy())
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    trajectories = torch.tanh(trajectories)
    
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed = variables_to_be_regressed.detach().numpy()
    
    X = choice_vector.reshape(len(choice_vector),1)
    
    reg = LinearRegression().fit(X,variables_to_be_regressed)
    r1 = reg.coef_                    
        

    context = 2
    
    inputs_array = 0*input_trial
    choice_vector = np.array([])
    choice_vector1 = np.array([])
    choice_vector2 = np.array([])
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            
    inputs_array = inputs_array[1:,:,:]
    inputs_array = torch.from_numpy(inputs_array)
    output, trajectories= net.forward(inputs_array,return_dynamics=True)
    choice_vector = np.sign(output[:,-1,0].detach().numpy())
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    trajectories = torch.tanh(trajectories)
    
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed = variables_to_be_regressed.detach().numpy()
    
    X = choice_vector.reshape(len(choice_vector),1)

    reg = LinearRegression().fit(X,variables_to_be_regressed)
    r2 = reg.coef_
    return r1, r2










def compute_choice_sensory_regressors_split_ctx(net,epoch_start,epoch_end,data_type='mante_expe'):
    context = +1
    coh1_vect = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    coh2_vect = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=context,std=0)
    inputs_array = 0*input_trial
    color_vector = np.array([])
    motion_vector = np.array([])
    choice_vector = np.array([])
    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            color_vector = np.append(color_vector,coh1)
            motion_vector = np.append(motion_vector,coh2)
            if context == +1:
                choice_vector = np.append(choice_vector,np.sign(coh1))
            else:
                choice_vector = np.append(choice_vector,np.sign(coh2))

    inputs_array = inputs_array[1:,:,:]  
    inputs_array = torch.from_numpy(inputs_array)
    output,trajectories= net.forward(inputs_array,return_dynamics=True)
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    trajectories = torch.tanh(trajectories)
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed = variables_to_be_regressed.detach().numpy()

    X = np.concatenate( (color_vector.reshape(len(color_vector),1),motion_vector.reshape(len(color_vector),1),choice_vector.reshape(len(choice_vector),1)),axis=1)
    print(X.shape)
    reg = LinearRegression().fit(X,variables_to_be_regressed)
    r1 = reg.coef_
    
    
    
    
    context = 2
    input_trial, y_train, mask_train,epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
    inputs_array = 0*input_trial
    color_vector = np.array([])
    motion_vector = np.array([])
    choice_vector = np.array([])
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            inputs_array = np.append(inputs_array,input_trial,axis=0)
            color_vector = np.append(color_vector,coh1)
            motion_vector = np.append(motion_vector,coh2)
            if context == +1:
                choice_vector = np.append(choice_vector,np.sign(coh1))
            else:
                choice_vector = np.append(choice_vector,np.sign(coh2))

            
    inputs_array = inputs_array[1:,:,:]  
    inputs_array = torch.from_numpy(inputs_array)
    output,trajectories= net.forward(inputs_array,return_dynamics=True)
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    trajectories = torch.tanh(trajectories)
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed = variables_to_be_regressed.detach().numpy()
    
    X = np.concatenate( (color_vector.reshape(len(color_vector),1),motion_vector.reshape(len(color_vector),1),choice_vector.reshape(len(choice_vector),1)),axis=1)
    print(X.shape)
    reg = LinearRegression().fit(X,variables_to_be_regressed)
    r2 = reg.coef_
    return r1,r2


def compute_ctx_regressors(net, epoch_start, epoch_end, rates=True):
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=+1,std=0)
    inputs_array = 0*input_trial

    context = +1
    coh1 = 0
    coh2 = 0
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
    input_trial = input_trial.numpy()
    inputs_array = np.append(inputs_array,input_trial,axis=0)
    

    context = 2
    coh1 = 0
    coh2 = 0
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
    input_trial = input_trial.numpy()    
    inputs_array = np.append(inputs_array,input_trial,axis=0)
    
    
    context = 0
    coh1 = 0
    coh2 = 0
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
    
    input_trial = input_trial.numpy()
    inputs_array = np.append(inputs_array,input_trial,axis=0)
    
    
    context1_vector = np.array([1,0,0])
    context2_vector = np.array([0,1,0])
    
    
    inputs_array = inputs_array[1:,:,:]  
    inputs_array = torch.from_numpy(inputs_array)

    output,trajectories= net.forward(inputs_array,return_dynamics=True)
    if rates:
        trajectories = torch.tanh(trajectories)
    stimulation_epoch = np.arange(epoch_start,epoch_end)
    variables_to_be_regressed = torch.mean(trajectories[:,stimulation_epoch,:],dim=1)
    variables_to_be_regressed1 = variables_to_be_regressed.detach().numpy()
    
    X = np.concatenate((context1_vector.reshape(len(context1_vector),1),context2_vector.reshape(len(context2_vector),1)),axis=1).T
    X = X.T
    reg = LinearRegression().fit(X,variables_to_be_regressed1)
    r = reg.coef_
#    plt.figure(1);plt.plot(r[:,0],r[:,1],'*')
    return r


def compute_single_ctx_regressors(net, epoch_start, epoch_end):
    """
    Compute regressors towards context, during the context-cue-only epoch.

    """
    inputs = torch.zeros((2, mante.total_duration, 4))
    input_trial = mante.generate_specified_input(0, 0, 1, 1, 0.)
    inputs[0] = input_trial
    input_trial = mante.generate_specified_input(0, 0, 2, 1, 0.)
    inputs[1] = input_trial
    context_vector = np.array([1, -1])

    output, trajectories = net.forward(inputs, return_dynamics=True)
    trajectories = torch.tanh(trajectories)
    stimulation_epoch = np.arange(epoch_start, epoch_end)
    variables_to_be_regressed = torch.mean(trajectories[:, stimulation_epoch, :], dim=1)
    variables_to_be_regressed1 = np.abs(variables_to_be_regressed.detach().numpy())
    X = context_vector.reshape(len(context_vector), 1)

    print(X.shape, variables_to_be_regressed1.shape)
    reg = LinearRegression(fit_intercept=True).fit(X, variables_to_be_regressed1)
    r = reg.coef_
    print(r.shape)
    return r


def plot_tuning_curves(net, ind_neuron, epoch_start, epoch_end, add_linreg=False, col1='m', col2='c'):
    """
    For a given neuron, compute and plot its sensitivy to stimulus coherences.
    :param net: network
    :param ind_neuron: int, a given neuron
    :param epoch_start: int
    :param epoch_end: int
    :param add_linreg: bool
    :return: 4-tuple of ndarrays, selectivity to signals A and B in both contexts
    """
    coh_vect = np.arange(0., 11., 2)
    print(coh_vect)
    color_selectivity_ctx1 = np.zeros_like(coh_vect)
    motion_selectivity_ctx1 = np.zeros_like(coh_vect)
    color_selectivity_ctx2 = np.zeros_like(coh_vect)
    motion_selectivity_ctx2 = np.zeros_like(coh_vect)

    for i in range(len(coh_vect)):
        coh1 = coh_vect[i]
        coh2 = 0
        context = 1
        input1 = mante.generate_specified_input(coh1, coh2, context, 1, 0.)
        z, x = net.forward(input1, return_dynamics=True)
        ra = float(np.mean(np.tanh(x[0, epoch_start:epoch_end, ind_neuron].detach().numpy())))
        color_selectivity_ctx1[i] = ra
        
        coh1 = coh_vect[i]
        coh2 = 0
        context = 2
        input1 = mante.generate_specified_input(coh1, coh2, context, 1, 0.)
        z, x = net.forward(input1, return_dynamics=True)
        ra = float(np.mean(np.tanh(x[0, epoch_start:epoch_end, ind_neuron].detach().numpy())))
        color_selectivity_ctx2[i] = ra
        
        coh1 = 0
        coh2 = coh_vect[i]
        context = 1
        input1 = mante.generate_specified_input(coh1, coh2, context, 1, 0.)
        z, x = net.forward(input1, return_dynamics=True)
        ra = float(np.mean(np.tanh(x[0, epoch_start:epoch_end, ind_neuron].detach().numpy())))
        motion_selectivity_ctx1[i] = ra
        
        coh1 = 0
        coh2 = coh_vect[i]
        context = 2
        input1 = mante.generate_specified_input(coh1, coh2, context, 1, 0.)
        z, x = net.forward(input1, return_dynamics=True)
        ra = float(np.mean(np.tanh(x[0, epoch_start:epoch_end, ind_neuron].detach().numpy())))
        motion_selectivity_ctx2[i] = ra
        
    plt.figure(0)
    plt.plot(coh_vect, (color_selectivity_ctx1 - color_selectivity_ctx1[0]) * np.sign(color_selectivity_ctx1[-1] - color_selectivity_ctx1[0]), 's-', c=col1)
    plt.plot(coh_vect, (color_selectivity_ctx2 - color_selectivity_ctx2[0]) * np.sign(color_selectivity_ctx2[-1] - color_selectivity_ctx2[0]), 's-', c=col2)
    plt.figure(1)
    plt.plot(coh_vect, (motion_selectivity_ctx1 - motion_selectivity_ctx1[0]) * np.sign(motion_selectivity_ctx1[-1] - motion_selectivity_ctx1[0]), 's-', c=col1)
    plt.plot(coh_vect, (motion_selectivity_ctx2 - motion_selectivity_ctx2[0]) * np.sign(motion_selectivity_ctx2[-1] - motion_selectivity_ctx2[0]), 's-', c=col2)

    if add_linreg:
        vtbr = (color_selectivity_ctx1 - color_selectivity_ctx1[0]) * np.sign(color_selectivity_ctx1[-1] - color_selectivity_ctx1[0])
        X = coh_vect.reshape((len(coh_vect), 1))
        vtbr_reshaped = vtbr.reshape(len(vtbr), 1)
        reg = LinearRegression().fit(X, vtbr_reshaped)
        r = reg.coef_
        plt.figure(0)
        plt.plot(coh_vect, coh_vect * r[0], c=col1, ls='--')

        vtbr = (color_selectivity_ctx2 - color_selectivity_ctx2[0]) * np.sign(color_selectivity_ctx2[-1] - color_selectivity_ctx2[0])
        X = coh_vect.reshape((len(coh_vect), 1))
        vtbr_reshaped = vtbr.reshape(len(vtbr), 1)
        reg = LinearRegression().fit(X, vtbr_reshaped)
        r = reg.coef_
        plt.plot(coh_vect, coh_vect * r[0], c=col2, ls='--')

        vtbr = (motion_selectivity_ctx1 - motion_selectivity_ctx1[0]) * np.sign(motion_selectivity_ctx1[-1] - motion_selectivity_ctx1[0])
        X = coh_vect.reshape((len(coh_vect), 1))
        vtbr_reshaped = vtbr.reshape(len(vtbr), 1)
        reg = LinearRegression().fit(X, vtbr_reshaped)
        r = reg.coef_
        plt.figure(1)
        plt.plot(coh_vect, coh_vect*r[0], c=col1, ls='--')

        vtbr = (motion_selectivity_ctx2 - motion_selectivity_ctx2[0]) * np.sign(motion_selectivity_ctx2[-1] - motion_selectivity_ctx2[0])
        X = coh_vect.reshape((len(coh_vect),1))
        vtbr_reshaped = vtbr.reshape(len(vtbr),1)
        reg = LinearRegression().fit(X, vtbr_reshaped)
        r = reg.coef_
        plt.plot(coh_vect, coh_vect*r[0], c=col2, ls='--')

    return color_selectivity_ctx1, motion_selectivity_ctx1, color_selectivity_ctx2, motion_selectivity_ctx2




def compute_FTV(net):
    
    context = +1
    coh1_vect = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    coh2_vect = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=context,std=0)
    input_ctx_plus_one = 0*input_trial    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            input_ctx_plus_one = np.append(input_ctx_plus_one,input_trial,axis=0)
    
    context = +2
    coh1_vect = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    coh2_vect = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
    input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=0, coh_motion_spec=0, context_spec=context,std=0)
    input_ctx_minus_one = 0*input_trial    
    for i in range(len(coh1_vect)):
        for j in range(len(coh2_vect)):
            coh1 = coh1_vect[i]
            coh2 = coh2_vect[j]
            input_trial, y_train, mask_train, epochs = mante.generate_mante_data(1,coh_color_spec=coh1, coh_motion_spec=coh2, context_spec=context,std=0)
            input_trial = input_trial.numpy()
            input_ctx_minus_one = np.append(input_ctx_minus_one,input_trial,axis=0)

    z,x = net.forward(torch.from_numpy(input_ctx_plus_one),return_dynamics = True)
    r_array = np.tanh(x[:,epochs[1][0]:,:].detach().numpy())
    r_av_over_trials = np.mean(r_array,axis=0)
    TV_A_vect = np.mean(np.mean((r_array-r_av_over_trials)**2,axis=0),axis=0)
    
    z,x = net.forward(torch.from_numpy(input_ctx_minus_one),return_dynamics = True)
    r_array = np.tanh(x[:,epochs[1][0]:,:].detach().numpy())
    r_av_over_trials = np.mean(r_array,axis=0)
    TV_B_vect = np.mean(np.mean((r_array-r_av_over_trials)**2,axis=0),axis=0)

    FTV = (TV_A_vect - TV_B_vect)/(TV_A_vect + TV_B_vect)
    
    
#    ind_out = np.where()
    
    return FTV, TV_A_vect, TV_B_vect
    
    
    
    