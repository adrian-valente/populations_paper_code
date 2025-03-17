import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from low_rank_rnns import rdm, romo, raposo, mante, dms


def regression_rdm(net, samples_per_coh=10, cohs=(-1, 0, 1)):
    # Building the inputs
    inputs = rdm.std_default * torch.randn(
        (samples_per_coh * len(cohs), rdm.total_duration, 1)
    )
    for i in range(len(cohs)):
        for j in range(samples_per_coh):
            inputs[
                i * samples_per_coh + j,
                rdm.fixation_duration_discrete : rdm.stimulus_end,
                :,
            ] += cohs[i] * rdm.SCALE

    # Run network
    outp, traj = net(inputs, return_dynamics=True)
    outp = outp.detach().squeeze().numpy()
    traj = traj.detach().numpy()
    rates = np.tanh(traj)

    # Regressing coherence
    coherence_trials = np.array(sum([[coh] * samples_per_coh for coh in cohs], []))
    target1 = np.mean(
        rates[
            :, rdm.fixation_duration_discrete : rdm.fixation_duration_discrete + 2, :
        ],
        axis=1,
    )
    reg1 = LinearRegression(fit_intercept=False).fit(
        coherence_trials.reshape((-1, 1)), target1
    )

    # Regressing choices
    choices = np.sign(np.mean(outp[:, rdm.response_begin :], axis=1))
    target2 = np.mean(rates[:, rdm.response_begin :, :], axis=1)
    reg2 = LinearRegression(fit_intercept=False).fit(choices.reshape((-1, 1)), target1)

    return np.stack([reg1.coef_.squeeze(), reg2.coef_.squeeze()], axis=1)


def regression_romo(net, nsamples=100):
    inputs, f1s, f2s = romo.data_for_regr(nsamples)
    delay_duration = np.floor(500 / romo.deltaT)
    response_begin = int(
        romo.stim1_end + delay_duration + romo.stimulus2_duration_discrete
    )
    response_end = int(response_begin + romo.decision_duration_discrete)

    outp, traj = net(inputs, return_dynamics=True)
    outp = outp.detach().squeeze().numpy()
    rates = np.tanh(traj.detach().numpy())
    decisions = np.sign(outp[:, response_begin:response_end].mean(axis=1))

    # Regression sensory
    predictors = np.stack([f1s, f2s], axis=1)
    target = np.mean(rates[:, response_begin:response_end, :], axis=1)
    reg = LinearRegression().fit(predictors, target)
    target_hat = reg.predict(predictors)
    # r2 = r2_score(target, target_hat)
    # cond = np.linalg.cond(predictors)
    reg_space = reg.coef_

    # # Regression decision
    # predictors = decisions.reshape((-1, 1))
    # reg = LinearRegression().fit(predictors, target)
    # target_hat = reg.predict(predictors)
    # # r2 = r2_score(target, target_hat)
    # # cond = np.linalg.cond(predictors)
    # reg_space = np.concatenate([reg_space, reg.coef_], axis=1)

    return reg_space


def regression_raposo(net, cohs=(-4, -2, -1, 1, 2, 4)):
    lc2 = len(cohs) ** 2
    inputs = torch.zeros((3 * lc2, raposo.total_duration, 4))

    cohs1 = []
    cohs2 = []
    for k, ctx in enumerate((-1, 0, 1)):
        for i, coh1 in enumerate(cohs):
            for j, coh2 in enumerate(cohs):
                trial_idx = k * lc2 + i * len(cohs) + j
                if ctx in (1, 0):
                    inputs[trial_idx, raposo.stim_begin : raposo.stim_end, 0] = (
                        coh1 * raposo.SCALE
                    )
                    inputs[trial_idx, raposo.stim_begin : raposo.stim_end, 2] = (
                        raposo.SCALE_CTX
                    )
                    cohs1.append(coh1)
                else:
                    cohs1.append(0)
                if ctx in (-1, 0):
                    inputs[trial_idx, raposo.stim_begin : raposo.stim_end, 1] = (
                        coh2 * raposo.SCALE
                    )
                    inputs[trial_idx, raposo.stim_begin : raposo.stim_end, 3] = (
                        raposo.SCALE_CTX
                    )
                    cohs2.append(coh2)
                else:
                    cohs2.append(0)
    ctx1 = [0] * lc2 + [1] * (2 * lc2)
    ctx2 = [1] * (2 * lc2) + [0] * lc2

    output, trajectories = net.forward(inputs, return_dynamics=True)
    trajectories = trajectories.detach().numpy()
    rates = np.tanh(trajectories)

    target = np.mean(rates[:, raposo.stim_begin : raposo.stim_begin + 10, :], axis=1)
    predictors = np.stack(
        [np.array(cohs1), np.array(cohs2), np.array(ctx1), np.array(ctx2)]
    ).T
    reg_sens = LinearRegression(fit_intercept=False).fit(predictors, target)
    target_hat = reg_sens.predict(predictors)
    r2 = r2_score(target, target_hat)

    reg_full = reg_sens.coef_
    reg_full = (reg_full - reg_full.mean(axis=0)) / reg_full.std(axis=0)
    return reg_full, r2


def regression_mante(net):
    cohs = [-4, -2, 2, 4]
    n_inputs = 3 * len(cohs) ** 2
    inputs = torch.zeros(n_inputs, mante.total_duration, 4)
    predictors = np.zeros((n_inputs, 4), dtype=np.float)

    # Generate all inputs
    trial_idx = 0
    for ctx in (-1, 0, 1):
        for i, coh1 in enumerate(cohs):
            for j, coh2 in enumerate(cohs):
                inputs[trial_idx, mante.stim_begin : mante.stim_end, 0] += (
                    coh1 * mante.SCALE
                )
                inputs[trial_idx, mante.stim_begin : mante.stim_end, 1] += (
                    coh2 * mante.SCALE
                )
                if ctx == 1:
                    inputs[trial_idx, mante.fixation_duration_discrete :, 2] = (
                        1 * mante.SCALE_CTX
                    )
                    predictors[trial_idx, 2] = 1
                elif ctx == -1:
                    inputs[trial_idx, mante.fixation_duration_discrete :, 3] = (
                        1 * mante.SCALE_CTX
                    )
                    predictors[trial_idx, 3] = 1
                predictors[trial_idx, 0] = coh1
                predictors[trial_idx, 1] = coh2
                trial_idx += 1

    output, trajectories = net.forward(inputs, return_dynamics=True)
    trajectories = trajectories.detach().numpy()
    rates = np.tanh(trajectories)

    target = np.mean(rates[:, mante.stim_begin : mante.stim_end, :], axis=1)
    reg = LinearRegression().fit(predictors, target)
    target_hat = reg.predict(predictors)
    r2 = r2_score(target, target_hat)

    reg_full = reg.coef_
    reg_full = (reg_full - reg_full.mean(axis=0)) / reg_full.std(axis=0)
    return reg_full, r2


def regression_dms(net, n_trials=100):
    x = dms.std_default * torch.randn(n_trials, dms.total_duration, 2)
    predictors = np.zeros((n_trials, 2))

    delay_duration = (dms.delay_duration_min + dms.delay_duration_max) / 2
    delay_duration_discrete = int(np.floor(delay_duration / dms.deltaT))
    stimulus1_duration = (dms.stimulus1_duration_min + dms.stimulus1_duration_max) / 2
    stimulus1_duration_discrete = int(np.floor(stimulus1_duration / dms.deltaT))
    stimulus2_duration = (dms.stimulus2_duration_min + dms.stimulus2_duration_max) / 2
    stimulus2_duration_discrete = int(np.floor(stimulus2_duration / dms.deltaT))
    stim1_begin = dms.fixation_duration_discrete
    stim1_end = stim1_begin + stimulus1_duration_discrete
    stim2_begin = stim1_end + delay_duration_discrete
    stim2_end = stim2_begin + stimulus2_duration_discrete
    decision_end = stim2_end + dms.decision_duration_discrete

    types = ["A-A", "A-B", "B-A", "B-B"]
    for i in range(n_trials):
        cur_type = types[int(np.random.rand() * 4)]

        if cur_type == "A-A":
            input1 = 1
            input2 = 1
        elif cur_type == "A-B":
            input1 = 1
            input2 = 0
        elif cur_type == "B-A":
            input1 = 0
            input2 = 1
        elif cur_type == "B-B":
            input1 = 0
            input2 = 0

        x[i, stim1_begin:stim1_end, 0] += input1
        x[i, stim1_begin:stim1_end, 1] += 1 - input1
        predictors[i, 0] = input1
        x[i, stim2_begin:stim2_end, 0] += input2
        x[i, stim2_begin:stim2_end, 1] += 1 - input2
        predictors[i, 1] = input2

    output, trajectories = net.forward(x, return_dynamics=True)
    output = output.detach().squeeze().numpy()
    trajectories = trajectories.detach().numpy()
    rates = np.tanh(trajectories)

    target = np.mean(rates[:, stim2_end:decision_end, :], axis=1)
    reg = LinearRegression().fit(predictors, target)
    target_hat = reg.predict(predictors)
    r2 = r2_score(target, target_hat)

    return reg.coef_, r2
