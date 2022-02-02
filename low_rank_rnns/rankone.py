import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


def plot_trajectory(net, input, I, c='blue', ax=None):
    m = net.m[:, 0].detach().numpy() * sqrt(net.hidden_size)
    I_orth = I - (I @ m) * m / (m @ m)

    if ax is None:
        fig, ax = plt.subplots()
    output, trajectories = net.forward(input.unsqueeze(0), return_dynamics=True)
    trajectories = trajectories[0].detach().numpy()
    projections1 = trajectories @ m / net.hidden_size
    projections2 = trajectories @ I_orth / net.hidden_size
    ax.plot(projections1, projections2, c=c)
    ax.scatter(projections1[0], projections2[0], marker='s', c='red', s=50)
    ax.scatter(projections1[-1], projections2[-1], marker='*', c='orange', s=200)
    ax.set_xlabel("$\kappa(t)$")
    ax.set_ylabel("$\mathbf{x}^T\mathbf{I}_\perp$")
    return ax


def plot_trial_averaged_trajectory(net, input, I, ax, c='blue', alpha=1, ymin=-3, ymax=3, rates=False):
    m = net.m[:, 0].detach().numpy() * sqrt(net.hidden_size)
    I_orth = I - (I @ m) * m / (m @ m)

    output, trajectories = net.forward(input, return_dynamics=True)
    if rates:
        trajectories = net.non_linearity(trajectories)
    trajectories = trajectories.detach().numpy()
    averaged = trajectories.mean(axis=0)
    projection1 = averaged @ m / net.hidden_size
    projection2 = averaged @ I_orth / net.hidden_size

    pl, = ax.plot(projection1, projection2, c=c, alpha=alpha, lw=1)
    #ax.scatter(projection1[-1], projection2[-1], marker='*', c='k', s=50)
    return pl

def plot_trial_averaged_trajectory_reconstructed(net, input, I, c='blue', alpha=1, label='', ymin=-3, ymax=3, rates=False, lw=3,
                                   ax=None):
    m = net.m_rec[:, 0].detach().numpy() * sqrt(net.hidden_size)
    I_orth = I - (I @ m) * m / (m @ m)

    output, trajectories = net.forward(input, return_dynamics=True)
    if rates:
        trajectories = net.non_linearity(trajectories)
    trajectories = trajectories.detach().numpy()
    averaged = trajectories.mean(axis=0)
    projection1 = averaged @ m / net.hidden_size
    projection2 = averaged @ I_orth / net.hidden_size

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(projection1, projection2, c=c, alpha=alpha, label=label, lw=lw)
    ax.scatter(projection1[0], projection2[0], marker='s', c='red', s=50)
    ax.scatter(projection1[-1], projection2[-1], marker='*', c='orange', s=200)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("$\kappa(t)$")
    ax.set_ylabel("$\mathbf{x}^T\mathbf{I}_\perp$")
    return ax
