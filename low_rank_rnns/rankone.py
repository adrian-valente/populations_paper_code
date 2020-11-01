
def plot_trial_averaged_trajectory(net, input, I, ax, c='blue', alpha=1, rates=False):
    m = net.m[:, 0].detach().numpy()
    I_orth = I - (I @ m) * m / (m @ m)

    output, trajectories = net.forward(input, return_dynamics=True)
    if rates:
        trajectories = net.non_linearity(trajectories)
    trajectories = trajectories.detach().numpy()
    averaged = trajectories.mean(axis=0)
    projection1 = averaged @ m / net.hidden_size
    projection2 = averaged @ I_orth / net.hidden_size

    pl, = ax.plot(projection1, projection2, c=c, alpha=alpha, lw=1)
    return pl
