# In this file we aim to save and visualize how thew weights of the agent dynamically change during the training process.
# First, we need to import from trainingUtils the stuff we need to perform a training

from trainingUtils import train_episode, OutOfDistributionTraining
from agent import LinearAgent, animate_weights
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from reservoirTrainingUtils import TrainingToInference


def test1():
    agent = LinearAgent()
    env = Environment()
    weights_over_time = []

    episodes = 600
    time_steps = 30

    env.reset(126)

    for episode in range(episodes):
        reward, traj = train_episode(agent, env, time_steps)
        weights_over_time.append(agent.weights.copy())
        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {reward}")

    weights_over_time = np.array(weights_over_time)  # (episodes, A, D)

    anim, fig = animate_weights(weights_over_time, interval=10, frame_skip=5)

    plt.show()  
    
    # anim, fig = animate_weights(weights_over_time, interval=120, save_path="weights.mp4")
    # anim, fig = animate_weights(weights_over_time, interval=120, save_path="weights.gif")

import numpy as np
import matplotlib.pyplot as plt

def pca_fit_all_paths(weights, n_components=2):
    """
    Fit PCA on all paths/episodes jointly and project.

    Parameters
    ----------
    weights : ndarray
        Shape (P, E, A, D) where:
          P = number of paths (rounds * positions, e.g., rounds*16)
          E = episodes per path
          A, D = agent weights shape (e.g., 4, 25)
    n_components : int
        PCA output dimensions

    Returns
    -------
    scores : ndarray
        Projected coordinates with shape (P, E, n_components)
    pca : PCA-like object
        The fitted PCA object (your PCA class or sklearn's)
    """
    P, E, A, D = weights.shape
    F = A * D
    X = weights.reshape(P * E, F)  
    pca = PCA(n_components=n_components, svd_solver='full')
    scores_all = pca.fit_transform(X) 
    scores = scores_all.reshape(P, E, n_components)
    return scores, pca

def compute_alpha_arrays(rewards, bin_size=15):
    """
    Compute alpha arrays for `plot_pca_trajectories` based on binned rewards for multiple paths.

    Parameters
    ----------
    rewards : ndarray
        Shape (P, E) where P is the number of paths (e.g., 16) and E is the number of episodes.
    bin_size : int
        The size of the bins to group rewards. Default is 15.

    Returns
    -------
    alpha_arrays : ndarray
        Shape (P, E) array of alpha values normalized between 0.05 and 1.0 for each path.
    """
    P, E = rewards.shape
    num_bins = (E + bin_size - 1) // bin_size 
    padded_rewards = np.zeros((P, num_bins * bin_size))  
    padded_rewards[:, -E:] = rewards  

    binned_rewards = padded_rewards.reshape(P, num_bins, bin_size).mean(axis=2)

    min_rewards = binned_rewards.min(axis=1, keepdims=True)
    max_rewards = binned_rewards.max(axis=1, keepdims=True)
    normalized_rewards = (binned_rewards - min_rewards) / (max_rewards - min_rewards)
    alpha_arrays = 0.05 + 0.75 * normalized_rewards  
    alpha_arrays = alpha_arrays.repeat(bin_size, axis=1)[:, -E:] 
    return alpha_arrays

def plot_pca_trajectories(scores, labels=None, alpha_arrays=None,
                          title="PCA of Agent Weights (All Positions)"):
    """
    Plot PCA trajectories per path with unique colors and increasing (or custom) alpha along time.

    Parameters
    ----------
    scores : ndarray
        Shape (P, E, 2) PCA scores for each path (P) across episodes (E).
    labels : list[str] | None
        Optional list of labels per path (length P). If None, uses 'Path i'.
    alpha_array : ndarray | None
        Shape (E,) array of alphas to use along time. If None, defaults to linspace(0.25, 1.0, E).
    title : str
        Plot title.
    """
    P, E, D = scores.shape
    assert D == 2, "scores must be 2D per point (PC1, PC2)"

    if P <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, P))
    elif P <= 32:
        base_colors = plt.cm.tab20b(np.linspace(0, 1, min(P, 16)))
        base_colors = np.vstack([base_colors, plt.cm.tab20c(np.linspace(0, 1, P - base_colors.shape[0]))])
    else:
        base_colors = plt.cm.hsv(np.linspace(0, 1, P, endpoint=False))

    if labels is None:
        labels = [f"Path {i}" for i in range(P)]

    if alpha_arrays is None:
        alpha_arrays = [np.linspace(0.05, 1.0, E)] * P

    plt.figure(figsize=(9.5, 7.5))
    for i in range(P):
        path = scores[i]
        color = base_colors[i % len(base_colors)]

        plt.plot(path[:, 0], path[:, 1], linestyle='-', color=color, alpha=0.25,
                 linewidth=1.5, zorder=1)

        rgba = np.tile(color, (E, 1))
        rgba[:, 3] = alpha_arrays[i]  

        plt.scatter(path[:, 0], path[:, 1], c=rgba, s=22, marker='o',
                    edgecolors='none', zorder=2)

        # plt.scatter(path[0, 0], path[0, 1], color=color, s=75, marker='*', zorder=3)
        # plt.scatter(path[-1, 0], path[-1, 1], color=color, s=75, marker='X', zorder=3)

    show_max = min(P, 16)
    legend_proxies = []
    legend_labels = []
    for i in range(show_max):
        c = base_colors[i % len(base_colors)]
        proxy = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=c, markersize=8, label=labels[i])
        legend_proxies.append(proxy)
        legend_labels.append(labels[i])

    if show_max > 0:
        plt.legend(legend_proxies, legend_labels, title="Paths (first 16)",
                   fontsize=9, title_fontsize=10, loc='best', ncol=2, framealpha=0.8)

    plt.title(title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pca_trajectory_3d(scores, labels=None, alpha_arrays=None,
                          title="3D PCA of Agent Weights (All Positions)",
                          elev=30, azim=45):
    """
    Plot 3D PCA trajectories per path with unique colors and increasing (or custom) alpha along time.

    Parameters
    ----------
    scores : ndarray
        Shape (P, E, 3) PCA scores for each path (P) across episodes (E).
    labels : list[str] | None
        Optional list of labels per path (length P). If None, uses 'Path i'.
    alpha_arrays : ndarray | None
        Shape (P, E) array of alphas to use along time. If None, defaults to linspace(0.05, 1.0, E).
    title : str
        Plot title.
    elev : float
        Elevation angle for 3D view
    azim : float
        Azimuth angle for 3D view
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        3D Axes object
    """
    from mpl_toolkits.mplot3d import Axes3D  
    
    P, E, D = scores.shape
    assert D == 3, "scores must be 3D per point (PC1, PC2, PC3)"

    if P <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, P))
    elif P <= 32:
        base_colors = plt.cm.tab20b(np.linspace(0, 1, min(P, 16)))
        base_colors = np.vstack([base_colors, plt.cm.tab20c(np.linspace(0, 1, P - base_colors.shape[0]))])
    else:
        base_colors = plt.cm.hsv(np.linspace(0, 1, P, endpoint=False))

    if labels is None:
        labels = [f"Path {i}" for i in range(P)]

    if alpha_arrays is None:
        alpha_arrays = [np.linspace(0.05, 1.0, E)] * P

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(P):
        path = scores[i]
        color = base_colors[i % len(base_colors)]
        rgb_color = color[:3]  

        ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                linestyle='-', color=rgb_color, alpha=0.25,
                linewidth=1.5)
        
        for j in range(E):
            alpha = alpha_arrays[i][j]
            rgba_color = (*rgb_color, alpha)  
            ax.scatter(path[j:j+1, 0], path[j:j+1, 1], path[j:j+1, 2], 
                      color=rgba_color,
                      s=22, marker='o', edgecolors='none')

    show_max = min(P, 16)
    legend_proxies = []
    legend_labels = []
    for i in range(show_max):
        c = base_colors[i % len(base_colors)]
        proxy = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=c, markersize=8, label=labels[i])
        legend_proxies.append(proxy)
        legend_labels.append(labels[i])

    if show_max > 0:
        ax.legend(legend_proxies, legend_labels, title="Paths (first 16)",
                 fontsize=9, title_fontsize=10, loc='best', ncol=2, framealpha=0.8)

    ax.set_title(title)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    
    ax.view_init(elev=elev, azim=azim)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_pca_trajectories_both(scores, labels=None, alpha_arrays=None,
                             title="PCA of Agent Weights (All Positions)",
                             elev=30, azim=45,
                             savefig=False, filename="pca_trajectories.png"):
    """
    Plot both 2D and 3D PCA trajectories in separate subplots.

    Parameters
    ----------
    scores : ndarray
        Shape (P, E, 3) PCA scores for each path (P) across episodes (E).
    labels : list[str] | None
        Optional list of labels per path (length P). If None, uses 'Path i'.
    alpha_arrays : ndarray | None
        Shape (P, E) array of alphas to use along time.
    title : str
        Plot title.
    elev : float
        Elevation angle for 3D view
    azim : float
        Azimuth angle for 3D view
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    P, E, D = scores.shape
    assert D >= 3, "scores must have at least 3 components"

    if P <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, P))
    elif P <= 32:
        base_colors = plt.cm.tab20b(np.linspace(0, 1, min(P, 16)))
        base_colors = np.vstack([base_colors, plt.cm.tab20c(np.linspace(0, 1, P - base_colors.shape[0]))])
    else:
        base_colors = plt.cm.hsv(np.linspace(0, 1, P, endpoint=False))

    if labels is None:
        labels = [f"Path {i}" for i in range(P)]

    if alpha_arrays is None:
        alpha_arrays = [np.linspace(0.05, 1.0, E)] * P

    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    
    # 2D plot
    ax1 = fig.add_subplot(121)
    
    for i in range(P):
        path = scores[i]
        color = base_colors[i % len(base_colors)]

        ax1.plot(path[:, 0], path[:, 1], linestyle='-', color=color, alpha=0.25,
                linewidth=1.5, zorder=1)

        rgba = np.tile(color, (E, 1))
        rgba[:, 3] = alpha_arrays[i]  

        ax1.scatter(path[:, 0], path[:, 1], c=rgba, s=22, marker='o',
                   edgecolors='none', zorder=2)
    
    ax1.set_title(f"{title} (2D)")
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i in range(P):
        path = scores[i]
        color = base_colors[i % len(base_colors)]
        rgb_color = color[:3]

        ax2.plot(path[:, 0], path[:, 1], path[:, 2], 
                linestyle='-', color=rgb_color, alpha=0.25,
                linewidth=1.5)
        
        for j in range(E):
            alpha = alpha_arrays[i][j]
            rgba_color = (*rgb_color, alpha)
            ax2.scatter(path[j:j+1, 0], path[j:j+1, 1], path[j:j+1, 2], 
                      color=rgba_color,
                      s=22, marker='o', edgecolors='none')
    
    ax2.set_title(f"{title} (3D)")
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    ax2.view_init(elev=elev, azim=azim)
    ax2.grid(True, alpha=0.3)
    
    show_max = min(P, 16)
    legend_proxies = []
    legend_labels = []
    for i in range(show_max):
        c = base_colors[i % len(base_colors)]
        proxy = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=c, markersize=8, label=labels[i])
        legend_proxies.append(proxy)
        legend_labels.append(labels[i])

    if show_max > 0:
        fig.legend(legend_proxies, legend_labels, title="Paths (first 16)",
                  fontsize=9, title_fontsize=10, loc='center right', 
                  bbox_to_anchor=(0.98, 0.5), ncol=1, framealpha=0.8)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    if savefig:
        plt.savefig(filename, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    
    return fig

def test_pca_over_all_positions(agent_mode='normal', pca_mode='2d'):
    agent = LinearAgent()
    env = Environment()
    from reservoir import initialize_reservoir
    reservoir = initialize_reservoir()

    rounds = 1
    episodes = 600
    time_steps = 30

    # Always get normal training weights to use as PCA reference frame
    reference_rewards, reference_trajectories, reference_weights = OutOfDistributionTraining(
        agent, env,
        rounds=rounds,
        episodes=episodes,
        time_steps=time_steps,
        mode='normal',
        verbose=False,
        return_weights=True
    )
    
    # Get the weights we want to analyze (based on agent_mode)
    if agent_mode == 'normal':
        rewards = reference_rewards
        trajectories = reference_trajectories
        weights = reference_weights
    elif agent_mode == 'reservoir':
        rewards, trajectories, _ ,_ , weights = TrainingToInference(
            agent, env, reservoir,
            rounds=rounds,
            episodes=episodes,
            time_steps=time_steps,
            verbose=False,
        )

    weights = np.array(weights)  
    reference_weights = np.array(reference_weights)
    print("Weights shape:", weights.shape)

    path_labels = [f"Angle {i * 22.5}" for r in range(rounds) for i in range(16)]
    alpha_arrays = compute_alpha_arrays(rewards, bin_size=15)

    # Fit PCA using reference weights (normal training)
    pca = PCA(n_components=3, svd_solver='full')
    P, E, A, D = reference_weights.shape
    F = A * D
    X_ref = reference_weights.reshape(P * E, F)
    pca.fit(X_ref)
    
    P, E, A, D = weights.shape
    F = A * D
    X = weights.reshape(P * E, F)
    scores_all = pca.transform(X)
    scores = scores_all.reshape(P, E, pca.n_components)

    if pca_mode == '2d':
        plot_pca_trajectories(
            scores[:, :, :2],  # Use only first 2 components
            labels=path_labels,
            alpha_arrays=alpha_arrays,
            title=f"PCA of {agent_mode.capitalize()} Agent Weights (All Positions)"
        )
    elif pca_mode == '3d':
        plot_pca_trajectory_3d(
            scores,
            labels=path_labels,
            alpha_arrays=alpha_arrays,
            title=f"3D PCA of {agent_mode.capitalize()} Agent Weights (All Positions)",
            elev=30,
            azim=45
        )
    elif pca_mode == 'both':
        plot_pca_trajectories_both(
            scores,
            labels=path_labels,
            alpha_arrays=alpha_arrays,
            title=f"PCA of {agent_mode.capitalize()} Agent Weights (All Positions)",
            elev=30,
            azim=45,
            savefig=True,
            filename=f"pca_trajectories_{agent_mode}.png"
        )

if __name__ == "__main__":
    test_pca_over_all_positions(agent_mode='reservoir', pca_mode='both')