# In this file we aim to save and visualize how thew weights of the agent dynamically change during the training process.
# First, we need to import from trainingUtils the stuff we need to perform a training

from trainingUtils import train_episode, OutOfDistributionTraining
from agent import LinearAgent, animate_weights
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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

def test_pca_over_all_positions():
    agent = LinearAgent()
    env = Environment()

    rounds = 1
    episodes = 600
    time_steps = 30

    rewards, trajectories, weights = OutOfDistributionTraining(
        agent, env,
        rounds=rounds,
        episodes=episodes,
        time_steps=time_steps,
        mode='normal',
        verbose=False,
        return_weights=True
    )

    weights = np.array(weights)  
    print("Weights shape:", weights.shape)

    scores, pca = pca_fit_all_paths(weights, n_components=2)

    P = scores.shape[0]
    path_labels = [f"Angle {i * 22.5}" for r in range(rounds) for i in range(16)]
    alpha_arrays = compute_alpha_arrays(rewards, bin_size=15)

    plot_pca_trajectories(scores, labels=path_labels, title="PCA Trajectories by Position (one PCA fit)", alpha_arrays=alpha_arrays)
    print(rewards.shape)

if __name__ == "__main__":
    test_pca_over_all_positions()