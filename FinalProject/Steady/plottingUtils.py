import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
from typing import Sequence, Mapping, Any

warnings.filterwarnings("ignore", message="There are no gridspecs with layoutgrids")

def agg(r, bin_size):
    """
    Aggregate runs over bins of size bin_size.

    Parameters:
    - r: numpy array of shape (runs, episodes)
    - bin_size: int, number of episodes per bin

    Returns:
    - mean_rewards: 1D array of length bins, mean reward across runs per bin
    - std_rewards: 1D array of length bins, std deviation across runs per bin
    - x_bins: 1D array of length bins, episode index for each bin (center)
    """
    runs, episodes = r.shape
    bins = episodes // bin_size
    # truncate and reshape
    binned = (
        r[:, : bins * bin_size]
        .reshape(runs, bins, bin_size)
        .mean(axis=2)
    )  # shape: (runs, bins)
    mean_rewards = binned.mean(axis=0)
    std_rewards = binned.std(axis=0)
    x_bins = np.arange(1, bins + 1) * bin_size
    return mean_rewards, std_rewards, x_bins


def plot_rewards(rewards, bin_size=10, high_point=1.5, figsize=(12, 16)):
    """
    Plot aggregated rewards for runs of different types in a grid of 4 rows x 2 columns.
    There are 8 types (0-7) cycling through runs by index modulo 8.

    Parameters:
    - rewards: numpy array of shape (n_runs, episodes)
    - bin_size: int, number of episodes per aggregation bin
    - high_point: float, draw a horizontal line at this reward level
    - figsize: tuple, figure size
    """
    n_runs, episodes = rewards.shape
    # Prepare figure and axes
    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette(n_colors=max(10, n_runs))

    for type_idx in range(8):
        ax = axes[type_idx]
        # select runs of this type
        idx = np.arange(n_runs)[np.arange(n_runs) % 8 == type_idx]
        group = rewards[idx]
        # plot each individual run
        for i_run, single in enumerate(group):
            mean_i, _, x = agg(single[np.newaxis, :], bin_size)
            ax.plot(x, mean_i, alpha=0.6, linewidth=1, color=palette[i_run])

        # if multiple runs, plot aggregated mean
        if len(group) > 1:
            mean_all, std_all, x = agg(group, bin_size)
            ax.plot(x, mean_all, color='pink', linewidth=2)
            # optional: shade std
            ax.fill_between(x, mean_all - std_all, mean_all + std_all, color='pink', alpha=0.2)

        # horizontal high point line
        ax.axhline(high_point, color='red', linestyle='--', linewidth=1)
        ax.set_title(f"Run type {type_idx}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(trajectories, batch_size=100, figsize=(12, 16)):
    """
    Plot average agent trajectories around multiple food positions.

    - For each batch of up to 8 food positions, create a subplot.
    - Draw an empty circle (radius=0.15) and filled dot (radius=0.075) at each food position.
    - Every `batch_size` episodes per food, compute the average path (ignoring NaN padding) and plot it.

    Parameters:
    - trajectories: list of dicts, each with keys:
        - 'food_position': array-like of shape (2,)
        - 'trajectory': np.ndarray of shape (n_episodes, n_steps, 2) with NaN for padded timesteps.
    - batch_size: int, number of episodes to average before plotting a path
    - figsize: tuple, figure size
    """
    n = len(trajectories)
    max_per_plot = 8
    n_plots = int(np.ceil(n / max_per_plot))
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for p in range(n_plots):
        ax = axes[p]
        start = p * max_per_plot
        end = min(start + max_per_plot, n)
        group = trajectories[start:end]

        for entry in group:
            food = np.array(entry['food_position'], dtype=float)
            trajs = np.array(entry['trajectory'], dtype=float)  # shape (E, T, 2)

            circle = plt.Circle(food, 0.15, fill=False, linewidth=1.5, linestyle='--')
            dot = plt.Circle(food, 0.075, fill=True)
            ax.add_patch(circle)
            ax.add_patch(dot)

            E, T, _ = trajs.shape
            n_batches = int(np.ceil(E / batch_size))

            for b in range(n_batches):
                idx0 = b * batch_size
                idx1 = min((b + 1) * batch_size, E)
                batch = trajs[idx0:idx1] 
                if np.isnan(batch).all():
                    continue
                mean_path = np.nanmean(batch, axis=0)  
                valid = ~np.isnan(mean_path[:, 0])
                ax.plot(mean_path[valid, 0], mean_path[valid, 1], alpha=0.7, linewidth=1)

        ax.set_title(f"Batch {p+1}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal')

    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def _plot_single_run(ax_perf: plt.Axes,
                     ax_env:  plt.Axes,
                     lr_values: Sequence[float],
                     rewards: np.ndarray,
                     agent_xy: tuple[float, float],
                     food_xy:  tuple[float, float],
                     run_idx:  int,
                     env_lims: float,
                     plotlog: bool) -> None:
    """Render a single run into the two supplied Axes."""
    rewards = np.asarray(rewards)
    if rewards.ndim != 2 or rewards.shape[0] != len(lr_values):
        raise ValueError("`rewards` must have shape (len(lr_values), n_episodes)")

    means = rewards[:, 0]          # first column is already the mean
    stds  = rewards[:, 1]

    ax_perf.errorbar(lr_values, means, yerr=stds, fmt="o-", capsize=3)
    if plotlog:
        ax_perf.set_xscale("log")
    else:
        ax_perf.set_xscale("linear")
    ax_perf.set_xlabel("Learning rate")
    ax_perf.set_ylabel("Mean reward")
    ax_perf.set_title(f"Run #{run_idx}", fontsize=10)
    ax_perf.grid(True, alpha=0.3)

    ax_env.scatter(*agent_xy, s=60, marker="o", label="Agent")
    ax_env.scatter(*food_xy,  s=60, marker="x", label="Food")
    ax_env.set_xlim(-env_lims, env_lims)
    ax_env.set_ylim(-env_lims, env_lims)
    ax_env.set_aspect("equal", "box")
    ax_env.set_xticks([]); ax_env.set_yticks([])
    ax_env.grid(True, alpha=0.2)
    ax_env.legend(frameon=False, fontsize=6, loc="upper right")

def plot_one_shot_eval(lr_values: Sequence[float],
                            data: Sequence[Mapping[str, Any]],
                            rows: int = 4,
                            cols: int = 4,
                            figsize: tuple[int, int] = (18, 18),
                            env_lims: float = 0.75,
                            width_ratio = (1, 1),
                            plotlog = True,
                            title = "One-shot evaluation across runs",
                            savefig = True,
                            filename = "one_shot_eval.png"):
    """
    Parameters
    ----------
    lr_values : sequence of float
        Learning-rate values (x-axis).
    data : sequence of dict
        Each entry must contain:
           'total_rewards'  -> 2-D array (#lr, #episodes)
           'agent_position' -> (x, y)
           'food_position'  -> (x, y)
    rows, cols : int
        Grid layout for runs.  rows*cols must equal len(data).
    figsize : (w, h)
        Inches for the overall figure.
    env_lims : float
        Half-width of the square environment box.
    width_ratio : (int, int)
        Widths of [performance, environment] panes inside every cell.
    title : str
        Figure-level title.
    save_to : str | None
        If given, saves the PNG at this path and returns the Figure.
    """
    if rows * cols != len(data):
        raise ValueError(f"rows*cols ({rows*cols}) must equal len(data) ({len(data)})")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer_gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.55)
    fig.suptitle(title, fontsize=20, y=0.98)

    for idx, entry in enumerate(data):
        inner_gs = gridspec.GridSpecFromSubplotSpec(  # two sub-axes in each cell
            nrows=1, ncols=2, subplot_spec=outer_gs[idx],
            width_ratios=width_ratio, wspace=0.25
        )
        ax_perf = fig.add_subplot(inner_gs[0])
        ax_env  = fig.add_subplot(inner_gs[1])

        _plot_single_run(
            ax_perf,
            ax_env,
            lr_values,
            entry["total_rewards"],
            entry["agent_position"],
            entry["food_position"],
            run_idx=idx + 1,
            env_lims=env_lims,
            plotlog=plotlog
        )

    if savefig:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()