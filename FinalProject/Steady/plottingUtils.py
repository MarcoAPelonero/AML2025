import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
from typing import Sequence, Mapping, Any, Union
import json
from pathlib import Path


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

def plot_single_run(rewards: np.ndarray, bin_size=10, high_point=1.5, color='blue', title="Single Run", figsize=(8, 4)):
    """
    Plot a single run's rewards aggregated over bins.

    Parameters:
    - rewards: 1D array of length (episodes)
    - bin_size: int, number of episodes per aggregation bin
    - high_point: float, draw a horizontal line at this reward level
    - color: line color
    - title: plot title
    - figsize: figure size (width, height)
    """
    if rewards.ndim != 1:
        raise ValueError("Expected a 1D array of rewards.")
    
    mean, _, x = agg(rewards[np.newaxis, :], bin_size)

    plt.figure(figsize=figsize)
    plt.plot(x, mean, color=color, linewidth=2)
    plt.axhline(high_point, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_rewards(rewards, bin_size=25, high_point=1.5, figsize=(12, 16), savefig=False, filename="rewards_plot.png"):
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
    if savefig:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def plot_rewards_as_article(rewards, bin_size=25, high_point=1.5, figsize=(10, 6),
                 savefig=False, filename="rewards_plot.png", show_individual=False):
   
    n_runs, episodes = rewards.shape

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if show_individual:
        palette = sns.color_palette(n_colors=max(10, n_runs))
        for i_run, single in enumerate(rewards):
            mean_i, _, x = agg(single[np.newaxis, :], bin_size)
            ax.plot(x, mean_i, alpha=0.35, linewidth=1, color=palette[i_run % len(palette)])

    mean_all, std_all, x = agg(rewards, bin_size)
    print(std_all)
    ax.plot(x, mean_all, color='pink', linewidth=2, label='Mean (all runs)')
    ax.fill_between(x, mean_all - std_all, mean_all + std_all, color='pink', alpha=0.2, label='±1 std')

    ax.axhline(high_point, color='red', linestyle='--', linewidth=1, label=f'high={high_point}')

    ax.set_title("Rewards for Gradient Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if savefig:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_out_of_distribution_comparison(
    rewards_grad_even, rewards_grad_odd,
    rewards_res_even, rewards_res_odd,
    bin_size=25, high_point=1.5,
    figsize=(12, 10), savefig=False, filename="ood_comparison.png"
):
    """
    Compare rewards between gradient-trained and reservoir-trained networks,
    separating even (in-distribution) and odd (out-of-distribution) angles.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    datasets = [
        (rewards_grad_even, "Gradient Training (Even / In-Distribution)", "blue"),
        (rewards_grad_odd,  "Gradient Training (Odd / Out-of-Distribution)", "orange"),
        (rewards_res_even,  "Reservoir Training (Even / In-Distribution)", "green"),
        (rewards_res_odd,   "Reservoir Training (Odd / Out-of-Distribution)", "purple"),
    ]

    for ax, (rewards, title, color) in zip(axes, datasets):
        rewards = np.array(rewards)
        mean_all, std_all, x = agg(rewards, bin_size)

        ax.plot(x, mean_all, color=color, linewidth=2, label='Mean')
        ax.fill_between(x, mean_all - std_all, mean_all + std_all,
                        color=color, alpha=0.2, label='±1 std')

        ax.axhline(high_point, color='red', linestyle='--', linewidth=1)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.1, 1.6)
        ax.legend()
        ax.tick_params(axis='both', labelsize=14)  # <-- Make axis numbers bigger

    plt.tight_layout()
    if savefig:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_rewards_ood(rewards, bin_size=10, high_point=1.5, figsize=(16, 16), title=None, savefig=False, filename="rewards_plot.png"):
    """
    Plot aggregated rewards for runs of different types in a grid of 4 rows x 4 columns.
    There are 16 types (0-15) cycling through runs by index modulo 16.

    Parameters:
    - rewards: numpy array of shape (n_runs, episodes)
    - bin_size: int, number of episodes per aggregation bin
    - high_point: float, draw a horizontal line at this reward level
    - figsize: tuple, figure size
    """
    n_runs, episodes = rewards.shape
    # Prepare figure and axes
    fig, axes = plt.subplots(4, 4, figsize=figsize, sharex=True, sharey=True)
    if title:
        fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    palette = sns.color_palette(n_colors=max(10, n_runs))

    for type_idx in range(16):
        ax = axes[type_idx]
        idx = np.arange(n_runs)[np.arange(n_runs) % 16 == type_idx]
        group = rewards[idx]
        for i_run, single in enumerate(group):
            mean_i, _, x = agg(single[np.newaxis, :], bin_size)
            ax.plot(x, mean_i, alpha=0.6, linewidth=1, color=palette[i_run])

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
    if savefig:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def plot_trajectories(trajectories, batch_size=100, figsize=(12, 16), savefig=False, filename="trajectories_plot.png"):
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
    if savefig:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def plot_trajectories_ood(trajectories, batch_size=100, figsize=(12, 16), savefig=False, filename="trajectories_plot.png"):
    """
    Plot average agent trajectories around multiple food positions.

    - Draw an empty circle (radius=0.15) and filled dot (radius=0.075) at each food position.
    - Every `batch_size` episodes per food, compute the average path (ignoring NaN padding) and plot it.

    Parameters:
    - trajectories: list of dicts, each with keys:
        - 'food_position': array-like of shape (2,)
        - 'trajectory': np.ndarray of shape (n_episodes, n_steps, 2) with NaN for padded timesteps.
    - batch_size: int, number of episodes to average before plotting a path
    - figsize: tuple, figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    for entry in trajectories:
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

    ax.set_title("Agent trajectories around food positions")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal')

    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
        plt.close(fig)
    else:
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

def _plot_grouped_runs(
    ax_perf,
    ax_env,
    lr_values_list: Sequence[Sequence[float]],
    total_rewards_list: Sequence[np.ndarray],
    agent_positions: Sequence[tuple[float, float]],
    food_position: tuple[float, float],
    labels: Sequence[str],
    env_lims: float = 0.75,
    plotlog: bool = True,
):
    for lr_values, total_rewards, label in zip(lr_values_list, total_rewards_list, labels):
        tr = np.asarray(total_rewards)
        if tr.ndim != 2:
            raise ValueError(f"total_rewards for {label} must be 2-D (#lr, #episodes) or (#lr, 2 summary), got shape {tr.shape}")

        # If the second dimension is 2, treat it as [mean, std]; otherwise aggregate raw episodes.
        if tr.shape[1] == 2:
            mean_rewards = tr[:, 0]
            std_rewards = tr[:, 1]
        else:
            mean_rewards = tr.mean(axis=1)
            std_rewards = tr.std(axis=1)
        ax_perf.plot(lr_values, mean_rewards, label=label)
        ax_perf.fill_between(
            lr_values,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )
    ax_perf.set_xlabel("Learning rate")
    ax_perf.set_ylabel("Total reward (mean ± std)")
    if plotlog:
        ax_perf.set_xscale("log")
    ax_perf.set_title("Performance (overlaid runs)")

    ax_env.set_aspect("equal", "box")
    lim = env_lims
    ax_env.set_xlim(-lim, lim)
    ax_env.set_ylim(-lim, lim)
    square = plt.Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False, linestyle="--", linewidth=1)
    ax_env.add_patch(square)
    fx, fy = food_position
    ax_env.scatter([fx], [fy], marker="*", s=150, edgecolors="black", zorder=3)
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    for idx, (agent_pos, label) in enumerate(zip(agent_positions, labels)):
        ax_env.scatter(
            [agent_pos[0]],
            [agent_pos[1]],
            marker=markers[idx % len(markers)],
            edgecolors="black",
            zorder=2,
        )
    # Remove axis numbers, labels, and individual titles
    ax_env.set_xticks([])
    ax_env.set_yticks([])
    ax_env.set_xlabel("")
    ax_env.set_ylabel("")
    ax_env.set_title("")

def plot_one_shot_eval_from_jsons(
    json_paths: Sequence[Union[str, Path]],
    lr_values: Sequence[float] | None = None,
    rows: int = 4,
    cols: int = 4,
    figsize: tuple[int, int] = (18, 18),
    env_lims: float = 0.75,
    width_ratio=(1, 1),
    plotlog: bool = True,
    title: str = "One-shot evaluation across food positions (grouped)",
    savefig: bool = True,
    filename: str = "one_shot_eval_grouped.png",
):
    """
    Loads multiple JSON files and groups runs by identical food_position.

    Supports these JSON formats:
      * A dict with 'lr_values' and 'data' (list of run dicts).
      * A single run dict with 'total_rewards' and optionally its own 'lr_values'.
      * A top-level list of run dicts (your current file), where each run must have
        'total_rewards', 'agent_position', and 'food_position'. Learning-rate values are taken from
        the provided `lr_values` argument or defaulted to 1..#lr if missing.

    Parameters
    ----------
    json_paths : sequence of paths to JSON files.
    lr_values : optional common learning-rate sequence to apply when the JSON(s) do not embed lr_values.
    rows, cols : grid layout for unique food positions.
    """
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}

    def _extract_runs(base_obj: Any, source_label: str, fallback_lr: Sequence[float] | None):
        runs = []
        if isinstance(base_obj, dict):
            # Case: dict with 'data' and 'lr_values'
            if "data" in base_obj and "lr_values" in base_obj:
                lr_vals = base_obj["lr_values"]
                for i, entry in enumerate(base_obj["data"], start=1):
                    run_label = f"{source_label}:{i}"
                    runs.append(
                        {
                            "lr_values": lr_vals,
                            "total_rewards": entry["total_rewards"],
                            "agent_position": tuple(entry["agent_position"]),
                            "food_position": tuple(entry["food_position"]),
                            "label": run_label,
                        }
                    )
            elif "total_rewards" in base_obj:
                if "lr_values" in base_obj:
                    lr_vals = base_obj["lr_values"]
                elif fallback_lr is not None:
                    lr_vals = fallback_lr
                else:
                    n_lr = len(base_obj["total_rewards"])
                    print(
                        f"Warning: no 'lr_values' found in {source_label}; inferring as 1..{n_lr}"
                    )
                    lr_vals = list(range(1, n_lr + 1))
                run_label = source_label
                runs.append(
                    {
                        "lr_values": lr_vals,
                        "total_rewards": base_obj["total_rewards"],
                        "agent_position": tuple(base_obj["agent_position"]),
                        "food_position": tuple(base_obj["food_position"]),
                        "label": run_label,
                    }
                )
            else:
                raise ValueError(
                    f"Unrecognized JSON structure in {source_label}: dict missing expected keys."
                )
        elif isinstance(base_obj, list):
            for i, entry in enumerate(base_obj, start=1):
                if not isinstance(entry, dict) or "total_rewards" not in entry:
                    raise ValueError(
                        f"Entry #{i} in list from {source_label} is not a valid run dict."
                    )
                if "lr_values" in entry:
                    lr_vals_entry = entry["lr_values"]
                elif fallback_lr is not None:
                    lr_vals_entry = fallback_lr
                else:
                    n_lr = len(entry["total_rewards"])
                    print(
                        f"Warning: entry #{i} in {source_label} lacks 'lr_values'; inferring as 1..{n_lr}"
                    )
                    lr_vals_entry = list(range(1, n_lr + 1))
                run_label = f"{source_label}:{i}"
                runs.append(
                    {
                        "lr_values": lr_vals_entry,
                        "total_rewards": entry["total_rewards"],
                        "agent_position": tuple(entry["agent_position"]),
                        "food_position": tuple(entry["food_position"]),
                        "label": run_label,
                    }
                )
        else:
            raise ValueError(f"Unrecognized top-level JSON type in {source_label}: {type(base_obj)}")
        return runs

    for path in json_paths:
        p = Path(path)
        with open(p, "r") as f:
            obj = json.load(f)
        runs = _extract_runs(obj, p.name, lr_values)
        for run in runs:
            food_pos = tuple(run["food_position"])
            grouped.setdefault(food_pos, []).append(run)

    unique_foods = list(grouped.keys())
    if rows * cols != len(unique_foods):
        raise ValueError(
            f"rows*cols ({rows*cols}) must equal number of unique food positions ({len(unique_foods)})"
        )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer_gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.55)
    fig.suptitle(title, fontsize=20, y=0.98)

    handles, labels = [], []

    for idx, food_pos in enumerate(unique_foods):
        runs = grouped[food_pos]
        lr_values_list = [run["lr_values"] for run in runs]
        total_rewards_list = [np.asarray(run["total_rewards"]) for run in runs]
        agent_positions = [run["agent_position"] for run in runs]
        labels = [run["label"] for run in runs]

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=1,
            ncols=2,
            subplot_spec=outer_gs[idx],
            width_ratios=width_ratio,
            wspace=0.25,
        )
        ax_perf = fig.add_subplot(inner_gs[0])
        ax_env = fig.add_subplot(inner_gs[1])

        _plot_grouped_runs(
            ax_perf,
            ax_env,
            lr_values_list,
            total_rewards_list,
            agent_positions,
            food_position=food_pos,
            labels=labels,
            env_lims=env_lims,
            plotlog=plotlog,
        )

        # Collect legend handles and labels from the first performance plot
        if idx == 0:
            handles, labels = ax_perf.get_legend_handles_labels()

    # Add a single legend to the figure
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, fontsize="small")

    if savefig:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()
    return fig

def main():
    json_paths = [
        "images_reservoir/one_shot_gradient_results.json",
        "images/one_shot_gradient_results.json"
    ]
    plot_one_shot_eval_from_jsons(
        json_paths,
        rows=4,
        cols=4,
        figsize=(18, 18),
        lr_values=[0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10],
        env_lims=0.75,
        width_ratio=(1, 1),
        plotlog=True,
        title="One-shot evaluation across food positions (grouped)",
        savefig=True,
        filename="one_shot_eval_grouped.png"
    )

if __name__ == "__main__":
    main()