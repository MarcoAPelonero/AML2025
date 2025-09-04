import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.patches import Arc
from matplotlib.lines import Line2D

"""
This module handles the plotting function that are mostly used in the presentation of the results, mostly regarding what's beyond the article's reproduction
This is only used in post processing, it loads back the json files in out of the simulations and handles them producing the graphs we need.
"""

def read_json_outfile(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def add_icon(ax, img_path, pos, zoom=0.1):
    """
    Reads a png image and adds it to the axis ax at position pos (x,y) with given zoom, this is a
    utility function for custom icons in the plots.
    """
    img = mpimg.imread(img_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, pos, frameon=False)
    ax.add_artist(ab)

def plot_single_angle(angle_data, lr_list=[0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10],
                      title=None, savefig=False, filename="oneshot_single_angle.png"):
    """
    Takes in the data coming from one of the oneShot simulations with the respective learning rate list,
    and plots the avg rewards plus error bars on the left, and the agent and food position with the angle arc on the right.
    """

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    if title:
        fig.suptitle(title)

    agent_position = np.array(angle_data["agent_position"], dtype=float)
    food_position  = np.array(angle_data["food_position"],  dtype=float)
    theta          = float(angle_data["theta0"])

    add_icon(ax1, "icons/person_icon.png", agent_position, zoom=0.08)
    add_icon(ax1, "icons/apple_icon.png",  food_position,  zoom=0.04)

    path_line, = ax1.plot([agent_position[0], food_position[0]],
                          [agent_position[1], food_position[1]],
                          linestyle=":", linewidth=2, label="Path")

    arc_radius = 0.22
    theta_deg  = theta
    ax1.plot([agent_position[0], agent_position[0] + arc_radius],
             [agent_position[1], agent_position[1]],
             linestyle="--", linewidth=1, alpha=0.5)

    arc_theta1, arc_theta2 = (theta_deg, 0) if theta_deg < 0 else (0, theta_deg)
    arc = Arc(xy=agent_position, width=2*arc_radius, height=2*arc_radius,
              angle=0, theta1=arc_theta1, theta2=arc_theta2,
              linestyle="--", linewidth=2)
    ax1.add_patch(arc)

    label_point = agent_position + arc_radius * np.array([np.cos(theta), np.sin(theta)])
    ax1.text(label_point[0], label_point[1], f"{theta_deg:.1f}°", ha="left", va="bottom", fontsize=11)

    lim = 0.6
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_title('Agent and Food Positions')
    ax1.tick_params(axis='both', labelsize=14)

    ax1.set_box_aspect(1)           

    agent_proxy = Line2D([0],[0], marker='o', color='none', markerfacecolor='k', label='Agent')
    food_proxy  = Line2D([0],[0], marker='o', color='none', markerfacecolor='r', label='Food')
    ax1.legend(handles=[agent_proxy, food_proxy, path_line], loc='upper left')

    means = [r[0] for r in angle_data["total_rewards"]]
    stds  = [r[1] for r in angle_data["total_rewards"]]
    ax0.errorbar(lr_list, means, yerr=stds, fmt='-o')
    ax0.set_xscale('log')
    ax0.set_xlabel('Learning Rate', fontsize=12)
    ax0.set_ylabel('Total Rewards', fontsize=12)
    ax0.set_title('Total Rewards vs Learning Rate')
    ax0.tick_params(axis='both', labelsize=14)
    ax0.axhline(y=1.5, color='r', linestyle='--', label='Max Reward')
    ax0.legend()

    if savefig:
        fig.savefig(filename, dpi=300)
    else:
        plt.show()

def plot_single_angle_one_shot(angle_data, 
                               k_list=[1, 2, 3, 5, 7, 10, 15, 20],
                               title=None, savefig=False, filename="oneshot_single_angle.png"):
    """
    Does the same thing as the previous function but for the one shot experiments where we vary K instead of the learning rate,
    so for data coming from trueOneShot files.
    """
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, 0]) 
    ax1 = fig.add_subplot(gs[0, 1])  

    if title:
        fig.suptitle(title)

    agent_position = np.array(angle_data["agent_position"], dtype=float)
    food_position  = np.array(angle_data["food_position"],  dtype=float)
    theta          = float(angle_data["theta0"])

    add_icon(ax1, "icons/person_icon.png", agent_position, zoom=0.08)
    add_icon(ax1, "icons/apple_icon.png",  food_position,  zoom=0.04)

    path_line, = ax1.plot(
        [agent_position[0], food_position[0]],
        [agent_position[1], food_position[1]],
        linestyle=":", linewidth=2, label="Path"
    )

    arc_radius = 0.22
    theta_deg  = theta

    ax1.plot(
        [agent_position[0], agent_position[0] + arc_radius],
        [agent_position[1], agent_position[1]],
        linestyle="--", linewidth=1, alpha=0.5
    )

    arc_theta1, arc_theta2 = 0, theta_deg
    if arc_theta2 < arc_theta1:
        arc_theta1, arc_theta2 = arc_theta2, arc_theta1

    arc = Arc(
        xy=agent_position, width=2*arc_radius, height=2*arc_radius,
        angle=0, theta1=arc_theta1, theta2=arc_theta2,
        linestyle="--", linewidth=2
    )
    ax1.add_patch(arc)

    label_point = agent_position + arc_radius * np.array([np.cos(theta), np.sin(theta)])
    ax1.text(label_point[0], label_point[1], f"{theta_deg:.1f}°",
             ha="left", va="bottom", fontsize=11)

    lim = 0.6
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_title('Agent and Food Positions')

    ax1.set_box_aspect(1) 

    agent_proxy = Line2D([0], [0], marker='o', color='none', markerfacecolor='k', label='Agent')
    food_proxy  = Line2D([0], [0], marker='o', color='none', markerfacecolor='r', label='Food')
    ax1.legend(handles=[agent_proxy, food_proxy, path_line], loc='upper left')

    means = [r[0] for r in angle_data["total_rewards"]]
    stds  = [r[1] for r in angle_data["total_rewards"]]

    ax0.errorbar(k_list, means, yerr=stds, fmt='-o')
    ax0.set_xlabel('K Value')
    ax0.set_ylabel('Total Rewards')
    ax0.set_title('Total Rewards vs K Value')
    ax0.axhline(y=1.5, color='r', linestyle='--', label='Max Reward')
    ax0.legend()

    if savefig:
        fig.savefig(filename, dpi=300)
    else:
        plt.show()

def plot_multiple_angles_grid(
    data_list,
    lr_list=[0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10],
    figsize=(20, 20),
    savefig=False,
    filename="datagrid.png",
    title=None
):
    """
    4x4 grid; each cell has two equally-sized subplots (left: rewards vs lr, right: positions).
    Uses constrained_layout to prevent layout from shifting widths between the two subplots. This is the plot_one_angle function
    extended to all 16 angles taken into consideration here, in one single plot. 
    """

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if title:
        fig.suptitle(title)
    outer = fig.add_gridspec(4, 4, wspace=0.2, hspace=0.25)

    data_to_plot = data_list[:16]

    for i, angle_data in enumerate(data_to_plot):
        row, col = divmod(i, 4)

        inner = outer[row, col].subgridspec(
            1, 2,
            wspace=0.1,  
            width_ratios=[1, 1]  
        )

        ax_left  = fig.add_subplot(inner[0, 0])
        ax_right = fig.add_subplot(inner[0, 1])

        theta = float(angle_data["theta0"])
        means = [r[0] for r in angle_data["total_rewards"]]
        stds  = [r[1] for r in angle_data["total_rewards"]]

        ax_left.errorbar(lr_list, means, yerr=stds, fmt='-o', markersize=4)
        ax_left.set_xlabel('K Value', fontsize=8)
        ax_left.set_ylabel('Total Rewards', fontsize=8)
        ax_left.set_title(f'Rewards vs K (θ={theta:.1f}°)', fontsize=9)
        ax_left.axhline(y=1.5, color='r', linestyle='--', alpha=0.7)
        ax_left.tick_params(labelsize=7)
        ax_left.set_xscale('log')

        agent_position = np.array(angle_data["agent_position"], dtype=float)
        food_position  = np.array(angle_data["food_position"], dtype=float)

        try:
            add_icon(ax_right, "icons/person_icon.png", agent_position, zoom=0.04)
            add_icon(ax_right, "icons/apple_icon.png",  food_position,  zoom=0.02)
        except Exception:
            ax_right.plot(agent_position[0], agent_position[1], 'ko', markersize=6, label='Agent')
            ax_right.plot(food_position[0],  food_position[1],  'ro', markersize=6, label='Food')

        ax_right.plot(
            [agent_position[0], food_position[0]],
            [agent_position[1], food_position[1]],
            linestyle=":", linewidth=1.5, alpha=0.7
        )

        arc_radius = 0.15
        ax_right.plot(
            [agent_position[0], agent_position[0] + arc_radius],
            [agent_position[1], agent_position[1]],
            linestyle="--", linewidth=1, alpha=0.5
        )
        arc_theta1, arc_theta2 = (0, theta) if theta >= 0 else (theta, 0)
        ax_right.add_patch(Arc(
            xy=agent_position, width=2*arc_radius, height=2*arc_radius,
            angle=0, theta1=arc_theta1, theta2=arc_theta2,
            linestyle="--", linewidth=1.5
        ))

        lim = 0.6
        ax_right.set_xlim(-lim, lim)
        ax_right.set_ylim(-lim, lim)
        ax_right.set_aspect('equal')  
        ax_right.set_title(f'Position (θ={theta:.1f}°)', fontsize=9)
        ax_right.tick_params(labelsize=7)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    if savefig:
        plt.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_multiple_angles_grid_comparison(
    data_lists, 
    lr_list=[0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10],
    labels=("Gradient", "Reservoir", "Res×Multiplier", "One-Shot×Multiplier"),
    colors=None,
    figsize=(22, 20),
    savefig=False,
    filename="datagrid_comparison.png",
    title=None,
    max_reward_line=1.5,
    show_position_from=0  
):
    """
    Final version of the plot_one_angle function, extended to a 4x4 grid of angles, and comparing four different methods in each subplot.
    Parameters
    ----------
    data_lists : list[list[dict]]
        A list of FOUR datasets; each is a list of angle_data dicts (>=16 items).
        Each angle_data has keys: "agent_position", "food_position", "theta0", "total_rewards".
    lr_list : list[float]
        Learning rates corresponding to total_rewards entries.
    labels : tuple[str, str, str, str]
        Legend labels for the four methods.
    colors : list | None
        Optional list of 4 matplotlib color specs. If None, uses tab10.
    figsize : tuple
    savefig : bool
    filename : str
    title : str | None
    max_reward_line : float
    show_position_from : int
        Index in [0..3] choosing which dataset supplies positions/arc.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    from matplotlib.lines import Line2D

    assert isinstance(data_lists, (list, tuple)) and len(data_lists) == 4, \
        "data_lists must be a list of FOUR datasets (one per method)."
    for dl in data_lists:
        assert len(dl) >= 16, "Each dataset must have at least 16 angle entries."
    assert 0 <= show_position_from < 4, "show_position_from must be 0..3"

    if colors is None:
        colors = [plt.get_cmap("tab10")(i) for i in range(4)]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if title:
        fig.suptitle(title)
    outer = fig.add_gridspec(4, 4, wspace=0.2, hspace=0.25)

    legend_handles = [
        Line2D([0], [0], color=colors[i], linewidth=2, marker='o', label=labels[i])
        for i in range(4)
    ]

    per_method_16 = [dl[:16] for dl in data_lists]

    for i in range(16):
        row, col = divmod(i, 4)
        inner = outer[row, col].subgridspec(
            1, 2, wspace=0.1, width_ratios=[1, 1]
        )
        ax_left  = fig.add_subplot(inner[0, 0])
        ax_right = fig.add_subplot(inner[0, 1])

        theta_here = float(per_method_16[0][i]["theta0"])  
        for m in range(4):
            angle_data = per_method_16[m][i]
            vals  = angle_data["total_rewards"]
            means = np.array([r[0] for r in vals], dtype=float)
            stds  = np.array([r[1] for r in vals], dtype=float)

            n = min(len(lr_list), len(means), len(stds))
            x = np.array(lr_list[:n], dtype=float)
            mvals = means[:n]
            svals = stds[:n]

            ax_left.plot(x, mvals, marker='o', linewidth=2, color=colors[m])
            ax_left.fill_between(x, mvals - svals, mvals + svals, alpha=0.2, edgecolor='none', color=colors[m])

        ax_left.set_xscale('log')
        ax_left.set_xlabel('K Value', fontsize=8)
        ax_left.set_ylabel('Total Rewards', fontsize=8)
        ax_left.set_title(f'Rewards vs K (θ={theta_here:.1f}°)', fontsize=9)
        ax_left.axhline(y=max_reward_line, color='r', linestyle='--', alpha=0.7)
        ax_left.tick_params(labelsize=7)
        ax_left.grid(True, which='both', linestyle=':', alpha=0.3)

        ref = per_method_16[show_position_from][i]
        agent_position = np.array(ref["agent_position"], dtype=float)
        food_position  = np.array(ref["food_position"],  dtype=float)
        theta          = float(ref["theta0"])

        try:
            add_icon(ax_right, "icons/person_icon.png", agent_position, zoom=0.04)
            add_icon(ax_right, "icons/apple_icon.png",  food_position,  zoom=0.02)
        except Exception:
            ax_right.plot(agent_position[0], agent_position[1], 'ko', markersize=6)
            ax_right.plot(food_position[0],  food_position[1],  'ro', markersize=6)

        ax_right.plot(
            [agent_position[0], food_position[0]],
            [agent_position[1], food_position[1]],
            linestyle=":", linewidth=1.5, alpha=0.7
        )

        arc_radius = 0.15
        ax_right.plot(
            [agent_position[0], agent_position[0] + arc_radius],
            [agent_position[1], agent_position[1]],
            linestyle="--", linewidth=1, alpha=0.5
        )
        arc_theta1, arc_theta2 = (0, theta) if theta >= 0 else (theta, 0)
        ax_right.add_patch(Arc(
            xy=agent_position, width=2*arc_radius, height=2*arc_radius,
            angle=0, theta1=arc_theta1, theta2=arc_theta2,
            linestyle="--", linewidth=1.5
        ))

        lim = 0.6
        ax_right.set_xlim(-lim, lim)
        ax_right.set_ylim(-lim, lim)
        ax_right.set_box_aspect(1)  
        ax_right.set_title(f'Position (θ={theta:.1f}°)', fontsize=9)
        ax_right.tick_params(labelsize=7)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title=None
    )

    if savefig:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig

def plot_multiple_angles_grid_one_shot(
    data_list,
    k_list=[1, 2, 3, 5, 7, 10, 15, 20],
    figsize=(20, 20),
    savefig=False,
    filename="datagrid.png",
    title=None
):
    """
    This is the same as the plot_multiple_angles_grid function but for the one shot experiments where we vary K instead of the learning rate,
    as in the trueOneShot network (except for the multiplier version), there is not really a notion of learning rate.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if title:
        fig.suptitle(title)
    outer = fig.add_gridspec(4, 4, wspace=0.2, hspace=0.25)

    data_to_plot = data_list[:16]

    for i, angle_data in enumerate(data_to_plot):
        row, col = divmod(i, 4)

        inner = outer[row, col].subgridspec(
            1, 2,
            wspace=0.1,  
            width_ratios=[1, 1]  
        )

        ax_left  = fig.add_subplot(inner[0, 0])
        ax_right = fig.add_subplot(inner[0, 1])

        theta = float(angle_data["theta0"])
        means = [r[0] for r in angle_data["total_rewards"]]
        stds  = [r[1] for r in angle_data["total_rewards"]]

        ax_left.errorbar(k_list, means, yerr=stds, fmt='-o', markersize=4)
        ax_left.set_xlabel('K Value', fontsize=8)
        ax_left.set_ylabel('Total Rewards', fontsize=8)
        ax_left.set_title(f'Rewards vs K (θ={theta:.1f}°)', fontsize=9)
        ax_left.axhline(y=1.5, color='r', linestyle='--', alpha=0.7)
        ax_left.tick_params(labelsize=7)

        agent_position = np.array(angle_data["agent_position"], dtype=float)
        food_position  = np.array(angle_data["food_position"], dtype=float)

        try:
            add_icon(ax_right, "icons/person_icon.png", agent_position, zoom=0.04)
            add_icon(ax_right, "icons/apple_icon.png",  food_position,  zoom=0.02)
        except Exception:
            ax_right.plot(agent_position[0], agent_position[1], 'ko', markersize=6, label='Agent')
            ax_right.plot(food_position[0],  food_position[1],  'ro', markersize=6, label='Food')

        ax_right.plot(
            [agent_position[0], food_position[0]],
            [agent_position[1], food_position[1]],
            linestyle=":", linewidth=1.5, alpha=0.7
        )

        arc_radius = 0.15
        ax_right.plot(
            [agent_position[0], agent_position[0] + arc_radius],
            [agent_position[1], agent_position[1]],
            linestyle="--", linewidth=1, alpha=0.5
        )
        arc_theta1, arc_theta2 = (0, theta) if theta >= 0 else (theta, 0)
        ax_right.add_patch(Arc(
            xy=agent_position, width=2*arc_radius, height=2*arc_radius,
            angle=0, theta1=arc_theta1, theta2=arc_theta2,
            linestyle="--", linewidth=1.5
        ))

        lim = 0.6
        ax_right.set_xlim(-lim, lim)
        ax_right.set_ylim(-lim, lim)
        ax_right.set_aspect('equal')  
        ax_right.set_title(f'Position (θ={theta:.1f}°)', fontsize=9)
        ax_right.tick_params(labelsize=7)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    if savefig:
        plt.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def _key_from_entry(entry, decimals=3):
    """Stable matching key: (rounded theta0, rounded food position)."""
    theta = round(float(entry["theta0"]), decimals)
    food = tuple(float(x) for x in entry["food_position"])
    food = tuple(round(x, decimals) for x in food)
    return (theta, food)

def _build_index(data, decimals=3):
    """Map key -> entry for quick matching."""
    index = {}
    for d in data:
        index[_key_from_entry(d, decimals=decimals)] = d
    return index

def plot_multiple_angles_grid_one_shot_compare(
    data_A,
    data_B,
    label_A="Set A",
    label_B="Set B",
    k_list=[1, 2, 3, 5, 7, 10, 15, 20],
    figsize=(20, 20),
    savefig=False,
    filename="datagrid_compare.png",
    title=None,
    max_cells=16,
):
    """
    This is the same function as before, but with the possibility of plotting 2 datasets in the same plot, for comparison.
    This was mainly used in pair with the entropyModulation module, to compare the results with and without entropy modulation.
    It matches the angles and food positions between the two datasets, and only plots those that are common to both.
    """
    idx_A = _build_index(data_A, decimals=3)
    idx_B = _build_index(data_B, decimals=3)
    common_keys = sorted(
        set(idx_A.keys()).intersection(idx_B.keys()),
        key=lambda k: (k[0], k[1]) 
    )

    if not common_keys:
        raise ValueError("No matching angles/food positions between the two datasets (after rounding).")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if title:
        fig.suptitle(title)
    outer = fig.add_gridspec(4, 4, wspace=0.2, hspace=0.25)

    keys_to_plot = common_keys[:max_cells]

    for i, key in enumerate(keys_to_plot):
        row, col = divmod(i, 4)
        inner = outer[row, col].subgridspec(
            1, 2,
            wspace=0.1,
            width_ratios=[1, 1]
        )
        ax_left  = fig.add_subplot(inner[0, 0])
        ax_right = fig.add_subplot(inner[0, 1])

        entry_A = idx_A[key]
        entry_B = idx_B[key]

        theta = float(entry_A["theta0"])  
        means_A = [r[0] for r in entry_A["total_rewards"]]
        stds_A  = [r[1] for r in entry_A["total_rewards"]]

        means_B = [r[0] for r in entry_B["total_rewards"]]
        stds_B  = [r[1] for r in entry_B["total_rewards"]]

        L = min(len(k_list), len(means_A), len(means_B), len(stds_A), len(stds_B))
        kk = k_list[:L]

        ax_left.errorbar(kk, means_A[:L], yerr=stds_A[:L], fmt='-o', markersize=4, label=label_A)
        ax_left.errorbar(kk, means_B[:L], yerr=stds_B[:L], fmt='-s', markersize=4, label=label_B)

        ax_left.axhline(y=1.5, color='r', linestyle='--', alpha=0.7)
        ax_left.set_xlabel('K Value', fontsize=8)
        ax_left.set_ylabel('Total Rewards', fontsize=8)
        ax_left.set_title(f'Rewards vs K (θ={theta:.1f}°)', fontsize=9)
        ax_left.tick_params(labelsize=7)
        ax_left.legend(fontsize=6, loc='lower right', frameon=False)

        agent_position = np.array(entry_A["agent_position"], dtype=float)
        food_position  = np.array(entry_A["food_position"], dtype=float)

        try:
            add_icon(ax_right, "icons/person_icon.png", agent_position, zoom=0.04)
            add_icon(ax_right, "icons/apple_icon.png",  food_position,  zoom=0.02)
        except Exception:
            ax_right.plot(agent_position[0], agent_position[1], 'ko', markersize=6, label='Agent')
            ax_right.plot(food_position[0],  food_position[1],  'ro', markersize=6, label='Food')

        ax_right.plot(
            [agent_position[0], food_position[0]],
            [agent_position[1], food_position[1]],
            linestyle=":", linewidth=1.5, alpha=0.7
        )

        arc_radius = 0.15
        ax_right.plot(
            [agent_position[0], agent_position[0] + arc_radius],
            [agent_position[1], agent_position[1]],
            linestyle="--", linewidth=1, alpha=0.5
        )
        arc_theta1, arc_theta2 = (0, theta) if theta >= 0 else (theta, 0)
        ax_right.add_patch(Arc(
            xy=agent_position, width=2*arc_radius, height=2*arc_radius,
            angle=0, theta1=arc_theta1, theta2=arc_theta2,
            linestyle="--", linewidth=1.5
        ))

        lim = 0.6
        ax_right.set_xlim(-lim, lim)
        ax_right.set_ylim(-lim, lim)
        ax_right.set_aspect('equal')
        ax_right.set_title(f'Position (θ={theta:.1f}°)', fontsize=9)
        ax_right.tick_params(labelsize=7)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    if savefig:
        plt.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_one_angle_comparison(
    angle_data_list,
    lr_list=[0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10],
    labels=("Gradient", "Reservoir", "Res×Multiplier", "One-Shot×Multiplier"),
    colors=None,
    title=None,
    savefig=False,
    filename="oneshot_one_angle_comparison.png",
    max_reward_line=1.5
):
    """
    Plot a single angle with four methods overlaid:
      - Left panel: Total Rewards vs Learning Rate (log-x), each with a mean curve and ±1σ band.
      - Right panel: Agent/Food positions and angle arc (taken from the first dataset).

    Parameters
    ----------
    angle_data_list : list[dict]
        List of FOUR angle_data dicts (same schema as your single-angle function).
        Each dict must contain:
            - "agent_position": [x, y]
            - "food_position" : [x, y]
            - "theta0"        : float (kept as-is, consistent with your function)
            - "total_rewards" : list of (mean, std) per LR
    lr_list : list[float]
        Learning rate values corresponding to total_rewards entries.
    labels : tuple[str, str, str, str]
        Legend labels for the four curves.
    colors : list[str] | None
        Optional list of 4 Matplotlib color specs. If None, uses tab10 cycle.
    title : str | None
    savefig : bool
    filename : str
    max_reward_line : float
        Horizontal reference line on the rewards plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Arc

    assert len(angle_data_list) == 4, "Provide exactly four angle_data dicts."
    if colors is None:
        colors = [plt.get_cmap("tab10")(i) for i in range(4)]

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, 0])  
    ax1 = fig.add_subplot(gs[0, 1])  

    if title:
        fig.suptitle(title)

    ref = angle_data_list[0]
    agent_position = np.array(ref["agent_position"], dtype=float)
    food_position  = np.array(ref["food_position"],  dtype=float)
    theta          = float(ref["theta0"])

    add_icon(ax1, "icons/person_icon.png", agent_position, zoom=0.08)
    add_icon(ax1, "icons/apple_icon.png",  food_position,  zoom=0.04)

    path_line, = ax1.plot(
        [agent_position[0], food_position[0]],
        [agent_position[1], food_position[1]],
        linestyle=":", linewidth=2, label="Path"
    )

    arc_radius = 0.22
    theta_deg  = theta  
    ax1.plot(
        [agent_position[0], agent_position[0] + arc_radius],
        [agent_position[1], agent_position[1]],
        linestyle="--", linewidth=1, alpha=0.5
    )

    arc_theta1, arc_theta2 = (theta_deg, 0) if theta_deg < 0 else (0, theta_deg)
    arc = Arc(
        xy=agent_position, width=2*arc_radius, height=2*arc_radius,
        angle=0, theta1=arc_theta1, theta2=arc_theta2,
        linestyle="--", linewidth=2
    )
    ax1.add_patch(arc)

    label_point = agent_position + arc_radius * np.array([np.cos(theta), np.sin(theta)])
    ax1.text(label_point[0], label_point[1], f"{theta_deg:.1f}°", ha="left", va="bottom", fontsize=11)

    lim = 0.6
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_title('Agent and Food Positions')
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_box_aspect(1)

    agent_proxy = Line2D([0],[0], marker='o', color='none', markerfacecolor='k', label='Agent')
    food_proxy  = Line2D([0],[0], marker='o', color='none', markerfacecolor='r', label='Food')
    ax1.legend(handles=[agent_proxy, food_proxy, path_line], loc='upper left')

    for i, angle_data in enumerate(angle_data_list):
        vals = angle_data["total_rewards"]
        means = np.array([r[0] for r in vals], dtype=float)
        stds  = np.array([r[1] for r in vals], dtype=float)

        n = min(len(lr_list), len(means), len(stds))
        x = np.array(lr_list[:n], dtype=float)
        m = means[:n]
        s = stds[:n]

        ax0.plot(x, m, marker='o', linewidth=2, label=labels[i], color=colors[i])
        ax0.fill_between(x, m - s, m + s, alpha=0.2, edgecolor='none', color=colors[i])

    ax0.set_xscale('log')
    ax0.set_xlabel('Learning Rate', fontsize=12)
    ax0.set_ylabel('Total Rewards', fontsize=12)
    ax0.set_title('Total Rewards vs Learning Rate')
    ax0.tick_params(axis='both', labelsize=14)
    ax0.axhline(y=max_reward_line, color='r', linestyle='--', linewidth=1.5, label='Max Reward')
    ax0.legend()
    ax0.grid(True, which='both', linestyle=':', alpha=0.35)

    if savefig:
        fig.savefig(filename, dpi=300)
    else:
        plt.show()

    return fig, (ax0, ax1)

def microscope_compare():
    """Imports two datasets and produces the comparison plot between the data produced with and without entropy modulation."""
    file_A = "true_one_shot_no_entropy/one_shot_meta_inference_results_without_entropy.json"
    file_B = "true_one_shot_entropy/one_shot_meta_inference_results.json"

    print("Loading datasets...")
    data_A = read_json_outfile(file_A)
    data_B = read_json_outfile(file_B)
    print(f"A entries: {len(data_A)} | B entries: {len(data_B)}")

    print("A keys:", data_A[0].keys())
    print("A theta0:", data_A[0]["theta0"])
    print("A agent_position:", data_A[0]["agent_position"])
    print("A food_position:", data_A[0]["food_position"])
    print("A total_rewards (len):", len(data_A[0]["total_rewards"]))

    plot_multiple_angles_grid_one_shot_compare(
        data_A, data_B,
        label_A="Without Entropy",
        label_B="With Entropy",
        figsize=(34, 19.5),
        savefig=True,
        filename="all_data_gradient_compare.png",
        title="All Data: A vs B (Overlapping Curves)",
    )

def microscope():
    """Mostly a testing function to load a dataset and print some of its contents, as well as plot it."""
    data = read_json_outfile("pure_grad_data/one_shot_gradient_results.json")
    # data = read_json_outfile("res_predicted_grad_data/one_shot_gradient_results.json")
    # data = read_json_outfile("true_one_shot\one_shot_meta_inference_results.json")

    print("Testing loaded data:")
    print(len(data))

    print(data[0].keys())

    print(data[0]['theta0']) 
    print(data[0]['agent_position'])  
    print(data[0]['food_position'])  
    print(data[0]['total_rewards'])  
    #plot_single_angle_one_shot(data[2])
    plot_multiple_angles_grid(data, figsize=(34, 19.5), savefig=True, filename="all_data_gradient.png", title="All Data Gradient")

def main():
    """
    The main produces all of the main figures used in the presentation, from the plot multiple angles grid (producing a figure of 
    16 individual angles in a 4x4 grid) for each datafile imported, to plotting the single angle figures for angles 0 and 2 (chosen arbitrarily, but representative)
    for each datafile as well.  It saves all of these figures in the final_pres_all_angles folder.
    """
    data_grad_only = read_json_outfile("one_shot_gradient_results.json")
    data_res_only = read_json_outfile("one_shot_gradient_res.json")
    data_res_multiplier = read_json_outfile("one_shot_gradient_res_multiplier.json")
    data_true_one_shot = read_json_outfile("one_shot_meta_inference_results_without_entropy.json")
    data_one_shot_multiplier = read_json_outfile("one_shot_meta_inference_multiplier.json")

    plot_multiple_angles_grid(data_res_only, figsize=(34, 19.5), savefig=True, filename="final_pres_all_angles/all_data_reservoir.png", title="All Data Reservoir")

    plot_multiple_angles_grid(data_res_multiplier, figsize=(34, 19.5), savefig=True, filename="final_pres_all_angles/all_data_reservoir_multiplier.png", title="All Data Reservoir Multiplier")

    plot_multiple_angles_grid_one_shot(data_true_one_shot, figsize=(34, 19.5), savefig=True, filename="final_pres_all_angles/all_data_true_one_shot.png", title="All Data True One-Shot")

    plot_multiple_angles_grid(data_one_shot_multiplier, figsize=(34, 19.5), savefig=True, filename="final_pres_all_angles/all_data_one_shot_multiplier.png", title="All Data One-Shot Multiplier")

    plot_multiple_angles_grid(data_grad_only, figsize=(34, 19.5), savefig=True, filename="final_pres_all_angles/all_data_gradient.png", title="All Data Gradient")

    plot_single_angle(data_grad_only[0], savefig=True, filename="final_pres_all_angles/one_shot_gradient_angle_0.png", title="One-Shot Gradient Angle 0")
    plot_single_angle(data_grad_only[2], savefig=True, filename="final_pres_all_angles/one_shot_gradient_angle_2.png", title="One-Shot Gradient Angle 2")

    plot_single_angle(data_res_only[0], savefig=True, filename="final_pres_all_angles/one_shot_reservoir_angle_0.png", title="One-Shot Reservoir Angle 0")
    plot_single_angle(data_res_only[2], savefig=True, filename="final_pres_all_angles/one_shot_reservoir_angle_2.png", title="One-Shot Reservoir Angle 2")

    plot_single_angle(data_res_multiplier[0], savefig=True, filename="final_pres_all_angles/one_shot_one_shot_multiplier_angle_0.png", title="One-Shot Res Multiplier Angle 0")
    plot_single_angle(data_res_multiplier[2], savefig=True, filename="final_pres_all_angles/one_shot_one_shot_multiplier_angle_2.png", title="One-Shot Res Multiplier Angle 2")

    plot_single_angle(data_one_shot_multiplier[0], savefig=True, filename="final_pres_all_angles/one_shot_true_one_shot_angle_0.png", title="One-Shot True One-Shot Multiplier Angle 0")
    plot_single_angle(data_one_shot_multiplier[2], savefig=True, filename="final_pres_all_angles/one_shot_true_one_shot_angle_2.png", title="One-Shot True One-Shot Multiplier Angle 2")

    plot_single_angle_one_shot(data_true_one_shot[0], savefig=True, filename="final_pres_all_angles/one_shot_true_one_shot_angle_0_one_shot.png", title="One-Shot True One-Shot Angle 0")
    plot_single_angle_one_shot(data_true_one_shot[2], savefig=True, filename="final_pres_all_angles/one_shot_true_one_shot_angle_2_one_shot.png", title="One-Shot True One-Shot Angle 2")

def main2():
    """
    This main function produces the comparison plots between all four methods, both for a single angle (angle 10, arbitrarily chosen) and for the full 4x4 grid of angles.
    It saves these figures in the final_pres_all_angles folder.
    """
    data_grad_only = read_json_outfile("one_shot_gradient_results.json")
    data_res_only = read_json_outfile("one_shot_gradient_res.json")
    data_res_multiplier = read_json_outfile("one_shot_gradient_res_multiplier.json")
    data_true_one_shot = read_json_outfile("one_shot_meta_inference_results_without_entropy.json")
    data_one_shot_multiplier = read_json_outfile("one_shot_meta_inference_multiplier.json")

    ang = 10
    fig, axes = plot_one_angle_comparison(
    [
        data_grad_only[ang],        # angle 0 from Gradient
        data_res_only[ang],         # angle 0 from Reservoir
        data_res_multiplier[ang],   # angle 0 from Res×Multiplier
        data_one_shot_multiplier[ang]  # angle 0 from One-Shot×Multiplier
    ],
    labels=("Gradient", "Reservoir", "Res×Mult", "One-Shot×Mult"),
    title="Angle 10 — Methods Comparison",
    savefig=True,
    filename="final_pres_all_angles/angle10_methods_comparison.png"
    )

    plot_multiple_angles_grid_comparison(
        [
            data_grad_only,
            data_res_only,
            data_res_multiplier,
            data_one_shot_multiplier
        ],
        labels=("Gradient", "Reservoir", "Res×Mult", "One-Shot×Mult"),
        title="Methods Comparison",
        savefig=True,
        filename="final_pres_all_angles/methods_comparison.png"
    )

if __name__ == "__main__":
    main2()