import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.patches import Arc
from matplotlib.lines import Line2D

def read_json_outfile(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def add_icon(ax, img_path, pos, zoom=0.1):
    img = mpimg.imread(img_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, pos, frameon=False)
    ax.add_artist(ab)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.patches import Arc
from matplotlib.lines import Line2D

def add_icon(ax, img_path, xy, zoom=0.1):
    img = mpimg.imread(img_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)

def plot_single_angle(angle_data, lr_list=[0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10]):
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))

    agent_position = np.array(angle_data["agent_position"], dtype=float)
    food_position  = np.array(angle_data["food_position"], dtype=float)
    theta          = float(angle_data["theta0"])  

    add_icon(ax[1], "icons/person_icon.png", agent_position, zoom=0.08)
    add_icon(ax[1], "icons/apple_icon.png",  food_position,  zoom=0.04)

    path_line, = ax[1].plot(
        [agent_position[0], food_position[0]],
        [agent_position[1], food_position[1]],
        linestyle=":", linewidth=2, label="Path"
    )

    arc_radius = 0.22  
    theta_deg  = theta

    ax[1].plot(
        [agent_position[0], agent_position[0] + arc_radius],
        [agent_position[1], agent_position[1]],
        linestyle="--", linewidth=1, alpha=0.5
    )

    arc_theta1 = 0
    arc_theta2 = theta_deg
    if arc_theta2 < arc_theta1:
        arc_theta1, arc_theta2 = arc_theta2, arc_theta1

    arc = Arc(
        xy=agent_position, width=2*arc_radius, height=2*arc_radius,
        angle=0, theta1=arc_theta1, theta2=arc_theta2,
        linestyle="--", linewidth=2
    )
    ax[1].add_patch(arc)

    label_point = agent_position + arc_radius * np.array([np.cos(theta), np.sin(theta)])
    ax[1].text(
        label_point[0], label_point[1],
        f"{theta_deg:.1f}°",
        ha="left", va="bottom", fontsize=11
    )

    lim = 0.6
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_aspect('equal')
    ax[1].set_title('Agent and Food Positions')

    agent_proxy = Line2D([0], [0], marker='o', color='none', markerfacecolor='k', label='Agent')
    food_proxy  = Line2D([0], [0], marker='o', color='none', markerfacecolor='r', label='Food')
    ax[1].legend(handles=[agent_proxy, food_proxy, path_line], loc='upper left')

    means = [r[0] for r in angle_data["total_rewards"]]
    stds  = [r[1] for r in angle_data["total_rewards"]]

    ax[0].errorbar(lr_list, means, yerr=stds, fmt='-o')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Learning Rate')
    ax[0].set_ylabel('Total Rewards')
    ax[0].set_title('Total Rewards vs Learning Rate')

    ax[0].axhline(y=1.5, color='r', linestyle='--', label='Max Reward')
    ax[0].legend()

    plt.tight_layout()
    plt.show()

def plot_single_angle_one_shot(angle_data, k_list = [1, 2, 3, 5, 7, 10, 15, 20]):
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))

    agent_position = np.array(angle_data["agent_position"], dtype=float)
    food_position  = np.array(angle_data["food_position"], dtype=float)
    theta          = float(angle_data["theta0"])  

    add_icon(ax[1], "icons/person_icon.png", agent_position, zoom=0.08)
    add_icon(ax[1], "icons/apple_icon.png",  food_position,  zoom=0.04)

    path_line, = ax[1].plot(
        [agent_position[0], food_position[0]],
        [agent_position[1], food_position[1]],
        linestyle=":", linewidth=2, label="Path"
    )

    arc_radius = 0.22  
    theta_deg  = theta

    ax[1].plot(
        [agent_position[0], agent_position[0] + arc_radius],
        [agent_position[1], agent_position[1]],
        linestyle="--", linewidth=1, alpha=0.5
    )

    arc_theta1 = 0
    arc_theta2 = theta_deg
    if arc_theta2 < arc_theta1:
        arc_theta1, arc_theta2 = arc_theta2, arc_theta1

    arc = Arc(
        xy=agent_position, width=2*arc_radius, height=2*arc_radius,
        angle=0, theta1=arc_theta1, theta2=arc_theta2,
        linestyle="--", linewidth=2
    )
    ax[1].add_patch(arc)

    label_point = agent_position + arc_radius * np.array([np.cos(theta), np.sin(theta)])
    ax[1].text(
        label_point[0], label_point[1],
        f"{theta_deg:.1f}°",
        ha="left", va="bottom", fontsize=11
    )

    lim = 0.6
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_aspect('equal')
    ax[1].set_title('Agent and Food Positions')

    agent_proxy = Line2D([0], [0], marker='o', color='none', markerfacecolor='k', label='Agent')
    food_proxy  = Line2D([0], [0], marker='o', color='none', markerfacecolor='r', label='Food')
    ax[1].legend(handles=[agent_proxy, food_proxy, path_line], loc='upper left')

    means = [r[0] for r in angle_data["total_rewards"]]
    stds  = [r[1] for r in angle_data["total_rewards"]]

    ax[0].errorbar(k_list, means, yerr=stds, fmt='-o')
    ax[0].set_xlabel('K Value')
    ax[0].set_ylabel('Total Rewards')
    ax[0].set_title('Total Rewards vs K Value')

    ax[0].axhline(y=1.5, color='r', linestyle='--', label='Max Reward')
    ax[0].legend()

    plt.tight_layout()
    plt.show()

def microscope():
    data = read_json_outfile("pure_grad_data/one_shot_gradient_results.json")
    data = read_json_outfile("res_predicted_grad_data/one_shot_gradient_results.json")
    data = read_json_outfile("true_one_shot\one_shot_meta_inference_results.json")

    print("Testing loaded data:")
    print(len(data))

    print(data[0].keys())

    print(data[0]['theta0'])  # Angle at which the environment was reset
    print(data[0]['agent_position'])  # Initial position of the agent
    print(data[0]['food_position'])  # Initial position of the food
    print(data[0]['total_rewards'])  
    plot_single_angle_one_shot(data[2])
    # plot_single_angle(data[2])

if __name__ == "__main__":
    microscope()