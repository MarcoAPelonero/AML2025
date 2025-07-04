# main.py

import pygame
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from agent import REINFORCEAgent   # updated agent.py with get_action_and_activations
from game import BreakoutPongEnv   # your existing environment

# ...existing code...

def main(policy_file):
    episodes = 5
    env = BreakoutPongEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, num_envs=1)
    agent.load(policy_file)

    # Visualization setup (as before)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis('off')
    fig.suptitle("Policy‐Network Neurons Firing (pre‐ReLU)\nBlue=low, Red=high", fontsize=12)

    h1_size = agent.policy.net[1].out_features
    h2_size = agent.policy.net[3].out_features
    out_size = agent.policy.net[5].out_features

    layer1_pos = np.column_stack((np.full(h1_size, 0.0), np.linspace(0.0, 1.0, h1_size)))
    layer2_pos = np.column_stack((np.full(h2_size, 1.0), np.linspace(0.0, 1.0, h2_size)))
    layer3_pos = np.column_stack((np.full(out_size, 2.0), np.linspace(0.0, 1.0, out_size)))

    edges12 = []
    for j in range(h1_size):
        y1 = layer1_pos[j, 1]
        for i in range(h2_size):
            y2 = layer2_pos[i, 1]
            edges12.append([(0.0, y1), (1.0, y2)])

    edges23 = []
    for i in range(h2_size):
        y2 = layer2_pos[i, 1]
        for k in range(out_size):
            y3 = layer3_pos[k, 1]
            edges23.append([(1.0, y2), (2.0, y3)])

    lc12 = LineCollection(edges12, linewidths=0.5, colors="blue", alpha=0.4)
    lc23 = LineCollection(edges23, linewidths=0.5, colors="blue", alpha=0.4)
    ax.add_collection(lc12)
    ax.add_collection(lc23)

    sc1 = ax.scatter(layer1_pos[:, 0], layer1_pos[:, 1], s=40, edgecolors='k', linewidths=0.5)
    sc2 = ax.scatter(layer2_pos[:, 0], layer2_pos[:, 1], s=40, edgecolors='k', linewidths=0.5)
    sc3 = ax.scatter(layer3_pos[:, 0], layer3_pos[:, 1], s=80, edgecolors='k', linewidths=0.5)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    # --- Visualization toggle ---
    visualize = False
    print("Press 'v' to toggle live network visualization ON/OFF.")

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    plt.close(fig)
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        visualize = not visualize
                        print(f"Visualization {'ON' if visualize else 'OFF'}")

            # Get action + pre‐ReLU activations for each layer
            action, activations = agent.get_action_and_activations(obs)
            pre1 = activations[0].squeeze(0).numpy()
            pre2 = activations[1].squeeze(0).numpy()
            pre3 = activations[2].squeeze(0).numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(np.array([reward]))

            if visualize:
                # Update node (circle) colors
                colors1 = cmap(norm(pre1))
                colors2 = cmap(norm(pre2))
                colors3 = cmap(norm(pre3))
                sc1.set_facecolors(colors1)
                sc2.set_facecolors(colors2)
                sc3.set_facecolors(colors3)

                # Update edge colors
                edge_vals12 = []
                for j in range(h1_size):
                    for i in range(h2_size):
                        edge_vals12.append((pre1[j] + pre2[i]) / 2.0)
                edge_vals12 = np.array(edge_vals12)
                lc12.set_colors(cmap(norm(edge_vals12)))

                edge_vals23 = []
                for i in range(h2_size):
                    for k in range(out_size):
                        edge_vals23.append((pre2[i] + pre3[k]) / 2.0)
                edge_vals23 = np.array(edge_vals23)
                lc23.set_colors(cmap(norm(edge_vals23)))

                fig.canvas.restore_region(background)
                ax.draw_artist(sc1)
                ax.draw_artist(sc2)
                ax.draw_artist(sc3)
                ax.draw_artist(lc12)
                ax.draw_artist(lc23)
                fig.canvas.blit(ax.bbox)
                fig.canvas.flush_events()
                # plt.pause(0.001)

        agent.finish_rollout()
        print(f"Episode {ep+1} finished.")
        pygame.time.delay(1000)

    env.close()
    plt.close(fig)

# ...existing code...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Pong visualizer with artistic neuron‐firing display."
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="godlike.pth",
        help="Path to the policy file"
    )
    args = parser.parse_args()
    main(args.policy)
