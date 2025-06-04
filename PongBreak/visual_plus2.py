# main.py

import argparse
import numpy as np
import torch
import pygame
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm
import matplotlib.animation as animation

from agent import REINFORCEAgent   # assumes get_action_and_activations is implemented
from game import BreakoutPongEnv   # your existing environment


def record_episode(policy_file):
    """
    Plays out one full episode (game) without rendering to screen, recording
    per-step game frames (RGB arrays) and pre-ReLU activations from the network.
    Returns:
        frames: List of H×W×3 uint8 arrays
        activations: List of tuples (pre1, pre2, pre3), each a NumPy array
    """
    # Instantiate the environment in 'rgb_array' mode so we can grab frames
    env = BreakoutPongEnv(render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load agent
    agent = REINFORCEAgent(state_dim, action_dim, num_envs=1)
    agent.load(policy_file)

    frames = []
    activations = []

    obs, _ = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        # Get action + pre-ReLU activations
        action, acts = agent.get_action_and_activations(obs)
        pre1 = acts[0].squeeze(0).cpu().numpy()
        pre2 = acts[1].squeeze(0).cpu().numpy()
        pre3 = acts[2].squeeze(0).cpu().numpy()

        # Step the env
        obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_reward(np.array([reward]))

        # Grab the rendered frame (rgb array)
        frame = env.render()  # should return an H×W×3 uint8 array
        frames.append(frame)
        activations.append((pre1, pre2, pre3))

    env.close()
    return frames, activations


def animate_episode(frames, activations, save_path=None):
    """
    Given recorded frames and activations, animate side-by-side:
    - Left: game frames (imshow)
    - Right: network visualization with nodes and edges colored by activations
    Uses a tqdm progress bar over time steps.
    """
    num_steps = len(frames)
    if num_steps == 0:
        print("No frames recorded.")
        return

    # Determine network sizes from the first activation
    pre1_0, pre2_0, pre3_0 = activations[0]
    h1_size = pre1_0.shape[0]
    h2_size = pre2_0.shape[0]
    out_size = pre3_0.shape[0]

    # Precompute node positions
    layer1_pos = np.column_stack((np.full(h1_size, 0.0), np.linspace(0.0, 1.0, h1_size)))
    layer2_pos = np.column_stack((np.full(h2_size, 1.0), np.linspace(0.0, 1.0, h2_size)))
    layer3_pos = np.column_stack((np.full(out_size, 2.0), np.linspace(0.0, 1.0, out_size)))

    # Precompute edges between layers
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
    edges12 = np.array(edges12)
    edges23 = np.array(edges23)

    # Set up matplotlib figure with two subplots side by side
    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.1)

    # Left subplot: game frames
    ax_frame = fig.add_subplot(gs[0, 0])
    ax_frame.axis('off')
    im = ax_frame.imshow(frames[0])

    # Right subplot: network visualization
    ax_net = fig.add_subplot(gs[0, 1])
    ax_net.axis('off')
    ax_net.set_xlim(-0.5, 2.5)
    ax_net.set_ylim(-0.1, 1.1)
    ax_net.set_title("Policy‐Network Neurons Firing\nBlue=low, Red=high", fontsize=10)

    # Initial LineCollections (all edges start blue)
    lc12 = LineCollection(edges12, linewidths=0.5, colors="blue", alpha=0.4)
    lc23 = LineCollection(edges23, linewidths=0.5, colors="blue", alpha=0.4)
    ax_net.add_collection(lc12)
    ax_net.add_collection(lc23)

    # Initial scatter plots (nodes)
    sc1 = ax_net.scatter(layer1_pos[:, 0], layer1_pos[:, 1], s=40, edgecolors='k', linewidths=0.5)
    sc2 = ax_net.scatter(layer2_pos[:, 0], layer2_pos[:, 1], s=40, edgecolors='k', linewidths=0.5)
    sc3 = ax_net.scatter(layer3_pos[:, 0], layer3_pos[:, 1], s=80, edgecolors='k', linewidths=0.5)

    # Colormap setup
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    # Draw once to cache the background
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    # Animation tqdm bar
    for t in tqdm(range(num_steps), desc="Animating"):
        frame = frames[t]
        pre1, pre2, pre3 = activations[t]

        # Update game frame
        im.set_data(frame)

        # Update node colors
        colors1 = cmap(norm(pre1))
        colors2 = cmap(norm(pre2))
        colors3 = cmap(norm(pre3))
        sc1.set_facecolors(colors1)
        sc2.set_facecolors(colors2)
        sc3.set_facecolors(colors3)

        # Update edge colors based on average activation of endpoints
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

        # Restore background and redraw artists
        fig.canvas.restore_region(bg)
        ax_frame.draw_artist(im)
        ax_net.draw_artist(lc12)
        ax_net.draw_artist(lc23)
        ax_net.draw_artist(sc1)
        ax_net.draw_artist(sc2)
        ax_net.draw_artist(sc3)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

    # Second tqdm bar for "video creation" (simulate work)
    for _ in tqdm(range(num_steps), desc="Creating video"):
        plt.pause(0.001)  # Simulate processing time per frame

    # Keep the final frame on screen until closed
    plt.ioff()
    plt.show()

    def update(t):
        frame = frames[t]
        pre1, pre2, pre3 = activations[t]
        im.set_data(frame)
        colors1 = cmap(norm(pre1))
        colors2 = cmap(norm(pre2))
        colors3 = cmap(norm(pre3))
        sc1.set_facecolors(colors1)
        sc2.set_facecolors(colors2)
        sc3.set_facecolors(colors3)
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
        return [im, sc1, sc2, sc3, lc12, lc23]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), blit=False, interval=30, repeat=False
    )

    if save_path:
        print(f"Saving animation to {save_path} ...")
        # Make sure FFmpeg is installed and on your PATH
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=30)
        ani.save(save_path, writer=writer)
        print("Saved.")

    plt.ioff()
    plt.show()


def main(policy_file):
    # Record one full episode
    print("Recording episode (no live render)...")
    frames, activations = record_episode(policy_file)
    print(f"Recorded {len(frames)} steps. Now animating and saving video...")
    animate_episode(frames, activations, save_path="output.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play, record, and then visualize Pong policy network alongside game frames."
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="godlike.pth",
        help="Path to the trained policy file"
    )
    args = parser.parse_args()
    main(args.policy)
