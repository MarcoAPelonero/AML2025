import numpy as np
from tqdm import tqdm

"""
This is the basis for all the other training modules in the repo, this handles everything from performing a single episode 
without training with a fixed weight agent, to all the utilies to a complete training session on ID and OOD angles. There are not 
really any differences between the two as long as we don't use a reservoir, but for convention we refer as ID angles
as multiples of 45 degrees and OOD as multiples of 22.5 degrees.
"""

def episode(agent, env, time_steps=30):
    """
    Handles an episode without touching the agent's weights, so this is for an already trained agent.
    Performs all the basic actions, which include getting the encoded position from the environment, using it to determine
    the policy trough softmax (multiplication with the agent's weights and softmax), sampling an action from the policy,
    and then performing the action in the environment, which returns a reward and a done signal.
    """
    env.reset_inner()
    done = False
    time = 0
    traj = []
    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action, _ = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
    return reward, np.array(traj)

def train_episode(agent, env, time_steps=30):
    """
    Handles a single training episode. This is the exact same as episode, with the added agent.update_weights call
    which updates the agent's weights using the REINFORCE algorithm after every action.
    """
    env.reset_inner()
    done = False
    time = 0
    traj = [env.agent_position.copy()]
    while not done and time < time_steps:
        agent_position = env.encoded_position
        action, _ = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        traj.append(env.agent_position.copy())
        agent.update_weights(agent_position.flatten(), action, reward)
        time += 1
    return reward, np.array(traj)

def train_episode_accumulation(agent, env, time_steps = 30):
    """
    Handles a single training episode. This is the exact same as episode, with the added agent.accumulate_gradients call
    which accumulates the gradients using the REINFORCE algorithm after every action, and then applies them at the end of the episode.
    """
    env.reset_inner()
    done = False
    time = 0
    traj = [env.agent_position.copy()]
    while not done and time < time_steps:
        agent_position = env.encoded_position
        action, _ = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        traj.append(env.agent_position.copy())
        agent.accumulate_gradients(agent_position.flatten(), action, reward)
        time += 1
    agent.apply_gradients()
    return reward, np.array(traj)

def train(agent, env, episodes=100, time_steps=30, verbose=False, return_weights=False):
    """
    Handles the training process over multiple episodes, for the same kind of environment (same angle),
    by launching a episodes time the train_episode function. It collects rewards and trajectories
    from each episode, and returns them as numpy arrays. The agent's parameters are reset at the beginning of the training process,
    since we don't want to carry over any learned parameters from previous angles.
    """
    rewards = []
    trajectories = []
    weights = []
    for episode in range(episodes):
        reward, traj = train_episode(agent, env, time_steps)
        max_length = time_steps + 1
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)  # Use np.nan for padding
        padded_traj[:traj.shape[0], :] = traj
        rewards.append(reward)
        trajectories.append(padded_traj)
        weights.append(agent.weights.copy())
        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    if not return_weights:
        return rewards, trajectories
    else:
        return rewards, trajectories, weights

def train_accumulation(agent, env, episodes=100, time_steps=30, verbose=False, return_weights=False):
    """
    Same as the precedent function, but uses the train_episode_accumulation function instead of train_episode,
    so the gradients are accumulated over the episode and applied at the end.
    """
    rewards = []
    trajectories = []
    weights = []
    for episode in range(episodes):
        reward, traj = train_episode_accumulation(agent, env, time_steps)
        max_length = time_steps + 1
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)  
        padded_traj[:traj.shape[0], :] = traj
        trajectories.append(padded_traj)
        rewards.append(reward)
        weights.append(agent.weights.copy())
        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")
    if not return_weights:
        return rewards, trajectories
    return rewards, trajectories, weights

def InDistributionTraining(agent, env, rounds = 1, episodes = 600, time_steps = 30, mode = 'normal', verbose = False, return_weights=False):
    """
    This function handles the complete training process over multiple angles (rounds), where each angle is
    a multiple of 45 degrees. It resets the environment and agent's parameters for each angle,
    and uses either the normal training or accumulation mode based on the mode parameter.
    It collects rewards and trajectories for each angle, and returns them as numpy arrays."""
    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []
    totalWeights = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()

        if mode == 'normal':
            if return_weights:
                rewards, trajectories, weights = train(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
            else:
                rewards, trajectories = train(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
        elif mode == 'accumulation':
            if return_weights:
                rewards, trajectories, weights = train_accumulation(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
            else:
                rewards, trajectories = train_accumulation(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})
        totalWeights.append(weights)

    if return_weights:
        return np.array(totalRewards), totalTrajectories, np.array(totalWeights)
    return np.array(totalRewards), totalTrajectories

def OutOfDistributionTraining(agent, env, rounds = 1, episodes = 600, time_steps = 30, mode = 'normal', verbose = False, return_weights=False):
    """
    Same as the precedent function, but this time the angles are multiples of 22.5 degrees,
    """
    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    n_resets = 16 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []
    totalWeights = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45 / 2 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()

        if mode == 'normal':
            if return_weights:
                rewards, trajectories, weights = train(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
            else:
                rewards, trajectories = train(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
        elif mode == 'accumulation':
            if return_weights:
                rewards, trajectories, weights = train_accumulation(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
            else:
                rewards, trajectories = train_accumulation(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose, return_weights=return_weights)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})
        totalWeights.append(weights)

    if return_weights:
        return np.array(totalRewards), totalTrajectories, np.array(totalWeights)
    return np.array(totalRewards), totalTrajectories

def generate_angle_gifs_with_food(agent,
                                  env,
                                  angles=None,
                                  episodes=600,
                                  time_steps=30,
                                  mode='normal',
                                  output_dir='angle_gifs',
                                  interval=40,
                                  frame_skip=10,
                                  verbose=False):
    """
    Run training for selected angles and save GIFs that combine the weight animation
    with a static subplot showing the agent and food configuration.
    """
    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    from matplotlib.lines import Line2D

    from presentationPlottingUtils import add_icon
    from agent import animate_weights

    if angles is None:
        angles = [45 * i for i in range(4)]
    else:
        angles = list(angles)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    icons_dir = Path(__file__).resolve().parent / "icons"
    agent_icon = icons_dir / "person_icon.png"
    food_icon = icons_dir / "apple_icon.png"

    if not agent_icon.exists() or not food_icon.exists():
        raise FileNotFoundError("Expected icon assets in the 'icons' directory.")

    train_fn = train if mode == 'normal' else train_accumulation
    saved_paths = []
    agent_start = np.zeros(2, dtype=float)

    for idx, theta in enumerate(angles):
        theta = float(theta)
        env.reset(theta)
        food_position = np.array(env.food_position, dtype=float)
        agent.reset_parameters()

        rewards, _, weights = train_fn(
            agent,
            env,
            episodes=episodes,
            time_steps=time_steps,
            verbose=verbose,
            return_weights=True
        )

        if verbose:
            print(f"Theta {theta:.1f}: avg reward {np.mean(rewards):.3f}")

        weights_array = np.asarray(weights, dtype=float)
        if weights_array.ndim != 3:
            raise ValueError("Weights history must have shape (episodes, output_dim, input_dim).")

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.4)
        ax_anim = fig.add_subplot(gs[0, 0], projection='3d')
        ax_food = fig.add_subplot(gs[0, 1])

        ax_food.clear()
        add_icon(ax_food, str(agent_icon), agent_start, zoom=0.08)
        add_icon(ax_food, str(food_icon), food_position, zoom=0.05)
        path_line, = ax_food.plot(
            [agent_start[0], food_position[0]],
            [agent_start[1], food_position[1]],
            linestyle=":",
            linewidth=2,
            color="#5a5a5a",
            label="Path"
        )

        arc_radius = 0.22
        if theta < 0:
            theta1, theta2 = theta, 0
        else:
            theta1, theta2 = 0, theta
        arc = Arc(agent_start,
                  width=2 * arc_radius,
                  height=2 * arc_radius,
                  angle=0,
                  theta1=theta1,
                  theta2=theta2,
                  linestyle="--",
                  linewidth=2)
        ax_food.add_patch(arc)

        theta_rad = np.deg2rad(theta)
        label_point = agent_start + arc_radius * np.array([np.cos(theta_rad), np.sin(theta_rad)])
        label_text = f"{theta:.1f}{chr(176)}"
        ax_food.text(label_point[0], label_point[1], label_text, ha="left", va="bottom", fontsize=11)

        lim = 0.6
        ax_food.set_xlim(-lim, lim)
        ax_food.set_ylim(-lim, lim)
        ax_food.set_box_aspect(1)
        ax_food.tick_params(axis='both', labelsize=12)
        ax_food.set_title("Agent and Food Positions")

        agent_proxy = Line2D([0], [0], marker='o', color='none', markerfacecolor='k', label='Agent')
        food_proxy = Line2D([0], [0], marker='o', color='none', markerfacecolor='r', label='Food')
        ax_food.legend(handles=[agent_proxy, food_proxy, path_line], loc='upper left')

        angle_string = f"{theta:.1f}".rstrip('0').rstrip('.')
        if not angle_string:
            angle_string = "0"
        slug = angle_string.replace('-', 'm').replace('.', 'p')
        gif_path = output_dir / f"weights_theta_{slug}_{idx:02d}.gif"

        animate_weights(
            weights_array,
            interval=interval,
            save_path=str(gif_path),
            dpi=120,
            frame_skip=frame_skip,
            fig=fig,
            ax=ax_anim
        )

        plt.close(fig)
        saved_paths.append(str(gif_path))

    return saved_paths


def test_a_single_run():
    """
    Testing of how everything comes together in a single run, and generation of four GIFs
    that combine weight evolution with the food-position subplot.
    """
    from environment import Environment
    from agent import LinearAgent
    from plottingUtils import plot_single_run

    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4

    learning_rate = 0.03
    temperature = 1.0

    agent = LinearAgent(input_dim, output_dim, learning_rate=learning_rate, temperature=temperature)
    env = Environment(grid_size=spatial_res, sigma=0.2)

    rewards, _, weights = train(
        agent,
        env,
        episodes=600,
        time_steps=30,
        verbose=False,
        return_weights=True
    )
    weights = np.array(weights)
    print(weights.shape)

    plot_single_run(np.array(rewards), bin_size=30)

    agent_for_gifs = LinearAgent(input_dim, output_dim, learning_rate=learning_rate, temperature=temperature)
    env_for_gifs = Environment(grid_size=spatial_res, sigma=0.2)

    gif_paths = generate_angle_gifs_with_food(
        agent_for_gifs,
        env_for_gifs,
        angles=[45 * i for i in range(4)],
        episodes=600,
        time_steps=30,
        mode='normal',
        output_dir='angle_gifs',
        interval=60,
        frame_skip=10,
        verbose=False
    )
    print("Saved GIFs with combined weight/food subplots:")
    for path in gif_paths:
        print(path)

def main():
    from environment import Environment
    from agent import LinearAgent  

    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.02, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    rewards, trajectories = InDistributionTraining(agent, env, rounds=2, episodes=600, time_steps=30, mode='normal', verbose=False)
    print(rewards.shape)
    print(trajectories[0]['trajectory'].shape)
    print("Training complete.")

    from plottingUtils import plot_rewards, plot_trajectories

    plot_rewards(rewards) 
    plot_trajectories(trajectories)

if __name__ == "__main__":
    test_a_single_run()
