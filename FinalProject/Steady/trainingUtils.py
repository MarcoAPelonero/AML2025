import numpy as np
from tqdm import tqdm

def episode(agent, env, time_steps=30):
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
        agent.update_weights(agent_position.flatten(), action, reward)
    return reward, np.array(traj)

def train_episode_accumulation(agent, env, time_steps = 30):
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
        agent.accumulate_gradients(agent_position.flatten(), action, reward)
    agent.apply_gradients()
    return reward, np.array(traj)

def train_accumulation(agent, env, episodes=100, time_steps=30, verbose=False):
    rewards = []
    trajectories = []
    for episode in range(episodes):
        reward, traj = train_episode_accumulation(agent, env, time_steps)
        max_length = time_steps
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)  # Use np.nan for padding
        padded_traj[:traj.shape[0], :] = traj
        trajectories.append(padded_traj)
        rewards.append(reward)
        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")
    return rewards, trajectories

def train(agent, env, episodes=100, time_steps=30, verbose=False):
    rewards = []
    trajectories = []
    for episode in range(episodes):
        reward, traj = train_episode(agent, env, time_steps)
        max_length = time_steps
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)  # Use np.nan for padding
        padded_traj[:traj.shape[0], :] = traj
        rewards.append(reward)
        trajectories.append(padded_traj)
        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")
    return rewards, trajectories

def InDistributionTraining(agent, env, rounds = 1, episodes = 600, time_steps = 30, mode = 'normal', verbose = False):

    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()

        if mode == 'normal':
            rewards, trajectories = train(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose)
        elif mode == 'accumulation':
            rewards, trajectories = train_accumulation(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(totalRewards), totalTrajectories

def OutOfDistributionTraining(agent, env, rounds = 1, episodes = 600, time_steps = 30, mode = 'normal', verbose = False):

    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    n_resets = 16 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45 / 2 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()

        if mode == 'normal':
            rewards, trajectories = train(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose)
        elif mode == 'accumulation':
            rewards, trajectories = train_accumulation(agent, env, episodes=episodes, time_steps=time_steps, verbose=verbose)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(totalRewards), totalTrajectories


def test_a_single_run():
    from environment import Environment
    from agent import LinearAgent

    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4

    learning_rate = 0.03
    temperature = 1.0

    agent = LinearAgent(input_dim, output_dim, learning_rate=learning_rate, temperature=temperature)
    env = Environment(grid_size=spatial_res, sigma=0.2)

    rewards, trajectories = train(agent, env, episodes=600, time_steps=30, verbose=False)
    from plottingUtils import plot_single_run
    plot_single_run(np.array(rewards), bin_size=30)

def main():
    from environment import Environment
    from agent import LinearAgent  

    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4

    learning_rate = 0.01
    temperature = 1.0

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.04, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    rewards, trajectories = InDistributionTraining(agent, env, rounds=2, episodes=600, time_steps=30, mode='accumulation', verbose=False)
    print(rewards.shape)
    print(trajectories[0]['trajectory'].shape)
    print("Training complete.")

    from plottingUtils import plot_rewards, plot_trajectories
    plot_rewards(rewards) 
    plot_trajectories(trajectories)

if __name__ == "__main__":
    main()