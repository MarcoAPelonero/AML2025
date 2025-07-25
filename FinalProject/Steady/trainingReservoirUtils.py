import numpy as np
from tqdm import tqdm

def train_reservoir_episode(agent, env, reservoir, time_steps = 30):
    env.reset_inner()
    reservoir.reset_state()

    done = False
    time = 0
    traj = []
    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        agent.accumulate_gradients(agent_position.flatten(), action, reward)

        res_input = np.concatenate((agent_position.flatten(), env.encode(action, res=env.grid_size).flatten()))
        res_modulation = np.array(reward)

        reservoir_state = reservoir.update(res_input, res_modulation)

    grad_weights, grad_bias = agent.apply_gradients()

    grad_weights = grad_weights.flatten()
    grad_bias = grad_bias.flatten()

    gradients = np.concatenate((grad_weights, grad_bias))

    return reward, np.array(traj), gradients, reservoir_state

def train_reservoir(agent, env, reservoir, episodes=100, time_steps=30, verbose=False):
    rewards = []
    trajectories = []
    gradients_list = []
    reservoir_states = []

    for episode in range(episodes):
        reward, traj, gradients, reservoir_state = train_reservoir_episode(agent, env, reservoir, time_steps)
        max_length = time_steps
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)  # Use np.nan for padding
        padded_traj[:traj.shape[0], :] = traj
        trajectories.append(padded_traj)
        rewards.append(reward)
        gradients_list.append(gradients)
        reservoir_states.append(reservoir_state)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return rewards, trajectories, gradients_list, reservoir_states

def train_with_reservoir_episode(agent, env, reservoir, time_steps=30):
    env.reset_inner()
    reservoir.reset_state()

    done = False
    time = 0
    traj = []
    rewards = []
    gradient= []

    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        res_input = np.concatenate((agent_position.flatten(), env.encode(action, res=env.grid_size).flatten()))
        res_modulation = env.encode(reward, res=env.grid_size).flatten()
        grad = reservoir.step(res_input, res_modulation)
        # THis grad vector has shape 104, the first 100 are the weights, the last 4 are the bias
        gradient.append((grad[:100] * reward, grad[100:] * reward))

    weights_array = np.stack([gw for gw, _ in gradient])
    bias_array    = np.stack([gb for _, gb in gradient])

    total_grad_weights = np.sum(weights_array, axis=0)
    total_grad_bias    = np.sum(bias_array, axis=0)

    total_grad_weights = total_grad_weights.reshape(agent.weights.shape)
    total_grad_bias = total_grad_bias.reshape(agent.bias.shape)

    agent.weights += total_grad_weights * agent.learning_rate
    agent.bias += total_grad_bias * agent.learning_rate

    return reward, np.array(traj)

def train_with_reservoir_episode(agent, env, reservoir, time_steps=30):
    env.reset_inner()
    reservoir.reset_state()

    done = False
    time = 0
    traj = []

    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        res_input = np.concatenate((agent_position.flatten(), env.encode(action, res=env.grid_size).flatten()))
        res_modulation = np.array(reward)
        reservoir.update(res_input, res_modulation)

    grads = reservoir.readout()

    total_grad_weights = grads[:100].reshape(agent.weights.shape)
    total_grad_bias = grads[100:].reshape(agent.bias.shape)

    agent.weights += total_grad_weights * agent.learning_rate
    agent.bias += total_grad_bias * agent.learning_rate

    return reward, np.array(traj)

def train_with_reservoir(agent, env, reservoir, episodes=100, time_steps=30, verbose=False):
    rewards = []
    trajectories = []

    for episode in range(episodes):
        reward, traj = train_with_reservoir_episode(agent, env, reservoir, time_steps)
        max_length = time_steps
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)  # Use np.nan for padding
        padded_traj[:traj.shape[0], :] = traj
        trajectories.append(padded_traj)
        rewards.append(reward)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return rewards, trajectories

def InDistributionTraining(agent, env, reservoir, rounds = 1, episodes = 600, time_steps = 30, verbose = False):

    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    gradients_list = []
    reservoir_states = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()

        rewards, trajectories, gradients, reservoir_out = train_reservoir(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=verbose)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})
        gradients_list.append(gradients)
        reservoir_states.append(reservoir_out)

    return np.array(totalRewards), totalTrajectories, np.array(gradients_list), np.array(reservoir_states)

def TrainingWithReservoir(agent, env, reservoir, rounds=1, episodes=600, time_steps=30, verbose=False):
    n_resets = 2 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0 = 45/2 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()

        rewards, trajectories = train_with_reservoir(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=verbose)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(totalRewards), totalTrajectories

def test_one():
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir

    spatial_res = 5
    input_dim = 25  
    output_dim = 4
    agent = LinearAgent(input_dim, output_dim)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    reservoir = initialize_reservoir(agent, env)

    reward, traj, gradients, reservoir_state = train_reservoir_episode(agent, env, reservoir, time_steps=30)
    print(f"Reward: {reward}")
    # print(f"Trajectory: {traj}")
    print(f"Gradients shape: {gradients.shape}")
    print(f"Reservoir state shape: {reservoir_state.shape}")

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.04, temperature=1.0)

    rewards, trajectories, gradients_list, reservoir_states = train_reservoir(agent, env, reservoir, episodes=300, time_steps=30, verbose=False)
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Rewards over episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def test_runs():
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir
    from plottingUtils import plot_rewards

    spatial_res = 5
    input_dim = 25  
    output_dim = 4
    agent = LinearAgent(input_dim, output_dim, learning_rate=0.05, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    reservoir = initialize_reservoir(agent, env, reservoir_size=500, spectral_radius=0.9)

    rewards, trajectories, gradients_list, reservoir_states = InDistributionTraining(agent, env, reservoir, rounds=1, episodes=200, time_steps=30, verbose=False)
    # plot_rewards(rewards)
    #plot_trajectories(trajectories)

    # Now let's see if the training worksw
    print(gradients_list.shape)
    print(reservoir_states.shape)

    # Flatten
    # Reshape from (n_runs, episodes, feature_dim) to (n_runs * episodes, feature_dim)
    gradients_list = gradients_list.reshape(-1, gradients_list.shape[-1])
    reservoir_states = reservoir_states.reshape(-1, reservoir_states.shape[-1])

    print(gradients_list.shape)
    print(reservoir_states.shape)

    reservoir.train(reservoir_states, gradients_list)

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.01, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    
    rewards, _ = TrainingWithReservoir(agent, env, reservoir, rounds=1, episodes=200, time_steps=30, verbose=False)
    plot_rewards(rewards)

if __name__ == "__main__":
    test_runs()