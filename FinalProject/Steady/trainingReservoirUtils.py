import numpy as np
from tqdm import tqdm

def train_reservoir_episode(agent, env, reservoir, time_steps=30):
    env.reset_inner()
    reservoir.reset_state()

    done = False
    time = 0
    traj = []

    weight_list, bias_list = [], []
    reservoir_states = []

    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        weights, bias = agent.update_weights(agent_position.flatten(), action, reward)

        res_input = np.concatenate((agent_position.flatten(), probs, env.encode(action, res=env.grid_size).flatten()))
        res_modulation = env.encode(reward, res=env.grid_size).flatten()
        res_out = reservoir.update(res_input, res_modulation, sigma_S=0.05)

        weight_list.append(weights.flatten())
        bias_list.append(bias.flatten())

        reservoir_states.append(res_out)

    max_length = time_steps
    padded_traj = np.full((max_length, traj[0].shape[0]), np.nan)  
    padded_traj[:len(traj), :] = traj
    padded_weights = np.full((max_length, len(weight_list[0])), np.nan)
    padded_weights[:len(weight_list), :] = weight_list
    padded_biases = np.full((max_length, len(bias_list[0])), np.nan)
    padded_biases[:len(bias_list), :] = bias_list
    padded_reservoir_states = np.full((max_length, len(reservoir_states[0])), np.nan)
    padded_reservoir_states[:len(reservoir_states), :] = reservoir_states

    return reward, np.array(padded_traj), np.array(padded_weights), np.array(padded_biases), np.array(padded_reservoir_states)

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
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        res_input = np.concatenate((agent_position.flatten(), probs, env.encode(action, res=env.grid_size).flatten()))
        res_modulation = env.encode(reward, res=env.grid_size).flatten()
        grads = reservoir.step(res_input, res_modulation)
        weight_grad = grads[:100].reshape(agent.weights.shape)
        bias_grad = grads[100:].reshape(agent.bias.shape)

        agent.apply_external_gradients((weight_grad, bias_grad))

    return reward, np.array(traj)


def train(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=False):
    reward_list = []
    trajectories_list = []
    weights_list = []
    biases_list = []
    reservoir_states_list = []
    for episode in tqdm(range(episodes), disable=not bar, desc='Training Reservoir'):
        reward, traj, weights, biases, reservoir_states = train_reservoir_episode(agent, env, reservoir, time_steps)
        reward_list.append(reward)
        trajectories_list.append(traj)
        weights_list.append(weights)
        biases_list.append(biases)
        reservoir_states_list.append(reservoir_states)
        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")
    return np.array(reward_list), np.array(trajectories_list), np.array(weights_list), np.array(biases_list), np.array(reservoir_states_list)

def train_with_reservoir(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=False):
    reward_list = []
    trajectories_list = []
    for episode in tqdm(range(episodes), disable=not bar, desc='Training Reservoir'):
        reward, traj= train_with_reservoir_episode(agent, env, reservoir, time_steps)
        max_length = time_steps
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)  # Use np.nan for padding
        padded_traj[:traj.shape[0], :] = traj
        reward_list.append(reward)
        trajectories_list.append(padded_traj)
        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")
    return np.array(reward_list), np.array(trajectories_list)

def prepare_arrays_for_training(weights, biases, reservoir_states, check=True):
    """
    Takes in weights, biases, and reservoir states with potential padding (NaNs),
    removes padding, and returns flattened arrays for training.
    
    Returns:
        X: Reservoir states of shape (N, reservoir_dim)
        Y: Gradients of shape (N, weight_dim + bias_dim)
    """
    if check:
        print(f"Original Weights shape: {weights.shape}")
        print(f"Original Biases shape: {biases.shape}")
        print(f"Original Reservoir States shape: {reservoir_states.shape}")

    flat_weights = weights.reshape(-1, weights.shape[-1])
    flat_biases = biases.reshape(-1, biases.shape[-1])
    flat_states = reservoir_states.reshape(-1, reservoir_states.shape[-1])

    if check:
        print(f"Reshaped Weights shape: {flat_weights.shape}")
        print(f"Reshaped Biases shape: {flat_biases.shape}")
        print(f"Reshaped Reservoir States shape: {flat_states.shape}")

    valid_mask = ~(
        np.isnan(flat_weights).any(axis=1) |
        np.isnan(flat_biases).any(axis=1) |
        np.isnan(flat_states).any(axis=1)
    )

    cleaned_weights = flat_weights[valid_mask]
    cleaned_biases = flat_biases[valid_mask]
    cleaned_states = flat_states[valid_mask]

    # non_zero_mask = np.any(cleaned_weights != 0, axis=1) | np.any(cleaned_biases != 0, axis=1)
    # cleaned_weights = cleaned_weights[non_zero_mask]
    # cleaned_biases = cleaned_biases[non_zero_mask]
    # cleaned_states = cleaned_states[non_zero_mask]

    Y = np.concatenate([cleaned_weights, cleaned_biases], axis=1)
    X = cleaned_states

    return X, Y    

def GenerateReservoirDataset(agent, env, reservoir, rounds=1, episodes=100, time_steps=30, verbose=False, bar=True):
    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []
    totalWeights = []
    totalBiases = []
    totalReservoirStates = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()
        rewards, trajectories, weights, biases, reservoir_states = train(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=verbose)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})
        totalWeights.append(weights)
        totalBiases.append(biases)
        totalReservoirStates.append(reservoir_states)

    totalRewards = np.array(totalRewards)
    totalTrajectories = np.array(totalTrajectories)
    totalWeights = np.array(totalWeights)
    totalBiases = np.array(totalBiases)
    totalReservoirStates = np.array(totalReservoirStates)
    return totalRewards, totalTrajectories, totalWeights, totalBiases, totalReservoirStates

def TrainingWithReservoir(agent, env, reservoir, rounds=1, episodes=100, time_steps=30, verbose=False, bar=True):
    n_resets = 16 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45/2 * n_angle
        n_angle += 1 
        env.reset(theta0)
        agent.reset_parameters()
        rewards, trajectories = train_with_reservoir(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=verbose)
        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards)}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(totalRewards), np.array(totalTrajectories)

def test_one_episode():
    from reservoir import initialize_reservoir
    from agent import LinearAgent
    from environment import Environment

    env = Environment()
    agent = LinearAgent()
    reservoir = initialize_reservoir(agent, env)

    rew = 0
    while rew != 1.5:
        rew, traj, weights, biases, reservoir_states = train_reservoir_episode(agent, env, reservoir)
    print(f"Episode Reward: {rew}")
    print(f"Trajectory: {traj}")
    print(f"Weights: {weights}")
    print(f"Biases: {biases}")
    print(f"Reservoir States: {reservoir_states}")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(traj[:, 0], traj[:, 1], marker='o')
    axs[0, 0].set_title('Trajectory')
    axs[0, 0].set_xlabel('X Position')
    axs[0, 0].set_ylabel('Y Position')
    axs[0, 1].hist(weights.flatten(), bins=30, alpha=0.7)
    axs[0, 1].set_title('Weights Distribution')
    axs[0, 1].set_xlabel('Weight Value')
    axs[0, 1].set_ylabel('Frequency')
    axs[1, 0].hist(biases.flatten(), bins=30, alpha=0.7)
    axs[1, 0].set_title('Biases Distribution')
    axs[1, 0].set_xlabel('Bias Value')
    axs[1, 0].set_ylabel('Frequency')
    time = np.arange(len(reservoir_states))
    N = reservoir_states.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    res_states = np.array(reservoir_states)
    for i in range(N):
        axs[1, 1].plot(time, res_states[:, i], color=colors[i], linewidth=0.7)
    axs[1, 1].set_title('Reservoir States')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Reservoir Neuron')
    plt.tight_layout()
    plt.show()

def test_one_run():
    from reservoir import initialize_reservoir
    from agent import LinearAgent
    from environment import Environment

    env = Environment()
    agent = LinearAgent()
    reservoir = initialize_reservoir(agent, env, mode='probs')

    rewards, trajectories, weights, biases, reservoir_states = train(agent, env, reservoir, episodes=600, time_steps=30, verbose=False, bar=True)
    
    from plottingUtils import plot_single_run
    # Add one extra dimension to weights biases and reservoir_states to simulate a real scenario
    weights = np.expand_dims(weights, axis=0)
    biases = np.expand_dims(biases, axis=0)
    reservoir_states = np.expand_dims(reservoir_states, axis=0)
    print(rewards.shape)
    print(trajectories.shape)
    print(weights.shape)
    print(biases.shape)
    print(reservoir_states.shape)

    print(f"Rewards size: {rewards.nbytes / 1e6:.2f} MB")
    print(f"Trajectories size: {trajectories.nbytes / 1e6:.2f} MB")
    print(f"Weights size: {weights.nbytes / 1e6:.2f} MB")
    print(f"Biases size: {biases.nbytes / 1e6:.2f} MB")
    print(f"Reservoir States size: {reservoir_states.nbytes / 1e6:.2f} MB")

    plot_single_run(rewards)
    X, Y = prepare_arrays_for_training(weights, biases, reservoir_states, check=True)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X size: {X.nbytes / 1e6:.2f} MB")
    print(f"Y size: {Y.nbytes / 1e6:.2f} MB")

    # Now let's see if the training works
    reservoir.train(X,Y)
    agent.reset_parameters()

    rewards, trajectories = train_with_reservoir(agent, env, reservoir, episodes=600, time_steps=30, verbose=False, bar=True)
    plot_single_run(rewards)
    
def test_dataset_generation():
    from reservoir import initialize_reservoir
    from agent import LinearAgent
    from environment import Environment

    env = Environment()
    agent = LinearAgent()
    reservoir = initialize_reservoir(agent, env)

    rewards, trajectories, weights, biases, reservoir_states = GenerateReservoirDataset(agent, env, reservoir, rounds=1, episodes=600, time_steps=30, verbose=False, bar=True)
    
    print(f"Rewards shape: {rewards.shape}")
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Biases shape: {biases.shape}")
    print(f"Reservoir States shape: {reservoir_states.shape}")

    X, Y = prepare_arrays_for_training(weights, biases, reservoir_states, check=True)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

def main():
    from reservoir import initialize_reservoir
    from agent import LinearAgent
    from environment import Environment
    from plottingUtils import plot_rewards, plot_trajectories, plot_rewards_ood

    env = Environment()
    agent = LinearAgent()
    reservoir = initialize_reservoir(agent, env)

    rewards, trajectories, weights, biases, reservoir_states = GenerateReservoirDataset(agent, env, reservoir, rounds=1, episodes=600, time_steps=30, verbose=False, bar=True)
    plot_rewards(rewards, savefig=True, filename="agent_rewards.png")
    plot_trajectories(trajectories, savefig=True, filename="agent_trajectories.png")
    X, Y = prepare_arrays_for_training(weights, biases, reservoir_states, check=True)
    reservoir.train(X, Y)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    rewards, trajectories = TrainingWithReservoir(agent, env, reservoir, rounds=1, episodes=600, time_steps=30, verbose=False, bar=True)
    plot_rewards_ood(rewards, savefig=True, filename="reservoir_rewards.png")

if __name__ == "__main__":
    main()