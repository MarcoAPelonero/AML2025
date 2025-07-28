from tqdm import tqdm
import numpy as np

def train_episode(agent, env, reservoir, time_steps=30, verbose=False):
    
    env.reset_inner()
    reservoir.reset_state()

    trajectories = []
    reservoir_states = []
    weight_updates = []

    done = False
    t = 0
    while not done and t < time_steps:
        agent_position = env.encoded_position.copy()
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        weights_update = agent.update_weights(agent_position.flatten(), action, reward)

        reservoir_input = np.concatenate((agent_position.flatten(), probs, env.encode(reward, res=10).flatten()))
        reservoir_state = reservoir.step(reservoir_input, reward)

        if verbose:
            print(f"Step {t + 1}/{time_steps}, Action: {action}, Reward: {reward}")

        trajectories.append(env.agent_position.copy())
        reservoir_states.append(reservoir_state.copy())
        weight_updates.append(weights_update.flatten().copy())

        t += 1

    max_length = time_steps
    padded_traj = np.full((max_length, env.agent_position.shape[0]), np.nan)  
    padded_traj[:len(trajectories), :] = trajectories
    
    padded_reservoir_states = np.full((max_length, reservoir_states[0].shape[0]), np.nan)
    padded_reservoir_states[:len(reservoir_states), :] = reservoir_states

    padded_weight_updates = np.full((max_length, weight_updates[0].shape[0]), np.nan)
    padded_weight_updates[:len(weight_updates), :] = weight_updates

    return reward, padded_traj, padded_reservoir_states, padded_weight_updates

def train(agent, env, reservoir, num_episodes, bar = False, verbose=False):

    agent.reset_parameters()
    rewards, trajs, res_states, weight_updates = [], [], [], []

    for episode in tqdm(range(num_episodes), disable=not bar, desc="Training Episodes"):

        reward, traj, res, weight = train_episode(agent, env, reservoir)

        rewards.append(reward)
        trajs.append(traj)
        res_states.append(res)
        weight_updates.append(weight)

        if verbose:
            if episode % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward}")

    return np.array(rewards), np.array(trajs), np.array(res_states), np.array(weight_updates)

def train_with_reservoir_episode(agent, env, reservoir, time_steps=30, verbose=False):

    env.reset_inner()
    reservoir.reset_state()

    trajectories = []
    reservoir_states = []
    weight_updates = []

    done = False
    t = 0

    while not done and t < time_steps:
        agent_position = env.encoded_position.copy()
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        reservoir_input = np.concatenate((agent_position.flatten(), probs, env.encode(reward, res=10).flatten()))
        reservoir_state = reservoir.step(reservoir_input, reward)
        weights_update = reservoir.predict(reservoir_state).copy()
        # MAYBE DO THESTEP AFTER THE PREDICT, TRY TOMORROW
        grad = weights_update.reshape(agent.weights.shape)

        agent.apply_external_gradients(grad,reward)

        if verbose:
            print(f"Step {t + 1}/{time_steps}, Action: {action}, Reward: {reward}")

        trajectories.append(env.agent_position.copy())
        reservoir_states.append(reservoir_state.copy())
        weight_updates.append(weights_update.flatten().copy())

        t += 1

    max_length = time_steps
    padded_traj = np.full((max_length, env.agent_position.shape[0]), np.nan)  
    padded_traj[:len(trajectories), :] = trajectories
    
    padded_reservoir_states = np.full((max_length, reservoir_states[0].shape[0]), np.nan)
    padded_reservoir_states[:len(reservoir_states), :] = reservoir_states

    padded_weight_updates = np.full((max_length, weight_updates[0].shape[0]), np.nan)
    padded_weight_updates[:len(weight_updates), :] = weight_updates

    return reward, padded_traj, padded_reservoir_states, padded_weight_updates

def train_with_reservoir(agent, env, reservoir, num_episodes, bar = False, verbose=False):

    agent.reset_parameters()
    rewards, trajs, res_states, weight_updates = [], [], [], []

    for episode in tqdm(range(num_episodes), disable=not bar, desc="Training Episodes"):

        reward, traj, res, weight = train_with_reservoir_episode(agent, env, reservoir)

        rewards.append(reward)
        trajs.append(traj)
        res_states.append(res)
        weight_updates.append(weight)

        if verbose:
            if episode % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward}")

    return np.array(rewards), np.array(trajs), np.array(res_states), np.array(weight_updates)

def prepare_data_for_training(reservoir_states, gradients):
    """
    Prepare the data for training by padding the reservoir states and gradients,
    filtering out any NaNs, and appending an intercept term to the inputs.

    Args:
        reservoir_states (np.ndarray): Array of shape (episodes, time_steps, n_res) of states.
        gradients        (np.ndarray): Array of shape (episodes, time_steps, n_params) of grad targets.

    Returns:
        X (np.ndarray): Array of shape (N, n_res + 1) of cleaned inputs with intercept.
        Y (np.ndarray): Array of shape (N, n_params) of cleaned gradient targets.
    """
    # Flatten episodes/time into samples
    X = reservoir_states.reshape(-1, reservoir_states.shape[-1])
    Y = gradients       .reshape(-1, gradients.shape[-1])

    print(f"Original X shape: {X.shape}, Original Y shape: {Y.shape}")

    # Build a combined mask to drop any sample with NaN in inputs or targets
    mask = ~ (np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X_clean = X[mask]
    Y_clean = Y[mask]
    print(f"Filtered X shape: {X_clean.shape}, Filtered Y shape: {Y_clean.shape}")

    # Append intercept (bias) term
    # intercept = np.ones((X_clean.shape[0], 1))
    # X_with_intercept = np.hstack([X_clean, intercept])
    # print(f"X with intercept shape: {X_with_intercept.shape}")

    return X_clean, Y_clean

def DatasetPreparation(agent, env, reservoir, num_episodes=100, time_steps=30, bar=False, verbose=False, full_return=False):
    """
    Prepare the dataset for training by running multiple episodes and collecting reservoir states and gradients.

    Args:
        agent: The agent to be trained.
        env: The environment in which the agent operates.
        reservoir: The reservoir to be used for training.
        num_episodes (int): Number of episodes to run.
        time_steps (int): Number of time steps per episode.
        bar (bool): Whether to show a progress bar.
        verbose (bool): Whether to print detailed information.

    Returns:
        np.ndarray: Padded reservoir states.
        np.ndarray: Padded gradients.
    """
    
    id_angles = np.arange(0, 360, 45).tolist()

    total_rewards = []
    total_trajectories = []
    total_reservoir_states = []
    total_weight_updates = []

    for theta0 in tqdm(id_angles, desc="ID Angles", total=len(id_angles), disable=not bar):
        env.reset(theta0=theta0)
        rewards, trajectories, reservoir_states, weight_updates = train(agent, env, reservoir, num_episodes, bar=False, verbose=False)

        total_rewards.append(rewards)
        total_trajectories.append(trajectories)
        total_reservoir_states.append(reservoir_states)
        total_weight_updates.append(weight_updates)

    total_rewards, total_trajectories, total_reservoir_states, total_weight_updates = map(np.array, (total_rewards, total_trajectories, total_reservoir_states, total_weight_updates))

    X, Y = prepare_data_for_training(total_reservoir_states, total_weight_updates)

    if full_return:
        return X, Y, total_rewards, total_trajectories, total_reservoir_states, total_weight_updates
    return X, Y

def TrainWithReservoir(agent, env, reservoir, num_episodes=100, time_steps=30, bar=False, verbose=False):
    ood_angles = np.arange(0, 45, 22.5).tolist()
    ood_angles = np.arange(0, 180, 22.5).tolist()

    total_rewards = []
    total_trajectories = []
    total_reservoir_states = []
    total_weight_updates = []

    for theta0 in tqdm(ood_angles, desc="OOD Angles", total=len(ood_angles), disable=not bar):
        env.reset(theta0=theta0)
        rewards, trajectories, reservoir_states, weight_updates = train_with_reservoir(agent, env, reservoir, num_episodes, bar=False, verbose=False)

        total_rewards.append(rewards)
        total_trajectories.append(trajectories)
        total_reservoir_states.append(reservoir_states)
        total_weight_updates.append(weight_updates)

    total_rewards, total_trajectories, total_reservoir_states, total_weight_updates = map(np.array, (total_rewards, total_trajectories, total_reservoir_states, total_weight_updates))

    return total_rewards, total_trajectories, total_reservoir_states, total_weight_updates

def overfit():
    
    from agent import LinearAgent
    from reservoir import ModulatedESN
    from environment import Environment

    from reservoir import plot_reservoir_states
    from plottingUtils import plot_single_run

    agent = LinearAgent(learning_rate=0.03, temperature=1.0)  
    env = Environment()  
    res = ModulatedESN(n_in = 39, n_res = 500, seed = 42)

    env.reset(30)

    num_episodes = 600
    rewards, trajectories, reservoir_states, weight_updates = train(agent, env, res, num_episodes, bar = True, verbose=False)

    X, Y = prepare_data_for_training(reservoir_states, weight_updates)

    res.fit(X, Y)

    rewards, trajectories, reservoir_states, weight_updates = train_with_reservoir(agent, env, res, num_episodes, bar = True, verbose=False)

    plot_single_run(rewards, bin_size=10)

def main():
    from agent import LinearAgent
    from reservoir import ModulatedESN
    from environment import Environment
    from plottingUtils import plot_rewards, plot_rewards_ood
    import matplotlib.pyplot as plt

    agent = LinearAgent(learning_rate=0.01, temperature=1.0)  
    env = Environment()  
    res = ModulatedESN(n_in=39, n_res=200, seed=42)

    X, Y, rewards, trajectories, reservoir_states, weight_updates = DatasetPreparation(agent, env, res, num_episodes=600, time_steps=30, bar=True, verbose=False, full_return=True)

    res.fit(X, Y)
    # plot_rewards(rewards)

    total_reservoir_rewards, total_reservoir_trajectories, total_reservoir_states, total_weight_updates = TrainWithReservoir(agent, env, res, num_episodes=600, time_steps=30, bar=True, verbose=False)

    plot_rewards_ood(total_reservoir_rewards, savefig=True)

    print("Weight updates shape:", weight_updates.shape)
    print("Total weight states shape:", total_weight_updates.shape)

    agent_grads = weight_updates.reshape(-1, weight_updates.shape[-1])
    reservoir_grads = total_weight_updates.reshape(-1, total_weight_updates.shape[-1])

    agent_norms      = np.linalg.norm(agent_grads,      axis=1)
    reservoir_norms  = np.linalg.norm(reservoir_grads,  axis=1)

    plt.figure(figsize=(10,4))
    plt.plot(agent_norms,     label="Agent (REINFORCE)", lw=1.2)
    plt.plot(reservoir_norms, label="Reservoir‑prop",    lw=1.2)
    plt.xlabel("Update step")
    plt.ylabel("L2 norm of gradient")
    plt.title("Gradient magnitude over time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.hist(agent_norms,     bins=60, alpha=0.6, label="Agent")
    plt.hist(reservoir_norms, bins=60, alpha=0.6, label="Reservoir")
    plt.xlabel("L2 norm")
    plt.ylabel("Frequency")
    plt.title("Gradient‑norm histogram")
    plt.legend()
    plt.tight_layout()
    plt.show()             

def boh():
    import os
    from agent import LinearAgent
    from reservoir import ModulatedESN
    from environment import Environment
    import matplotlib.pyplot as plt

    os.makedirs("gradient_traces", exist_ok=True)

    agent = LinearAgent(learning_rate=0.01, temperature=1.0)  
    env = Environment()  
    res = ModulatedESN(n_in=39, n_res=200, seed=42)

    # Step 1: Train the reservoir on true agent gradients
    X, Y, rewards, trajectories, reservoir_states, weight_updates = DatasetPreparation(
        agent, env, res, num_episodes=600, time_steps=30, bar=True, verbose=False, full_return=True)
    res.fit(X, Y)

    # Step 2: Track gradients during test episodes
    agent.reset_parameters()
    env.reset(45)

    num_episodes = 600
    time_steps = 30

    for episode in range(num_episodes):
        env.reset_inner()
        res.reset_state()

        agent_grads = []
        res_grads = []

        done = False
        t = 0

        while not done and t < time_steps:
            state = env.encoded_position.copy()
            action, probs = agent.sample_action(state.flatten())
            reward, done = env.step(action)

            # Agent computes true gradient
            agent_grad = agent.update_weights(state.flatten(), action, reward)
            agent_grads.append(agent_grad.flatten().copy())

            # Reservoir prediction
            res_input = np.concatenate((state.flatten(), probs, env.encode(reward, res=10).flatten()))
            res_state = res.step(res_input, reward)
            pred_grad = res.predict(res_state).copy()
            res_grads.append(pred_grad)

            t += 1

        # Convert and clean padded data
        agent_grads = np.array(agent_grads)
        res_grads   = np.array(res_grads)

        if len(agent_grads) == 0 or len(res_grads) == 0:
            continue

        agent_norms = np.linalg.norm(agent_grads, axis=1)
        res_norms   = np.linalg.norm(res_grads,   axis=1)

        # Plot every 100 episodes
        if (episode + 1) % 100 == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(agent_norms, label="Agent (REINFORCE)", lw=1.5)
            plt.plot(res_norms,   label="Reservoir", lw=1.5)
            plt.xlabel("Timestep")
            plt.ylabel("L2 norm of gradient")
            plt.title(f"Gradient norm evolution – Episode {episode + 1}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"gradient_traces/episode_{episode + 1}_gradients.png")
            plt.close()

    print("Saved gradient trace plots for both agent and reservoir.")


if __name__ == "__main__":
    boh()  # Call the function to run the training and plotting
    # overfit()