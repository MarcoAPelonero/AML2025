import numpy as np
from tqdm import tqdm

"""
This module is the equivalent of trainingUtils.py but adapted to work with the reservoir computing model, so it has functions
that handle the pure training part, where gradients are generated trough reinforce online, and the inference part, where 
instead they are generaterd by the reservoir
"""

# Hyperparameters for trainining and inference, we want to introduce noise in training but not in inference
GAMMA_GRAD = 0.15  
noise_in_train = 1e-4
noise_in_inference = 0 

def train_episode(agent, env, reservoir, time_steps: int = 30):
    """
    This function handles a single episode. Starts by resetting the position of the agent and the reservoir state
    cleaning the temporal states. Then for a number of time steps or until the episode is done, it performs:
    - Gets the current encoded position of the agent
    - Multiplies that position (environment state) by the agent weights, applies softmax to get action probabilities, and selects an action
    - Steps the environment with the selected action, receiving a reward and a done flag
    - Updates the agent weights based on the received reward
    - Encodes the reward and the angle to the origin, and prepares the input vector for the reservoir
    - Steps the reservoir with the input vector and modulation based on the reward

    Once the episode is done, everything that needs to be returned is padded with NaNs to ensure consistent shapes.
    This includes the trajectory of the agent, the reservoir states, and the gradients used for updating the agent.
    Args:
        agent: The agent interacting with the environment.
        env: The environment in which the agent operates.
        reservoir: The reservoir computing model used for processing inputs.
        time_steps (int): Maximum number of time steps for the episode.
    Returns:
        reward (float): Total reward obtained in the episode.
        padded_traj (np.ndarray): Padded trajectory of the agent's positions.
        padded_res_states (np.ndarray): Padded reservoir states over time.
        padded_grads (np.ndarray): Padded gradients used for updating the agent.
    """

    env.reset_inner()
    reservoir.reset()

    done = False
    t = 0

    traj = []          
    grads = []         
    res_states = []    

    while not done and t < time_steps:
        t += 1
        traj.append(env.agent_position.copy())  # (x, y)

        agent_position_enc = env.encoded_position  # shape (5, 5)
        flat_pos_enc = agent_position_enc.flatten()

        action, probs = agent.sample_action(flat_pos_enc)
        reward, done = env.step(action)  

        grad = agent.update_weights(flat_pos_enc, action, reward)

        r_encoded = env.encode(reward)  

        x, y = env.agent_position  # current position AFTER the step
        angle = np.arctan2(y, x)   # range (‑π, π]
        angle_encoded = env.encode(angle, angle=True)  # same size as reward encoding

        input_modulation = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        input_modulation = input_modulation.flatten()

        input_vec = np.concatenate([
            flat_pos_enc,            # 25 dims (5×5 grid)
            probs,                   # |A| dims
            r_encoded.flatten(),     # reward encoding
            angle_encoded.flatten()   # angle encoding (NEW)
        ]).reshape(-1)

        reservoir.step_rate(input_vec, input_modulation, noise_in_train)

        res_states.append(reservoir.S.copy())
        grads.append(grad.flatten().copy())

    padded_traj = np.full((time_steps, traj[0].shape[0]), np.nan)
    padded_traj[: len(traj), :] = traj

    padded_res_states = np.full((time_steps, reservoir.S.shape[0]), np.nan)
    padded_res_states[: len(res_states), :] = res_states

    padded_grads = np.full((time_steps, grads[0].shape[0]), np.nan)
    padded_grads[: len(grads), :] = grads

    return reward, padded_traj, padded_res_states, padded_grads

def train(agent, env, reservoir, episodes=100, time_steps=30, verbose=False):
    """
    This function handles the training process over multiple episodes, for the same kind of environment (same angle), 
    by launching a episodes time the train_episode function. It collects rewards, trajectories, reservoir states, and gradients
    from each episode, and returns them as numpy arrays. The agent's parameters are reset at the beginning of the training process,
    since we don't want to carry over any learned parameters from previous angles. 
    Args:
        agent: The agent to be trained.
        env: The environment in which the agent operates.
        reservoir: The reservoir computing model used for processing inputs.
        episodes (int): Number of episodes to train the agent.
        time_steps (int): Maximum number of time steps per episode.
        verbose (bool): If True, prints progress information during training.
    Returns:
        rewards (list): List of total rewards obtained in each episode.
        trajectories (np.ndarray): Array of padded trajectories from all episodes.
        reservoir_states (np.ndarray): Array of padded reservoir states from all episodes.
        gradients (np.ndarray): Array of padded gradients from all episodes.
    """
    rewards = []
    trajectories = []
    reservoir_states = []
    gradients = []

    agent.reset_parameters()

    for episode in range(episodes):
        reward, traj, res_states, grads = train_episode(agent, env, reservoir, time_steps)
        trajectories.append(traj)
        rewards.append(reward)
        reservoir_states.append(res_states)
        gradients.append(grads)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return rewards, np.array(trajectories), np.array(reservoir_states), np.array(gradients)

def InDistributionTraining(agent, env, reservoir, rounds = 2, episodes = 600, time_steps = 30, verbose = False, bar=True):
    """
    Handles the generation of the entire training dataset across the in‑distribution angles (multiples of 45°).
    For each angle, the environment is reset, and the agent's parameters are reset to ensure no carry-over learning.
    For each environment reset, we launch the train function for that angle. 
    """
    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []
    total_reservoir_states = []
    total_gradients = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets, disable=not bar):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)

        agent.reset_parameters()

        rewards, trajectories, reservoir_states, gradients = train(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=False)

        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards[-50:])}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})
        total_reservoir_states.append(reservoir_states)
        total_gradients.append(gradients)

    return np.array(totalRewards), totalTrajectories, np.array(total_reservoir_states), np.array(total_gradients)

def organize_dataset(reservoir_states, gradients):
    """
    Organizes the reservoir states and gradients into proper arrays that are easy to use to build the readout matrix.
    Practically, this simply reshapes the original arrays from (n_resets, episodes, time_steps, dim) to (n_resets * episodes * time_steps, dim),
    and removes any rows that contain NaN values, which were used for padding shorter episodes.
    """
    res_states = reservoir_states.reshape(-1, reservoir_states.shape[-1])
    grads = gradients.reshape(-1, gradients.shape[-1])
    res_states = res_states[~np.isnan(res_states).any(axis=1)]
    grads = grads[~np.isnan(grads).any(axis=1)]
    return res_states, grads

def inference_episode(agent, env, reservoir, time_steps=30):
    """
    Once the reservoir is trained, we can use it to perform inference. This function handles a single inference episode.
    This is the exact same as the train_episode function, except that:
    - We use noise_in_inference instead of noise_in_train when stepping the reservoir
    - We do not update the agent weights based on the received reward, 
    - Instead we use the reward to build the input vector for the reservoir, and we use the reservoir output to update the agent weights.
    Args:
        agent: The agent interacting with the environment.
        env: The environment in which the agent operates.
        reservoir: The reservoir computing model used for processing inputs.
        time_steps (int): Maximum number of time steps for the episode.
    Returns:
        reward (float): Total reward obtained in the episode.
        padded_traj (np.ndarray): Padded trajectory of the agent's positions.
        padded_res_states (np.ndarray): Padded reservoir states over time.
        padded_grads (np.ndarray): Padded gradients used for updating the agent.
    """
    env.reset_inner()
    reservoir.reset()
    done = False
    time = 0

    traj = []
    grads = []
    res_states = []

    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position_enc = env.encoded_position  
        flat_pos_enc = agent_position_enc.flatten()
        action, probs = agent.sample_action(agent_position_enc.flatten())
        reward, done = env.step(action)
        
        r_encoded = env.encode(reward)  

        x, y = env.agent_position  
        angle = np.arctan2(y, x)   
        angle_encoded = env.encode(angle, angle=True)  

        input_modulation = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        input_modulation = input_modulation.flatten()

        input_vec = np.concatenate([ 
            flat_pos_enc,            
            probs,                  
            r_encoded.flatten(),     
            angle_encoded.flatten()  
        ]).reshape(-1)

        reservoir.step_rate(input_vec, input_modulation, 0.0)
        res_states.append(reservoir.S.copy())
        dw_out = np.copy(np.reshape(reservoir.y,(4,5**2)))
        agent.weights += dw_out 
        grads.append(dw_out.flatten().copy())

    padded_traj = np.full((time_steps, traj[0].shape[0]), np.nan)  
    padded_traj[:len(traj), :] = traj
    padded_res_states = np.full((time_steps, reservoir.S.shape[0]), np.nan)  
    padded_res_states[:len(res_states), :] = res_states
    padded_grads = np.full((time_steps, grads[0].shape[0]), np.nan)  
    padded_grads[:len(grads), :] = grads

    return reward, np.array(padded_traj), np.array(padded_res_states), np.array(padded_grads)

def inference_episode_multiplier(agent, env, reservoir, multiplier, time_steps=30):
    """
    Same as inference_episode, but we multiply the reservoir output by a multiplier before updating the agent weights.
    This is useful to test the effect of different learning rates during inference when using the oneShotReservoirMultiplier module.
    """
    env.reset_inner()
    reservoir.reset()
    done = False
    time = 0

    traj = []
    grads = []
    res_states = []

    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position_enc = env.encoded_position  
        flat_pos_enc = agent_position_enc.flatten()
        action, probs = agent.sample_action(agent_position_enc.flatten())
        reward, done = env.step(action)
        
        r_encoded = env.encode(reward)  

        x, y = env.agent_position  
        angle = np.arctan2(y, x)   
        angle_encoded = env.encode(angle, angle=True)  
        input_modulation = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        input_modulation = input_modulation.flatten()

        input_vec = np.concatenate([ 
            flat_pos_enc,            
            probs,                   
            r_encoded.flatten(),     
            angle_encoded.flatten()  
        ]).reshape(-1)

        reservoir.step_rate(input_vec, input_modulation, 0.0)
        res_states.append(reservoir.S.copy())
        dw_out = np.copy(np.reshape(reservoir.y,(4,5**2)))
        agent.weights += dw_out * multiplier
        grads.append(dw_out.flatten().copy())

    padded_traj = np.full((time_steps, traj[0].shape[0]), np.nan)  
    padded_traj[:len(traj), :] = traj
    padded_res_states = np.full((time_steps, reservoir.S.shape[0]), np.nan)  
    padded_res_states[:len(res_states), :] = res_states
    padded_grads = np.full((time_steps, grads[0].shape[0]), np.nan) 
    padded_grads[:len(grads), :] = grads

    return reward, np.array(padded_traj), np.array(padded_res_states), np.array(padded_grads)

def inference(agent, env, reservoir, episodes=100, time_steps=30, verbose=False):
    """
    This is the copy of the train function, but uses inference_episode instead of train_episode.
    """
    rewards = []
    trajectories = []
    reservoir_states = []
    gradients = []

    agent.reset_parameters()

    for episode in range(episodes):
        reward, traj, res_states, grads = inference_episode(agent, env, reservoir, time_steps)
        trajectories.append(traj)
        rewards.append(reward)
        reservoir_states.append(res_states)
        gradients.append(grads)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return rewards, np.array(trajectories), np.array(reservoir_states), np.array(gradients)

def InDistributionInference(agent, env, reservoir, rounds = 1, episodes = 600, time_steps = 30, verbose = False, bar=True):
    """
    Handles the entire inference process across the in‑distribution angles (multiples of 45°).
    For each angle, the environment is reset, and the agent's parameters are reset to ensure no carry-over learning.
    For each environment reset, we launch the inference function for that angle.
    Args:
        agent: The agent to be used for inference.
        env: The environment in which the agent operates.
        reservoir: The reservoir computing model used for processing inputs.
        rounds (int): Number of times to cycle through all in-distribution angles.
        episodes (int): Number of episodes to run for each angle.
        time_steps (int): Maximum number of time steps per episode.
        verbose (bool): If True, prints progress information during inference.
        bar (bool): If True, displays a progress bar using tqdm.
    Returns:
        totalRewards (np.ndarray): Array of total rewards obtained in each episode across all angles.
        totalTrajectories (list): List of dictionaries containing food positions and trajectories for each angle.
        reservoir_states (np.ndarray): Array of padded reservoir states from all episodes across all angles.
        gradients (np.ndarray): Array of padded gradients from all episodes across all angles.
    """
    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets, disable=not bar):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)

        agent.reset_parameters()

        rewards, trajectories, reservoir_states, gradients = inference(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=False)

        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards[-50:])}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(totalRewards), totalTrajectories, np.array(reservoir_states), np.array(gradients)


def OODInference(agent, env, reservoir, rounds = 1, episodes = 600, time_steps = 30, verbose = False):
    """
    Handles the entire inference process across the out‑of‑distribution angles (multiples of 22.5° that are not multiples of 45°).
    For each angle, the environment is reset, and the agent's parameters are reset to ensure no carry-over learning.
    For each environment reset, we launch the inference function for that angle.
    Args:
        agent: The agent to be used for inference.
        env: The environment in which the agent operates.
        reservoir: The reservoir computing model used for processing inputs.
        rounds (int): Number of times to cycle through all out-of-distribution angles.
        episodes (int): Number of episodes to run for each angle.
        time_steps (int): Maximum number of time steps per episode.
        verbose (bool): If True, prints progress information during inference.
    Returns:
        totalRewards (np.ndarray): Array of total rewards obtained in each episode across all angles.
        totalTrajectories (list): List of dictionaries containing food positions and trajectories for each angle
        reservoir_states (np.ndarray): Array of padded reservoir states from all episodes across all angles.
        gradients (np.ndarray): Array of padded gradients from all episodes across all angles.
    """
    n_resets = 16 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 22.5 * n_angle
        n_angle += 1 
        env.reset(theta0)

        agent.reset_parameters()

        rewards, trajectories, reservoir_states, gradients = inference(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=False)

        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards[-50:])}")

        totalRewards.append(rewards)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(totalRewards), totalTrajectories, np.array(reservoir_states), np.array(gradients)
        
def test1():
    """
    Simple test to check if everything is working as expected, the shapes match and the reservoir is 
    not at saturation during either training or inference.
    """
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir

    input_dim = 5**2 
    output_dim = 4

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.02, temperature=1.0)
    env = Environment(grid_size=5, sigma=0.2)
    env.reset()
    reservoir = initialize_reservoir()

    agent.reset_parameters()

    time_steps = 30
    rewards = 0
    while rewards != 1.5:
        agent.reset_parameters()
        rewards, trajectories, res_states, grads = train_episode(agent, env, reservoir, time_steps)

    print("Rewards:", rewards)
    print("Trajectories:", trajectories)
    print("Reservoir States:", res_states)
    print("Gradients:", grads)

    print(res_states.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(res_states.shape[1]):
        plt.plot(res_states[:, i], label=f'Neuron {i+1}')
    plt.title('Reservoir States Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('State Value')
    plt.show()

    rewards, trajectories, reservoir_states, gradients = InDistributionTraining(agent, env, reservoir, rounds=4, episodes=600, time_steps=time_steps, verbose=True)
    
    from reservoir import build_W_out

    res_states = reservoir_states.reshape(-1, reservoir_states.shape[-1])
    grads = gradients.reshape(-1, gradients.shape[-1])
    res_states = res_states[~np.isnan(res_states).any(axis=1)]
    grads = grads[~np.isnan(grads).any(axis=1)]
    print("Res States Shape:", res_states.shape)
    print("Gradients Shape:", grads.shape)

    W_out = build_W_out(res_states, grads)

    print("W_out Shape:", W_out.shape)
    print("reservoir out shape:", reservoir.Jout.shape)
    reservoir.Jout = W_out.T

    rewards,trajectories,_,_ = OODInference(agent, env, reservoir, rounds=1, episodes=600, time_steps=time_steps, verbose=True)

    from plottingUtils import plot_rewards_ood, plot_trajectories_ood
    plot_trajectories_ood(trajectories, batch_size=200, savefig=True, filename="presentation_figures/trajectories_plot_reservoir.png")

    plot_rewards_ood(rewards, savefig=True, filename="presentation_figures/rewards_plot_reservoir.png")


def test2():
    # Additional testing
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir

    input_dim = 5**2 
    output_dim = 4

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.02, temperature=1.0)
    env = Environment(grid_size=5, sigma=0.2)
    env.reset(45)
    reservoir = initialize_reservoir()

    agent.reset_parameters()

    time_steps = 30
    rewards, trajectories, reservoir_states_train, gradients_train = train(agent, env, reservoir, episodes=600, time_steps=time_steps, verbose=False)

    import os
    import matplotlib.pyplot as plt
    if not os.path.exists('grads'):
        os.makedirs('grads')
    for i in range(0, len(gradients_train), 100):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for j in range(gradients_train[i].shape[1]):
            plt.plot(gradients_train[i][:, j])
        plt.xlabel('Time Steps')
        plt.ylabel('Gradient Value')
        plt.title(f'Gradients at Episode {i}')

        plt.subplot(1, 2, 2)
        for j in range(reservoir_states_train[i].shape[1]):
            plt.plot(reservoir_states_train[i][:, j])
        plt.xlabel('Time Steps')
        plt.ylabel('Reservoir State Value')
        plt.title(f'Reservoir States at Episode {i}')

        plt.savefig(f'grads/grad_reservoir_{i}.png')
        plt.close()

    rewards, trajectories, reservoir_states, gradients = inference(agent, env, reservoir, episodes=600, time_steps=time_steps, verbose=False)
    if not os.path.exists('inference_results'):
        os.makedirs('inference_results')
    for i in range(0, len(gradients), 100):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for j in range(gradients[i].shape[1]):
            plt.plot(gradients[i][:, j])
        plt.xlabel('Time Steps')
        plt.ylabel('Gradient Value')
        plt.title(f'Gradients at Episode {i} (Inference)')

        plt.subplot(1, 2, 2)
        for j in range(reservoir_states[i].shape[1]):
            plt.plot(reservoir_states[i][:, j])
        plt.xlabel('Time Steps')
        plt.ylabel('Reservoir State Value')
        plt.title(f'Reservoir States at Episode {i} (Inference)')

        plt.savefig(f'inference_results/inference_grad_reservoir_{i}.png')
        plt.close()
    print("Inference results saved.")

    from reservoir import build_W_out
    res_states = reservoir_states_train.reshape(-1, reservoir_states_train.shape[-1])
    grads = gradients_train.reshape(-1, gradients_train.shape[-1])
    res_states = res_states[~np.isnan(res_states).any(axis=1)]
    grads = grads[~np.isnan(grads).any(axis=1)]
    print("Res States Shape:", res_states.shape)
    print("Gradients Shape:", grads.shape)
    W_out = build_W_out(res_states, grads)
    print("W_out Shape:", W_out.shape)
    print("reservoir out shape:", reservoir.Jout.shape)
    reservoir.Jout = W_out.T
    print("Reservoir Jout updated.")
    rewards, trajectories, reservoir_states, gradients = inference(agent, env, reservoir, episodes=600, time_steps=time_steps, verbose=False)
    if not os.path.exists('inference_results_after_training'):
        os.makedirs('inference_results_after_training')
    for i in range(0, len(gradients), 100):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for j in range(gradients[i].shape[1]):
            plt.plot(gradients[i][:, j])
        plt.xlabel('Time Steps')
        plt.ylabel('Gradient Value')
        plt.title(f'Gradients at Episode {i} (Inference After Training)')

        plt.subplot(1, 2, 2)
        for j in range(reservoir_states[i].shape[1]):
            plt.plot(reservoir_states[i][:, j])
        plt.xlabel('Time Steps')
        plt.ylabel('Reservoir State Value')
        plt.title(f'Reservoir States at Episode {i} (Inference After Training)')

        plt.savefig(f'inference_results_after_training/inference_grad_reservoir_{i}.png')
        plt.close()
    print("Inference results after training saved.")

    from plottingUtils import plot_single_run
    plot_single_run(np.array(rewards), bin_size=30)

if __name__ == "__main__":
    test1()
    