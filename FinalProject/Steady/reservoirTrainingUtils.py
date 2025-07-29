import numpy as np
from tqdm import tqdm

# NOTE: This snippet only shows the updated function. Make sure the rest of your
# codebase (e.g. Agent, Environment, Reservoir classes) is imported or defined
# elsewhere in your project before running.
GAMMA_GRAD = 0.05  # Modulation factor for the input to the reservoir
noise_in_train = 1e-4
noise_in_inference = 1e-4
def train_episode(agent, env, reservoir, time_steps: int = 30):
    """Run one episode, feeding the reservoir with:
    - the agent's one‑hot position encoding (grid flattened)
    - the agent's action probabilities  
    - the reward encoding  
    - **NEW:** the angle (radians) between the agent's current position and the
      origin, encoded with ``env.encode``.

    Returns
    -------
    reward : float
        Final reward at the end of the episode.
    padded_traj : np.ndarray, shape (time_steps, 2)
        Trajectory (x, y) padded with NaNs.
    padded_res_states : np.ndarray, shape (time_steps, |reservoir|)
        Reservoir states for each step, padded with NaNs.
    padded_grads : np.ndarray, shape (time_steps, |theta|)
        Policy‑gradient vectors for each step, padded with NaNs.
    """

    # --- Episode setup -----------------------------------------------------
    env.reset_inner()
    reservoir.reset()

    done = False
    t = 0

    traj = []          # (x, y) positions
    grads = []         # policy‑gradient vectors
    res_states = []    # reservoir states

    while not done and t < time_steps:
        # ------------------------------------------------------------------
        # 1. Observe the CURRENT state (before moving)
        t += 1
        traj.append(env.agent_position.copy())  # (x, y)

        # Encoded (one‑hot) position used by the policy network
        agent_position_enc = env.encoded_position  # shape (5, 5)
        flat_pos_enc = agent_position_enc.flatten()

        # 2. Sample action & execute it -------------------------------------
        action, probs = agent.sample_action(flat_pos_enc)
        reward, done = env.step(action)  # env.agent_position is now updated

        # 3. Policy‑gradient update ----------------------------------------
        grad = agent.update_weights(flat_pos_enc, action, reward)

        # 4. Encode scalars -------------------------------------------------
        r_encoded = env.encode(reward)  # reward encoding

        # NEW: angle to origin (in radians) & its encoding
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

    # --- Padding to fixed length ------------------------------------------
    padded_traj = np.full((time_steps, traj[0].shape[0]), np.nan)
    padded_traj[: len(traj), :] = traj

    padded_res_states = np.full((time_steps, reservoir.S.shape[0]), np.nan)
    padded_res_states[: len(res_states), :] = res_states

    padded_grads = np.full((time_steps, grads[0].shape[0]), np.nan)
    padded_grads[: len(grads), :] = grads

    return reward, padded_traj, padded_res_states, padded_grads

def train(agent, env, reservoir, episodes=100, time_steps=30, verbose=False):
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

def InDistributionTraining(agent, env, reservoir, rounds = 2, episodes = 600, time_steps = 30, verbose = False):
    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []
    total_reservoir_states = []
    total_gradients = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
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

def inference_episode(agent, env, reservoir, time_steps=30):
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
        agent_position_enc = env.encoded_position  # shape (5, 5)
        flat_pos_enc = agent_position_enc.flatten()
        action, probs = agent.sample_action(agent_position_enc.flatten())
        reward, done = env.step(action)
        
        r_encoded = env.encode(reward)  # reward encoding

        # NEW: angle to origin (in radians) & its encoding
        x, y = env.agent_position  # current position AFTER the step
        angle = np.arctan2(y, x)   # range (‑π, π]
        angle_encoded = env.encode(angle, angle=True)  # same size as reward encoding

        input_modulation = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        input_modulation = input_modulation.flatten()

        input_vec = np.concatenate([ 
            flat_pos_enc,            # 25 dims (5×5 grid)
            probs,                   # |A| dims
            r_encoded.flatten(),     # reward encoding
            angle_encoded.flatten()  # angle encoding (NEW)
        ]).reshape(-1)

        reservoir.step_rate(input_vec, input_modulation, 0.0)
        res_states.append(reservoir.S.copy())
        dw_out = np.copy(np.reshape(reservoir.y,(4,5**2)))
        agent.weights += dw_out 
        grads.append(dw_out.flatten().copy())
        '''
        if reward != 0:
            agent.weights += dw_out
            grads.append(dw_out.flatten().copy())
        else:
            grads.append(np.zeros_like(dw_out).flatten().copy())
        '''

    padded_traj = np.full((time_steps, traj[0].shape[0]), np.nan)  # Use np.nan for padding
    padded_traj[:len(traj), :] = traj
    padded_res_states = np.full((time_steps, reservoir.S.shape[0]), np.nan)  # Use np.nan for padding
    padded_res_states[:len(res_states), :] = res_states
    padded_grads = np.full((time_steps, grads[0].shape[0]), np.nan)  # Use np.nan for padding
    padded_grads[:len(grads), :] = grads

    return reward, np.array(padded_traj), np.array(padded_res_states), np.array(padded_grads)

def inference(agent, env, reservoir, episodes=100, time_steps=30, verbose=False):
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

def InDistributionInference(agent, env, reservoir, rounds = 1, episodes = 600, time_steps = 30, verbose = False):
    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
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

    rewards, trajectories, reservoir_states, gradients = InDistributionTraining(agent, env, reservoir, rounds=2, episodes=600, time_steps=time_steps, verbose=True)
    
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
    plot_trajectories_ood(trajectories)

    plot_rewards_ood(rewards)
    

def test2():
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

    # Now sin a "grads" folder save every 100 episodes an image of the evolving gradients and the evolving reservoir states side by side
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

        # Now do the same with inference results BEFORE training and aftyer in 2 separate folders
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
    # Now train 
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
    # Now do the inference again
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
    