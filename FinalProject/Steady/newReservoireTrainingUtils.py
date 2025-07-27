import numpy as np
from tqdm import tqdm

import reservoir
from trainingUtils import InDistributionTraining

def reservoir_episode(agent, env, reservoir, time_steps=30, mod_amplification=5):
    env.reset_inner()
    reservoir.reset()
    done = False
    time = 0
    traj = []
    reservoir_states = []
    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        r_encoded = env.encode(reward)

        input_modulation = .1 + reservoir.Jin_mult * reward * mod_amplification
        input_modulation = input_modulation.flatten() 

        reservoir_input = np.concatenate((env.encoded_position.copy().flatten(), probs, r_encoded.flatten()))
        reservoir.step_rate(reservoir_input, input_modulation)
        reservoir_states.append(reservoir.S.copy())

    return reward, np.array(traj), reservoir_states

def train_reservoir_episode(agent, env, reservoir, time_steps=30, mod_amplification=1):
    env.reset_inner()
    reservoir.reset()

    done = False
    time = 0
    traj = []

    reservoir_states = []
    gradients = []

    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        pre_weights, pre_bias = agent.weights.copy(), agent.bias.copy()
        agent.update_weights(agent_position.flatten(), action, reward)
        gradient_weights = agent.weights.copy() - pre_weights
        gradient_bias = agent.bias.copy() - pre_bias

        gradient = np.concatenate((gradient_weights.flatten(), gradient_bias.flatten()))

        r_encoded = env.encode(reward)

        input_modulation = .1 + reservoir.Jin_mult * reward * mod_amplification
        input_modulation = input_modulation.flatten()

        reservoir_input = np.concatenate((agent_position, probs, r_encoded.flatten()))
        reservoir.step_rate(reservoir_input, input_modulation)
        gradients.append(gradient)
        reservoir_states.append(reservoir.S.copy())

    max_length = time_steps
    padded_traj = np.full((max_length, traj[0].shape[0]), np.nan)  
    padded_traj[:len(traj), :] = traj
    padded_reservoir_states = np.full((max_length, reservoir.N), np.nan)  
    padded_reservoir_states[:len(reservoir_states), :] = reservoir_states
    padded_gradients = np.full((max_length, len(gradient)), np.nan)
    padded_gradients[:len(gradients), :] = gradients

    return reward, np.array(padded_traj), np.array(padded_reservoir_states), np.array(padded_gradients)

def train_with_reservoir_episode(agent, env, reservoir, time_steps=30, mod_amplification=1):
    env.reset_inner()
    reservoir.reset()
    done = False
    time = 0
    traj = []

    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        r_encoded = env.encode(reward)

        input_modulation = .1 + reservoir.Jin_mult * reward * mod_amplification
        input_modulation = input_modulation.flatten()

        reservoir_input = np.concatenate((env.encoded_position.copy().flatten(), probs, r_encoded.flatten()))
        reservoir.step_rate(reservoir_input, input_modulation)
        gradient_weights = np.reshape(reservoir.y.copy()[:100], (agent.weights.shape))
        gradient_bias = np.reshape(reservoir.y.copy()[100:], (agent.bias.shape))

        agent.weights += gradient_weights 
        agent.bias += gradient_bias 

    max_length = time_steps
    padded_traj = np.full((max_length, traj[0].shape[0]), np.nan)  
    padded_traj[:len(traj), :] = traj

    return reward, np.array(padded_traj)

def train_reservoir(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=False):
    rewards = []
    trajectories = []
    reservoir_states = []
    gradients = []

    agent.reset_parameters()

    for episode in tqdm(range(episodes), disable=not bar, desc='Training Reservoir'):
        reward, traj, res_states, grads = train_reservoir_episode(agent, env, reservoir, time_steps)

        trajectories.append(traj)
        reservoir_states.append(res_states)
        gradients.append(grads)
        rewards.append(reward)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return rewards, trajectories, reservoir_states, gradients

def train_with_reservoir(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=False):
    rewards = []
    trajectories = []

    agent.reset_parameters()

    for episode in tqdm(range(episodes), disable=not bar, desc='Training Reservoir'):
        reward, traj = train_with_reservoir_episode(agent, env, reservoir, time_steps)

        trajectories.append(traj)
        rewards.append(reward)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return rewards, trajectories

def TestOneReservoirPerFood(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=True):
    n_resets = 8 
    n_angle = 0 

    all_rewards = []
    all_trajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)
        print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Food Position: {env.food_position}")
        reward, _, reservoir_states, gradients = train_reservoir(agent, env, reservoir, episodes=600, time_steps=30, bar=False)

        X, Y = prepare_arrays_for_training(np.array(gradients), np.array(reservoir_states), check=True)
        
        reservoir.train(X, Y, reg=0.0001)

        reward_res, traj = train_with_reservoir(agent, env, reservoir, episodes=600, time_steps=30, bar=False)

        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Food Position: {env.food_position}")
            print(f"Average Reward Agent: {np.mean(reward[-50:])}")
            print(f"Average Reward Reservoir: {np.mean(reward_res[-50:])}")

        all_rewards.append(reward_res)
        all_trajectories.append({'food_position': env.food_position, 'trajectory': np.array(traj)})

    return np.array(all_rewards), all_trajectories

def TrainOrIWillKillTheAgentaaaaaaaaaaaa(agent, env, reservoir, episodes=600, time_steps=30, verbose=False, bar=False):
    id_angles  = np.arange(0, 360, 45).tolist()
    ood_angles = np.arange(0, 360, 22.5).tolist()


    print(id_angles)
    print(ood_angles)

    all_reservoirs = []
    all_gradients = []

    for angle in tqdm(id_angles, desc='In Distribution Training', total=len(id_angles), disable=not bar):
        env.reset(theta0=angle)
        rew, traj, reservoir_states, gradients = train_reservoir(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=verbose, bar=False)
        all_reservoirs.append(reservoir_states)
        all_gradients.append(gradients)
    
    all_reservoirs = np.array(all_reservoirs)
    all_gradients = np.array(all_gradients)

    X, Y = prepare_arrays_for_training(np.array(all_gradients), np.array(all_reservoirs), check=True)   
    reservoir.train(X, Y, reg=0.0001)

    rewards = []
    trajectories = []

    for angle in tqdm(ood_angles, desc='Out of Distribution Training', total=len(ood_angles), disable=not bar):
        env.reset(theta0=angle)
        rew, traj = train_with_reservoir(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=verbose, bar=False)

        rewards.append(rew)
        trajectories.append(traj)

    return np.array(rewards), trajectories

def prepare_arrays_for_training(gradients, reservoir_states, check=False):
    if check:
        print(f"gradients shape: {gradients.shape}, reservoir_states shape: {reservoir_states.shape}")

    X = np.array(reservoir_states)
    Y = np.array(gradients)

    if X.ndim < 4:
        X = np.expand_dims(X, axis=0)
    if Y.ndim < 4:
        Y = np.expand_dims(Y, axis=0)

    X = X.reshape(-1, X.shape[-1])
    Y = Y.reshape(-1, Y.shape[-1])

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    X = X[mask]
    Y = Y[mask]

    if check:
        print(f"Prepared X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y

def test_one_train_reservoir():
    from newReservoir import Reservoir
    from environment import Environment
    from agent import LinearAgent

    reservoir = Reservoir()
    env = Environment()
    agent = LinearAgent()

    reward = 0
    while reward != 1.5:
        reward, traj, reservoir_states, gradients = train_reservoir_episode(agent, env, reservoir, time_steps=30)

    print(f"Final reward: {reward}")
    print(f"Trajectory shape: {traj.shape}")
    print(f"Reservoir states shape: {reservoir_states.shape}")
    print(f"Gradients shape: {gradients.shape}")

    _, trajectories, reservoir_states, gradients = train_reservoir(agent, env, reservoir, episodes=600, time_steps=30, bar=True)

    from plottingUtils import plot_single_run

    X, Y = prepare_arrays_for_training(np.array(gradients), np.array(reservoir_states), check=True)

    reservoir.train(X, Y, reg=0.0001)

    reward, traj = train_with_reservoir(agent, env, reservoir, episodes=600, time_steps=30, bar=True)
    plot_single_run(np.array(reward),bin_size=30)
    rew_mean = np.mean(reward[400:])
    return rew_mean

def test():
    from newReservoir import Reservoir
    from environment import Environment
    from agent import LinearAgent

    from plottingUtils import plot_rewards, plot_trajectories   

    reservoir = Reservoir()
    env = Environment()
    agent = LinearAgent()

    rewards, trajectories = TestOneReservoirPerFood(agent, env, reservoir, verbose=True, bar=False)

    plot_rewards(rewards)
    plot_trajectories(trajectories)

def main():
    from newReservoir import Reservoir
    from environment import Environment
    from agent import LinearAgent
    from plottingUtils import plot_rewards, plot_rewards_ood, plot_trajectories_ood

    reservoir = Reservoir()
    env = Environment()
    agent = LinearAgent()

    rewards, trajectories = TrainOrIWillKillTheAgentaaaaaaaaaaaa(agent, env, reservoir, episodes=600, time_steps=30, verbose=False, bar=True)
    plot_rewards_ood(rewards)

def losing_my_mind():
    from newReservoir import Reservoir
    from environment import Environment
    from agent import LinearAgent
    import matplotlib.pyplot as plt
    import numpy as np

    # Initialize components
    reservoir = Reservoir()
    env = Environment()
    agent = LinearAgent()
    
    # Train reservoir normally
    rewards, trajectories, reservoir_states, gradients = train_reservoir(
        agent, env, reservoir, episodes=600, time_steps=30, bar=True
    )
    X, Y = prepare_arrays_for_training(np.array(gradients), np.array(reservoir_states), check=True)
    reservoir.train(X, Y, reg=0.0001)

    # Find a successful REINFORCE episode
    reward = 0
    while reward != 1.5:
        reward, traj, true_states, true_gradients = train_reservoir_episode(
            agent, env, reservoir, time_steps=30
        )
    
    # Remove padding from the successful episode
    mask = ~np.isnan(true_gradients).any(axis=1)
    true_states = true_states[mask]
    true_gradients = true_gradients[mask]
    
    # Get reservoir predictions for the same states
    predicted_gradients = []
    for state in true_states:
        reservoir.y = reservoir.Jout @ state  # Manual prediction
        predicted_gradients.append(reservoir.y.copy())
    predicted_gradients = np.array(predicted_gradients)

    # Calculate and print MSE
    mse = np.mean((true_gradients - predicted_gradients) ** 2)
    print(f"MSE between true and predicted gradients: {mse:.6f}")
    print(f"True gradients magnitude: {np.linalg.norm(true_gradients):.4f}")
    print(f"Predicted gradients magnitude: {np.linalg.norm(predicted_gradients):.4f}")

    # Plot comparison for first 5 gradient components
    plt.figure(figsize=(12, 8))
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(true_gradients[:, i], 'b-', label='True Grad')
        plt.plot(predicted_gradients[:, i], 'r--', label='Predicted Grad')
        plt.ylabel(f'Component {i}')
        if i == 0:
            plt.legend()
            plt.title('Gradient Component Comparison')
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('gradient_comparison.png')
    plt.show()

    return true_gradients, predicted_gradients

if __name__ == "__main__": 
    losing_my_mind()