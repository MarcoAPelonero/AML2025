import numpy as np
from tqdm import tqdm

GAMMA_GRAD = 1

def train_episode(agent, env, reservoir, time_steps=30):
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
        agent_position = env.encoded_position
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        grad = agent.update_weights(agent_position.flatten(), action, reward)

        r_encoded = env.encode(reward)

        input_modulation = .1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        input_modulation= input_modulation.flatten()

        input = np.concatenate((agent_position.reshape(5**2,), probs, r_encoded.flatten()))
        input = np.reshape(input,(np.shape(input)[0],))

        reservoir.step_rate(input,input_modulation,0)
        res_states.append(reservoir.S.copy())
        grads.append(grad.flatten().copy())

    padded_traj = np.full((time_steps, traj[0].shape[0]), np.nan)  # Use np.nan for padding
    padded_traj[:len(traj), :] = traj
    padded_res_states = np.full((time_steps, reservoir.S.shape[0]), np.nan)  # Use np.nan for padding
    padded_res_states[:len(res_states), :] = res_states
    padded_grads = np.full((time_steps, grads[0].shape[0]), np.nan)  # Use np.nan for padding
    padded_grads[:len(grads), :] = grads

    return reward, np.array(padded_traj), np.array(padded_res_states), np.array(padded_grads)

def train(agent, env, reservoir, episodes=100, time_steps=30, verbose=False):
    rewards = []
    trajectories = []
    reservoir_states = []
    gradients = []

    for episode in range(episodes):
        reward, traj, res_states, grads = train_episode(agent, env, reservoir, time_steps)
        trajectories.append(traj)
        rewards.append(reward)
        reservoir_states.append(res_states)
        gradients.append(grads)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return rewards, np.array(trajectories), np.array(reservoir_states), np.array(gradients)

def InDistributionTraining(agent, env, reservoir, rounds = 1, episodes = 600, time_steps = 30, verbose = False):
    n_resets = 8 * rounds
    n_angle = 0

    totalRewards = []
    totalTrajectories = []

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

    return np.array(totalRewards), totalTrajectories, np.array(reservoir_states), np.array(gradients)

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
        agent_position = env.encoded_position
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        
        r_encoded = env.encode(reward)

        input_modulation = .1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        input_modulation= input_modulation.flatten()

        input = np.concatenate((agent_position.reshape(5**2,), probs, r_encoded.flatten()))
        input = np.reshape(input,(np.shape(input)[0],))

        reservoir.step_rate(input,input_modulation,0)
        res_states.append(reservoir.S.copy())
        dw_out = np.copy(np.reshape(reservoir.y,(4,5**2)))
        agent.weights += dw_out
        grads.append(dw_out.flatten().copy())

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

    for episode in range(episodes):
        reward, traj, res_states, grads = train_episode(agent, env, reservoir, time_steps)
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

    '''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(res_states.shape[1]):
        plt.plot(res_states[:, i], label=f'Neuron {i+1}')
    plt.title('Reservoir States Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('State Value')
    plt.show()
    '''

    rewards, trajectories, reservoir_states, gradients = InDistributionTraining(agent, env, reservoir, rounds=1, episodes=600, time_steps=time_steps, verbose=True)
    
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

    rewards,_,_,_ = OODInference(agent, env, reservoir, rounds=1, episodes=600, time_steps=time_steps, verbose=True)

    from plottingUtils import plot_rewards_ood

    plot_rewards_ood(rewards)

if __name__ == "__main__":
    test1()
    