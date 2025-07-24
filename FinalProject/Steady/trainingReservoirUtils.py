import numpy as np

def train_reservoir_episode(agent, env, reservoir, time_steps = 30):
    env.reset_inner()
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
        res_modulation = env.encode(reward, res=env.grid_size).flatten()

        reservoir_state = reservoir.update(res_input, res_modulation)

    grad_weights, grad_bias = agent.apply_gradients()

    grad_weights = grad_weights.flatten()
    grad_bias = grad_bias.flatten()

    # Those are 2 arrays of shape 25 4 and 4. I need one array of shape 104
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

def InDistributionTraining(agent, env, reservoir, rounds=2, episodes=600, time_steps=30, mode='accumulation', verbose=False):
    for round in range(rounds):
        if verbose:
            print(f"Round {round + 1}/{rounds}")
        train_reservoir(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=verbose)

def test_one():
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir

    spatial_res = 5
    input_dim = 25  
    output_dim = 4
    agent = LinearAgent(input_dim, output_dim)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    reservoir = initialize_reservoir(agent, env, reservoir_size=100, spectral_radius=0.9)

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

if __name__ == "__main__":
    test_one()