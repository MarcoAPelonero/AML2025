# The first objective is to define 2 separate reservoirs that will work in parallel in a optimal weight prediction task. We will follow the 
# progress, and see how the entropy scalar evolves over time. We have the functions for the distance predictor already, but they work on entropy, so without the entropy term we need to adapt them
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from stagePredictorReservoir import InDistributionMetaTraining

GAMMA_GRAD = 0.05  
noise_in_train = 1e-4
noise_in_inference = 1e-4


def train_episode_meta_no_entropy(agent, env, reservoir, time_steps: int = 30):
    env.reset_inner()
    reservoir.reset()

    t, reward, done = 0, 0.0, False

    while not done and t < time_steps:
        t += 1
        flat_pos_enc = env.encoded_position.flatten()
        action, probs = agent.sample_action(flat_pos_enc)
        reward, done  = env.step(action)
        true_probs = probs.copy()
        true_probs[action] += 1

        agent.update_weights(flat_pos_enc, action, reward)

        r_enc = env.encode(reward)
        x, y  = env.agent_position
        angle_enc = env.encode(np.arctan2(y, x), angle=True)
        inp_vec   = np.concatenate([flat_pos_enc, probs,
                                    r_enc.flatten(), angle_enc.flatten()])

        inp_mod = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        reservoir.step_rate(inp_vec, inp_mod.flatten(), noise_in_train)

    S_final        = reservoir.S.copy()
    W_snapshot     = agent.weights.copy()
    return reward, S_final, W_snapshot.flatten()

def train_meta_no_entropy(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=False):
    rewards, res_states, entropy_scalars, W_snapshots = [], [], [], []

    agent.reset_parameters()

    for ep in tqdm(range(episodes), disable=not bar):
        reward, res_state, W_snapshot = train_episode_meta_no_entropy(agent, env, reservoir, time_steps)
        rewards.append(reward)
        if reward != 0.0:  
            res_states.append(res_state)
            W_snapshots.append(W_snapshot)

        if verbose:
            print(f"Episode {ep + 1}/{episodes}, Reward: {reward}")

    return (np.array(rewards),
            np.array(res_states),
            np.array(W_snapshots),
           )

def InDistributionMetaTrainingWithoutEntropy(agent, env, reservoir, rounds=1, episodes=600, time_steps=30, verbose=False, bar=True):
    n_resets = 8 * rounds
    totalRewards, totalReservoirStates, totalWSnapshots = [], [], []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets, disable=not bar):
        theta0 = 45 * (n % 8)  
        env.reset(theta0)
        agent.reset_parameters()

        rewards, res_states, W_snapshots= train_meta_no_entropy(
            agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=False, bar=False
        )

        if verbose:
            avg_last = np.mean(rewards[-50:]) if len(rewards) >= 1 else float('nan')
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, #Rewarded Episodes: {len(rewards)}, Average Reward (last 50): {avg_last:.3f}")

        totalRewards.append(rewards)                     
        totalReservoirStates.append(res_states)          
        totalWSnapshots.append(W_snapshots)               
        

    return totalRewards, totalReservoirStates, totalWSnapshots

def testing():
    # First, let's pair the 2 training against each other on a couple epochs and confirm they work
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir
    from plottingUtils import plot_rewards

    agent = LinearAgent()
    entropic_agent = LinearAgent()

    environment = Environment()
    entropic_environment = Environment()

    reservoir = initialize_reservoir(1000)
    entropic_reservoir = initialize_reservoir(1000)

    rewards, resStates, W_snapshots = InDistributionMetaTrainingWithoutEntropy(agent, environment, reservoir, rounds=1, episodes=600, bar=True)
    entropic_rewards, entropic_resStates, entropic_W_snapshots = InDistributionMetaTrainingWithoutEntropy(entropic_agent, entropic_environment, entropic_reservoir, rounds=1, episodes=600, bar=True)

    rewards = np.array(rewards)
    entropic_rewards = np.array(entropic_rewards)

    plot_rewards(rewards)
    plot_rewards(entropic_rewards)

if __name__ == "__main__":
    testing()   