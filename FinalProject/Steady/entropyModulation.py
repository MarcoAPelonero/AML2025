# The first objective is to define 2 separate reservoirs that will work in parallel in a optimal weight prediction task. We will follow the 
# progress, and see how the entropy scalar evolves over time. We have the functions for the distance predictor already, but they work on entropy, so without the entropy term we need to adapt them
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from stagePredictorReservoir import InDistributionMetaTraining
from trainingUtils import episode

GAMMA_GRAD = 0.15  
noise_in_train = 1e-4
noise_in_inference = 0

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

def inference_episode_meta_without_entropy(agent, env, reservoir, time_steps: int = 30):
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

        r_enc = env.encode(reward)
        x, y  = env.agent_position
        angle_enc = env.encode(np.arctan2(y, x), angle=True)
        inp_vec   = np.concatenate([flat_pos_enc, probs,
                                    r_enc.flatten(), angle_enc.flatten()])

        inp_mod = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        reservoir.step_rate(inp_vec, inp_mod.flatten(), 0.0)

    S_final        = reservoir.S.copy()
    return reward, S_final

def run_meta_inference_without_entropy(agent, env, reservoir,
                       k=1, mode="last", episodes_total=600,
                       time_steps=30, eta=1.0, clip_norm=0.6, verbose=False):
    rewards_hist, dW_acc = [], None
    agent.reset_parameters()
    if mode not in ["last", "average"]:
        raise ValueError("Mode must be either 'last' or 'average'")
    
    counter = 0
    while counter < k:
        reward, S_final = inference_episode_meta_without_entropy(agent, env, reservoir, time_steps)
        if reward == 1.5:
            counter += 1
            rewards_hist.append(reward)
            dW_pred = np.tanh(reservoir.W_meta.T @ S_final)
            dW_acc  = dW_pred if (mode == "last" or dW_acc is None) else dW_acc + dW_pred

    if mode == "average":
        dW_acc /= k
    norm = np.linalg.norm(dW_acc)
    if norm > clip_norm:
        dW_acc *= clip_norm / (norm + 1e-12)
    agent.weights += eta * dW_acc.reshape(agent.weights.shape)
    rewards_hist = []
    trajectories = []
    for ep in range(episodes_total):
        reward, traj = episode(agent, env)
        # Pad trajectory to fixed length of time_steps
        max_length = time_steps
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)
        padded_traj[:traj.shape[0], :] = traj
        trajectories.append(padded_traj)
        rewards_hist.append(reward)
        if verbose and (ep % 50 == 0):
            print(f"Episode {ep+1}/{episodes_total}  R={reward:.3f}")

    return np.array(rewards_hist), np.array(trajectories)

def OutOfDistributionMetaInferenceWithoutEntropy(agent, env, reservoir, k=1, episodes=600, time_steps=30, verbose=False, bar=True):
    n_resets = 16 

    rewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets, disable=not bar):
        theta0 = (45 / 2) * n
        env.reset(theta0)

        rewards_hist, trajectories = run_meta_inference_without_entropy(agent, env, reservoir, k=1, mode="average", episodes_total=episodes, time_steps=time_steps, eta=1.0, clip_norm=10, verbose=False)
        rewards.append(rewards_hist)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(rewards), totalTrajectories

def testing():
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir
    from plottingUtils import plot_rewards
    from stagePredictorReservoir import build_meta_weights 
    from stagePredictorReservoir import OutOfDistributionMetaInference
    from plottingUtils import plot_rewards_ood

    agent = LinearAgent()
    entropic_agent = LinearAgent()

    environment = Environment()
    entropic_environment = Environment()

    reservoir = initialize_reservoir(1000)
    entropic_reservoir = initialize_reservoir(1000)

    rewards, resStates, W_snapshots = InDistributionMetaTrainingWithoutEntropy(agent, environment, reservoir, rounds=1, episodes=600, bar=True)
    entropic_rewards, entropic_resStates, _, entropic_W_snapshots = InDistributionMetaTraining(entropic_agent, entropic_environment, entropic_reservoir, rounds=1, episodes=600, bar=True)

    W_out = build_meta_weights(resStates, W_snapshots)
    reservoir.W_meta = W_out

    W_meta = build_meta_weights(entropic_resStates, entropic_W_snapshots)
    entropic_reservoir.W_meta = W_meta

    rewards_inf, trajs = OutOfDistributionMetaInferenceWithoutEntropy(agent, environment, reservoir, k=1, episodes=600, bar=True)
    entropic_rewards_inf, entropic_trajs = OutOfDistributionMetaInference(entropic_agent, entropic_environment, entropic_reservoir, k=1, episodes=600, bar=True)

    plot_rewards_ood(rewards_inf, title="No Entropy Rewards", savefig=True, filename="rewards_inf_no_entropy.png")
    plot_rewards_ood(entropic_rewards_inf, title="Entropy Rewards", savefig=True, filename="rewards_inf_entropy.png")

if __name__ == "__main__":
    testing()   