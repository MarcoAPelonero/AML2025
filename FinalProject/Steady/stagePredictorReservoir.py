from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from plottingUtils import plot_trajectories
from trainingUtils import episode

GAMMA_GRAD = 0.05  
noise_in_train = 1e-4
noise_in_inference = 1e-4

def scale_entropy_to_match(res_state, encoded_entropy, gamma=1.0, eps=1e-12):
    # gamma lets you up-/down-weight beyond simple matching
    norm_res = np.linalg.norm(res_state)
    norm_enc = np.linalg.norm(encoded_entropy)
    if norm_enc < eps:
        return encoded_entropy  # avoid divide by zero
    alpha = (norm_res / (norm_enc + eps)) * gamma
    return alpha * encoded_entropy

def train_episode_meta(agent, env, reservoir, time_steps: int = 30):
    env.reset_inner()
    reservoir.reset()

    ent_acc, t, reward, done = 0.0, 0, 0.0, False

    while not done and t < time_steps:
        t += 1
        flat_pos_enc = env.encoded_position.flatten()
        action, probs = agent.sample_action(flat_pos_enc)
        reward, done  = env.step(action)
        true_probs = probs.copy()
        true_probs[action] += 1
        ent_acc += -np.sum(true_probs * np.log(true_probs + 1e-12))

        agent.update_weights(flat_pos_enc, action, reward)

        r_enc = env.encode(reward)
        x, y  = env.agent_position
        angle_enc = env.encode(np.arctan2(y, x), angle=True)
        inp_vec   = np.concatenate([flat_pos_enc, probs,
                                    r_enc.flatten(), angle_enc.flatten()])

        inp_mod = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        reservoir.step_rate(inp_vec, inp_mod.flatten(), noise_in_train)

    S_final        = reservoir.S.copy()
    entropy_scalar = ent_acc / t
    W_snapshot     = agent.weights.copy()
    encoded_entropy = env.encode_entropy(entropy_scalar, res = 20)
    encoded_entropy = scale_entropy_to_match(S_final, encoded_entropy, gamma=1.0)
    return reward, S_final, entropy_scalar, W_snapshot.flatten(), encoded_entropy.flatten()

def train_meta(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=False):
    rewards, res_states, entropy_scalars, W_snapshots = [], [], [], []

    agent.reset_parameters()

    for ep in tqdm(range(episodes), disable=not bar):
        reward, res_state, entropy_scalar, W_snapshot, encoded_entropy = train_episode_meta(agent, env, reservoir, time_steps)
        rewards.append(reward)
        if reward != 0.0:  # only keep episodes that yielded reward
            res_states.append(np.concatenate([res_state, encoded_entropy]))
            entropy_scalars.append(entropy_scalar)
            W_snapshots.append(W_snapshot)

        if verbose:
            print(f"Episode {ep + 1}/{episodes}, Reward: {reward}")

    return (np.array(rewards),
            np.array(res_states),
            np.array(entropy_scalars),
            np.array(W_snapshots),
           )

def InDistributionMetaTraining(agent, env, reservoir, rounds=1, episodes=600, time_steps=30, verbose=False, bar=True):
    n_resets = 8 * rounds
    totalRewards, totalReservoirStates, totalEntropyScalars, totalWSnapshots = [], [], [], []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets, disable=not bar):
        theta0 = 45 * (n % 8)  # or whatever angle logic you intend
        env.reset(theta0)
        agent.reset_parameters()

        rewards, res_states, entropy_scalars, W_snapshots= train_meta(
            agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=False, bar=False
        )

        if verbose:
            avg_last = np.mean(rewards[-50:]) if len(rewards) >= 1 else float('nan')
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, #Rewarded Episodes: {len(rewards)}, Average Reward (last 50): {avg_last:.3f}")

        totalRewards.append(rewards)                     # list of shape (n_i,)
        totalReservoirStates.append(res_states)           # list of shape (n_i, feat_dim+1)
        totalEntropyScalars.append(entropy_scalars)       # list of shape (n_i,)
        totalWSnapshots.append(W_snapshots)               # list of shape (n_i, weight_dim)
        

    return totalRewards, totalReservoirStates, totalEntropyScalars, totalWSnapshots

def inference_episode_meta(agent, env, reservoir, time_steps: int = 30):
    env.reset_inner()
    reservoir.reset()
    ent_acc, t, reward, done = 0.0, 0, 0.0, False

    while not done and t < time_steps:
        t += 1
        flat_pos_enc = env.encoded_position.flatten()
        action, probs = agent.sample_action(flat_pos_enc)
        reward, done  = env.step(action)
        true_probs = probs.copy()
        true_probs[action] += 1
        ent_acc += -np.sum(true_probs * np.log(true_probs + 1e-12))

        r_enc = env.encode(reward)
        x, y  = env.agent_position
        angle_enc = env.encode(np.arctan2(y, x), angle=True)
        inp_vec   = np.concatenate([flat_pos_enc, probs,
                                    r_enc.flatten(), angle_enc.flatten()])

        inp_mod = 0.1 + GAMMA_GRAD * reservoir.Jin_mult * reward
        reservoir.step_rate(inp_vec, inp_mod.flatten(), 0.0)

    S_final        = reservoir.S.copy()
    entropy_scalar = ent_acc / t
    encoded_entropy = env.encode_entropy(entropy_scalar, res=20).flatten()
    encoded_entropy = scale_entropy_to_match(S_final, encoded_entropy, gamma=1.0)
    return reward, S_final, entropy_scalar, encoded_entropy


def run_meta_inference(agent, env, reservoir,
                       k=1, mode="last", episodes_total=600,
                       time_steps=30, eta=1.0, clip_norm=0.6, verbose=False):
    rewards_hist, dW_acc = [], None
    agent.reset_parameters()
    if mode not in ["last", "average"]:
        raise ValueError("Mode must be either 'last' or 'average'")
    
    counter = 0
    while counter < k:
        reward, S_final, H, encoded_entropy = inference_episode_meta(agent, env, reservoir, time_steps)
        if reward == 1.5:
            counter += 1
            rewards_hist.append(reward)
            S_aug  = np.concatenate([S_final, encoded_entropy])
            dW_pred = np.tanh(reservoir.W_meta.T @ S_aug)
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

def OutOfDistributionMetaInference(agent, env, reservoir, k=1, episodes=600, time_steps=30, verbose=False, bar=True):
    n_resets = 16 

    rewards = []
    totalTrajectories = []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets, disable=not bar):
        theta0 = (45 / 2) * n
        env.reset(theta0)

        rewards_hist, trajectories = run_meta_inference(agent, env, reservoir, k=1, mode="average", episodes_total=episodes, time_steps=time_steps, eta=1.0, clip_norm=10, verbose=False)
        rewards.append(rewards_hist)
        totalTrajectories.append({'food_position': env.food_position, 'trajectory': np.array(trajectories)})

    return np.array(rewards), totalTrajectories

def build_meta_weights(res_states, W_snapshots):
    S_list = []
    DeltaW_list = []

    for res_states_task, W_snapshots_task in zip(res_states, W_snapshots):
        if res_states_task.size == 0:
            continue  # skip tasks with no rewarded episodes
        W_star = W_snapshots_task[-1]  # final successful weights for this task, shape (weight_dim,)
        # ΔW for each successful episode in this task: W_star - W_snapshot
        DeltaW_task = W_star[None, :] - W_snapshots_task  # (n_i, weight_dim)
        S_list.append(res_states_task)                    # (n_i, feat_dim+1)
        DeltaW_list.append(DeltaW_task)                   # (n_i, weight_dim)

    # Stack across all tasks
    S_all = np.vstack(S_list)           # (total_successful_episodes, feat_dim+1)
    ΔW_all = np.vstack(DeltaW_list)     # (total_successful_episodes, weight_dim)

    # Sanity checks
    assert S_all.shape[0] == ΔW_all.shape[0]
    assert not np.isnan(S_all).any()
    assert not np.isnan(ΔW_all).any()

    # Prepare regression targets
    X = S_all.copy()
    Y = np.arctanh(np.clip(ΔW_all, -0.999, 0.999))  # shape (N, weight_dim)

    lam = 1e-5
    W_meta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    return W_meta

def testing():
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir

    agent = LinearAgent()
    env = Environment()
    reservoir = initialize_reservoir()

    reward, S_final, entropy_scalar, W_snapshot, _ = train_episode_meta(agent, env, reservoir)

    print("Reward:", reward)
    print("Final Reservoir State:", S_final.shape)
    print("Entropy Scalar:", entropy_scalar)
    print("Weights Snapshot:", W_snapshot.shape)

    rewards, res_states, entropy_scalars, W_snapshots = train_meta(agent, env, reservoir, episodes=600, time_steps=30, verbose=False, bar=True)

    from plottingUtils import plot_single_run

    print("Rewards shape:", rewards.shape)
    print("Reservoir States shape:", res_states.shape)
    print("Entropy Scalars shape:", entropy_scalars.shape)
    print("Weights Snapshots shape:", W_snapshots.shape)

    plot_single_run(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(entropy_scalars)
    plt.xlabel('Episode')
    plt.ylabel('Entropy Scalar')
    plt.title('Entropy Scalar Over Episodes')
    plt.show()

    # total* are lists per task of variable-length arrays
    totalRewards, totalReservoirStates, totalEntropyScalars, totalWSnapshots = InDistributionMetaTraining(agent, env, reservoir, rounds=3, episodes=600, time_steps=30, verbose=False, bar=True)

    S_list = []
    DeltaW_list = []

    for res_states_task, W_snapshots_task in zip(totalReservoirStates, totalWSnapshots):
        if res_states_task.size == 0:
            continue  # skip tasks with no rewarded episodes
        W_star = W_snapshots_task[-1]  # final successful weights for this task, shape (weight_dim,)
        # ΔW for each successful episode in this task: W_star - W_snapshot
        DeltaW_task = W_star[None, :] - W_snapshots_task  # (n_i, weight_dim)
        S_list.append(res_states_task)                    # (n_i, feat_dim+1)
        DeltaW_list.append(DeltaW_task)                   # (n_i, weight_dim)

    # Stack across all tasks
    S_all = np.vstack(S_list)           # (total_successful_episodes, feat_dim+1)
    ΔW_all = np.vstack(DeltaW_list)     # (total_successful_episodes, weight_dim)

    # Sanity checks
    assert S_all.shape[0] == ΔW_all.shape[0]
    assert not np.isnan(S_all).any()
    assert not np.isnan(ΔW_all).any()

    # Prepare regression targets
    X = S_all.copy()
    Y = np.arctanh(np.clip(ΔW_all, -0.999, 0.999))  # shape (N, weight_dim)

    lam = 1e-5
    W_meta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    reservoir.W_meta = W_meta

    env.reset(22.5)
    rewards_hist, _ = run_meta_inference(agent, env, reservoir, k=1, mode="average", episodes_total=600, time_steps=30, eta=1.0, clip_norm=10, verbose=True)

    print("Inferred Rewards:", rewards_hist)
    plt.figure(figsize=(10, 5))
    plt.hist(rewards_hist, bins=20, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inferred Rewards')
    plt.show()

def main():
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir

    agent = LinearAgent()
    env = Environment()
    reservoir = initialize_reservoir(1000)

    totalRewards, totalReservoirStates, totalEntropyScalars, totalWSnapshots = InDistributionMetaTraining(agent, env, reservoir, rounds=3, episodes=600, time_steps=30, verbose=False, bar=True)

    W_meta = build_meta_weights(totalReservoirStates, totalWSnapshots)
    reservoir.W_meta = W_meta

    rewards, trajectories = OutOfDistributionMetaInference(agent, env, reservoir, k=1, episodes=600, time_steps=30, verbose=True, bar=True)

    print("Out-of-Distribution Rewards:", rewards.shape)
    print("Trajectories shape:", trajectories[0]['trajectory'].shape)
    print("Length of trajectories:", len(trajectories))

    from plottingUtils import plot_trajectories_ood, plot_rewards_ood
    plot_rewards_ood(rewards)
    plot_trajectories_ood(trajectories)

if __name__ == "__main__":
    testing()