from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from trainingUtils import episode

GAMMA_GRAD = 0.05  
noise_in_train = 1e-4
noise_in_inference = 1e-4

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
    return reward, S_final, entropy_scalar, W_snapshot.flatten()

def train_meta(agent, env, reservoir, episodes=100, time_steps=30, verbose=False, bar=False):
    rewards, res_states, entropy_scalars, W_snapshots = [], [], [], []

    agent.reset_parameters()

    for episode in tqdm(range(episodes), disable=not bar):
        reward, res_state, entropy_scalar, W_snapshot = train_episode_meta(agent, env, reservoir, time_steps)
        rewards.append(reward)
        res_states.append(np.concatenate([res_state, [entropy_scalar]]))
        entropy_scalars.append(entropy_scalar)
        W_snapshots.append(W_snapshot)

        if verbose:
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

    return np.array(rewards), np.array(res_states), np.array(entropy_scalars), np.array(W_snapshots)

def InDistributionMetaTraining(agent, env, reservoir, rounds = 1, episodes = 600, time_steps = 30, verbose = False, bar=True):
    n_resets = 8 * rounds
    n_angle = 0

    totalRewards, totalReservoirStates, totalEntropyScalars, totalWSnapshots = [], [], [], []

    for n in tqdm(range(n_resets), desc='Resets', total=n_resets, disable=not bar):
        theta0= 45 * n_angle
        n_angle += 1 
        env.reset(theta0)

        agent.reset_parameters()

        rewards, res_states, entropy_scalars, W_snapshots = train_meta(agent, env, reservoir, episodes=episodes, time_steps=time_steps, verbose=False)

        if verbose:
            print(f"Reset {n + 1}/{n_resets}, Angle: {theta0}, Average Reward: {np.mean(rewards[-50:])}")

        totalRewards.append(rewards)
        totalReservoirStates.append(res_states)
        totalEntropyScalars.append(entropy_scalars)
        totalWSnapshots.append(W_snapshots)

    return np.array(totalRewards), np.array(totalReservoirStates), np.array(totalEntropyScalars), np.array(totalWSnapshots)

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
    return reward, S_final, entropy_scalar


def run_meta_inference(agent, env, reservoir,
                       k=1, mode="last", episodes_total=600,
                       time_steps=30, eta=1.0, clip_norm=0.6, verbose=False):
    rewards_hist, dW_acc = [], None
    agent.reset_parameters()
    if mode not in ["last", "average"]:
        raise ValueError("Mode must be either 'last' or 'average'")
    
    counter = 0
    while counter < k:
        reward, S_final, H = inference_episode_meta(agent, env, reservoir, time_steps)
        if reward == 1.5:
            counter += 1
            rewards_hist.append(reward)
            S_aug  = np.concatenate([S_final, [H]])
            dW_pred = np.tanh(reservoir.W_meta.T @ S_aug)
            dW_acc  = dW_pred if (mode == "last" or dW_acc is None) else dW_acc + dW_pred

    if mode == "average":
        dW_acc /= k
    norm = np.linalg.norm(dW_acc)
    if norm > clip_norm:
        dW_acc *= clip_norm / (norm + 1e-12)
    agent.weights += eta * dW_acc.reshape(agent.weights.shape)
    rewards_hist = []
    for ep in range(k, episodes_total):
        reward, *_ = episode(agent, env)
        rewards_hist.append(reward)
        if verbose and (ep % 50 == 0):
            print(f"Episode {ep+1}/{episodes_total}  R={reward:.3f}")

    return np.array(rewards_hist)

def testing():
    from agent import LinearAgent
    from environment import Environment
    from reservoir import initialize_reservoir

    agent = LinearAgent()
    env = Environment()
    reservoir = initialize_reservoir()

    reward, S_final, entropy_scalar, W_snapshot = train_episode_meta(agent, env, reservoir)

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

    # plot_single_run(rewards)

    # plt.figure(figsize=(10, 5))
    # plt.plot(entropy_scalars)
    # plt.xlabel('Episode')
    # plt.ylabel('Entropy Scalar')
    # #plt.title('Entropy Scalar Over Episodes')
    # plt.show()

    totalRewards, totalReservoirStates, totalEntropyScalars, totalWSnapshots = InDistributionMetaTraining(agent, env, reservoir, rounds=3, episodes=600, time_steps=30, verbose=False)

    print("Total Rewards:", totalRewards.shape)
    print("Total Reservoir States:", totalReservoirStates.shape)
    print("Total Entropy Scalars:", totalEntropyScalars.shape)
    print("Total Weights Snapshots:", totalWSnapshots.shape)

    num_tasks, num_eps, feat_dim = totalReservoirStates.shape
    S_all = totalReservoirStates.reshape(-1, feat_dim)           # (4800, 601)

    # 2. build ΔW dataset  --------------------------------------------
    W_star  = totalWSnapshots[:, -1, :]                           # (8, 100)
    ΔW_list = []

    for task in range(num_tasks):
        ΔW_task = W_star[task] - totalWSnapshots[task]            # (600, 100)
        ΔW_list.append(ΔW_task)

    ΔW_all = np.vstack(ΔW_list)

    print("ΔW_all shape:", ΔW_all.shape)  # (4800, 100)
    print("S_all shape:", S_all.shape)    # (4800, 601)

    # mean = S_all.mean(axis=0, keepdims=True)
    # std  = S_all.std(axis=0,  keepdims=True) + 1e-8
    # X = (S_all - mean) / std                                      # (4800, 601)
    X = S_all.copy()  # No normalization for now

    Y = np.arctanh(np.clip(ΔW_all, -0.999, 0.999))                # (4800, 100)

    print("X shape:", X.shape)   # (4800, 601)
    print("Y shape:", Y.shape)   # (4800, 100)

    lam = 1e-5
    W_meta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    print("W_meta shape:", W_meta.shape)

    reservoir.W_meta = W_meta

    # Test the inference 
    env.reset(45)
    rewards_hist = run_meta_inference(agent, env, reservoir, k=1, mode="average", episodes_total=600, time_steps=30, eta=1.0, clip_norm=10, verbose=True)

    print("Inferred Rewards:", rewards_hist)
    plt.figure(figsize=(10, 5))
    plt.hist(rewards_hist, bins=20, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inferred Rewards')
    plt.show()

if __name__ == "__main__":
    testing()