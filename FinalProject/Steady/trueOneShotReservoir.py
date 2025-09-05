import copy
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from agent import LinearAgent
from environment import Environment
from reservoir import initialize_reservoir
from stagePredictorReservoir import InDistributionMetaTraining, build_meta_weights, inference_episode_meta
from entropyModulation import InDistributionMetaTrainingWithoutEntropy, inference_episode_meta_without_entropy
from trainingUtils import episode

from plottingUtils import plot_one_shot_eval

"""
This is the first of the 2 main modules to perform one-shot meta inference using a reservoir, and the backbone for the second one that.
This module evaluates the one shot inference capabilities using different k values (number of successful episodes before updating the agent's weights)
with and without entropy modulation. Most of the complexities regarding the module are tied to the parallelization of the evaluation of different angles,
the true low level complexity is handled in the stagePredictorReservoir.py module. because of that and because these 
complexities are common to the oneShot... modules, we don't explore them in much detail.
"""

def convert_numpy(obj):
    """Convert NumPy data types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def run_meta_inference_single_theta(agent, env, reservoir,
                                   theta0,
                                   k=1, mode="average", episodes_total=600,
                                   time_steps=30, eta=1.0, clip_norm=0.6, verbose=False, entropy=True):
    
    """
    Run meta inference for a single angle theta0 with a pretrained reservoir, this is a more structured version
    of the run_meta_inference function.
    """
    if mode not in ["last", "average"]:
        raise ValueError("Mode must be either 'last' or 'average'")

    agent.reset_parameters()
    env.reset(theta0)

    dW_acc = None
    counter = 0
    if entropy:
        while counter < k:
            reward, S_final, H, encoded_entropy = inference_episode_meta(agent, env, reservoir, time_steps)
            if reward == 1.5:
                counter += 1
                S_aug = np.concatenate([S_final, encoded_entropy])
                dW_pred = np.tanh(reservoir.W_meta.T @ S_aug)
                if mode == "last" or dW_acc is None:
                    dW_acc = dW_pred
                else:
                    dW_acc = dW_acc + dW_pred
    else:
        while counter < k:
            reward, S_final = inference_episode_meta_without_entropy(agent, env, reservoir, time_steps)
            if reward == 1.5:
                counter += 1
                dW_pred = np.tanh(reservoir.W_meta.T @ S_final)
                if mode == "last" or dW_acc is None:
                    dW_acc = dW_pred
                else:
                    dW_acc = dW_acc + dW_pred

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
        max_length = time_steps
        padded_traj = np.full((max_length, traj.shape[1]), np.nan)
        padded_traj[:traj.shape[0], :] = traj
        trajectories.append(padded_traj)
        rewards_hist.append(reward)
        if verbose and (ep % 50 == 0):
            print(f"Episode {ep+1}/{episodes_total}  R={reward:.3f}")

    return np.array(rewards_hist), np.array(trajectories)


def OutOfDistributionMetaInference_single_theta(agent, env, reservoir,
                                               theta0,
                                               k=1,
                                               episodes=600,
                                               time_steps=30,
                                               mode="average",
                                               eta=1.0,
                                               clip_norm=0.6,
                                               verbose=False,
                                               entropy=True):
    rewards, trajectories = run_meta_inference_single_theta(
        agent, env, reservoir,
        theta0=theta0,
        k=k,
        mode=mode,
        episodes_total=episodes,
        time_steps=time_steps,
        eta=eta,
        clip_norm=clip_norm,
        verbose=verbose,
        entropy=entropy
    )
    traj_dict = {
        "food_position": env.food_position.tolist(),
        "trajectory": trajectories
    }
    return rewards, traj_dict

def _eval_theta_meta_worker(theta0, agent, env, reservoir, k, rounds, episodes, time_steps, mode, eta, clip_norm, entropy):
    all_rewards = []
    example_traj = None
    for _ in range(rounds):
        agent_eval = copy.deepcopy(agent)
        env_eval = copy.deepcopy(env)
        rewards, traj_dict = OutOfDistributionMetaInference_single_theta(
            agent_eval, env_eval, reservoir,
            theta0=theta0,
            k=k,
            episodes=episodes,
            time_steps=time_steps,
            mode=mode,
            eta=eta,
            clip_norm=clip_norm,
            verbose=False,
            entropy=entropy
        )
        all_rewards.extend(rewards)
        if example_traj is None:
            example_traj = traj_dict
    mu = np.mean(all_rewards)
    sem = np.std(all_rewards, ddof=1) / np.sqrt(rounds)
    return theta0, mu, sem, all_rewards, example_traj


def EvalOneShotMetaInference(agent, env, reservoir,
                             rounds: int = 1,
                             k_list: list[int] = (1, 2, 4, 8, 16, 32, 64, 128),
                             episodes: int = 600,
                             time_steps: int = 30,
                             meta_train_rounds: int = 3,
                             episodes_train: int = 600,
                             parallel: bool = False,
                             bar: bool = True,
                             verbose: bool = False,
                             store_raw: bool = False,
                             mode="entropic", outfile="one_shot_meta_inference_results.json"):
    n_resets = 16
    theta_list = [45 / 2 * i for i in range(n_resets)]

    results = []
    for theta in theta_list:
        env_tmp = copy.deepcopy(env)
        env_tmp.reset(theta)
        entry = {
            "theta0": theta,
            "agent_position": env_tmp.agent_position.tolist(),
            "food_position": env_tmp.food_position.tolist(),
            "total_rewards": []
        }
        if store_raw:
            entry["raw_rewards"] = []
        entry["example_trajectory"] = None
        results.append(entry)

    for k in tqdm(k_list, desc="k values", disable=not bar):
        agent_train = copy.deepcopy(agent)
        env_train = copy.deepcopy(env)
        reservoir_train = copy.deepcopy(reservoir)

        if verbose:
            print(f"[Meta Training] k={k}")
        if mode == "entropic":
            totalRewards, totalReservoirStates, totalEntropyScalars, totalWSnapshots = InDistributionMetaTraining(
                agent_train, env_train, reservoir_train,
                rounds=meta_train_rounds,
                episodes=episodes_train,
                time_steps=time_steps,
                verbose=False,
                bar=False
            )
            entropy=True
        elif mode == "no_entropic":
            totalRewards, totalReservoirStates, totalWSnapshots = InDistributionMetaTrainingWithoutEntropy(
                agent_train, env_train, reservoir_train,
                rounds=meta_train_rounds,
                episodes=episodes_train,
                time_steps=time_steps,
                verbose=False,
                bar=False
            )
            entropy=False
        else:
            raise ValueError("Invalid mode")
        
        W_meta = build_meta_weights(totalReservoirStates, totalWSnapshots)
        res_trained = copy.deepcopy(reservoir_train)
        res_trained.W_meta = W_meta

        if parallel:
            with ProcessPoolExecutor(max_workers=5) as pool:
                partial_eval = partial(
                    _eval_theta_meta_worker,
                    agent=agent, env=env, reservoir=res_trained,
                    k=k, rounds=rounds, episodes=episodes,
                    time_steps=time_steps, mode="average",
                    eta=1.0, clip_norm=10, entropy=entropy
                )
                futures = [pool.submit(partial_eval, theta) for theta in theta_list]
                for fut in tqdm(as_completed(futures), total=n_resets,
                                disable=not bar, desc=f'Eval k={k}'):
                    theta0, mu, sem, all_rewards, example_traj = fut.result()
                    idx = theta_list.index(theta0)
                    results[idx]["total_rewards"].append((mu, sem))
                    if store_raw:
                        results[idx]["raw_rewards"].append(all_rewards)
                    if results[idx]["example_trajectory"] is None:
                        results[idx]["example_trajectory"] = example_traj
        else:
            for idx, theta in enumerate(theta_list):
                theta0, mu, sem, all_rewards, example_traj = _eval_theta_meta_worker(
                    theta, agent, env, res_trained,
                    k, rounds, episodes, time_steps,
                    mode="average", eta=1.0, clip_norm=10, entropy=entropy
                )
                results[idx]["total_rewards"].append((mu, sem))
                if store_raw:
                    results[idx]["raw_rewards"].append(all_rewards)
                if results[idx]["example_trajectory"] is None:
                    results[idx]["example_trajectory"] = example_traj

    results.sort(key=lambda d: d["theta0"])
    with open(outfile, "w") as f:
        json.dump(results, f, indent=4, default=convert_numpy)  # Use the custom serializer

    return results


def main():
    agent = LinearAgent(learning_rate=0.03)
    env = Environment()
    reservoir = initialize_reservoir(1000)
    k_list = [1, 2, 3, 5, 7, 10, 15, 20]

    data = EvalOneShotMetaInference(
        agent, env, reservoir,
        rounds=20,
        k_list=k_list,
        episodes=600,
        time_steps=30,
        meta_train_rounds=3,
        episodes_train=600,
        parallel=True,
        bar=True,
        verbose=True,
        store_raw=True, 
        mode="no_entropic",
        outfile="one_shot_meta_inference_results_without_entropy.json"
    )

    plot_one_shot_eval(k_list, data, plotlog=False, savefig=True,
                       filename="one_shot_meta_inference_plot_without_entropy.png")

if __name__ == "__main__":
    main()