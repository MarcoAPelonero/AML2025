from reservoirTrainingUtils import InDistributionTraining, inference_episode_multiplier, organize_dataset
from trainingUtils import episode
from reservoir import build_W_out, initialize_reservoir

import numpy as np
from tqdm import tqdm
import json
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def one_shot_gradient(agent, env, res, multiplier,episodes=100, time_steps=30, verbose=False):
    """
    Run inference until perfect performance (reward == 1.5), then evaluate average reward over `episodes`.
    Returns a list of rewards.
    """
    reward = None
    while reward != 1.5:
        agent.reset_parameters()
        reward, _, _, _ = inference_episode_multiplier(agent, env, res, multiplier, time_steps)
    if verbose:
        print(f"Reached one-shot inference reward: {reward}")

    rewards = []
    for _ in range(episodes):
        r, _ = episode(agent, env, time_steps)
        rewards.append(r)
    if verbose:
        print(f"One-shot Gradient Average Reward: {np.mean(rewards):.3f}")

    return rewards


def evaluate_position(agent, env, res, rounds, multiplier, episodes=100, time_steps=30,
                      mode='normal', verbose=False, bar=False):
    """
    For each of `rounds`, run one-shot gradient evaluation and collect rewards.
    Compute mean and SEM over all collected rewards.
    """
    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    all_rewards = []
    for _ in tqdm(range(rounds), disable=not bar, desc='Evaluating position'):
        rewards = one_shot_gradient(agent, env, res,
                                     episodes=episodes,
                                     multiplier=multiplier,
                                     time_steps=time_steps,
                                     verbose=verbose)
        all_rewards.extend(rewards)

    mu = np.mean(all_rewards)
    sem = np.std(all_rewards, ddof=1) / np.sqrt(rounds)
    if verbose:
        print(f"Learning rate {agent.learning_rate} → μ = {mu:.3f} ± {sem:.3f} (SEM, n={len(all_rewards)})")
    return mu, sem


def _eval_theta_worker(theta0, agent, env, res, lr_multiplier, rounds, episodes, time_steps, mode):
    """Helper function for parallel evaluation of theta positions."""
    agent_eval = copy.deepcopy(agent)
    env_eval = copy.deepcopy(env)
    agent_eval.learning_rate = lr_multiplier * agent.learning_rate
    env_eval.reset(theta0)
    mu, sem = evaluate_position(
        agent_eval, env_eval, res,
        rounds=rounds,
        multiplier=lr_multiplier,
        episodes=episodes,
        time_steps=time_steps,
        mode=mode,
        verbose=False,
        bar=False
    )
    return theta0, mu, sem


def EvalOneShotGradient(agent, env, res,
                         rounds: int = 1,
                         lr_list: list[float] = (0.01, 0.1, 1.0),
                         episodes: int = 10,
                         time_steps: int = 30,
                         mode: str = "normal",
                         parallel: bool = False,
                         bar: bool = True):
    """
    Evaluate one-shot gradient performance over a grid of initial orientations.
    Retrains reservoir only when the learning rate changes. Supports parallel evaluation.
    Saves results to 'one_shot_gradient_results.json'.
    Returns a list of dicts per orientation.
    """
    if mode not in {"normal", "accumulation"}:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    # list of initial orientations
    n_resets = 16
    theta_list = [45 / 2 * i for i in range(n_resets)]

    # prepare result container
    results = []
    for theta in theta_list:
        env_tmp = copy.deepcopy(env)
        env_tmp.reset(theta)
        results.append({
            "theta0": theta,
            "agent_position": env_tmp.agent_position.tolist(),
            "food_position": env_tmp.food_position.tolist(),
            "total_rewards": []
        })

    # loop over learning rates
    for lr in tqdm(lr_list, desc='Learning rates', disable=not bar):
        # train reservoir for this lr
        agent_train = copy.deepcopy(agent)
        env_train = copy.deepcopy(env)
        res_train = copy.deepcopy(res)
        agent_train.learning_rate = lr_list[0]

        k = lr / lr_list[0]

        _, _, reservoir_states, grads = InDistributionTraining(
            agent_train, env_train, res_train,
            rounds=2,
            episodes=episodes,
            time_steps=time_steps,
            verbose=False,
            bar=False
        )
        X, Y = organize_dataset(reservoir_states, grads)
        W_out = build_W_out(X, Y)
        res_trained = copy.deepcopy(res_train)
        res_trained.Jout = W_out.T

        # evaluate across orientations
        if parallel:
            with ProcessPoolExecutor(max_workers=5) as pool:
                partial_eval = partial(_eval_theta_worker, 
                                     agent=agent, env=env, res=res_trained, lr_multiplier=k,
                                     rounds=rounds, episodes=episodes, 
                                     time_steps=time_steps, mode=mode)
                futures = [pool.submit(partial_eval, theta) for theta in theta_list]
                for fut in tqdm(as_completed(futures), total=n_resets, disable=not bar,
                                desc=f'Eval lr={lr}'):
                    theta0, mu, sem = fut.result()
                    idx = theta_list.index(theta0)
                    results[idx]["total_rewards"].append((mu, sem))
        else:
            for idx, theta in enumerate(theta_list):
                _, mu, sem = _eval_theta_worker(theta, agent, env, res_trained, k, rounds, episodes, time_steps, mode)
                results[idx]["total_rewards"].append((mu, sem))

    # sort and serialize
    results.sort(key=lambda d: d["theta0"])
    with open("one_shot_gradient_res_gradient_multiplier.json", "w") as f:
        # convert numpy floats to native lists
        json.dump(results, f, indent=4)

    return results

def agent_mode():
    from agent import LinearAgent
    from environment import Environment
    from plottingUtils import plot_one_shot_eval
    from reservoir import initialize_reservoir

    env = Environment()
    agent = LinearAgent()
    res = initialize_reservoir()

    lr_list = [0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10]

    data = EvalOneShotGradient(
        agent, env, res,
        rounds=20,
        lr_list=lr_list,
        episodes=600,
        time_steps=30,
        parallel=True,
        bar=True
    )
    plot_one_shot_eval(lr_list, data,
                       plotlog=True,
                       savefig=True,
                       filename="one_shot_eval_res_gradient_multiplier.png")


if __name__ == "__main__":
    agent_mode()
