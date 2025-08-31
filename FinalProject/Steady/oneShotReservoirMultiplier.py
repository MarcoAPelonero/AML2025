import numpy as np
from tqdm import tqdm
import json
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from reservoirTrainingUtils import InDistributionTraining, inference_episode, organize_dataset
from trainingUtils import episode
from reservoir import build_W_out, initialize_reservoir


def _train_base_reservoir_at_lr(agent, env, res,
                                base_lr: float,
                                episodes: int,
                                time_steps: int,
                                rounds: int,
                                verbose: bool = False):
    """
    Train reservoir only once at base_lr, collecting an extensive dataset.
    Returns a copy of `res` with Jout set (trained).
    """
    agent_train = copy.deepcopy(agent)
    env_train   = copy.deepcopy(env)
    res_train   = copy.deepcopy(res)
    agent_train.learning_rate = base_lr

    # Collect a large dataset at the base LR
    _, _, reservoir_states, grads = InDistributionTraining(
        agent_train, env_train, res_train,
        rounds=rounds,             # <-- make this 'extensive'
        episodes=episodes,
        time_steps=time_steps,
        verbose=verbose,
        bar=False
    )
    X, Y = organize_dataset(reservoir_states, grads)
    W_out = build_W_out(X, Y)

    res_trained = copy.deepcopy(res_train)
    res_trained.Jout = W_out.T  # (out_dim x hidden_dim), consistent with your usage
    return res_trained


def one_shot_gradient(agent, env, res, episodes=100, time_steps=30, verbose=False):
    """
    Run inference until perfect performance (reward == 1.5), then evaluate average reward over `episodes`.
    Returns a list of rewards.
    """
    reward = None
    while reward != 1.5:
        agent.reset_parameters()
        reward, _, _, _ = inference_episode(agent, env, res, time_steps)
    if verbose:
        print(f"Reached one-shot inference reward: {reward}")

    rewards = []
    for _ in range(episodes):
        r, _ = episode(agent, env, time_steps)
        rewards.append(r)
    if verbose:
        print(f"One-shot Gradient Average Reward: {np.mean(rewards):.3f}")

    return rewards


def evaluate_position(agent, env, res, rounds, episodes=100, time_steps=30,
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
                                    time_steps=time_steps,
                                    verbose=verbose)
        all_rewards.extend(rewards)

    mu  = np.mean(all_rewards)
    n   = len(all_rewards)
    sem = np.std(all_rewards, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    if verbose:
        print(f"Learning rate {agent.learning_rate} → μ = {mu:.3f} ± {sem:.3f} (SEM, n={n})")
    return mu, sem


def _eval_theta_worker(theta0, agent, env, res, lr, rounds, episodes, time_steps, mode):
    """Helper function for parallel evaluation of theta positions."""
    agent_eval = copy.deepcopy(agent)
    env_eval   = copy.deepcopy(env)
    res_eval   = copy.deepcopy(res)

    agent_eval.learning_rate = lr
    env_eval.reset(theta0)
    mu, sem = evaluate_position(
        agent_eval, env_eval, res_eval,
        rounds=rounds,
        episodes=episodes,
        time_steps=time_steps,
        mode=mode,
        verbose=False,
        bar=False
    )
    return theta0, mu, sem


def _scaled_reservoir(res_base, scale: float):
    """
    Return a copy of res_base whose output mapping is scaled so that
    the predicted gradient is multiplied by `scale`.
    This preserves the base training while adjusting the gradient magnitude.
    """
    res_scaled = copy.deepcopy(res_base)
    # Scaling Jout scales the predicted gradient linearly.
    # grad_pred = Jout @ state  --> scale * grad_pred if Jout *= scale
    res_scaled.Jout = res_scaled.Jout * scale
    return res_scaled


def EvalOneShotGradient(agent, env, res,
                         rounds: int = 1,
                         lr_list: list[float] = (0.01, 0.1, 1.0),
                         episodes: int = 10,
                         time_steps: int = 30,
                         mode: str = "normal",
                         parallel: bool = False,
                         bar: bool = True,
                         base_rounds: int = 6):
    """
    Evaluate one-shot gradient performance over a grid of initial orientations.

    NEW BEHAVIOR:
    - Train reservoir ONCE at the smallest LR in `lr_list`, using `base_rounds` for extensive data.
    - For each other LR, reuse the base reservoir and scale its predicted gradients by
      `scale = base_lr / lr_cur` so that the *effective* update lr_cur * grad_cur matches base_lr * grad_base.

    Saves results to 'one_shot_gradient_results.json'.
    Returns a list of dicts per orientation.
    """
    if mode not in {"normal", "accumulation"}:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    # list of initial orientations
    n_resets   = 16
    theta_list = [45 / 2 * i for i in range(n_resets)]

    # prepare result container
    results = []
    for theta in theta_list:
        env_tmp = copy.deepcopy(env)
        env_tmp.reset(theta)
        results.append({
            "theta0": theta,
            "agent_position": env_tmp.agent_position.tolist(),
            "food_position":  env_tmp.food_position.tolist(),
            "total_rewards":  []
        })

    # 1) Train a single reservoir at the smallest LR, extensively
    base_lr = min(lr_list)
    res_base = _train_base_reservoir_at_lr(
        agent, env, res,
        base_lr=base_lr,
        episodes=episodes,
        time_steps=time_steps,
        rounds=base_rounds,  # make this large as needed
        verbose=False
    )

    # 2) Evaluate across LRs by scaling the base reservoir's gradients
    for lr in tqdm(lr_list, desc='Learning rates', disable=not bar):
        # scale so that lr_cur * grad_scaled == base_lr * grad_base
        scale = (base_lr / lr) if lr != 0 else 0.0
        res_for_lr = _scaled_reservoir(res_base, scale=scale)

        if parallel:
            with ProcessPoolExecutor(max_workers=5) as pool:
                partial_eval = partial(_eval_theta_worker,
                                       agent=agent, env=env, res=res_for_lr, lr=lr,
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
                theta0, mu, sem = _eval_theta_worker(theta, agent, env, res_for_lr, lr,
                                                     rounds, episodes, time_steps, mode)
                results[idx]["total_rewards"].append((mu, sem))

    # sort and serialize
    results.sort(key=lambda d: d["theta0"])
    with open("one_shot_gradient_results.json", "w") as f:
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
                       filename="one_shot_eval.png")


if __name__ == "__main__":
    agent_mode()