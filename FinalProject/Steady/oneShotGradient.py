from trainingUtils import episode, train_episode, train_episode_accumulation

import numpy as np
from tqdm import tqdm
import json

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def one_shot_gradient(agent, env, episodes=100, time_steps=30, mode='normal', verbose=False):

    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")
    reward = 0
    while reward != 1.5:
        agent.reset_parameters()
        if mode == 'normal':
            reward, _ = train_episode(agent, env, time_steps)
        elif mode == 'accumulation':
            reward, _ = train_episode_accumulation(agent, env, time_steps)
        if verbose:
            print(f"Average Reward: {np.mean(reward)}")

    rewards = []
    for _ in range(episodes):
        reward, _ = episode(agent, env, time_steps)
        rewards.append(reward)
    if verbose:
        print(f"One-shot Gradient Average Reward: {np.mean(rewards)}") 

    return rewards

def evaluate_position(agent, env, rounds, episodes=100, time_steps=30, mode='normal', verbose=False, bar=False):
    if mode not in ['normal', 'accumulation']:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")
    
    all_rewards = []
    for _ in tqdm(range(rounds), disable=not bar, desc='Evaluating position'):
        episode_rewards = one_shot_gradient(agent, env, episodes, time_steps, mode)
        all_rewards.extend(episode_rewards)
    
    mu = np.mean(all_rewards)
    sem = np.std(all_rewards, ddof=1) / np.sqrt(rounds)
    if verbose:
        print(f"o = {agent.learning_rate} ⇒ μ = {mu:.3f} ± {sem:.3f} (SEM, n={len(all_rewards)})")
    return mu, sem

def _run_one_reset(
    theta0,
    agent, env,
    lr_list,
    rounds, episodes, time_steps, mode,
):
    """
    Executed in a worker process.  Does *one* env reset and evaluates
    the whole lr_list serially inside that process.
    Returns the dict that EvalOneShotGradient appends to its results.
    """
    # Local, independent copies so that workers never fight over state
    agent = copy.deepcopy(agent)
    env   = copy.deepcopy(env)

    env.reset(theta0)
    agent.reset_parameters()

    agent_pos = env.agent_position.copy()
    food_pos  = env.food_position.copy()

    rewards = []
    for lr in lr_list:
        agent.learning_rate = lr
        mean, std = evaluate_position(
            agent, env,
            rounds=rounds,
            episodes=episodes,
            time_steps=time_steps,
            mode=mode,
            verbose=False,
            bar=False,
        )
        rewards.append((mean, std))

    return {
        "theta0": theta0,
        "agent_position": agent_pos,
        "food_position" : food_pos,
        "total_rewards" : np.asarray(rewards),
    }

def EvalOneShotGradient(
        agent,
        env,
        rounds: int = 1,
        lr_list: list[float] = (0.01, 0.1, 1.0),
        episodes: int = 10,
        time_steps: int = 30,
        mode: str = "normal",
        parallel: bool = False,
        bar: bool = True,
):
    if mode not in {"normal", "accumulation"}:
        raise ValueError("Mode must be either 'normal' or 'accumulation'")

    n_resets = 16
    theta_list = [45 / 2 * i for i in range(n_resets)]

    if parallel:
        with ProcessPoolExecutor(max_workers=4) as pool:
            partial_reset = partial(
                _run_one_reset,
                agent=agent,
                env=env,
                lr_list=lr_list,
                rounds=rounds,
                episodes=episodes,
                time_steps=time_steps,
                mode=mode,
            )

            futures = [pool.submit(partial_reset, th) for th in theta_list]

            results = []
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            disable=not bar,
                            desc="Resets"):
                results.append(fut.result())

        results.sort(key=lambda d: d["theta0"])  
        # Convert np.ndarray to list for JSON serialization
        for result in results:
            result["total_rewards"] = result["total_rewards"].tolist()
            result["agent_position"] = result["agent_position"].tolist()
            result["food_position"] = result["food_position"].tolist()

        with open("one_shot_gradient_results.json", "w") as f:
            json.dump(results, f, indent=4)

        return results

    results = []
    for theta0 in tqdm(theta_list, disable=not bar, desc="Resets", total=n_resets):
        result = _run_one_reset(
            theta0, agent, env, lr_list,
            rounds, episodes, time_steps, mode
        )
        # Convert numpy arrays to lists for JSON serialization
        result["total_rewards"] = result["total_rewards"].tolist()
        result["agent_position"] = result["agent_position"].tolist()
        result["food_position"] = result["food_position"].tolist()
        results.append(result)
    
    with open("one_shot_gradient_results.json", "w") as f:
        json.dump(results, f, indent=4)
    return results

def testing():
    from agent import LinearAgent
    from environment import Environment
    from plottingUtils import plot_one_shot_eval

    env = Environment()
    agent = LinearAgent()
    agent.learning_rate = 0.1 
    one_shot_gradient(agent, env, episodes=1000, time_steps=30, mode='normal', verbose=True)
    evaluate_position(agent, env, rounds=10, episodes=100, time_steps=30, mode='normal', verbose=True, bar = True)
    data = EvalOneShotGradient(agent, env, rounds=1, lr_list=[0.01, 0.1, 1], 
                        episodes=10, time_steps=30, mode='normal', 
                        parallel=False, bar=True)
    
    print(len(data), "data entries found.")
    plot_one_shot_eval([0.01, 0.1, 1], data, savefig=True, filename="one_shot_eval.png")

def test_parallel():
    from agent import LinearAgent
    from environment import Environment
    import time
    from plottingUtils import plot_one_shot_eval

    env = Environment()
    agent = LinearAgent()
    agent.learning_rate = 0.1 

    start_time = time.time()
    data = EvalOneShotGradient(agent, env, rounds=40, lr_list=[0.01, 0.1, 1], 
                        episodes=100, time_steps=30, mode='normal', 
                        parallel=True, bar=False)

    end_time = time.time()
    plot_one_shot_eval([0.01, 0.1, 1], data, plotlog=True, savefig=True, filename="one_shot_eval_parallel.png")
    print("Parallel execution time:", end_time - start_time)

    start_time = time.time()
    data = EvalOneShotGradient(agent, env, rounds=40, lr_list=[0.01, 0.1, 1], 
                        episodes=100, time_steps=30, mode='normal', 
                        parallel=False, bar=False)
    end_time = time.time()
    plot_one_shot_eval([0.01, 0.1, 1], data, plotlog=True, savefig=True, filename="one_shot_eval_sequential.png")
    print("Sequential execution time:", end_time - start_time)

def main():
    from agent import LinearAgent
    from environment import Environment
    from plottingUtils import plot_one_shot_eval

    env = Environment()
    agent = LinearAgent()

    lr_list = [0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10]

    data = EvalOneShotGradient(agent, env, rounds=20, lr_list=lr_list,
                              episodes=400, time_steps=30, mode='normal',
                                parallel=True, bar=True)
    plot_one_shot_eval(lr_list, data, plotlog=True, savefig=True, filename="one_shot_eval.png")


if __name__ == "__main__":
    main()