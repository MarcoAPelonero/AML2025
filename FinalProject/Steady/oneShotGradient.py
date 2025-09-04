from trainingUtils import episode, train_episode, train_episode_accumulation

import numpy as np
from tqdm import tqdm
import json

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

"""First idea of one-shot gradient: highier lr"""

def one_shot_gradient(agent, env, episodes=100, time_steps=30, mode='normal', verbose=False):
    """
    Run a trainining episode until the food is reached one time. Then, exit the loop while keeping the agent parameters
    updated according to that one episode only. Finally, run `episodes` episodes with that set of weights and return the rewards 
    for later analysis.
    The modes are the ones used in trainingUtils.py: 'normal' for online weight updates, 'accumulation' for out of episode weight updates.
    Args:
        agent: the agent to be trained
        env: the environment in which the agent will be trained
        episodes: number of episodes to run after the one-shot training episode
        time_steps: maximum number of time steps per episode
        mode: 'normal' or 'accumulation', see trainingUtils.py
        verbose: whether to print progress messages
    Returns:
        rewards: list of rewards obtained in the `episodes` episodes after the one-shot training episode
    """
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
    '''
    Keeping the environment the same, we run "rounds". 
    We don't want this to get static, so we reset the agent parameters at each round and retrain for a number of specified rounds.
    For each round, we run one_shot_gradient() and collect the rewards.
    We then compute the mean and SEM over all collected rewards.
    Args:
        agent: the agent to be trained
        env: the environment in which the agent will be trained
        rounds: number of one-shot training episodes to run
        episodes: number of episodes to run after each one-shot training episode
        time_steps: maximum number of time steps per episode
        mode: 'normal' or 'accumulation', see trainingUtils.py
        verbose: whether to print progress messages
        bar: whether to show a progress bar
    Returns:
        mu: mean of all rewards collected over all rounds and episodes
        sem: standard error of the mean of all rewards collected over all rounds
    '''
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
    The important part is the setup to run this in parallel, which is why we perform deepcopies of the agent and env.
    Args:
        theta0: initial angle to reset the environment
        agent: the agent to be trained
        env: the environment in which the agent will be trained
        lr_list: list of learning rates to evaluate
        rounds: number of one-shot training episodes to run
        episodes: number of episodes to run after each one-shot training episode
        time_steps: maximum number of time steps per episode
        mode: 'normal' or 'accumulation', see trainingUtils.py
    Returns:
        A dict with the following keys
        "theta0": the initial angle used to reset the environment
        "agent_position": the position of the agent after the reset
        "food_position": the position of the food after the reset
        "total_rewards": a 2D numpy array of shape (len(lr_list),
            2) where each row is (mean, sem) for the corresponding learning rate

    """
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
    '''
    Evaluate one-shot gradient performance over a grid of initial orientations. 
    Seeing as the simulation is intensive, and the evaluations over different orientations are independent,
    we can parallelize the process by using ProcessPoolExecutor. Including GPU acceleration is futile, and possibly counterproductive,
    if one accounts for the time it takes to transfer data to and from the GPU.
    Once all the processed are done, some post-processing is needed to save the results in a json file, and then the function returns the results.
    Args:
        agent: the agent to be trained
        env: the environment in which the agent will be trained
        rounds: number of one-shot training episodes to run per orientation
        lr_list: list of learning rates to evaluate
        episodes: number of episodes to run after each one-shot training episode
        time_steps: maximum number of time steps per episode
        mode: 'normal' or 'accumulation', see trainingUtils.py
        parallel: whether to run the evaluations in parallel
        bar: whether to show a progress bar
    Returns:
        A list of dicts, one per orientation, each with the following keys
        "theta0": the initial angle used to reset the environment
        "agent_position": the position of the agent after the reset
        "food_position": the position of the food after the reset
        "total_rewards": a 2D numpy array of shape (len(lr_list),
            2) where each row is (mean, sem) for the corresponding learning rate
    Raises:
        ValueError: if mode is not 'normal' or 'accumulation'
    '''
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
        result["total_rewards"] = result["total_rewards"].tolist()
        result["agent_position"] = result["agent_position"].tolist()
        result["food_position"] = result["food_position"].tolist()
        results.append(result)
    
    with open("one_shot_gradient_only_grad.json", "w") as f:
        json.dump(results, f, indent=4)
    return results

def testing():
    '''
    Test the individual functions in this module.
    '''
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

def agent_mode():
    '''
    Define an agent and an environment, then run EvalOneShotGradient over a grid of learning rates, to get the one-shot results
    for this approach, then plot the results. Both the figure and the data are saved to file.
    '''
    from agent import LinearAgent
    from environment import Environment
    from plottingUtils import plot_one_shot_eval

    env = Environment()
    agent = LinearAgent()

    lr_list = [0.01, 0.03, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 10]

    data = EvalOneShotGradient(agent, env, rounds=20, lr_list=lr_list,
                              episodes=400, time_steps=30, mode='normal',
                                parallel=True, bar=True)
    plot_one_shot_eval(lr_list, data, plotlog=True, savefig=True, filename="one_shot_eval_only_grad.png")

if __name__ == "__main__":
    agent_mode()
    # jdf