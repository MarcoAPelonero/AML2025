# train.py

import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import trange
import gymnasium as gym

from agent import REINFORCEAgent
from game import BreakoutPongEnv


def make_env(render: bool):
    """
    Returns a function that creates a single BreakoutPongEnv instance.
    """
    return lambda: BreakoutPongEnv(
        render_mode="human" if render else None,
        max_episode_steps=None
    )


def train(
    num_episodes: int = 1000,
    save_path: str | Path = "policy.pth",
    render: bool = False,
    seed: int | None = None,
    num_envs: int = 4,
):
    """
    Trains a REINFORCEAgent on BreakoutPongEnv using a SyncVectorEnv of num_envs parallel environments.
    Uses gymnasium's default “NEXT_STEP” autoreset behavior and masks out finished environments
    so that each episode counts exactly one termination per worker.
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create a vectorized environment with num_envs parallel BreakoutPongEnv instances.
    # We rely on the default autoreset_mode=NEXT_STEP, so that as soon as an env returns done=True,
    # it is automatically reset on the next env.step() call. We will keep track of which
    # workers have terminated this episode via a done_mask.
    env = gym.vector.SyncVectorEnv([make_env(render) for _ in range(num_envs)])

    # Figure out state and action dimensions from a single sub‐environment
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    # Create the REINFORCE agent
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr_actor=1e-3,      # Higher learning rate for policy
        lr_critic=5e-4,     # Lower learning rate for critic
        gamma=0.95,
        grad_clip=1.0,
        num_envs=num_envs,
        device=None
    )

    reward_hist = []
    best_avg_reward = float('-inf')
    best_weights = None
    progress_bar = trange(1, num_episodes + 1, desc="Training", unit="ep")

    # Reset all sub‐environments once before starting
    obs, _ = env.reset()  # obs has shape (num_envs, state_dim)

    try:
        for ep in progress_bar:
            # ep_return[i] will accumulate the total reward for env i in this episode
            ep_return = np.zeros(num_envs, dtype=np.float32)
            # done_mask[i] will flip to True as soon as env i terminates this episode
            done_mask = np.zeros(num_envs, dtype=bool)

            # Run until every sub‐environment has terminated (masked by done_mask)
            while not done_mask.all():
                # 1) Let the agent pick one action per parallel environment
                action = agent.select_action(obs)  # shape: (num_envs,)

                # 2) Step all environments in parallel
                next_obs, reward, terminated, truncated, _ = env.step(action)
                just_done = np.logical_or(terminated, truncated)

                # 3) Accumulate reward only for environments that haven't yet terminated
                ep_return += reward * (~done_mask)

                # 4) Store reward for policy-gradient update (agent stores all rewards per step)
                agent.store_reward(reward)

                # 5) Mark those environments as “finished” for this episode
                done_mask |= just_done

                # 6) If rendering is requested, display the frames
                if render:
                    env.render()

                # 7) Move to the next observations
                obs = next_obs

            # When we exit the while-loop, every env has terminated at least once.
            # Because we are using NEXT_STEP autoreset, each sub-environment that terminated
            # was already reset to its initial state on the very next call to env.step().
            # Moreover, we masked out rewards after termination, so ep_return is
            # exactly the sum over one episode per worker.

            # Finish this rollout (compute returns, update policy, zero buffers, etc.)
            agent.finish_rollout()
            agent.scheduler.step()  # Step the learning rate scheduler

            # Log the mean episodic return across all num_envs
            reward_hist.append(ep_return.mean())
            avg_last_10 = float(np.mean(reward_hist[-10:]))
            actor_lr  = agent.optim.param_groups[0]['lr']
            critic_lr = agent.optim.param_groups[1]['lr']
            progress_bar.set_postfix(
                avg10=avg_last_10,
                actor_lr=f"{actor_lr:.6f}",
                critic_lr=f"{critic_lr:.6f}"
            )
            # Save best model so far
            if ep % (num_episodes // 10) == 0:
                if avg_last_10 > best_avg_reward:
                    best_avg_reward = avg_last_10
                    best_weights = agent.policy.state_dict()

        # Save best model at the end
        if best_weights is not None:
            torch.save(best_weights, str(save_path))
        else:
            agent.save(str(save_path))
        env.close()
        print(f"\nTraining complete – best policy saved to '{save_path}'.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save best model so far
        if best_weights is not None:
            torch.save(best_weights, str(save_path))
            print(f"Best policy saved to '{save_path}'.")
        else:
            agent.save(str(save_path))
            print(f"Policy saved to '{save_path}'.")
        env.close()


# ---------------- CLI -----------------
if __name__ == "__main__":
    n_eps, path = 1000, "policy.pth"
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        n_eps = int(sys.argv[1])
    if len(sys.argv) >= 3:
        path = sys.argv[2]

    train(num_episodes=n_eps, save_path=path, num_envs=4)
